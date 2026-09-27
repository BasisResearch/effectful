"""Re-run edited code while a script served by the launcher keeps running.

Under ``python -m effectful.handlers.llm.harness --autoreload <script>``, `hmr
<https://pypi.org/project/hmr/>`_ re-runs each edited module that was imported from a
directory on `sys.path` -- not the standard library or installed packages, and of
effectful only this package -- the harness stack is rebuilt from the launcher's
flags, and each subscriber is told, so it can move its agents onto the new classes
with `Reloader.refresh`. The running script is ``__main__`` and is never re-run;
`Reloader.module`, the script imported under its own name on first use, supplies its
current classes. A module that holds the running process sets ``__autoreload__ =
False`` and imports reloadable modules only inside functions.

Edits apply between calls, on the thread that calls `Reloader.apply`: the launcher's
watcher thread, or the event loop of a host that awaits `Reloader.watch`, which a host
with a loop should, since hmr is not thread-safe. `linecache` serves the running
version of each reloadable file. An operation keeps its identity across edits (see
`~effectful.ops.types.REDEFINING`), and handlers a script installs over the launcher's
stack follow the stack as it is rebuilt.

Limits. A file that does not parse, or a stack that does not build, keeps its previous
version. A module that raises while re-running keeps the names bound before the error
updated and the rest as they were. The script's module-level code runs again in
`Reloader.module`, so what should run once belongs under ``if __name__ ==
"__main__"``. An agent whose class an edit renamed or removed keeps its old class.
"""

import ast
import asyncio
import collections.abc
import contextlib
import dataclasses
import functools
import gc
import importlib
import importlib.abc
import importlib.util
import linecache
import os
import pathlib
import symtable
import sys
import threading
import traceback
import types
import typing

from effectful.internals.runtime import get_interpretation
from effectful.ops.semantics import LiveInterpretation, coproduct, fwd, handler
from effectful.ops.types import REDEFINING, Interpretation, Operation

HARNESS = "effectful.handlers.llm.harness"

INSTANCE_STALE = "__autoreload_stale__"
"""Set in an agent's ``__dict__`` by `rebind`; its next call replaces its system message."""


@Operation.define
def current() -> "Reloader | None":
    """The reloader of the launcher's ``--autoreload``, under the stack it installed."""
    return None


def rebind(agent: object, cls: type) -> None:
    """Move `agent` onto `cls` in place, and have its next call refresh its system message."""
    if type(agent) is not cls:
        try:
            object.__setattr__(agent, "__class__", cls)
        except TypeError as e:
            raise RuntimeError(
                f"cannot move a {type(agent).__qualname__} onto {cls}"
            ) from e
        # A field's plain default is a class attribute; one from a factory is not.
        for field in dataclasses.fields(cls) if dataclasses.is_dataclass(cls) else ():
            if (
                field.name not in vars(agent)
                and field.default_factory is not dataclasses.MISSING
            ):
                object.__setattr__(agent, field.name, field.default_factory())
    vars(agent)[INSTANCE_STALE] = True


def gate_interpretation(reloader: "Reloader | None" = None) -> Interpretation:
    """Handlers that hold `reloader` off during a call, answer `current` with it, and
    replace a rebound agent's system message; built from the hooks as they now are."""
    hooks = importlib.import_module(f"{HARNESS}.hooks")
    transaction = importlib.import_module(f"{HARNESS}.durability.transaction")

    def call_system(*args, **kwargs):
        message = fwd()
        # The transaction's buffer, which the call adopts as a rewrite.
        history = transaction.HistoryBuilder.get_history()
        if history and history[0]["role"] == "system":
            history[0] = message
        return message

    def call_agent(skill, *args, **kwargs):
        state = getattr(getattr(skill, "__self__", None), "__dict__", {})
        with reloader.hold() if reloader is not None else contextlib.nullcontext():
            if not state.pop(INSTANCE_STALE, False):
                return fwd()
            with handler({hooks.call_system: call_system}):
                return fwd(skill, *args, **kwargs)

    return {hooks.call_agent: call_agent, current: lambda: reloader}


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """hmr's finder, minus effectful outside this package, pinning each module as it runs."""

    def __init__(self, hmr: typing.Any, pin) -> None:
        self._hmr, self._pin = hmr, pin

    def find_spec(self, name, path, target=None):
        if name.partition(".")[0] == "effectful" and (
            not name.startswith(f"{HARNESS}.")
            or name in (__name__, f"{HARNESS}.__main__")
        ):
            return None
        spec = self._hmr.find_spec(name, path, target)
        if spec is not None:
            spec.loader_state, spec.loader = spec.loader, self
        return spec

    def create_module(self, spec):
        return spec.loader_state.create_module(spec)

    def exec_module(self, module):
        self._pin(module.__spec__.origin)
        module.__spec__.loader_state.exec_module(module)


def _read(file: str) -> str | None:
    try:
        return pathlib.Path(file).read_text("utf-8")
    except OSError:
        return None


def _reloadable(module: types.ModuleType) -> bool:
    return vars(module).get("__autoreload__", True) is not False


def _prune(module: types.ModuleType, file: str, source: str) -> None:
    """Drop the names `module` binds that its `source` no longer does."""
    tree = ast.parse(source)
    if "__path__" in vars(module) or any(
        isinstance(node, ast.ImportFrom) and node.names[0].name == "*"
        for node in ast.walk(tree)
    ):
        return
    table = symtable.symtable(source, file, "exec")
    bound = {
        s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()
    }
    tables = table.get_children()
    while tables:
        child = tables.pop()
        bound |= {s.get_name() for s in child.get_symbols() if s.is_declared_global()}
        tables += child.get_children()
    proxy = module._ReactiveModule__namespace_proxy
    for name in [n for n in proxy.raw if n not in bound]:
        if not name.startswith(("__", "_ReactiveModule__")):
            del proxy[name]


class Reloader(LiveInterpretation):
    """The launcher's harness stack, rebuilt as the modules it and the script import change."""

    def __init__(
        self,
        script: str | os.PathLike[str],
        build: collections.abc.Callable[[], Interpretation],
    ) -> None:
        try:
            from reactivity.hmr._common import HMR_CONTEXT
            from reactivity.hmr.core import BaseReloader
        except ImportError as e:
            raise ImportError(
                "--autoreload needs hmr: pip install effectful[llm]"
            ) from e
        import effectful

        self.script = pathlib.Path(script).resolve()
        self._subscribers: list[collections.abc.Callable[[Reloader], None]] = []
        self._pinned: dict[str, tuple] = {}
        self._stop = threading.Event()
        # Held while calls run or a reload applies; the first call in takes it, the
        # last one out releases it.
        self._room, self._count, self._calls = threading.Lock(), threading.Lock(), 0

        # hmr finds modules through `sys.path`, where an editable install need not
        # put effectful.
        root = str(pathlib.Path(effectful.__file__).parents[1])
        if root not in sys.path:
            sys.path.append(root)
        self._hmr = BaseReloader(
            str(self.script), [p or "." for p in sys.path if os.path.isdir(p or ".")]
        )
        self._finder = sys.meta_path[0] = _Finder(sys.meta_path[0], self._pin)

        base = get_interpretation()
        self._derived = HMR_CONTEXT.derived(
            lambda: coproduct(base, coproduct(build(), gate_interpretation(self)))
        )
        # A build that reads nothing reloadable is legitimate, not a lost dependency.
        self._derived.reactivity_loss_strategy = "ignore"
        self._stack: Interpretation = self._derived()
        self._generation = 0

    @property
    def version(self) -> int:
        return self._generation

    def snapshot(self) -> Interpretation:
        return self._stack

    @functools.cached_property
    def module(self) -> types.ModuleType | None:
        """The script imported under its own name, which supplies its current classes."""
        existing = sys.modules.get(self.script.stem)
        if existing is not None and vars(existing).get("__file__") == str(self.script):
            return existing
        spec = self._finder.find_spec(self.script.stem, None)
        if spec is None or spec.origin != str(self.script):
            print(
                f"note: {self.script.name} cannot be imported by name", file=sys.stderr
            )
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def subscribe(self, callback: collections.abc.Callable[["Reloader"], None]) -> None:
        """Call `callback` after each reload, before any call runs under it."""
        self._subscribers.append(callback)

    def current_class(self, cls: type) -> type:
        """The class `cls`'s module now defines under its name, else `cls`, with a note."""
        from reactivity.hmr.core import ReactiveModule

        name = cls.__module__
        module = self.module if name == "__main__" else sys.modules.get(name)
        if not isinstance(module, ReactiveModule) or "<locals>" in cls.__qualname__:
            return cls
        found = functools.reduce(
            lambda scope, part: getattr(scope, part, None),
            cls.__qualname__.split("."),
            module,
        )
        if isinstance(found, type):
            return found
        print(f"note: {name}.{cls.__qualname__} is no longer defined", file=sys.stderr)
        return cls

    def refresh(self, agent: object) -> None:
        """Move `agent` onto the class its module now defines; see `rebind`."""
        rebind(agent, self.current_class(type(agent)))

    @contextlib.contextmanager
    def hold(self) -> collections.abc.Iterator[None]:
        """Keep reloads off while this is entered, as a call does; nests, across threads."""
        with self._count:
            self._calls += 1
            if self._calls == 1:
                self._room.acquire()
        try:
            yield
        finally:
            with self._count:
                self._calls -= 1
                if self._calls == 0:
                    self._room.release()

    def apply(
        self,
        files: collections.abc.Iterable[str | os.PathLike[str]] | None = None,
        *,
        wait: bool = True,
    ) -> bool | None:
        """Re-run the edited `files`, by default every reloadable file changed on disk.

        `None` if a call holds reloads off and `wait` is false; otherwise whether
        anything was re-run. Subscribers must not call skills or apply.
        """
        from reactivity.hmr.core import ReactiveModule
        from watchfiles import Change

        if files is None:
            files = [f for f, e in self._pinned.items() if _read(f) != "".join(e[2])]
        paths = {pathlib.Path(file).resolve() for file in files}
        modules = [
            ReactiveModule.instances.get(path) for path in paths if path.is_file()
        ]
        if not any(m is not None and _reloadable(m) for m in modules):
            return False
        if not self._room.acquire(blocking=wait):
            return None
        try:
            token = REDEFINING.set(True)
            built = False
            try:
                self._hmr.on_events(
                    [(Change.modified, str(m.__file__)) for m in modules if m]
                )
                self._pull()
                with self._hmr.error_filter:
                    stack, built = self._derived(), True
            finally:
                REDEFINING.reset(token)
            if not built:
                self._derived.dirty = True  # rebuilt on the next edit
            elif stack is not self._stack:
                self._stack, self._generation = stack, self._generation + 1
            linecache.cache.update(self._pinned)
            for callback in self._subscribers:
                with self._hmr.error_filter:
                    callback(self)
        finally:
            self._room.release()
        return True

    def _pull(self) -> None:
        """Re-run each dirty module once, here, so no other thread does it lazily later."""
        from reactivity.hmr.core import ReactiveModule

        ran: set[types.ModuleType] = set()
        while dirty := [
            module
            for module in list(ReactiveModule.instances.values())
            if module not in ran
            and _reloadable(module)
            and module._ReactiveModule__load.dirty
        ]:
            for module in dirty:
                ran.add(module)
                file = str(module.__file__)
                self._pin(file)
                with self._hmr.error_filter:
                    module._ReactiveModule__load()
                if (entry := self._pinned.get(file)) is not None:
                    _prune(module, file, "".join(entry[2]))

    def _pin(self, file: str) -> None:
        """Serve `file` from `linecache` as it is about to run, if it compiles."""
        source = _read(file)
        try:
            compile(source or "", file, "exec", dont_inherit=True)
        except (SyntaxError, ValueError):
            return
        if source is not None:
            lines = source.splitlines(keepends=True)
            linecache.cache[file] = self._pinned[file] = (
                len(source),
                None,
                lines,
                file,
            )

    def _dirs(self) -> set[str]:
        """The directories of the modules hmr loaded, which are all an edit can reach."""
        from reactivity.hmr.core import ReactiveModule

        return {
            str(p.parent) for p in list(ReactiveModule.instances) if p.parent.is_dir()
        }

    def _applied(self, files: set[str], wait: bool) -> bool | None:
        """`apply`, for a watcher, which must outlive an error in a reload."""
        try:
            return self.apply(files, wait=wait)
        except Exception:
            traceback.print_exc()
            return False

    def _watch_options(self) -> dict[str, typing.Any]:
        from watchfiles import PythonFilter

        return dict(
            watch_filter=PythonFilter(),
            recursive=False,
            debounce=300,
            rust_timeout=250,
            yield_on_timeout=True,
        )

    def start(self) -> None:
        """Apply edits from a daemon thread, until `watch` or `close`."""
        threading.Thread(
            target=self._watch_thread, name="autoreload", daemon=True
        ).start()

    def _watch_thread(self) -> None:
        from watchfiles import watch

        while not self._stop.wait(0.25):
            if dirs := self._dirs():
                for events in watch(
                    *dirs, stop_event=self._stop, **self._watch_options()
                ):
                    self._applied({file for _, file in events}, wait=True)
                    if self._dirs() != dirs:
                        break

    async def watch(self) -> None:
        """Apply edits on the running event loop between its calls; stops `start`'s thread."""
        from watchfiles import awatch

        self._stop.set()
        pending: set[str] = set()
        while True:
            if not (dirs := self._dirs()):
                await asyncio.sleep(0.25)
                continue
            async for events in awatch(*dirs, **self._watch_options()):
                # Deferred rather than awaited: waiting for a call here would block the
                # loop the call needs. The next yield retries.
                pending |= {file for _, file in events}
                if self._applied(pending, wait=False) is not None:
                    pending = set()
                if self._dirs() != dirs:
                    break

    def close(self) -> None:
        """Stop watching and forget every module hmr loaded; for tests."""
        from reactivity.hmr.core import ReactiveModule

        self._stop.set()
        sys.meta_path.remove(self._finder)
        self._derived.dispose()
        for file in self._pinned:
            linecache.cache.pop(file, None)
        for name, module in list(sys.modules.items()):
            if isinstance(module, ReactiveModule):
                del sys.modules[name]
        gc.collect()
