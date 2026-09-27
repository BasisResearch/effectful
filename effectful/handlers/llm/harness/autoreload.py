"""Re-run edited code while a script served by the launcher keeps running.

Under ``python -m effectful.handlers.llm.harness --autoreload <script>``, `hmr
<https://pypi.org/project/hmr/>`_ re-runs each edited module that was imported from a
directory on `sys.path` -- not the standard library or installed packages, nor any
module imported before it started, which includes effectful's core -- the harness
stack is rebuilt from the launcher's flags, and each subscriber is told. A class keeps
its identity across edits: the class its instances already have is updated from the
re-run and bound to its name again, so every instance, held anywhere, runs the edited
code, and an agent's next call replaces its conversation's system message. The running
script is ``__main__`` and is never re-run; `Reloader.module`, the script imported
under its own name when it is first edited, updates its classes. A module that holds
the running process sets ``__autoreload__ = False`` and imports reloadable modules
only inside functions.

Edits apply between calls, on the thread that calls `Reloader.apply`: the launcher's
watcher thread, or the event loop of a host that awaits `Reloader.watch`, which a host
with a loop should, since hmr is not thread-safe. An operation keeps its identity
across edits (see
`~effectful.ops.types.REDEFINING`), and handlers a script installs over the launcher's
stack follow the stack as it is rebuilt.

Limits. A file that does not parse, or a stack that does not build, keeps its previous
version. A module that raises while re-running keeps the names bound before the error
updated and the rest as they were. A name an edit removes stays bound, as under
`importlib.reload` or in a notebook, since nothing can tell a stale definition from
state; so a deleted tool is still offered, and an instance of a renamed class keeps
the old one, until a restart. A class whose layout an edit changed -- its bases,
metaclass or slots -- is bound anew, and its existing instances keep the old one. The
script's module-level code runs again in `Reloader.module`, so what should run once
belongs under ``if __name__ == "__main__"``.
"""

import asyncio
import collections.abc
import contextlib
import functools
import gc
import importlib
import importlib.abc
import importlib.util
import os
import pathlib
import sys
import threading
import traceback
import types
import typing

from effectful.internals.runtime import get_interpretation
from effectful.ops.semantics import LiveInterpretation, coproduct, fwd, handler
from effectful.ops.types import REDEFINING, Interpretation, Operation

HARNESS = "effectful.handlers.llm.harness"

GENERATION = "__autoreload_generation__"
"""The reload an agent's conversation last ran under, kept in the agent's ``__dict__``."""


@Operation.define
def current() -> "Reloader | None":
    """The reloader of the launcher's ``--autoreload``, under the stack it installed."""
    return None


def _layout(cls: type) -> tuple:
    """What an existing class cannot be changed to match."""
    bases = [(base.__module__, base.__qualname__) for base in cls.__mro__[1:]]
    return type(cls), cls.__basicsize__, cls.__dict__.get("__slots__"), bases


def _update(old: type, new: type) -> bool:
    """Make `old` define what `new` does, if their layouts match; whether it could."""
    if _layout(old) != _layout(new):
        return False
    for name in set(vars(old)) - set(vars(new)):
        with contextlib.suppress(AttributeError, TypeError):
            delattr(old, name)
    for name, value in vars(new).items():
        if name in ("__dict__", "__weakref__", "__module__") or name.startswith(
            "_abc_"
        ):
            continue
        inner = vars(old).get(name)
        if (
            isinstance(value, type)
            and isinstance(inner, type)
            and _update(inner, value)
        ):
            continue
        # Zero-argument `super()` reads the class from a `__class__` cell.
        for fn in (
            value,
            *(getattr(value, a, None) for a in ("__func__", "fget", "fset", "fdel")),
        ):
            code = getattr(fn, "__code__", None)
            if code is not None and "__class__" in code.co_freevars:
                cell = fn.__closure__[code.co_freevars.index("__class__")]
                if cell.cell_contents is new:
                    cell.cell_contents = old
        with contextlib.suppress(AttributeError, TypeError):
            setattr(old, name, value)
    return True


def _keep_classes(namespace: dict[str, typing.Any], before: dict[str, type]) -> None:
    """Bind each of the classes `namespace` had `before` again, updated from its re-run."""
    for name, old in before.items():
        new = namespace.get(name)
        if (
            new is old
            or not isinstance(new, type)
            or new.__qualname__ != old.__qualname__
        ):
            continue
        if _update(old, new):
            namespace[name] = old
        else:
            print(
                f"note: {name} changed layout; its instances keep the old class",
                file=sys.stderr,
            )


def _classes(namespace: dict[str, typing.Any]) -> dict[str, type]:
    """The classes `namespace` defines at its top level."""
    return {
        name: value
        for name, value in namespace.items()
        if isinstance(value, type)
        and value.__module__ == namespace["__name__"]
        and value.__qualname__ == name
    }


def gate_interpretation(reloader: "Reloader | None" = None) -> Interpretation:
    """Handlers that hold `reloader` off during a call, answer `current` with it, and
    replace the system message of a conversation last run before a reload; built from
    the hooks as they now are."""
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
        if reloader is None:
            return fwd()
        state = getattr(getattr(skill, "__self__", None), "__dict__", {})
        with reloader.hold():
            last, state[GENERATION] = state.get(GENERATION), reloader.version
            if last is None or last == reloader.version:
                return fwd()
            with handler({hooks.call_system: call_system}):
                return fwd(skill, *args, **kwargs)

    return {hooks.call_agent: call_agent, current: lambda: reloader}


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
        self._finder = sys.meta_path[0]

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
        """The script imported under its own name, which updates the classes of
        ``__main__``; executed when first read."""
        existing = sys.modules.get(self.script.stem)
        if existing is not None and vars(existing).get("__file__") == str(self.script):
            return existing
        spec = self._finder.find_spec(self.script.stem, None)
        if spec is None or spec.loader is None or spec.origin != str(self.script):
            print(
                f"note: {self.script.name} cannot be imported by name", file=sys.stderr
            )
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        main = vars(sys.modules["__main__"])
        before = {n: c for n, c in _classes(main | {"__name__": "__main__"}).items()}
        spec.loader.exec_module(module)
        _keep_classes(vars(module), before)
        return module

    def subscribe(self, callback: collections.abc.Callable[["Reloader"], None]) -> None:
        """Call `callback` after each reload, before any call runs under it."""
        self._subscribers.append(callback)

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
        files: collections.abc.Iterable[str | os.PathLike[str]],
        *,
        wait: bool = True,
    ) -> bool | None:
        """Re-run the edited `files`.

        `None` if a call holds reloads off and `wait` is false; otherwise whether
        anything was re-run. Subscribers must not call skills or apply.
        """
        from reactivity.hmr.core import ReactiveModule
        from watchfiles import Change

        paths = {pathlib.Path(file).resolve() for file in files}
        # The script's copy is imported by its first edit, which it then already runs.
        first = self.script in paths and "module" not in vars(self)
        modules = [
            ReactiveModule.instances.get(path)
            for path in paths
            if path.is_file() and not (first and path == self.script)
        ]
        if not first and not any(
            m and vars(m).get("__autoreload__") is not False for m in modules
        ):
            return False
        if not self._room.acquire(blocking=wait):
            return None
        try:
            token = REDEFINING.set(True)
            built = False
            try:
                if first:
                    self.module
                self._hmr.on_events(
                    [(Change.modified, str(m.__file__)) for m in modules if m]
                )
                self._pull()
                with self._hmr.error_filter:
                    stack, built = self._derived(), True
            finally:
                REDEFINING.reset(token)
            if built:
                self._stack = stack
            else:
                self._derived.dirty = True  # rebuilt on the next edit
            # Every reload, not only one that rebuilt the stack: classes changed too.
            self._generation += 1
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
            and vars(module).get("__autoreload__") is not False
            and module._ReactiveModule__load.dirty
        ]:
            for module in dirty:
                ran.add(module)
                before = _classes(vars(module))
                with self._hmr.error_filter:
                    module._ReactiveModule__load()
                _keep_classes(vars(module), before)

    def _dirs(self) -> set[str]:
        """The directories of the modules hmr loaded, which are all an edit can reach."""
        from reactivity.hmr.core import ReactiveModule

        paths = [*ReactiveModule.instances, self.script]
        return {str(path.parent) for path in paths if path.parent.is_dir()}

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
        for name, module in list(sys.modules.items()):
            if isinstance(module, ReactiveModule):
                del sys.modules[name]
        gc.collect()
