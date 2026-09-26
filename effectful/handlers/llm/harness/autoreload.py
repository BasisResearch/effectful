"""Re-run edited code while a script served by the launcher keeps running.

Under ``python -m effectful.handlers.llm.harness --autoreload <script>``, `hmr
<https://pypi.org/project/hmr/>`_ re-runs the modules of the script's project -- the
nearest directory above it with a ``pyproject.toml``, ``setup.py``, ``setup.cfg`` or
``.git`` -- and of this package as they are edited, the rest of effectful never, the
harness stack is rebuilt from the launcher's flags, and each
subscriber is told so it can move its agents onto the new classes with
`Reloader.refresh`. The running script is ``__main__`` and is never re-run; a copy
under its module name supplies the current classes. A module that holds the running
process declares itself with `keep`, and must import reloadable modules only inside
functions. A file that does not parse keeps its previous version, and `linecache`
serves the running version of every reloadable file.

Limits. The script's module-level code runs again in its copy, so anything that
should run once belongs under ``if __name__ == "__main__"``. A module that raises
while re-running is left with the names bound before the error updated and the rest
as they were. An agent whose class the edit renamed or removed stays on its old
class, with a note on stderr. A stack that fails to build is kept as it was and
rebuilt on the next edit.

Reloads apply on the thread that calls `Reloader.apply` -- the launcher's watcher
thread, or the event loop of a host that awaits `Reloader.watch` -- and never while a
call runs under `gate_interpretation`. hmr's reactive graph is not thread-safe, so a
host with an event loop should adopt the watcher.

The launcher installs the stack as a `~effectful.ops.semantics.LiveInterpretation`,
so handlers a script installs on top with `~effectful.ops.semantics.handler` follow
the stack as it is rebuilt; and a module is re-executed under
`~effectful.ops.types.REDEFINING`, so an operation it defines keeps its
identity across edits and handlers keyed by it, wherever they were made, still apply.
"""

import ast
import collections.abc
import contextlib
import contextvars
import dataclasses
import gc
import importlib
import importlib.abc
import importlib.machinery
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
from effectful.ops.semantics import LiveInterpretation, coproduct, fwd
from effectful.ops.types import REDEFINING, Interpretation

INSTANCE_STALE = "__autoreload_stale__"
"""Set in an agent's ``__dict__`` by `rebind`; its next call replaces its system message."""

HARNESS = "effectful.handlers.llm.harness"

_IN_SHARED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "autoreload_shared", default=False
)
_TURN: contextvars.ContextVar[dict[str, typing.Any] | None] = contextvars.ContextVar(
    "autoreload_turn", default=None
)
_CURRENT: "Reloader | None" = None
_KEPT_EARLY: list[typing.Any] = []


class _Gate:
    """A reader/writer lock: calls hold the shared side, `Reloader.apply` the exclusive."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0
        self._writer: threading.Thread | None = None

    def idle(self) -> bool:
        with self._cond:
            return not self._readers and self._writer is None

    def acquire_exclusive(self, *, wait: bool = True) -> bool:
        with self._cond:
            if self._writer is threading.current_thread():
                raise RuntimeError("a reload cannot be applied from within a reload")
            while self._readers or self._writer is not None:
                if not wait:
                    return False
                self._cond.wait()
            self._writer = threading.current_thread()
            return True

    def release_exclusive(self) -> None:
        with self._cond:
            self._writer = None
            self._cond.notify_all()

    def _acquire_shared(self) -> bool:
        with self._cond:
            if self._writer is threading.current_thread():
                return False
            while self._writer is not None:
                self._cond.wait()
            self._readers += 1
            return True

    def _release_shared(self) -> None:
        with self._cond:
            self._readers -= 1
            self._cond.notify_all()

    @contextlib.contextmanager
    def shared(self) -> collections.abc.Iterator[None]:
        """Hold the shared side; reentrant within a context and for the applying thread."""
        if _IN_SHARED.get() or not self._acquire_shared():
            yield
            return
        token = _IN_SHARED.set(True)
        try:
            yield
        finally:
            _IN_SHARED.reset(token)
            self._release_shared()


def rebind(agent: object, cls: type) -> None:
    """Move `agent` onto `cls` in place, and have its next call refresh its system message."""
    if type(agent) is not cls:
        try:
            object.__setattr__(agent, "__class__", cls)
        except TypeError as e:
            raise RuntimeError(
                f"cannot move a {type(agent).__qualname__} onto the edited class: {e}; "
                f"restart instead"
            ) from e
        if dataclasses.is_dataclass(cls):
            for field in dataclasses.fields(cls):
                if field.name in vars(agent):
                    continue
                if field.default is not dataclasses.MISSING:
                    object.__setattr__(agent, field.name, field.default)
                elif field.default_factory is not dataclasses.MISSING:
                    object.__setattr__(agent, field.name, field.default_factory())
    vars(agent)[INSTANCE_STALE] = True


def gate_interpretation(gate: _Gate | None = None) -> Interpretation:
    """Handlers that keep a reload out of a running call and refresh a rebound agent's system message.

    Built from the hooks module as it now is, so it is rebuilt with the stack.
    """
    gate = _Gate() if gate is None else gate
    hooks = importlib.import_module(f"{HARNESS}.hooks")
    transaction = importlib.import_module(f"{HARNESS}.durability.transaction")

    def call_agent(skill, *args, **kwargs):
        state = getattr(getattr(skill, "__self__", None), "__dict__", None)
        stale = bool(state is not None and state.pop(INSTANCE_STALE, False))
        with gate.shared():
            token = _TURN.set({"refresh": stale})
            try:
                return fwd()
            finally:
                _TURN.reset(token)

    def call_system(*args, **kwargs):
        message = fwd()
        turn = _TURN.get()
        if turn is not None and turn.pop("refresh", False):
            # The transaction's buffer, which the call adopts as a rewrite.
            history = transaction.HistoryBuilder.get_history()
            if history and history[0]["role"] == "system":
                history[0] = message
        return message

    return {hooks.call_agent: call_agent, hooks.call_system: call_system}


class _PinningLoader(importlib.abc.Loader):
    """hmr's loader, pinning a file's source before it first runs."""

    def __init__(self, inner: typing.Any, reloader: "Reloader") -> None:
        self._inner = inner
        self._reloader = reloader

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> typing.Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: types.ModuleType) -> None:
        if (file := vars(module).get("__file__")) is not None:
            self._reloader._pin_file(file)
        self._inner.exec_module(module)


class _PinningFinder(importlib.abc.MetaPathFinder):
    """hmr's finder, with its loader wrapped in `_PinningLoader`."""

    def __init__(self, inner: typing.Any, reloader: "Reloader") -> None:
        self._inner = inner
        self._reloader = reloader

    def find_spec(self, fullname, path, target=None):
        spec = self._inner.find_spec(fullname, path, target)
        if spec is not None:
            spec.loader = _PinningLoader(spec.loader, self._reloader)
        return spec


class _Live(LiveInterpretation):
    """An interpretation that is whatever the reloader's stack now is."""

    def __init__(self, reloader: "Reloader") -> None:
        self._reloader = reloader

    @property
    def version(self) -> int:
        return self._reloader._generation

    def snapshot(self) -> Interpretation:
        return self._reloader._stack


def _bound_names(file: pathlib.Path) -> set[str] | None:
    """The names `file` binds at module level, or `None` if they cannot be known."""
    source = file.read_text()
    try:
        table = symtable.symtable(source, str(file), "exec")
    except SyntaxError:
        return None
    bound = {
        s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()
    }
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.names[0].name == "*":
            if node.level or node.module not in sys.modules:
                return None
            names = vars(sys.modules[node.module])
            bound |= set(
                names.get("__all__") or [n for n in names if not n.startswith("_")]
            )
    return bound


PROJECT_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg", ".git")


def _module_name(script: pathlib.Path) -> tuple[str, pathlib.Path]:
    """`script`'s dotted module name, and the directory it is importable from."""
    parts, directory = [script.stem], script.parent
    while (directory / "__init__.py").is_file():
        parts.insert(0, directory.name)
        directory = directory.parent
    return ".".join(parts), directory


def _project_root(script: pathlib.Path) -> pathlib.Path:
    """The nearest enclosing project of `script`, else where its top-level package is."""
    stop = {pathlib.Path.home().resolve(), pathlib.Path(script.anchor)}
    for directory in script.parents:
        if directory in stop:
            break
        if any((directory / marker).exists() for marker in PROJECT_MARKERS):
            return directory
    name, importable_from = _module_name(script)
    return importable_from / name.split(".")[0] if "." in name else script.parent


def _effectful_excludes() -> list[pathlib.Path]:
    """Every part of effectful but the harness, and the harness's own machinery."""
    import effectful

    here = pathlib.Path(__file__).parent
    excludes = [here / name for name in ("__init__.py", "__main__.py", "autoreload.py")]
    directory = pathlib.Path(effectful.__file__).parent
    for keep in ("handlers", "llm", "harness"):
        excludes += [child for child in directory.iterdir() if child.name != keep]
        directory = directory / keep
    return excludes


class Reloader:
    """One process's reloader; see the module docstring and `install`."""

    def __init__(
        self,
        script: str | os.PathLike[str],
        build: collections.abc.Callable[[], Interpretation],
        *,
        includes: collections.abc.Iterable[str | os.PathLike[str]] = (),
    ) -> None:
        try:
            from reactivity.hmr._common import HMR_CONTEXT
            from reactivity.hmr.core import (
                BaseReloader,
                ReactiveModuleFinder,
                ReactiveModuleLoader,
            )
        except ImportError as e:
            raise ImportError(
                "--autoreload needs hmr; install it with `pip install effectful[llm]`"
            ) from e

        import effectful

        package = importlib.import_module(HARNESS)

        self.script = pathlib.Path(script).resolve()
        self.name, importable_from = _module_name(self.script)
        self.root = _project_root(self.script)
        self._build = build
        self._gate = _Gate()
        self._subscribers: list[collections.abc.Callable[[Reloader], None]] = []
        self._kept: set[pathlib.Path] = set()
        self._pending: set[pathlib.Path] = set()
        self._pinned: dict[str, tuple] = {}
        self._suspended = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # hmr finds modules only through `sys.path`, which an editable install
        # need not put a repository on.
        for directory in (
            importable_from,
            self.root,
            self.root / "src",
            pathlib.Path(effectful.__file__).parents[1],
        ):
            if directory.is_dir() and str(directory) not in sys.path:
                sys.path.append(str(directory))
        self._hmr = BaseReloader(
            str(self.script),
            [str(self.root), *package.__path__, *map(str, includes)],
            [str(path) for path in _effectful_excludes()],
        )
        finder = sys.meta_path[0]
        assert isinstance(finder, ReactiveModuleFinder)
        self._finder = sys.meta_path[0] = _PinningFinder(finder, self)
        self.includes = [str(path) for path in finder.includes]

        # The reloadable copy of the script, executed on first use; under a
        # private name if its own already belongs to another module.
        existing = sys.modules.get(self.name)
        if existing is not None and self._file(existing) == self.script:
            self.module = existing
        else:
            if existing is not None:
                self.name = f"__autoreload__.{self.name}"
            loader = ReactiveModuleLoader()
            spec = importlib.machinery.ModuleSpec(
                self.name, loader, origin=str(self.script)
            )
            self.module = loader.create_module(spec)
            sys.modules[self.name] = self.module

        base = get_interpretation()
        self._derived = HMR_CONTEXT.derived(
            lambda: coproduct(base, coproduct(build(), gate_interpretation(self._gate)))
        )
        # A build that reads nothing reloadable is legitimate, not a lost dependency.
        self._derived.reactivity_loss_strategy = "ignore"
        self._stack: Interpretation = self._derived()
        self._generation = 0
        self._pin_sources()

    # -- what a host reads ---------------------------------------------------------

    def stack(self) -> Interpretation:
        """The harness stack as last built."""
        return self._stack

    def live(self) -> Interpretation:
        """An interpretation that follows the stack across reloads, as do coproducts with it."""
        return _Live(self)

    def subscribe(self, callback: collections.abc.Callable[["Reloader"], None]) -> None:
        """Call `callback` after each applied reload, before any call runs under it."""
        self._subscribers.append(callback)

    def unsubscribe(
        self, callback: collections.abc.Callable[["Reloader"], None]
    ) -> None:
        with contextlib.suppress(ValueError):
            self._subscribers.remove(callback)

    def keep(self, module: str | types.ModuleType | os.PathLike[str]) -> None:
        """Never re-run `module`, a name, module or file that holds the running process."""
        if isinstance(module, str) and not module.endswith(".py"):
            found = sys.modules.get(module)
            if found is None:
                return
            module = found
        if isinstance(module, types.ModuleType):
            file = self._file(module)
        else:
            file = pathlib.Path(module).resolve()
        if file is not None:
            self._kept.add(file)
            self._pending.discard(file)

    def current_class(self, cls: type) -> type:
        """The class the reloadable module now defines under `cls`'s name, else `cls`."""
        if "<locals>" in cls.__qualname__:
            return cls
        if cls.__module__ in ("__main__", self.name):
            module: typing.Any = self.module
        else:
            module = sys.modules.get(cls.__module__)
        reloadable = module is self.module or self._load(module) is not None
        for part in cls.__qualname__.split("."):
            module = getattr(module, part, None)
            if module is None:
                if reloadable:
                    print(
                        f"note: {cls.__module__}.{cls.__qualname__} is no longer "
                        f"defined; its instances keep the old class",
                        file=sys.stderr,
                    )
                return cls
        return module if isinstance(module, type) else cls

    def refresh(self, agent: object) -> None:
        """Move `agent` onto the class its module now defines; see `rebind`."""
        rebind(agent, self.current_class(type(agent)))

    def turn(self) -> contextlib.AbstractContextManager[None]:
        """Hold reloads off for a whole turn, beyond the calls `gate_interpretation` guards."""
        return self._gate.shared()

    @contextlib.contextmanager
    def suspended(self) -> collections.abc.Iterator[None]:
        """Queue edits instead of applying them; they apply shortly after."""
        self._suspended += 1
        try:
            yield
        finally:
            self._suspended -= 1

    # -- applying ------------------------------------------------------------------

    def apply(
        self,
        files: collections.abc.Iterable[str | os.PathLike[str]] | None = None,
        *,
        wait: bool = True,
    ) -> bool | None:
        """Re-run the edited files now.

        `None` if deferred -- a call is running and `wait` is false, or reloads are
        suspended -- with the files queued; otherwise whether anything was re-run.
        With no `files`, every reloadable file that differs from its running version.
        """
        changed = set(self._pending)
        changed |= (
            self._changed()
            if files is None
            else {pathlib.Path(file).resolve() for file in files}
        )
        self._pending.clear()
        changed &= self._loaded()
        changed -= self._kept
        if not changed:
            return False
        if self._suspended or not self._gate.acquire_exclusive(wait=wait):
            self._pending |= changed
            return None
        try:
            from watchfiles import Change

            self._unpin_sources()
            token = REDEFINING.set(True)
            stack, built = self._stack, False
            try:
                self._hmr.on_events([(Change.modified, str(file)) for file in changed])
                self._pull()
                with self._hmr.error_filter:
                    stack = self._derived()
                    built = True
            finally:
                REDEFINING.reset(token)
            if not built:
                # A failed build keeps the running stack and is retried on the next edit.
                self._derived.dirty = True
            elif stack is not self._stack:
                self._stack, self._generation = stack, self._generation + 1
            self._pin_sources()
            for callback in list(self._subscribers):
                with self._hmr.error_filter:
                    callback(self)
        finally:
            self._gate.release_exclusive()
        return True

    def _pull(self) -> None:
        """Re-run every dirty module here, so no other thread does it lazily later."""
        from reactivity.hmr.core import ReactiveModule

        for _ in range(100):
            dirty = [
                module
                for module in list(ReactiveModule.instances.values())
                if self._file(module) not in self._kept
                and (load := self._load(module)) is not None
                and load.dirty
                and self._executed(load)
            ]
            for module in dirty:
                with self._hmr.error_filter:
                    self._load(module)()
            if not self._forget_removed_names() and not dirty:
                return
        print("note: modules keep re-running each other; giving up", file=sys.stderr)

    def _forget_removed_names(self) -> bool:
        """Drop names a re-run module's file no longer binds; whether any were."""
        from reactivity.hmr.core import ReactiveModule

        pruned = False
        for module in list(ReactiveModule.instances.values()):
            names = vars(module)
            file = self._file(module)
            if "__path__" in names or file is None or file in self._kept:
                continue
            if names["__name__"].startswith(f"{HARNESS}."):
                continue
            try:
                bound = _bound_names(file)
            except OSError:
                continue
            if bound is None:
                continue
            proxy = module._ReactiveModule__namespace_proxy
            for key in list(proxy.raw):
                if key.startswith(("__", "_ReactiveModule__")):
                    continue
                if key not in bound:
                    del proxy[key]
                    pruned = True
        return pruned

    @staticmethod
    def _loaded() -> set[pathlib.Path]:
        """The files of the modules hmr has loaded, which are all an edit can affect."""
        from reactivity.hmr.core import ReactiveModule

        return set(ReactiveModule.instances)

    def _changed(self) -> set[pathlib.Path]:
        changed = set()
        for file, (_, _, lines, _) in self._pinned.items():
            try:
                if pathlib.Path(file).read_text("utf-8") != "".join(lines):
                    changed.add(pathlib.Path(file))
            except OSError:
                continue
        return changed

    def _pin_sources(self) -> None:
        """Serve the running version of each reloadable file from `linecache`.

        Pinned without an mtime, which `linecache.checkcache` leaves alone. A file that
        does not compile keeps its earlier pin, as hmr keeps its earlier version.
        """
        from reactivity.hmr.core import ReactiveModule

        for module in list(ReactiveModule.instances.values()):
            if (file := vars(module).get("__file__")) is not None:
                self._pin_file(file)
        linecache.cache.update(self._pinned)

    def _pin_file(self, file: str) -> None:
        try:
            source = pathlib.Path(file).read_text("utf-8")
            compile(source, file, "exec", dont_inherit=True)
        except (OSError, SyntaxError, ValueError):
            return
        lines = source.splitlines(keepends=True)
        linecache.cache[file] = self._pinned[file] = (len(source), None, lines, file)

    def _unpin_sources(self) -> None:
        for file in self._pinned:
            linecache.cache.pop(file, None)

    @staticmethod
    def _file(module: types.ModuleType) -> pathlib.Path | None:
        file = vars(module).get("__file__")
        return pathlib.Path(file).resolve() if file else None

    @staticmethod
    def _load(module: types.ModuleType) -> typing.Any:
        """hmr's per-module computation, or `None` for a module hmr does not run."""
        return getattr(module, "_ReactiveModule__load", None)

    @staticmethod
    def _executed(load: typing.Any) -> bool:
        from reactivity.primitives import Derived

        return load._value is not Derived.UNSET

    # -- watching ------------------------------------------------------------------

    @staticmethod
    def _paths(events: collections.abc.Iterable[tuple[typing.Any, str]]) -> set[str]:
        from watchfiles import Change

        return {file for change, file in events if change is not Change.deleted}

    def _apply_watched(
        self, events: collections.abc.Iterable[tuple[typing.Any, str]], *, wait: bool
    ) -> None:
        """`apply` for the watcher, which must outlive an error in a reload."""
        try:
            self.apply(self._paths(events), wait=wait)
        except Exception:
            traceback.print_exc()

    def start(self) -> None:
        """Watch from a daemon thread, until `watch` or `close` takes over."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._watch_thread, name="autoreload", daemon=True
        )
        self._thread.start()

    def _watch_thread(self) -> None:
        from watchfiles import PythonFilter, watch

        for events in watch(
            *self.includes,
            watch_filter=PythonFilter(),
            debounce=300,
            stop_event=self._stop,
            rust_timeout=250,
            yield_on_timeout=True,
        ):
            if self._stop.is_set():
                return
            self._apply_watched(events, wait=True)

    async def watch(self) -> None:
        """Watch on the running event loop, applying there; stops the thread `start` began."""
        from watchfiles import PythonFilter, awatch

        self._stop.set()
        async for events in awatch(
            *self.includes,
            watch_filter=PythonFilter(),
            debounce=300,
            rust_timeout=250,
            yield_on_timeout=True,
        ):
            # Deferred rather than awaited: waiting for a turn here would block the
            # loop the turn needs. The next yield retries.
            self._apply_watched(events, wait=False)

    def close(self) -> None:
        """Undo `install`: stop watching and forget every module hmr loaded."""
        global _CURRENT
        from reactivity.hmr.core import ReactiveModule

        self._stop.set()
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(self._finder)
        self._unpin_sources()
        self._derived.dispose()
        for name, module in list(sys.modules.items()):
            if isinstance(module, ReactiveModule):
                del sys.modules[name]
        gc.collect()
        if _CURRENT is self:
            _CURRENT = None


def install(
    script: str | os.PathLike[str],
    build: collections.abc.Callable[[], Interpretation],
    *,
    includes: collections.abc.Iterable[str | os.PathLike[str]] = (),
) -> Reloader:
    """Set hmr up for `script` and build the stack; call before importing what should reload."""
    global _CURRENT
    if _CURRENT is not None:
        raise RuntimeError("autoreload is already installed")
    _CURRENT = Reloader(script, build, includes=includes)
    for module in _KEPT_EARLY:
        _CURRENT.keep(module)
    _KEPT_EARLY.clear()
    return _CURRENT


def current() -> Reloader | None:
    """The process's reloader, if the launcher was given ``--autoreload``."""
    return _CURRENT


def keep(module: str | types.ModuleType | os.PathLike[str]) -> None:
    """`Reloader.keep` on the current reloader; remembered for one installed later."""
    if _CURRENT is not None:
        _CURRENT.keep(module)
    else:
        _KEPT_EARLY.append(module)
