"""Re-run edited code while a script served by the launcher keeps running.

Under ``python -m effectful.handlers.llm.harness --autoreload <script>``, `hmr
<https://pypi.org/project/hmr/>`_ re-runs each edited module imported from a directory
on `sys.path` (not the standard library, installed packages, or any module imported
before the reloader started), and what depends on it, as its file watcher sees edits;
an agent's own writes are handed to hmr when its call starts and ends. As hmr runs an
entry file, the reloader runs the script's copy, `Reloader.module`, in an effect; the
running script is ``__main__`` and never re-runs, and the copy's re-runs update its
classes. A re-run keeps the identity of the operations and classes it defines,
updating classes in place, and an agent's next call picks up its edited skills and
system prompt. A module that holds the running process sets ``__autoreload__ =
False``.

Limits: a call may see an edit partway through; a file that does not parse keeps its
previous version, and one that raises keeps what ran before the error; a name an edit
removes stays bound; a class whose layout changed is bound anew; and the script's
module-level code runs again in `Reloader.module`.
"""

import collections.abc
import contextlib
import gc
import importlib.util
import inspect
import os
import pathlib
import sys
import threading
import types
import typing
import weakref

from effectful.internals.runtime import get_interpretation
from effectful.ops.semantics import LiveInterpretation, coproduct, fwd, handler
from effectful.ops.types import REDEFINING, Interpretation


def _layout(cls: type) -> tuple:
    """What an existing class cannot be changed to match."""
    bases = [(base.__module__, base.__qualname__) for base in inspect.getmro(cls)[1:]]
    return type(cls), cls.__basicsize__, cls.__dict__.get("__slots__"), bases


def _update(old: type, new: type, replaced: dict[type, type]) -> bool:
    """Make `old` define what `new` does, if their layouts match, noting `new` in
    `replaced`; whether it could."""
    if _layout(old) != _layout(new):
        return False
    replaced[new] = old
    # Through `type`, past a metaclass's own rules, such as an enum's for its members.
    for name in set(vars(old)) - set(vars(new)):
        with contextlib.suppress(AttributeError, TypeError):
            type.__delattr__(old, name)
    for name, value in vars(new).items():
        if name in ("__dict__", "__weakref__", "__module__") or name.startswith(
            "_abc_"
        ):
            continue
        inner = vars(old).get(name)
        if (
            inspect.isclass(value)
            and inspect.isclass(inner)
            and _update(inner, value, replaced)
        ):
            continue
        with contextlib.suppress(AttributeError, TypeError):
            type.__setattr__(old, name, value)
    return True


def _keep_classes(namespace: dict[str, typing.Any], before: dict[str, type]) -> None:
    """Bind each of the classes `namespace` had `before` again, updated from its re-run,
    and point everything that refers to a re-run's class at the class it updated."""
    replaced: dict[type, type] = {}
    for name, old in before.items():
        new = namespace.get(name)
        if (
            new is old
            or not inspect.isclass(new)
            or new.__qualname__ != old.__qualname__
        ):
            continue
        if _update(old, new, replaced):
            namespace[name] = old
        else:
            print(
                f"note: {name} changed layout; its instances keep the old class",
                file=sys.stderr,
            )
    if not replaced:
        return
    for value in list(namespace.values()):  # a class the edit added, on an updated base
        if inspect.isclass(value) and any(b in replaced for b in value.__bases__):
            with contextlib.suppress(TypeError):
                value.__bases__ = tuple(replaced.get(b, b) for b in value.__bases__)
    for ref in gc.get_referrers(*replaced):
        if isinstance(ref, types.CellType):  # zero-argument `super()`, and closures
            if (kept := replaced.get(ref.cell_contents)) is not None:
                ref.cell_contents = kept
        elif (kept := replaced.get(type(ref))) is not None:  # e.g. enum members
            with contextlib.suppress(TypeError):
                object.__setattr__(ref, "__class__", kept)


def _classes(namespace: dict[str, typing.Any]) -> dict[str, type]:
    """The classes `namespace` defines at its top level."""
    return {
        name: value
        for name, value in namespace.items()
        if inspect.isclass(value)
        and value.__module__ == namespace["__name__"]
        and value.__qualname__ == name
    }


def gate_interpretation(reloader: "Reloader | None" = None) -> Interpretation:
    """Handlers that bring a called agent up to date with `reloader`'s edits, and
    replace the system message of a conversation last run before a reload."""
    # Here rather than at the top, so that hmr, installed after this module was
    # imported, loads them and can re-run them.
    from effectful.handlers.llm.harness.durability.transaction import HistoryBuilder
    from effectful.handlers.llm.harness.hooks import call_agent as agent_op
    from effectful.handlers.llm.harness.hooks import call_system as system_op

    def call_system(*args, **kwargs):
        message = fwd()
        # The transaction's buffer, which the call adopts as a rewrite.
        history = HistoryBuilder.get_history()
        if history and history[0]["role"] == "system":
            history[0] = message
        return message

    def call_agent(skill, *args, **kwargs):
        agent = getattr(skill, "__self__", None)
        state = getattr(agent, "__dict__", None)
        if reloader is None or state is None:
            return fwd()
        reloader.refresh()
        # Looked up before `refresh` brought the class up to date; this looks it up
        # again, which rebuilds it only if an edit changed it.
        if (op := getattr(skill, "__classop__", None)) is not None:
            skill = op.__get__(agent, type(agent))
        # The reload count this agent's conversation last ran under.
        key = "__autoreload_reloads__"
        last, state[key] = state.get(key), reloader.reloads
        try:
            if last is None or last == reloader.reloads:
                return fwd(skill, *args, **kwargs)
            with handler({system_op: call_system}):
                return fwd(skill, *args, **kwargs)
        finally:
            reloader.refresh()  # what the call wrote is live when it returns

    return {agent_op: call_agent}


class Reloader(LiveInterpretation):
    """The launcher's harness stack, rebuilt as the modules it and the script import change."""

    reloads: int
    """How many times a module has re-run."""

    module: types.ModuleType | None
    """The script imported under its own name, or `None` if another module has it."""

    def __init__(
        self,
        script: str | os.PathLike[str],
        build: collections.abc.Callable[[], Interpretation],
    ) -> None:
        try:
            from reactivity.hmr._common import HMR_CONTEXT
            from reactivity.hmr.core import BaseReloader, ReactiveModule
            from reactivity.primitives import Derived
        except ImportError as e:
            raise ImportError(
                "--autoreload needs hmr: pip install effectful[llm]"
            ) from e
        import effectful

        self.script = pathlib.Path(script).resolve()
        self.reloads = self._builds = 0
        self._lock = threading.RLock()
        self._pending: set[pathlib.Path] = set()
        self._closed = False

        # hmr finds modules through `sys.path`, where an editable install need not
        # put effectful.
        root = str(pathlib.Path(effectful.__file__).parents[1])
        if root not in sys.path:
            sys.path.append(root)
        self._hmr = BaseReloader(
            str(self.script), [p or "." for p in sys.path if os.path.isdir(p or ".")]
        )
        self._finder = sys.meta_path[0]
        on_changes = self._hmr.on_changes

        def serialized(files: set[pathlib.Path]) -> None:
            with self._lock:
                on_changes(files)

        self._hmr.on_changes = serialized
        # Every module hmr runs, first or again, runs through here.
        self._load = vars(ReactiveModule)["_ReactiveModule__load"]
        self._original = self._load.method
        self._load.method = self._runner(Derived.UNSET)
        audit = weakref.WeakMethod(self._audit)
        sys.addaudithook(lambda event, args: (hook := audit()) and hook(event, args))

        base = get_interpretation()
        self._derived = HMR_CONTEXT.derived(
            lambda: coproduct(base, coproduct(build(), gate_interpretation(self)))
        )
        # A build that reads nothing reloadable is legitimate, not a lost dependency.
        self._derived.reactivity_loss_strategy = "ignore"
        self._stack: Interpretation = {}
        spec = self._finder.find_spec(self.script.stem, None)
        self.module = None
        if spec is not None and spec.loader is not None:
            self.module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = self.module
        self._effect = HMR_CONTEXT.effect(self._track)

    def _runner(
        self, unset: object
    ) -> collections.abc.Callable[[types.ModuleType], None]:
        """hmr's way of running a module, keeping the identities a re-run replaces."""
        original, lock = self._original, self._lock

        def run(module: types.ModuleType) -> None:
            with lock:
                load = self._load.find(module)
                if load._value is unset:  # its first run
                    original(module)
                    return
                if vars(module).get("__autoreload__") is False:
                    load.reactivity_loss_strategy = "restore"
                    return
                before = _classes(vars(module))
                if module is self.module:  # which stands in for ``__main__``
                    main = vars(sys.modules["__main__"])
                    before |= _classes(main | {"__name__": "__main__"})
                token = REDEFINING.set(True)
                # Reported rather than raised, so one broken edit leaves the rest to run.
                with self._hmr.error_filter:
                    original(module)
                REDEFINING.reset(token)
                _keep_classes(vars(module), before)
                self.reloads += 1

        return run

    def _track(self) -> None:
        """Run the script's copy and read the stack, as hmr runs an entry file, so an
        edit to anything they import re-runs it and what depends on it."""
        if self.module is not None:
            getattr(self.module, "__autoreload__", None)
        with self._hmr.error_filter:  # a stack that fails to build keeps the last
            if (stack := self._derived()) is not self._stack:
                self._stack, self._builds = stack, self._builds + 1

    @property
    def version(self) -> int:
        return self._builds

    def snapshot(self) -> Interpretation:
        return self._stack

    def refresh(self) -> None:
        """Tell hmr about the files this process wrote; it re-runs what they reach."""
        if self._pending:
            paths, self._pending = self._pending, set()
            self._hmr.on_changes(paths)

    def _audit(self, event: str, args: tuple) -> None:
        """Note a module whose file this process opens for writing, for `refresh`."""
        if event != "open" or self._closed:
            return
        path, _, flags = args
        if not isinstance(path, str | os.PathLike) or not isinstance(flags, int):
            return
        if flags & (os.O_WRONLY | os.O_RDWR):
            from reactivity.hmr.core import ReactiveModule

            if (resolved := pathlib.Path(path).resolve()) in ReactiveModule.instances:
                self._pending.add(resolved)

    def start(self) -> None:
        """Watch for edits from a daemon thread, until `close`."""
        from reactivity.hmr.core import SyncReloader

        threading.Thread(
            target=SyncReloader.start_watching, args=(self._hmr,), daemon=True
        ).start()

    def close(self) -> None:
        """Stop watching and forget every module hmr loaded; for tests."""
        from reactivity.hmr.core import ReactiveModule

        self._closed = True
        self._hmr.stop_watching()
        sys.meta_path.remove(self._finder)
        self._load.method = self._original
        self._effect.dispose()
        self._derived.dispose()
        for name, module in list(sys.modules.items()):
            if isinstance(module, ReactiveModule):
                del sys.modules[name]
        gc.collect()
