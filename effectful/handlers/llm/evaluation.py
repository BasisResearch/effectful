import ast
import builtins
import code
import codeop
import collections.abc
import contextlib
import inspect
import io
import json
import linecache
import logging
import os
import shutil
import sys
import tempfile
import typing
from collections.abc import MutableMapping
from types import CodeType
from typing import Any

from mypy import api as mypy_api
from RestrictedPython import (
    Eval,
    Guards,
    RestrictingNodeTransformer,
    compile_restricted,
    safe_globals,
)
from RestrictedPython.PrintCollector import PrintCollector

from effectful.handlers.llm.template import Tool
from effectful.ops.syntax import ObjectInterpretation, defop, implements


@defop
def parse(source: str, filename: str) -> ast.Module:
    """
    Parse source text into an AST.

    source: The Python source code to parse.
    filename: The filename recorded in the resulting AST for tracebacks and tooling.

    Returns the parsed AST.
    """
    raise NotImplementedError(
        "An eval provider must be installed in order to parse code."
    )


@defop
def type_check_anchor() -> Any:
    """The Template's underlying function, anchoring ``type_check`` to the module
    the Template is defined in.

    ``None`` (the default) means there is no Template in scope -- the case for
    tool-argument decoding, which therefore skips the source-anchored type check.
    ``Template.__apply__`` binds this per call via a handler, so nested/recursive
    synthesis sees the right anchor (innermost wins); the tool-argument decode
    rebinds it to ``None``.
    """
    return None


@defop
def type_check(generated: ast.Module, anchor: Any) -> None:
    """
    Type check generated code against the lexical context of a Template.

    generated: The parsed module of LLM-generated code; its last top-level
        function definition is the synthesized target, preceded by any helpers.
    anchor: The Template's underlying function, whose module source supplies the
        lexical context the generated code is checked against.

    Returns None, raises TypeError on an in-region typechecking failure.
    """
    raise NotImplementedError(
        "An eval provider must be installed in order to type check code."
    )


@defop
def compile(module: ast.Module, filename: str) -> CodeType:
    """
    Compile an AST into a Python code object.

    module: The AST to compile (typically produced by parse()).
    filename: The filename recorded in the resulting code object (CodeType.co_filename), used in tracebacks and by inspect.getsource().

    Returns the compiled code object.
    """
    raise NotImplementedError(
        "An eval provider must be installed in order to compile code."
    )


@defop
def exec(
    bytecode: CodeType,
    env: dict[str, Any],
) -> None:
    """
    Execute a compiled code object.

    bytecode: A code object to execute (typically produced by compile()).
    env: The namespace mapping used during execution.

    After ``exec(bytecode, env)`` returns, ``env`` reflects all top-level
    binding effects of the executed code (new names and rebindings alike).
    """
    raise NotImplementedError(
        "An eval provider must be installed in order to execute code."
    )


logger = logging.getLogger(__name__)


def scan_non_nestable(generated: ast.Module) -> None:
    """Reject constructs legal at module level but illegal once nested in a function.

    ``from ... import *`` and ``from __future__ import ...`` are both ``SyntaxError``s
    inside a function body, but mypy *accepts* a nested star import silently, so the
    splice would slip an illegal construct past the type check and fail later at
    ``compile``/``exec``. Detect them explicitly and raise before splicing.
    """
    for stmt in generated.body:
        if isinstance(stmt, ast.ImportFrom):
            if stmt.module == "__future__":
                raise TypeError(
                    "generated code uses `from __future__ import ...`, which is "
                    "illegal once spliced into a function body"
                )
            if any(alias.name == "*" for alias in stmt.names):
                raise TypeError(
                    "generated code uses a star import (`from ... import *`), which "
                    "is illegal once spliced into a function body"
                )


def _unwrap(fn: Any) -> Any:
    """Unwrap a ``staticmethod``/``classmethod`` to its underlying function."""
    return fn.__func__ if isinstance(fn, staticmethod | classmethod) else fn


def _recover_module_source(fn: Any) -> str | None:
    """Recover the full source of the module that lexically contains ``fn``.

    Reads the real file when one exists, else the ``linecache``-registered source
    (REPL/``exec``-defined templates, e.g. ``<synthesis:...>`` filenames). Returns
    ``None`` when neither is available, so the caller can skip rather than guess.
    ``inspect.getsourcefile`` and ``os.path.isfile`` are both given the *unwrapped*
    function.
    """
    try:
        filename = inspect.getsourcefile(fn)
    except TypeError:
        return None
    if not filename:
        return None
    if os.path.isfile(filename):
        try:
            with open(filename, encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None
    lines = linecache.getlines(filename)
    return "".join(lines) if lines else None


def _def_nodes(
    module: ast.Module,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """All function definitions in ``module``, in a stable order that an
    ``ast.unparse`` -> ``ast.parse`` round-trip preserves (so a def keeps its
    index across it)."""
    return [
        n
        for n in ast.walk(module)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    ]


def _find_def_at_lineno(
    module: ast.Module, lineno: int
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Locate the function definition whose definition site is ``lineno``.

    Matches ``fn.__code__.co_firstlineno`` -- the first decorator line, or the
    ``def`` line when undecorated -- which identifies the def directly and
    unambiguously (no name matching, and nesting-agnostic). Returns None only if
    no def starts there: a dynamically generated ``fn`` with no source def, or
    source that has drifted since import.
    """
    for node in _def_nodes(module):
        start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        if start == lineno:
            return node
    return None


def _region_errors(stdout: str, lo: int, hi: int) -> list[dict[str, Any]]:
    """mypy ``--output=json`` diagnostics of severity ``error`` whose reported
    line falls within ``[lo, hi]`` -- the spliced region.

    ``--output=json`` emits one JSON object per diagnostic carrying mypy's own
    ``severity`` and ``line`` fields, so we filter on those directly rather than
    parsing (and risking mis-parsing) its human-readable format.  Only reached
    for exit status < 2; a fatal status emits text, not JSON, and is handled by
    the caller before this runs.
    """
    errors: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        diag = json.loads(line)
        if diag["severity"] == "error" and lo <= diag["line"] <= hi:
            errors.append(diag)
    return errors


def mypy_type_check(generated: ast.Module, anchor: Any) -> None:
    """Type-check LLM-generated code by splicing it into the real module source.

    The generated function — and any helpers it defines alongside — is spliced as
    the body of the Template's own function, at its real (possibly nested) position
    in the module, and the whole module is type-checked with mypy. Only diagnostics
    that fall inside the spliced region are raised, so unrelated pre-existing errors
    elsewhere in the module never block synthesis. The generated code is thereby
    checked in its real lexical scope, with no synthesized type stubs.

    ``anchor`` is the Template's underlying function, whose module supplies the
    lexical context. When its source can't be recovered, the check is skipped with
    a logged reason (never a silent pass on a real error). Raises ``TypeError`` with
    the mypy report on an in-region failure.
    """
    if not generated.body:
        raise TypeError("mypy_type_check: generated module is empty")
    last = generated.body[-1]
    if not isinstance(last, ast.FunctionDef | ast.AsyncFunctionDef):
        raise TypeError(
            f"mypy_type_check: last statement must be a function definition, "
            f"got {type(last).__name__}"
        )
    target_name = last.name

    fn = _unwrap(anchor)
    module_source = _recover_module_source(fn)
    if module_source is None:
        logger.warning("skipping type check: cannot recover source for %r", fn)
        return None
    module_ast = ast.parse(module_source)

    template_def = _find_def_at_lineno(module_ast, fn.__code__.co_firstlineno)
    if template_def is None:
        logger.warning(
            "skipping type check: cannot locate %r in its module source",
            getattr(fn, "__qualname__", fn),
        )
        return None

    # Splice in place: replace the body with the generated body and bind the
    # target against the (source) return annotation via `return`. Decorators are
    # left untouched -- mypy checks a function's body against its declared return
    # type regardless of decorators (even an unresolvable / `Any` one), and the
    # decorator application itself doesn't spuriously fail, so touching the
    # surrounding source as little as possible keeps the splice robust.
    template_def.body = [
        *generated.body,
        ast.Return(ast.Name(target_name, ast.Load())),
    ]

    # mypy reports line numbers in the coordinates of `checked_source`, so we need
    # the spliced def's span there. ast.unparse reassigns line numbers but preserves
    # def order, so the def keeps its index in walk order -- take the def at that
    # same index in the re-parsed source.
    def_index = _def_nodes(module_ast).index(template_def)
    checked_source = ast.unparse(ast.fix_missing_locations(module_ast))
    spliced = _def_nodes(ast.parse(checked_source))[def_index]
    lo, hi = spliced.lineno, (spliced.end_lineno or spliced.lineno)

    # Type-check the spliced module by writing it to a file and running mypy on
    # it. Each call gets an isolated temp dir holding both the module and mypy's
    # cache, for two reasons: (1) `--command`, which the previous implementation
    # used on a tiny stub, passes the whole source as one argv string and hits an
    # OS filename-length limit on large modules (`[Errno 36] File name too long`);
    # (2) parallel decodes (pytest-xdist, concurrent synthesis) must not share
    # mypy's SQLite cache, which otherwise deadlocks ("database is locked"). A
    # persistent shared `--cache-dir`/incremental setup, and preserving the
    # module's sibling-import resolution by writing alongside it, are perf
    # follow-ups.
    tmpdir = tempfile.mkdtemp(prefix="effectful_typecheck_")
    try:
        tf_path = os.path.join(tmpdir, "_synthesized.py")
        with open(tf_path, "w", encoding="utf-8") as f:
            f.write(checked_source)
        stdout, stderr, status = mypy_api.run(
            [
                tf_path,
                "--cache-dir",
                os.path.join(tmpdir, "cache"),
                "--no-error-summary",
                "--output=json",
                "--ignore-missing-imports",
                "--disable-error-code=import-untyped",
            ]
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    if status >= 2:
        # Exit code >= 2 means mypy itself failed (fatal/usage/internal/syntax);
        # it emits text rather than JSON, so surface it rather than parse or
        # silently pass.
        raise TypeError(
            f"mypy could not check the spliced module:\n{(stdout or '') + (stderr or '')}"
        )
    errors = _region_errors(stdout or "", lo, hi)
    if errors:
        report = "\n".join(
            f"{e['line']}: {e['message']}  [{e['code']}]" for e in errors
        )
        raise TypeError("mypy type check failed:\n" + report + "\n\n" + checked_source)
    return None


# Eval Providers


class UnsafeEvalProvider(ObjectInterpretation):
    """UNSAFE provider that handles parse, comple and exec operations
    by shelling out to python *without* any further checks. Only use for testing."""

    @implements(type_check)
    def type_check(self, generated: ast.Module, anchor: Any) -> None:
        mypy_type_check(generated, anchor)

    @implements(parse)
    def parse(self, source: str, filename: str) -> ast.Module:
        # Cache source under `filename` so inspect.getsource() can retrieve it later.
        # inspect uses f.__code__.co_filename -> linecache.getlines(filename)
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(True),
            filename,
        )

        return ast.parse(source, filename=filename, mode="exec")

    @implements(compile)
    def compile(self, module: ast.AST, filename: str) -> CodeType:
        return builtins.compile(typing.cast(typing.Any, module), filename, "exec")

    @implements(exec)
    def exec(
        self,
        bytecode: CodeType,
        env: dict[str, Any],
    ) -> None:
        # Ensure builtins exist in the execution environment.
        env.setdefault("__builtins__", __builtins__)

        # Execute module-style so top-level defs land in `env`.
        builtins.exec(bytecode, env, env)


class _StdoutPrintCollector(PrintCollector):
    """`_print_` factory whose `print(...)` writes to the real `sys.stdout`
    (so output-capturing callers see it) rather than accumulating into the
    collector's discarded `printed` buffer."""

    def _call_print(self, *objects, **kwargs):
        kwargs.setdefault("file", sys.stdout)
        builtins.print(*objects, **kwargs)


class RestrictedEvalProvider(ObjectInterpretation):
    """
    Safer provider using RestrictedPython.

    RestrictedPython is not a complete sandbox, but it enforces a restricted
    language subset and expects you to provide a constrained exec environment.

    policy : dict[str, Any], optional
        RestrictedPython compile_restricted policy for compilation
    """

    policy: type[RestrictingNodeTransformer] | None = None

    def __init__(
        self,
        *,
        policy: type[RestrictingNodeTransformer] | None = None,
    ):
        self.policy = policy

    @implements(type_check)
    def type_check(self, generated: ast.Module, anchor: Any) -> None:
        mypy_type_check(generated, anchor)

    @implements(parse)
    def parse(self, source: str, filename: str) -> ast.Module:
        # Keep inspect.getsource() working for dynamically-defined objects.
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(True),
            filename,
        )
        return ast.parse(source, filename=filename, mode="exec")

    @implements(compile)
    def compile(self, module: ast.Module, filename: str) -> CodeType:
        # RestrictedPython can compile from an AST directly.
        return compile_restricted(
            module,
            filename=filename,
            mode="exec",
            policy=self.policy or RestrictingNodeTransformer,
        )

    @implements(exec)
    def exec(
        self,
        bytecode: CodeType,
        env: dict[str, Any],
    ) -> None:
        # Build restricted globals from RestrictedPython's defaults
        rglobals: dict[str, Any] = safe_globals.copy()

        # Enable class definitions (required for Python 3)
        rglobals["__metaclass__"] = type
        rglobals["__name__"] = "restricted"

        # Layer `env` on top (without letting callers replace the restricted builtins).
        rglobals.update({k: v for k, v in env.items() if k != "__builtins__"})

        # Enable for loops and comprehensions
        rglobals["_getiter_"] = Eval.default_guarded_getiter
        # Enable sequence unpacking in comprehensions and for loops
        rglobals["_iter_unpack_sequence_"] = Guards.guarded_iter_unpack_sequence

        rglobals["getattr"] = Guards.safer_getattr
        rglobals["setattr"] = Guards.guarded_setattr
        rglobals["_write_"] = lambda x: x

        # RestrictedPython rewrites `print(...)` into its `_print_` collector
        # protocol; route it to the real stdout so output-capturing callers
        # (e.g. redirect_stdout) see it instead of a discarded collector.
        rglobals["_print_"] = _StdoutPrintCollector

        # Snapshot value identities before execution so we can copy back every
        # *binding effect* — both new names and rebindings of seeded names.
        before = dict(rglobals)
        builtins.exec(bytecode, rglobals, rglobals)

        sentinel = object()
        env.update(
            {
                key: value
                for key, value in rglobals.items()
                if key != "__builtins__" and before.get(key, sentinel) is not value
            }
        )


class _OpCommandCompiler(codeop.CommandCompiler):
    """A `codeop.CommandCompiler` that routes compilation through the
    `parse`/`compile` effect operations (so the installed eval provider owns it
    and `parse` populates `linecache`), replacing the native single-mode
    compiler that `code.InteractiveInterpreter` installs.
    """

    def __call__(
        self, source: str, filename: str = "<input>", symbol: str = "single"
    ) -> CodeType:
        # `runsource` passes symbol="single"; we ignore it and compile in the
        # exec mode the ops produce, so a complete multi-statement block runs in
        # one shot.  Incomplete/invalid input raises SyntaxError, which
        # `runsource` routes to `showsyntaxerror` (we do not buffer partial input
        # -- there is no line-at-a-time protocol).
        return compile(parse(source, filename), filename)


class ReplSession(code.InteractiveInterpreter):
    """A persistent, output-capturing Python session seeded from a lexical
    context.

    `exec_code(source)` runs a pre-compiled code object in `self.locals` through
    the `exec` effect operation.  Both bindings and captured stdout/stderr
    persist across calls -- variables, imports and definitions accumulate exactly
    like a REPL -- and the session (with its buffer) is discarded as a whole when
    it goes out of scope.  Each call returns only the output it produced; a
    snippet that raises has its traceback appended to that output rather than
    propagating -- mirroring `code.InteractiveInterpreter`, only `SystemExit`
    propagates -- so failures are surfaced as text.  There is no bare-expression
    auto-echo, so use `print()` to surface values.

    Compilation -- and therefore syntax checking -- happens earlier, at the
    `Encodable[CodeType]` boundary; this session only executes.
    """

    # The session's captured output, accumulated across calls and exposed for
    # introspection.  stdout (`print` output) and stderr (writes plus tracebacks)
    # are kept separate; `exec_code` returns each call's slice of both.
    stdout: io.StringIO
    stderr: io.StringIO

    def __init__(self, env: MutableMapping[str, Any]):
        # Run in a fresh writable dict seeded with a flat view of `env`.  This is
        # forced by `exec`: its globals must be one real dict (a ChainMap is
        # rejected), and a REPL needs a single persistent namespace so a function
        # defined in one snippet sees a name a later snippet binds.  Seeding a flat
        # copy also leaves the lexical seed untouched, so REPL assignments never
        # leak into the surrounding scope.
        scope: dict[str, Any] = dict(env)
        # When `env` is the per-call `ChainMap` (its outer layers are read-only
        # frame proxies), splice this dict in as an extra shadowing first layer so
        # the bindings are *also* visible to the rest of the Template call
        # (mirroring `exec`) -- still scoped to the call, since that ChainMap is.
        if isinstance(env, collections.ChainMap):
            env.maps.insert(0, scope)
        # `InteractiveInterpreter.__init__` stores it as `self.locals`, so we reuse
        # the base's runcode/showtraceback/write machinery.
        super().__init__(scope)
        # Route `runsource`'s compilation through the `parse`/`compile` ops too, so
        # it stays consistent with our `runcode` (which execs through the `exec`
        # op) rather than the native single-mode compiler the base installed.
        self.compile = _OpCommandCompiler()
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def runcode(self, code: CodeType) -> None:
        # Mirrors `InteractiveInterpreter.runcode` exactly; the only difference
        # is that `exec` here is the effect operation, so execution routes
        # through the installed eval provider.  `showtraceback` reports failures
        # via `self.write`, which `exec_code` has redirected into `self.stderr`.
        try:
            exec(code, self.locals)
        except SystemExit:
            raise
        except:
            self.showtraceback()

    @Tool.define
    def exec_code(self, code: CodeType) -> str:
        """Run Python in a persistent, stateful session and return its output.

        This is a long-lived REPL, not a one-shot sandbox: every call runs in the
        SAME namespace, so names you bind in one call stay available in later
        calls within the same task.  Imports, function/class definitions and
        variable assignments all accumulate during the session of this template.
        The namespace starts seeded with the in-scope variables of the surrounding context, which you may read and
        rebind.

        Output: returns this call's output -- its stdout (what `print` wrote)
        followed by its stderr (which includes the traceback if the code raised).
        There is NO automatic echoing of results -- a bare expression on its own
        line (e.g. `1 + 1`) displays nothing, so call `print(...)` for anything
        you want to see.  A snippet that raises has its traceback returned and the
        session survives, so you can read the error and continue in the next call
        (only `SystemExit` aborts).

        Provide `code` as a string of Python source.  It must be a complete,
        compilable snippet -- incomplete or invalid source is rejected before it
        runs.
        """
        out_start = self.stdout.tell()
        err_start = self.stderr.tell()
        with (
            contextlib.redirect_stdout(self.stdout),
            contextlib.redirect_stderr(self.stderr),
        ):
            self.runcode(code)
        return self.stdout.getvalue()[out_start:] + self.stderr.getvalue()[err_start:]
