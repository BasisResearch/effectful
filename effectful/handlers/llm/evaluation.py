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
def type_check(source: str, lo: int | None = None, hi: int | None = None) -> None:
    """
    Type check a module source, reporting only diagnostics inside a line region.

    source: A complete module source to check (e.g. produced by
        ``splice_into_source``, which splices generated code into a Template's real
        module source).
    lo, hi: Inclusive line range within ``source`` to report errors from; when
        omitted, the whole source is in scope. Errors outside the region are
        ignored so unrelated pre-existing code never blocks synthesis.

    Returns None, raises TypeError on an in-region failure.
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


def _region_errors(
    stdout: str, lo: int | None, hi: int | None
) -> list[dict[str, Any]]:
    """mypy ``--output=json`` diagnostics of severity ``error`` whose reported
    line falls within ``[lo, hi]`` -- the spliced region. An open bound (``None``)
    is unbounded on that side, so ``lo=hi=None`` reports every error.

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
        if diag["severity"] != "error":
            continue
        if (lo is None or lo <= diag["line"]) and (hi is None or diag["line"] <= hi):
            errors.append(diag)
    return errors


def splice_into_source(
    generated: ast.Module, anchor: Any
) -> tuple[str, int, int] | None:
    """Splice `generated` into the anchor Template's own function body, in its real
    module source.

    Returns the modified module source and the ``[lo, hi]`` line span of the
    spliced body within it, or ``None`` when the anchor's source can't be recovered
    (the caller skips rather than guesses). Raises ``RuntimeError`` if the source is
    recovered but the anchor's def can't be located in it (source drift) -- a real
    error, not a silent pass.

    The generated function -- and any helpers it defines alongside -- becomes the
    body of the Template's own function at its real (possibly nested) position, so
    the generated code is checked in its real lexical scope with no synthesized
    type stubs.
    """
    if not generated.body:
        raise TypeError("splice: generated module is empty")
    last = generated.body[-1]
    if not isinstance(last, ast.FunctionDef | ast.AsyncFunctionDef):
        raise TypeError(
            f"splice: last statement must be a function definition, "
            f"got {type(last).__name__}"
        )
    target_name = last.name

    fn = inspect.unwrap(anchor)  # staticmethod/classmethod -> underlying function
    # Recover the module source via fn's own filename -- a real path or a
    # linecache-registered synthetic name (e.g. <synthesis:...>) for REPL/exec/
    # notebook templates; linecache.getlines reads real files from disk too.
    try:
        source_file = inspect.getsourcefile(fn)
    except TypeError:
        source_file = None
    module_source = "".join(linecache.getlines(source_file)) if source_file else ""
    if not module_source:
        logger.warning("skipping type check: cannot recover source for %r", fn)
        return None
    module_ast = ast.parse(module_source)

    template_def = _find_def_at_lineno(module_ast, fn.__code__.co_firstlineno)
    if template_def is None:
        # The source drifted from what fn was compiled from (e.g. the file was edited after
        # import).
        raise RuntimeError(
            f"cannot locate {getattr(fn, '__qualname__', fn)!r} in its module "
            f"source (source drifted since import?)"
        )

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
    # the spliced *body's* span there. ast.unparse reassigns line numbers but
    # preserves def order, so the def keeps its index in walk order -- take the def
    # at that same index in the re-parsed source.
    #
    # The region is the body (the generated code) only, NOT the def header: the
    # signature and decorators are the Template author's own pre-existing source,
    # which we must not attribute to synthesis. This matters for templates whose
    # module source can't be fully recovered -- notably notebook/REPL cells, which
    # share a runtime namespace but whose recovered source is a single cell missing
    # the other cells' imports, so the signature's own annotations (e.g. `Literal`,
    # `Callable`) look undefined to mypy. Flagging only the body keeps those
    # spurious signature-line diagnostics out of the gate.
    def_index = _def_nodes(module_ast).index(template_def)
    checked_source = ast.unparse(ast.fix_missing_locations(module_ast))
    spliced = _def_nodes(ast.parse(checked_source))[def_index]
    lo = spliced.body[0].lineno  # first generated statement (body is non-empty)
    hi = spliced.end_lineno or lo
    return checked_source, lo, hi


def _mypy_check_region(
    source: str, lo: int | None = None, hi: int | None = None
) -> None:
    """Run mypy on `source` and raise ``TypeError`` if any error diagnostic falls
    within ``[lo, hi]``; raise ``RuntimeError`` if mypy itself fails to run.

    Applies mypy to whatever source it's given -- spliced or otherwise -- and
    reports only the region's errors (the whole source when the region is
    omitted), so pre-existing errors elsewhere in `source` never block synthesis.
    """
    # Run mypy on the source as a temp file (not --command: it hits an argv
    # length limit on large modules). Each call gets an isolated temp dir + cache
    # so parallel decodes don't share -- and deadlock on -- mypy's SQLite cache.
    tmpdir = tempfile.mkdtemp(prefix="effectful_typecheck_")
    try:
        tf_path = os.path.join(tmpdir, "_synthesized.py")
        with open(tf_path, "w", encoding="utf-8") as f:
            f.write(source)
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
    # Exit status >= 2 means mypy itself failed (fatal/usage/internal/syntax) -- a
    # tool failure, not a type error -- and it emits text rather than JSON, so
    # raise `RuntimeError` rather than parse or silently pass.
    if status >= 2:
        raise RuntimeError(
            f"mypy could not check the source:\n{(stdout or '') + (stderr or '')}"
        )
    errors = _region_errors(stdout or "", lo, hi)
    if errors:
        # Not the source: it's large and the model already has the generated code.
        report = "\n".join(json.dumps(e) for e in errors)
        raise TypeError("mypy type check failed:\n" + report)


# Eval Providers


class UnsafeEvalProvider(ObjectInterpretation):
    """UNSAFE provider that handles parse, comple and exec operations
    by shelling out to python *without* any further checks. Only use for testing."""

    @implements(type_check)
    def type_check(
        self, source: str, lo: int | None = None, hi: int | None = None
    ) -> None:
        _mypy_check_region(source, lo, hi)

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
    def type_check(
        self, source: str, lo: int | None = None, hi: int | None = None
    ) -> None:
        _mypy_check_region(source, lo, hi)

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
