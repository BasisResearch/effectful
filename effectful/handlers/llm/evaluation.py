import ast
import builtins
import code
import codeop
import collections.abc
import contextlib
import copy
import doctest
import inspect
import io
import json
import linecache
import logging
import operator
import os
import shutil
import string
import subprocess
import sys
import tempfile
import types
import typing
import warnings

from RestrictedPython import (
    Eval,
    Guards,
    RestrictingNodeTransformer,
    compile_restricted,
    safe_globals,
)
from RestrictedPython.PrintCollector import PrintCollector
from RestrictedPython.transformer import (
    FORBIDDEN_FUNC_NAMES,
    INSPECT_ATTRIBUTES,
    copy_locations,
)

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
def type_check(
    source: str,
    lo: int | None = None,
    hi: int | None = None,
    *,
    lenient: bool = False,
) -> None:
    """
    Type check a module source, reporting only diagnostics inside a line region.

    source: A complete module source to check (e.g. produced by
        ``splice_into_source``, which splices generated code into a Template's real
        module source).
    lo, hi: Inclusive line range within ``source`` to report errors from; when
        omitted, the whole source is in scope. Errors outside the region are
        ignored so unrelated pre-existing code never blocks synthesis.
    lenient: when True, relax mypy for incrementally-built REPL code spliced into a
        Template body -- allow redefinition (a cell may rebind or redefine a name)
        and don't require the body to satisfy the Template's return type. Off (strict)
        for a synthesized ``Callable`` or ``TemplateBody``, which must honor its
        signature and gets no redefinition slack.

    Returns None, raises TypeError on an in-region failure.
    """
    raise NotImplementedError(
        "An eval provider must be installed in order to type check code."
    )


@defop
def run_doctests(
    obj: collections.abc.Callable | type | types.ModuleType,
    globs: collections.abc.Mapping[str, typing.Any],
) -> None:
    """Run the doctests found in a synthesized object's docstring.

    obj: The synthesized object (typically a function) whose docstring may
        contain interactive ``>>>`` examples.
    globs: The namespace the examples execute in (typically the exec namespace,
        which already contains the function plus its lexical context).

    Returns None, raises TypeError if any doctest example fails. A docstring
    with no examples is a no-op (passes trivially).

    Unlike the other operations here, this one carries its mechanics in its
    default rule: running examples needs nothing an eval provider owns beyond
    the compiler `doctest` itself uses, so it works with no provider installed.
    A provider overrides it only to change *how* the examples are executed --
    `RestrictedEvalProvider` does, to run them under the same restrictions as
    the code they exercise.
    """
    return _run_doctests(obj, globs)


@defop
def compile(module: ast.Module, filename: str) -> types.CodeType:
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
    bytecode: types.CodeType,
    env: dict[str, typing.Any],
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


# The shared output of the three splicers (`splice_into_source`,
# `splice_template_body`, `splice_repl_code_into_body`): the module ``source`` to
# type-check and the inclusive ``[lo, hi]`` line span within it to report
# diagnostics from -- exactly the leading arguments of `type_check`. ``None`` (not
# this type) is returned when the anchor's source can't be recovered.
type SplicedRegion = tuple[str, int, int]


def scan_non_nestable(generated: ast.Module) -> None:
    """Reject constructs legal at module level but illegal once nested in a function.

    ``from ... import *`` and ``from __future__ import ...`` are both ``SyntaxError``s
    inside a function body, but mypy *accepts* a nested star import silently, so the
    splice would slip an illegal construct past the type check and fail later at
    ``compile``/``exec``. Detect them explicitly and raise before splicing. Raises
    ``ValueError`` (this is rejecting invalid generated *source*, not signaling a type
    error), so a decoder can catch it alongside ``SyntaxError`` without swallowing a real
    ``TypeError`` from a broken provider.
    """
    for stmt in generated.body:
        if isinstance(stmt, ast.ImportFrom):
            if stmt.module == "__future__":
                raise ValueError(
                    "generated code uses `from __future__ import ...`, which is "
                    "illegal once spliced into a function body"
                )
            if any(alias.name == "*" for alias in stmt.names):
                raise ValueError(
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
) -> list[dict[str, typing.Any]]:
    """mypy ``--output=json`` diagnostics of severity ``error`` whose reported
    line falls within ``[lo, hi]`` -- the spliced region. An open bound (``None``)
    is unbounded on that side, so ``lo=hi=None`` reports every error.

    ``--output=json`` emits one JSON object per diagnostic carrying mypy's own
    ``severity`` and ``line`` fields, so we filter on those directly rather than
    parsing (and risking mis-parsing) its human-readable format.  Only reached
    for exit status < 2; a fatal status emits text, not JSON, and is handled by
    the caller before this runs.
    """
    errors: list[dict[str, typing.Any]] = []
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
    generated: ast.Module, anchor: collections.abc.Callable[..., typing.Any]
) -> SplicedRegion | None:
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

    This is the splice for a Template whose *return type* is a callable (the model
    writes a function and the Template returns it). Example. For the Template ::

        @Template.define
        def make_adder(n: int) -> Callable[[int], int]:
            '''Return a function that adds {n}.'''

    a model that submits this ``generated`` (its last statement is the function to
    return) ::

        def adder(x: int) -> int:
            return x + n

    becomes the whole Template body followed by ``return <its name>`` ::

        @Template.define
        def make_adder(n: int) -> Callable[[int], int]:
            def adder(x: int) -> int:
                return x + n
            return adder

    so mypy checks that ``adder`` satisfies ``Callable[[int], int]`` and that its
    body may reference the Template's ``n``. Contrast `splice_template_body`, which
    grafts the model's function *body* under the Template's own header (for a
    Template whose body -- not return value -- is synthesized). The returned
    ``[lo, hi]`` spans the generated statements only, not the ``def`` header.
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

    recovered = _recover_template_def(anchor)
    if recovered is None:
        return None
    module_ast, template_def = recovered

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


def splice_template_body(
    generated: ast.Module, anchor: collections.abc.Callable[..., typing.Any]
) -> SplicedRegion | None:
    """Splice a synthesized function in as the anchor Template's *own body*.

    Unlike `splice_into_source` (which appends ``return <fn>`` and checks that the
    Template returns the synthesized *function*), this treats the synthesized
    function as the Template's implementation: the Template keeps its own
    authoritative signature and its body becomes ``[<helpers/imports the model
    wrote>, *<the synthesized function's body>]``.  mypy then checks that body
    against the Template's declared parameter and return types -- so a body that
    fails to return the declared type is rejected.  The synthesized function's own
    parameter list (including any ``self``) is intentionally discarded: the
    Template's real signature is the contract.

    ``generated`` is the model's whole ``module_code`` parsed to a module; its
    *last* statement is the implementation, any earlier statements are helper
    definitions/imports.  For example, given the Template ::

        @Template.define
        def parity(numbers: Sequence[int]) -> bool:
            '''True iff the sum of {numbers} is odd.
            >>> parity([1, 2])
            True
            '''

    a model that submits this ``generated`` (note the header on its final ``def``
    -- ``numbers: list`` -- is discarded) ::

        import math
        def _odd(n: int) -> bool:
            return n % 2 == 1
        def parity(numbers: list) -> bool:
            return _odd(sum(numbers))

    is spliced into the Template's real source as ::

        @Template.define
        def parity(numbers: Sequence[int]) -> bool:   # authoritative header kept
            import math
            def _odd(n: int) -> bool:
                return n % 2 == 1
            return _odd(sum(numbers))                  # from the final def's body

    so mypy checks the grafted body against ``numbers: Sequence[int]`` and
    ``-> bool``.  The helper ``_odd`` and ``import math`` (everything before the
    final ``def``) become locals at the top of the body; only the final ``def``'s
    *body* is taken, under the Template's own header.

    Returns the modified module source and the ``[lo, hi]`` line span from the
    ``def`` line through the last body line, or ``None`` when the anchor's source
    can't be recovered (REPL/notebook template -- the caller skips rather than
    guesses). Raises ``RuntimeError`` on source drift, via `_recover_template_def`.
    """
    if not generated.body:
        raise TypeError("splice: generated module is empty")
    last = generated.body[-1]
    if not isinstance(last, ast.FunctionDef | ast.AsyncFunctionDef):
        raise TypeError(
            f"splice: last statement must be a function definition, "
            f"got {type(last).__name__}"
        )

    recovered = _recover_template_def(anchor)
    if recovered is None:
        return None
    module_ast, template_def = recovered

    # Keep the Template's real header (authoritative annotations, `self` for
    # methods); replace only its body with the model's helpers/imports followed by
    # the synthesized function's body statements, so the declared return type is
    # enforced. Any docstring/doctests in the recovered source are dropped.
    template_def.body = [*generated.body[:-1], *last.body]

    # Report the def line through the end of the body. Unlike `splice_into_source`,
    # the region starts at the `def` line (not the first body statement): mypy
    # anchors "Missing return statement"/"empty-body" there, and a body that doesn't
    # return the Template's declared type is a real defect we want to catch. The
    # header is the Template's own (recovered, resolvable) signature -- sourceless
    # templates return `None` above and skip -- so including it adds no spurious
    # signature diagnostics. Decorator lines sit above `spliced.lineno` and stay out.
    # `template_def` is still a node in `module_ast` (only its body changed), so its
    # walk-order index is stable across the unparse round-trip.
    def_index = _def_nodes(module_ast).index(template_def)
    checked_source = ast.unparse(ast.fix_missing_locations(module_ast))
    spliced = _def_nodes(ast.parse(checked_source))[def_index]
    lo = spliced.lineno
    hi = spliced.body[-1].end_lineno or lo
    return checked_source, lo, hi


def _recover_template_def(
    anchor: collections.abc.Callable[..., typing.Any],
) -> tuple[ast.Module, ast.FunctionDef | ast.AsyncFunctionDef] | None:
    """Locate the anchor Template's own ``def`` in its real module source.

    Returns the parsed module AST and the def node, or ``None`` when the source can't
    be recovered (REPL/exec/notebook Template with no linecache entry -- the caller
    skips rather than guesses). Raises ``RuntimeError`` on source drift (source
    recovered but the def no longer sits where ``fn`` was compiled from).
    """
    # `anchor` is the enclosing `Template` (an `Operation`), a bound method, or a
    # plain function; `inspect.unwrap` follows the `__wrapped__` chain that
    # `Operation`/method binding sets up, resolving all of them to the original
    # source-backed function (staticmethod/classmethod included).
    fn = inspect.unwrap(anchor)
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
        raise RuntimeError(
            f"cannot locate {getattr(fn, '__qualname__', fn)!r} in its module "
            f"source (source drifted since import?)"
        )
    return module_ast, template_def


def splice_repl_code_into_body(
    generated: ast.Module, anchor: collections.abc.Callable[..., typing.Any]
) -> SplicedRegion | None:
    """Splice REPL code -- ``generated`` -- into the anchor Template's body, in its
    real module source, and return the modified source with the ``[lo, hi]`` line
    span of the spliced statements.

    ``generated`` is the cumulative session code (any already-run snippets followed
    by the current one; the caller prepends them). It becomes the Template function's
    body at its real (possibly nested) position, so the Template's parameters and
    enclosing scope -- i.e. the session's seed env -- are in scope and each statement
    sees the ones before it (they are function locals). No ``return`` is appended;
    the REPL code doesn't produce the Template's declared type, and that contract is
    waived by ``lenient`` type checking. The whole spliced body is reported, but the
    already-run snippets are type-clean (they passed this same check when *they* were
    the current one), so only the new statements can raise.

    Example. For the Template ::

        @Template.define
        def analyze(data: list[int]) -> str:
            '''Analyze {data}.'''

    a ``generated`` module of accumulated session statements ::

        total = sum(data)
        print(total / len(data))

    becomes the Template's body ::

        @Template.define
        def analyze(data: list[int]) -> str:
            total = sum(data)
            print(total / len(data))

    so each statement sees the Template's ``data`` and the earlier statements'
    bindings (here ``total``).

    Returns ``None`` when ``generated`` has no statements to check, or when the
    Template's source can't be recovered -- a Template defined at a REPL, in a notebook, or
    via ``exec()`` is sourceless, so we skip the check and run the code unchecked, exactly
    as ``splice_into_source`` does for a sourceless Callable anchor. Raises ``RuntimeError``
    only on source *drift* (source recovered but the def no longer sits where it was
    compiled from), which ``_recover_template_def`` surfaces.
    """
    # An empty or comment-only module parses to zero statements: nothing to check.
    if not generated.body:
        return None
    # None means the Template's source can't be recovered (REPL/exec/notebook-defined) --
    # skip, like the Callable path, rather than break the tool; `_recover_template_def`
    # raises on source drift, which is a real error and propagates.
    recovered = _recover_template_def(anchor)
    if recovered is None:
        return None
    module_ast, template_def = recovered
    template_def.body = list(generated.body)

    # `template_def` is still a node in `module_ast` (only its body changed), so its
    # walk-order index is stable across the unparse round-trip.
    def_index = _def_nodes(module_ast).index(template_def)
    checked_source = ast.unparse(ast.fix_missing_locations(module_ast))
    spliced = _def_nodes(ast.parse(checked_source))[def_index]
    lo = spliced.body[0].lineno
    hi = spliced.body[-1].end_lineno or lo
    return checked_source, lo, hi


def _mypy_check_region(
    source: str,
    lo: int | None = None,
    hi: int | None = None,
    lenient: bool = False,
) -> None:
    """Run mypy on `source` and raise ``TypeError`` if any error diagnostic falls
    within ``[lo, hi]``; raise ``RuntimeError`` if mypy itself fails to run.

    Applies mypy to whatever source it's given -- spliced or otherwise -- and
    reports only the region's errors (the whole source when the region is
    omitted), so pre-existing errors elsewhere in `source` never block synthesis.

    When ``lenient`` (for REPL code spliced into a Template body): allow a variable to be
    redefined with a new type across cells (``--allow-redefinition``), a def/class/import
    to be redefined (``no-redef``), and the body not to return the Template's declared type
    (``return``/``empty-body``). All normal for an incrementally-built REPL, not real errors.
    """
    lenient_flags = (
        [
            "--allow-redefinition",
            "--disable-error-code=no-redef",
            "--disable-error-code=return",
            "--disable-error-code=empty-body",
        ]
        if lenient
        else []
    )
    # Run mypy as a subprocess, not the in-process `mypy.api.run`: the API builds
    # typeshed and a full module graph inside this process and never returns that
    # memory, so under a test/agent session doing many checks it accumulates to many
    # GB (OOM). A subprocess reclaims all of it on exit. Pass a file (not --command:
    # it hits an argv length limit on large modules); each call gets an isolated temp
    # dir + cache so parallel decodes don't share -- and deadlock on -- mypy's cache.
    tmpdir = tempfile.mkdtemp(prefix="effectful_typecheck_")
    try:
        tf_path = os.path.join(tmpdir, "_synthesized.py")
        with open(tf_path, "w", encoding="utf-8") as f:
            f.write(source)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                tf_path,
                "--cache-dir",
                os.path.join(tmpdir, "cache"),
                "--no-error-summary",
                "--output=json",
                "--ignore-missing-imports",
                "--disable-error-code=import-untyped",
                *lenient_flags,
            ],
            capture_output=True,
            text=True,
        )
        stdout, stderr, status = proc.stdout, proc.stderr, proc.returncode
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


def _run_doctests(
    obj: collections.abc.Callable | type | types.ModuleType,
    globs: collections.abc.Mapping[str, typing.Any],
) -> None:
    """The mechanics behind the `run_doctests` operation: find the interactive
    examples in ``obj``'s own docstring and run them in ``globs``, raising
    ``TypeError`` with the failure report if any example fails.

    Compilation of each example is `doctest`'s own (see `_doctest_compiled_with`,
    which a provider uses to substitute its compiler), so this is shared by every
    provider; only the compiler differs.
    """
    assert hasattr(obj, "__name__")
    name = obj.__name__
    finder = doctest.DocTestFinder(recurse=False)
    runner = doctest.DocTestRunner(verbose=False)
    # Collect each example's want/got report via `out=...` and read failure
    # counts from `run`'s return value, avoiding `summarize`, which would print
    # to stdout instead of returning the report.
    output: list[str] = []
    failed = attempted = 0
    for test in finder.find(obj, name=name, globs=dict(globs)):
        results = runner.run(test, out=output.append)
        failed += results.failed
        attempted += results.attempted
    if failed:
        report = "".join(output).strip()
        if not report:
            report = f"{failed} doctest(s) failed out of {attempted} attempted."
        raise TypeError(f"doctest failed:\n{report}")
    return None


@contextlib.contextmanager
def _doctest_compiled_with(
    compiler: collections.abc.Callable[..., types.CodeType],
) -> collections.abc.Iterator[None]:
    """Run the examples inside this block through ``compiler`` instead of the
    built-in `compile`.

    `doctest.DocTestRunner` compiles and runs every example with a bare
    ``exec(compile(source, filename, "single", flags, True), test.globs)``, with no
    hook to supply a different compiler. Both names resolve as globals of the
    `doctest` module before falling back to builtins, so binding ``compile`` there
    for the duration of the run redirects example compilation without
    reimplementing (and having to track) the runner. ``compiler`` is called with
    `compile`'s positional signature, which `RestrictedPython.compile_restricted`
    already matches.

    This rebinds module state, so it is not safe against another thread running
    doctests concurrently under a *different* compiler; the block is short and
    holds only for the examples of one synthesized object.
    """
    sentinel = object()
    original = doctest.__dict__.get("compile", sentinel)
    doctest.compile = compiler  # type: ignore[attr-defined]
    try:
        yield
    finally:
        # Restore `doctest` exactly as we found it -- normally by *removing* the
        # global again, so `compile` resolves to the builtin as it did before.
        if original is sentinel:
            del doctest.compile  # type: ignore[attr-defined]
        else:
            doctest.compile = original  # type: ignore[attr-defined]


class UnsafeEvalProvider(ObjectInterpretation):
    """UNSAFE provider that handles parse, comple and exec operations
    by shelling out to python *without* any further checks. Only use for testing."""

    @implements(type_check)
    def type_check(
        self,
        source: str,
        lo: int | None = None,
        hi: int | None = None,
        *,
        lenient: bool = False,
    ) -> None:
        _mypy_check_region(source, lo, hi, lenient)

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
    def compile(self, module: ast.AST, filename: str) -> types.CodeType:
        return builtins.compile(typing.cast(typing.Any, module), filename, "exec")

    @implements(exec)
    def exec(
        self,
        bytecode: types.CodeType,
        env: dict[str, typing.Any],
    ) -> None:
        # Ensure builtins exist in the execution environment.
        env.setdefault("__builtins__", __builtins__)

        # Execute module-style so top-level defs land in `env`.
        builtins.exec(bytecode, env, env)

    # `run_doctests` is not implemented here: its default rule already runs the
    # examples with the ordinary Python compiler, which is exactly this provider's
    # (absent) policy.


class _StdoutPrintCollector(PrintCollector):
    """`_print_` factory whose `print(...)` writes to the real `sys.stdout`
    (so output-capturing callers see it) rather than accumulating into the
    collector's discarded `printed` buffer."""

    def _call_print(self, *objects, **kwargs):
        kwargs.setdefault("file", sys.stdout)
        builtins.print(*objects, **kwargs)


# ----------------------------------------------------------------------------
# The RestrictedPython policy: what generated code may name, touch and import
#
# RestrictedPython's job is to keep generated code away from the interpreter's
# internals. The names below draw that line once, and both halves of the sandbox
# read from it: the compile-time policy (`RestrictedPythonPolicy`) and the runtime
# guards installed in the exec environment (`_guarded_getattr`, `_guarded_import`).
# ----------------------------------------------------------------------------

# Names the RestrictedPython transformer itself emits into compiled code -- the
# guarded accessors (`_getattr_(x, "y")`), the print collector (`_print`), the
# class metaclass. Generated code must not bind or read them, or it could hand
# itself an unguarded accessor (or quietly disable one).
_GUARD_NAMES = frozenset(
    {
        "_apply_",
        "_getattr_",
        "_getitem_",
        "_getiter_",
        "_inplacevar_",
        "_iter_unpack_sequence_",
        "_print",
        "_print_",
        "_unpack_sequence_",
        "_write_",
        "__builtins__",
        "__metaclass__",
    }
)

# Dunder attributes generated code may read, call and define: the operator/context
# protocols plus a few purely descriptive names. Every *other* ``__dunder__`` --
# `__class__`, `__bases__`, `__subclasses__`, `__mro__`, `__globals__`, `__code__`,
# `__dict__`, `__getattribute__`, `__reduce__`, ... -- is the road from sandboxed
# code back to the interpreter, and stays closed.
_SAFE_DUNDER_ATTRS = frozenset(
    {
        "__abs__",
        "__add__",
        "__bool__",
        "__call__",
        "__contains__",
        "__doc__",
        "__enter__",
        "__eq__",
        "__exit__",
        "__ge__",
        "__getitem__",
        "__gt__",
        "__hash__",
        "__init__",
        "__iter__",
        "__le__",
        "__len__",
        "__lt__",
        "__mul__",
        "__name__",
        "__ne__",
        "__neg__",
        "__next__",
        "__post_init__",
        "__radd__",
        "__repr__",
        "__reversed__",
        "__rmul__",
        "__setitem__",
        "__str__",
        "__sub__",
    }
)

# Modules generated code may import. Pure computation and data structures only:
# nothing that reaches the filesystem, the network, the process, or the import
# system itself (`os`, `sys`, `subprocess`, `socket`, `importlib`, `builtins`,
# `inspect`, `types`, `pickle`, `ctypes`, ...). Submodules must be listed in full,
# since the check is on the imported name.
#
# Two near-misses worth recording, so they don't get added later by analogy:
# `io` is not here because `io.open` *is* `builtins.open`, which would hand back
# the filesystem that omitting `open` closes; and `numpy`/`matplotlib` are not here
# because `numpy.load(..., allow_pickle=True)` executes arbitrary code. Synthesized
# code that genuinely needs those belongs under `UnsafeEvalProvider`, not behind a
# widened allowlist.
_ALLOWED_MODULES = frozenset(
    {
        "abc",
        "array",
        "base64",
        "binascii",
        "bisect",
        "calendar",
        "cmath",
        "collections",
        "collections.abc",
        "copy",
        "csv",
        "dataclasses",
        "datetime",
        "decimal",
        "difflib",
        "enum",
        "fractions",
        "functools",
        "graphlib",
        "hashlib",
        "heapq",
        "itertools",
        "json",
        "math",
        "numbers",
        "operator",
        "queue",
        "random",
        "re",
        "statistics",
        "string",
        "struct",
        "textwrap",
        "typing",
        "unicodedata",
        "uuid",
    }
)

# `n += 1` compiles to `n = _inplacevar_("+=", n, 1)`, so the environment has to
# supply the operators; without this every augmented assignment is a NameError.
_INPLACE_OPS: dict[
    str, collections.abc.Callable[[typing.Any, typing.Any], typing.Any]
] = {
    "+=": operator.iadd,
    "-=": operator.isub,
    "*=": operator.imul,
    "/=": operator.itruediv,
    "//=": operator.ifloordiv,
    "%=": operator.imod,
    "**=": operator.ipow,
    "<<=": operator.ilshift,
    ">>=": operator.irshift,
    "&=": operator.iand,
    "^=": operator.ixor,
    "|=": operator.ior,
    "@=": operator.imatmul,
}

# Builtins beyond RestrictedPython's deliberately minimal `safe_builtins`. Each is
# pure and reaches nothing outside its arguments; the omissions are the point --
# `open`, `input`, `compile`, `eval`, `exec`, `globals`, `locals`, `vars`, `dir`
# and `breakpoint` are all absent, and `__import__`/`getattr`/`setattr`/`delattr`
# are installed as guarded wrappers rather than taken from `builtins`.
_EXTRA_SAFE_BUILTIN_NAMES = frozenset(
    {
        "all",
        "any",
        "ascii",
        "bin",
        "bytearray",
        "classmethod",
        "dict",
        "enumerate",
        "filter",
        "format",
        "frozenset",
        "iter",
        "list",
        "map",
        "max",
        "memoryview",
        "min",
        "next",
        "object",
        "property",
        "reversed",
        "set",
        "staticmethod",
        "sum",
        "super",
        "type",
        # Exceptions `safe_builtins` happens to omit but recursive/brute-force
        # generated code routinely names.
        "RecursionError",
        "StopAsyncIteration",
        "TimeoutError",
        "NotImplemented",
    }
)


# Dunder *bindings* a class or module body may make. Binding these declares
# something about the code being written; it is not a way to read a dunder
# attribute off an object, which `_is_allowed_attribute` governs separately.
_SAFE_DUNDER_BINDINGS = frozenset({"__all__", "__slots__"})


def _is_allowed_name(name: str) -> bool:
    """Whether generated code may bind or read ``name`` as an identifier.

    RestrictedPython rejects *every* name starting with ``_``, which costs a great
    deal (``_helper``, ``_memo``, ``_solve`` are how Python spells "private") and
    buys little: an identifier is only dangerous when it is one of the guard names
    the transformer emits, or a dunder. Both of those stay rejected; a plain
    single-underscore private does not.
    """
    if name in _GUARD_NAMES or name in FORBIDDEN_FUNC_NAMES:
        return False
    if name == "_":  # the conventional throwaway
        return True
    if name in _SAFE_DUNDER_BINDINGS:
        return True
    if not name.startswith("_"):
        return True
    # `_private` is fine; `__dunder__`-shaped and `_guard_`-shaped names are not.
    return not name.startswith("__") and not name.endswith("_")


def _is_allowed_attribute(name: str) -> bool:
    """Whether generated code may read, write or define the attribute ``name``.

    Same trade as `_is_allowed_name`, one level in: single-underscore attributes
    (``self._items``) are ordinary Python, while dunders outside
    `_SAFE_DUNDER_ATTRS` -- and the frame/code/traceback attributes RestrictedPython
    lists in ``INSPECT_ATTRIBUTES`` -- are how sandboxed code climbs out.
    """
    if name in INSPECT_ATTRIBUTES or name.endswith("__roles__"):
        return False
    if name.startswith("__"):
        return name in _SAFE_DUNDER_ATTRS
    return True


class RestrictedPythonPolicy(RestrictingNodeTransformer):
    """RestrictedPython's policy, relaxed where it rejects ordinary modern Python.

    `RestrictingNodeTransformer` predates a good deal of the language a model
    writes today, and its rejections are not all security-carrying. Four in
    particular make it unusable as-is for synthesized code:

    * **annotated assignments** (``total: int = 0``) are rejected outright, and with
      them every ``@dataclass`` and every class-level field -- while this library
      *asks* models for annotated code and type-checks what it gets;
    * **any name starting with ``_``**, so a helper called ``_solve`` fails to
      compile;
    * **``nonlocal``**, so a closure cannot update the variable it closes over;
    * **``counts[k] += 1``**, so nothing can accumulate into a container in place.

    This subclass allows those four (plus ``type`` aliases and the dunder methods
    in `_SAFE_DUNDER_ATTRS`, so a model can define ``__len__``/``__repr__`` on its
    own classes) and changes nothing else. Everything the sandbox actually rests on
    is inherited untouched: ``exec``/``eval`` calls, star imports, ``async``,
    ``except*``, ``match`` and any other unreviewed syntax are still rejected, and
    attribute access, subscripting and iteration are still rewritten to the guarded
    accessors that `RestrictedEvalProvider` installs.
    """

    def check_name(
        self, node: typing.Any, name: str | None, allow_magic_methods: bool = False
    ) -> None:
        if name is None or _is_allowed_name(name):
            return
        # Defining a dunder *method* (never at column 0, i.e. always inside a class
        # or function body) is how you write `__len__`; reading one is governed by
        # `_is_allowed_attribute`, which agrees on the same set.
        if (
            allow_magic_methods
            and name in _SAFE_DUNDER_ATTRS
            and getattr(node, "col_offset", 0) != 0
        ):
            return
        super().check_name(node, name, allow_magic_methods)

    def visit_Attribute(self, node: ast.Attribute) -> typing.Any:
        """``a.b`` -> ``_getattr_(a, 'b')``, ``a.b = c`` -> ``_write_(a).b = c``.

        Identical to the base transform except that the name is checked against
        `_is_allowed_attribute` rather than "does it start with an underscore".
        """
        if not _is_allowed_attribute(node.attr):
            self.error(node, f'"{node.attr}" is a restricted attribute name.')
        node = self.node_contents_visit(node)
        if isinstance(node.ctx, ast.Load):
            new_node: ast.expr = ast.Call(
                func=ast.Name("_getattr_", ast.Load()),
                args=[node.value, ast.Constant(node.attr)],
                keywords=[],
            )
            copy_locations(new_node, node)
            return new_node
        # Store/Del: the base returns the attribute node itself with its *object*
        # routed through the write guard.
        new_value = ast.Call(
            func=ast.Name("_write_", ast.Load()), args=[node.value], keywords=[]
        )
        copy_locations(new_value, node.value)
        node.value = new_value
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> typing.Any:
        """Allow annotated assignment (``x: int = 1``).

        It carries no capability a plain assignment doesn't: there is a single
        target and no unpacking, and an attribute or subscript target is rewritten
        to the write guard by the visitors for those nodes, exactly as in
        ``visit_Assign``. The annotation is just another expression.
        """
        return self.node_contents_visit(node)

    def visit_TypeAlias(self, node: ast.AST) -> typing.Any:
        """Allow ``type X = ...`` aliases: a binding of a lazily-evaluated
        annotation expression, with no more reach than the expression itself."""
        return self.node_contents_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> typing.Any:
        """Allow ``nonlocal``.

        Like the ``global`` the base already allows, it only rebinds a name in an
        *enclosing scope of the same generated code* -- and `check_name` still
        governs which names those can be."""
        return self.node_contents_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> typing.Any:
        """Allow augmented assignment to an item or attribute (``counts[k] += 1``).

        The base allows it only for a plain name (rewritten to ``_inplacevar_``) and
        rejects ``a[i] += x`` / ``a.b += x`` outright, which rules out most code that
        accumulates into a container. Left alone, those compile to a subscript/attribute
        read, the in-place operation, and a store -- and the target still goes through
        the ordinary `visit_Subscript`/`visit_Attribute` rewriting, so the store lands on
        ``_write_(a)`` and an attribute name is still checked. The read is the one thing
        not routed through ``_getitem_``/``_getattr_``; that costs nothing here, where
        `RestrictedEvalProvider` reads are unrestricted, but a policy paired with a
        *restricting* ``_getitem_`` should not use this class.
        """
        if isinstance(node.target, ast.Name):
            return super().visit_AugAssign(node)
        return self.node_contents_visit(node)

    # -- `match` -------------------------------------------------------------
    #
    # RestrictedPython has no visitor for any of the match nodes, so its
    # `generic_visit` rejects the whole statement. Supporting it takes more than
    # forwarding to `node_contents_visit`, for two reasons.
    #
    # First, a pattern is not an expression, even where it holds expression nodes:
    # the value of a `case Color.RED` and the class of a `case Point(...)` must stay
    # a literal or a dotted name. Rewriting them into `_getattr_(...)` calls -- what
    # `visit_Attribute` does everywhere else -- produces a program CPython refuses to
    # compile ("patterns may only match literals and attribute lookups"). So those
    # positions are *checked* against the same name policy and left alone.
    #
    # Second, a pattern reads attributes and items of the subject without any of the
    # rewriting the guards rely on: a class pattern reads the attributes named in
    # `kwd_attrs`, and a mapping pattern subscripts. The `kwd_attrs` names are in the
    # AST, so they get the same `_is_allowed_attribute` check as `obj.attr`. The
    # subscripting is unguarded, exactly as in `visit_AugAssign` above, and costs
    # nothing while this provider's `_getitem_`/`_getiter_` are unrestricted.
    #
    # The residual: a *positional* class pattern (`case Point(x, y)`) reads whatever
    # attribute names the class's own `__match_args__` lists, which is not in the AST
    # and cannot be checked here. Generated code cannot supply those names -- binding
    # `__match_args__` is refused by `_is_allowed_name` and setting it by
    # `_guarded_setattr` -- so they can only come from a class the host program put
    # in scope, which would have to name an interpreter internal there deliberately.

    def visit_Match(self, node: ast.Match) -> typing.Any:
        """Allow ``match``; the subject is an ordinary expression."""
        return self.node_contents_visit(node)

    def visit_match_case(self, node: ast.match_case) -> typing.Any:
        """Allow a ``case``: its guard and body are ordinary code (and are rewritten
        as such); its pattern dispatches to the visitors below."""
        return self.node_contents_visit(node)

    def _check_pattern_expression(self, node: ast.AST) -> None:
        """Check the names and attributes of an expression sitting in a pattern
        position, without rewriting it.

        Every such expression is a literal or a dotted name, and is only ever
        *compared* against the subject -- never bound -- so checking it under the
        same policy as an ordinary read is enough.
        """
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute):
                if not _is_allowed_attribute(child.attr):
                    self.error(child, f'"{child.attr}" is a restricted attribute name.')
            elif isinstance(child, ast.Name):
                self.check_name(child, child.id)

    def visit_MatchValue(self, node: ast.MatchValue) -> typing.Any:
        """``case 3:`` / ``case Color.RED:`` -- checked, deliberately not rewritten."""
        self._check_pattern_expression(node.value)
        return node

    def visit_MatchSingleton(self, node: ast.MatchSingleton) -> typing.Any:
        """``case None:`` / ``case True:`` -- a bare constant, nothing to check."""
        return node

    def visit_MatchSequence(self, node: ast.MatchSequence) -> typing.Any:
        """``case [a, b]:`` -- sub-patterns are visited; see the note on unguarded
        iteration above."""
        return self.node_contents_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> typing.Any:
        """``case [first, *rest]:`` -- ``rest`` is a binding."""
        self.check_name(node, node.name)
        return node

    def visit_MatchAs(self, node: ast.MatchAs) -> typing.Any:
        """``case x:`` / ``case [1] as pair:`` / ``case _:`` -- a binding plus an
        optional sub-pattern."""
        self.check_name(node, node.name)
        return self.node_contents_visit(node)

    def visit_MatchOr(self, node: ast.MatchOr) -> typing.Any:
        """``case 1 | 2:`` -- alternatives are themselves patterns."""
        return self.node_contents_visit(node)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> typing.Any:
        """``case {"k": v, **rest}:`` -- keys are checked in place (they are pattern
        expressions), sub-patterns are visited, ``rest`` is a binding."""
        for key in node.keys:
            self._check_pattern_expression(key)
        self.check_name(node, node.rest)
        node.patterns = [self.visit(pattern) for pattern in node.patterns]
        return node

    def visit_MatchClass(self, node: ast.MatchClass) -> typing.Any:
        """``case Point(x=0, y=y):`` -- the class is checked in place, and each
        keyword names an attribute read off the subject, so it gets the attribute
        policy."""
        self._check_pattern_expression(node.cls)
        for attr in node.kwd_attrs:
            if not _is_allowed_attribute(attr):
                self.error(node, f'"{attr}" is a restricted attribute name.')
        node.patterns = [self.visit(pattern) for pattern in node.patterns]
        node.kwd_patterns = [self.visit(pattern) for pattern in node.kwd_patterns]
        return node


_RAISE = object()

# `str.format`'s field names, the part before any `!conversion` or `:spec`. A field
# that *reaches* -- `{0.__class__}`, `{0[1]}` -- is the whole reason RestrictedPython
# refuses these methods, since the traversal happens inside CPython's formatter,
# where the `_getattr_`/`_getitem_` rewriting cannot see it.
_FORMAT_REACHING_CHARS = (".", "[")


def _reject_reaching_format_fields(template: str) -> None:
    """Raise unless every replacement field in ``template`` names its argument and
    stops there -- no attribute or item traversal from it.

    Nested fields inside a format spec (``"{0:{1}}"``) are checked too; the
    recursion terminates because a spec with no braces yields a single field of
    ``None``, and CPython caps spec nesting at one level regardless.
    """
    for _, field_name, format_spec, _ in string.Formatter().parse(template):
        if field_name is not None and any(
            char in field_name for char in _FORMAT_REACHING_CHARS
        ):
            raise NotImplementedError(
                f"format field {'{'}{field_name}{'}'} reaches into an attribute or "
                f"item of its argument, which is not allowed in restricted code; "
                f"index or attribute it yourself, or use an f-string"
            )
        if format_spec:
            _reject_reaching_format_fields(format_spec)


def _checked_str_format(
    target: typing.Any, name: str
) -> collections.abc.Callable[..., str]:
    """``str.format``/``str.format_map``, wrapped to check the template first.

    RestrictedPython refuses these outright because ``"{0.__class__.__mro__}"``
    walks attributes inside CPython's formatter, out of reach of the guards -- but
    refusing them wholesale is expensive and, worse, invisible: the call compiles and
    type-checks and only fails when the code runs, possibly inside a doctest. What is
    actually dangerous is the *traversal*, and that is written in the template, where
    it can be read off before anything is formatted. So check the template and allow
    the rest, which is nearly all real uses.
    """
    # Reached either bound (``"...".format(x)``, template in hand) or unbound
    # (``str.format("...", x)``, template passed as the first argument).
    unbound = isinstance(target, type)

    def formatter(*args: typing.Any, **kwargs: typing.Any) -> str:
        if unbound and not args:
            return typing.cast(str, getattr(str, name)())  # let Python raise TypeError
        _reject_reaching_format_fields(args[0] if unbound else target)
        return typing.cast(str, getattr(target, name)(*args, **kwargs))

    return formatter


def _guarded_getattr(
    obj: typing.Any, name: str, default: typing.Any = _RAISE
) -> typing.Any:
    """The runtime half of `_is_allowed_attribute`: the ``_getattr_`` every
    attribute access in restricted code compiles to, and the ``getattr`` builtin
    generated code sees (so a dynamically-computed name is checked too)."""
    if type(name) is not str:
        raise TypeError("type(name) must be str")
    if name in ("format", "format_map") and (
        isinstance(obj, str) or (isinstance(obj, type) and issubclass(obj, str))
    ):
        # The one place this guard is *more* permissive than RestrictedPython's, and
        # only because it checks what that one could not. `string.Formatter` and its
        # methods stay blocked by the delegation below: there the template is not in
        # hand at attribute-access time, so there is nothing to check.
        return _checked_str_format(obj, name)
    if not name.startswith("_"):
        # Public name: defer to RestrictedPython's own guard, which additionally
        # blocks `string.Formatter` (it can reach attributes of whatever is
        # interpolated) and the `inspect` attributes.
        if default is _RAISE:
            return Guards.safer_getattr_raise(obj, name)
        return Guards.safer_getattr(obj, name, default)
    if not _is_allowed_attribute(name):
        raise AttributeError(f'"{name}" is a restricted attribute name.')
    if default is _RAISE:
        return getattr(obj, name)
    return getattr(obj, name, default)


def _guarded_hasattr(obj: typing.Any, name: str) -> bool:
    """`hasattr` that agrees with `_guarded_getattr`: a restricted attribute reads
    as absent rather than as a way to probe for one."""
    try:
        _guarded_getattr(obj, name)
    except (AttributeError, TypeError, NotImplementedError):
        return False
    return True


def _guarded_setattr(obj: typing.Any, name: str, value: typing.Any) -> None:
    """`setattr` under `_is_allowed_attribute`, matching what ``obj.name = value``
    compiles to (RestrictedPython checks *that* name at compile time; this is the
    same rule for a name computed at run time)."""
    if type(name) is not str:
        raise TypeError("type(name) must be str")
    if not _is_allowed_attribute(name):
        raise AttributeError(f'"{name}" is a restricted attribute name.')
    setattr(obj, name, value)


def _guarded_delattr(obj: typing.Any, name: str) -> None:
    """`delattr` under `_is_allowed_attribute`; see `_guarded_setattr`."""
    if type(name) is not str:
        raise TypeError("type(name) must be str")
    if not _is_allowed_attribute(name):
        raise AttributeError(f'"{name}" is a restricted attribute name.')
    delattr(obj, name)


def _guarded_import(
    name: str,
    globals: typing.Any = None,
    locals: typing.Any = None,
    fromlist: collections.abc.Sequence[str] = (),
    level: int = 0,
) -> types.ModuleType:
    """The ``__import__`` restricted code sees: `_ALLOWED_MODULES` only.

    Without an ``__import__`` at all, ``import math`` fails and most generated code
    with it; with the real one, ``import os`` hands back the process. Relative
    imports are refused outright -- there is no package for generated code to be
    relative to."""
    if level != 0:
        raise ImportError("relative imports are not allowed in restricted code")
    if name not in _ALLOWED_MODULES:
        raise ImportError(
            f"import of {name!r} is not allowed in restricted code; "
            f"the allowed modules are: {', '.join(sorted(_ALLOWED_MODULES))}"
        )
    return builtins.__import__(name, globals, locals, fromlist, level)


def _guarded_inplacevar(op: str, x: typing.Any, y: typing.Any) -> typing.Any:
    """``x <op>= y``, as `_INPLACE_OPS` spells it."""
    try:
        return _INPLACE_OPS[op](x, y)
    except KeyError:
        raise NotImplementedError(f"augmented assignment {op!r} is not supported")


def _guarded_apply(
    func: collections.abc.Callable, *args: typing.Any, **kwargs: typing.Any
) -> typing.Any:
    """``f(*args, **kwargs)`` -- the form the transformer routes starred calls
    through. Argument *values* need no guarding here: whatever built them was
    itself compiled under the same policy."""
    return func(*args, **kwargs)


class RestrictedEvalProvider(ObjectInterpretation):
    """
    Safer provider using RestrictedPython.

    RestrictedPython is not a complete sandbox, but it enforces a restricted
    language subset and expects you to provide a constrained exec environment.
    This provider supplies that environment: a `RestrictedPythonPolicy` at compile
    time, and at run time the guarded accessors that policy's output calls into
    (`_guarded_getattr`, `_guarded_import`, ...) over a builtins namespace with no
    I/O, no introspection and no way back to `compile`/`eval`/`exec`.

    Doctests are executed under the same policy as the code they exercise, so a
    model cannot smuggle past the sandbox in a docstring.

    policy : type[RestrictingNodeTransformer], optional
        RestrictedPython compile_restricted policy for compilation. Defaults to
        `RestrictedPythonPolicy`.
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
        self,
        source: str,
        lo: int | None = None,
        hi: int | None = None,
        *,
        lenient: bool = False,
    ) -> None:
        _mypy_check_region(source, lo, hi, lenient)

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

    def _compile_restricted(
        self, source: typing.Any, filename: str, mode: str = "exec", *args: typing.Any
    ) -> types.CodeType:
        """`compile_restricted` under this provider's policy, with `compile`'s
        positional signature so it can stand in for the built-in (which is how
        `_doctest_compiled_with` uses it)."""
        # RestrictedPython's transformer rewrites its argument *in place*, so after
        # this call `tree` is the checked, guard-injected program -- which we then
        # compile ourselves. That second compile is not redundant: RestrictedPython
        # calls `compile()` without `dont_inherit`, from a module that begins with
        # `from __future__ import annotations`, so its future flag leaks into every
        # program it compiles and turns generated code's annotations into strings
        # (which breaks, among others, every `@dataclass`). Compiling the same tree
        # here, from a module with no future imports and with `dont_inherit`, gives
        # generated code the semantics its source actually asks for.
        #
        # Transform a *copy*, so the caller's AST is left as it passed it: it is the
        # `parse` op's output, which callers also read (and could compile again --
        # re-transforming an already-transformed tree would reject its own guards).
        tree = (
            copy.deepcopy(source)
            if isinstance(source, ast.AST)
            else ast.parse(source, filename, mode)
        )
        with warnings.catch_warnings():
            # RestrictedPython warns ("Prints, but never reads 'printed'") whenever
            # the code prints, because its protocol expects the collected text to be
            # read back; we route `print` to stdout instead, so the warning is noise.
            warnings.filterwarnings(
                "ignore", ".*Prints, but never reads", SyntaxWarning
            )
            compile_restricted(  # raises SyntaxError if the policy rejects `tree`
                typing.cast(typing.Any, tree),
                filename=filename,
                mode=typing.cast(typing.Any, mode),
                policy=self.policy or RestrictedPythonPolicy,
            )
        return builtins.compile(
            typing.cast(typing.Any, tree),
            filename,
            mode,
            dont_inherit=True,
        )

    def _restricted_globals(
        self, env: collections.abc.Mapping[str, typing.Any]
    ) -> dict[str, typing.Any]:
        """The namespace restricted code runs in: RestrictedPython's safe builtins
        (extended by `_EXTRA_SAFE_BUILTIN_NAMES` and the guarded
        ``__import__``/``getattr``/``hasattr``), the guarded accessors compiled code
        calls into, then ``env`` layered on top."""
        rglobals: dict[str, typing.Any] = safe_globals.copy()

        # `safe_globals` is module-level shared state and its `__builtins__` is one
        # dict, so extend a *copy* rather than mutating everyone's builtins.
        restricted_builtins = dict(rglobals["__builtins__"])
        for name in _EXTRA_SAFE_BUILTIN_NAMES:
            restricted_builtins.setdefault(name, getattr(builtins, name))
        restricted_builtins["__import__"] = _guarded_import
        restricted_builtins["getattr"] = _guarded_getattr
        restricted_builtins["hasattr"] = _guarded_hasattr
        restricted_builtins["setattr"] = _guarded_setattr
        restricted_builtins["delattr"] = _guarded_delattr
        rglobals["__builtins__"] = restricted_builtins

        # Enable class definitions (required for Python 3)
        rglobals["__metaclass__"] = type
        rglobals["__name__"] = "restricted"

        # The accessors compiled code is rewritten to call. Without every one of
        # these, perfectly ordinary code -- indexing, unpacking, `+=` -- dies with a
        # `NameError` on a guard name the model never wrote.
        rglobals["_getattr_"] = _guarded_getattr
        rglobals["_getitem_"] = Eval.default_guarded_getitem
        rglobals["_getiter_"] = Eval.default_guarded_getiter
        rglobals["_iter_unpack_sequence_"] = Guards.guarded_iter_unpack_sequence
        rglobals["_unpack_sequence_"] = Guards.guarded_unpack_sequence
        rglobals["_inplacevar_"] = _guarded_inplacevar
        rglobals["_apply_"] = _guarded_apply
        # Attribute/item *writes* land on objects the generated code was handed or
        # built itself; the names it may write are already fixed at compile time by
        # `RestrictedPythonPolicy.visit_Attribute`.
        rglobals["_write_"] = lambda x: x

        # RestrictedPython rewrites `print(...)` into its `_print_` collector
        # protocol; route it to the real stdout so output-capturing callers
        # (e.g. redirect_stdout) see it instead of a discarded collector.
        rglobals["_print_"] = _StdoutPrintCollector
        # The transformer injects `_print = _print_(_getattr_)` at the top of each
        # module and function that prints -- but not into `single`-mode code, which
        # is what a doctest example is. Seed one so `print` works there too.
        rglobals["_print"] = _StdoutPrintCollector(_guarded_getattr)

        # Layer `env` on top (without letting callers replace the restricted
        # builtins, or any guard, with something of their choosing).
        rglobals.update(
            {
                k: v
                for k, v in env.items()
                if k != "__builtins__" and k not in _GUARD_NAMES
            }
        )
        return rglobals

    @implements(compile)
    def compile(self, module: ast.Module, filename: str) -> types.CodeType:
        # RestrictedPython can compile from an AST directly.
        return self._compile_restricted(module, filename, "exec")

    @implements(exec)
    def exec(
        self,
        bytecode: types.CodeType,
        env: dict[str, typing.Any],
    ) -> None:
        rglobals = self._restricted_globals(env)

        # Snapshot value identities before execution so we can copy back every
        # *binding effect* — both new names and rebindings of seeded names.
        before = dict(rglobals)
        builtins.exec(bytecode, rglobals, rglobals)

        sentinel = object()
        env.update(
            {
                key: value
                for key, value in rglobals.items()
                # The guards (and the `_print` the transformer injects) are the
                # sandbox's own furniture, not a binding effect of the code.
                if key != "__builtins__"
                and key not in _GUARD_NAMES
                and before.get(key, sentinel) is not value
            }
        )

    @implements(run_doctests)
    def run_doctests(
        self,
        obj: collections.abc.Callable | type | types.ModuleType,
        globs: collections.abc.Mapping[str, typing.Any],
    ) -> None:
        # A docstring's examples are as much model output as the code they
        # document, so run them under the same policy and in the same guarded
        # namespace -- otherwise `>>> __import__("os").system(...)` in a synthesized
        # docstring would execute with nothing restricting it at all.
        with _doctest_compiled_with(self._compile_restricted):
            _run_doctests(obj, self._restricted_globals(globs))


class _OpCommandCompiler(codeop.CommandCompiler):
    """A `codeop.CommandCompiler` that routes compilation through the
    `parse`/`compile` effect operations (so the installed eval provider owns it
    and `parse` populates `linecache`), replacing the native single-mode
    compiler that `code.InteractiveInterpreter` installs.
    """

    def __call__(
        self, source: str, filename: str = "<input>", symbol: str = "single"
    ) -> types.CodeType:
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

    def __init__(self, env: collections.abc.MutableMapping[str, typing.Any]):
        # Run in a fresh writable dict seeded with a flat view of `env`.  This is
        # forced by `exec`: its globals must be one real dict (a ChainMap is
        # rejected), and a REPL needs a single persistent namespace so a function
        # defined in one snippet sees a name a later snippet binds.  Seeding a flat
        # copy also leaves the lexical seed untouched, so REPL assignments never
        # leak into the surrounding scope.
        scope: dict[str, typing.Any] = dict(env)
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
        self._prior_snippets: list[str] = []

    @property
    def prior_snippets(self) -> list[str]:
        """Sources of the actual error-free executed snippets, in order -- the type-check
        context the `Encodable[CodeType]` decoder splices before the current snippet."""
        return self._prior_snippets

    def runcode(self, code: types.CodeType) -> None:
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

    def exec_code(self, code: types.CodeType) -> str:
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
        # Record this snippet's source so the *next* snippet's decode-time type check can
        # splice the accumulated session code into the Template body. The type check itself
        # lives in the `Encodable[CodeType]` decoder (as it does for synthesized Callables),
        # not here -- this session only runs code.
        self._prior_snippets.append("".join(linecache.getlines(code.co_filename)))
        with (
            contextlib.redirect_stdout(self.stdout),
            contextlib.redirect_stderr(self.stderr),
        ):
            self.runcode(code)
        return self.stdout.getvalue()[out_start:] + self.stderr.getvalue()[err_start:]
