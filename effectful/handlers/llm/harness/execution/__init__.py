import ast
import collections.abc
import doctest
import inspect
import json
import linecache
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import types
import typing

from effectful.ops.syntax import defop


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
            >>> parity([1, 2])  # doctest: +SKIP
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
    redefined with a new type across cells (``--allow-redefinition-new``, which supersedes
    the narrower ``--allow-redefinition`` and requires ``--local-partial-types``), a
    def/class/import to be redefined (``no-redef``), and the body not to return the
    Template's declared type (``return``/``empty-body``). All normal for an
    incrementally-built REPL, not real errors.
    """
    lenient_flags = (
        [
            "--allow-redefinition-new",
            "--local-partial-types",
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
