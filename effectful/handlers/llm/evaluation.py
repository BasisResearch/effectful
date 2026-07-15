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
import symtable
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


def _region_errors(stdout: str, lo: int | None, hi: int | None) -> list[dict[str, Any]]:
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


# Builtin names mypy resolves at module scope. dir(builtins) also contains module
# dunders (__loader__/__spec__/__package__) that mypy does NOT resolve -- including
# them would let a read of __loader__ pass the gate and get a spurious name-defined.
# Non-dunder builtins are a clean subset of mypy's builtin scope; over-declining the
# rare dunder read is cheap under report-not-gate, a spurious error is not.
_BUILTIN_NAMES = frozenset(n for n in dir(builtins) if not n.startswith("_"))


class _ModuleBinds(ast.NodeVisitor):
    """Names freshly bound at MODULE scope: recurse control-flow (if/for/while/with/
    try/match) but HALT at nested scopes (def/class/lambda/comprehensions), so nested
    locals aren't counted. Collects the names that would `no-redef` a prior module bind;
    used only for the rebind check, so the AugAssign exclusion (a `+=` target is not a
    fresh bind) has to be per-*node* (a name both `+=`-ed and plainly assigned is still a
    fresh bind) -- which symtable's scope-level ``is_assigned`` cannot express. So this
    stays hand-rolled over the AST while the read side (``_external_reads``) delegates
    scope resolution to symtable. `del`/`global`/`nonlocal` targets need no override:
    their targets aren't `Store` ``Name``s, so ``visit_Name`` already skips them.

    walrus (``:=``) and match captures leak past a comprehension / into the enclosing
    scope, so they ARE module binds even though the comprehension halts normal
    recursion. A def/class also *evaluates* its decorators, default arguments and
    annotations in THIS scope (only its body is the nested scope), so a walrus there
    binds at module scope too -- collected from every field but the body."""

    def __init__(self) -> None:
        # A list, not a set: a name bound twice at module scope (`x = 1; x = "s"`) must
        # count twice so the rebind tally in `repl_check_source` sees the retype. Extra
        # counts only ever over-decline, so an inexact tally stays sound.
        self.names: list[str] = []

    def visit_FunctionDef(self, n):  # bind the name; body is a nested scope, skip it
        self.names.append(n.name)
        self._walrus_outside_body(n)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, n):
        self.names.append(n.name)
        self._walrus_outside_body(n)

    def visit_Lambda(
        self, n
    ):  # own scope; a walrus in its default binds here, not there
        self._collect_walrus_in(*n.args.defaults, *(d for d in n.args.kw_defaults if d))

    def visit_ListComp(self, n):
        # A comprehension is its own scope (its `for` targets don't leak), but a walrus
        # inside it binds in THIS scope.
        self._collect_walrus(n)

    visit_SetComp = visit_DictComp = visit_GeneratorExp = visit_ListComp

    def _walrus_outside_body(self, n) -> None:
        # A def/class evaluates its decorators, defaults and annotations in the enclosing
        # (here module) scope, so walruses there bind at module scope; the body is the
        # nested scope and is skipped.
        self._collect_walrus_in(*(v for f, v in ast.iter_fields(n) if f != "body"))

    def _collect_walrus_in(self, *nodes) -> None:
        for node in nodes:
            for item in node if isinstance(node, list) else [node]:
                if isinstance(item, ast.AST):
                    self._collect_walrus(item)

    def _collect_walrus(self, node) -> None:
        # Node-inclusive: a walrus may be the node itself (`x=(w := 5)` -> the default IS
        # the NamedExpr). Descend through everything but nested function scopes, through
        # which a walrus does NOT leak.
        if isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda
        ):
            return
        if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            self.names.append(node.target.id)
        for child in ast.iter_child_nodes(node):
            self._collect_walrus(child)

    def visit_NamedExpr(self, n):  # `(y := ...)` outside a comprehension binds here
        if isinstance(n.target, ast.Name):
            self.names.append(n.target.id)
        self.visit(n.value)

    def visit_MatchAs(self, n):  # `case ... as name` / bare `case name` capture
        if n.name:
            self.names.append(n.name)
        self.generic_visit(n)

    def visit_MatchStar(self, n):  # `case [*rest]`
        if n.name:
            self.names.append(n.name)

    def visit_MatchMapping(self, n):  # `case {**rest}`
        if n.rest:
            self.names.append(n.rest)
        self.generic_visit(n)

    def visit_AugAssign(self, n):  # target is read-modify-write, not a fresh bind
        self.visit(n.value)

    def visit_ExceptHandler(self, n):  # `except X as e` binds `e`; recurse the body
        if n.name:
            self.names.append(n.name)
        self.generic_visit(n)

    def visit_Import(self, n):
        for a in n.names:
            self.names.append(a.asname or a.name.split(".")[0])

    def visit_ImportFrom(self, n):
        for a in n.names:
            self.names.append(a.asname or a.name)

    def visit_Name(self, n):  # only module-level Names reach here (no nested recursion)
        if isinstance(n.ctx, ast.Store):
            self.names.append(n.id)


def _module_binds(tree: ast.AST) -> set[str]:
    return set(_module_bind_counts(tree))


def _module_bind_counts(tree: ast.AST) -> "collections.Counter[str]":
    v = _ModuleBinds()
    v.visit(tree)
    return collections.Counter(v.names)


def _external_reads(source: str) -> set[str]:
    """Names ``source`` looks up in the module/global namespace at runtime -- module-
    level reads AND nested free reads that escape their local scope chain (a closure's
    read of an enclosing local does NOT escape; a nested read of an unbound name does),
    plus read-modify-write (``+=``) and ``del`` targets.

    Delegates the lexical scope resolution to ``symtable.Symbol.is_global`` -- which
    walks the real scope chain the compiler uses -- rather than re-deriving it over the
    AST, so a nested-scope local or parameter can never spuriously satisfy a module-
    level read (the false positive a flat all-scopes bind set would allow). ``source``
    is always compilable here (it already decoded through ``Encodable[CodeType]``).

    Intersected with the identifiers that actually appear in the source: symtable also
    reports compiler-synthesized globals absent from the user's code (Python 3.14's
    ``__conditional_annotations__`` from PEP 649, injected for any annotated statement),
    which mypy never sees and so must not count as an unresolved read."""
    names_in_source = {
        n.id for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Name)
    }
    reads: set[str] = set()
    stack = [symtable.symtable(source, "<repl>", "exec")]
    while stack:
        table = stack.pop()
        reads |= {s.get_name() for s in table.get_symbols() if s.is_global()}
        stack.extend(table.get_children())
    return reads & names_in_source


def _annotation_exprs(tree: ast.AST) -> "list[ast.expr]":
    """Every annotation expression in the snippet: function parameter and return
    annotations, and variable (``AnnAssign``) annotations."""
    out: list[ast.expr] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef):
            a = n.args
            out += [
                arg.annotation
                for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg)
                if arg and arg.annotation
            ]
            if n.returns:
                out.append(n.returns)
        elif isinstance(n, ast.AnnAssign):
            out.append(n.annotation)
    return out


def _is_subscript_of(node: ast.Subscript, name: str) -> bool:
    v = node.value
    return (isinstance(v, ast.Name) and v.id == name) or (
        isinstance(v, ast.Attribute) and v.attr == name
    )


def _has_string_forward_ref(node: ast.expr) -> bool:
    """A string used in *type* position within an annotation (a forward ref mypy resolves
    but the interpreter never evaluates). String *values* inside ``Literal[...]`` and the
    metadata of ``Annotated[T, ...]`` are not forward refs and are skipped, so common
    ``Literal["a", "b"]`` / ``Annotated[int, "doc"]`` annotations aren't over-declined."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.Subscript):
        if _is_subscript_of(node, "Literal"):
            return False  # every Literal arg is a value, never a type
        if _is_subscript_of(node, "Annotated"):
            elts = (
                node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            )
            return bool(elts) and _has_string_forward_ref(elts[0])  # only T is a type
        return _has_string_forward_ref(node.value) or _has_string_forward_ref(
            node.slice
        )
    return any(
        _has_string_forward_ref(c)
        for c in ast.iter_child_nodes(node)
        if isinstance(c, ast.expr)
    )


def _has_deferred_annotation(tree: ast.AST, future_annotations: bool) -> bool:
    """True if the snippet carries an annotation mypy resolves but the interpreter never
    evaluates -- a *string* forward-ref (``x: "Foo"``), or *any* annotation once ``from
    __future__ import annotations`` is active in the module. mypy raises ``name-defined``
    on an unresolved name in such an annotation while the code runs fine, and
    ``_external_reads`` can't see those names (``symtable`` doesn't evaluate string or
    deferred annotations), so decline rather than risk that spurious diagnostic. Bare,
    eager annotations need no special case: the interpreter evaluates them too, so an
    unresolved one is a real ``NameError`` that ``_external_reads`` already declines."""
    return any(
        future_annotations or _has_string_forward_ref(ann)
        for ann in _annotation_exprs(tree)
    )


def _imports_future_annotations(tree: ast.AST) -> bool:
    return any(
        isinstance(n, ast.ImportFrom)
        and n.module == "__future__"
        and any(a.name == "annotations" for a in n.names)
        for n in ast.walk(tree)
    )


def repl_check_source(prior: list[str], snippet: str) -> tuple[str, int, int] | None:
    """Assemble `prior + snippet` and the [lo, hi] line span of `snippet` for
    ``type_check``, or ``None`` when checking it would spuriously fail: it freshly
    rebinds a name a prior snippet bound (no-redef), or reads a name not resolvable
    from prior / its own binds / builtins (name-defined). Both are normal in a REPL,
    neither a real type error.

    The REPL is a module built incrementally, so earlier snippets' source is the
    type-check context. This static decision runs first; the source is assembled and
    checked only when it holds.
    """
    tree = ast.parse(snippet)
    # Newline-terminate each prior snippet before concatenating: snippets recovered
    # from linecache need not end in "\n", and joining them raw would fuse two lines
    # into a syntax error (`a = 1` + `b = 2` -> `a = 1b = 2`).
    body = "".join(s if s.endswith("\n") else s + "\n" for s in prior)
    # Tally prior module binds WITH multiplicity: a name bound more than once across the
    # prior -- whether in two cells or twice in one -- was rebound, and mypy pins it to its
    # first-seen type in the concatenation, refusing the later reassignment, so its type
    # there is stale w.r.t. the runtime's latest value.
    prior_bind_counts: collections.Counter[str] = collections.Counter()
    future = _imports_future_annotations(tree)
    for p in prior:
        p_tree = ast.parse(p)
        prior_bind_counts += _module_bind_counts(p_tree)
        future = future or _imports_future_annotations(p_tree)
    prior_binds = set(prior_bind_counts)
    rebound = {name for name, count in prior_bind_counts.items() if count > 1}
    snippet_binds = _module_binds(tree)
    if snippet_binds & prior_binds:  # cross-snippet rebind -> no-redef
        return None
    # A read is resolvable iff it is bound at module scope (in the snippet or a prior
    # one) or is a builtin; anything else needs the seed env. The seed env (the session's
    # lexical context, e.g. `readings`) is deliberately NOT threaded in: declaring it is
    # #578/#577 runtime-type territory, so a read of a seed name declines (safe: never a
    # spurious name-defined) rather than being checked. This means the first snippet of a
    # session -- which typically reads seeded context -- is usually skipped, by design.
    reads = _external_reads(snippet)
    if reads - snippet_binds - prior_binds - _BUILTIN_NAMES:
        return None
    # A read of a name already rebound across prior cells is also declined: mypy's stale
    # type for it (above) would spuriously reject a use valid for its latest runtime value.
    if reads & rebound:
        return None
    # Deferred annotations (string forward-refs, or any annotation under a module-active
    # `from __future__ import annotations` -- `future`, computed above across prior + this
    # snippet) hold names mypy resolves but the runtime never evaluates and `_external_reads`
    # can't see -> an unresolved one would spuriously raise.
    if _has_deferred_annotation(tree, future):
        return None
    n_lines = snippet.count("\n") + (0 if snippet.endswith("\n") else 1)
    lo = body.count("\n") + 1
    hi = lo + n_lines - 1
    return body + snippet, lo, hi


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
        # Ordered prior-snippet sources, the type-check context for repl_check_source.
        # linecache keys snippets by random per-call <exec_code-{uuid}> filenames and
        # retains no ordered key list, so the accumulated module can't be rebuilt from
        # it. Single-session state -- not the multi-session self._buffers smell of #687.
        self._prior_snippets: list[str] = []

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
        # Type-check the snippet against the accumulated module when feasible, and
        # report any error as text (the snippet still runs -- a type error is not a
        # runtime error, and the session's contract is that errors are text it reads).
        snippet = "".join(linecache.getlines(code.co_filename))
        checked = repl_check_source(self._prior_snippets, snippet)
        if checked is not None:
            try:
                type_check(*checked)
            except TypeError as e:
                self.stderr.write(f"{e}\n")  # in-region type error: a mypy diagnostic
            except NotImplementedError:
                pass  # provider implements exec but not type_check -> unavailable, skip
            except RuntimeError as e:
                logger.warning("REPL type-check skipped: mypy failed to run: %s", e)
        self._prior_snippets.append(snippet)
        with (
            contextlib.redirect_stdout(self.stdout),
            contextlib.redirect_stderr(self.stderr),
        ):
            self.runcode(code)
        return self.stdout.getvalue()[out_start:] + self.stderr.getvalue()[err_start:]
