"""Effect-row inference — the ``ε`` engine.

Sibling of :func:`effectful.ops.semantics.typeof` / :func:`fvsof`: a fold over the
universal ``apply`` operation that computes which operations a term performs (its
**effect row**), plus the ``Uses`` / ``Computation`` / ``Requires`` annotations that
operations declare and the fold reads.

The per-op rule and its annotation live on the core types, mirroring the ``τ`` / ``fvs``
machinery exactly:

======================  =====================================================
symbol                  home
======================  =====================================================
``Operation.__uses_rule__``   ``ops/types.py``, a ``@final`` method next to ``__type_rule__`` / ``__fvs_rule__``
``Uses``                    ``ops/syntax.py``, next to ``Scoped`` (read by ``__uses_rule__``)
``usesof`` / ``effectsof``   this module (fold, next to ``typeof`` / ``fvsof`` in spirit)
``Computation`` / ``Requires``   this module; argument annotations read by the fold
======================  =====================================================

Like ``Uses``, ``Computation`` and ``Requires`` are plain *read*-metadata, not
:class:`~effectful.ops.types.Annotation` signature-transforms: enforcement is at
``usesof``-time (``_fold_computation_args`` fails loudly on an unclassified callable), so
there is no build-time ``infer_annotations`` gate to wire.

The static LLM tool-governance layer (``toolsof`` / ``reachable_tools`` / ``check_tools``
— transitive tool-graph reachability with no LLM call) is built on top of this in
``handlers/llm/governance.py``.

**Not yet implemented:** ``usagesof`` (usage multiset) and handler discharge; polymorphic
``Operation[[A], B]`` ``Uses`` members; and the *runtime* LLM tool-governance layer — tool
**restriction** as an off-by-default handler filtering the offered tool set (which must
leave synthetic ``LexicalReaders`` tools untouched — they are prompt-variable plumbing,
not effectful tools), a decode-time ``ε`` validator, and ``tool_choice`` forcing. This
module is the ``ε`` core (fold + argument annotations).
"""

import typing
from dataclasses import dataclass
from typing import Annotated, Any

from effectful.internals.runtime import interpreter
from effectful.ops.semantics import apply, evaluate, typeof
from effectful.ops.syntax import Uses
from effectful.ops.types import Expr, Operation

__all__ = [
    "Computation",
    "Requires",
    "UndeclaredCallable",
    "UnsoundCallbackFold",
    "usesof",
    "effectsof",
    "effect_type",
    "check_uses",
    "requires_rule",
    "check_requires",
]


# ---------------------------------------------------------------------------
# Annotations an op declares (read off ``Annotated[T, ...]``).
# ``Uses`` itself lives in ``ops/syntax.py`` (next to ``Scoped``) and is read by
# ``Operation.__uses_rule__``; ``Computation`` / ``Requires`` are argument annotations
# read by the fold below.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Computation:
    """``Annotated[Callable[[A], B], Computation]`` on an *argument*: a suspended
    computation whose effect row joins the op's row when the op runs it. Higher-order
    combinators (``map``/``filter``/…) mark their callback arg with this instead of the
    checker hard-coding which ops are higher-order.

    Plain read-metadata (like :class:`~effectful.ops.syntax.Uses`), not an
    :class:`~effectful.ops.types.Annotation` — it is *read* by the fold, not a signature
    transform. An unclassified callable argument is caught loudly at fold time by
    :func:`_fold_computation_args`, so no build-time gate is needed."""


#: Singleton marker (data-less, like ``IsRecursive``) — use in ``Annotated[C, Computation]``.
Computation = _Computation()


@dataclass(frozen=True, init=False)
class Requires:
    """``Annotated[T, Requires(op, ...)]`` on an *argument*: the value's provenance must
    cover these ops — ``{op,...} ⊆ usesof(arg)``. The precondition dual of ``Uses``
    (#664). Plain read-metadata (like :class:`~effectful.ops.syntax.Uses`), read by
    :func:`requires_rule`."""

    ops: frozenset[Operation]

    def __init__(self, *ops: Operation) -> None:
        object.__setattr__(self, "ops", frozenset(ops))

    def missing(self, arg: Any) -> frozenset[Operation]:
        """Required ops absent from the argument's provenance — the whole check is one
        :func:`usesof`."""
        return self.ops - usesof(arg)


class UndeclaredCallable(Exception):
    """Raised by the fold on a callable argument that is neither ``Computation`` nor
    ``Uses[()]`` — the checker refuses to guess rather than silently under-approximate."""


class UnsoundCallbackFold(Exception):
    """Raised when folding a ``Computation`` callback that *inspects* its argument
    (branches on it, accesses its attributes, iterates/indexes it). The fold runs the
    callback on an opaque placeholder to collect its effects; a callback that inspects
    that placeholder would path/structure-under-approximate, so the fold refuses loudly
    (a symbolic-execution provider is needed for such callbacks)."""


# ---------------------------------------------------------------------------
# The fold (upstream: ops/semantics.py, next to typeof/fvsof)
# ---------------------------------------------------------------------------
def usesof[S](term: Expr[S]) -> frozenset[Operation]:
    """Return the effect row of a term: the set of operations it performs. The
    effect-typing sibling of :func:`typeof` / :func:`fvsof`.

    Each applied op contributes :meth:`Operation.__uses_rule__` (default ``{self}``,
    ``Uses[()]`` = pure). ``Computation``-marked callback args are *entered* so their
    effects fold too; an undeclared callable arg raises :class:`UndeclaredCallable`
    (never silent).
    """
    used: set[Operation] = set()

    def _update(op: Operation, *args: Any, **kwargs: Any) -> Any:
        used.update(op.__uses_rule__())
        _fold_computation_args(op, args, kwargs)  # enters callbacks; loud on undeclared

    with interpreter({apply: _update}):
        evaluate(term)
    return frozenset(used)


#: Reads better at effect-typing call sites; same function.
effectsof = usesof


def effect_type[S](term: Expr[S]) -> tuple[type[S], frozenset[Operation]]:
    """The effect type ``(τ, ε)`` of a term: its result type and its effect row —
    ``τ`` from :func:`typeof`, ``ε`` from :func:`usesof`. The two folds compose over the
    same ``apply`` op."""
    return typeof(term), usesof(term)


def check_uses(op: Operation, body: Expr[Any]) -> frozenset[Operation]:
    """Effects ``body`` performs that ``op``'s declared ``Uses[...]`` does not cover —
    empty == the declaration is sound (and transitively closed, since ``usesof`` unions
    the whole DAG). This is the checker for a composite op: ``usesof(body) ⊆ declared``.
    An op with no ``Uses`` annotation declares nothing, so every effect is reported."""
    declared = Uses.declared(op.__signature__)
    return usesof(body) - (declared if declared is not None else frozenset())


# ---------------------------------------------------------------------------
# Requires verification (upstream: with usesof, in semantics.py)
# ---------------------------------------------------------------------------
def requires_rule(op: Operation, *args: Any, **kwargs: Any) -> dict[str, frozenset[Operation]]:
    """Per-argument unmet provenance for ``op``: ``{arg_name: missing_ops}``. Empty ==
    every ``Requires`` on ``op`` is satisfied by the given args."""
    bound = op.__signature__.bind(*args, **kwargs)
    bound.apply_defaults()
    unmet: dict[str, frozenset[Operation]] = {}
    for name, p in op.__signature__.parameters.items():
        for anno in _annotations(p.annotation):
            if isinstance(anno, Requires) and (m := anno.missing(bound.arguments[name])):
                unmet[name] = m
    return unmet


def check_requires(term: Expr[Any]) -> dict[Operation, dict[str, frozenset[Operation]]]:
    """Provenance violations in ``term``: ``{op: {arg: missing_ops}}``. Empty == OK.
    This is #664's "public hook to read a Term's effective row" — one fold over ``apply``."""
    violations: dict[Operation, dict[str, frozenset[Operation]]] = {}

    def _update(op: Operation, *args: Any, **kwargs: Any) -> Any:
        if unmet := requires_rule(op, *args, **kwargs):
            violations[op] = unmet
        return op.__default_rule__(*args, **kwargs)

    with interpreter({apply: _update}):
        evaluate(term)
    return violations


# ---------------------------------------------------------------------------
# helpers (annotation reading)
# ---------------------------------------------------------------------------
def _annotations(annotation: Any) -> tuple[Any, ...]:
    """All metadata of a (possibly *nested*) ``Annotated`` — ``defop`` wraps params in
    ``Annotated[..., Scoped]`` so a manual annotation can end up one layer deep."""
    out: list[Any] = []
    while typing.get_origin(annotation) is Annotated:
        args = typing.get_args(annotation)
        annotation, meta = args[0], args[1:]
        out.extend(meta)
    return tuple(out)


def _has(annotation: Any, kinds: tuple[type, ...]) -> bool:
    return any(isinstance(a, kinds) for a in _annotations(annotation))


def _fold_computation_args(op: Operation, args: Any, kwargs: Any) -> None:
    """Enter each ``Computation`` callback arg (its ops route back through the active
    ``apply`` fold), and refuse any *undeclared* callable arg loudly."""
    try:
        bound = op.__signature__.bind(*args, **kwargs)
    except TypeError:
        return
    bound.apply_defaults()
    for name, p in op.__signature__.parameters.items():
        val = bound.arguments.get(name)
        if _has(p.annotation, (_Computation,)):
            if callable(val):
                val(_Opaque())  # run under the active interpreter -> its ops fold; loud if it inspects its arg
        elif callable(val) and not isinstance(val, _Opaque) and not _has(p.annotation, (Uses,)):
            raise UndeclaredCallable(
                f"{op}: argument {name!r} is callable but not declared `Computation`/`Uses[()]`; "
                "its effects can't be soundly folded — annotate it, or the check is unsound."
            )


def _refuse(*_a: Any, **_k: Any) -> Any:
    raise UnsoundCallbackFold(
        "usesof ran a Computation callback on an opaque placeholder to collect its "
        "effects, but the callback *inspected* its argument (compared it, did arithmetic on "
        "it, called it, branched on it, took its length, accessed an attribute, iterated or "
        "indexed it, …). Folding on a fake value would path/structure-under-approximate; "
        "refusing rather than under-approximating."
    )


class _Opaque:
    """Placeholder fed to a ``Computation`` callback so its op-calls fire. Usable only as
    opaque *data* — passed straight through to operations, which never inspect their
    argument *values* under the fold. **Any** other use (comparison, arithmetic, ``len``,
    ``bool``, call, attribute access, iteration, indexing, formatting, …) raises
    :class:`UnsoundCallbackFold`. So ``lambda x: op()`` and ``lambda x: op(x)`` fold
    soundly, while ``lambda x: a() if x == 0 else b()``, ``lambda x: op(x + 1)`` or
    ``lambda x: x.field`` are refused *loudly* — never silently folded on a fake value.

    The refusal is default-deny: every inspection dunder is bound to :func:`_refuse`
    (below), so an operator we did not anticipate raises rather than silently returning a
    wrong answer (e.g. the identity ``__eq__`` would return ``False`` and drop a branch)."""

    __slots__ = ()


# Default-deny: bind every operation a callback could perform on its argument — other than
# handing it to an op — to a loud refusal. Enumerated so a missed operator fails closed.
_INSPECTION_DUNDERS = (
    # truth / identity / hashing / formatting / conversions
    "__bool__", "__hash__", "__eq__", "__ne__", "__repr__", "__str__", "__format__",
    "__bytes__", "__int__", "__float__", "__complex__", "__index__", "__round__",
    "__trunc__", "__floor__", "__ceil__",
    # ordering
    "__lt__", "__le__", "__gt__", "__ge__",
    # attribute access / call
    "__getattr__", "__call__",
    # container protocol
    "__len__", "__length_hint__", "__contains__", "__getitem__", "__setitem__",
    "__delitem__", "__iter__", "__next__", "__reversed__",
    # context / async
    "__enter__", "__exit__", "__await__", "__aiter__", "__anext__",
    # unary numeric
    "__neg__", "__pos__", "__abs__", "__invert__",
)
# binary numeric, plus reflected (r) and in-place (i) forms
for _binop in (
    "add", "sub", "mul", "matmul", "truediv", "floordiv", "mod", "divmod", "pow",
    "lshift", "rshift", "and", "xor", "or",
):
    _INSPECTION_DUNDERS += (f"__{_binop}__", f"__r{_binop}__", f"__i{_binop}__")

for _dunder in _INSPECTION_DUNDERS:
    setattr(_Opaque, _dunder, _refuse)
