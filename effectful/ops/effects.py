"""Effect-row inference for effectful — the ``ε`` engine.

Sibling of :func:`effectful.ops.semantics.typeof` / :func:`fvsof`: a fold over the
universal ``apply`` operation that computes which operations a term performs (its
**effect row**), plus the ``Uses`` / ``Computation`` / ``Requires`` annotations
(#448, #664) that operations declare and the fold reads.

This branches off ``master``: #694 (the doctest-validation PR) touches only
``handlers/llm/`` and **never** ``ops/``, so the engine has no dependency on it. The
LLM integration (an ``ε`` validator at #694's decode stage + tool governance on its
``tool_types`` seam) is a separate, later follow-up in ``handlers/llm/``.

**Final upstream placement** (kept in one module here to be reviewable without editing
core files):

======================  =====================================================
this module             upstream home
======================  =====================================================
``usesof`` / ``effectsof``   ``ops/semantics.py``, next to ``typeof`` / ``fvsof``
``uses_rule``               a method ``Operation.__uses_rule__`` in ``ops/types.py``
``Uses`` / ``Computation`` / ``Requires``   ``ops/syntax.py``, next to ``Scoped``
======================  =====================================================

**Deferred (not in this module yet):** `__uses_rule__` as a real ``@final`` method on
``Operation`` (so subclassed ops like ``Tool``/``Template`` can override their row) —
currently the free ``uses_rule``; wiring ``Annotation.infer_annotations`` into ``defop``
(the build-time gate); ``usagesof`` (usage multiset) and handler discharge (2nd PR);
polymorphic ``Operation[[A], B]`` ``Uses`` members; the autumn thin consumer; and the whole
LLM layer (Branch B, off #694) — ``toolsof``/``reachable_tools``, the ``ε``-validator at the
decode seam, tool governance. This module is the ``ε`` core only.
"""

import collections.abc
import inspect
import typing
from dataclasses import dataclass
from typing import Annotated, Any

from effectful.internals.runtime import interpreter
from effectful.ops.semantics import apply, evaluate, typeof
from effectful.ops.types import Annotation, Expr, Operation

__all__ = [
    "Uses",
    "Computation",
    "Requires",
    "UndeclaredCallable",
    "UnsoundCallbackFold",
    "uses_rule",
    "usesof",
    "effectsof",
    "effect_type",
    "check_uses",
    "requires_rule",
    "check_requires",
]


# ---------------------------------------------------------------------------
# Annotations an op declares (read off ``Annotated[T, ...]``)
# ---------------------------------------------------------------------------
class Uses:
    """Effect-row annotation metadata: ``Annotated[T, Uses[op1, op2, ...]]`` on a
    *return* type. ``Uses[()]`` is the empty row — explicitly pure. Read by
    :func:`uses_rule`. (Not an :class:`Annotation` subtype: it is plain metadata, not
    a signature transform.)"""

    __slots__ = ("members",)

    def __class_getitem__(cls, items: Any) -> "Uses":
        return cls(items if isinstance(items, tuple) else (items,))

    def __init__(self, members: tuple[Any, ...]) -> None:
        self.members = members

    def __repr__(self) -> str:
        return f"Uses{list(self.members)!r}"


@dataclass(frozen=True)
class _Computation(Annotation):
    """``Annotated[Callable[[A], B], Computation]`` on an *argument*: a suspended
    computation whose effect row joins the op's row when the op runs it. Higher-order
    combinators (``map``/``filter``/…) mark their callback arg with this instead of the
    checker hard-coding which ops are higher-order."""

    @classmethod
    def infer_annotations(cls, sig: inspect.Signature) -> inspect.Signature:
        # Validates that every callable-typed parameter is classified — Computation (its
        # effects fold) or Uses[()] (declared not-effectful). NOTE: this is NOT yet wired
        # into `defop`, so it is a validate-if-called helper, not an automatic build-time
        # gate. Actual enforcement is at `usesof`-time: `_fold_computation_args` raises
        # `UndeclaredCallable` on an unclassified callable. Wiring generic
        # `Annotation.infer_annotations` dispatch into op construction is deferred.
        for name, p in sig.parameters.items():
            if _is_callable_annotation(p.annotation) and not _has(p.annotation, (_Computation, Uses)):
                raise TypeError(
                    f"{name!r}: a callable argument must be declared `Computation` or "
                    "`Uses[()]` — else its effects would be silently dropped."
                )
        return sig


#: Singleton marker (data-less, like ``IsRecursive``) — use in ``Annotated[C, Computation]``.
Computation = _Computation()


@dataclass(frozen=True, init=False)
class Requires(Annotation):
    """``Annotated[T, Requires(op, ...)]`` on an *argument*: the value's provenance must
    cover these ops — ``{op,...} ⊆ usesof(arg)``. The precondition dual of ``Uses``
    (#664)."""

    ops: frozenset[Operation]

    def __init__(self, *ops: Operation) -> None:
        object.__setattr__(self, "ops", frozenset(ops))

    @classmethod
    def infer_annotations(cls, sig: inspect.Signature) -> inspect.Signature:
        ret = sig.return_annotation
        if typing.get_origin(ret) is Annotated and any(
            isinstance(a, cls) for a in typing.get_args(ret)[1:]
        ):
            raise TypeError("Requires annotates arguments, not return types.")
        return sig

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
# The per-op rule (upstream: Operation.__uses_rule__, default {self})
# ---------------------------------------------------------------------------
def uses_rule(op: Operation) -> frozenset[Operation]:
    """The effect row an op contributes *itself*: its declared ``Uses[...]`` if the
    return is annotated, else ``{op}``. (``Computation`` args are folded by
    :func:`usesof`, not here.)"""
    declared = _declared_uses(op)
    return declared if declared is not None else frozenset({op})


# ---------------------------------------------------------------------------
# The fold (upstream: ops/semantics.py, next to typeof/fvsof)
# ---------------------------------------------------------------------------
def usesof[S](term: Expr[S]) -> frozenset[Operation]:
    """Return the effect row of a term: the set of operations it performs. The
    effect-typing sibling of :func:`typeof` / :func:`fvsof`.

    Each applied op contributes :func:`uses_rule` (default ``{self}``, ``Uses[()]`` =
    pure). ``Computation``-marked callback args are *entered* so their effects fold too;
    an undeclared callable arg raises :class:`UndeclaredCallable` (never silent).
    """
    used: set[Operation] = set()

    def _update(op: Operation, *args: Any, **kwargs: Any) -> Any:
        used.update(uses_rule(op))
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
    declared = _declared_uses(op)
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


def _is_callable_annotation(annotation: Any) -> bool:
    while typing.get_origin(annotation) is Annotated:
        annotation = typing.get_args(annotation)[0]
    return annotation is collections.abc.Callable or typing.get_origin(annotation) is collections.abc.Callable


def _member_ops(m: Any) -> frozenset[Operation]:
    if typing.get_origin(m) is typing.Literal:
        return frozenset(a for a in typing.get_args(m) if isinstance(a, Operation))
    if isinstance(m, Operation):
        return frozenset({m})
    raise NotImplementedError(  # loud, not a silent frozenset() drop
        f"Uses member {m!r} is not supported: use `Literal[op]` or a bare `Operation`. "
        "Polymorphic `Operation[[A], B]` members are a TODO (#448 §8 Q4)."
    )


def _declared_uses(op: Operation) -> frozenset[Operation] | None:
    """The ``Uses[...]`` row off ``op``'s return annotation, or ``None`` if no ``Uses``
    metadata is present (distinct from ``Uses[()]`` = present-and-empty = pure)."""
    ret = op.__signature__.return_annotation
    found: frozenset[Operation] | None = None
    for anno in _annotations(ret):
        if isinstance(anno, Uses):
            found = (found or frozenset())
            for m in anno.members:
                found |= _member_ops(m)
    return found


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
        elif callable(val) and not _has(p.annotation, (Uses,)):
            raise UndeclaredCallable(
                f"{op}: argument {name!r} is callable but not declared `Computation`/`Uses[()]`; "
                "its effects can't be soundly folded — annotate it, or the check is unsound."
            )


def _refuse(*_a: Any, **_k: Any) -> Any:
    raise UnsoundCallbackFold(
        "usesof ran a Computation callback on an opaque placeholder to collect its "
        "effects, but the callback inspected its argument (branched on it, accessed an "
        "attribute, or destructured it). Folding it on a fake value would "
        "path/structure-under-approximate; refusing rather than under-approximating."
    )


class _Opaque:
    """Placeholder fed to a ``Computation`` callback so its op-calls fire. Usable only as
    opaque *data* (passed on to ops); any *inspection* — bool-test, attribute access,
    iteration, indexing — raises :class:`UnsoundCallbackFold`. So ``lambda x: op()`` and
    ``lambda x: op(x)`` fold soundly, while ``lambda x: a() if x else b()`` or
    ``lambda x: x.field`` are refused loudly rather than folded on a fake value."""

    __slots__ = ()

    __bool__ = _refuse
    __getattr__ = _refuse
    __iter__ = _refuse
    __getitem__ = _refuse
