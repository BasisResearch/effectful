import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, get_args, get_origin

import jax
from hypothesis import given, settings
from hypothesis import strategies as st

import effectful.handlers.jax.numpy as _jnp
from effectful.internals.runtime import interpreter
from effectful.ops.monoid import NormalizeIntp
from effectful.ops.semantics import apply, evaluate, handler
from effectful.ops.syntax import _BaseTerm, defdata, deffn, syntactic_eq
from effectful.ops.types import NotHandled, Operation, Term

_JAX_ARRAY_SHAPE = (2,)


def _jax_array_value_strategy() -> st.SearchStrategy[jax.Array]:
    return st.lists(
        st.integers(min_value=-5, max_value=5),
        min_size=_JAX_ARRAY_SHAPE[0],
        max_size=_JAX_ARRAY_SHAPE[0],
    ).map(lambda xs: jax.numpy.asarray(xs, dtype=jax.numpy.float32))


# Shape-preserving unary jax fns: scalar → scalar (counterpart of
# ``_UNARY_NUM_FNS`` for ints). Used for ops declared with ``ret="scalar"``.
_UNARY_JAX_SCALAR_FNS: list[Callable[[jax.Array], jax.Array]] = [
    lambda a: a,
    lambda a: a + 1,
    lambda a: a - 1,
    lambda a: -a,
    lambda a: 2 * a,
]

# Unary jax fns map a scalar to a 1-D array (analogous to ``_UNARY_LIST_FNS``
# for ints). Uses the effectful-wrapped jnp so named-dim broadcasting works.
# Used for ops declared with ``ret="stream"``.
_UNARY_JAX_FNS: list[Callable[[jax.Array], jax.Array]] = [
    lambda a: _jnp.stack([a, a + 1]),
    lambda a: _jnp.stack([a, -a]),
    lambda a: _jnp.stack([a, a + 1, 2 * a]),
]

_BINARY_JAX_FNS: list[Callable[[jax.Array, jax.Array], jax.Array]] = [
    lambda a, b: a + b,
    lambda a, b: a - b,
    lambda a, b: a * b,
]


def _value_strategy_for(annotation: Any) -> st.SearchStrategy[Any]:
    """Strategy for the value an *0-arg* Operation should return."""
    if annotation is int:
        return st.integers(min_value=-100, max_value=100)
    if annotation is float:
        return st.floats(allow_nan=False)
    if get_origin(annotation) is list and get_args(annotation) == (int,):
        return st.lists(st.integers(min_value=-100, max_value=100), max_size=2)
    if annotation is jax.Array:
        return _jax_array_value_strategy()
    if get_origin(annotation) is list and get_args(annotation) == (jax.Array,):
        return st.lists(_jax_array_value_strategy(), max_size=2)
    raise NotImplementedError(
        f"No value strategy for return annotation {annotation!r}; "
        "supported: int, list[int], jax.Array, list[jax.Array]"
    )


_UNARY_NUM_FNS: list[Callable[[int], int]] = [
    lambda x: x,
    lambda x: x + 1,
    lambda x: x - 1,
    lambda x: -x,
    lambda x: 2 * x,
    lambda x: 3 * x + 1,
]

_BINARY_NUM_FNS: list[Callable[[int, int], int]] = [
    lambda x, y: x + y,
    lambda x, y: x - y,
    lambda x, y: x * y,
    lambda x, y: x + 2 * y,
    lambda x, y: 2 * x - y,
]

_UNARY_LIST_FNS: list[Callable[[int], list[int]]] = [
    lambda _x: [],
    lambda x: [x],
    lambda x: [x, x + 1],
    lambda x: [x, -x],
    lambda x: [0, x, x + 1],
]

_UNARY_JAX_LIST_FNS: list[Callable[[jax.Array], list[jax.Array]]] = [
    lambda _x: [],
    lambda x: [x],
    lambda x: [x, x + 1],
    lambda x: [x, -x],
]


def _is_stream(annotation: Any) -> bool:
    """True if ``annotation`` carries the ``"stream"`` Annotated marker.

    On the JAX backend ``scalar_typ`` and ``stream_typ`` are both ``jax.Array``,
    so :meth:`Backend.fresh_op` tags stream returns as
    ``Annotated[jax.Array, "stream"]`` to keep them distinguishable here.
    """
    return get_origin(annotation) is Annotated and "stream" in annotation.__metadata__


def _strip(annotation: Any) -> Any:
    """Strip an ``Annotated`` wrapper to its underlying type."""
    if get_origin(annotation) is Annotated:
        return get_args(annotation)[0]
    return annotation


def _strategy_for_op(op: Operation) -> st.SearchStrategy[Callable[..., Any]]:
    """Pick a strategy producing a callable suitable for binding `op` in an
    interpretation. Inspects the operation's signature.
    """
    sig = op.__signature__
    params = list(sig.parameters.values())
    ret_annot = sig.return_annotation
    ret = _strip(ret_annot)
    ret_is_stream = _is_stream(ret_annot)
    param_types = tuple(_strip(p.annotation) for p in params)

    if not params:
        return _value_strategy_for(ret).map(deffn)
    if ret in (int, float) and param_types == (int,):
        return st.sampled_from(_UNARY_NUM_FNS)
    if ret in (int, float) and param_types == (int, int):
        return st.sampled_from(_BINARY_NUM_FNS)
    if get_origin(ret) is list and get_args(ret) == (int,) and param_types == (int,):
        return st.sampled_from(_UNARY_LIST_FNS)
    if ret is jax.Array and param_types == (jax.Array,):
        if ret_is_stream:
            return st.sampled_from(_UNARY_JAX_FNS)
        return st.sampled_from(_UNARY_JAX_SCALAR_FNS)
    if ret is jax.Array and param_types == (jax.Array, jax.Array):
        return st.sampled_from(_BINARY_JAX_FNS)
    if (
        get_origin(ret) is list
        and get_args(ret) == (jax.Array,)
        and param_types == (jax.Array,)
    ):
        return st.sampled_from(_UNARY_JAX_LIST_FNS)
    raise NotImplementedError(
        f"No callable strategy for free var with return {ret!r}, params {param_types!r}"
    )


@st.composite
def random_interpretation(
    draw: st.DrawFn, free_vars: Sequence[Operation]
) -> Mapping[Operation, Callable[..., Any]]:
    """Draw an Interpretation binding every Operation in `case.free_vars` to
    a randomly chosen value/callable. Keys are Operation identities.
    """
    intp: dict[Operation, Callable[..., Any]] = {}
    for op in free_vars:
        intp[op] = draw(_strategy_for_op(op))
    return intp


def define_vars(*names, typ=int):
    if len(names) == 1:
        return Operation.define(typ, name=names[0])
    return tuple(Operation.define(typ, name=n) for n in names)


def syntactic_eq_alpha(x, y) -> bool:
    """Alpha-equivalence-respecting variant of ``syntactic_eq``.

    Walks each expression bottom-up with :func:`evaluate` and renames
    every bound variable to a deterministic canonical Operation. The
    canonical names are assigned by a counter that increments in
    ``evaluate``'s natural traversal order, so two alpha-equivalent
    expressions canonicalize to syntactically identical results.
    """

    _op_cache: dict[int, Operation] = {}

    def _canonical_op(idx: int, op: Operation) -> Operation:
        """Cached canonical Operation, keyed by encounter index.

        Cached so that two independent canonicalize runs return the same
        Operation object for the same index — letting ``syntactic_eq``
        compare canonical forms by Operation identity.
        """
        if idx in _op_cache:
            return _op_cache[idx]

        op = Operation.define(op, name=f"__cv_{idx}")
        _op_cache[idx] = op
        return op

    cx = _canonicalize(x, _canonical_op)
    cy = _canonicalize(y, _canonical_op)
    return syntactic_eq(cx, cy)


def _canonicalize(expr, _canonical_op):
    counter = itertools.count()

    def _substitute(arg, renaming):
        """Apply a bound-variable renaming using ``evaluate`` for traversal."""
        if not renaming:
            return arg
        with interpreter({apply: _BaseTerm, **renaming}):
            return evaluate(arg)

    def _bound_var_order(args, kwargs, bound_set: set[Operation]) -> list[Operation]:
        """Return bound variables in deterministic encounter order."""
        seen: list[Operation] = []
        seen_set: set[Operation] = set()

        def _capture(op, *a, **kw):
            if op in bound_set and op not in seen_set:
                seen.append(op)
                seen_set.add(op)
            return defdata(op, *a, **kw)

        # ``evaluate`` walks Terms, lists, tuples, mappings, dataclasses,
        # etc. for free; the apply handler captures bound vars used as
        # ``x()`` anywhere in the body.
        with interpreter({apply: _capture}):
            evaluate((args, kwargs))

        # Binders bypass the apply handler. Pick them up with a small structural
        # walk that visits dict keys too.
        def _walk_bare(obj):
            if isinstance(obj, Operation):
                if obj in bound_set and obj not in seen_set:
                    seen.append(obj)
                    seen_set.add(obj)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _walk_bare(k)
                    _walk_bare(v)
            elif isinstance(obj, list | set | frozenset | tuple):
                for v in obj:
                    _walk_bare(v)

        _walk_bare((args, kwargs))
        return seen

    def _apply_canonical(op, *args, **kwargs) -> Term:
        bindings = op.__fvs_rule__(*args, **kwargs)
        all_bound: set[Operation] = set().union(
            *bindings.args, *bindings.kwargs.values()
        )
        if not all_bound:
            return _BaseTerm(op, *args, **kwargs)

        order = _bound_var_order(args, kwargs, all_bound)
        canonical = {var: _canonical_op(next(counter), var) for var in order}
        assert all_bound <= set(order)

        new_args = tuple(
            _substitute(
                arg, {v: canonical[v] for v in bindings.args[i] if v in canonical}
            )
            for i, arg in enumerate(args)
        )
        new_kwargs = {
            k: _substitute(
                v,
                {var: canonical[var] for var in bindings.kwargs[k] if var in canonical},
            )
            for k, v in kwargs.items()
        }

        # avoid the renaming from defdata
        return _BaseTerm(op, *new_args, **new_kwargs)

    with interpreter({apply: _apply_canonical}):
        return evaluate(expr)


@dataclass(frozen=True)
class Backend:
    """A value-domain spec used to share monoid tests across int and jax.Array
    backends. Provides the concrete value type, the hypothesis strategy for
    drawing scalars in property tests, and an equality predicate that works
    for that domain.
    """

    name: str
    scalar_typ: Any
    stream_typ: Any
    scalar_strategy: st.SearchStrategy[Any]
    eq: Callable[[Any, Any], bool]

    def fresh_op(self, name: str, n_args: int = 1, ret: str = "scalar") -> Operation:
        """Build a fresh, unhandled Operation whose parameter and return
        annotations are derived from this backend.

        ``ret`` is ``"scalar"`` for a scalar return or ``"stream"`` for a
        stream-of-scalar return. The operation has ``n_args`` parameters,
        each of type ``scalar_typ``.
        """
        scalar = self.scalar_typ
        if ret == "stream":
            out = self.stream_typ
            # When scalar_typ == stream_typ (e.g. jax backend), tag the return
            # with an Annotated marker so ``_strategy_for_op`` can pick the
            # right (shape-changing) function family.
            if scalar is out:
                out = Annotated[out, "stream"]
        else:
            out = scalar
        params = ", ".join(f"_a{i}" for i in range(n_args))
        ns: dict[str, Any] = {"NotHandled": NotHandled}
        exec(f"def _fn({params}):\n    raise NotHandled\n", ns)
        fn = ns["_fn"]
        fn.__annotations__ = {
            **{f"_a{i}": scalar for i in range(n_args)},
            "return": out,
        }
        return Operation.define(fn, name=name)


def _int_eq(a: Any, b: Any) -> bool:
    return not isinstance(a, Term) and not isinstance(b, Term) and a == b


def _jax_eq(a: Any, b: Any) -> bool:
    def _leaf_eq(x: Any, y: Any) -> bool:
        return bool(jax.numpy.all(jax.numpy.isclose(x, y, equal_nan=True)))

    try:
        leaves = jax.tree.leaves(jax.tree.map(_leaf_eq, a, b))
    except (ValueError, TypeError):
        return False
    return all(leaves)


def check_rewrite(
    lhs,
    rhs,
    rule,
    *,
    backend: Backend,
    free_vars=[],
    max_examples: int = 25,
    deadline=None,
    normalize=NormalizeIntp,
) -> None:
    with handler(rule):
        norm = evaluate(lhs)
    assert syntactic_eq_alpha(norm, rhs)

    @given(intp=random_interpretation(free_vars))
    @settings(max_examples=max_examples, deadline=deadline)
    def _check_semantics(intp):
        with handler(normalize), handler(intp):
            lhs_val = evaluate(lhs)
            rhs_val = evaluate(rhs)
        assert backend.eq(lhs_val, rhs_val)

    _check_semantics()


INT_BACKEND = Backend(
    name="int",
    scalar_typ=int,
    stream_typ=list[int],
    scalar_strategy=st.integers(min_value=-100, max_value=100),
    eq=_int_eq,
)


JAX_BACKEND = Backend(
    name="jax",
    scalar_typ=jax.Array,
    stream_typ=jax.Array,
    scalar_strategy=_jax_array_value_strategy(),
    eq=_jax_eq,
)


__all__ = [
    "Backend",
    "INT_BACKEND",
    "JAX_BACKEND",
    "random_interpretation",
    "define_vars",
    "syntactic_eq_alpha",
    "check_rewrite",
]
