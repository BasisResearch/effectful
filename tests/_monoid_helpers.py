import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any, get_args, get_origin

import jax
from hypothesis import strategies as st

import effectful.handlers.jax.numpy as _jnp
from effectful.internals.runtime import interpreter
from effectful.ops.semantics import apply, evaluate
from effectful.ops.syntax import _BaseTerm, defdata, deffn, syntactic_eq
from effectful.ops.types import Operation

_JAX_ARRAY_SHAPE = (3,)


def _jax_array_value_strategy() -> st.SearchStrategy[jax.Array]:
    return st.integers(min_value=0, max_value=2**31 - 1).map(
        lambda seed: jax.random.uniform(
            jax.random.PRNGKey(seed), _JAX_ARRAY_SHAPE, minval=0.5, maxval=1.5
        )
    )


# Unary jax fns map a scalar to a 1-D array (analogous to ``_UNARY_LIST_FNS``
# for ints). Uses the effectful-wrapped jnp so named-dim broadcasting works.
_UNARY_JAX_FNS: list[Callable[[jax.Array], jax.Array]] = [
    lambda a: _jnp.stack([a, a + 1.0]),
    lambda a: _jnp.stack([a, -a]),
    lambda a: _jnp.stack([a, a + 1.0, 2.0 * a]),
]

_BINARY_JAX_FNS: list[Callable[[jax.Array, jax.Array], jax.Array]] = [
    lambda a, b: a + b,
    lambda a, b: a - b,
    lambda a, b: a * b,
]


def _value_strategy_for(annotation: Any) -> st.SearchStrategy[Any]:
    """Strategy for the value an *0-arg* Operation should return."""
    if annotation is int:
        return st.integers()
    if annotation is float:
        return st.floats(allow_nan=False)
    if get_origin(annotation) is list and get_args(annotation) == (int,):
        return st.lists(st.integers())
    if annotation is jax.Array:
        return _jax_array_value_strategy()
    raise NotImplementedError(
        f"No value strategy for return annotation {annotation!r}; "
        "supported: int, list[int], jax.Array"
    )


_UNARY_INT_FNS: list[Callable[[int], int]] = [
    lambda x: x,
    lambda x: x + 1,
    lambda x: x - 1,
    lambda x: -x,
    lambda x: 2 * x,
    lambda x: 3 * x + 1,
]

_BINARY_INT_FNS: list[Callable[[int, int], int]] = [
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


def _strategy_for_op(op: Operation) -> st.SearchStrategy[Callable[..., Any]]:
    """Pick a strategy producing a callable suitable for binding `op` in an
    interpretation. Inspects the operation's signature.
    """
    sig = op.__signature__
    params = list(sig.parameters.values())
    ret = sig.return_annotation
    param_types = tuple(p.annotation for p in params)

    if not params:
        return _value_strategy_for(ret).map(deffn)
    if ret is int and param_types == (int,):
        return st.sampled_from(_UNARY_INT_FNS)
    if ret is int and param_types == (int, int):
        return st.sampled_from(_BINARY_INT_FNS)
    if get_origin(ret) is list and get_args(ret) == (int,) and param_types == (int,):
        return st.sampled_from(_UNARY_LIST_FNS)
    if ret is jax.Array and param_types == (jax.Array,):
        return st.sampled_from(_UNARY_JAX_FNS)
    if ret is jax.Array and param_types == (jax.Array, jax.Array):
        return st.sampled_from(_BINARY_JAX_FNS)
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

    def _bound_var_order(args, kwargs, bound_set):
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

    def _apply_canonical(op, *args, **kwargs):
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


__all__ = ["random_interpretation", "define_vars", "syntactic_eq_alpha"]
