import builtins
import itertools
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Literal, overload

import jax
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

import effectful.handlers.jax.numpy as _jnp
from effectful.internals.runtime import interpreter
from effectful.ops.monoid import NormalizeIntp, Stream, _is_monoid_weighted
from effectful.ops.semantics import apply, evaluate, fvsof, handler
from effectful.ops.syntax import _BaseTerm, defdata, deffn, syntactic_eq
from effectful.ops.types import NotHandled, Operation, Term


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


class Backend(ABC):
    """A value-domain spec used to share monoid tests across int and jax.Array
    backends. Provides the concrete value type, the hypothesis strategy for
    drawing scalars in property tests, and an equality predicate that works
    for that domain.
    """

    name: str
    scalar_typ: Any
    stream_typ: Any
    strategy_for_op: dict[Operation, st.SearchStrategy[Callable[..., Any]]]

    def __init__(self):
        self.strategy_for_op = {}

    @abstractmethod
    def eq(self, a: Any, b: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def strategy(
        self,
        arg_types: tuple[type, ...] = (),
        ret: Literal["scalar", "stream"] = "scalar",
    ) -> SearchStrategy:
        raise NotImplementedError

    def _fresh_op(
        self,
        name: str,
        arg_types: tuple[type, ...] = (),
        ret: Literal["scalar", "stream"] = "scalar",
    ) -> Operation:
        """Build a fresh, unhandled Operation whose parameter and return
        annotations are derived from this backend.

        ``ret`` is ``"scalar"`` for a scalar return or ``"stream"`` for a
        stream-of-scalar return. The operation has ``n_args`` parameters,
        each of type ``scalar_typ``.
        """
        scalar = self.scalar_typ
        out = self.stream_typ if ret == "stream" else scalar
        params = ", ".join(f"_a{i}" for i in range(len(arg_types)))
        ns: dict[str, Any] = {"NotHandled": NotHandled}
        exec(f"def _fn({params}):\n    raise NotHandled\n", ns)
        fn = ns["_fn"]
        fn.__annotations__ = {
            **{f"_a{i}": t for i, t in enumerate(arg_types)},
            "return": out,
        }
        op = Operation.define(fn, name=name)
        self.strategy_for_op[op] = self.strategy(arg_types, ret)
        return op

    @overload
    def define_vars(self, name: str, /, **kwargs) -> Operation: ...

    @overload
    def define_vars(
        self, n1: str, n2: str, /, *names: str, **kwargs
    ) -> tuple[Operation, ...]: ...

    def define_vars(self, *names: str, **kwargs) -> Operation | tuple[Operation, ...]:  # type: ignore[misc]
        if len(names) == 1:
            return self._fresh_op(names[0], **kwargs)
        return tuple(self._fresh_op(n, **kwargs) for n in names)

    def check_rewrite(
        self,
        lhs,
        rhs,
        rule,
        *,
        max_examples: int = 25,
        deadline=None,
        normalize=NormalizeIntp,
    ) -> None:
        with handler(rule):
            norm = evaluate(lhs)
        assert syntactic_eq_alpha(norm, rhs)

        fvs = fvsof(lhs) | fvsof(rhs)

        @st.composite
        def random_interpretation(
            draw: st.DrawFn,
        ) -> Mapping[Operation, Callable[..., Any]]:
            """Draw an Interpretation binding every Operation in `free_vars` to
            a randomly chosen value/callable. Keys are Operation identities.
            """
            intp: dict[Operation, Callable[..., Any]] = {}
            for op, strategy in self.strategy_for_op.items():
                if op in fvs:
                    intp[op] = draw(strategy)
            return intp

        @given(intp=random_interpretation())
        @settings(
            max_examples=max_examples, deadline=deadline, report_multiple_bugs=False
        )
        def _check_semantics(intp):
            with handler(normalize), handler(intp):
                lhs_val = evaluate(lhs)
                rhs_val = evaluate(rhs)
                assert self.eq(lhs_val, rhs_val)

        _check_semantics()


def _is_weighted(x: Any) -> bool:
    return isinstance(x, Term) and _is_monoid_weighted(x.op)


def _weight_pairs(x: Any, monoid: Any) -> list[tuple[Any, Any]] | None:
    """Return ``(element, weight)`` pairs for a stream.

    A weighted-monoid Term yields each element paired with its weight. A plain
    (unweighted) stream yields each element paired with ``monoid.identity`` --
    the no-op weight -- so an unweighted stream compares equal to a weighted one
    exactly when every weight reduces to the identity (e.g. ``[()]`` vs a
    weighted ``[()]`` whose single empty row reduces to the identity, and, more
    generally, whenever both streams are empty). Returns ``None`` for a
    non-stream Term, which never compares equal to a weighted stream.
    """
    if isinstance(x, Term):
        if not _is_monoid_weighted(x.op):
            return None
        stream, weight = x.args
        assert not isinstance(stream, Term)
        return [(e, typing.cast(Callable, weight)(e)) for e in stream]
    return [(e, monoid.identity) for e in x]


def _weighted_stream_eq(a, b, leaf_eq: Callable[[Any, Any], bool]) -> bool:
    monoids = {x.op.__self__ for x in (a, b) if _is_weighted(x)}
    # distinct weight monoids can never be equal
    if len(monoids) != 1:
        return False
    monoid = next(iter(monoids))

    a_pairs = _weight_pairs(a, monoid)
    b_pairs = _weight_pairs(b, monoid)
    if a_pairs is None or b_pairs is None or len(a_pairs) != len(b_pairs):
        return False
    for (ea, wa), (eb, wb) in zip(a_pairs, b_pairs):
        if not leaf_eq(ea, eb) or not leaf_eq(wa, wb):
            return False
    return True


class IntBackend(Backend):
    name = "int"
    scalar_typ = int
    stream_typ = Stream[int]

    _unary_num_fns: list[Callable[[int], int]] = [
        lambda x: x,
        lambda x: x + 1,
        lambda x: x - 1,
        lambda x: -x,
        lambda x: 2 * x,
        lambda x: 3 * x + 1,
    ]

    _binary_num_fns: list[Callable[[int, int], int]] = [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: x * y,
        lambda x, y: x + 2 * y,
        lambda x, y: 2 * x - y,
    ]

    _unary_list_fns: list[Callable[[int], list[int]]] = [
        lambda _x: [],
        lambda x: [x],
        lambda x: [x, x + 1],
        lambda x: [x, -x],
        lambda x: [0, x, x + 1],
    ]

    def strategy(
        self,
        arg_types: tuple[type, ...] = (),
        ret: Literal["scalar", "stream"] = "scalar",
    ) -> SearchStrategy:
        match arg_types, ret:
            case (), "scalar":
                return st.integers(min_value=-100, max_value=100).map(deffn)
            case (), "stream":
                scalars = st.integers(min_value=-100, max_value=100)
                return st.lists(scalars, max_size=2).map(deffn)
            case (builtins.int,), "scalar":
                return st.sampled_from(self._unary_num_fns)
            case (builtins.int, builtins.int), "scalar":
                return st.sampled_from(self._binary_num_fns)
            case (builtins.int,), "stream":
                return st.sampled_from(self._unary_list_fns)
        raise NotImplementedError(
            f"No int strategy for op with return {ret!r} and {arg_types} args"
        )

    def eq(self, a: Any, b: Any) -> bool:
        if _is_weighted(a) or _is_weighted(b):
            return _weighted_stream_eq(a, b, self.eq)
        return not isinstance(a, Term) and not isinstance(b, Term) and a == b


class JaxBackend(Backend):
    name = "jax"
    scalar_typ = jax.Array
    stream_typ = jax.Array

    _unary_jax_scalar_fns: list[Callable[[jax.Array], jax.Array]] = [
        lambda a: a,
        lambda a: a + 1,
        lambda a: a - 1,
        lambda a: -a,
        lambda a: 2 * a,
    ]

    _unary_jax_stream_fns: list[Callable[[jax.Array], Stream[jax.Array]]] = [
        lambda a: _jnp.stack([a, a + 1]),
        lambda a: _jnp.stack([a, -a]),
        lambda a: _jnp.stack([a, a + 1, 2 * a]),
    ]

    _binary_jax_scalar_fns: list[Callable[[jax.Array, jax.Array], jax.Array]] = [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
    ]

    def strategy(
        self,
        arg_types: tuple[type, ...] = (),
        ret: Literal["scalar", "stream"] = "scalar",
    ) -> st.SearchStrategy[Callable]:
        match arg_types, ret:
            case (), "scalar":
                return (
                    st.lists(
                        st.integers(min_value=-5, max_value=5),
                        min_size=2,
                        max_size=2,
                    )
                    .map(lambda xs: jax.numpy.asarray(xs, dtype=jax.numpy.float32))
                    .map(deffn)
                )
            case (), "stream":
                return (
                    st.lists(
                        st.integers(min_value=-5, max_value=5),
                        min_size=1,
                        max_size=2,
                    )
                    .map(lambda xs: jax.numpy.asarray(xs, dtype=jax.numpy.float32))
                    .map(deffn)
                )
            case (jax.Array,), "scalar":
                return st.sampled_from(self._unary_jax_scalar_fns)
            case (jax.Array, jax.Array), "scalar":
                return st.sampled_from(self._binary_jax_scalar_fns)
            case (jax.Array,), "stream":
                return st.sampled_from(self._unary_jax_stream_fns)

        raise NotImplementedError(
            f"No jax strategy for op with return {ret!r} and {arg_types} args"
        )

    def eq(self, a: Any, b: Any) -> bool:
        if _is_weighted(a) or _is_weighted(b):
            return _weighted_stream_eq(a, b, self.eq)

        def _leaf_eq(x: Any, y: Any) -> bool:
            return bool(jax.numpy.all(jax.numpy.isclose(x, y, equal_nan=True)))

        try:
            leaves = jax.tree.leaves(jax.tree.map(_leaf_eq, a, b))
        except (ValueError, TypeError):
            return False
        return all(leaves)


__all__ = ["Backend", "IntBackend", "JaxBackend", "syntactic_eq_alpha"]
