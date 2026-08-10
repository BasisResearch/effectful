import functools
import itertools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from effectful.internals.runtime import interpreter
from effectful.ops.monoid import Max, Min, NormalizeIntp, Product, Semilattice, Sum
from effectful.ops.semantics import apply, evaluate, fvsof, handler
from effectful.ops.syntax import _BaseTerm, defdata, syntactic_eq
from effectful.ops.types import NotHandled, Operation
from tests._monoid_helpers import random_interpretation

_INT = st.integers(min_value=-100, max_value=100)

ALL_MONOIDS = [
    pytest.param(Sum, id="Sum"),
    pytest.param(Product, id="Product"),
    pytest.param(Min, id="Min"),
    pytest.param(Max, id="Max"),
]

COMMUTATIVE = [
    pytest.param(Sum, id="Sum"),
    pytest.param(Product, id="Product"),
    pytest.param(Min, id="Min"),
    pytest.param(Max, id="Max"),
]

IDEMPOTENT = [
    pytest.param(Min, id="Min"),
    pytest.param(Max, id="Max"),
]

WITH_ZERO = [
    pytest.param(Product, id="Product"),
]


def define_vars(*names, typ=int):
    if len(names) == 1:
        return Operation.define(typ, name=names[0])
    return tuple(Operation.define(typ, name=n) for n in names)


@functools.cache
def _canonical_op(idx: int) -> Operation:
    """Globally cached canonical Operation, keyed by encounter index.

    Cached so that two independent canonicalize runs return the same
    Operation object for the same index — letting ``syntactic_eq``
    compare canonical forms by Operation identity.
    """
    return Operation.define(int, name=f"__cv_{idx}")


def syntactic_eq_alpha(x, y) -> bool:
    """Alpha-equivalence-respecting variant of ``syntactic_eq``.

    Walks each expression bottom-up with :func:`evaluate` and renames
    every bound variable to a deterministic canonical Operation. The
    canonical names are assigned by a counter that increments in
    ``evaluate``'s natural traversal order, so two alpha-equivalent
    expressions canonicalize to syntactically identical results.
    """
    return syntactic_eq(_canonicalize(x), _canonicalize(y))


def _canonicalize(expr):
    counter = itertools.count()

    def _passthrough(op, *args, **kwargs):
        return defdata(op, *args, **kwargs)

    def _substitute(arg, renaming):
        """Apply a bound-variable renaming using ``evaluate`` for traversal."""
        if not renaming:
            return arg
        with interpreter({apply: _passthrough, **renaming}):
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
            return defdata(op, *args, **kwargs)

        order = _bound_var_order(args, kwargs, all_bound)
        canonical = {var: _canonical_op(next(counter)) for var in order}
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


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
@given(a=_INT, b=_INT, c=_INT)
@settings(max_examples=50, deadline=None)
def test_associativity(monoid, a, b, c):
    left = monoid.plus(monoid.plus(a, b), c)
    right = monoid.plus(a, monoid.plus(b, c))
    assert left == right


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
@given(a=_INT)
@settings(max_examples=50, deadline=None)
def test_identity(monoid, a):
    assert monoid.plus(monoid.identity, a) == a
    assert monoid.plus(a, monoid.identity) == a


@pytest.mark.parametrize("monoid", COMMUTATIVE)
@given(a=_INT, b=_INT)
@settings(max_examples=50, deadline=None)
def test_commutativity(monoid, a, b):
    assert monoid.plus(a, b) == monoid.plus(b, a)


@pytest.mark.parametrize("monoid", IDEMPOTENT)
@given(a=_INT)
@settings(max_examples=50, deadline=None)
def test_idempotence(monoid, a):
    assert monoid.plus(a, a) == a


@pytest.mark.parametrize("monoid", WITH_ZERO)
@given(a=_INT)
@settings(max_examples=50, deadline=None)
def test_zero_absorbs(monoid, a):
    assert monoid.plus(monoid.zero, a) == monoid.zero
    assert monoid.plus(a, monoid.zero) == monoid.zero


def _check_pair(lhs, rhs, *, free_vars=[], max_examples: int = 25) -> None:
    """Run structural + semantic checks on a TermPair."""
    with handler(NormalizeIntp):
        norm = evaluate(lhs)

    assert syntactic_eq_alpha(norm, rhs)

    @given(intp=random_interpretation(free_vars))
    @settings(max_examples=max_examples, deadline=None)
    def _check_semantics(intp):
        with handler(intp):
            lhs_val = evaluate(lhs)
            rhs_val = evaluate(rhs)
        assert lhs_val == rhs_val

    _check_semantics()


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_empty(monoid):
    _check_pair(lhs=monoid.plus(), rhs=monoid.identity)


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_single(monoid):
    x = define_vars("x", typ=type(monoid.identity))
    _check_pair(lhs=monoid.plus(x()), rhs=x(), free_vars=[x])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_identity_right(monoid):
    x = define_vars("x", typ=type(monoid.identity))
    _check_pair(lhs=monoid.plus(x(), monoid.identity), rhs=x(), free_vars=[x])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_identity_left(monoid):
    x = define_vars("x", typ=type(monoid.identity))
    _check_pair(lhs=monoid.plus(monoid.identity, x()), rhs=x(), free_vars=[x])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_assoc_right(monoid):
    x, y, z = define_vars("x", "y", "z", typ=type(monoid.identity))
    _check_pair(
        lhs=monoid.plus(x(), monoid.plus(y(), z())),
        rhs=monoid.plus(x(), y(), z()),
        free_vars=[x, y, z],
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_assoc_left(monoid):
    x, y, z = define_vars("x", "y", "z", typ=type(monoid.identity))
    _check_pair(
        lhs=monoid.plus(monoid.plus(x(), y()), z()),
        rhs=monoid.plus(x(), y(), z()),
        free_vars=[x, y, z],
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_sequence(monoid):
    a, b, c, d = define_vars("a", "b", "c", "d", typ=type(monoid.identity))
    _check_pair(
        lhs=monoid.plus([a(), b()], [c(), d()]),
        rhs=[monoid.plus(a(), c()), monoid.plus(b(), d())],
        free_vars=[a, b, c, d],
    )


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_plus_mapping(monoid):
    a, b, c, d = define_vars("a", "b", "c", "d", typ=type(monoid.identity))
    _check_pair(
        lhs=monoid.plus({"x": a(), "y": b()}, {"x": c(), "z": d()}),
        rhs={"x": monoid.plus(a(), c()), "y": b(), "z": d()},
        free_vars=[a, b, c, d],
    )


def test_plus_distributes():
    a, b, c, d = define_vars("a", "b", "c", "d")
    lhs = Product.plus(Sum.plus(a(), b()), Sum.plus(c(), d()))
    rhs = Sum.plus(
        Product.plus(a(), c()),
        Product.plus(a(), d()),
        Product.plus(b(), c()),
        Product.plus(b(), d()),
    )
    _check_pair(lhs=lhs, rhs=rhs, free_vars=[a, b, c, d])


def test_plus_distributes_constant():
    a, b, c, d = define_vars("a", "b", "c", "d")
    lhs = Product.plus(Sum.plus(a(), b()), Sum.plus(c(), d()), 5)
    rhs = Product.plus(
        5,
        Sum.plus(
            Product.plus(a(), c()),
            Product.plus(a(), d()),
            Product.plus(b(), c()),
            Product.plus(b(), d()),
        ),
    )
    _check_pair(lhs=lhs, rhs=rhs, free_vars=[a, b, c, d])


def test_plus_distributes_multiple():
    a, b, c, d = define_vars("a", "b", "c", "d")
    lhs = Sum.plus(
        Min.plus(a(), b()),
        Min.plus(c(), d()),
        Max.plus(a(), b()),
        Max.plus(c(), d()),
    )
    rhs = Sum.plus(
        Min.plus(
            Sum.plus(a(), c()),
            Sum.plus(a(), d()),
            Sum.plus(b(), c()),
            Sum.plus(b(), d()),
        ),
        Max.plus(
            Sum.plus(a(), c()),
            Sum.plus(a(), d()),
            Sum.plus(b(), c()),
            Sum.plus(b(), d()),
        ),
    )
    _check_pair(lhs=lhs, rhs=rhs, free_vars=[a, b, c, d])


@pytest.mark.parametrize("monoid", IDEMPOTENT)
def test_plus_idempotent_consecutive(monoid):
    """``a, a, b → a, b`` — only consecutive duplicates collapse."""
    a, b = define_vars("a", "b")
    lhs = monoid.plus(a(), a(), b())
    return _check_pair(lhs=lhs, rhs=monoid.plus(a(), b()), free_vars=[a, b])


@pytest.mark.parametrize("monoid", IDEMPOTENT)
def test_plus_idempotent_non_consecutive(monoid):
    """``a, b, a`` — Semilattice (Min/Max) collapses via commutative
    PlusDups; plain IdempotentMonoid leaves it as-is (consecutive-only)."""
    a, b = define_vars("a", "b")
    lhs = monoid.plus(a(), b(), a())
    if isinstance(monoid, Semilattice):
        rhs = monoid.plus(a(), b())
    else:
        rhs = monoid.plus(a(), b(), a())
    _check_pair(lhs=lhs, rhs=rhs, free_vars=[a, b])


def test_plus_commutative_idempotent_long():
    """Long alternation collapses via commutative dedup (Min/Max only)."""
    a, b = define_vars("a", "b")
    lhs = Min.plus(a(), b(), a(), b(), b(), a(), a())
    _check_pair(lhs=lhs, rhs=Min.plus(a(), b()), free_vars=[a, b])


@pytest.mark.parametrize("monoid", WITH_ZERO)
def test_plus_zero(monoid):
    a = define_vars("a")
    lhs_right = monoid.plus(a(), monoid.zero)
    lhs_left = monoid.plus(monoid.zero, a())
    _check_pair(lhs=lhs_right, rhs=monoid.zero, free_vars=[a])
    _check_pair(lhs=lhs_left, rhs=monoid.zero, free_vars=[a])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_body_sequence(monoid):
    x = Operation.define(int, name="x")
    X = Operation.define(list[int], name="X")

    @Operation.define
    def f(_x: int) -> int:
        raise NotHandled

    g = Operation.define(f, name="g")

    lhs = monoid.reduce([f(x()), g(x())], {x: X()})
    rhs = [monoid.reduce(f(x()), {x: X()}), monoid.reduce(g(x()), {x: X()})]

    _check_pair(lhs=lhs, rhs=rhs, free_vars=[X, f, g])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_body_sequence_2(monoid):
    x, y = define_vars("x", "y")
    X, Y = define_vars("X", "Y", typ=list[int])

    @Operation.define
    def f(_x: int) -> int:
        raise NotHandled

    g = Operation.define(f, name="g")

    lhs = monoid.reduce([f(x()), g(y())], {x: X(), y: Y()})
    rhs = [
        monoid.reduce(f(x()), {x: X(), y: Y()}),
        monoid.reduce(g(y()), {x: X(), y: Y()}),
    ]

    _check_pair(lhs=lhs, rhs=rhs, free_vars=[X, Y, f, g])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_body_mapping(monoid):
    x = Operation.define(int, name="x")
    X = Operation.define(list[int], name="X")

    @Operation.define
    def f(_x: int) -> int:
        raise NotHandled

    g = Operation.define(f, name="g")

    lhs = monoid.reduce({"a": f(x()), "b": g(x())}, {x: X()})
    rhs = {
        "a": monoid.reduce(f(x()), {x: X()}),
        "b": monoid.reduce(g(x()), {x: X()}),
    }
    _check_pair(lhs=lhs, rhs=rhs, free_vars=[X, f, g])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_no_streams(monoid):
    a = define_vars("a")
    lhs = monoid.reduce(a(), {})
    rhs = monoid.identity

    _check_pair(lhs=lhs, rhs=rhs, free_vars=[a])


@pytest.mark.parametrize("monoid", ALL_MONOIDS)
def test_reduce_reduce(monoid):
    a, b = define_vars("a", "b")
    A, B = define_vars("A", "B", typ=list[int])

    @Operation.define
    def f(_x: int, _y: int) -> int:
        raise NotHandled

    lhs = monoid.reduce(monoid.reduce(f(a(), b()), {a: A()}), {b: B()})
    rhs = monoid.reduce(f(a(), b()), {a: A(), b: B()})

    _check_pair(lhs=lhs, rhs=rhs, free_vars=[A, B, f])


@pytest.mark.parametrize("monoid", COMMUTATIVE)
def test_reduce_plus(monoid):
    a, b = define_vars("a", "b")
    A, B = define_vars("A", "B", typ=list[int])
    lhs = monoid.reduce(monoid.plus(a(), b()), {a: A(), b: B()})
    rhs = monoid.plus(
        monoid.reduce(a(), {a: A(), b: B()}),
        monoid.reduce(b(), {a: A(), b: B()}),
    )
    _check_pair(lhs=lhs, rhs=rhs, free_vars=[A, B])


def test_reduce_independent_1():
    a, b = define_vars("a", "b")
    A, B = define_vars("A", "B", typ=list[int])
    lhs = Sum.reduce(Product.plus(a(), b()), {a: A(), b: B()})
    rhs = Product.plus(Sum.reduce(a(), {a: A()}), Sum.reduce(b(), {b: B()}))
    _check_pair(lhs=lhs, rhs=rhs, free_vars=[A, B])


def test_reduce_independent_2():
    a, b, c = define_vars("a", "b", "c")
    A, B, C = define_vars("A", "B", "C", typ=list[int])

    @Operation.define
    def f(_x: int, _y: int) -> int:
        raise NotHandled

    lhs = Sum.reduce(Product.plus(a(), b(), f(b(), c())), {a: A(), b: B(), c: C()})
    rhs = Product.plus(
        Sum.reduce(a(), {a: A()}),
        Sum.reduce(Product.plus(b(), f(b(), c())), {b: B(), c: C()}),
    )
    _check_pair(lhs=lhs, rhs=rhs, free_vars=[A, B, C, f])


def test_reduce_independent_3_negative():
    """Stream `b` depends on `a` (b: g(a())), so the proposed factorization
    is unsound — the normalizer must NOT apply it."""
    a, b, c = define_vars("a", "b", "c")
    A, C = define_vars("A", "C", typ=list[int])

    @Operation.define
    def f(_x: int, _y: int) -> int:
        raise NotHandled

    @Operation.define
    def g(_x: int) -> list[int]:
        raise NotHandled

    with handler(NormalizeIntp):
        lhs = Sum.reduce(
            Product.plus(a(), b(), f(b(), c())), {a: A(), b: g(a()), c: C()}
        )
    bogus_rhs = Product.plus(
        Sum.reduce(a(), {a: A()}),
        Sum.reduce(Product.plus(b(), f(b(), c())), {b: g(a()), c: C()}),
    )
    assert fvsof(bogus_rhs) != fvsof(lhs)
    # Structural-only negative check: the normalizer correctly refused to apply
    # the bogus factorization.
    assert not syntactic_eq_alpha(lhs, bogus_rhs)


def test_reduce_independent_4():
    a, b, c = define_vars("a", "b", "c")
    A, B, C = define_vars("A", "B", "C", typ=list[int])

    @Operation.define
    def f(_x: int, _y: int) -> int:
        raise NotHandled

    lhs = Sum.reduce(Product.plus(a(), b(), f(b(), c()), 7), {a: A(), b: B(), c: C()})
    rhs = Product.plus(
        7,
        Sum.reduce(a(), {a: A()}),
        Sum.reduce(Product.plus(b(), f(b(), c())), {b: B(), c: C()}),
    )
    _check_pair(lhs=lhs, rhs=rhs, free_vars=[A, B, C, f])
