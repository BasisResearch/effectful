import functools
import itertools
from collections.abc import Callable

from effectful.internals.runtime import interpreter
from effectful.ops.semantics import apply, evaluate
from effectful.ops.syntax import _BaseTerm, defdata, syntactic_eq
from effectful.ops.types import NotHandled, Operation
from effectful.ops.weighted.monoid import IdempotentMonoid, Max, Min, Product, Sum


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


def test_plus_single():
    x = define_vars("x")
    assert syntactic_eq_alpha(Sum.plus(x()), x())


def test_plus_identity():
    x = define_vars("x")
    assert syntactic_eq_alpha(Sum.plus(x(), Sum.identity), x())
    assert syntactic_eq_alpha(Sum.plus(Sum.identity, x()), x())


def test_plus_plus():
    (x, y, z) = define_vars("x", "y", "z")
    assert syntactic_eq_alpha(
        Sum.plus(x(), Sum.plus(y(), z())), Sum.plus(x(), y(), z())
    )
    assert syntactic_eq_alpha(
        Sum.plus(Sum.plus(x(), y()), z()), Sum.plus(x(), y(), z())
    )


def test_plus_sequence():
    (a, b, c, d) = define_vars("a", "b", "c", "d")
    assert syntactic_eq_alpha(
        Sum.plus([a(), b()], [c(), d()]), [Sum.plus(a(), c()), Sum.plus(b(), d())]
    )


def test_plus_mapping():
    (a, b, c, d) = define_vars("a", "b", "c", "d")
    assert syntactic_eq_alpha(
        Sum.plus({"x": a(), "y": b()}, {"x": c(), "z": d()}),
        {"x": Sum.plus(a(), c()), "y": b(), "z": d()},
    )


def test_plus_distributes():
    (a, b, c, d) = define_vars("a", "b", "c", "d")
    assert syntactic_eq_alpha(
        Product.plus(Sum.plus(a(), b()), Sum.plus(c(), d())),
        Sum.plus(
            Product.plus(a(), c()),
            Product.plus(a(), d()),
            Product.plus(b(), c()),
            Product.plus(b(), d()),
        ),
    )


def test_plus_distributes_multiple():
    (a, b, c, d) = define_vars("a", "b", "c", "d")
    assert syntactic_eq_alpha(
        Sum.plus(
            Min.plus(a(), b()),
            Min.plus(c(), d()),
            Max.plus(a(), b()),
            Max.plus(c(), d()),
        ),
        Sum.plus(
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
        ),
    )


def test_plus_idempotent():
    (a, b, identity) = define_vars("a", "b", "identity")

    IdMonoid = IdempotentMonoid(
        kernel=Operation.define(Callable[[int, int], int]), identity=identity()
    )

    assert syntactic_eq_alpha(IdMonoid.plus(a(), a(), b()), IdMonoid.plus(a(), b()))
    assert syntactic_eq_alpha(
        IdMonoid.plus(a(), b(), a()), IdMonoid.plus(a(), b(), a())
    )
    assert syntactic_eq_alpha(
        IdMonoid.plus(a(), b(), a(), b(), b(), a(), a()),
        IdMonoid.plus(a(), b(), a(), b(), a()),
    )


def test_plus_commutative_idempotent():
    (a, b) = define_vars("a", "b")

    assert syntactic_eq_alpha(Min.plus(a(), a(), b()), Min.plus(a(), b()))
    assert syntactic_eq_alpha(Min.plus(b(), a(), b()), Min.plus(b(), a()))
    assert syntactic_eq_alpha(
        Min.plus(a(), b(), a(), b(), b(), a(), a()), Min.plus(a(), b())
    )


def test_plus_zero():
    a = define_vars("a")
    assert syntactic_eq_alpha(Product.plus(a(), Product.zero), Product.zero)
    assert syntactic_eq_alpha(Product.plus(Product.zero, a()), Product.zero)


def test_reduce_body_sequence():
    x = Operation.define(int)
    X = Operation.define(list[int])

    @Operation.define
    def f(x: int) -> int:
        raise NotHandled

    g = Operation.define(f, name="g")

    assert syntactic_eq_alpha(
        Sum.reduce([f(x()), g(x())], {x: X()}),
        [Sum.reduce(f(x()), {x: X()}), Sum.reduce(g(x()), {x: X()})],
    )


def test_reduce_body_sequence_2():
    x, y = define_vars("x", "y")
    X, Y = define_vars("X", "Y", typ=list[int])

    @Operation.define
    def f(x: int) -> int:
        raise NotHandled

    g = Operation.define(f, name="g")

    assert syntactic_eq_alpha(
        Sum.reduce([f(x()), g(y())], {x: X(), y: Y()}),
        [Sum.reduce(f(x()), {x: X(), y: Y()}), Sum.reduce(g(y()), {x: X(), y: Y()})],
    )


def test_reduce_body_mapping():
    x = Operation.define(int)
    X = Operation.define(list[int])

    @Operation.define
    def f(x: int) -> int:
        raise NotHandled

    g = Operation.define(f, name="g")

    assert syntactic_eq_alpha(
        Sum.reduce({"a": f(x()), "b": g(x())}, {x: X()}),
        {"a": Sum.reduce(f(x()), {x: X()}), "b": Sum.reduce(g(x()), {x: X()})},
    )


def test_reduce_no_streams():
    a = define_vars("a")
    assert syntactic_eq_alpha(Sum.reduce(a(), {}), Sum.identity)


def test_reduce_empty():
    a, b, c = define_vars("a", "b", "c")
    A = define_vars("A", typ=list[int])

    @Operation.define
    def C(x: int) -> list[int]:
        raise NotHandled

    assert syntactic_eq_alpha(Sum.reduce(c(), {a: A(), b: [], c: C(a())}), Sum.identity)


def test_reduce_plus():
    a, b = define_vars("a", "b")
    A, B = define_vars("A", "B", typ=list[int])
    assert syntactic_eq_alpha(
        Sum.reduce(Sum.plus(a(), b()), {a: A(), b: B()}),
        Sum.plus(Sum.reduce(a(), {a: A(), b: B()}), Sum.reduce(b(), {a: A(), b: B()})),
    )


def test_reduce_reduce():
    a, b = define_vars("a", "b")
    A, B = define_vars("A", "B", typ=list[int])

    @Operation.define
    def f(x: int, y: int) -> int:
        raise NotHandled

    assert syntactic_eq_alpha(
        Sum.reduce(Sum.reduce(f(a(), b()), {a: A()}), {b: B()}),
        Sum.reduce(f(a(), b()), {a: A(), b: B()}),
    )


def test_reduce_idempotent_unused_1():
    a, b = define_vars("a", "b")
    A = Operation.define(list[int])
    assert syntactic_eq_alpha(Min.reduce(b(), {a: A()}), b())


def test_reduce_idempotent_unused_2():
    a, b, c = define_vars("a", "b", "c")
    C = define_vars("C", typ=list[int])

    @Operation.define
    def f(x: int) -> int:
        raise NotHandled

    assert syntactic_eq_alpha(
        Min.reduce(b(), {a: f(b()), b: f(c()), c: C()}),
        Min.reduce(b(), {b: f(c()), c: C()}),
    )
