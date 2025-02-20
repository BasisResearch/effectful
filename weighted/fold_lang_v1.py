import collections.abc
import dataclasses
import functools
import itertools
import operator
from typing import (
    Annotated,
    Callable,
    Concatenate,
    Generic,
    Literal,
    ParamSpec,
    TypeVar,
)

import effectful.handlers.numbers  # noqa: F401
import tree
from effectful.ops.semantics import (
    apply,
    call,
    coproduct,
    evaluate,
    fvsof,
    fwd,
    handler,
    product,
    typeof,
)
from effectful.ops.syntax import Scoped, defdata, deffn, defop, defterm
from effectful.ops.types import Interpretation, Operation, Term

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
A = TypeVar("A")
B = TypeVar("B")


# Expectation(
#     f(x)
#     for z1 in sample(z1_dist)
#     for z2 in sample(z2_dist(z1))
#     for x in sample(x_dist(z1, z2))
# )
#
#
# # unnormalized
# Expectation(
#     weight * vars[-1]
#     for (weight, vars) in Infer(
#         (w1(z1) * w2(z1, z2) * w3(z1, z2, x), (z1, z2, x))
#         for z1 in sample(z1_dist)
#         # if factor(w1(z1)) != 0
#         for z2 in sample(z2_dist(z1))
#         # if factor(w2(z1, z2)) != 0
#         for x in sample(x_dist(z1, z2))
#         # if factor(w3(z1, z2, x)) != 0
#     )
# )


@dataclasses.dataclass
class Semiring(Generic[T]):
    add: Callable[[T, T], T]
    mul: Callable[[T, T], T]
    zero: T
    one: T


Runner = Interpretation[T, collections.abc.Iterable[T]]


StreamAlg: Semiring[collections.abc.Generator] = Semiring(
    add=lambda a, b: (v for v in itertools.chain(a, b)),
    mul=lambda a, b: ((v1, v2) for (v1, v2) in itertools.product(a, b)),
    zero=(),
    one=(),
)

LinAlg: Semiring[float] = Semiring(operator.add, operator.mul, 0.0, 1.0)

DenseLinAlg: Semiring[dict[tuple[int, ...], float]] = None


Vec = (
    T
    | tree.StructureKV[object, T]
    | collections.abc.Callable[..., T]
    | collections.abc.Generator[T, None, None]
)


@defop
def unfold(
    streams: Runner[S],
    body: T,
    guard: bool | None = None,
) -> collections.abc.Iterable[T]:
    if guard is not None:
        return (b for (b, g) in unfold(streams, (body, guard)) if g)

    if not streams:
        return (b for b in (body,))

    if isinstance(body, Operation) and body in streams:
        return handler(streams)(body)
    elif isinstance(body, collections.abc.Callable):
        return functools.wraps(body)(lambda *a, **k: unfold(streams, body(*a, **k)))
    elif isinstance(body, Term):
        streams = product(
            streams, {op: streams[op] for op in fvsof(body) if op in streams}
        )
        return fold(
            StreamAlg,
            unfold(streams, (body.args, body.kwargs)),
            lambda ak: unfold(streams, body.op)(*ak[0], **ak[1]),
        )
    elif isinstance(body, collections.abc.Generator):
        return fold(StreamAlg, (unfold(streams, b) for b in body))
    elif tree.is_nested(body) and any(isinstance(b, Term) for b in tree.flatten(body)):
        return (
            tree.unflatten_as(body, it)
            for it in itertools.product(
                *map(functools.partial(unfold, streams), tree.flatten(body))
            )
        )
    else:
        return unfold({}, body)


@defop
def fold(
    semiring: Semiring[T],
    stream: collections.abc.Iterable[S],
    body: Callable[[S], Vec[T]] = lambda s: s,
    guard: Callable[[S], bool] = lambda _: True,
) -> Vec[T]:
    def promote_add(add: Callable[[V, V], V], a: V, b: V) -> V:
        if isinstance(b, collections.abc.Generator) or isinstance(
            a, collections.abc.Generator
        ):
            a = a if isinstance(a, collections.abc.Generator) else (a,)
            b = b if isinstance(b, collections.abc.Generator) else (b,)
            return (v for v in (*a, *b))
        elif isinstance(b, collections.abc.Mapping):
            result = {
                k: a[k]
                if k not in b
                else b[k]
                if k not in a
                else promote_add(add, a[k], b[k])
                for k in set(a) | set(b)
            }
            return result
        elif isinstance(b, collections.abc.Callable):
            return lambda *args, **kwargs: promote_add(
                add, a(*args, **kwargs), b(*args, **kwargs)
            )
        elif tree.is_nested(b):
            return tree.map_structure(functools.partial(promote_add, add), a, b)
        else:
            return add(a, b)

    return functools.reduce(
        functools.partial(promote_add, semiring.add),
        (body(s) for s in stream if guard(s)),
    )


@defop
def unfold_weighted(
    semiring: Semiring[V],
    streams: Runner[S],
    body: T,
) -> collections.abc.Iterable[tuple[V, T]]:
    if isinstance(body, Term):
        args_kwargs = unfold_weighted(semiring, streams, (body.args, body.kwargs))
        if body.op in streams:
            return (
                (semiring.mul(w_args, w), v)
                for (w_args, (a, k)) in args_kwargs
                for (w, v) in handler(streams)(body.op)(*a, **k)
            )
        else:
            # TODO track weight through function body
            return ((w_args, body.op(*a, **k)) for (w_args, (a, k)) in args_kwargs)
    elif tree.is_nested(body):
        return (
            (
                functools.reduce(semiring.mul, (w for (w, _) in it), semiring.one),
                tree.unflatten_as(body, [v for (_, v) in it]),
            )
            for it in itertools.product(
                *tree.flatten(
                    tree.map_structure(
                        functools.partial(unfold_weighted, semiring, streams), body
                    )
                )
            )
        )
    else:
        return (
            (semiring.one, b)
            for b in (body if isinstance(body, collections.abc.Iterable) else (body,))
        )


@defop
def fold_weighted(
    semiring: Semiring[T],
    streams: Runner[S],
    body: T,
) -> T:
    return functools.reduce(
        lambda a, b: a + [b], unfold_weighted(semiring, streams, body), []
    )


if __name__ == "__main__":
    x, y = defop(int), defop(int)

    print(list(unfold({x: lambda: range(2), y: lambda: range(2)}, x() + y())))

    print(
        fold(LinAlg, unfold({x: lambda: range(3), y: lambda: range(3)}, x() * y() ** 2))
    )

    print(
        fold_weighted(
            LinAlg,
            {
                x: lambda: [(i**2, i) for i in range(3)],
                y: lambda: [(i**2, i) for i in range(3)],
            },
            x() * y() ** 2,
        )
    )

    print(fold(LinAlg, [1, 2, 3], lambda x: x**2, lambda x: x > 0))

    sums = functools.partial(fold, LinAlg)

    print(
        list(sums(((i, i / k) for i in range(k) if i % 2 == 1) for k in range(1, 10)))
    )
