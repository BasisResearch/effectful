import collections.abc
import dataclasses
import functools
import itertools
import operator
from collections.abc import Iterable
from typing import Callable, Generic, ParamSpec, TypeVar

import effectful.handlers.numbers  # noqa: F401
import torch
import tree
from effectful.handlers.torch import Indexable, to_tensor
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
from effectful.ops.syntax import ObjectInterpretation, Scoped, defdata, deffn, defop, defterm, implements
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


# actually a near-semiring
StreamAlg: Semiring[collections.abc.Generator] = Semiring(
    add=lambda a, b: (v for v in itertools.chain(a, b)),
    mul=lambda a, b: ((v1, v2) for (v1, v2) in itertools.product(a, b)),
    zero=(),
    one=(),  # note: empty tuple is not a valid identity for multiplication
)

LinAlg: Semiring[float] = Semiring(operator.add, operator.mul, 0.0, 1.0)

MinAlg: Semiring[float] = Semiring(min, operator.mul, float("inf"), 1.0)

MaxAlg: Semiring[float] = Semiring(max, operator.mul, float("-inf"), 1.0)

Vec = T | tree.StructureKV[object, T] | collections.abc.Callable[..., T] | collections.abc.Generator[T, None, None]


# @defop
# def unfold(
#     streams: Runner[S],
#     body: T,
#     guard: bool | None = None,
# ) -> collections.abc.Iterable[T]:
#     if guard is not None:
#         return (b for (b, g) in unfold(streams, (body, guard)) if g)

#     if not streams:
#         return (b for b in (body,))

#     if isinstance(body, Operation) and body in streams:
#         return handler(streams)(body)

#     if isinstance(body, collections.abc.Callable):
#         return functools.wraps(body)(lambda *a, **k: unfold(streams, body(*a, **k)))

#     if isinstance(body, Term):
#         # select streams that are used the body of the term
#         used_streams = {op: streams[op] for op in streams}
#         streams = product(streams, used_streams)

#         def fold_body(ak):
#             return unfold(streams, body.op)(*ak[0], **ak[1])

#         unfolded_body = list(unfold(streams, (body.args, body.kwargs)))
#         folded_body = fold(StreamAlg, unfolded_body, fold_body)
#         return folded_body

#     if isinstance(body, collections.abc.Generator):
#         return fold(StreamAlg, (unfold(streams, b) for b in body))

#     if tree.is_nested(body) and any(isinstance(b, Term) for b in tree.flatten(body)):
#         if (
#             isinstance(body, tuple)
#             and len(body) == 2
#             and isinstance(body[0], tuple)
#             and len(body[0]) == 2
#         ):
#             breakpoint()
#         flat_body = tree.flatten(body)
#         unfolded_bodies = [list(unfold(streams, b)) for b in flat_body]
#         unflattened_result = [tree.unflatten_as(body, x) for x in zip(*unfolded_bodies)]
#         return unflattened_result

#     return (body for _ in itertools.product(streams.values()))


@defop
def unfold(streams: Runner, body: T) -> collections.abc.Iterable[T]:
    def generator():
        all_vals = itertools.product(*list(streams.values()))
        for vals in all_vals:
            keys = streams.keys()
            with handler({k: deffn(v) for (k, v) in zip(keys, vals)}):
                yield evaluate(body)

    return generator()


@defop
def fold(semiring: Semiring[T], streams: Runner, body: Vec[T], guard: bool = True) -> Vec[T]:
    def promote_add(add: Callable[[V, V], V], a: V, b: V) -> V:
        if isinstance(b, collections.abc.Generator) or isinstance(a, collections.abc.Generator):
            a = a if isinstance(a, collections.abc.Generator) else (a,)
            b = b if isinstance(b, collections.abc.Generator) else (b,)
            return (v for v in (*a, *b))
        elif isinstance(b, collections.abc.Mapping):
            result = {
                k: a[k] if k not in b else b[k] if k not in a else promote_add(add, a[k], b[k]) for k in set(a) | set(b)
            }
            return result
        elif isinstance(b, collections.abc.Callable):
            return lambda *args, **kwargs: promote_add(add, a(*args, **kwargs), b(*args, **kwargs))
        elif tree.is_nested(b):
            return tree.map_structure(functools.partial(promote_add, add), a, b)
        else:
            return add(a, b)

    def generator():
        all_vals = itertools.product(*list(streams.values()))
        for vals in all_vals:
            keys = streams.keys()
            with handler({k: deffn(v) for (k, v) in zip(keys, vals)}):
                if evaluate(guard):
                    yield evaluate(body)

    return functools.reduce(functools.partial(promote_add, semiring.add), generator())


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
                *tree.flatten(tree.map_structure(functools.partial(unfold_weighted, semiring, streams), body))
            )
        )
    else:
        return ((semiring.one, b) for b in (body if isinstance(body, collections.abc.Iterable) else (body,)))


@defop
def fold_weighted(
    semiring: Semiring[T],
    streams: Runner[S],
    body: T,
) -> T:
    return functools.reduce(lambda a, b: a + [b], unfold_weighted(semiring, streams, body), [])


def unfold_fn(intp: Runner[S], fn: Callable[P, T] | None = None):
    if fn is None:
        return functools.partial(unfold_fn, intp)

    def _trace_op(env, op, *args, **kwargs):
        val = fwd()
        var = defop(typeof(defdata(op, *args, **kwargs)))
        env[var] = deffn(val)
        return var()

    @functools.wraps(fn)
    def _wrapped(*args: P.args, **kwargs: P.kwargs) -> tuple[Interpretation, T]:
        env = {}

        with (
            handler(intp),
            handler({op: functools.wraps(op)(functools.partial(_trace_op, env, op)) for op in intp}),
        ):
            result = fn(*args, **kwargs)

        return env, result

    return _wrapped


@defop
def D(*args) -> dict:
    if any(len(fvsof(k)) > 0 for (k, _) in args):
        raise NotImplementedError
    return dict(args)


class DenseTensorFold(ObjectInterpretation):
    @implements(fold)
    def fold(self, semiring, streams, body, **kwargs):
        if not (
            semiring in (LinAlg, MinAlg, MaxAlg)
            and all(isinstance(s, collections.abc.Sized) for s in streams.values())
            and all(typeof(k()) is int for k in streams.keys())
        ):
            return fwd()

        if isinstance(body, Term):
            if not (body.op is D and all(isinstance(args, tuple) and len(args) == 2 for args in body.args)):
                return fwd()
            if len(body.args) <= 0:
                return torch.tensor([])
            if len(body.args) > 1:
                # todo: handle multiple output indices
                return fwd()
            indices, value = body.args[0]
        elif isinstance(body, dict):
            if len(body) <= 0:
                return torch.tensor([])
            if len(body) > 1:
                return fwd()
            indices, value = next(iter(body.items()))
        else:
            return fwd()

        if not isinstance(indices, tuple):
            indices = (indices,)

        # Check that the output is indexed in a subset of the input indices, and
        # that there are no index transformations
        if not all(isinstance(i, Term) and i.op in streams for i in indices):
            return fwd()
        indices = [i.op for i in indices]

        fresh_indices = {k: defop(k) for k in streams.keys()}
        indexed_streams = {k: deffn(Indexable(torch.tensor(v))[fresh_indices[k]()]) for k, v in streams.items()}
        with handler(indexed_streams):
            result = evaluate(value)

        reduction_indices = [fresh_indices[i] for i in streams.keys() if i not in indices]
        result = to_tensor(result, reduction_indices)

        reductor = None
        if semiring is LinAlg:
            reductor = torch.sum
        elif semiring is MinAlg:
            reductor = lambda *args, **kwargs: torch.min(*args, **kwargs).values
        elif semiring is MaxAlg:
            reductor = lambda *args, **kwargs: torch.max(*args, **kwargs).values
        else:
            assert False, f"unexpected semiring: {semiring}"

        for _ in range(len(reduction_indices)):
            result = reductor(result, dim=0)

        return to_tensor(result, [fresh_indices[i] for i in indices])


@defop
def reals() -> Iterable[float]:
    raise NotImplementedError


class GradientOptimizationFold(ObjectInterpretation):
    def __init__(self, optimizer=torch.optim.Adam, steps=100, **kwargs):
        self.optimizer = optimizer
        self.optimizer_kwargs = kwargs
        self.steps = steps

    def _optimizer(self, params):
        return self.optimizer(params, self.optimizer_kwargs)

    @implements(fold)
    def fold(self, semiring, streams, body, **kwargs):
        if not (semiring is MinAlg and all(isinstance(v, Term) and v.op is reals for v in streams.values())):
            return fwd()

        if isinstance(body, Term):
            if not (body.op is D and all(isinstance(args, tuple) and len(args) == 2 for args in body.args)):
                return fwd()
            if len(body.args) <= 0:
                return torch.tensor([])
            if len(body.args) > 1:
                # todo: handle multiple output indices
                return fwd()
            indices, value = body.args[0]
        elif isinstance(body, dict):
            if len(body) <= 0:
                return torch.tensor([])
            if len(body) > 1:
                return fwd()
            indices, value = next(iter(body.items()))
        else:
            return fwd()

        if not isinstance(indices, tuple):
            indices = (indices,)


if __name__ == "__main__":
    x, y = defop(int), defop(int)

    print(list(unfold({x: lambda: range(2), y: lambda: range(2)}, x() + y())))

    print(fold(LinAlg, unfold({x: lambda: range(3), y: lambda: range(3)}, x() * y() ** 2)))

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

    print(list(sums(((i, i / k) for i in range(k) if i % 2 == 1) for k in range(1, 10))))
