import collections.abc
import dataclasses
import functools
import itertools
import operator
from collections.abc import Iterable
from typing import Any, Callable, Generic, Mapping, ParamSpec, TypeAlias, TypeVar, cast

import effectful.handlers.numbers  # noqa: F401
import torch
import tree
from effectful.handlers.indexed import sizesof
from effectful.handlers.torch import to_tensor
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


Runner: TypeAlias = Interpretation[T, collections.abc.Iterable[T]]


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


def arg_min(a, b):
    return a if a[0] < b[0] else b


def arg_max(a, b):
    return a if a[0] > b[0] else b


ArgMinAlg: Semiring[tuple[float, Any]] = Semiring(arg_min, operator.mul, (float("inf"), None), (1.0, None))

ArgMaxAlg: Semiring[tuple[float, Any]] = Semiring(arg_max, operator.mul, (float("-inf"), None), (1.0, None))


@defop
def semi_ring_product(*args: Semiring[Any]) -> Semiring[tuple]:
    flat_args = []
    for semiring in args:
        if isinstance(semiring, Term) and semiring.op is semi_ring_product:
            flat_args.extend(semiring.args)
    return defdata(semi_ring_product, *flat_args)


def semi_ring_product_value(*args: Semiring[Any]) -> Semiring[tuple]:
    return Semiring(
        add=lambda a, b: tuple(semiring.add(a[i], b[i]) for i, semiring in enumerate(args)),
        mul=lambda a, b: tuple(semiring.mul(a[i], b[i]) for i, semiring in enumerate(args)),
        zero=tuple(semiring.zero for semiring in args),
        one=tuple(semiring.one for semiring in args),
    )


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
def fold(semiring: Semiring[T], streams: Runner, body: Mapping[K, T], guard: bool = True) -> Mapping[K, T]:
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
        else:
            return add(a, b)

    def generator() -> collections.abc.Iterable[Mapping[K, T]]:
        all_vals = itertools.product(*list(streams.values()))
        for vals in all_vals:
            keys = streams.keys()
            with handler({k: deffn(v) for (k, v) in zip(keys, vals)}):
                if evaluate(guard):
                    with handler({D: lambda *args: dict(args)}):
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
    if not all(isinstance(kv, tuple) and len(kv) == 2 for kv in args):
        raise ValueError("Expected a sequence of key-value pairs")
    raise NotImplementedError


class NormalizeValueFold(ObjectInterpretation):
    """Normalization rule for the body of folds."""

    @implements(fold)
    def fold(self, semiring, streams, body, **kwargs):
        modified_body = False
        if isinstance(body, Term) and body.op is D:
            kvs = []
            for k, v in body.args:
                if not isinstance(k, tuple):
                    k = (k,)
                    modified_body = True
                kvs.append((k, v))
            body = D(*kvs)
        elif isinstance(body, dict):
            body = D(*body.items())
            modified_body = True
        else:
            body = D(((), body))
            modified_body = True

        if modified_body:
            return fold(semiring, streams, body)
        return fwd()


class ProductFold(ObjectInterpretation):
    """Handles products of semirings."""

    @implements(fold)
    def fold(self, semiring, streams, body, **kwargs):
        if not (isinstance(semiring, Term) and semiring.op is semi_ring_product):
            return fwd()

        semi_rings = semiring.args
        return tree.map_structure(lambda s: fold(s, streams, body), semi_rings)


class DenseTensorArgFold(ObjectInterpretation):
    @implements(fold)
    def fold(self, semiring, streams, body, **kwargs):
        if not (
            semiring in (ArgMinAlg, ArgMaxAlg)
            and all(isinstance(s, collections.abc.Sized) for s in streams.values())
            and all(typeof(k()) is torch.Tensor for k in streams.keys())
        ):
            return fwd()

        if not (isinstance(body, Term) and body.op is D):
            return fwd()

        if len(body.args) <= 0:
            return torch.tensor([])

        if len(body.args) > 1:
            # todo: handle multiple output indices
            return fwd()

        indices, value = body.args[0]
        if not isinstance(value, tuple) and len(value) == 2:
            raise ValueError("Expected a tuple of (value, arg) for ArgMinAlg")
        min_value, argmin_value = value

        # Check that the output is indexed in a subset of the input indices, and
        # that there are no index transformations
        if not all(isinstance(i, Term) and i.op in streams for i in indices):
            return fwd()
        indices = [i.op for i in indices]

        old_to_fresh = {k: defop(k) for k in streams.keys()}
        fresh_to_old = {v: k for (k, v) in old_to_fresh.items()}
        indexed_streams = {k: deffn(v[old_to_fresh[k]()]) for k, v in streams.items()}
        with handler(indexed_streams):
            result = evaluate(min_value)

        result_indices = sizesof(result)
        reduction_indices = [i for i in result_indices if fresh_to_old[i] not in indices]

        result = to_tensor(result, reduction_indices)

        flat_result = result.flatten(start_dim=0, end_dim=len(reduction_indices) - 1)
        mins, flat_indices = flat_result.min(dim=0) if semiring is ArgMinAlg else result.max(dim=0)
        min_indices = torch.unravel_index(flat_indices, result.shape)
        with handler(
            {fresh_to_old[k]: deffn(streams[fresh_to_old[k]][v]) for k, v in zip(reduction_indices, min_indices)}
        ):
            argmins = evaluate(argmin_value)

        final_result = tree.map_structure(
            lambda t: to_tensor(t, [i for i in result_indices if i not in reduction_indices]), (mins, argmins)
        )
        return final_result


class DenseTensorFold(ObjectInterpretation):
    def _sum_reductor(self, tensor, dims):
        for _ in range(dims):
            tensor = torch.sum(tensor, dim=0)
        return tensor

    def _min_reductor(self, tensor, dims):
        for _ in range(dims):
            tensor = torch.min(tensor, dim=0).values
        return tensor

    def _max_reductor(self, tensor, dims):
        for _ in range(dims):
            tensor = torch.max(tensor, dim=0).values
        return tensor

    def _get_reductor(self, semi_ring):
        if semi_ring is LinAlg:
            return self._sum_reductor
        elif semi_ring is MinAlg:
            return self._min_reductor
        elif semi_ring is MaxAlg:
            return self._max_reductor
        else:
            return None

    @implements(fold)
    def fold(self, semiring, streams, body, **kwargs):
        reductor = self._get_reductor(semiring)
        if reductor is None or not (
            all(isinstance(s, collections.abc.Sized) for s in streams.values())
            and all(typeof(k()) is torch.Tensor for k in streams.keys())
        ):
            return fwd()

        if not (isinstance(body, Term) and body.op is D):
            return fwd()

        if len(body.args) <= 0:
            return torch.tensor([])

        if len(body.args) > 1:
            # todo: handle multiple output indices
            return fwd()

        indices, value = body.args[0]

        # Check that the output is indexed in a subset of the input indices, and
        # that there are no index transformations
        if not all(isinstance(i, Term) and i.op in streams for i in indices):
            return fwd()
        indices = [i.op for i in indices]

        old_to_fresh = {k: defop(k) for k in streams.keys()}
        fresh_to_old = {v: k for (k, v) in old_to_fresh.items()}
        indexed_streams = {k: deffn(v[old_to_fresh[k]()]) for k, v in streams.items()}
        with handler(indexed_streams):
            result = evaluate(value)

        result_indices = sizesof(result)
        reduction_indices = [i for i in result_indices if fresh_to_old[i] not in indices]

        result = to_tensor(result, reduction_indices)
        result = reductor(result, len(reduction_indices))
        return to_tensor(result, [i for i in result_indices if i not in reduction_indices])


@defop
def reals() -> Iterable[float]:
    raise NotImplementedError


class GradientOptimizationFold(ObjectInterpretation):
    def __init__(self, optimizer=torch.optim.Adam, steps=1000, **kwargs):
        self.optimizer = optimizer
        self.optimizer_kwargs = kwargs
        self.steps = steps

    def _optimizer(self, params):
        return self.optimizer(params, **self.optimizer_kwargs)

    @implements(fold)
    def fold(self, semiring, streams, body, **kwargs):
        breakpoint()
        if not (
            semiring in (MinAlg, ArgMinAlg) and all(isinstance(v, Term) and v.op is reals for v in streams.values())
        ):
            return fwd()

        if not (isinstance(body, Term) and body.op is D):
            return fwd()

        if len(body.args) <= 0:
            return torch.tensor([])

        if len(body.args) > 1:
            # todo: handle multiple output indices
            return fwd()

        indices, value = body.args[0]
        if indices != ():
            return fwd()

        if semiring is ArgMinAlg:
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError("Expected a tuple of (value, arg) for ArgMinAlg")
            value, arg = value

        params = [torch.tensor(0.0, requires_grad=True) for _ in streams.values()]
        param_ctx = {v: deffn(p) for (v, p) in zip(streams.keys(), params)}

        optimizer = self._optimizer(params)
        assert self.steps > 0
        for _ in range(self.steps):
            optimizer.zero_grad()
            with handler(param_ctx):
                loss = evaluate(value)
            tree.map_structure(lambda p: p.backward(), loss)
            optimizer.step()

        loss = tree.map_structure(lambda p: p.detach(), loss)
        if semiring is MinAlg:
            return loss

        with handler(param_ctx):
            arg = evaluate(arg)
        return loss, arg


dense_fold_intp = functools.reduce(coproduct, [NormalizeValueFold(), DenseTensorArgFold(), DenseTensorFold()])
