import collections.abc
import dataclasses
import functools
import itertools
import logging
import operator
from collections.abc import Iterable
from typing import Any, Callable, Generic, Mapping, ParamSpec, TypeAlias, TypeVar

import effectful.handlers.jax.numpy as jnp
import effectful.handlers.numpyro as dist
import jax
import numpyro
import optax
import tree
from effectful.handlers.jax import bind_dims, jax_getitem, sizesof
from effectful.handlers.jax._handlers import _register_jax_op
from effectful.handlers.numbers import _wrap_binop
from effectful.ops.semantics import (
    coproduct,
    evaluate,
    fvsof,
    fwd,
    handler,
    typeof,
)
from effectful.ops.syntax import ObjectInterpretation, defdata, deffn, defop, implements
from effectful.ops.types import Interpretation, Operation, Term

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
A = TypeVar("A")
B = TypeVar("B")

logger = logging.getLogger(__name__)

min = _wrap_binop(min)
max = _wrap_binop(max)

Runner: TypeAlias = Interpretation[T, collections.abc.Iterable[T]]


@dataclasses.dataclass
class Semiring(Generic[T]):
    add: Callable[[T, T], T]
    mul: Callable[[T, T], T]
    zero: T
    one: T
    name: str | None = None

    def __init__(self, add, mul, zero, one, name=None):
        self.add = add
        self.mul = mul
        self.zero = zero
        self.one = one
        self.name = name

    def __str__(self):
        if self.name is None:
            return repr(self)
        return self.name


# Semiring laws:
# (R, +) is a commutative monoid with identity 0
# (R, *) is a monoid with identity 1
# a * 0 = 0 = 0 * a
# a * (b + c) = (a * b) + (a * c)
# (b + c) * a = (b * a) + (c * a)


@defop
def add(a, b):
    if a == 0:
        return b
    if b == 0:
        return a
    if any(isinstance(x, Term) for x in (a, b)):
        raise NotImplementedError
    return a + b


@defop
def mul(a, b):
    if a == 1:
        return b
    if b == 1:
        return a
    if any(isinstance(x, Term) for x in (a, b)):
        raise NotImplementedError
    return a * b


@defop
def arg_min(a, b):
    if isinstance(a, tuple) and a[0] is float("inf"):
        return b
    if isinstance(b, tuple) and b[0] is float("inf"):
        return a
    if any(isinstance(x, Term) for x in tree.flatten((a, b))):
        raise NotImplementedError
    return a if a[0] < b[0] else b


@defop
def arg_max(a, b):
    if isinstance(a, tuple) and a[0] is float("-inf"):
        return b
    if isinstance(b, tuple) and b[0] is float("-inf"):
        return a
    if any(isinstance(x, Term) for x in tree.flatten((a, b))):
        raise NotImplementedError
    return a if a[0] > b[0] else b


# actually a near-semiring
StreamAlg: Semiring[collections.abc.Generator] = Semiring(
    add=lambda a, b: (v for v in itertools.chain(a, b)),
    mul=lambda a, b: ((v1, v2) for (v1, v2) in itertools.product(a, b)),
    zero=(),
    one=(),  # note: empty tuple is not a valid identity for multiplication
    name="StreamAlg",
)

LinAlg: Semiring[float] = Semiring(add, mul, 0.0, 1.0, "LinAlg")

MinAlg: Semiring[float] = Semiring(min, mul, float("inf"), 1.0, "MinAlg")

MaxAlg: Semiring[float] = Semiring(max, mul, float("-inf"), 1.0, "MaxAlg")


ArgMinAlg: Semiring[tuple[float, Any]] = Semiring(
    arg_min, mul, (float("inf"), None), (1.0, None), "ArgMinAlg"
)

ArgMaxAlg: Semiring[tuple[float, Any]] = Semiring(
    arg_max, mul, (float("-inf"), None), (1.0, None), "ArgMaxAlg"
)


@defop
def semi_ring_product(*args: Semiring[Any]) -> Semiring[tuple]:
    flat_args = []
    for semiring in args:
        if isinstance(semiring, Term) and semiring.op is semi_ring_product:
            flat_args.extend(semiring.args)
        else:
            flat_args.append(semiring)
    return defdata(semi_ring_product, *flat_args)


def semi_ring_product_value(*args: Semiring[Any]) -> Semiring[tuple]:
    return Semiring(
        add=lambda a, b: tuple(
            semiring.add(a[i], b[i]) for i, semiring in enumerate(args)
        ),
        mul=lambda a, b: tuple(
            semiring.mul(a[i], b[i]) for i, semiring in enumerate(args)
        ),
        zero=tuple(semiring.zero for semiring in args),
        one=tuple(semiring.one for semiring in args),
    )


Vec = (
    T
    | tree.StructureKV[object, T]
    | collections.abc.Callable[..., T]
    | collections.abc.Generator[T, None, None]
)


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


def fold_spec(
    semiring: Semiring[T], streams: Runner, body: Mapping[K, T]
) -> Mapping[K, T]:
    if any(isinstance(v, Term) for v in streams.values()):
        raise NotImplementedError

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
        else:
            return add(a, b)

    def generator() -> collections.abc.Iterable[Mapping[K, T]]:
        all_vals = itertools.product(*list(streams.values()))
        for vals in all_vals:
            keys = streams.keys()
            with handler({k: deffn(v) for (k, v) in zip(keys, vals)}):
                with handler({D: lambda *args: dict(args)}):
                    yield evaluate(body)

    return functools.reduce(functools.partial(promote_add, semiring.add), generator())


@defop
def fold(semiring: Semiring[T], streams: Runner, body: Mapping[K, T]) -> Mapping[K, T]:
    return fold_spec(semiring, streams, body)


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
            handler(
                {
                    op: functools.wraps(op)(functools.partial(_trace_op, env, op))
                    for op in intp
                }
            ),
        ):
            result = fn(*args, **kwargs)

        return env, result

    return _wrapped


@defop
def D(*args) -> dict:
    if not all(isinstance(kv, tuple) and len(kv) == 2 for kv in args):
        raise ValueError("Expected a sequence of key-value pairs")
    raise NotImplementedError


@defop
def sample(key: jax.Array, d: dist.Distribution, sample_shape: tuple[int]) -> jax.Array:
    if not (
        isinstance(d, numpyro.distributions.Distribution)
        and (not isinstance(d, Term) or all(isinstance(a, jax.Array) for a in d.args))
    ):
        raise NotImplementedError
    return d.sample(key, sample_shape=sample_shape)


@defop
def rsample(key, d: dist.Distribution, sample_shape: tuple[int]) -> jax.Array:
    if not (
        isinstance(d, numpyro.distributions.Distribution)
        and isinstance(sample_shape, tuple)
    ):
        raise NotImplementedError
    return d.rsample(key, sample_shape=sample_shape)


@defop
def log_prob(d: dist.Distribution, value: jax.Array) -> jax.Array:  # todo
    if (
        isinstance(d, numpyro.distributions.Distribution) and not isinstance(d, Term)
    ) and isinstance(value, Term):
        return _register_jax_op(d.log_prob)(value)
    if not (
        (
            isinstance(d, numpyro.distributions.Distribution)
            and isinstance(value, jax.Array)
        )
        or (isinstance(d, numpyro.distributions.Distribution) and isinstance(d, Term))
    ):
        raise NotImplementedError

    return d.log_prob(value)


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
            new_body = D(*kvs)
        elif isinstance(body, dict):
            new_body = D(*body.items())
            modified_body = True
        else:
            new_body = D(((), body))
            modified_body = True

        if modified_body:
            print(str(body), str(new_body))
            return fold(semiring, streams, new_body)
        return fwd()


class ProductFold(ObjectInterpretation):
    """Handles products of semirings."""

    @implements(fold)
    def fold(self, semiring, streams, body, **kwargs):
        if not (isinstance(semiring, Term) and semiring.op is semi_ring_product):
            return fwd()

        semi_rings = semiring.args
        if not (isinstance(body, tuple) and len(body) == len(semi_rings)):
            raise ValueError(
                "Expected a tuple of the same length as the product of semirings"
            )

        return tree.map_structure(lambda r, b: fold(r, streams, b), semi_rings, body)


class DenseTensorArgFold(ObjectInterpretation):
    @implements(fold)
    def fold(self, semiring, streams, body, **kwargs):
        if not (
            semiring in (ArgMinAlg, ArgMaxAlg)
            and all(isinstance(s, collections.abc.Sized) for s in streams.values())
            and all(typeof(k()) is jax.Array for k in streams.keys())
        ):
            return fwd()

        if not (isinstance(body, Term) and body.op is D):
            return fwd()

        if len(body.args) <= 0:
            return jnp.array([])

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
        indexed_streams = {
            k: deffn(jax_getitem(v, [old_to_fresh[k]()])) for k, v in streams.items()
        }
        with handler(indexed_streams):
            result = evaluate(min_value)

        result_indices = sizesof(result)
        reduction_indices = [i for i in result_indices if fresh_to_old[i] not in indices]

        result = bind_dims(result, *reduction_indices)

        # Flatten the leading len(reduction_indices) dimensions
        reduction_shape = result.shape[: len(reduction_indices)]
        flat_shape = (functools.reduce(operator.mul, reduction_shape, 1),) + result.shape[
            len(reduction_indices) :
        ]
        flat_result = jnp.reshape(result, flat_shape)

        mins = (
            jnp.min(flat_result, axis=0)
            if semiring is ArgMinAlg
            else jnp.max(flat_result, axis=0)
        )
        flat_indices = (
            jnp.argmin(flat_result, axis=0)
            if semiring is ArgMinAlg
            else jnp.argmax(flat_result, axis=0)
        )
        min_indices = jnp.unravel_index(flat_indices, reduction_shape)
        with handler(
            {
                fresh_to_old[k]: deffn(streams[fresh_to_old[k]][v])
                for k, v in zip(reduction_indices, min_indices)
            }
        ):
            argmins = evaluate(argmin_value)

        final_result = tree.map_structure(
            lambda t: bind_dims(
                t, *[i for i in result_indices if i not in reduction_indices]
            ),
            (mins, argmins),
        )
        return final_result


class DenseTensorFold(ObjectInterpretation):
    def _sum_reductor(self, tensor, dims):
        for _ in range(dims):
            tensor = jnp.sum(tensor, axis=0)
        return tensor

    def _min_reductor(self, tensor, dims):
        for _ in range(dims):
            tensor = jnp.min(tensor, axis=0)
        return tensor

    def _max_reductor(self, tensor, dims):
        for _ in range(dims):
            tensor = jnp.max(tensor, axis=0)
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
    def fold(self, semiring, streams, body):
        reductor = self._get_reductor(semiring)
        if reductor is None or not (
            all(isinstance(s, collections.abc.Sized) for s in streams.values())
            and all(typeof(k()) is jax.Array for k in streams.keys())
        ):
            return fwd()

        if not (isinstance(body, Term) and body.op is D):
            return fwd()

        if len(body.args) <= 0:
            return jnp.array([])

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
        indexed_streams = {
            k: deffn(jax_getitem(v, [old_to_fresh[k]()])) for k, v in streams.items()
        }

        fvars = fvsof(value)
        unused_streams = {k: v for k, v in streams.items() if k not in fvars}
        with handler(indexed_streams):
            result = evaluate(value)
        result = result * functools.reduce(
            operator.mul, (len(v) for v in unused_streams.values()), 1
        )

        result_indices = sizesof(result)
        reduction_indices = [i for i in result_indices if fresh_to_old[i] not in indices]

        result = bind_dims(result, *reduction_indices)
        result = reductor(result, len(reduction_indices))
        return bind_dims(
            result, *[i for i in result_indices if i not in reduction_indices]
        )


@defop
def reals(*, shape: tuple[int, ...] = ()) -> Iterable[jax.Array]:
    raise NotImplementedError


@defop
def key() -> jax.Array:
    return jax.random.key(0)


class FlipOptimizationFold(ObjectInterpretation):
    """Convert Max/ArgMax problems to Min/ArgMin by negating values.

    This handler transforms maximization problems into minimization problems
    by negating the objective function, allowing reuse of minimization algorithms.
    """

    @implements(fold)
    def fold(self, semiring, streams, body, **kwargs):
        # Only handle MaxAlg and ArgMaxAlg
        if semiring not in (MaxAlg, ArgMaxAlg):
            return fwd()

        # Determine the target semiring (Min for Max, ArgMin for ArgMax)
        target_semiring = MinAlg if semiring is MaxAlg else ArgMinAlg

        # Normalize the body to use D if it's not already
        if not (isinstance(body, Term) and body.op is D):
            # For ArgMaxAlg, body should be a tuple of (value, arg)
            if semiring is ArgMaxAlg:
                if not isinstance(body, tuple) or len(body) != 2:
                    return fwd()
                body = D(((), body))
            else:
                body = D(((), body))

        # For each key-value pair in the body
        new_args = []
        for indices, value in body.args:
            if semiring is MaxAlg:
                # For MaxAlg, just negate the value
                new_value = -value
            else:  # ArgMaxAlg
                # For ArgMaxAlg, negate the first element of the tuple (the value)
                # but keep the second element (the arg) unchanged
                if not (isinstance(value, tuple) and len(value) == 2):
                    raise ValueError("Expected a tuple of (value, arg) for ArgMaxAlg")
                val, arg = value
                new_value = (-val, arg)

            new_args.append((indices, new_value))

        # Create a new body with negated values
        new_body = D(*new_args)

        # Solve as a minimization problem
        result = fold(target_semiring, streams, new_body, **kwargs)

        # For MaxAlg, negate the result back
        if semiring is MaxAlg:
            if isinstance(result, dict):
                return {k: -v for k, v in result.items()}
            else:
                return -result
        else:  # ArgMaxAlg
            # For ArgMaxAlg, negate the first element of the result tuple back
            if isinstance(result, dict):
                return {k: (-v[0], v[1]) for k, v in result.items()}
            elif isinstance(result, tuple):
                return (-result[0], result[1])
            else:
                fwd()


class GradientOptimizationFold(ObjectInterpretation):
    """Handle min/argmin over reals using gradient descent.

    Notes:
    - A single empty output index is expected. Nontrivial output indexes would in
    principle allow us to represent partial optimization problems like the following:
    fold(MinAlg, {x: reals(), y: reals()}, {x(): f(x(), y())}) = \\lambda x. min_{y\\in R} f(x, y).

    """

    def __init__(
        self, optimizer=optax.adam, steps=1000, init=None, progress=False, **kwargs
    ):
        self.optimizer = optimizer
        self.optimizer_kwargs = kwargs
        if steps <= 0:
            raise ValueError("Expected a positive number of steps")
        self.steps = steps
        self.init = {} if init is None else init
        self.progress = progress

    @implements(fold)
    def fold(self, semiring, streams, body):
        if not (
            semiring in (MinAlg, ArgMinAlg)
            and all(isinstance(v, Term) and v.op is reals for v in streams.values())
        ):
            return fwd()

        if not (isinstance(body, Term) and body.op is D):
            return fwd()

        if len(body.args) <= 0:
            return jnp.array([])

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

        # Initialize parameters using provided init values or zeros
        param_keys = list(streams.keys())
        param_values = []
        for k in param_keys:
            r = streams[k]
            if r.op is not reals:
                raise ValueError("Expected reals as stream values")
            shape = r.args[0] if len(r.args) > 0 else ()

            if k in self.init:
                # Use provided initialization
                init_value = self.init[k]
                if not isinstance(init_value, jax.Array):
                    init_value = jnp.array(init_value)

                # Ensure the shape matches
                if init_value.shape != shape and shape != ():
                    raise ValueError(
                        f"Init shape mismatch for {k}: expected {shape}, got {init_value.shape}"
                    )
                param = init_value
            else:
                # Default to zeros
                param = jnp.zeros(shape)
            param_values.append(param)

        loss = deffn(value, key, *param_keys)

        # we must be able to fully evaluate the loss function
        loss_value = loss(key(), *param_values)
        if not isinstance(loss_value, jax.Array):
            raise ValueError(f"Loss must evaluate to an array, but got {loss_value}")

        loss_grad = jax.jit(jax.grad(loss, argnums=range(1, len(param_values) + 1)))

        optimizer = self.optimizer(**self.optimizer_kwargs)
        opt_state = optimizer.init(param_values)

        steps_iter = range(self.steps)
        if self.progress:
            from tqdm import tqdm

            steps_iter = tqdm(steps_iter)

        keys = jax.random.split(key(), self.steps)
        for i in steps_iter:
            grads = list(loss_grad(keys[i], *param_values))
            assert all(isinstance(g, jax.Array) for g in tree.flatten(grads))
            updates, opt_state = optimizer.update(grads, opt_state)
            param_values = optax.apply_updates(param_values, updates)

        final_loss = loss(key, *param_values)

        if semiring is MinAlg:
            return final_loss

        with handler({v: deffn(p) for (v, p) in zip(param_keys, param_values)}):
            final_arg = evaluate(arg)

        return final_loss, final_arg


class LikelihoodWeightingFold(ObjectInterpretation):
    """Handle expectation computation using likelihood weighting."""

    def __init__(self, samples=2):
        self.samples = samples

    @implements(fold)
    def fold(self, semiring, streams, body):
        if not (
            semiring is LinAlg
            and all(
                issubclass(typeof(v), numpyro.distributions.Distribution)
                for v in streams.values()
            )
        ):
            reason = "semiring" if semiring is not LinAlg else "streams"
            logger.debug(
                f"Skipping likelihood weighting (reason {reason}): fold({semiring}, {streams}, {body}"
            )
            return fwd()

        sample_streams = {}
        index_streams = {}
        for k, v in streams.items():
            s = defop(jax.Array, name="sample")

            if isinstance(k, Operation):
                value = k
                samples = sample(key(), v, (self.samples,))[s()]
                sample_streams[value] = deffn(samples)
            elif isinstance(k, tuple):
                if not (len(k) == 2 and all(isinstance(i, Operation) for i in k)):
                    raise ValueError(
                        "Expected a tuple of (value, weight) for likelihood weighting"
                    )
                (value, weight) = k
                samples = sample(key(), v, (self.samples,))
                weights = log_prob(v, samples)
                weights = weights - jnp.logsumexp(weights)
                samples = jax_getitem(samples, [s()])
                weights = jax_getitem(weights, [s()])
                sample_streams[value] = deffn(samples)
                sample_streams[weight] = deffn(weights)
            else:
                raise ValueError("Unexpected key type")

            index_streams[s] = jnp.arange(self.samples)

        with handler(sample_streams):
            body = evaluate(body)

        return fold(LinAlg, index_streams, body)


class FoldFusion(ObjectInterpretation):
    """Implements the identity: fold(R, S1, fold(R, S2, body)) = fold(R, S1 x S2, body)

    This optimization fuses nested folds with the same semiring into a single fold
    over the product of their streams, which can be more efficient.
    """

    @implements(fold)
    def fold(self, semiring, streams, body):
        # Only proceed if body is a fold operation
        if not (isinstance(body, Term) and body.op is fold):
            return fwd()

        # Extract the inner fold's parameters
        inner_semiring, inner_streams, inner_body = body.args

        # Only fuse if both folds use the same semiring
        if not (semiring == inner_semiring):
            return fwd()

        # Return the fused fold
        return fold(semiring, streams | inner_streams, inner_body)


class FoldIndexDistributivity(ObjectInterpretation):
    """Implements the identity: fold(R, S, D((I1, X1), ..., (IN, XN))) = fold(R, S, D((I1, X1))) R.+ ... R.+ fold(R, S, D((IN, XN)))"""

    @implements(fold)
    def fold(self, semiring, streams, body):
        # Check if the body is a D term with multiple arguments (representing addition)
        if not (isinstance(body, Term) and body.op is D):
            return fwd()

        # If there's only 0 or 1 argument, no distribution needed
        if len(body.args) <= 1:
            return fwd()

        # Create separate fold operations for each term
        results = []
        for indices, value in body.args:
            # Create a new D term with just this key-value pair
            term_body = D((indices, value))
            # Compute fold for this term
            term_result = fold(semiring, streams, term_body)
            results.append(term_result)

        # Combine results using semiring addition
        return functools.reduce(lambda a, b: semiring.add(a, b), results, semiring.zero)


class FoldAddDistributivity(ObjectInterpretation):
    """Implements the identity: fold(R, S, D((I, X1 R.+ ... R.+ XN))) = fold(R, S, D((I1, X1), ..., (IN, XN)))

    This optimization distributes fold over addition within a single index, allowing
    for parallel computation of individual terms.
    """

    @implements(fold)
    def fold(self, semiring, streams, body):
        if not isinstance(body, Term):
            return fwd()

        # Check if the body is a D term with a single argument
        if body.op is D and len(body.args) == 1:
            indices, value = body.args[0]

            if not (isinstance(value, Term) and value.op is semiring.add):
                return fwd()

            terms = value.args

            # Create separate D terms for each addend
            new_terms = []
            for term in terms:
                new_terms.append((indices, term))

                # Create a new body with separate terms
                new_body = D(*new_terms)

            # Apply fold to the new body
            return fold(semiring, streams, new_body)

        elif body.op is semiring.add:
            # Create separate D terms for each addend
            new_terms = []
            for term in body.args:
                new_terms.append(fold(semiring, streams, term))

            # Apply fold to the new body
            return functools.reduce(semiring.add, new_terms, semiring.zero)

        else:
            return fwd()


class FoldFactorization(ObjectInterpretation):
    """Implements the identity: fold(R, S, A * B), free(A) ∩ S = {} => A * fold(R, S, B)

    This optimization factors out terms that don't depend on the fold variables,
    which can significantly reduce computation by avoiding redundant calculations.
    """

    def _mul_op(self, semiring):
        if semiring is LinAlg:
            return jnp.multiply
        elif semiring is MinAlg:
            return jnp.min
        elif semiring is MaxAlg:
            return jnp.max
        else:
            return None

    @staticmethod
    def _separate_factors(factors, stream_vars):
        indep_factors = []
        dep_factors = []
        for f in factors:
            if len(fvsof(f) & stream_vars) == 0:
                indep_factors.append(f)
            else:
                dep_factors.append(f)

        return indep_factors, dep_factors

    @implements(fold)
    def fold(self, semiring, streams, body):
        if not isinstance(body, Term):
            return fwd()

        # Check if the body is a D term
        if body.op is D:
            # We only handle single-term bodies for now
            if len(body.args) != 1:
                return fwd()

            indices, value = body.args[0]

            # Check if value is a multiplication operation
            if not (isinstance(value, Term) and value.op is self._mul_op(semiring)):
                return fwd()

            indep_factors, dep_factors = FoldFactorization._separate_factors(
                value.args, set(tree.flatten(streams.keys()))
            )

            if indep_factors == []:
                return fwd()

            indep_prod = functools.reduce(semiring.mul, indep_factors, semiring.one)
            dep_prod = functools.reduce(semiring.mul, dep_factors, semiring.one)
            dep_result = fold(semiring, streams, D((indices, dep_prod)))
            return semiring.mul(indep_prod, dep_result)

        elif body.op is semiring.mul:
            indep_factors, dep_factors = FoldFactorization._separate_factors(
                body.args, set(tree.flatten(streams.keys()))
            )

            if indep_factors == []:
                return fwd()

            indep_prod = functools.reduce(semiring.mul, indep_factors, semiring.one)
            dep_prod = functools.reduce(semiring.mul, dep_factors, semiring.one)
            return semiring.mul(indep_prod, fold(semiring, streams, dep_prod))
        else:
            return fwd()


class PushMulFold(ObjectInterpretation):
    @implements(mul)
    def mul(self, lhs, rhs):
        if isinstance(rhs, Term) and rhs.op is fold:
            semiring, streams, body = rhs.args
            if fvs_lhs & streams.keys() == {}:
                return fold(semiring, streams, lhs * body)
        return fwd()


# fold(R, S, A * B), free(A) \intersect S = {} => fold(R, S, A) = A * fold(R, S, B)


simplify_intp = functools.reduce(
    coproduct,
    [
        ProductFold(),
        FoldFusion(),
        # FoldIndexDistributivity(),
        # FoldAddDistributivity(),
        # PushMulFold(),
        # FoldFactorization(),
    ],
)

dense_fold_intp = functools.reduce(
    coproduct,
    [
        # NormalizeValueFold(),
        DenseTensorArgFold(),
        DenseTensorFold(),
        # FlipOptimizationFold(),
        ProductFold(),
        # FoldFusion(),
        FoldIndexDistributivity(),
        FoldAddDistributivity(),
        FoldFactorization(),
    ],
)
