"""Mirrored JAX transforms, with source retracing for each interpretation."""

import functools
import inspect
from dataclasses import dataclass

import jax

from effectful.internals.runtime import _get_args, get_interpretation, interpreter
from effectful.internals.weak import StrongIdRef
from effectful.ops.semantics import fwd
from effectful.ops.syntax import defop
from effectful.ops.types import Term

from ._callables import wrap_callable

_POLICY = "_effectful_interpretation"


@functools.partial(
    jax.tree_util.register_dataclass, data_fields=("value",), meta_fields=("names",)
)
@dataclass
class _NamedResult:
    value: object
    names: tuple


def _pack_named(value):
    from ._handlers import bind_dims, sizesof

    if isinstance(value, Term):
        names = tuple(sizesof(value))
        if names:
            return _NamedResult(bind_dims(value, *names), names)
    return value


def _unpack_named(value):
    from ._handlers import jax_getitem

    if isinstance(value, _NamedResult):
        return jax_getitem(value.value, tuple(name() for name in value.names))
    return value


@defop
def jit(fun=None, **options):
    """Compile the original callable separately for each effective interpretation.

    The hidden static keyword does not shift public positional argument indices.
    Keys use the runtime's identity-reference machinery. Snapshots exclude
    call-bound forwarding state, and are scoped to this compiled wrapper.
    """
    if fun is None:
        return functools.partial(jit, **options)
    signature = inspect.signature(fun)
    if _POLICY in signature.parameters:
        raise ValueError(f"{_POLICY!r} is reserved by effectful JIT")
    parameters = list(signature.parameters.values())
    offset = next(
        (i for i, p in enumerate(parameters) if p.kind == p.VAR_KEYWORD),
        len(parameters),
    )
    parameters.insert(
        offset, inspect.Parameter(_POLICY, inspect.Parameter.KEYWORD_ONLY)
    )

    @functools.wraps(fun)
    def staged(*args, _effectful_interpretation, **kwargs):
        with interpreter(_effectful_interpretation()):
            result = fun(*args, **kwargs)
            return jax.tree.map(
                _pack_named, result, is_leaf=lambda x: isinstance(x, Term)
            )

    staged.__signature__ = signature.replace(parameters=parameters)
    static_names = options.get("static_argnames")
    if static_names is None:
        positions = [
            p.name
            for p in signature.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        nums = options.get("static_argnums") or ()
        nums = (nums,) if isinstance(nums, int) else nums
        static_names = tuple(
            positions[i] for i in nums if -len(positions) <= i < len(positions)
        )
    elif isinstance(static_names, str):
        static_names = (static_names,)
    compiled = jax.jit(
        wrap_callable(staged),
        **{**options, "static_argnames": (*static_names, _POLICY)},
    )
    policies = {}

    def policy():
        current = {
            op: impl
            for op, impl in get_interpretation().items()
            if op is not fwd and op is not _get_args
        }
        key = tuple(sorted((id(op), id(impl)) for op, impl in current.items()))
        if key not in policies:
            # Holding the snapshot also prevents identity reuse in this cache.
            policies[key] = StrongIdRef(current)
        return policies[key]

    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        result = compiled(*args, **kwargs, **{_POLICY: policy()})
        return jax.tree.map(
            _unpack_named, result, is_leaf=lambda x: isinstance(x, _NamedResult)
        )

    wrapped.lower = lambda *a, **k: compiled.lower(*a, **k, **{_POLICY: policy()})
    wrapped._cache_size = compiled._cache_size
    wrapped.clear_cache = compiled.clear_cache
    return wrapped


def _callable_api(original):
    """Wrap a callable at an immediately executing JAX API boundary."""

    @functools.wraps(original)
    def default(fun, *args, **kwargs):
        return original(wrap_callable(fun), *args, **kwargs)

    return defop(default)


def _transform(original):
    """Defer wrapping until invocation, when the active interpretation is known."""

    @functools.wraps(original)
    def default(fun, *transform_args, **transform_options):
        @functools.wraps(fun)
        def transformed(*args, **kwargs):
            return original(wrap_callable(fun), *transform_args, **transform_options)(
                *args, **kwargs
            )

        return transformed

    return defop(default)


# These APIs act on callables/pytrees, not numerical operands. In particular,
# custom derivative implementations on handlers must not be densified here.
grad = _transform(jax.grad)
value_and_grad = _transform(jax.value_and_grad)
jacfwd = _transform(jax.jacfwd)
jacrev = _transform(jax.jacrev)
hessian = _transform(jax.hessian)
vmap = _transform(jax.vmap)
checkpoint = _transform(jax.checkpoint)
remat = checkpoint
make_jaxpr = _transform(jax.make_jaxpr)

jvp = _callable_api(jax.jvp)
vjp = _callable_api(jax.vjp)
linearize = _callable_api(jax.linearize)
linear_transpose = _callable_api(jax.linear_transpose)
eval_shape = _callable_api(jax.eval_shape)
custom_jvp = _callable_api(jax.custom_jvp)
custom_vjp = _callable_api(jax.custom_vjp)
