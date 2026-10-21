"""Experimental interpretation of ordinary JAX primitive calls.

The adapter is deliberately independent of tensors, sparsity, and linear algebra.
It supplies logical avals and bounded pytree representations. JAX continues to
compile, batch, and differentiate the numerical buffers in those representations.
"""

import functools
import inspect
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from typing import Any, Protocol

import jax
from prettyprinter import pprint

from effectful.internals.runtime import get_interpretation
from effectful.ops.semantics import handler
from effectful.ops.syntax import ObjectInterpretation, deffn, defop, implements
from effectful.ops.types import Operation

from ._compat import (
    bind_custom,
    bind_primitive,
    check_version,
    core,
    eval_program,
    with_constants,
)

"""Contract for values with a logical aval different from their buffers.

Values must be registered JAX pytrees. ``normalize`` bounds expression
growth. ``join`` returns a representation template valid for both values;
``coerce`` changes a value to that template without changing its meaning.
Templates may contain ShapeDtypeStruct leaves, rather than concrete arrays.
"""


@defop
def is_value(value) -> bool:
    return False


@defop
def aval(value) -> bool:
    return core.typeof(value)


@defop
def materialize(value):
    return value


@defop
def normalize(value):
    return value


def _id(x):
    return x


@defop
def upcast[U, V, T](u: U, v: V) -> tuple[Callable[[U], T], Callable[[V], T]]:
    if type(u) == type(v):
        return (_id, _id)
    raise NotImplementedError


@defop
def _parent_trace() -> core.Trace:
    raise RuntimeError("BUG: _parent_trace is not set.")


class _JaxInnerPrimitiveOperation(Operation):
    @classmethod
    def define_primitive(cls, primitive):
        def default(*args, **kwargs):
            with core.set_current_trace(_parent_trace()):
                return bind_primitive(primitive, *args, **kwargs)

        return super().define(default, name=primitive.name)


@functools.cache
def _inner_primop(primitive):
    """Return the unique effectful Operation for a JAX Primitive.

    Handlers receive primitive operands positionally and primitive parameters
    as keyword arguments. Multiple-result primitives return a list of values.
    ``fwd`` delegates to the next handler or the original primitive binding.
    """

    return _JaxInnerPrimitiveOperation.define_primitive(primitive)


class JaxPrimitiveOperation(Operation):
    @classmethod
    def define_primitive(cls, primitive):
        inner_op = _inner_primop(primitive)

        def default(*args, **kwargs):
            return inner_op(*args, **kwargs)

        return super().define(default, name=primitive.name)


@functools.cache
def primitive_op(primitive):
    """Return the unique effectful Operation for a JAX Primitive.

    Handlers receive primitive operands positionally and primitive parameters
    as keyword arguments. Multiple-result primitives return a list of values.
    ``fwd`` delegates to the next handler or the original primitive binding.
    """

    return JaxPrimitiveOperation.define_primitive(primitive)


class DefaultIntp(ObjectInterpretation):
    @implements(_inner_primop(jax._src.pjit.jit_p))
    def _jit_p_default(self, *args, **kwargs):
        if any(kwargs.get("donated_invars", ())):
            raise NotImplementedError(
                "effectful interception does not support buffer donation"
            )
        shardings = (*kwargs.get("in_shardings", ()), *kwargs.get("out_shardings", ()))
        if any(getattr(s, "is_fully_replicated", True) is False for s in shardings):
            raise NotImplementedError(
                "effectful interception does not support distributed sharding"
            )

        @intercept
        def run(*xs):
            return _eval_jaxpr(kwargs["jaxpr"], *map(represented_value, xs))

        with core.set_current_trace(_parent_trace()):
            return jax.jit(run)(*args)

    @implements(_inner_primop(jax.lax.scan_p))
    def _scan_p_default(self, *args, **params):
        nc, nk = params["num_consts"], params["num_carry"]
        consts = args[:nc]
        carry = args[nc : nc + nk]
        xs = args[nc + nk :]

        @intercept
        def body(c, x):
            out = _eval_jaxpr(
                params["jaxpr"],
                *consts,
                *map(represented_value, c),
                *map(represented_value, x),
            )
            return tuple(out[:nk]), tuple(out[nk:])

        with core.set_current_trace(_parent_trace()):
            carry, ys = jax.lax.scan(
                body,
                carry,
                xs,
                length=params["length"],
                reverse=params["reverse"],
                unroll=params["unroll"],
            )
        return [*carry, *ys]

    @implements(_inner_primop(jax.lax.cond_p))
    def _cond_p_default(self, *args, **params):
        index, *operands = args

        def branch(jaxpr):
            @intercept
            def run(*xs):
                return _eval_jaxpr(jaxpr, *map(represented_value, xs))

            return run

        with core.set_current_trace(_parent_trace()):
            return jax.lax.switch(
                index, tuple(branch(b) for b in params["branches"]), *operands
            )

    @implements(_inner_primop(jax.lax.while_p))
    def _while_p_default(self, *args, **params):
        nc, nb = params["cond_nconsts"], params["body_nconsts"]
        cond_consts, body_consts, init = args[:nc], args[nc : nc + nb], args[nc + nb :]

        @intercept
        def cond(xs):
            return _eval_jaxpr(
                params["cond_jaxpr"], *cond_consts, *map(represented_value, xs)
            )[0]

        @intercept
        def body(xs):
            return tuple(
                _eval_jaxpr(
                    params["body_jaxpr"], *body_consts, *map(represented_value, xs)
                )
            )

        with core.set_current_trace(_parent_trace()):
            return list(jax.lax.while_loop(cond, body, init))


class _InterceptTracer(core.Tracer):
    def __init__(self, trace, value):
        self._trace = trace
        self.value = value

    @property
    def aval(self):
        return aval(self.value)

    def full_lower(self):
        return self


def represented_value(value):
    """Unwrap a value in an active interception trace (for semantic bridges)."""
    return value.value if isinstance(value, _InterceptTracer) else value


def _eval_jaxpr(closed, *args):
    """Interpret a program without flattening represented operands."""
    jaxpr = closed.jaxpr if hasattr(closed, "jaxpr") else closed
    consts = closed.consts if hasattr(closed, "consts") else ()
    env = dict(zip(jaxpr.constvars, consts, strict=True))
    env.update(zip(jaxpr.invars, args, strict=True))

    def read(var):
        return var.val if isinstance(var, core.Literal) else env[var]

    for eqn in jaxpr.eqns:
        result = primitive_op(eqn.primitive)(
            *(read(v) for v in eqn.invars), **eqn.params
        )
        results = result if eqn.primitive.multiple_results else [result]
        env.update(
            {
                var: value
                for var, value in zip(eqn.outvars, results, strict=True)
                if not isinstance(var, core.DropVar)
            }
        )
    return [read(var) for var in jaxpr.outvars]


class _InterceptTrace(core.Trace):
    def _unwrap(self, value):
        if isinstance(value, _InterceptTracer) and value._trace is self:
            return value.value
        return value

    def _wrap(self, value):
        return _InterceptTracer(self, value)

    def process_primitive(self, primitive, tracers, params, /):
        args = [self._unwrap(x) for x in tracers]
        op = primitive_op(primitive)
        with core.set_current_trace(self):
            result = op(*args, **params)
        return (
            [self._wrap(self._unwrap(v)) for v in result]
            if primitive.multiple_results
            else self._wrap(self._unwrap(result))
        )

    def process_custom_jvp_call(
        self, primitive, fun, jvp, tracers, /, *, symbolic_zeros
    ):
        with core.set_current_trace(_parent_trace()):
            args = [materialize(self._unwrap(x)) for x in tracers]
            out = bind_custom(
                primitive, args, (fun, jvp), dict(symbolic_zeros=symbolic_zeros)
            )
        return [self._wrap(v) for v in out]

    def process_custom_vjp_call(
        self, primitive, fun, fwd, bwd, tracers, /, *, out_trees, symbolic_zeros
    ):
        with core.set_current_trace(_parent_trace()):
            args = [materialize(self._unwrap(x)) for x in tracers]
            out = bind_custom(
                primitive,
                args,
                (fun, fwd, bwd),
                dict(out_trees=out_trees, symbolic_zeros=symbolic_zeros),
            )
        return [self._wrap(v) for v in out]


def intercept[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    """Transform ``fn`` so ordinary JAX primitives dispatch through effectful."""
    check_version()

    @functools.wraps(fn)
    @handler(DefaultIntp())
    def wrapped(*args, **kwargs):
        trace = _InterceptTrace()

        def wrap(value):
            if is_value(value) or isinstance(value, (jax.Array, core.Tracer)):
                return trace._wrap(value)
            return value

        with core.take_current_trace() as parent:
            with handler({_parent_trace: deffn(parent)}):
                inputs = jax.tree.map(wrap, (args, kwargs), is_leaf=is_value)
                # Compose the complete interpretation once per trace. Handler
                # implementations may still invoke other effectful operations.
                with core.set_current_trace(trace):
                    out = fn(*inputs[0], **inputs[1])

                out = jax.tree.map(
                    trace._unwrap,
                    out,
                    is_leaf=lambda x: isinstance(x, _InterceptTracer) or is_value(x),
                )
                return jax.tree.map(normalize, out, is_leaf=is_value)

    return wrapped


def intercept_jit[**P, T](fn: Callable[P, T], *args, **kwargs) -> Callable[P, T]:
    """Compile `fn` using `jax.jit`, passing the current interpretation as a
    static argument so that the function re-jits when called in a different
    interpretation.

    """
    raise NotImplementedError


__all__ = ["intercept", "primitive_op", "represented_value"]
