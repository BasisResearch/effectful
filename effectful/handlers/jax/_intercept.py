"""Experimental interpretation of ordinary JAX primitive calls."""

import functools
import inspect
import warnings
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from typing import Any, Protocol

import jax
from prettyprinter import pprint

from effectful.internals.runtime import get_interpretation
from effectful.ops.semantics import coproduct, fwd, handler
from effectful.ops.syntax import ObjectInterpretation, deffn, defop, implements
from effectful.ops.types import Operation

from ._compat import (
    UNSUPPORTED_HIGHER_ORDER_PRIMITIVES,
    bind_custom,
    bind_primitive,
    check_version,
    core,
    custom_bind_params,
    custom_jvp_call_p,
    custom_vjp_call_p,
    eval_program,
    lu,
    with_constants,
)


def _id(x):
    return x


def _compose(outer, inner):
    return lambda x: outer(inner(x))


@defop
def upcast[S, T](left: S, right: T) -> Callable[[S | T], Any]:
    """Choose one smart constructor for both abstract representation witnesses.

    Inputs are outputs of ``jax.eval_shape``. The returned function must accept
    either source representation with real or tracer-backed buffers, preserve
    meaning, and produce a shared layout without capturing either input's buffers.
    JAX validates the resulting physical layouts at the control-flow boundary.
    """
    if type(left) is type(right):
        return _id
    raise NotImplementedError(
        f"No upcast for {type(left).__name__} and {type(right).__name__}"
    )


def upcast_many(*values):
    """Return one conversion per abstract value to a common representation."""
    if not values:
        return ()
    common = values[0]
    conversions = [_id]
    for other in values[1:]:
        convert = upcast(common, other)
        conversions = [_compose(convert, previous) for previous in conversions]
        conversions.append(convert)
        common = jax.eval_shape(lambda x: convert(x), common)
    return tuple(conversions)


def _apply_conversions(conversions, values):
    return [convert(x) for convert, x in zip(conversions, values, strict=True)]


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


def _warn_unsupported(primitive, *args, **params):
    warnings.warn(
        f"effectful interception does not support higher-order primitive "
        f"{primitive.name!r}; operations inside its subprograms will not be intercepted",
        RuntimeWarning,
        stacklevel=2,
    )
    return fwd(*args, **params)


_unsupported_intp = {
    _inner_primop(p): functools.partial(_warn_unsupported, p)
    for p in UNSUPPORTED_HIGHER_ORDER_PRIMITIVES
}


class DefaultIntp(ObjectInterpretation):
    @implements(_inner_primop(custom_jvp_call_p))
    def _custom_jvp_call_default(self, *args, **params):
        subfuns, keywords = custom_bind_params(custom_jvp_call_p, params)
        return _bind_intercepted_custom(custom_jvp_call_p, args, subfuns, keywords)

    @implements(_inner_primop(custom_vjp_call_p))
    def _custom_vjp_call_default(self, *args, **params):
        subfuns, keywords = custom_bind_params(custom_vjp_call_p, params)
        return _bind_intercepted_custom(custom_vjp_call_p, args, subfuns, keywords)

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
    def _cond_p_default(self, index, *operands, branches, **params):
        def branch(jaxpr):
            @intercept
            def run(*xs):
                return _eval_jaxpr(jaxpr, *map(represented_value, xs))

            return run

        def converted(run, conversions):
            return lambda *xs: _apply_conversions(conversions, run(*xs))

        with core.set_current_trace(_parent_trace()):
            runs = tuple(branch(b) for b in branches)
            outputs = [jax.eval_shape(run, *operands) for run in runs]
            conversions = [[] for _ in runs]
            for alternatives in zip(*outputs, strict=True):
                for casts, convert in zip(
                    conversions, upcast_many(*alternatives), strict=True
                ):
                    casts.append(convert)
            # switch validates complete output pytrees and buffer avals.
            return jax.lax.switch(
                index,
                tuple(converted(run, casts) for run, casts in zip(runs, conversions)),
                *operands,
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
            initial = jax.eval_shape(_id, init)
            outputs = jax.eval_shape(body, init)
            conversions = [upcast(a, b) for a, b in zip(initial, outputs, strict=True)]
            init = tuple(_apply_conversions(conversions, init))

            def converted(xs):
                return tuple(_apply_conversions(conversions, body(xs)))

            # Retracing on the widened carry must produce the same physical
            # layout; while_loop validates this after the single widening step.
            return list(jax.lax.while_loop(cond, converted, init))


class _InterceptTracer(core.Tracer):
    def __init__(self, trace, value):
        self._trace = trace
        self.value = value

    @property
    def aval(self):
        return core.typeof(self.value)

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
    def _wrap_inputs(self, args, kwargs):
        def wrap(value):
            if isinstance(value, (jax.Array, core.Tracer)):
                return self._wrap(value)
            return value

        return jax.tree.map(wrap, (args, kwargs))

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
        args = [self._unwrap(x) for x in tracers]
        out = _bind_intercepted_custom(
            primitive, args, (fun, jvp), dict(symbolic_zeros=symbolic_zeros)
        )
        return [self._wrap(v) for v in out]

    def process_custom_vjp_call(
        self, primitive, fun, fwd, bwd, tracers, /, *, out_trees, symbolic_zeros
    ):
        args = [self._unwrap(x) for x in tracers]
        out = _bind_intercepted_custom(
            primitive,
            args,
            (fun, fwd, bwd),
            dict(out_trees=out_trees, symbolic_zeros=symbolic_zeros),
        )
        return [self._wrap(v) for v in out]


def _bind_intercepted_custom(primitive, args, subfuns, params):
    """Bind a custom derivative call on the parent trace with intercepted bodies.

    ``subfuns`` are JAX WrappedFuns for the primal and derivative rules. Wrapping
    each one re-enters interception when the parent trace executes or stages it,
    so operations inside derivative rules are handled too. Both live trace hooks
    and defaults evaluating stored jaxprs can use this binding path.
    """
    with core.set_current_trace(_parent_trace()):
        return bind_custom(
            primitive, args, tuple(_intercept_fun(f) for f in subfuns), params
        )


def _intercept_fun(fun):
    """Intercept a JAX subfunction while preserving its transformation stores."""
    return lu.wrap_init(intercept(fun.call_wrapped), debug_info=fun.debug_info)


@defop
def _new_trace() -> _InterceptTrace:
    """Construct the trace used by this interception, including nested bodies."""
    return _InterceptTrace()


def intercept[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    """Transform ``fn`` so ordinary JAX primitives dispatch through effectful."""
    check_version()

    @functools.wraps(fn)
    @handler(coproduct(_unsupported_intp, DefaultIntp()))
    def wrapped(*args, **kwargs):
        trace = _new_trace()

        with core.take_current_trace() as parent:
            with handler({_parent_trace: deffn(parent)}):
                inputs = trace._wrap_inputs(args, kwargs)
                # Compose the complete interpretation once per trace. Handler
                # implementations may still invoke other effectful operations.
                with core.set_current_trace(trace):
                    out = fn(*inputs[0], **inputs[1])

                return jax.tree.map(
                    trace._unwrap,
                    out,
                    is_leaf=lambda x: isinstance(x, _InterceptTracer),
                )

    return wrapped


def intercept_jit[**P, T](fn: Callable[P, T], *args, **kwargs) -> Callable[P, T]:
    """Compile `fn` using `jax.jit`, passing the current interpretation as a
    static argument so that the function re-jits when called in a different
    interpretation.

    """
    raise NotImplementedError


__all__ = ["intercept", "primitive_op", "represented_value"]
