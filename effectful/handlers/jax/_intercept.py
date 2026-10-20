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

from effectful.internals.runtime import get_interpretation
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, defop, implements
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


class JaxPrimitiveOperation(Operation):
    @classmethod
    def define(cls, primitive):
        return super().define(
            functools.partial(bind_primitive, primitive), name=primitive.name
        )


@functools.cache
def primitive_op(primitive):
    """Return the unique effectful Operation for a JAX Primitive.

    Handlers receive primitive operands positionally and primitive parameters
    as keyword arguments. Multiple-result primitives return a list of values.
    ``fwd`` delegates to the next handler or the original primitive binding.
    """

    return JaxPrimitiveOperation.define(primitive)


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


def _join_lists(left, right):
    return [join(a, b) for a, b in zip(left, right, strict=True)]


def _coerce_list(values, templates):
    return [coerce(a, b) for a, b in zip(values, templates, strict=True)]


class HigherOrderJaxIntp(ObjectInterpretation):
    @implements(primitive_op(jax._src.pjit.jit_p))
    def jit(self, *args, **kwargs):
        if any(kwargs.get("donated_invars", ())):
            raise NotImplementedError(
                "effectful interception does not support buffer donation"
            )
        shardings = (*kwargs.get("in_shardings", ()), *kwargs.get("out_shardings", ()))
        if any(getattr(s, "is_fully_replicated", True) is False for s in shardings):
            raise NotImplementedError(
                "effectful interception does not support distributed sharding"
            )
        return _eval_jaxpr(kwargs["jaxpr"], *args)

    @implements(primitive_op(jax.lax.scan_p))
    def scan(self, *args, **params):
        args = [normalize(v) for v in args]
        nc, nk = params["num_consts"], params["num_carry"]
        consts = args[:nc]
        carry = [normalize(v) for v in args[nc : nc + nk]]
        xs = args[nc + nk :]

        # if hasattr(self.adapter, "mapped"):
        #     xs = [self.adapter.mapped(v) for v in xs]  # ??

        # Operate on buffer pytrees, not on logical dense array placeholders.
        def run(c, x):
            out = _eval_jaxpr(params["jaxpr"], [*consts, *c, *x])
            return [normalize(v) for v in out[:nk]], [normalize(v) for v in out[nk:]]

        sample_xs = jax.tree.map(
            lambda a: jax.ShapeDtypeStruct(a.shape[1:], a.dtype), xs
        )
        for _ in range(8):
            out_c, _ = jax.eval_shape(run, carry, sample_xs)
            templates = _join_lists(carry, out_c)
            joined = _coerce_list(carry, templates)
            if jax.tree.structure(joined) == jax.tree.structure(carry):
                carry = joined
                break
            carry = joined
        else:
            raise RuntimeError("scan representation join did not converge")

        def body(c, x):
            out_c, out_y = run(c, x)
            return _coerce_list(out_c, carry), out_y

        out_c, out_y = fwd(
            body,
            carry,
            xs,
            length=params["length"],
            reverse=params["reverse"],
            unroll=params["unroll"],
        )

        # if hasattr(self.adapter, "stacked"):  # ??
        #     out_y = [self.adapter.stacked(v, params["length"]) for v in out_y]

        return [*out_c, *out_y]

    @implements(primitive_op(jax.lax.cond_p))
    def cond(self, args, params):
        index, *operands = args
        branches = params["branches"]
        outputs = [
            jax.eval_shape(lambda *xs, b=b: _eval_jaxpr(b, xs), *operands)
            for b in branches
        ]
        templates = functools.reduce(_join_lists, outputs)

        def branch(b):
            return lambda *xs: _coerce_list(_eval_jaxpr(b, xs), templates)

        return jax.lax.switch(index, tuple(branch(b) for b in branches), *operands)

    @implements(primitive_op(jax.lax.while_p))
    def while_loop(self, args, params):
        nc, nb = params["cond_nconsts"], params["body_nconsts"]
        cond_consts, body_consts, init = args[:nc], args[nc : nc + nb], args[nc + nb :]
        init = [normalize(v) for v in init]

        def run_body(xs):
            return [
                normalize(v)
                for v in _eval_jaxpr(params["body_jaxpr"], [*body_consts, *xs])
            ]

        for _ in range(8):
            out = jax.eval_shape(run_body, init)
            new = _coerce_list(init, _join_lists(init, out))
            if jax.tree.structure(new) == jax.tree.structure(init):
                init = new
                break
            init = new
        else:
            raise RuntimeError("while representation join did not converge")

        def cond(xs):
            return _eval_jaxpr(params["cond_jaxpr"], [*cond_consts, *xs])[0]

        return fwd(cond, lambda xs: _coerce_list(run_body(xs), init), init)


class _InterceptTrace(core.Trace):
    def __init__(self, parent, specializations=None):
        super().__init__()
        self.parent = parent
        # Scoped to one interception trace: no stale handlers or retained tracers.
        self.specializations = {} if specializations is None else specializations

    def _unwrap(self, value):
        if isinstance(value, _InterceptTracer) and value._trace is self:
            return value.value
        return value

    def _wrap(self, value):
        return _InterceptTracer(self, value)

    def process_primitive(self, primitive, tracers, params, /):
        args = [self._unwrap(x) for x in tracers]
        with core.set_current_trace(self.parent):
            op = primitive_op(primitive)
            result = op(*args, **params)
        return (
            [self._wrap(v) for v in result]
            if primitive.multiple_results
            else self._wrap(result)
        )

    def process_call(self, primitive, fun, tracers, params, /):
        # Ordinary JIT is handled by its closed jaxpr above. Keep opaque calls
        # and their transformation rules intact, rather than tracing their primal.
        with core.set_current_trace(self.parent):
            args = [materialize(self._unwrap(x)) for x in tracers]
            out = primitive.bind(fun, *args, **params)
        return [self._wrap(v) for v in out]

    def process_custom_jvp_call(
        self, primitive, fun, jvp, tracers, /, *, symbolic_zeros
    ):
        with core.set_current_trace(self.parent):
            args = [materialize(self._unwrap(x)) for x in tracers]
            out = bind_custom(
                primitive, args, (fun, jvp), dict(symbolic_zeros=symbolic_zeros)
            )
        return [self._wrap(v) for v in out]

    def process_custom_vjp_call(
        self, primitive, fun, fwd, bwd, tracers, /, *, out_trees, symbolic_zeros
    ):
        with core.set_current_trace(self.parent):
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
    interpretation = get_interpretation()

    @functools.wraps(fn)
    @handler(HigherOrderJaxIntp())
    @handler(interpretation)
    def wrapped(*args, **kwargs):
        with core.take_current_trace() as parent:
            trace = _InterceptTrace(parent)

            def wrap(value):
                if is_value(value) or isinstance(value, (jax.Array, core.Tracer)):
                    return trace._wrap(value)
                return value

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

            with core.set_current_trace(parent):
                return jax.tree.map(normalize, out, is_leaf=is_value)

    return wrapped


__all__ = ["intercept", "primitive_op", "represented_value"]
