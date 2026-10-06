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

from effectful.ops.semantics import handler
from effectful.ops.syntax import defop

from ._compat import (
    bind_custom,
    bind_primitive,
    check_version,
    core,
    eval_program,
    with_constants,
)

_active_dispatch = ContextVar("effectful_jax_dispatch")
_active_interpretation = ContextVar("effectful_jax_interpretation", default=None)


class ValueAdapter(Protocol):
    """Contract for values with a logical aval different from their buffers.

    Values must be registered JAX pytrees. ``normalize`` bounds expression
    growth. ``join`` returns a representation template valid for both values;
    ``coerce`` changes a value to that template without changing its meaning.
    Templates may contain ShapeDtypeStruct leaves, rather than concrete arrays.
    """

    def is_value(self, value: Any) -> bool: ...
    def aval(self, value: Any) -> Any: ...
    def materialize(self, value: Any) -> Any: ...
    def normalize(self, value: Any) -> Any: ...
    def join(self, left: Any, right: Any) -> Any: ...
    def coerce(self, value: Any, template: Any) -> Any: ...


class _ArrayAdapter:
    def is_value(self, value):
        return False

    def aval(self, value):
        return core.typeof(value)

    def materialize(self, value):
        return value

    def normalize(self, value):
        return value

    def join(self, left, right):
        return right

    def coerce(self, value, template):
        return value


@functools.cache
def primitive_op(primitive):
    """Return the unique effectful Operation for a JAX Primitive.

    Handlers receive primitive operands positionally and primitive parameters
    as keyword arguments. Multiple-result primitives return a list of values.
    ``fwd`` delegates to the next handler or the original primitive binding.
    """

    @defop
    def operation(*args, **params) -> Any:
        return bind_primitive(primitive, args, params)

    operation.__name__ = f"jax_{primitive.name}"
    operation._jax_primitive = primitive
    return operation


class _InterceptTracer(core.Tracer):
    def __init__(self, trace, value):
        self._trace = trace
        self.value = value

    @property
    def aval(self):
        return self._trace.adapter.aval(self.value)

    def full_lower(self):
        return self


def represented_value(value):
    """Unwrap a value in an active interception trace (for semantic bridges)."""
    return value.value if isinstance(value, _InterceptTracer) else value


class _InterceptTrace(core.Trace):
    def __init__(self, parent, interpretation, adapter, specializations=None):
        super().__init__()
        self.parent = parent
        self.interpretation = interpretation
        self.adapter = adapter
        # Scoped to one interception trace: no stale handlers or retained tracers.
        self.specializations = {} if specializations is None else specializations

    def unwrap(self, value):
        if isinstance(value, _InterceptTracer) and value._trace is self:
            return value.value
        return value

    def wrap(self, value):
        return _InterceptTracer(self, value)

    def normalize(self, value):
        return self.adapter.normalize(value)

    def eval_jaxpr(self, closed, args):
        """Interpret a program without flattening represented operands."""
        jaxpr = closed.jaxpr if hasattr(closed, "jaxpr") else closed
        consts = closed.consts if hasattr(closed, "consts") else ()
        env = dict(zip(jaxpr.constvars, consts, strict=True))
        env.update(zip(jaxpr.invars, args, strict=True))

        def read(var):
            return var.val if isinstance(var, core.Literal) else env[var]

        for eqn in jaxpr.eqns:
            result = self.dispatch(
                eqn.primitive, [read(v) for v in eqn.invars], eqn.params
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

    def eval_child(self, closed, args):
        """Reuse buffer programs during abstract joins and control-flow staging.

        Captured source constants are explicit dynamic inputs, so changing a
        captured value does not change the cached specialization's semantics.
        The cache is shared only by children of the current interception trace.
        """
        source = closed.jaxpr if hasattr(closed, "jaxpr") else closed
        constants = closed.consts if hasattr(closed, "consts") else ()
        leaves, inputs = jax.tree.flatten((constants, args))
        signature = tuple(core.typeof(x) for x in leaves)
        key = (source, inputs, signature)
        cached = self.specializations.get(key)
        if cached is None:

            def stage(*buffers):
                consts, operands = jax.tree.unflatten(inputs, buffers)
                with core.take_current_trace() as parent:
                    child = _InterceptTrace(
                        parent, self.interpretation, self.adapter, self.specializations
                    )
                    with core.set_current_trace(parent):
                        return child.eval_jaxpr(
                            with_constants(source, consts), operands
                        )

            program, output = jax.make_jaxpr(stage, return_shape=True)(*leaves)
            cached = (program, jax.tree.structure(output))
            self.specializations[key] = cached
        program, output = cached
        return jax.tree.unflatten(output, eval_program(program, leaves))

    def default(self, primitive, args, params):
        name = primitive.name
        if name == "jit":
            if any(params.get("donated_invars", ())):
                raise NotImplementedError(
                    "effectful interception does not support buffer donation"
                )
            shardings = (
                *params.get("in_shardings", ()),
                *params.get("out_shardings", ()),
            )
            if any(getattr(s, "is_fully_replicated", True) is False for s in shardings):
                raise NotImplementedError(
                    "effectful interception does not support distributed sharding"
                )
            return self.eval_jaxpr(params["jaxpr"], args)
        if name == "scan":
            return self.scan(args, params)
        if name == "cond":
            return self.cond(args, params)
        if name == "while":
            return self.while_loop(args, params)
        return bind_primitive(
            primitive, [self.adapter.materialize(v) for v in args], params
        )

    def dispatch(self, primitive, args, params):
        op = primitive_op(primitive)
        token = _active_dispatch.set(self)
        try:
            with core.set_current_trace(self.parent):
                if _active_interpretation.get() is not self.interpretation:
                    # Crossing nested interception wrappers must select this
                    # trace's semantics, rather than the innermost wrapper's.
                    def original(*values, **kw):
                        return self.default(primitive, values, kw)

                    with handler({op: original}), handler(self.interpretation):
                        return op(*args, **params)
                if op in self.interpretation:
                    return op(*args, **params)

                # Unhandled primitives still need adapter fallback or recursive
                # interpretation, and remain observable to enclosing handlers.
                def original(*values, **kw):
                    return self.default(primitive, values, kw)

                with handler({op: original}):
                    return op(*args, **params)
        finally:
            _active_dispatch.reset(token)

    def process_primitive(self, primitive, tracers, params, /):
        args = [self.unwrap(x) for x in tracers]
        with core.set_current_trace(self.parent):
            result = self.dispatch(primitive, args, params)
        return (
            [self.wrap(v) for v in result]
            if primitive.multiple_results
            else self.wrap(result)
        )

    def process_call(self, primitive, fun, tracers, params, /):
        # Ordinary JIT is handled by its closed jaxpr above. Keep opaque calls
        # and their transformation rules intact, rather than tracing their primal.
        with core.set_current_trace(self.parent):
            args = [self.adapter.materialize(self.unwrap(x)) for x in tracers]
            out = primitive.bind(fun, *args, **params)
        return [self.wrap(v) for v in out]

    def process_custom_jvp_call(
        self, primitive, fun, jvp, tracers, /, *, symbolic_zeros
    ):
        with core.set_current_trace(self.parent):
            args = [self.adapter.materialize(self.unwrap(x)) for x in tracers]
            out = bind_custom(
                primitive, args, (fun, jvp), dict(symbolic_zeros=symbolic_zeros)
            )
        return [self.wrap(v) for v in out]

    def process_custom_vjp_call(
        self, primitive, fun, fwd, bwd, tracers, /, *, out_trees, symbolic_zeros
    ):
        with core.set_current_trace(self.parent):
            args = [self.adapter.materialize(self.unwrap(x)) for x in tracers]
            out = bind_custom(
                primitive,
                args,
                (fun, fwd, bwd),
                dict(out_trees=out_trees, symbolic_zeros=symbolic_zeros),
            )
        return [self.wrap(v) for v in out]

    def _join_lists(self, left, right):
        return [self.adapter.join(a, b) for a, b in zip(left, right, strict=True)]

    def _coerce_list(self, values, templates):
        return [
            self.adapter.coerce(a, b) for a, b in zip(values, templates, strict=True)
        ]

    def scan(self, args, params):
        args = [self.normalize(v) for v in args]
        nc, nk = params["num_consts"], params["num_carry"]
        consts = args[:nc]
        carry = [self.normalize(v) for v in args[nc : nc + nk]]
        xs = args[nc + nk :]
        if hasattr(self.adapter, "mapped"):
            xs = [self.adapter.mapped(v) for v in xs]

        # Operate on buffer pytrees, not on logical dense array placeholders.
        def run(c, x):
            out = self.eval_child(params["jaxpr"], [*consts, *c, *x])
            return [self.normalize(v) for v in out[:nk]], [
                self.normalize(v) for v in out[nk:]
            ]

        sample_xs = jax.tree.map(
            lambda a: jax.ShapeDtypeStruct(a.shape[1:], a.dtype), xs
        )
        for _ in range(8):
            out_c, _ = jax.eval_shape(run, carry, sample_xs)
            templates = self._join_lists(carry, out_c)
            joined = self._coerce_list(carry, templates)
            if jax.tree.structure(joined) == jax.tree.structure(carry):
                carry = joined
                break
            carry = joined
        else:
            raise RuntimeError("scan representation join did not converge")

        def body(c, x):
            out_c, out_y = run(c, x)
            return self._coerce_list(out_c, carry), out_y

        with core.set_current_trace(self.parent):
            out_c, out_y = jax.lax.scan(
                body,
                carry,
                xs,
                length=params["length"],
                reverse=params["reverse"],
                unroll=params["unroll"],
            )
        if hasattr(self.adapter, "stacked"):
            out_y = [self.adapter.stacked(v, params["length"]) for v in out_y]
        return [*out_c, *out_y]

    def cond(self, args, params):
        index, *operands = args
        branches = params["branches"]
        outputs = [
            jax.eval_shape(lambda *xs, b=b: self.eval_child(b, xs), *operands)
            for b in branches
        ]
        templates = functools.reduce(self._join_lists, outputs)

        def branch(b):
            return lambda *xs: self._coerce_list(self.eval_child(b, xs), templates)

        with core.set_current_trace(self.parent):
            return jax.lax.switch(index, tuple(branch(b) for b in branches), *operands)

    def while_loop(self, args, params):
        nc, nb = params["cond_nconsts"], params["body_nconsts"]
        cond_consts, body_consts, init = args[:nc], args[nc : nc + nb], args[nc + nb :]
        init = [self.normalize(v) for v in init]

        def run_body(xs):
            return [
                self.normalize(v)
                for v in self.eval_child(params["body_jaxpr"], [*body_consts, *xs])
            ]

        for _ in range(8):
            out = jax.eval_shape(run_body, init)
            new = self._coerce_list(init, self._join_lists(init, out))
            if jax.tree.structure(new) == jax.tree.structure(init):
                init = new
                break
            init = new
        else:
            raise RuntimeError("while representation join did not converge")

        def cond(xs):
            return self.eval_child(params["cond_jaxpr"], [*cond_consts, *xs])[0]

        with core.set_current_trace(self.parent):
            return jax.lax.while_loop(
                cond, lambda xs: self._coerce_list(run_body(xs), init), init
            )


def _signature_snapshot(fn):
    # Handler composition repeatedly consults callable signatures. In
    # particular, inspecting a functools.partial reconstructs its signature.
    # Snapshot on a bridge-owned wrapper, without mutating user callables.
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapped.__signature__ = inspect.signature(fn)
    return wrapped


def intercept(
    fn: Callable, *, interpretation: Mapping, value_adapter: ValueAdapter | None = None
):
    """Transform ``fn`` so ordinary JAX primitives dispatch through effectful.

    The interpretation is snapshotted at wrapper creation. Create a fresh
    wrapper to change its semantics; JAX's normal callable/pytree/static-argument
    caching then distinguishes the resulting compiled programs. The callable
    may itself contain warmed JIT functions, scan, cond, and vmap.
    """
    check_version()
    adapter = value_adapter or _ArrayAdapter()
    interpretation = {op: _signature_snapshot(fn) for op, fn in interpretation.items()}

    def default_for(primitive):
        def original(*values, **params):
            return _active_dispatch.get().default(primitive, values, params)

        return original

    defaults = {
        op: default_for(op._jax_primitive)
        for op in interpretation
        if hasattr(op, "_jax_primitive")
    }

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with core.take_current_trace() as parent:
            trace = _InterceptTrace(parent, interpretation, adapter)

            def wrap(value):
                if adapter.is_value(value) or isinstance(
                    value, (jax.Array, core.Tracer)
                ):
                    return trace.wrap(value)
                return value

            inputs = jax.tree.map(wrap, (args, kwargs), is_leaf=adapter.is_value)
            token = _active_dispatch.set(trace)
            interpretation_token = _active_interpretation.set(interpretation)
            try:
                # Compose the complete interpretation once per trace. Handler
                # implementations may still invoke other effectful operations.
                with (
                    core.set_current_trace(trace),
                    handler(defaults),
                    handler(interpretation),
                ):
                    out = fn(*inputs[0], **inputs[1])
            finally:
                _active_interpretation.reset(interpretation_token)
                _active_dispatch.reset(token)
            out = jax.tree.map(
                trace.unwrap,
                out,
                is_leaf=lambda x: (
                    isinstance(x, _InterceptTracer) or adapter.is_value(x)
                ),
            )
            with core.set_current_trace(parent):
                return jax.tree.map(adapter.normalize, out, is_leaf=adapter.is_value)

    return wrapped


__all__ = ["ValueAdapter", "intercept", "primitive_op", "represented_value"]
