"""Whole-pytree custom arrays with logical avals and dense baseline fallback."""

from collections.abc import Callable

import jax

from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import defop

from ._compat import core
from ._intercept import (
    _InterceptTrace,
    _InterceptTracer,
    _JaxInnerPrimitiveOperation,
    _new_trace,
    _parent_trace,
    intercept,
)


@defop
def is_custom_array(value) -> bool:
    """Whether a registered pytree represents one logical array operand."""
    return False


@defop
def aval(value) -> core.AbstractValue:
    """Return the logical aval, which may differ from the storage buffers' avals."""
    return core.typeof(value)


@defop
def to_jax_array(value):
    """Convert a custom array to its dense value; leave ordinary operands alone."""
    if is_custom_array(value):
        raise NotImplementedError(
            f"No conversion from {type(value).__name__} to jax.Array available"
        )
    return value


class _CustomArrayTracer(_InterceptTracer):
    @property
    def aval(self):
        return aval(self.value)


class _CustomArrayTrace(_InterceptTrace):
    def _wrap(self, value):
        return _CustomArrayTracer(self, value)

    def _wrap_inputs(self, args, kwargs):
        def wrap(value):
            if is_custom_array(value) or isinstance(value, (jax.Array, core.Tracer)):
                return self._wrap(value)
            return value

        return jax.tree.map(wrap, (args, kwargs), is_leaf=is_custom_array)


def _dense_apply(op, /, *args, **params):
    # Only unhandled inner operations reach apply. Convert on the parent trace
    # so densification itself does not recursively enter interception.
    with core.set_current_trace(_parent_trace()):
        return fwd(op, *(to_jax_array(x) for x in args), **params)


def custom_arrays[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    """Intercept whole custom arrays, densifying only at baseline primitives."""
    return handler(
        {
            _new_trace: _CustomArrayTrace,
            _JaxInnerPrimitiveOperation.__apply__: _dense_apply,
        }
    )(intercept(fn))
