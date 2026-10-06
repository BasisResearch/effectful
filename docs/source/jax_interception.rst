Experimental ordinary JAX interception
=====================================

``effectful.handlers.jax.intercept`` interprets ordinary JAX primitives through
effectful operations, without replacing ``Primitive.bind`` or requiring callers
to use the effectful NumPy namespace::

    import jax
    from effectful.handlers.jax import intercept, primitive_op
    from effectful.ops.semantics import fwd

    def observe_dot(*operands, **parameters):
        return fwd()

    transformed = intercept(
        lambda a, b: a @ b,
        interpretation={primitive_op(jax.lax.dot_general_p): observe_dot},
    )
    compiled = jax.jit(transformed)

``primitive_op`` returns a stable operation for each primitive. Operands are
positional; primitive parameters are keywords. ``fwd`` composes with the
original parent-trace binding. Multiple-result primitives return lists.
Interpretations are snapshotted when the callable is constructed. Reuse the
resulting callable: ordinary ``jax.jit`` caching keys on this callable identity,
static arguments, logical argument avals and representation pytree layouts;
numerical buffer values stay dynamic.

Nested JIT programs (including previously compiled ones) are recursively
interpreted. Scan, while and conditional programs are restaged over buffer
pytrees, with conservative representation joins. Reverse scans, mapped inputs
and captured constants are supported. Opaque custom differentiation boundaries
bind on the parent trace with their original derivative rules. An adapter may
materialize represented operands at an unsupported boundary.

Represented values
------------------

Supply ``value_adapter=adapter`` for values whose logical shape/dtype differs
from their buffers. Register them as JAX pytrees: numerical arrays are leaves,
layout is static auxiliary metadata. Implement ``is_value``, ``aval``,
``materialize``, ``normalize``, ``join`` and ``coerce`` as documented by
``ValueAdapter``. Joining must preserve meaning and reach a bounded fixed
representation for loop carries. Abstract templates can contain
``ShapeDtypeStruct`` leaves. Optionally implement ``stacked(value, length)``
when scan outputs require a static shape update after stacking numerical leaves.
Optionally implement ``mapped(value)`` to prepare a value whose logical leading
axis is removed by scan; it may widen that operand if its buffers cannot support
the resulting logical value.
The bridge contains no matrix-specific rules.

Compatibility and boundaries
----------------------------

This experimental interpreter supports JAX 0.9.1 and 0.9.2. Private trace APIs
are imported through ``_compat.py``. Distributed sharding and donated JIT inputs
raise explicit errors; they are not supported. Existing compiled executables
are unaffected when called directly. Semantic Python patches must be installed
while staging a function; previously staged custom calls keep their original
derivative semantics unless a domain-specific handler recognizes them.
