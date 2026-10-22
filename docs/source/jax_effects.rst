High-level JAX effects
======================

Use the mirrored APIs in ``effectful.handlers.jax`` to select numerical
algorithms with ordinary Effectful handlers. These APIs do not require a custom
JAX trace or replaying jaxprs::

    import jax.numpy as raw_numpy
    from effectful.handlers import jax
    from effectful.handlers.jax import numpy
    from effectful.ops.semantics import fwd, handler

    @jax.jit
    def compute(x):
        return numpy.add(x, 1)

    x = raw_numpy.array(2.0)
    assert compute(x) == 3
    with handler({numpy.add: lambda *a, **k: fwd() * 2}):
        assert compute(x) == 6
    assert compute(x) == 3

Mirrored ``numpy``, ``numpy.linalg``, ``lax``, and ``scipy.linalg`` functions are
stable effects. Handlers receive public API operands and keywords, not primitive
parameters. ``fwd`` delegates to the next handler or the original JAX algorithm.
Calls made by a handler can themselves use mirrored effects.

All operations that should be interpreted must go through these mirrors.
Operators on ordinary JAX arrays and calls to the original JAX modules are not
intercepted. Use explicit mirrored functions, or implement custom-array operator
methods that delegate to them. Original JAX is appropriate inside an intentional
baseline kernel or a storage conversion.

Interpretation-aware compilation
-------------------------------

The mirrored ``jit`` retains the original Python callable and supplies a hidden
static interpretation key to JAX. Changing the interpretation retraces that
callable, including its high-level effects. Nested mirrored JIT functions also
specialize on the interpretation, even if they have already been compiled under
another one.

Repeated calls under the same interpretation and argument signature reuse
compilation. Numerical values remain dynamic. Static argument options retain
their public positional indices; decorator options and ``lower`` are supported.
Existing named-dimension result handling is preserved.

Handlers must have only benign side effects: logging and tracing counters may
observe Python executions, but numerical results must not depend on how often
tracing, abstract planning, or cache hits occur. Changing handler configuration
that changes semantics requires a changed interpretation. There is no automatic
invalidation for mutable closure state or equivalence analysis of separately
constructed handler contexts.

Functions compiled outside mirrored JIT, cached unmirrored aliases, and AOT
executables do not gain this retracing behavior. Replaying an old numerical
jaxpr cannot recover high-level effects that were expanded during source tracing.

Custom arrays and dense fallback
--------------------------------

Register custom arrays as ordinary JAX pytrees. Data buffers are leaves;
representation metadata is auxiliary data. Provide handlers for
``is_custom_array`` and ``to_jax_array`` from ``effectful.handlers.jax``.

Recognition treats a custom array as one logical operand during handler dispatch
and fallback conversion. JAX still sees its physical pytree when compiling or
differentiating the selected algorithm. No logical tracer shell or custom aval
registration is required.

Specialized handlers receive whole custom arrays and may return custom arrays.
At the numerical operation's default implementation, unhandled custom operands
are converted to ordinary arrays with ``to_jax_array``. Ordinary operands pass
through unchanged. A missing conversion raises ``NotImplementedError``.
Conversion is differentiable with respect to the original floating-point buffers.
Handle ``to_jax_array`` differently to prohibit or report dense fallbacks.

Control-flow joins
------------------

Mirrored ``lax.cond``, ``switch``, ``while_loop``, and ``scan`` wrap their
callbacks, then delegate to JAX. Representation negotiation is opt-in::

    from effectful.handlers.jax import CustomArrayControlFlow

    with handler(CustomArrayControlFlow()):
        # Install representation-specific recognition/upcast handlers too.
        ...

This interpretation uses ``jax.eval_shape`` to plan physical representations,
then forwards converted branches and carries to the baseline effects.

Callback wrapping is independent of this interpretation. Higher-order defaults
call the ``wrap_callable`` effect before passing functions to JAX. An active
``effectful.ops.semantics.cache()`` scope reuses a wrapper for the same callable
and interpretation, allowing JAX to reuse its traces. A different interpretation
gets a different wrapper; without a cache, wrappers are fresh.

Handle ``upcast(left, right)`` to select one smart constructor accepting either
abstract representation witness. The constructor must preserve meaning and
produce a common physical layout. Singledispatch constructors on custom-array
types are convenient. They must operate on concrete or tracer-backed buffers and
must not capture either witness's numerical buffers.

``upcast_many`` composes conversions when successive alternatives widen the
shared representation. Custom arrays are treated as logical leaves while joins
are negotiated; their buffer layouts need not initially match.

Conditionals apply conversions to their branch outputs. While and scan convert
the initial carry and body outputs after one widening step; JAX validates that
the resulting physical layout is stable. Zero-iteration loops and empty scans
are supported. General custom-array scan inputs, stacked custom outputs, and
custom-array batching semantics are not yet supported.

Differentiation
---------------

Mirrored ``grad``, ``value_and_grad``, ``jvp``, ``vjp``, and related transforms
use JAX's normal buffer-level AD. Sparse gradients are with respect to stored
nonzeros; integer coordinates and static metadata are not differentiable.

Handler implementations can define representation-aware ``custom_jvp`` or
``custom_vjp`` rules on registered pytrees. Compatible sparse primal/tangent
outputs remain sparse. Custom derivative boundaries are not blanket-densified.

Arbitrary logical derivative rules are not adapted to sparse storage. A rule
that returns an incompatible tangent/cotangent layout fails JAX's normal
validation. Consumers must supply representation-aware rules or deliberately
choose a dense implementation at the relevant semantic boundary.

Implementation layout
---------------------

``_custom_array.py`` defines representation effects, shared upcasting, and the
``CustomArrayControlFlow`` interpretation for branch and carry negotiation.
``_transforms.py`` implements interpretation-aware JIT and transform mirrors.
``_callables.py`` supplies interpretation-scoped callback identities.
``lax/_control_flow.py`` wraps callbacks and delegates to original JAX.
``_handlers.py`` retains numerical operation registration and named dimensions.

The earlier primitive-interception prototype has been removed. There is no
``intercept`` decorator, primitive-operation registry, or jaxpr replay path;
source code must call the mirrored APIs.
