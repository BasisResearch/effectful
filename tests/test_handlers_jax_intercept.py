import jax
import jax.numpy as jnp
import numpy as np

from effectful.handlers.jax import intercept, primitive_op
from effectful.ops.semantics import fwd


def test_nested_warmed_jit_and_handler_composition():
    seen = []

    def observe(*args, **params):
        seen.append(params["dimension_numbers"])
        return fwd() + 1

    original = jax.jit(lambda a, b: a @ b)
    a, b = jnp.eye(3), jnp.arange(3.0)
    original(a, b).block_until_ready()
    transformed = intercept(
        original, interpretation={primitive_op(jax.lax.dot_general_p): observe}
    )
    np.testing.assert_allclose(jax.jit(transformed)(a, b), b + 1)
    assert seen
    np.testing.assert_allclose(original(a, b), b)
    assert primitive_op(jax.lax.dot_general_p) is primitive_op(jax.lax.dot_general_p)


def test_scan_cond_reverse_and_grad():
    def fn(xs, reverse):
        def body(c, x):
            new = jax.lax.cond(x > 1, lambda: c + x, lambda: c - x)
            return new, new

        return jax.lax.scan(body, jnp.zeros(()), xs, reverse=reverse)[1]

    x = jnp.arange(5.0)
    for reverse in (False, True):

        def original(x):
            return fn(x, reverse)

        transformed = intercept(original, interpretation={})
        np.testing.assert_allclose(jax.jit(transformed)(x), original(x))
        np.testing.assert_allclose(
            jax.grad(lambda x: transformed(x).sum())(x),
            jax.grad(lambda x: original(x).sum())(x),
        )


def test_custom_derivative_is_preserved():
    @jax.custom_jvp
    def fn(x):
        return x * x

    @fn.defjvp
    def jvp(p, t):
        return fn(p[0]), 3 * p[0] * t[0]

    transformed = intercept(fn, interpretation={})
    assert jax.grad(jax.jit(transformed))(2.0) == 6.0


def test_custom_vjp_and_cached_jit_values_remain_dynamic():
    @jax.custom_vjp
    def custom(x):
        return x * x

    def forward(x):
        return custom(x), x

    def backward(x, g):
        return (5 * x * g,)

    custom.defvjp(forward, backward)
    fn = jax.jit(intercept(custom, interpretation={}))
    assert jax.grad(fn)(2.0) == 10.0
    assert jax.grad(fn)(3.0) == 15.0
    assert fn(2.0) == 4.0
    assert fn(3.0) == 9.0
    assert fn._cache_size() == 1


def test_scan_captured_constants_and_mapped_numerical_values():
    def original(xs, rate):
        return jax.lax.scan(lambda c, x: (c * rate + x, c), 0.0, xs)[0]

    transformed = jax.jit(intercept(original, interpretation={}))
    xs = jnp.arange(5.0)
    for rate in (0.4, 0.8):
        np.testing.assert_allclose(transformed(xs, rate), original(xs, rate))


def test_control_flow_operation_composes_with_recursive_interpretation():
    seen = []

    def observe(*args, **params):
        seen.append(params["reverse"])
        return fwd()

    fn = intercept(
        lambda xs: jax.lax.scan(lambda c, x: (c + x, c), 0.0, xs)[0],
        interpretation={primitive_op(jax.lax.scan_p): observe},
    )
    np.testing.assert_allclose(jax.jit(fn)(jnp.arange(5.0)), 10.0)
    assert seen == [False]


def test_while_and_vmap():
    def original(x):
        return jax.lax.while_loop(
            lambda v: v[0] < 4, lambda v: (v[0] + 1, v[1] + x), (0, 0.0)
        )[1]

    transformed = intercept(original, interpretation={})
    np.testing.assert_allclose(
        jax.jit(jax.vmap(transformed))(jnp.arange(3.0)), jnp.arange(3.0) * 4
    )


def test_represented_custom_derivative_materializes_on_parent_trace():
    from effectful.handlers.jax._compat import core

    @jax.tree_util.register_pytree_node_class
    class Box:
        def __init__(self, buffer):
            self.buffer = buffer

        def tree_flatten(self):
            return (self.buffer,), None

        @classmethod
        def tree_unflatten(cls, aux, children):
            return cls(children[0])

    class Adapter:
        def is_value(self, x):
            return isinstance(x, Box)

        def aval(self, x):
            return (
                core.ShapedArray(x.buffer.shape, x.buffer.dtype)
                if isinstance(x, Box)
                else core.typeof(x)
            )

        def materialize(self, x):
            return 2 * x.buffer if isinstance(x, Box) else x

        def normalize(self, x):
            return x

        def join(self, a, b):
            return b

        def coerce(self, x, template):
            return x

    @jax.custom_vjp
    def fn(x):
        return jnp.sum(x**2)

    def forward(x):
        return fn(x), x

    def backward(x, g):
        return (4 * x * g,)

    fn.defvjp(forward, backward)
    transformed = intercept(fn, interpretation={}, value_adapter=Adapter())
    x = jnp.array([2.0, 3.0])
    np.testing.assert_allclose(jax.grad(lambda x: transformed(Box(x)))(x), 16 * x)


def test_child_program_reuse_and_captured_constants_are_dynamic():
    seen = []

    def observe(*args, **params):
        seen.append(params["dimension_numbers"])
        return fwd()

    def original(xs, weights):
        def body(c, x):
            new = c + weights @ x
            return new, new

        return jax.lax.scan(body, jnp.zeros(2), xs)[1]

    fn = jax.jit(
        intercept(
            original, interpretation={primitive_op(jax.lax.dot_general_p): observe}
        )
    )
    xs = jnp.arange(8.0, dtype=float).reshape(4, 2)
    with jax.check_tracer_leaks():
        for weights in (jnp.eye(2), jnp.eye(2) * 2):
            np.testing.assert_allclose(fn(xs, weights), original(xs, weights))
    assert len(seen) == 1
    assert fn._cache_size() == 1


def test_handlers_can_invoke_other_primitive_operations():
    dot = primitive_op(jax.lax.dot_general_p)
    add = primitive_op(jax.lax.add_p)

    def dot_handler(*args, **params):
        value = fwd()
        return add(value, jnp.ones_like(value))

    def add_handler(*args, **params):
        return fwd() * 2

    fn = jax.jit(
        intercept(
            lambda a, x: a @ x, interpretation={dot: dot_handler, add: add_handler}
        )
    )
    np.testing.assert_allclose(fn(jnp.eye(2), jnp.ones(2)), 4)


def test_primitive_keyword_parameters_are_not_shadowed():
    from effectful.handlers.jax._compat import core

    primitive = core.Primitive("keyword_parameter")
    primitive.def_impl(lambda x, *, _primitive: x + _primitive)
    primitive.def_abstract_eval(lambda x, *, _primitive: x)

    def observe(*args, **params):
        return fwd()

    fn = intercept(
        lambda x: primitive.bind(x, _primitive=3),
        interpretation={primitive_op(primitive): observe},
    )
    np.testing.assert_allclose(fn(jnp.array(2.0)), 5.0)


def test_nested_interceptions_select_each_traces_interpretation():
    op = primitive_op(jax.lax.add_p)
    inner = intercept(lambda x: x + 1, interpretation={op: lambda *a, **k: fwd() * 2})
    outer = intercept(inner, interpretation={op: lambda *a, **k: fwd() + 3})
    for fn in (outer, jax.jit(outer)):
        np.testing.assert_allclose(fn(jnp.array(0.0)), 8)
        np.testing.assert_allclose(fn(jnp.array(2.0)), 12)
        np.testing.assert_allclose(jax.grad(fn)(jnp.array(2.0)), 2)
