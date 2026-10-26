"""Named dimensions, indexing, and partial evaluation in jax._handlers."""

import jax

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, jax_getitem, sizesof
from effectful.ops.semantics import evaluate, fvsof, handler
from effectful.ops.syntax import defdata, defop, syntactic_eq
from effectful.ops.types import Term


def test_bind_dims():
    """Test bind_dims's handling of free variables and tensor shapes"""
    i, j, k = (
        defop(jax.Array, name="i"),
        defop(jax.Array, name="j"),
        defop(jax.Array, name="k"),
    )
    key = jax.random.PRNGKey(0)
    t = jax.random.normal(key, (2, 3, 4))

    # Test case 1: Converting all named dimensions to positional
    t_ijk = jax_getitem(t, (i(), j(), k()))
    assert fvsof(t_ijk) >= {i, j, k}

    t1 = bind_dims(t_ijk, i, j, k)
    assert not (fvsof(t1) & {i, j, k})
    assert t1.shape == (2, 3, 4)

    # Test case 2: Different ordering of dimensions
    t2 = bind_dims(t_ijk, k, j, i)
    assert not (fvsof(t1) & {i, j, k})
    assert t2.shape == (4, 3, 2)

    # Test case 3: Keeping some dimensions as free variables
    t3 = bind_dims(t_ijk, i)  # Convert only i to positional
    assert fvsof(t3) >= {j, k}  # j and k remain free
    assert isinstance(t3, Term)
    assert t3.shape == (2,)

    t4 = bind_dims(t_ijk, i, j)  # Convert i and j to positional
    assert fvsof(t4) >= {k} and not (fvsof(t4) & {i, j})  # only k remains free
    assert isinstance(t4, Term)
    assert t4.shape == (2, 3)

    # Test case 4: Empty order list keeps all variables free
    t5 = bind_dims(t_ijk)
    assert fvsof(t5) >= {i, j, k}  # All variables remain free
    assert isinstance(t5, Term)
    assert t5.shape == tuple()

    # Test case 5: Verify permuted tensors maintain correct relationships
    t_kji = jax_getitem(jnp.permute_dims(t, (2, 1, 0)), (k(), j(), i()))
    t6 = bind_dims(t_kji, i, j, k)
    t7 = bind_dims(t_ijk, i, j, k)
    assert jnp.allclose(t6, t7)

    # Test case 6: Mixed operations with free variables
    x = jnp.sin(t_ijk)  # Apply operation to indexed tensor
    x1 = bind_dims(x, i, j)  # Convert some dimensions
    assert fvsof(x1) >= {k}  # k remains free
    assert isinstance(x1, Term)
    assert x1.shape == (2, 3)

    # Test case 7: Multiple tensors sharing variables
    t2_ijk = jax_getitem(jax.random.normal(key, (2, 3, 4)), (i(), j(), k()))
    sum_t = t_ijk + t2_ijk
    sum1 = bind_dims(sum_t, i, j)
    assert fvsof(sum1) >= {k}  # k remains free
    assert isinstance(sum1, Term)
    assert sum1.shape == (2, 3)

    # Test case 8: Tensor term with non-sized free variables
    w = defop(jax.Array, name="w")
    t_ijk = jax_getitem(t, (i(), j(), k())) + w()
    t8 = bind_dims(t_ijk, i, j, k)
    assert fvsof(t8) >= {w}

    # Test case 9: Eliminate remaining free variables in result
    with handler({w: lambda: jnp.array(1.0)}):
        t9 = evaluate(t8)
    assert not (fvsof(t9) & {i, j, k, w})


def test_tpe_1():
    i, j = defop(jax.Array), defop(jax.Array)
    key = jax.random.PRNGKey(0)
    xval, y1_val, y2_val = (
        jax.random.normal(key, (2, 3)),
        jax.random.normal(key, (2)),
        jax.random.normal(key, (3)),
    )

    expected = xval + y1_val[..., None] + y2_val[None]

    x_ij = jax_getitem(xval, (i(), j()))
    x_plus_y1_ij = x_ij + jax_getitem(y1_val, (i(),))
    actual = x_plus_y1_ij + jax_getitem(y2_val, (j(),))

    assert actual.op == jax_getitem
    assert fvsof(actual) >= {i, j}
    assert actual.shape == ()
    assert actual.size == 1
    assert actual.ndim == 0

    assert (bind_dims(actual, i, j) == expected).all()


def test_tpe_2():
    key = jax.random.PRNGKey(0)
    xval = jax.random.normal(key, (2, 3))
    ival = jnp.arange(2)
    expected = jnp.sum(xval[ival, :], axis=0)

    j = defop(jax.Array)
    x_j = jax_getitem(xval, (ival, j()))

    assert x_j.shape == (2,)
    actual = jnp.sum(x_j, axis=0)

    assert actual.op == jax_getitem
    assert fvsof(actual) >= {j}
    assert actual.shape == ()
    assert actual.size == 1

    assert (bind_dims(actual, j) == expected).all()


def test_tpe_3():
    key = jax.random.PRNGKey(0)
    xval = jax.random.normal(key, (4, 2, 3))
    ival = jnp.arange(2)
    expected = jnp.sum(xval, axis=1)

    j, k = defop(jax.Array), defop(jax.Array)
    x_j = jax_getitem(xval, (k(), ival, j()))
    actual = jnp.sum(x_j, axis=0)

    assert actual.op == jax_getitem
    assert fvsof(actual) >= {j, k}
    assert actual.shape == ()
    assert actual.size == 1

    assert (bind_dims(actual, k, j) == expected).all()


def test_tpe_known_index():
    """Constant indexes are partially evaluated away."""
    i, j = defop(jax.Array, name="i"), defop(jax.Array, name="j")

    cases = [
        jax_getitem(jnp.ones((2, 3)), (i(), 1)),
        jax_getitem(jnp.ones((2, 3)), (0, i())),
        jax_getitem(jnp.ones((2, 3, 4)), (0, i(), 1)),
        jax_getitem(jnp.ones((2, 3, 4)), (0, i(), j())),
        jax_getitem(jnp.ones((2, 3, 4)), (i(), j(), 3)),
    ]

    for case_ in cases:
        assert all(isinstance(a, Term) for a in case_.args[1])
        assert not any(isinstance(a, int) for a in case_.args[1])


def test_tpe_constant_eval():
    """Constant indexes are partially evaluated away."""
    height, width = (
        defop(jax.Array, name="height"),
        defop(jax.Array, name="width"),
    )
    t = jnp.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
    A = jax_getitem(t, (height(), width()))

    layer = defop(jax.Array, name="layer")
    with handler(
        {
            height: lambda: layer() // jnp.array(3),
            width: lambda: layer() % jnp.array(3),
        }
    ):
        A_layer = evaluate(A)
    with handler({layer: lambda: jnp.array(2)}):
        A_final = evaluate(A_layer)

    assert not isinstance(A_final, Term)


def test_tpe_stack():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    xval = jax.random.normal(key1, (10, 5))
    yval = jax.random.normal(key2, (10, 5))

    i = defop(jax.Array)
    j = defop(jax.Array)
    x_ij = jax_getitem(xval, (i(), j()))
    y_ij = jax_getitem(yval, (i(), j()))
    actual = jnp.stack((x_ij, y_ij))
    assert actual.shape == (2,)
    assert (
        jnp.transpose(bind_dims(actual, i, j), [2, 0, 1]) == jnp.stack((xval, yval))
    ).all()


def test_jax_nested_getitem():
    t = jnp.ones((2, 3))
    i, j = defop(jax.Array), defop(jax.Array)
    t_i = jax_getitem(t, (i(),))

    t_ij = defdata(jax_getitem, t_i, (j(),))
    assert sizesof(t_ij) == {i: 2, j: 3}

    t_ij = jax_getitem(t_i, (j(),))
    assert sizesof(t_ij) == {i: 2, j: 3}


def test_sizesof_compound_index():
    """A compound index is not a named dimension.

    Only a bare call names a dimension. An index built out of one, like
    ``a() + 1``, names nothing -- neither the operation applied nor the
    variable underneath it. The entry still holds its place, so the names
    beside it are found against the right dimensions.
    """
    a, b = defop(jax.Array, name="a"), defop(jax.Array, name="b")

    assert sizesof(jax_getitem(jnp.ones((4, 5)), (a() + 1, b()))) == {b: 5}
    assert sizesof(jax_getitem(jnp.ones((4, 5)), (a() + 1, a() * 2))) == {}
    assert sizesof(jax_getitem(jnp.ones((2, 3)), (a() + 1, None, b()))) == {b: 3}
    assert sizesof(jax_getitem(jnp.ones((2, 3)), (None, a() + 1, b()))) == {b: 3}
    assert sizesof(jax_getitem(jnp.ones((2, 3, 4)), (a() + 1, ..., b()))) == {b: 4}

    # A bare named dimension is still reported, alongside a compound sibling.
    assert sizesof(jax_getitem(jnp.ones((4, 5)), (a(), b()))) == {a: 4, b: 5}


def test_sizesof_nested_getitem():
    """Sizes are found through an inner getitem, using its residual shape.

    ``sizesof`` computes that shape itself rather than reading it off a
    rebuilt term, so these pin it against the conditions ``_embed_array``
    builds an eager term under: a concrete array whose every term entry is a
    bare name.
    """
    a, b = defop(jax.Array, name="a"), defop(jax.Array, name="b")

    # A named dimension is consumed by the indexing; the rest stays.
    inner = jax_getitem(jnp.ones((2, 3, 4)), (a(), slice(None), slice(None)))
    assert sizesof(jax_getitem(inner, (b(), slice(None)))) == {a: 2, b: 3}

    inner = jax_getitem(jnp.ones((2, 3)), (slice(None), a()))
    assert sizesof(jax_getitem(inner, (b(),))) == {a: 3, b: 2}

    # A concrete index keeps its dimension.
    inner = jax_getitem(jnp.ones((4, 5)), (jnp.arange(4), a()))
    assert sizesof(jax_getitem(inner, (b(),))) == {a: 5, b: 4}

    # Unlike torch, an index is not required to be array-typed.
    n = defop(int, name="n")
    assert sizesof(jax_getitem(jnp.ones((4, 5)), (n(), a()))) == {n: 4, a: 5}


def test_sizesof_symbolic_key():
    """A key can be a term rather than a sequence of index entries.

    Indexing the result of an earlier indexing this way is what reaches the
    analysis: the inner term has a shape to read while being a term, which is
    what stops ``_embed_array`` from inspecting the key at all.
    """
    a = defop(jax.Array, name="a")
    key = defop(tuple, name="key")

    inner = jax_getitem(jnp.ones((2, 3, 4)), (a(), slice(None), slice(None)))
    assert tuple(inner.shape) == (3, 4)

    assert sizesof(jax_getitem(inner, key())) == {a: 2}


def test_jax_dimension_addition():
    """Test jax_getitem with dimension addition via None indexing."""
    i = defop(jax.Array, name="i")
    i2 = defop(i)

    # Basic case: indexing with slice and defop
    x = jax_getitem(jnp.eye(3), (i(), slice(None)))
    assert x.shape == (3,)
    assert fvsof(x) >= {i}

    # Multiple defops with None - this should work fine
    x2 = jax_getitem(jnp.eye(3), (i(), i2(), None))
    assert x2.shape == (1,)
    assert fvsof(x2) >= {i, i2}

    # The problematic case: indexing a Term with defop and None
    # This should work but may currently fail
    x3 = jax_getitem(x, (i2(), None))
    assert x3.shape == (1,)
    assert fvsof(x3) >= {i, i2}

    # Additional test cases for dimension addition
    # Test with multiple None dimensions
    x4 = jax_getitem(x, (i2(), None, None))
    assert x4.shape == (1, 1)
    assert fvsof(x4) >= {i, i2}

    # Test with slice and None
    x5 = jax_getitem(x, (slice(None), None))
    assert x5.shape == (3, 1)
    assert fvsof(x5) >= {i}

    # Test with Ellipsis and None
    x6 = jax_getitem(x, (..., None))
    assert x6.shape == (3, 1)
    assert fvsof(x6) >= {i}

    # Test nested indexing with dimension addition
    base = jnp.ones((2, 3, 4))
    y = jax_getitem(base, (i(), slice(None), slice(None)))
    assert y.shape == (3, 4)
    assert fvsof(y) >= {i}

    # Index the result with another defop and add dimension
    y2 = jax_getitem(y, (i2(), slice(None), None))
    assert y2.shape == (4, 1)
    assert fvsof(y2) >= {i, i2}


def test_array_eq():
    integer = jnp.array([[1, 2], [3, 4]])
    equal = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    unequal = jnp.array([[1, 2], [3, 5]])

    assert syntactic_eq(integer, equal)
    assert syntactic_eq(equal, integer)
    assert not syntactic_eq(integer, unequal)
    assert not syntactic_eq(integer, jnp.array([1, 2, 3, 4]))
    assert not syntactic_eq(jnp.empty((0,)), jnp.empty((0, 2)))
    assert not syntactic_eq(integer, [[1, 2], [3, 4]])
    assert not syntactic_eq(jnp.array([float("nan")]), jnp.array([float("nan")]))
    assert syntactic_eq({"array": integer}, {"array": equal})

    x = defop(jax.Array)()
    assert syntactic_eq(x + integer, x + equal)
    assert not syntactic_eq(x + integer, x + unequal)
    assert not syntactic_eq(x + integer, x + jnp.array([1, 2, 3, 4]))
