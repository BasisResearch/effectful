"""Integration of the public JAX package with Effectful evaluation."""

import effectful.handlers.jax.numpy as jnp
from effectful.ops.semantics import evaluate
from effectful.ops.syntax import syntactic_eq


def test_jax_rotation():
    import jax.scipy.spatial.transform

    x = jax.scipy.spatial.transform.Rotation.from_rotvec(jnp.array([1, 2, 3]))
    y = evaluate(x)
    assert syntactic_eq(x, y)
