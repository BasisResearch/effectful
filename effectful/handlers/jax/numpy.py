import jax.numpy

from .handlers import _register_jax_op

for name, op in jax.numpy.__dict__.items():
    if callable(op):
        globals()[name] = _register_jax_op(op)
