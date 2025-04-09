import jax.numpy

from .handlers import _register_jax_op

_no_overload = ["array", "asarray"]

for name, op in jax.numpy.__dict__.items():
    if callable(op):
        globals()[name] = _register_jax_op(op)

for name in _no_overload:
    globals()[name] = jax.numpy.__dict__[name]

logsumexp = _register_jax_op(jax.scipy.special.logsumexp)
