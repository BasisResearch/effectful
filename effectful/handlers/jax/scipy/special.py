from typing import TYPE_CHECKING

import jax.scipy.special

from effectful.handlers.jax._handlers import _register_jax_op

logsumexp = _register_jax_op(jax.scipy.special.logsumexp)

# Tell mypy about our wrapped functions.
if TYPE_CHECKING:
    from jax.scipy.special import logsumexp  # noqa: F401
