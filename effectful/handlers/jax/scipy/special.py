from typing import TYPE_CHECKING

import jax.scipy.special

from effectful.handlers.jax._handlers import _reduce_named, _register_jax_op
from effectful.ops.semantics import handler

logsumexp = _register_jax_op(jax.scipy.special.logsumexp)
logsumexp = handler({logsumexp: _reduce_named})(logsumexp)

# Tell mypy about our wrapped functions.
if TYPE_CHECKING:
    from jax.scipy.special import logsumexp  # noqa: F401
