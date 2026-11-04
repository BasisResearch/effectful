"""Interpretation-specific callable identities for JAX's callback caches."""

import functools

import jax

from effectful.internals.runtime import (
    EVAL_CACHE,
    _get_args,
    cache_get,
    cache_put,
    get_interpretation,
)
from effectful.internals.weak import AutoIdKeyDictionary
from effectful.ops.semantics import fwd
from effectful.ops.syntax import defop

_WRAPPERS = object()


@defop
def wrap_callable(fun):
    """Give a callback an identity specific to the current interpretation.

    An active effectful cache reuses wrappers for the same callable and
    interpretation. Without a cache, fresh wrappers prevent JAX from silently
    reusing another interpretation's callback trace. Execution remains under
    the ambient interpretation; this operation does not adapt derivative rules.
    """
    current = get_interpretation()
    store = EVAL_CACHE.get()
    # These contexts contain call-bound arguments/continuations, not a stable
    # policy. Do not retain their values (possibly tracers) in a shared cache.
    if fwd in current or _get_args in current:
        store = None
    wrappers = cache_get(store, _WRAPPERS, current) if store is not None else None
    if wrappers is not None and fun in wrappers:
        return wrappers[fun]

    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        return fun(*args, **kwargs)

    if store is not None:
        if wrappers is None:
            wrappers = AutoIdKeyDictionary()
            cache_put(store, _WRAPPERS, current, wrappers)
        wrappers[fun] = wrapped
    return wrapped


def wrap_arguments(args, kwargs):
    """Wrap callable operands of numerical APIs, leaving dtype classes alone."""
    return jax.tree.map(
        lambda x: wrap_callable(x) if callable(x) and not isinstance(x, type) else x,
        (args, kwargs),
    )
