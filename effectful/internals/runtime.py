import contextlib
import contextvars
import functools
import inspect
import typing
from collections.abc import Callable, Mapping

from effectful.internals.weak import AutoIdKeyDictionary, weak_memoize
from effectful.ops.types import Interpretation, Operation

type CacheEntry = AutoIdKeyDictionary[Interpretation, typing.Any]
type EvalCache = AutoIdKeyDictionary[typing.Any, CacheEntry]

EVAL_CACHE: contextvars.ContextVar[EvalCache | None] = contextvars.ContextVar(
    "EVAL_CACHE", default=None
)


INTERPRETATION: contextvars.ContextVar[Interpretation] = contextvars.ContextVar(
    "INTERPRETATION", default=typing.cast(Interpretation, {})
)


def get_interpretation():
    return INTERPRETATION.get()


@contextlib.contextmanager
def interpreter(intp: "Interpretation"):
    token = INTERPRETATION.set(intp)
    try:
        yield intp
    finally:
        INTERPRETATION.reset(token)


@contextlib.contextmanager
def cache(store: EvalCache | None = None):
    """Memoize evaluation under any interpretation for the duration of this block.

    Installs ``store``, or a fresh cache if none is given, and yields it so that a
    later block can reuse it::

        with cache() as store:
            ...
        with cache(store):
            ...

    :func:`effectful.ops.semantics.evaluate` installs one for the duration of a
    call when none is active, so a lone call is memoized internally whether or not
    a scope is open. Holding a scope is what shares that work *between* calls.
    """
    store = AutoIdKeyDictionary() if store is None else store
    token = EVAL_CACHE.set(store)
    try:
        yield store
    finally:
        EVAL_CACHE.reset(token)


def cache_get(
    store: EvalCache,
    expr: typing.Any,
    intp: "Interpretation",
    default: typing.Any = None,
) -> typing.Any:
    """Look ``expr`` up under ``intp``, returning ``default`` if it is not cached."""
    inner = store.get(expr)
    return default if inner is None else inner.get(intp, default)


def cache_put(
    store: EvalCache, expr: typing.Any, intp: "Interpretation", value: typing.Any
) -> None:
    """Record that ``expr`` evaluates to ``value`` under ``intp``."""
    inner = store.get(expr)
    if inner is None:
        # Same flavour as the outer store, so the inner map accepts the plain
        # ``dict`` interpretations that ``coproduct`` builds.
        inner = type(store)()
        store[expr] = inner
    inner[intp] = value


def copy_cache_entries(src, dst) -> None:
    """Copy everything cached for ``src`` onto ``dst``.

    :func:`effectful.ops.syntax._build_term` computes a node's type analysis on a
    throwaway term and then needs it attributed to the term it actually returns.
    """
    store = EVAL_CACHE.get()
    if store is None:
        return
    inner = store.get(src)
    if inner is not None:
        for intp, value in inner.items():
            cache_put(store, dst, intp, value)


@Operation.define
def _get_args() -> tuple[tuple, Mapping]:
    return ((), {})


@weak_memoize
def _restore_args[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    sig = inspect.signature(fn)
    if not sig.parameters:
        return fn

    @functools.wraps(fn)
    def _cont_wrapper(*a: P.args, **k: P.kwargs) -> T:
        a, k = (a, k) if a or k else _get_args()
        return fn(*a, **k)

    return _cont_wrapper


@weak_memoize
def _save_args[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    from effectful.ops.semantics import handler

    sig = inspect.signature(fn)
    if not sig.parameters:
        return fn

    @functools.wraps(fn)
    def _cont_wrapper(*a: P.args, **k: P.kwargs) -> T:
        with handler({_get_args: lambda: (a, k)}):
            return fn(*a, **k)

    return _cont_wrapper


def _set_prompt[**P, T](
    prompt: Operation[P, T], cont: Callable[P, T], body: Callable[P, T]
) -> Callable[P, T]:
    from effectful.ops.semantics import handler

    @functools.wraps(body)
    def bound_body(*a: P.args, **k: P.kwargs) -> T:
        next_cont = get_interpretation().get(prompt, prompt.__default_rule__)
        with handler({prompt: handler({prompt: next_cont})(cont)}):
            return body(*a, **k)

    return bound_body
