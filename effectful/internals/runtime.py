import contextlib
import dataclasses
import functools
import typing
import weakref
from typing import Callable, Generic, TypeVar

from typing_extensions import ParamSpec

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")
T_co = TypeVar("T_co", covariant=True)

if typing.TYPE_CHECKING:
    from ..ops.core import Interpretation


@dataclasses.dataclass
class Runtime(Generic[S, T]):
    interpretation: "Interpretation[S, T]"


@functools.lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    return Runtime(interpretation={})


def get_interpretation():
    return get_runtime().interpretation


def weak_memoize(f: Callable[[S], T]) -> Callable[[S], T]:
    """
    Memoize a one-argument function using a dictionary
    whose keys are weak references to the arguments.
    """

    cache: weakref.WeakKeyDictionary[S, T] = weakref.WeakKeyDictionary()

    @functools.wraps(f)
    def wrapper(x: S) -> T:
        try:
            return cache[x]
        except KeyError:
            result = f(x)
            cache[x] = result
            return result

    return wrapper


@contextlib.contextmanager
def interpreter(intp: "Interpretation", *, unset: bool = True):

    r = get_runtime()
    old_intp = r.interpretation
    try:
        new_intp = {
            op: intp[op] if op in intp else old_intp[op]
            for op in set(intp.keys()) | set(old_intp.keys())
        }
        old_intp, r.interpretation = r.interpretation, new_intp
        yield intp
    finally:
        if unset:
            r.interpretation = old_intp
        else:
            if len(list(old_intp.keys())) == 0 and len(list(intp.keys())) > 0:
                raise RuntimeError(f"Dangling interpretation on stack: {intp}")
