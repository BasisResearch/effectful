import dataclasses
import functools
import typing
import weakref
from typing import Callable, Generic, TypeVar

from typing_extensions import ParamSpec

Q = ParamSpec("Q")
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


def swap_interpretation(intp: "Interpretation[S, V]") -> "Interpretation[S, T]":
    old_intp = get_runtime().interpretation
    get_runtime().interpretation = intp
    return old_intp


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
