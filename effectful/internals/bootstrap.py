from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

from typing_extensions import ParamSpec

from effectful.internals.runtime import get_interpretation

_T = TypeVar("_T")
_V = TypeVar("_V")
_P = ParamSpec("_P")


class InjectedMeta(type, Generic[_P]):
    _con: "Optional[Operation[_P, InjectedMeta]]" = None

    @property
    def constructor(self) -> "Operation[_P, InjectedMeta]":
        if not self._con:
            self._con = object.__new__(Operation)
            self._con._default = lambda *a, **k: type.__call__(self, *a, *k)

        return self._con

    def __call__(self, *a: _P.args, **k: _P.kwargs) -> "InjectedMeta":
        return self.constructor(*a, **k)


class InjectedType(metaclass=InjectedMeta):
    pass


@dataclass(unsafe_hash=True)
class Operation(Generic[_P, _V], InjectedType):
    _default: Callable[_P, _V]

    def __init__(self, defl: Callable[_P, _V]):
        self._default = defl

    @property
    def default(self) -> Callable[_P, _V]:
        return self._default

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _V:
        return get_interpretation().get(self, self._default)(*args, **kwargs)
