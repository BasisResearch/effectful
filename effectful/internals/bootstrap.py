from dataclasses import dataclass
from typing import Callable, Generic, TypeVar
from weakref import WeakKeyDictionary

from typing_extensions import ParamSpec, dataclass_transform

from effectful.internals.runtime import get_interpretation

_T = TypeVar("_T")
_V = TypeVar("_V")
_P = ParamSpec("_P")


class InjectedMeta(type):
    def __call__(self: type[_T], *a, **k) -> _T:
        return define(self)(*a, **k)


class InjectedType(metaclass=InjectedMeta):
    pass


@dataclass(frozen=True, repr=False)
class Operation(Generic[_P, _V], InjectedType):
    default: Callable[_P, _V]

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _V:
        return get_interpretation().get(self, self.default)(*args, **kwargs)

    def __repr__(self):
        if hasattr(self.default, "__name__"):
            return f"<Operation {self.default.__name__} at {hex(id(self))}>"
        else:
            return f"<Operation at {hex(id(self))}>"


def _blank_con(ty: type[_T]) -> Callable[_P, _T]:
    sup = type(ty)
    if issubclass(sup, InjectedMeta):
        sup = type

    def make(*a: _P.args, **k: _P.kwargs) -> _T:
        return sup.__call__(ty, *a, **k)

    make.__name__ = f"define_{ty.__name__}"
    return make


def _blank_op(default: Callable[_P, _T]) -> Operation[_P, _T]:
    return _blank_con(Operation)(default)


_CONSTRUCTOR_MAP: WeakKeyDictionary[type, Operation] = WeakKeyDictionary()
_CONSTRUCTOR_MAP[Operation] = _blank_op(_blank_op)


def define(ty: type[_T]) -> "Operation[..., _T]":
    if ty not in _CONSTRUCTOR_MAP:
        _CONSTRUCTOR_MAP[ty] = Operation(_blank_con(ty))

    return _CONSTRUCTOR_MAP[ty]


@dataclass_transform()
class InjectedDataclass(InjectedType):
    def __init_subclass__(sub, **kwargs):
        dataclass(sub, unsafe_hash=True)
