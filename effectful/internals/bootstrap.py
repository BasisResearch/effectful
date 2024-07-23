from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar, final

from typing_extensions import ParamSpec, Self, dataclass_transform

from effectful.internals.runtime import get_interpretation

_T = TypeVar("_T")
_V = TypeVar("_V")
_P = ParamSpec("_P")


class InjectedType(Generic[_P], ABC):
    _constructor_cache: Optional["Operation[_P, Self]"] = None

    @classmethod
    @abstractmethod
    def constructor(cls) -> "Operation[_P, Self]":
        raise NotImplementedError

    def __new__(cls, *args: _P.args, **kwargs: _P.kwargs) -> Self:
        if cls._constructor_cache:
            return cls._constructor_cache(*args, **kwargs)
        else:
            cls._constructor_cache = cls.constructor()
            return cls._constructor_cache(*args, **kwargs)


class Operation(Generic[_P, _V], InjectedType[Callable[_P, _V]]):
    default: Callable[_P, _V]

    def __hash__(self):
        return id(self)

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _V:
        return get_interpretation().get(self, self.default)(*args, **kwargs)

    @classmethod
    def constructor(cls) -> "Operation[[Callable[_P, _V]], Operation[_P, _V]]":
        return _OperationConstructor()


@final
class _OperationConstructor(
    Generic[_P, _V], Operation[[Callable[_P, _V]], Operation[_P, _V]]
):
    _me: Optional["_OperationConstructor[_P, _V]"] = None

    def __hash__(self):
        # This type is a singleton, so no other instances should exist, so hash collisions are impossible
        return 1

    def __new__(cls) -> Self:
        def new_operation(cl: Callable[_P, _V]) -> Operation[_P, _V]:
            ob = object.__new__(Operation)
            ob.default = cl
            return ob

        if cls._me is None:
            cls._me = object.__new__(cls)
            cls._me.default = new_operation
            return cls._me

        return cls._me

    def __init__(self):
        self.default = self._me.default


@dataclass_transform()
def define(ty: type) -> type[InjectedType]:
    ty = dataclass(ty)

    class NewClass(ty, InjectedType):
        @classmethod
        def constructor(cls) -> "Operation[..., Self]":
            return Operation(ty)

    NewClass.__name__ = ty.__name__
    NewClass.__doc__ = ty.__doc__
    NewClass.__parameters__ = getattr(ty, "__parameters__", ())
    NewClass.__wrapped__ = ty

    return NewClass
