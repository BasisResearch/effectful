import functools
import inspect
import typing
from collections.abc import Callable
from typing import Concatenate

from effectful.ops.types import Operation, _CustomSingleDispatchCallable


class _ClassMethodOperation[**P, S, T]:
    def __init__(self, default):
        self._default = default
        self._operation = {}

    def __get__(self, instance, owner: type[S]) -> Callable[P, T]:
        op = self._operation.get(owner, None)
        if op is None:
            func = self._default.__func__
            op = Operation.define(functools.partial(func, owner), name=func.__name__)
            self._operation[owner] = op
        return op


class _StaticMethodOperation[**P, S, T](Operation[P, T]):
    def __init__(self, default: staticmethod, **kwargs):
        super().__init__(default=default.__func__, **kwargs)

    def __get__(self, instance: S, owner: type[S] | None = None) -> Callable[P, T]:
        return self


# class _PropertyOperation[S, T](Operation[[S], T]):
#     def __init__(self, default: property, **kwargs):
#         assert not default.fset, "property with setter is not supported"
#         assert not default.fdel, "property with deleter is not supported"
#         super().__init__(default=typing.cast(Callable[[S], T], default.fget), **kwargs)

#     @typing.overload
#     def __get__(
#         self, instance: None, owner: type[S] | None = None
#     ) -> "_PropertyOperation[S, T]": ...

#     @typing.overload
#     def __get__(self, instance: S, owner: type[S] | None = None) -> T: ...

#     def __get__(self, instance, owner: type[S] | None = None):
#         if instance is not None:
#             return self(instance)
#         else:
#             return self


class _SingleDispatchMethodOperation[**P, S, T]:
    __default__: Callable[Concatenate[S, P], T]

    def __init__(self, default: functools.singledispatchmethod, **kwargs):
        if isinstance(default.func, classmethod):
            raise NotImplementedError("Operations as classmethod are not yet supported")

        @functools.wraps(default.func)
        def _wrapper(obj: S, *args: P.args, **kwargs: P.kwargs) -> T:
            return default.__get__(obj)(*args, **kwargs)

        self._registry: functools.singledispatchmethod = default
        super().__init__(_wrapper, **kwargs)

    @typing.overload
    def __get__(
        self, instance: None, owner: type[S] | None = None
    ) -> "_SingleDispatchMethodOperation[P, S, T]": ...

    @typing.overload
    def __get__(self, instance: S, owner: type[S] | None = None) -> Callable[P, T]: ...

    def __get__(self, instance, owner: type[S] | None = None):
        if instance is not None:
            return functools.partial(self, instance)
        else:
            return self

    @property
    def register(self):
        return self._registry.register

    @property
    def __isabstractmethod__(self):
        return self._registry.__isabstractmethod__


class _SingleDispatchOperation[**P, S, T](Operation[Concatenate[S, P], T]):
    __default__: "functools._SingleDispatchCallable[T]"

    @property
    def register(self):
        return self.__default__.register

    @property
    def dispatch(self):
        return self.__default__.dispatch


class _CustomSingleDispatchOperation[**P, **Q, S, T](Operation[P, T]):
    _default: _CustomSingleDispatchCallable[P, Q, S, T]

    def __init__(self, default: _CustomSingleDispatchCallable[P, Q, S, T], **kwargs):
        super().__init__(default, **kwargs)
        self.__signature__ = inspect.signature(functools.partial(default.func, None))  # type: ignore

    @property
    def dispatch(self):
        return self._registry.dispatch

    @property
    def register(self):
        return self._registry.register
