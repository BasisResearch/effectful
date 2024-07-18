import dataclasses
import typing
from typing import Type, TypeVar

from typing_extensions import dataclass_transform

from . import runtime

T = TypeVar("T")


if typing.TYPE_CHECKING:
    from ..ops.core import Operation
else:

    class _overloadmeta(type):

        def __getitem__(cls, item):
            try:
                return cls.__class_getitem__(item)
            except TypeError as e:
                if e.args[0].endswith("is not a generic class"):
                    return cls  # TODO
                else:
                    raise

        def __or__(cls, other):
            return typing.Union[cls, other]

        def __ror__(cls, other):
            return typing.Union[other, cls]

        def __call__(cls, *args, **kwargs):
            return cls.__cons_op__(*args, **kwargs)


@dataclass_transform()
@runtime.weak_memoize
def base_define(m: Type[T]) -> "Operation[..., T]":
    if typing.TYPE_CHECKING:
        return m  # type: ignore
    else:
        m = typing.get_origin(m) if typing.get_origin(m) is not None else m

        try:
            from ..ops.core import Operation

            if issubclass(m, Operation):
                return (
                    m
                    if dataclasses.is_dataclass(m)
                    else dataclasses.dataclass(unsafe_hash=True)(m)
                )
            else:
                cons_op = base_define(Operation)(m)
                return _overloadmeta(
                    m.__name__,
                    (dataclasses.dataclass(unsafe_hash=True)(m),),
                    {"__cons_op__": staticmethod(cons_op)},
                )
        except ImportError:
            return dataclasses.dataclass(unsafe_hash=True)(m)
