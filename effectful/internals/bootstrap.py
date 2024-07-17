import dataclasses
import typing
from typing import Callable, Generic, Type, TypeVar

from typing_extensions import ParamSpec, TypeGuard, dataclass_transform

from effectful.ops.core import (
    Constant,
    Context,
    Interpretation,
    Operation,
    Symbol,
    Term,
    Variable,
)

from ..ops import core
from . import runtime


@runtime.weak_memoize
def base_define(m: Type[T]):  # | Callable[Q, T]) -> Operation[..., T]:
    if not typing.TYPE_CHECKING:
        if typing.get_origin(m) not in (m, None):
            return base_define(typing.get_origin(m))

    return _BaseOperation(dataclasses.dataclass)
