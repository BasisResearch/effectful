import functools
from typing import Annotated, Any, TypeVar

import tree

from effectful.ops.syntax import Scoped, defop
from effectful.ops.types import Operation

T = TypeVar("T")
A = TypeVar("A")
B = TypeVar("B")


@functools.singledispatch
def _bind_dims(value, *names: Operation[[], Any]):
    if tree.is_nested(value):
        return tree.map_structure(lambda v: _bind_dims(v, *names), value)
    raise NotImplementedError


@defop
def bind_dims(
    value: Annotated[T, Scoped[A | B]],
    *names: Annotated[Operation[[], Any], Scoped[B]],
) -> Annotated[T, Scoped[A]]:
    """Convert named dimensions to positional dimensions.

    :param t: A tensor.
    :type t: T
    :param args: Named dimensions to convert to positional dimensions.
                  These positional dimensions will appear at the beginning of the
                  shape.
    :type args: Operation[[], torch.Tensor]
    :return: A tensor with the named dimensions in ``args`` converted to positional dimensions.

    **Example usage**:

    >>> a, b = defop(torch.Tensor, name='a'), defop(torch.Tensor, name='b')
    >>> t = torch.ones(2, 3)
    >>> bind_dims(t[a(), b()], b, a).shape
    torch.Size([3, 2])
    """
    return _bind_dims(value, *names)


@functools.singledispatch
def _unbind_dims(value, *names: Operation[[], Any]):
    if tree.is_nested(value):
        return tree.map_structure(lambda v: _unbind_dims(v, *names), value)
    raise NotImplementedError


@defop
def unbind_dims(
    value: Annotated[T, Scoped[A | B]],
    *names: Annotated[Operation[[], Any], Scoped[B]],
) -> Annotated[T, Scoped[A | B]]:
    return _unbind_dims(value, *names)
