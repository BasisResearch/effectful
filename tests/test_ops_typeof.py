from collections.abc import Callable
from typing import Annotated, Any, Generic, TypeVar, Union

import pytest

from effectful.ops.semantics import typeof
from effectful.ops.syntax import Scoped, defop
from effectful.ops.types import Operation


def test_typeof_basic():
    """Test typeof with basic operations that have simple return types."""

    @defop
    def add(x: int, y: int) -> int:
        raise NotImplementedError

    @defop
    def is_positive(x: int) -> bool:
        raise NotImplementedError

    @defop
    def get_name() -> str:
        raise NotImplementedError

    assert typeof(add(1, 2)) is int
    assert typeof(is_positive(5)) is bool
    assert typeof(get_name()) is str


def test_typeof_nested():
    """Test typeof with nested operations."""

    @defop
    def add(x: int, y: int) -> int:
        raise NotImplementedError

    @defop
    def multiply(x: int, y: int) -> int:
        raise NotImplementedError

    @defop
    def is_even(x: int) -> bool:
        raise NotImplementedError

    assert typeof(add(multiply(2, 3), 4)) is int
    assert typeof(is_even(add(1, 2))) is bool


def test_typeof_polymorphic():
    """Test typeof with operations that have polymorphic return types."""
    T = TypeVar("T")
    U = TypeVar("U")

    @defop
    def identity(x: T) -> T:
        raise NotImplementedError

    @defop
    def first(x: T, y: U) -> T:
        raise NotImplementedError

    @defop
    def if_then_else(cond: bool, then_val: T, else_val: T) -> T:
        raise NotImplementedError

    assert typeof(identity(42)) is int
    assert typeof(identity("hello")) is str
    assert typeof(first(42, "hello")) is int
    assert typeof(first("hello", 42)) is str
    assert typeof(if_then_else(True, 42, 43)) is int
    assert typeof(if_then_else(False, "hello", "world")) is str


def test_typeof_none():
    """Test typeof with operations that return None."""

    @defop
    def do_nothing() -> None:
        raise NotImplementedError

    @defop
    def print_value(x: Any) -> None:
        raise NotImplementedError

    assert typeof(do_nothing()) is type(None)
    assert typeof(print_value(42)) is type(None)


def test_typeof_scoped():
    """Test typeof with operations that have scoped annotations."""
    A = TypeVar("A")
    B = TypeVar("B")
    S = TypeVar("S")
    T = TypeVar("T")

    @defop
    def Lambda(
        var: Annotated[Operation[[], S], Scoped[A]], body: Annotated[T, Scoped[A | B]]
    ) -> Annotated[Callable[[S], T], Scoped[B]]:
        raise NotImplementedError

    x = defop(int, name="x")

    # Lambda that adds 1 to its argument
    lambda_term = Lambda(x, x() + 1)
    assert typeof(lambda_term) is Callable


def test_typeof_no_annotations():
    """Test typeof with operations that lack type annotations."""

    @defop
    def untyped_op(x, y):
        raise NotImplementedError

    @defop
    def partially_typed_op(x: int, y):
        raise NotImplementedError

    # Without annotations, the default is object
    assert typeof(untyped_op(1, 2)) is object
    assert typeof(partially_typed_op(1, 2)) is object


@pytest.mark.xfail(reason="Union types are not yet supported")
def test_typeof_union():
    """Test typeof with union types."""

    @defop
    def maybe_int(b: bool) -> int | str:
        raise NotImplementedError

    # Union types are simplified to their origin type
    assert typeof(maybe_int(True)) is Union


@pytest.mark.xfail(reason="Union types are not yet supported")
def test_typeof_optional():
    """Test typeof with Optional types."""

    @defop
    def maybe_value(b: bool) -> int | None:
        raise NotImplementedError

    # Optional[int] is Union[int, None], so it simplifies to Union
    assert typeof(maybe_value(True)) is Union


def test_typeof_generic():
    """Test typeof with generic classes."""
    T = TypeVar("T")

    class Box(Generic[T]):
        def __init__(self, value: T):
            self.value = value

    @defop
    def box_value(x: T) -> Box[T]:
        raise NotImplementedError

    # Generic types are simplified to their origin type
    assert typeof(box_value(42)) is Box
