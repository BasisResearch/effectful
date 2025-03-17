import inspect
from typing import Any, Callable, Generic, Optional, TypeVar, Union

import pytest

from effectful.ops.semantics import typeof, handler
from effectful.ops.syntax import Scoped, defop, defterm
from effectful.ops.types import Annotation, Operation, Term


def test_typeof_basic():
    """Test typeof with basic operations that have simple return types."""
    @defop
    def add(x: int, y: int) -> int:
        return x + y

    @defop
    def is_positive(x: int) -> bool:
        return x > 0

    @defop
    def get_name() -> str:
        return "test"

    assert typeof(add(1, 2)) is int
    assert typeof(is_positive(5)) is bool
    assert typeof(get_name()) is str


def test_typeof_nested():
    """Test typeof with nested operations."""
    @defop
    def add(x: int, y: int) -> int:
        return x + y

    @defop
    def multiply(x: int, y: int) -> int:
        return x * y

    @defop
    def is_even(x: int) -> bool:
        return x % 2 == 0

    assert typeof(add(multiply(2, 3), 4)) is int
    assert typeof(is_even(add(1, 2))) is bool


def test_typeof_polymorphic():
    """Test typeof with operations that have polymorphic return types."""
    T = TypeVar('T')
    U = TypeVar('U')

    @defop
    def identity(x: T) -> T:
        return x

    @defop
    def first(x: T, y: U) -> T:
        return x

    @defop
    def if_then_else(cond: bool, then_val: T, else_val: T) -> T:
        return then_val if cond else else_val

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
        pass

    @defop
    def print_value(x: Any) -> None:
        print(x)

    assert typeof(do_nothing()) is type(None)
    assert typeof(print_value(42)) is type(None)


def test_typeof_scoped():
    """Test typeof with operations that have scoped annotations."""
    A = TypeVar('A')
    B = TypeVar('B')
    S = TypeVar('S')
    T = TypeVar('T')

    @defop
    def Lambda(
        var: Annotation[Operation[[], S], Scoped[A]],
        body: Annotation[T, Scoped[A | B]]
    ) -> Annotation[Callable[[S], T], Scoped[B]]:
        raise NotImplementedError

    x = defop(int, name='x')
    y = defop(int, name='y')

    # Lambda that adds 1 to its argument
    lambda_term = Lambda(x, x() + 1)
    assert typeof(lambda_term) is Callable


def test_typeof_no_annotations():
    """Test typeof with operations that lack type annotations."""
    @defop
    def untyped_op(x, y):
        return x + y

    @defop
    def partially_typed_op(x: int, y):
        return x + y

    # Without annotations, the default is object
    assert typeof(untyped_op(1, 2)) is object
    assert typeof(partially_typed_op(1, 2)) is object


def test_typeof_callable():
    """Test typeof with callable terms."""
    def add_one(x: int) -> int:
        return x + 1

    term = defterm(add_one)
    assert typeof(term) is Callable

    # When called, it should return an int
    assert typeof(term(5)) is int


def test_typeof_union():
    """Test typeof with union types."""
    @defop
    def maybe_int(b: bool) -> Union[int, str]:
        return 42 if b else "hello"

    # Union types are simplified to their origin type
    assert typeof(maybe_int(True)) is Union


def test_typeof_optional():
    """Test typeof with Optional types."""
    @defop
    def maybe_value(b: bool) -> Optional[int]:
        return 42 if b else None

    # Optional[int] is Union[int, None], so it simplifies to Union
    assert typeof(maybe_value(True)) is Union


def test_typeof_generic():
    """Test typeof with generic classes."""
    T = TypeVar('T')

    class Box(Generic[T]):
        def __init__(self, value: T):
            self.value = value

    @defop
    def box_value(x: T) -> Box[T]:
        return Box(x)

    # Generic types are simplified to their origin type
    assert typeof(box_value(42)) is Box
