from dataclasses import dataclass
from typing import Any, Generic, ParamSpec, TypeVar

from effectful.ops.core import Interpretation, Operation

T = TypeVar("T")
V = TypeVar("V")
P = ParamSpec("P")
Q = ParamSpec("Q")


@dataclass
class Box(Generic[T]):
    contents: T

    def set(self, new: T):
        self.contents = new

    def get(self) -> T:
        return self.contents


@dataclass
class State(Generic[T]):
    """
    A generic mutable state effect with an optional default value.

    >>> from effectful.ops.handler import handler
    >>> debug = State(False)
    >>> def log(value):
    ...     if debug.get():
    ...         print(f"LOG: {value}")
    ...
    >>> def factorial(value):
    ...     log(value)
    ...     return 1 if value == 0 else value * factorial(value - 1)
    >>> factorial(3)
    6
    >>> with handler(debug.bound_to(True)):
    ...     print(factorial(3))
    LOG: 3
    LOG: 2
    LOG: 1
    LOG: 0
    6
    """

    get: Operation[[], T]
    set: Operation[[T], None]
    bound_to: Operation[[T], Interpretation[T | None, T | None]]

    class _Empty:
        pass

    def __init__(self, initial_state: T | _Empty = _Empty()):
        if not isinstance(initial_state, self._Empty):
            box = Box(initial_state)

            self.get = Operation(box.get)
            self.set = Operation(box.set)
        else:

            def explicit_operation(msg: str) -> Operation[..., Any]:
                def default(*args, **kwargs):
                    raise RuntimeError(msg, *args, **kwargs)

                return Operation(default)

            self.get = explicit_operation("Cannot read from an empty box")
            self.set = explicit_operation("Cannot write to an empty box")

        def bound_to(new: T):
            new_box = Box(new)
            return {self.get: new_box.get, self.set: new_box.set}

        self.bound_to = Operation(bound_to)

    def __call__(self) -> T:
        return self.get()
