"""
A generic mutable state effect with an optional default value.

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
>>> with interpreter(debug.bound_to(True)):
...     print(factorial(3))
LOG: 3
LOG: 2
LOG: 1
LOG: 0
6
"""

from dataclasses import dataclass
from typing import Any, Callable, Generic, ParamSpec, TypeVar

from effectful.ops.core import Interpretation, Operation, define, invalid_operation
from effectful.ops.interpreter import interpreter

T = TypeVar("T")
P = ParamSpec("P")
Q = ParamSpec("Q")


@dataclass
class Box(Generic[T]):
    contents: T

    def set(self, new: T):
        self.contents = new

    def get(self) -> T:
        return self.contents


@define
class State(Generic[T]):
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
            self.get = invalid_operation("Cannot read from an empty box")
            self.set = invalid_operation("Cannot write to an empty box")

        def bound_to(new: T):
            new_box = Box(new)
            return {self.get: new_box.get, self.set: new_box.set}

        self.bound_to = Operation(bound_to)

    def __call__(self) -> T:
        return self.get()
