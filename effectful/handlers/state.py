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
>>> with debug.bound_to(True):
...     print(factorial(3))
LOG: 3
LOG: 2
LOG: 1
LOG: 0
6
"""

from collections import namedtuple
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from effectful.ops.core import Operation, define, register
from effectful.ops.interpreter import interpreter


T = TypeVar("T")


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

    def __init__(self, initial_state: T):
        box = Box(initial_state)

        self.get = Operation(box.get)
        self.set = Operation(box.set)

        def bound_to(new: T):
            new_box = Box(new)
            return interpreter({self.get: new_box.get, self.set: new_box.set})

        self.bound_to = Operation(bound_to)
