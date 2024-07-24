from collections.abc import Mapping
from typing import Callable, Generic, Optional, ParamSpec, TypeVar

from effectful.ops.core import Operation

P = ParamSpec("P")
V = TypeVar("V")


class ObjectInterpretation(Mapping):
    """
    A helper superclass for defining `Interpretations`s of `Operation`s with shared state or behavior.

    You can mark specific methods in the definition of an `ObjectInterpretation` with operations
    using the `implements` decorator. The `ObjectInterpretation` object itself is an `Interpretation`
    (mapping from `Operation`s to `Callable`s)

    >>> from effectful.ops.core import define
    >>> from effectful.ops.handler import handler
    >>> @define(Operation)
    ... def read_box():
    ...     pass
    ...
    >>> @define(Operation)
    ... def write_box(new_value):
    ...     pass
    ...
    >>> class StatefulBox(ObjectInterpretation):
    ...     def __init__(self, init=None):
    ...         super().__init__()
    ...         self.stored = init
    ...     @implements(read_box)
    ...     def whatever(self):
    ...         return self.stored
    ...     @implements(write_box)
    ...     def write_box(self, new_value):
    ...         self.stored = new_value
    ...
    >>> first_box = StatefulBox(init="First Starting Value")
    >>> second_box = StatefulBox(init="Second Starting Value")
    >>> with handler(first_box):
    ...     print(read_box())
    ...     write_box("New Value")
    ...     print(read_box())
    ...
    First Starting Value
    New Value
    >>> with handler(second_box):
    ...     print(read_box())
    Second Starting Value
    >>> with handler(first_box):
    ...     print(read_box())
    New Value
    """

    # This is a weird hack to get around the fact that
    # the default meta-class runs __set_name__ before __init__subclass__.
    # We basically store the implementations here temporarily
    # until __init__subclass__ is called.
    # This dict is shared by all `Implementation`s,
    # so we need to clear it when we're done.
    _temporary_implementations: dict[Operation, Callable] = dict()
    implementations: dict[Operation, Callable] = dict()

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.implementations = ObjectInterpretation._temporary_implementations.copy()

        for sup in cls.mro():
            if issubclass(sup, ObjectInterpretation):
                cls.implementations = {**sup.implementations, **cls.implementations}

        ObjectInterpretation._temporary_implementations.clear()

    def __iter__(self):
        return iter(self.implementations)

    def __len__(self):
        return len(self.implementations)

    def __getitem__(self, item):
        return self.implementations[item].__get__(self, type(self))


class _ImplementedOperation(Generic[P, V]):
    impl: Optional[Callable[P, V]]

    def __init__(self, op: Optional[Operation[P, V]]):
        self.op = op
        self.impl = None

    def __get__(self, instance, owner):
        return self.impl.__get__(instance, owner)

    def __call__(self, impl: Callable[P, V]):
        self.impl = impl
        return self

    def __set_name__(self, owner: ObjectInterpretation, name):
        assert self.impl is not None
        assert self.op is not None
        owner._temporary_implementations[self.op] = self.impl


def implements(op: Operation[P, V]):
    """
    Makrs a method in an `ObjectInterpretation` as the implementation of a
    particular abstract `Operation`.

    When passed an `Operation`, returns a method decorator which installs the given
    method as the implementation of the given `Operation`.
    """
    return _ImplementedOperation(op)
