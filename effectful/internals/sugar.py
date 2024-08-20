from functools import wraps
from inspect import signature
from typing import Callable, Generic, Optional, ParamSpec, TypeVar, get_origin

from effectful.internals.prompts import Prompt
from effectful.ops.core import Interpretation, Operation
from effectful.ops.handler import fwd

P = ParamSpec("P")
V = TypeVar("V")
T = TypeVar("T")


class ObjectInterpretation(Generic[T, V], Interpretation[T, V]):
    """
    A helper superclass for defining an :type:`Interpretation`s of many :type:`Operation` instances with shared
    state or behavior.

    You can mark specific methods in the definition of an :class:`ObjectInterpretation` with operations
    using the :func:`implements` decorator. The :class:`ObjectInterpretation` object itself is an :type:`Interpretation`
    (mapping from :type:`Operation` to :type:`Callable`)

    >>> from effectful.ops.core import define
    >>> from effectful.ops.handler import handler
    >>> @Operation
    ... def read_box():
    ...     pass
    ...
    >>> @Operation
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
    _temporary_implementations: dict[Operation[..., T], Callable[..., V]] = dict()
    implementations: dict[Operation[..., T], Callable[..., V]] = dict()

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

    def __getitem__(self, item: Operation[..., T]) -> Callable[..., V]:
        return self.implementations[item].__get__(self, type(self))


P1 = ParamSpec("P1")
P2 = ParamSpec("P2")


class _ImplementedOperation(Generic[P1, P2, T, V]):
    impl: Optional[Callable[P2, V]]
    op: Operation[P1, T]

    def __init__(self, op: Operation[P1, T]):
        self.op = op
        self.impl = None

    def __get__(
        self, instance: ObjectInterpretation[T, V], owner: type
    ) -> Callable[..., V]:
        assert self.impl is not None

        return self.impl.__get__(instance, owner)

    def __call__(self, impl: Callable[P2, V]):
        self.impl = impl
        return self

    def __set_name__(self, owner: ObjectInterpretation[T, V], name):
        assert self.impl is not None
        assert self.op is not None
        owner._temporary_implementations[self.op] = self.impl


def implements(op: Operation[P, V]):
    """
    Marks a method in an `ObjectInterpretation` as the implementation of a
    particular abstract `Operation`.

    When passed an `Operation`, returns a method decorator which installs the given
    method as the implementation of the given `Operation`.
    """
    return _ImplementedOperation(op)


def _adheres_to(obj, cond):
    while get_origin(cond) is not None:
        cond = get_origin(cond)

    try:
        return isinstance(obj, cond)
    except TypeError:
        return True


def type_guard(
    prompt: Prompt = fwd, **kwargs
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    A helper for defining implementations of operations with type dispatch.
    When arguments are annotated with runtime-checkable types, the arguments
    are checked and dispatched upon.
    If they match, then the implementation is called as normal.
    If it they don't, then `prompt(None)` (by default, `fwd(None)`) is called.
    Other keyword arguments are passed to `inspect.signature`.

    Because much of the type system's information is thrown out at runtime, we
    can only check a subset of annotations. See `isinstance` for more information.

    >>> from effectful.ops.handler import handler
    >>> @Operation
    ... def count_vowels(obj):
    ...     return 0
    >>> vowels = "aeiou"
    >>> @type_guard()
    ... def count_string_vowels(obj: str):
    ...     return sum(int(c in vowels) for c in obj)
    >>> @type_guard()
    ... def count_list_vowels(obj: list):
    ...     return sum(map(count_vowels, obj))
    >>> text_sample = "an example sentence"
    >>> with handler({count_vowels: count_string_vowels}):
    ...     with handler({count_vowels: count_list_vowels}):
    ...         print(f"String: {count_vowels(text_sample)}")
    ...         print(f"List: {count_vowels(text_sample.split())}")
    String: 7
    List: 7
    """

    def take_fn(fn: Callable[..., T]) -> Callable[..., T]:
        sig = signature(fn, **kwargs)

        @wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            bound = sig.bind(*args, **kwargs)

            for n, k in sig.parameters.items():
                v = bound.arguments[n]

                if not _adheres_to(v, k.annotation):
                    return prompt(None)

            return fn(*args, **kwargs)

        return wrapper

    return take_fn
