import collections
import dataclasses
import functools
import typing
from typing import Annotated, Callable, Generic, Optional, Sequence, Type, TypeVar

import tree
from typing_extensions import Concatenate, ParamSpec

from effectful.ops.types import (
    ArgAnnotation,
    Expr,
    Interpretation,
    MaybeResult,
    Operation,
    Term,
)

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@dataclasses.dataclass
class Bound(ArgAnnotation):
    scope: int = 0


@dataclasses.dataclass
class Scoped(ArgAnnotation):
    scope: int = 0


class NoDefaultRule(Exception):
    pass


@typing.overload
def defop(t: Type[T], *, name: Optional[str] = None) -> Operation[[], T]: ...


@typing.overload
def defop(t: Callable[P, T], *, name: Optional[str] = None) -> Operation[P, T]: ...


@typing.overload
def defop(t: Operation[P, T], *, name: Optional[str] = None) -> Operation[P, T]: ...


def defop(t, *, name=None):
    """defop creates fresh Operations.

    This is useful for creating fresh variables.

    :param t: May be a type or a callable. If a type, the Operation will have no arguments. If a callable, the Operation
    will have the same signature as the callable, but with no default rule.
    :param name: Optional name for the Operation.
    :returns: A fresh Operation.

    """

    if isinstance(t, Operation):

        def func(*args, **kwargs):  # type: ignore
            raise NoDefaultRule

        functools.update_wrapper(func, t)
        return defop(func, name=name)
    elif isinstance(t, type):

        def func() -> t:  # type: ignore
            raise NoDefaultRule

        func.__name__ = name or t.__name__
        return typing.cast(Operation[[], T], defop(func, name=name))
    elif isinstance(t, collections.abc.Callable):
        from effectful.internals.base_impl import _BaseOperation

        op = _BaseOperation(t)
        op.__name__ = name or t.__name__
        return op
    else:
        raise ValueError(f"expected type or callable, got {t}")


@defop
def deffn(
    body: T,
    *args: Annotated[Operation, Bound()],
    **kwargs: Annotated[Operation, Bound()],
) -> Callable[..., T]:
    raise NoDefaultRule


class _CustomSingleDispatchCallable(Generic[P, T]):
    def __init__(
        self, func: Callable[Concatenate[Callable[[type], Callable[P, T]], P], T]
    ):
        self._func = func
        self._registry = functools.singledispatch(func)

    @property
    def dispatch(self):
        return self._registry.dispatch

    @property
    def register(self):
        return self._registry.register

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self._func(self.dispatch, *args, **kwargs)


@_CustomSingleDispatchCallable
def defterm(dispatch, value: T) -> Expr[T]:
    if isinstance(value, Term):
        return value
    else:
        return dispatch(type(value))(value)


@_CustomSingleDispatchCallable
def defdata(dispatch, expr: Term[T]) -> Expr[T]:
    from effectful.ops.semantics import typeof

    if isinstance(expr, Term):
        impl: Callable[
            [Operation[..., T], Sequence, Sequence[tuple[str, object]]], Expr[T]
        ]
        impl = dispatch(typeof(expr))  # type: ignore
        return impl(expr.op, expr.args, expr.kwargs)  # type: ignore
    else:
        return expr


@defterm.register(object)
@defterm.register(Operation)
@defterm.register(Term)
def _unembed_literal(value: T) -> T:
    return value


@defdata.register(object)
def _(op, args, kwargs):
    from effectful.internals.base_impl import _BaseTerm

    return _BaseTerm(op, args, kwargs)


@defdata.register(collections.abc.Callable)
def _(op, args, kwargs):
    from effectful.internals.base_impl import _CallableTerm

    return _CallableTerm(op, args, kwargs)


@defterm.register(collections.abc.Callable)
def _(fn: Callable[P, T]):
    from effectful.internals.base_impl import _unembed_callable

    return _unembed_callable(fn)


def syntactic_eq(x: Expr[T], other: Expr[T]) -> bool:
    """Syntactic equality, ignoring the interpretation of the terms."""
    match x, other:
        case Term(op, args, kwargs), Term(op2, args2, kwargs2):
            kwargs_d, kwargs2_d = dict(kwargs), dict(kwargs2)
            try:
                tree.assert_same_structure(
                    (op, args, kwargs_d), (op2, args2, kwargs2_d), check_types=True
                )
            except (TypeError, ValueError):
                return False
            return all(
                tree.flatten(
                    tree.map_structure(
                        syntactic_eq, (op, args, kwargs_d), (op2, args2, kwargs2_d)
                    )
                )
            )
        case Term(_, _, _), _:
            return False
        case _, Term(_, _, _):
            return False
        case _, _:
            return x == other
    return False


def bind_result(fn: Callable[Concatenate[MaybeResult[T], P], T]) -> Callable[P, T]:
    from effectful.internals.runtime import _get_result

    return lambda *a, **k: fn(_get_result(), *a, **k)


def bind_result_to_method(
    fn: Callable[Concatenate[V, MaybeResult[T], P], T]
) -> Callable[Concatenate[V, P], T]:
    return bind_result(lambda r, s, *a, **k: fn(s, r, *a, **k))


class ObjectInterpretation(Generic[T, V], Interpretation[T, V]):
    """
    A helper superclass for defining an :type:`Interpretation`s of many :type:`Operation` instances with shared
    state or behavior.

    You can mark specific methods in the definition of an :class:`ObjectInterpretation` with operations
    using the :func:`implements` decorator. The :class:`ObjectInterpretation` object itself is an :type:`Interpretation`
    (mapping from :type:`Operation` to :type:`Callable`)

    >>> from effectful.ops.handler import handler
    >>> @defop
    ... def read_box():
    ...     pass
    ...
    >>> @defop
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


class _ImplementedOperation(Generic[P, Q, T, V]):
    impl: Optional[Callable[Q, V]]
    op: Operation[P, T]

    def __init__(self, op: Operation[P, T]):
        self.op = op
        self.impl = None

    def __get__(
        self, instance: ObjectInterpretation[T, V], owner: type
    ) -> Callable[..., V]:
        assert self.impl is not None

        return self.impl.__get__(instance, owner)

    def __call__(self, impl: Callable[Q, V]):
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
