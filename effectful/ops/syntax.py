import collections
import dataclasses
import functools
import typing
from typing import Annotated, Callable, Generic, Optional, Type, TypeVar, Union

import tree
from typing_extensions import Concatenate, ParamSpec

from effectful.ops.types import ArgAnnotation, Expr, Interpretation, Operation, Term

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
    """Raised in an operation's signature to indicate that the operation has no default rule."""

    pass


@typing.overload
def defop(t: Type[T], *, name: Optional[str] = None) -> Operation[[], T]: ...


@typing.overload
def defop(t: Callable[P, T], *, name: Optional[str] = None) -> Operation[P, T]: ...


@typing.overload
def defop(t: Operation[P, T], *, name: Optional[str] = None) -> Operation[P, T]: ...


def defop(t, *, name=None):
    """Creates a fresh :class:`Operation`.

    :param t: May be a type, callable, or :class:`Operation`. If a type, the
              operation will have no arguments and return the type. If a callable,
              the operation will have the same signature as the callable, but with
              no default rule. If an operation, the operation will be a distinct
              copy of the operation.
    :param name: Optional name for the operation.
    :returns: A fresh operation.

    .. note::

      The result of :func:`defop` is always fresh (i.e. ``defop(f) != defop(f)``).

    **Example usage**:

    * Defining an operation:

      This example defines an operation that selects one of two integers:

      >>> @defop
      ... def select(x: int, y: int) -> int:
      ...     return x

      The operation can be called like a regular function. By default, ``select``
      returns the first argument:

      >>> select(1, 2)
      1

      We can change its behavior by installing a ``select`` handler:

      >>> from effectful.ops.semantics import handler
      >>> with handler({select: lambda x, y: y}):
      ...     print(select(1, 2))
      2

    * Defining an operation with no default rule:

      We can use :func:`defop` and the
      :exc:`effectful.internals.sugar.NoDefaultRule` exception to define an
      operation with no default rule:

      >>> @defop
      ... def add(x: int, y: int) -> int:
      ...     raise NoDefaultRule
      >>> add(1, 2)
      add(1, 2)

      When an operation has no default rule, the free rule is used instead, which
      constructs a term of the operation applied to its arguments. This feature
      can be used to conveniently define the syntax of a domain-specific language.

    * Defining free variables:

      Passing :func:`defop` a type is a handy way to create a free variable.

      >>> import effectful.handlers.operator
      >>> from effectful.ops.semantics import evaluate
      >>> x = defop(int, name='x')
      >>> y = x() + 1

      ``y`` is free in ``x``, so it is not fully evaluated:

      >>> y
      add(x(), 1)

      We bind ``x`` by installing a handler for it:

      >>> with handler({x: lambda: 2}):
      ...     print(evaluate(y))
      3

      .. note::

        Because the result of :func:`defop` is always fresh, it's important to
        be careful with variable identity.

        Two variables with the same name are not equal:

        >>> x1 = defop(int, name='x')
        >>> x2 = defop(int, name='x')
        >>> x1 == x2
        False

        This means that to correctly bind a variable, you must use the same
        operation object. In this example, ``scale`` returns a term with a free
        variable ``x``:

        >>> import effectful.handlers.operator
        >>> def scale(a: float) -> float:
        ...     x = defop(float, name='x')
        ...     return x() * a

        Binding the variable ``x`` by creating a fresh operation object does not

        >>> term = scale(3.0)
        >>> x = defop(float, name='x')
        >>> with handler({x: lambda: 2.0}):
        ...     print(evaluate(term))
        mul(x(), 3.0)

        This does:

        >>> from effectful.ops.semantics import fvsof
        >>> correct_x = [v for v in fvsof(term) if str(x) == 'x'][0]
        >>> with handler({correct_x: lambda: 2.0}):
        ...     print(evaluate(term))
        6.0

    * Defining a fresh :class:`Operation`:

      Passing :func:`defop` an :class:`Operation` creates a fresh operation with
      the same name and signature, but no default rule.

      >>> fresh_select = defop(select)
      >>> fresh_select(1, 2)
      select(1, 2)

      The new operation is distinct from the original:

      >>> with handler({select: lambda x, y: y}):
      ...     print(select(1, 2), fresh_select(1, 2))
      2 select(1, 2)

      >>> with handler({fresh_select: lambda x, y: y}):
      ...     print(select(1, 2), fresh_select(1, 2))
      1 2

    """

    if isinstance(t, Operation):

        def func(*args, **kwargs):
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
    """An operation that represents a lambda function.

    :param body: The body of the function.
    :type body: T
    :param args: Operations representing the positional arguments of the function.
    :type args: Annotated[Operation, Bound()]
    :param kwargs: Operations representing the keyword arguments of the function.
    :type kwargs: Annotated[Operation, Bound()]
    :returns: A callable term.
    :rtype: Callable[..., T]

    :func:`deffn` terms are eliminated by the :func:`call` operation, which
    performs beta-reduction.

    **Example usage**:

    Here :func:`deffn` is used to define a term that represents the function
    ``lambda x, y=1: 2 * x + y``:

    >>> import effectful.handlers.operator
    >>> x, y = defop(int, name='x'), defop(int, name='y')
    >>> term = deffn(2 * x() + y(), x, y=y)
    >>> term
    deffn(add(mul(2, x()), y()), x, y=y)
    >>> term(3, y=4)
    10

    .. note::

      In general, avoid using :func:`deffn` directly. Instead, use
      :func:`defterm` to convert a function to a term because it will
      automatically create the right free variables.

    """
    raise NoDefaultRule


class _CustomSingleDispatchCallable(Generic[P, T]):
    def __init__(
        self, func: Callable[Concatenate[Callable[[Type[T]], Callable[P, T]], P], T]
    ):
        self._func = func
        self._registry = functools.singledispatch(func)
        functools.update_wrapper(self, func)

    @property
    def dispatch(self):
        return self._registry.dispatch

    @property
    def register(self):
        return self._registry.register

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self._func(self.dispatch, *args, **kwargs)


@_CustomSingleDispatchCallable
def defterm(
    __dispatch: Callable[[Type[T]], Callable[[T], Expr[T]]], value: T
) -> Expr[T]:
    """Convert a value to a term, using the type of the value to dispatch.

    :param value: The value to convert.
    :type value: T
    :returns: A term.
    :rtype: Expr[T]

    **Example usage**:

    :func:`defterm` can be passed a function, and it will convert that function
    to a term by calling it with appropriately typed free variables:

    >>> def incr(x: int) -> int:
    ...     return x + 1
    >>> term = defterm(incr)
    >>> term
    deffn(add(int(), 1), int)
    >>> term(2)
    3

    """
    if isinstance(value, Term):
        return value
    else:
        return __dispatch(type(value))(value)


@_CustomSingleDispatchCallable
def defdata(
    __dispatch: Callable[[Type[T]], Callable[Concatenate[Operation[P, T], P], Expr[T]]],
    op: Operation[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> Expr[T]:
    """Constructs a Term that is an instance of its semantic type.

    :returns: An instance of ``T``.
    :rtype: Expr[T]

    This function is the only way to construct a :class:`Term` from an :class:`Operation`.

    .. note::

      This function is not likely to be called by users of the effectful
      library, but they may wish to register implementations for additional
      types.

    **Example usage**:

    This is how callable terms are implemented:

    .. code-block:: python

      class _CallableTerm(Generic[P, T], Term[collections.abc.Callable[P, T]]):
          def __init__(
              self,
              op: Operation[..., T],
              *args: Expr,
              **kwargs: Expr,
          ):
              self._op = op
              self._args = args
              self._kwargs = kwargs

          @property
          def op(self):
              return self._op

          @property
          def args(self):
              return self._args

          @property
          def kwargs(self):
              return self._kwargs

          def __call__(self, *args: Expr, **kwargs: Expr) -> Expr[T]:
              from effectful.ops.semantics import call

              return call(self, *args, **kwargs)

      @defdata.register(collections.abc.Callable)
      def _(op, *args, **kwargs):
          return _CallableTerm(op, *args, **kwargs)

    When an Operation whose return type is `Callable` is passed to :func:`defdata`,
    it is reconstructed as a :class:`_CallableTerm`, which implements the :func:`__call__` method.
    """
    from effectful.ops.semantics import apply, evaluate, typeof

    arg_ctxs, kwarg_ctxs = op.__fvs_rule__(*args, **kwargs)
    args_kwargs: dict[Union[int, str], tuple[Expr, Interpretation]] = {
        **dict(enumerate(zip(args, arg_ctxs))),
        **{k: (v, kwarg_ctxs[k]) for k, v in kwargs.items()},
    }
    args_, kwargs_ = list(args), dict(kwargs)
    for k, (v, c) in list(args_kwargs.items()):
        if c:
            v = tree.map_structure(
                lambda a: c.get(a, a) if isinstance(a, Operation) else a, v
            )
            res = evaluate(
                v, intp={apply: lambda _, op, *a, **k: defdata(op, *a, **k), **c}
            )
            if isinstance(k, int):
                args_.append(res)
            else:
                kwargs_[k] = res

    tp: Type[T] = typeof(
        __dispatch(typing.cast(Type[T], object))(
            op,
            *args_,  # type: ignore
            **kwargs_,  # type: ignore
        )
    )
    return __dispatch(tp)(
        op,
        *args_,  # type: ignore
        **kwargs_,  # type: ignore
    )


@defterm.register(object)
@defterm.register(Operation)
@defterm.register(Term)
def _(value: T) -> T:
    return value


@defdata.register(object)
def _(op: Operation[P, T], *args: P.args, **kwargs: P.kwargs):
    from effectful.internals.base_impl import _BaseTerm

    return _BaseTerm(op, *args, **kwargs)


@defdata.register(collections.abc.Callable)
def _(op: Operation[P, Callable[Q, T]], *args: P.args, **kwargs: P.kwargs):
    from effectful.internals.base_impl import _CallableTerm

    return typing.cast(_CallableTerm[Q, T], _CallableTerm(op, *args, **kwargs))


@defterm.register(collections.abc.Callable)
def _(fn: Callable[P, T]) -> Expr[Callable[P, T]]:
    from effectful.internals.base_impl import _unembed_callable

    return _unembed_callable(fn)


def syntactic_eq(x: Expr[T], other: Expr[T]) -> bool:
    """Syntactic equality, ignoring the interpretation of the terms.

    :param x: A term.
    :type x: Expr[T]
    :param other: Another term.
    :type other: Expr[T]
    :returns: ``True`` if the terms are syntactically equal and ``False`` otherwise.
    """
    if isinstance(x, Term) and isinstance(other, Term):
        op, args, kwargs = x.op, x.args, x.kwargs
        op2, args2, kwargs2 = other.op, other.args, other.kwargs
        try:
            tree.assert_same_structure(
                (op, args, kwargs), (op2, args2, kwargs2), check_types=True
            )
        except (TypeError, ValueError):
            return False
        return all(
            tree.flatten(
                tree.map_structure(
                    syntactic_eq, (op, args, kwargs), (op2, args2, kwargs2)
                )
            )
        )
    elif isinstance(x, Term) or isinstance(other, Term):
        return False
    else:
        return x == other


class ObjectInterpretation(Generic[T, V], Interpretation[T, V]):
    """A helper superclass for defining an ``Interpretation`` of many
    :class:`~effectful.ops.types.Operation` instances with shared state or behavior.

    You can mark specific methods in the definition of an
    :class:`ObjectInterpretation` with operations using the :func:`implements`
    decorator. The :class:`ObjectInterpretation` object itself is an
    ``Interpretation`` (mapping from :class:`~effectful.ops.types.Operation` to :class:`~typing.Callable`)

    >>> from effectful.ops.semantics import handler
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
    """Marks a method in an :class:`ObjectInterpretation` as the implementation of a
    particular abstract :class:`Operation`.

    When passed an :class:`Operation`, returns a method decorator which installs
    the given method as the implementation of the given :class:`Operation`.

    """
    return _ImplementedOperation(op)
