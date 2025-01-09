from __future__ import annotations

import abc
import typing
from typing import (
    Any,
    Callable,
    Generic,
    Mapping,
    Optional,
    Sequence,
    Set,
    Type,
    TypeVar,
    Union,
)

from typing_extensions import ParamSpec

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


class Operation(abc.ABC, Generic[Q, V]):
    """An abstract class representing an effect that can be implemented by an effect handler.

    .. note::

       Do not use :class:`Operation` directly. Instead, use :func:`defop` to define operations.

    """

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abc.abstractmethod
    def __hash__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __default_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> "Expr[V]":
        """The default rule is used when the operation is not handled.

        If no default rule is supplied, the free rule is used instead.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __free_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> "Expr[V]":
        """Returns a term for the operation applied to arguments."""
        raise NotImplementedError

    @abc.abstractmethod
    def __type_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> Type[V]:
        """Returns the type of the operation applied to arguments."""
        raise NotImplementedError

    @abc.abstractmethod
    def __fvs_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> Set["Operation"]:
        """Returns the free variables of the operation applied to arguments."""
        raise NotImplementedError

    @abc.abstractmethod
    def __repr_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> str:
        raise NotImplementedError

    @typing.final
    def __call__(self, *args: Q.args, **kwargs: Q.kwargs) -> V:
        from effectful.internals.runtime import get_interpretation
        from effectful.ops.semantics import apply

        return apply.__default_rule__(get_interpretation(), self, *args, **kwargs)  # type: ignore


class Term(abc.ABC, Generic[T]):
    """A term in an effectful computation is a is a tree of :class:`Operation`
    applied to values.

    """

    __match_args__ = ("op", "args", "kwargs")

    @property
    @abc.abstractmethod
    def op(self) -> Operation[..., T]:
        """Abstract property for the operation."""
        pass

    @property
    @abc.abstractmethod
    def args(self) -> Sequence["Expr[Any]"]:
        """Abstract property for the arguments."""
        pass

    @property
    @abc.abstractmethod
    def kwargs(self) -> Mapping[str, "Expr[Any]"]:
        """Abstract property for the keyword arguments."""
        pass

    def __repr__(self) -> str:
        from effectful.internals.runtime import interpreter
        from effectful.ops.semantics import apply, evaluate

        with interpreter({apply: lambda _, op, *a, **k: op.__repr_rule__(*a, **k)}):
            return evaluate(self)  # type: ignore


#: An expression is either a value or a term.
Expr = Union[T, Term[T]]

#: An interpretation is a mapping from operations to their implementations.
Interpretation = Mapping[Operation[..., T], Callable[..., V]]

MaybeResult = Optional[T]


class ArgAnnotation:
    pass
