import abc
import typing
from typing import (
    Any,
    Callable,
    Generic,
    Mapping,
    Sequence,
    Tuple,
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

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abc.abstractmethod
    def __hash__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __default_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> "Expr[V]":
        raise NotImplementedError

    @abc.abstractmethod
    def __free_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> "Expr[V]":
        raise NotImplementedError

    @abc.abstractmethod
    def __type_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> Type[V]:
        raise NotImplementedError

    @abc.abstractmethod
    def __fvs_rule__(
        self, *args: Q.args, **kwargs: Q.kwargs
    ) -> "Interpretation[T, Type[T]]":
        raise NotImplementedError

    @typing.final
    def __call__(self, *args: Q.args, **kwargs: Q.kwargs) -> V:
        from effectful.internals.runtime import get_interpretation
        from effectful.ops.semantics import apply

        return apply.__default_rule__(get_interpretation(), self, *args, **kwargs)  # type: ignore


class Term(abc.ABC, Generic[T]):
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
    def kwargs(self) -> Sequence[Tuple[str, "Expr[Any]"]]:
        """Abstract property for the keyword arguments."""
        pass


Expr = Union[T, Term[T]]

Interpretation = Mapping[Operation[..., T], Callable[..., V]]