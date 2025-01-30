from __future__ import annotations

import abc
import collections.abc
import inspect
import typing
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Generic, TypeVar

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

    __signature__: inspect.Signature
    __name__: str

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abc.abstractmethod
    def __hash__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __default_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> Expr[V]:
        """The default rule is used when the operation is not handled.

        If no default rule is supplied, the free rule is used instead.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __type_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> type[V]:
        """Returns the type of the operation applied to arguments."""
        raise NotImplementedError

    @abc.abstractmethod
    def __fvs_rule__(
        self, *args: Q.args, **kwargs: Q.kwargs
    ) -> tuple[
        tuple[collections.abc.Set[Operation], ...],
        dict[str, collections.abc.Set[Operation]],
    ]:
        """
        Returns the sets of variables that appear free in each argument and keyword argument
        but not in the result of the operation, i.e. the variables bound by the operation.

        These are used by :func:`fvsof` to determine the free variables of a term by
        subtracting the results of this method from the free variables of the subterms,
        allowing :func:`fvsof` to be implemented in terms of :func:`evaluate` .
        """
        raise NotImplementedError

    @typing.final
    def __call__(self, *args: Q.args, **kwargs: Q.kwargs) -> V:
        from effectful.internals.runtime import get_interpretation
        from effectful.ops.semantics import apply

        return apply.__default_rule__(get_interpretation(), self, *args, **kwargs)  # type: ignore

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__name__}, {self.__signature__})"


class Term(abc.ABC, Generic[T]):
    """A term in an effectful computation is a is a tree of :class:`Operation`
    applied to values.

    """

    __match_args__ = ("op", "args", "kwargs")

    @property
    @abc.abstractmethod
    def op(self) -> Operation[..., T]:
        """Abstract property for the operation."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def args(self) -> Sequence[Expr[Any]]:
        """Abstract property for the arguments."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def kwargs(self) -> Mapping[str, Expr[Any]]:
        """Abstract property for the keyword arguments."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.op!r}, {self.args!r}, {self.kwargs!r})"

    def __str__(self) -> str:
        from effectful.internals.runtime import interpreter
        from effectful.ops.semantics import apply, evaluate

        fresh: dict[str, dict[Operation, int]] = collections.defaultdict(dict)

        def op_str(op):
            """Return a unique (in this term) name for the operation."""
            name = op.__name__
            if name not in fresh:
                fresh[name] = {op: 0}
            if op not in fresh[name]:
                fresh[name][op] = len(fresh[name])

            n = fresh[name][op]
            if n == 0:
                return name
            return f"{name}!{n}"

        def term_str(term):
            if isinstance(term, Operation):
                return op_str(term)
            return str(term)

        def _apply(_, op, *args, **kwargs) -> str:
            args_str = ", ".join(map(term_str, args)) if args else ""
            kwargs_str = (
                ", ".join(f"{k}={term_str(v)}" for k, v in kwargs.items())
                if kwargs
                else ""
            )

            ret = f"{op_str(op)}({args_str}"
            if kwargs:
                ret += f"{', ' if args else ''}"
            ret += f"{kwargs_str})"
            return ret

        with interpreter({apply: _apply}):
            return typing.cast(str, evaluate(self))


#: An expression is either a value or a term.
Expr = T | Term[T]

#: An interpretation is a mapping from operations to their implementations.
Interpretation = Mapping[Operation[..., T], Callable[..., V]]


class Annotation(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def infer_annotations(cls, sig: inspect.Signature) -> inspect.Signature:
        raise NotImplementedError
