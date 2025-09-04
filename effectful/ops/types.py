from __future__ import annotations

import abc
import collections.abc
import functools
import inspect
import typing
from collections.abc import Callable, Mapping, Sequence
from typing import Any, _ProtocolMeta, overload, runtime_checkable


class NotHandled(Exception):
    """Raised by an operation when the operation should remain unhandled."""

    pass


@functools.total_ordering
class Operation[**Q, V](abc.ABC):
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
    def __lt__(self, other):
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
    def __fvs_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> inspect.BoundArguments:
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
        from effectful.ops.semantics import apply

        return apply.__default_rule__(self, *args, **kwargs)  # type: ignore

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__name__}, {self.__signature__})"


class Term[T](abc.ABC):
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
            elif isinstance(term, list):
                return "[" + ", ".join(map(term_str, term)) + "]"
            elif isinstance(term, tuple):
                return "(" + ", ".join(map(term_str, term)) + ")"
            elif isinstance(term, dict):
                return (
                    "{"
                    + ", ".join(
                        f"{term_str(k)}:{term_str(v)}" for (k, v) in term.items()
                    )
                    + "}"
                )
            return str(term)

        def _apply(op, *args, **kwargs) -> str:
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
type Expr[T] = T | Term[T]


class _InterpretationMeta(_ProtocolMeta):
    def __instancecheck__(cls, instance):
        return isinstance(instance, collections.abc.Mapping) and all(
            isinstance(k, Operation) and callable(v) for k, v in instance.items()
        )


@runtime_checkable
class Interpretation[T, V](typing.Protocol, metaclass=_InterpretationMeta):
    """An interpretation is a mapping from operations to their implementations."""

    def keys(self):
        raise NotImplementedError

    def values(self):
        raise NotImplementedError

    def items(self):
        raise NotImplementedError

    @overload
    def get(self, key: Operation[..., T], /) -> Callable[..., V] | None:
        raise NotImplementedError

    @overload
    def get(
        self, key: Operation[..., T], default: Callable[..., V], /
    ) -> Callable[..., V]:
        raise NotImplementedError

    @overload
    def get[S](self, key: Operation[..., T], default: S, /) -> Callable[..., V] | S:
        raise NotImplementedError

    def __getitem__(self, key: Operation[..., T]) -> Callable[..., V]:
        raise NotImplementedError

    def __contains__(self, key: Operation[..., T]) -> bool:
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class Annotation(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def infer_annotations(cls, sig: inspect.Signature) -> inspect.Signature:
        raise NotImplementedError
