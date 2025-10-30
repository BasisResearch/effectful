from __future__ import annotations

import abc
import collections.abc
import functools
import inspect
import typing
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, _ProtocolMeta, Concatenate, overload, runtime_checkable


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
    __default__: Callable[Q, V]

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abc.abstractmethod
    def __hash__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __lt__(self, other):
        raise NotImplementedError

    @functools.singledispatchmethod
    @classmethod
    def define(cls, obj) -> Operation[Q, V]:
        """Define an operation from a callable object."""
        raise NotImplementedError

    @typing.final
    def __default_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> Expr[V]:
        """The default rule is used when the operation is not handled.

        If no default rule is supplied, the free rule is used instead.
        """
        try:
            try:
                return self.__default__(*args, **kwargs)
            except NotImplementedError:
                warnings.warn(
                    "Operations should raise effectful.ops.types.NotHandled instead of NotImplementedError.",
                    DeprecationWarning,
                )
                raise NotHandled
        except NotHandled:
            from effectful.ops.syntax import defdata

            return typing.cast(
                Callable[Concatenate[Operation[Q, V], Q], Expr[V]], defdata
            )(self, *args, **kwargs)

    @typing.final
    def __type_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> type[V]:
        """Returns the type of the operation applied to arguments."""
        from effectful.internals.unification import (
            freetypevars,
            nested_type,
            substitute,
            unify,
        )

        return_anno = self.__signature__.return_annotation
        if typing.get_origin(return_anno) is typing.Annotated:
            return_anno = typing.get_args(return_anno)[0]

        if return_anno is inspect.Parameter.empty:
            return typing.cast(type[V], object)
        elif return_anno is None:
            return type(None)  # type: ignore
        elif not freetypevars(return_anno):
            return return_anno

        type_args = tuple(nested_type(a) for a in args)
        type_kwargs = {k: nested_type(v) for k, v in kwargs.items()}
        bound_sig = self.__signature__.bind(*type_args, **type_kwargs)
        return substitute(return_anno, unify(self.__signature__, bound_sig))  # type: ignore

    @typing.final
    def __fvs_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> inspect.BoundArguments:
        """
        Returns the sets of variables that appear free in each argument and keyword argument
        but not in the result of the operation, i.e. the variables bound by the operation.

        These are used by :func:`fvsof` to determine the free variables of a term by
        subtracting the results of this method from the free variables of the subterms,
        allowing :func:`fvsof` to be implemented in terms of :func:`evaluate` .
        """
        from effectful.ops.syntax import Scoped

        sig = Scoped.infer_annotations(self.__signature__)
        bound_sig = sig.bind(*args, **kwargs)
        bound_sig.apply_defaults()

        result_sig = sig.bind(
            *(frozenset() for _ in bound_sig.args),
            **{k: frozenset() for k in bound_sig.kwargs},
        )
        for name, param in sig.parameters.items():
            if typing.get_origin(param.annotation) is typing.Annotated:
                for anno in typing.get_args(param.annotation)[1:]:
                    if isinstance(anno, Scoped):
                        param_bound_vars = anno.analyze(bound_sig)
                        if param.kind is inspect.Parameter.VAR_POSITIONAL:
                            result_sig.arguments[name] = tuple(
                                param_bound_vars for _ in bound_sig.arguments[name]
                            )
                        elif param.kind is inspect.Parameter.VAR_KEYWORD:
                            for k in bound_sig.arguments[name]:
                                result_sig.arguments[name][k] = param_bound_vars
                        else:
                            result_sig.arguments[name] = param_bound_vars

        return result_sig

    @typing.final
    def __call__(self, *args: Q.args, **kwargs: Q.kwargs) -> V:
        from effectful.ops.semantics import apply

        return apply.__default_rule__(self, *args, **kwargs)  # type: ignore

    def __get__(self, instance, owner):
        if instance is not None:
            # This is an instance-level operation, so we need to bind the instance
            return functools.partial(self, instance)
        else:
            # This is a static operation, so we return the operation itself
            return self

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__name__}, {self.__signature__})"

    def __str__(self):
        return self.__name__


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


try:
    from prettyprinter import install_extras, pretty_call, register_pretty

    install_extras({"dataclasses"})

    @register_pretty(Term)
    def pretty_term(value: Term, ctx):
        default_op_name = str(value.op)

        fresh_by_name = ctx.get("fresh_by_name") or {}
        new_ctx = ctx.assoc("fresh_by_name", fresh_by_name)

        fresh = fresh_by_name.get(default_op_name, {})
        fresh_by_name[default_op_name] = fresh

        fresh_ctr = fresh.get(value.op, len(fresh))
        fresh[value.op] = fresh_ctr

        op_name = str(value.op) + (f"!{fresh_ctr}" if fresh_ctr > 0 else "")
        return pretty_call(new_ctx, op_name, *value.args, **value.kwargs)

except ImportError:
    pass


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
