import abc
import collections
import dataclasses
import functools
import inspect
import typing
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import tree
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
    def __scope_rule__(
        self, *args: Q.args, **kwargs: Q.kwargs
    ) -> "Interpretation[T, Type[T]]":
        raise NotImplementedError

    @typing.final
    def __call__(self, *args: Q.args, **kwargs: Q.kwargs) -> V:
        from effectful.internals.runtime import get_interpretation

        return apply.__default_rule__(get_interpretation(), self, *args, **kwargs)  # type: ignore


class Annotation:
    pass


@dataclasses.dataclass
class Bound(Annotation):
    scope: int = 0


@dataclasses.dataclass
class Scoped(Annotation):
    scope: int = 0


@typing.overload
def gensym(t: Type[T], *, name: Optional[str] = None) -> Operation[[], T]: ...


@typing.overload
def gensym(t: Callable[P, T], *, name: Optional[str] = None) -> Operation[P, T]: ...


def gensym(t, *, name=None):
    """gensym creates fresh Operations.

    This is useful for creating fresh variables.

    :param t: May be a type or a callable. If a type, the Operation will have no arguments. If a callable, the Operation
    will have the same signature as the callable, but with no default rule.
    :param name: Optional name for the Operation.
    :returns: A fresh Operation.

    """
    # curiously, typing.Callable[..., T] is not a subtype of typing.Type[T]
    is_type = (
        isinstance(t, typing.Type) or typing.get_origin(t) is collections.abc.Callable
    )

    if is_type:

        def func() -> t:  # type: ignore
            return NotImplemented

    elif isinstance(t, collections.abc.Callable):

        def func(*args, **kwargs):  # type: ignore
            return NotImplemented

        functools.update_wrapper(func, t)

    else:
        raise ValueError(f"expected type or callable, got {t}")

    func.__name__ = name or t.__name__
    op = defop(func)

    if is_type:
        return typing.cast(Operation[[], T], op)
    else:
        return typing.cast(Operation[P, T], op)


class _BaseOperation(Generic[Q, V], Operation[Q, V]):
    signature: Callable[Q, V]

    def __init__(self, signature: Callable[Q, V]):
        functools.update_wrapper(self, signature)
        self.signature = signature

    def __eq__(self, other):
        if not isinstance(other, Operation):
            return NotImplemented
        return self.signature == other.signature

    def __hash__(self):
        return hash(self.signature)

    def __default_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> "Expr[V]":
        maybe_result = self.signature(*args, **kwargs)
        if maybe_result is NotImplemented:
            return self.__free_rule__(*args, **kwargs)
        else:
            return maybe_result

    def __free_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> "Expr[V]":
        from effectful.internals.sugar import _embed_registry, embed, rename

        sig = inspect.signature(self.signature)
        bound_sig = sig.bind(*args, **kwargs)
        bound_sig.apply_defaults()

        bound_vars: dict[int, set[Operation]] = collections.defaultdict(set)
        scoped_args: dict[int, set[str]] = collections.defaultdict(set)
        unscoped_args: set[str] = set()
        for param_name, param in bound_sig.signature.parameters.items():
            if typing.get_origin(param.annotation) is typing.Annotated:
                for anno in param.annotation.__metadata__:
                    if isinstance(anno, Bound):
                        scoped_args[anno.scope].add(param_name)
                        if param.kind is inspect.Parameter.VAR_POSITIONAL:
                            assert isinstance(bound_sig.arguments[param_name], tuple)
                            for bound_var in bound_sig.arguments[param_name]:
                                bound_vars[anno.scope].add(bound_var)
                        elif param.kind is inspect.Parameter.VAR_KEYWORD:
                            assert isinstance(bound_sig.arguments[param_name], dict)
                            for bound_var in bound_sig.arguments[param_name].values():
                                bound_vars[anno.scope].add(bound_var)
                        else:
                            bound_vars[anno.scope].add(bound_sig.arguments[param_name])
                    elif isinstance(anno, Scoped):
                        scoped_args[anno.scope].add(param_name)
            else:
                unscoped_args.add(param_name)

        # TODO replace this temporary check with more general scope level propagation
        if bound_vars:
            min_scope = min(bound_vars.keys(), default=0)
            scoped_args[min_scope] |= unscoped_args
            max_scope = max(bound_vars.keys(), default=0)
            assert all(s in bound_vars or s > max_scope for s in scoped_args.keys())

        # recursively rename bound variables from innermost to outermost scope
        for scope in sorted(bound_vars.keys()):
            # create fresh variables for each bound variable in the scope
            renaming_map = {var: gensym(var) for var in bound_vars[scope]}
            # get just the arguments that are in the scope
            for name in scoped_args[scope]:
                bound_sig.arguments[name] = tree.map_structure(
                    lambda a: rename(renaming_map, a),
                    bound_sig.arguments[name],
                )

        tm = _embed_registry.dispatch(object)(
            self, tuple(bound_sig.args), tuple(bound_sig.kwargs.items())
        )
        return embed(tm)  # type: ignore

    def __type_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> Type[V]:
        sig = inspect.signature(self.signature)
        bound_sig = sig.bind(*args, **kwargs)
        bound_sig.apply_defaults()

        anno = sig.return_annotation
        if anno is inspect.Signature.empty:
            return typing.cast(Type[V], object)
        elif isinstance(anno, typing.TypeVar):
            # rudimentary but sound special-case type inference sufficient for syntax ops:
            # if the return type annotation is a TypeVar,
            # look for a parameter with the same annotation and return its type,
            # otherwise give up and return Any/object
            for name, param in bound_sig.signature.parameters.items():
                if param.annotation is anno and param.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    arg = bound_sig.arguments[name]
                    tp: Type[V] = type(arg) if not isinstance(arg, type) else arg
                    return tp
            return typing.cast(Type[V], object)
        elif typing.get_origin(anno) is typing.Annotated:
            tp = typing.get_args(anno)[0]
            if not typing.TYPE_CHECKING:
                tp = tp if typing.get_origin(tp) is None else typing.get_origin(tp)
            return tp
        elif typing.get_origin(anno) is not None:
            return typing.get_origin(anno)
        else:
            return anno

    def __scope_rule__(
        self, *args: Q.args, **kwargs: Q.kwargs
    ) -> "Interpretation[T, Type[T]]":

        sig = inspect.signature(self.signature)
        bound_sig = sig.bind(*args, **kwargs)
        bound_sig.apply_defaults()

        bound_vars: dict[Operation[..., T], Callable[..., Type[T]]] = {}
        for param_name, param in bound_sig.signature.parameters.items():
            if typing.get_origin(param.annotation) is typing.Annotated:
                for anno in param.annotation.__metadata__:
                    if isinstance(anno, Bound):
                        if param.kind is inspect.Parameter.VAR_POSITIONAL:
                            for bound_var in bound_sig.arguments[param_name]:
                                bound_vars[bound_var] = bound_var.__type_rule__
                        elif param.kind is inspect.Parameter.VAR_KEYWORD:
                            for bound_var in bound_sig.arguments[param_name].values():
                                bound_vars[bound_var] = bound_var.__type_rule__
                        else:
                            bound_var = bound_sig.arguments[param_name]
                            bound_vars[bound_var] = bound_var.__type_rule__

        return bound_vars

    def __repr__(self):
        return self.signature.__name__


def defop(signature: Callable[Q, V]) -> Operation[Q, V]:
    return _BaseOperation(signature)


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


Interpretation = Mapping[Operation[..., T], Callable[..., V]]


@defop  # type: ignore
def apply(
    intp: Interpretation[S, T], op: Operation[P, S], *args: P.args, **kwargs: P.kwargs
) -> T:
    if op in intp:
        return intp[op](*args, **kwargs)
    elif apply in intp:
        return intp[apply](intp, op, *args, **kwargs)
    else:
        return op.__default_rule__(*args, **kwargs)  # type: ignore


def evaluate(expr: Expr[T], *, intp: Optional[Interpretation[S, T]] = None) -> Expr[T]:
    if intp is None:
        from effectful.internals.runtime import get_interpretation

        intp = get_interpretation()

    match as_term(expr):
        case Term(op, args, kwargs):
            (args, kwargs) = tree.map_structure(
                functools.partial(evaluate, intp=intp), (args, dict(kwargs))
            )
            return apply.__default_rule__(intp, op, *args, **kwargs)  # type: ignore
        case literal:
            return literal


def ctxof(term: Expr[S]) -> Interpretation[T, Type[T]]:
    from effectful.internals.runtime import interpreter

    _ctx: Dict[Operation[..., T], Callable[..., Type[T]]] = {}

    def _update_ctx(_, op, *args, **kwargs):
        _ctx.setdefault(op, op.__type_rule__(*args, **kwargs))
        for bound_var in op.__scope_rule__(*args, **kwargs):
            _ctx.pop(bound_var, None)

    with interpreter({apply: _update_ctx}):  # type: ignore
        evaluate(as_term(term))  # type: ignore

    return _ctx


def to_string(term: Expr[S]) -> str:
    from effectful.internals.runtime import interpreter

    def _to_str(_, op, *args, **kwargs):
        return f"{op}({', '.join(str(a) for a in args)}, {', '.join(f'{k}={v}' for k, v in kwargs.items())})"

    with interpreter({apply: _to_str}):  # type: ignore
        return evaluate(as_term(term))  # type: ignore


def typeof(term: Expr[T]) -> Type[T]:
    from effectful.internals.runtime import interpreter

    with interpreter({apply: lambda _, op, *a, **k: op.__type_rule__(*a, **k)}):  # type: ignore
        return evaluate(term)  # type: ignore


def as_term(value: Expr[T]) -> Expr[T]:
    from effectful.internals.sugar import _as_term_registry

    if isinstance(value, Term):
        return value  # type: ignore
    else:
        impl: Callable[[T], Expr[T]]
        impl = _as_term_registry.dispatch(type(value))  # type: ignore
        return impl(value)
