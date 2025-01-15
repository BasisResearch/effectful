import collections
import functools
import inspect
import typing
from typing import Callable, Generic, Mapping, Optional, Sequence, Set, Type, TypeVar

import tree
from typing_extensions import ParamSpec

from effectful.ops.types import Expr, Operation, Term

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


def rename(
    subs: Mapping[Operation[..., S], Operation[..., S]],
    leaf_value: V,  # Union[Term[V], Operation[..., V], V],
) -> V:  # Union[Term[V], Operation[..., V], V]:
    from effectful.internals.runtime import interpreter
    from effectful.ops.semantics import apply, evaluate

    if isinstance(leaf_value, Operation):
        return subs.get(leaf_value, leaf_value)  # type: ignore
    elif isinstance(leaf_value, Term):
        with interpreter(
            {apply: lambda _, op, *a, **k: op.__free_rule__(*a, **k), **subs}
        ):
            return evaluate(leaf_value)  # type: ignore
    else:
        return leaf_value


class _BaseOperation(Generic[Q, V], Operation[Q, V]):
    signature: Callable[Q, V]

    def __init__(self, signature: Callable[Q, V], *, name: Optional[str] = None):
        functools.update_wrapper(self, signature)
        self.signature = signature
        self.__name__ = name or signature.__name__

    def __eq__(self, other):
        if not isinstance(other, Operation):
            return NotImplemented
        return self.signature == other.signature

    def __hash__(self):
        return hash(self.signature)

    def __default_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> "Expr[V]":
        try:
            return self.signature(*args, **kwargs)
        except NotImplementedError:
            return self.__free_rule__(*args, **kwargs)

    def __free_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> "Expr[V]":
        from effectful.ops.syntax import Bound, Scoped, defdata, defop

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
            renaming_map = {var: defop(var) for var in bound_vars[scope]}
            # get just the arguments that are in the scope
            for name in scoped_args[scope]:
                bound_sig.arguments[name] = tree.map_structure(
                    lambda a: rename(renaming_map, a),
                    bound_sig.arguments[name],
                )

        return defdata(_BaseTerm(self, bound_sig.args, bound_sig.kwargs))

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

    def __fvs_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> Set[Operation]:
        from effectful.ops.syntax import Bound

        sig = inspect.signature(self.signature)
        bound_sig = sig.bind(*args, **kwargs)
        bound_sig.apply_defaults()

        bound_vars: Set[Operation] = set()
        for param_name, param in bound_sig.signature.parameters.items():
            if typing.get_origin(param.annotation) is typing.Annotated:
                for anno in param.annotation.__metadata__:
                    if isinstance(anno, Bound):
                        if param.kind is inspect.Parameter.VAR_POSITIONAL:
                            for bound_var in bound_sig.arguments[param_name]:
                                bound_vars.add(bound_var)
                        elif param.kind is inspect.Parameter.VAR_KEYWORD:
                            for bound_var in bound_sig.arguments[param_name].values():
                                bound_vars.add(bound_var)
                        else:
                            bound_var = bound_sig.arguments[param_name]
                            bound_vars.add(bound_var)

        return bound_vars

    def __repr_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> str:
        args_str = ", ".join(map(str, args)) if args else ""
        kwargs_str = (
            ", ".join(f"{k}={str(v)}" for k, v in kwargs.items()) if kwargs else ""
        )

        ret = f"{self.signature.__name__}({args_str}"
        if kwargs:
            ret += f"{', ' if args else ''}"
        ret += f"{kwargs_str})"
        return ret

    def __repr__(self):
        return self.__name__


class _BaseTerm(Generic[T], Term[T]):
    _op: Operation[..., T]
    _args: Sequence[Expr]
    _kwargs: Mapping[str, Expr]

    def __init__(
        self,
        op: Operation[..., T],
        args: Sequence[Expr],
        kwargs: Mapping[str, Expr],
    ):
        self._op = op
        self._args = args
        self._kwargs = kwargs

    def __eq__(self, other) -> bool:
        from effectful.ops.syntax import syntactic_eq

        return syntactic_eq(self, other)

    @property
    def op(self):
        return self._op

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs


class _CallableTerm(Generic[P, T], _BaseTerm[collections.abc.Callable[P, T]]):
    def __call__(self, *args: Expr, **kwargs: Expr) -> Expr[T]:
        from effectful.ops.semantics import call

        return call(self, *args, **kwargs)  # type: ignore


def _unembed_callable(value: Callable[P, T]) -> Expr[Callable[P, T]]:
    from effectful.internals.runtime import interpreter
    from effectful.ops.semantics import apply, call
    from effectful.ops.syntax import deffn, defop

    assert not isinstance(value, Term)

    try:
        sig = inspect.signature(value)
    except ValueError:
        return value

    for name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise ValueError(f"cannot unembed {value}: parameter {name} is variadic")

    bound_sig = sig.bind(
        **{name: defop(param.annotation) for name, param in sig.parameters.items()}
    )
    bound_sig.apply_defaults()

    with interpreter(
        {
            apply: lambda _, op, *a, **k: op.__free_rule__(*a, **k),
            call: call.__default_rule__,
        }
    ):
        body = value(
            *[a() for a in bound_sig.args],
            **{k: v() for k, v in bound_sig.kwargs.items()},
        )

    return deffn(body, *bound_sig.args, **bound_sig.kwargs)
