import collections
import functools
import inspect
import typing
from typing import Callable, Generic, Mapping, Sequence, Set, Type, TypeVar

import tree
from typing_extensions import ParamSpec

from effectful.ops.types import Expr, Interpretation, Operation, Term

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


def _rename_leaf_op(
    v: T, renaming: Mapping[Operation[P, S], Operation[P, S]]
) -> tuple[T, Interpretation[S, S]]:
    return tree.map_structure(
        lambda a: renaming.get(a, a) if isinstance(a, Operation) else a, v
    ), dict(renaming)


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

    def __default_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> Expr[V]:
        from effectful.ops.syntax import NoDefaultRule

        try:
            return self.signature(*args, **kwargs)
        except NoDefaultRule:
            return self.__free_rule__(*args, **kwargs)

    def __free_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> Term[V]:
        from effectful.ops.semantics import apply, evaluate
        from effectful.ops.syntax import defdata

        bound_vars = self.__fvs_rule__(*args, **kwargs)
        if not bound_vars:
            return defdata(_BaseTerm(self, args, kwargs))
        else:
            intp = {
                apply: lambda _, op, *a, **k: defdata(_BaseTerm(op, a, k)),
                **{op: op for op in bound_vars},
            }
            return evaluate(_BaseTerm(self, args, kwargs), intp=intp)

    def __evalctx_rule__(
        self, intp: Interpretation[S, T], *args: Q.args, **kwargs: Q.kwargs
    ) -> tuple[
        Interpretation[S, S | T],
        tuple[tuple[Expr, Interpretation[S, S | T]], ...],
        dict[str, tuple[Expr, Interpretation[S, S | T]]],
    ]:
        from effectful.ops.semantics import coproduct
        from effectful.ops.syntax import Bound, Scoped, defop

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

        if not bound_vars:  # fast path for no bound variables
            return (
                {},
                tuple((a, {}) for a in bound_sig.args),
                {k: (v, {}) for k, v in bound_sig.kwargs.items()},
            )

        # TODO replace this temporary check with more general scope level propagation
        min_scope = min(bound_vars.keys(), default=0)
        scoped_args[min_scope] |= unscoped_args
        max_scope = max(bound_vars.keys(), default=0)
        assert all(s in bound_vars or s > max_scope for s in scoped_args.keys())

        # recursively rename bound variables from innermost to outermost scope
        subs: dict[Operation, Operation] = {}
        for scope in sorted(scoped_args.keys()):
            # create fresh variables for each bound variable in the scope
            subs = coproduct(
                {var: defop(var) for var in bound_vars[scope] if var in intp}, subs
            )

            # get just the arguments that are in the scope
            for name in scoped_args[scope]:
                if sig.parameters[name].kind is inspect.Parameter.VAR_POSITIONAL:
                    bound_sig.arguments[name] = tuple(
                        _rename_leaf_op(a, subs) for a in bound_sig.arguments[name]
                    )
                elif sig.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
                    bound_sig.arguments[name] = {
                        k: _rename_leaf_op(v, subs)
                        for k, v in bound_sig.arguments[name].items()
                    }
                else:
                    bound_sig.arguments[name] = _rename_leaf_op(
                        bound_sig.arguments[name], subs
                    )

        return {}, tuple(bound_sig.args), dict(bound_sig.kwargs)

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
        args_str = ", ".join(map(repr, args)) if args else ""
        kwargs_str = (
            ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items()) if kwargs else ""
        )
        return (
            f"{self.signature.__name__}({args_str}{', ' if args else ''}{kwargs_str})"
        )

    def __repr__(self):
        return self.signature.__name__


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
            raise NotImplementedError(
                f"cannot unembed {value}: parameter {name} is variadic"
            )

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
