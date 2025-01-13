import collections
import functools
import inspect
import typing
from typing import Callable, Generic, Type, TypeVar

from typing_extensions import ParamSpec

from effectful.ops.types import Expr, Interpretation, Operation

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


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
        from effectful.ops.syntax import NoDefaultRule, defdata

        try:
            return self.signature(*args, **kwargs)
        except NoDefaultRule:
            return defdata(self, *args, **kwargs)

    def __fvs_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> tuple[
        tuple[Interpretation[S, S | T], ...],
        dict[str, Interpretation[S, S | T]],
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
                tuple({} for _ in bound_sig.args),
                {k: {} for k in bound_sig.kwargs},
            )

        # TODO replace this temporary check with more general scope level propagation
        min_scope = min(bound_vars.keys(), default=0)
        scoped_args[min_scope] |= unscoped_args
        max_scope = max(bound_vars.keys(), default=0)
        assert all(s in bound_vars or s > max_scope for s in scoped_args.keys())

        # recursively rename bound variables from innermost to outermost scope
        subs: dict[Operation[..., S], Operation[..., S]] = {}
        for scope in sorted(scoped_args.keys()):
            # create fresh variables for each bound variable in the scope
            subs = coproduct({var: defop(var) for var in bound_vars[scope]}, subs)  # type: ignore

            # get just the arguments that are in the scope
            for name in scoped_args[scope]:
                if sig.parameters[name].kind is inspect.Parameter.VAR_POSITIONAL:
                    bound_sig.arguments[name] = tuple(
                        subs for _ in bound_sig.arguments[name]
                    )
                elif sig.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
                    bound_sig.arguments[name] = {
                        k: subs for k in bound_sig.arguments[name]
                    }
                else:
                    bound_sig.arguments[name] = subs

        return tuple(bound_sig.args), dict(bound_sig.kwargs)

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
        return self.signature.__name__
