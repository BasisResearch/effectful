import collections
import functools
import inspect
import typing
from typing import Callable, Generic, Mapping, Optional, Sequence, Type, TypeVar

from typing_extensions import Concatenate, ParamSpec

from effectful.ops.types import Expr, Operation, Term

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


class _BaseOperation(Generic[Q, V], Operation[Q, V]):
    __signature__: inspect.Signature
    __name__: str

    _default: Callable[Q, V]

    def __init__(self, default: Callable[Q, V], *, name: Optional[str] = None):
        functools.update_wrapper(self, default)
        self._default = default
        self.__name__ = name or default.__name__
        self.__signature__ = inspect.signature(default)

    def __eq__(self, other):
        if not isinstance(other, Operation):
            return NotImplemented
        return self._default == other._default

    def __hash__(self):
        return hash(self._default)

    def __default_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> "Expr[V]":
        from effectful.ops.syntax import defdata

        try:
            return self._default(*args, **kwargs)
        except NotImplementedError:
            return typing.cast(
                Callable[Concatenate[Operation[Q, V], Q], Expr[V]], defdata
            )(self, *args, **kwargs)

    def __fvs_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> tuple[
        tuple[collections.abc.Set[Operation], ...],
        dict[str, collections.abc.Set[Operation]],
    ]:
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
                            result_sig.kwargs[name] = {
                                k: param_bound_vars for k in bound_sig.arguments[name]
                            }
                        else:
                            result_sig.arguments[name] = param_bound_vars

        return tuple(result_sig.args), dict(result_sig.kwargs)

    def __type_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> Type[V]:
        sig = inspect.signature(self._default)
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

        ret = f"{self._default.__name__}({args_str}"
        if kwargs:
            ret += f"{', ' if args else ''}"
        ret += f"{kwargs_str})"
        return ret

    def __repr__(self):
        return self._default.__name__


class _BaseTerm(Generic[T], Term[T]):
    _op: Operation[..., T]
    _args: Sequence[Expr]
    _kwargs: Mapping[str, Expr]

    def __init__(
        self,
        op: Operation[..., T],
        *args: Expr,
        **kwargs: Expr,
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
    from effectful.ops.syntax import defdata, deffn, defop

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
            apply: lambda _, op, *a, **k: defdata(op, *a, **k),
            call: call.__default_rule__,
        }
    ):
        body = value(
            *[a() for a in bound_sig.args],
            **{k: v() for k, v in bound_sig.kwargs.items()},
        )

    return deffn(body, *bound_sig.args, **bound_sig.kwargs)
