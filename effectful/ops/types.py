import abc
import collections.abc
import functools
import inspect
import types
import typing
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import (
    Any,
    Concatenate,
    Protocol,
    _ProtocolMeta,
    overload,
    runtime_checkable,
)


class NotHandled(Exception):
    """Raised by an operation when the operation should remain unhandled."""

    pass


class _CustomSingleDispatchCallable[**P, **Q, S, T]:
    def __init__(
        self, func: Callable[Concatenate[Callable[[type], Callable[Q, S]], P], T]
    ):
        self.func = func
        self._registry = functools.singledispatch(func)
        functools.update_wrapper(self, func)

    @property
    def dispatch(self):
        return self._registry.dispatch

    @property
    def register(self):
        return self._registry.register

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.func(self.dispatch, *args, **kwargs)


class _ClassMethodOpDescriptor(classmethod):
    def __init__(self, define, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._define = define

    def __set_name__(self, owner, name):
        assert not hasattr(self, "_name_on_owner"), "should only be called once"
        self._name_on_owner = f"_descriptorop_{name}"

    def __get__(self, instance, owner: type | None = None):
        owner = owner if owner is not None else type(instance)
        try:
            return owner.__dict__[self._name_on_owner]
        except KeyError:
            bound_op = self._define(super().__get__(instance, owner))
            setattr(owner, self._name_on_owner, bound_op)
            return bound_op


@functools.total_ordering
class Operation[**Q, V]:
    """An abstract class representing an effect that can be implemented by an effect handler.

    .. note::

       Do not instantiate :class:`Operation` directly. Instead, use
       :func:`define` to define operations.

    """

    __signature__: inspect.Signature
    __name__: str
    __default__: Callable[Q, V]

    def __init__(
        self,
        default: Callable[Q, V],
        *,
        name: str | None = None,
        signature: inspect.Signature | None = None,
    ):
        functools.update_wrapper(self, default)

        self.__default__ = default
        self.__name__ = name or default.__name__
        self.__signature__ = signature or inspect.signature(default)

    def __eq__(self, other):
        if not isinstance(other, Operation):
            return NotImplemented
        return self is other

    def __lt__(self, other):
        if not isinstance(other, Operation):
            return NotImplemented
        return id(self) < id(other)

    def __hash__(self):
        return hash(self.__default__)

    @functools.singledispatchmethod
    @classmethod
    def define[**P, T](
        cls: Callable[P, T], default: Callable[Q, V], *, name: str | None = None
    ) -> "Operation[P, T]":
        """Creates a fresh :class:`Operation`.

        :param t: May be a type, callable, or :class:`Operation`. If a type, the
                  operation will have no arguments and return the type. If a
                  callable, the operation will have the same signature as the
                  callable, but with no default rule. If an operation, the
                  operation will be a distinct copy of the operation.
        :param name: Optional name for the operation.
        :returns: A fresh operation.

        .. note::

          The result of :func:`Operation.define` is always fresh (i.e.
          ``Operation.define(f) != Operation.define(f)``).

        **Example usage**:

        * Defining an operation:

          This example defines an operation that selects one of two integers:

          >>> @Operation.define
          ... def select(x: int, y: int) -> int:
          ...     return x

          The operation can be called like a regular function. By default,
          ``select`` returns the first argument:

          >>> select(1, 2)
          1

          We can change its behavior by installing a ``select`` handler:

          >>> from effectful.ops.semantics import handler
          >>> with handler({select: lambda x, y: y}):
          ...     print(select(1, 2))
          2

        * Defining an operation with no default rule:

          We can use :func:`Operation.define` and the :exc:`NotHandled`
          exception to define an operation with no default rule:

          >>> @Operation.define
          ... def add(x: int, y: int) -> int:
          ...     raise NotHandled
          >>> print(str(add(1, 2)))
          add(1, 2)

          When an operation has no default rule, the free rule is used instead,
          which constructs a term of the operation applied to its arguments.
          This feature can be used to conveniently define the syntax of a
          domain-specific language.

        * Defining free variables:

          Passing :func:`Operation.define` a type creates a free variable.

          >>> from effectful.ops.semantics import evaluate
          >>> x = Operation.define(int, name='x')
          >>> y = x() + 1

          ``y`` is free in ``x``, so it is not fully evaluated:

          >>> print(str(y))
          __add__(x(), 1)

          We bind ``x`` by installing a handler for it:

          >>> with handler({x: lambda: 2}):
          ...     print(evaluate(y))
          3

          .. note::

            Because the result of :func:`Operation.define` is always fresh, it's
            important to be careful with variable identity.

            Two operations with the same name that come from different calls to
            ``Operation.define`` are not equal:

            >>> x1 = Operation.define(int, name='x')
            >>> x2 = Operation.define(int, name='x')
            >>> x1 == x2
            False

            This means that to correctly bind a variable, you must use the same
            operation object. In this example, ``scale`` returns a term with a
            free variable ``x``:

            >>> x = Operation.define(float, name='x')
            >>> def scale(a: float) -> float:
            ...     return x() * a

            Binding the variable ``x`` as follows does not work:

            >>> term = scale(3.0)
            >>> fresh_x = Operation.define(float, name='x')
            >>> with handler({fresh_x: lambda: 2.0}):
            ...     print(str(evaluate(term)))
            __mul__(x(), 3.0)

            Only the original operation object will work:

            >>> from effectful.ops.semantics import fvsof
            >>> with handler({x: lambda: 2.0}):
            ...     print(evaluate(term))
            6.0

        * Defining a fresh :class:`Operation`:

          Passing :func:`Operation.define` an :class:`Operation` creates a fresh
          operation with the same name and signature, but no default rule.

          >>> fresh_select = Operation.define(select)
          >>> print(str(fresh_select(1, 2)))
          select(1, 2)

          The new operation is distinct from the original:

          >>> with handler({select: lambda x, y: y}):
          ...     print(select(1, 2), fresh_select(1, 2))
          2 select(1, 2)

          >>> with handler({fresh_select: lambda x, y: y}):
          ...     print(select(1, 2), fresh_select(1, 2))
          1 2

        """
        raise NotImplementedError

    @define.register(classmethod)
    @classmethod
    def _define_classmethod(cls, default, **kwargs):
        return Operation._ClassMethodOpDescriptor(cls.define, default.__func__)

    @define.register(
        typing.cast(type[collections.abc.Callable], collections.abc.Callable)
    )
    @classmethod
    def _define_callable[**P, T](
        cls, t: Callable[P, T], *, name: str | None = None, **kwargs
    ) -> "Operation[P, T]":
        if isinstance(t, Operation):

            @functools.wraps(t)
            def func(*args, **kwargs):
                raise NotHandled

            name = name or getattr(t, "__name__", str(t))
            op = cls.define(func, name=name, **kwargs)
        else:
            op = cls(t, name=name, **kwargs)  # type: ignore[arg-type]

        return op  # type: ignore[return-value]

    @define.register(type)
    @define.register(typing.cast(type, types.GenericAlias))
    @define.register(typing.cast(type, typing._GenericAlias))  # type: ignore[attr-defined]
    @define.register(typing.cast(type, types.UnionType))
    @classmethod
    def _define_type[T](
        cls, t: type[T], *, name: str | None = None
    ) -> "Operation[[], T]":
        def func():
            raise NotHandled

        sig = inspect.Signature(return_annotation=t)
        name = name or t.__name__
        return typing.cast(Operation[[], T], cls.define(func, name=name, signature=sig))

    @define.register(types.BuiltinFunctionType)
    @classmethod
    def _define_builtinfunctiontype[**P, T](
        cls, t: Callable[P, T], *, name: str | None = None
    ) -> "Operation[P, T]":
        @functools.wraps(t)
        def func(*args, **kwargs):
            from effectful.ops.semantics import fvsof

            if not fvsof((args, kwargs)):
                return t(*args, **kwargs)
            else:
                raise NotHandled

        return typing.cast(Operation[P, T], cls.define(func, name=name))

    @define.register(staticmethod)
    @classmethod
    def _define_staticmethod[**P, T](cls, t: "staticmethod[P, T]", **kwargs):
        return staticmethod(cls.define(t.__func__, **kwargs))

    @define.register(functools.singledispatchmethod)
    @classmethod
    def _define_singledispatchmethod(cls, default, **kwargs):
        if isinstance(default.func, classmethod):
            raise NotImplementedError("Operations as classmethod are not yet supported")

        @functools.wraps(default.func)
        def _wrapper(obj, *args, **kwargs):
            return default.__get__(obj)(*args, **kwargs)

        op = cls.define(_wrapper, **kwargs)
        op.register = default.register
        op.__isabstractmethod__ = default.__isabstractmethod__
        return op

    @define.register(_CustomSingleDispatchCallable)
    @classmethod
    def _defop_customsingledispatchcallable(
        cls, default: _CustomSingleDispatchCallable, **kwargs
    ):
        sig = inspect.signature(functools.partial(default.func, None))
        op = cls._define_callable(default, signature=sig)
        op.dispatch = default._registry.dispatch
        op.register = default._registry.register
        return op

    @typing.final
    def __default_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> "Expr[V]":
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
        """Returns the type of the operation applied to arguments.

        .. note::

           The arguments are expected to be either
           :class:`effectful.internals.unification.Box`ed types or collections
           containing values and :class:`effectful.internals.unification.Box`ed
           types. Callers are expected to apply the appropriate boxing. Boxing
           the input types prevents confusion between types and values and
           allows for terms that compute on type-valued arguments.

        """
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

        type_args = tuple(nested_type(a).value for a in args)
        type_kwargs = {k: nested_type(v).value for k, v in kwargs.items()}
        bound_sig = self.__signature__.bind(*type_args, **type_kwargs)
        subst_type = substitute(return_anno, unify(self.__signature__, bound_sig))
        return typing.cast(type[V], subst_type)

    @typing.final
    def __fvs_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> inspect.BoundArguments:
        """Returns the sets of variables that appear free in each argument and
        keyword argument but not in the result of the operation, i.e. the
        variables bound by the operation.

        These are used by :func:`fvsof` to determine the free variables of a
        term by subtracting the results of this method from the free variables
        of the subterms, allowing :func:`fvsof` to be implemented in terms of
        :func:`evaluate` .

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

    def __call__(self, *args: Q.args, **kwargs: Q.kwargs) -> V:
        from effectful.internals.runtime import get_interpretation
        from effectful.ops.semantics import apply

        intp = get_interpretation()

        self_handler = intp.get(self)
        if self_handler is not None:
            return self_handler(*args, **kwargs)

        class_apply_handler = intp.get(type(self).apply)
        if class_apply_handler is not None:
            return class_apply_handler(self, *args, **kwargs)

        global_apply_handler = intp.get(apply)
        if global_apply_handler is not None:
            return global_apply_handler(self, *args, **kwargs)

        return type(self).apply(self, *args, **kwargs)  # type: ignore[return-value]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__name__}, {self.__signature__})"

    def __str__(self):
        return self.__name__

    def __get__(self, instance, owner):
        if instance is not None:
            # This is an instance-level operation, so we need to bind the instance
            return functools.partial(self, instance)
        else:
            # This is a static operation, so we return the operation itself
            return self

    @classmethod
    def apply[**A, B](
        cls, op: "Operation[A, B]", *args: A.args, **kwargs: A.kwargs
    ) -> "Expr[B]":
        """Apply an operation to arguments.

        In subclasses of Operation, `apply` is an operation that may be handled.

        """
        return op.__default_rule__(*args, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.apply = cls.define(cls.apply, name=f"{cls.__name__}_apply")


if typing.TYPE_CHECKING:

    @runtime_checkable
    class _OperationDefine(Protocol):
        def __call__[**Q, V](
            self, op: Callable[Q, V], *, name: str | None = None
        ) -> Operation[Q, V]: ...

    assert isinstance(Operation.define, _OperationDefine)


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
    def args(self) -> Sequence["Expr[Any]"]:
        """Abstract property for the arguments."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def kwargs(self) -> Mapping[str, "Expr[Any]"]:
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
