import collections.abc
import dataclasses
import functools
import inspect
import random
import types
import typing
from typing import Annotated, Callable, Generic, Optional, Type, TypeVar

import tree
from typing_extensions import Concatenate, ParamSpec

from effectful.ops.types import Annotation, Expr, Interpretation, Operation, Term

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


@dataclasses.dataclass
class Scoped(Annotation):
    """
    A special type annotation that indicates the relative scope of a parameter
    in the signature of an :class:`Operation` created with :func:`defop` .

    :class:`Scoped` makes it easy to describe higher-order :class:`Operation` s
    that take other :class:`Term` s and :class:`Operation` s as arguments,
    inspired by a number of recent proposals to view syntactic variables
    as algebraic effects and environments as effect handlers.

    As a result, in ``effectful`` many complex higher-order programming constructs,
    such as lambda-abstraction, let-binding, loops, try-catch exception handling,
    nondeterminism, capture-avoiding substitution and algebraic effect handling,
    can be expressed uniformly using :func:`defop` as ordinary :class:`Operation` s
    and evaluated or transformed using generalized effect handlers that respect
    the scoping semantics of the operations.

    .. warning::

        :class:`Scoped` instances are typically constructed using indexing
        syntactic sugar borrowed from generic types like :class:`typing.Generic` .
        For example, ``Scoped[A]`` desugars to a :class:`Scoped` instances
        with ``ordinal={A}``, and ``Scoped[A | B]`` desugars to a :class:`Scoped`
        instance with ``ordinal={A, B}`` .

        However, :class:`Scoped` is not a generic type, and the set of :class:`typing.TypeVar` s
        used for the :class:`Scoped` annotations in a given operation must be disjoint
        from the set of :class:`typing.TypeVar` s used for generic types of the parameters.

    **Example usage**:

    We illustrate the use of :class:`Scoped` with a few case studies of classical
    syntactic variable binding constructs expressed as :class:`Operation` s.

    >>> from typing import Annotated, TypeVar
    >>> from effectful.ops.syntax import Scoped, defop
    >>> from effectful.ops.semantics import fvsof
    >>> from effectful.handlers.numbers import add
    >>> A, B, S, T = TypeVar('A'), TypeVar('B'), TypeVar('S'), TypeVar('T')
    >>> x, y = defop(int, name='x'), defop(int, name='y')

    * For example, we can define a higher-order operation :func:`Lambda`
      that takes an :class:`Operation` representing a bound syntactic variable
      and a :class:`Term` representing the body of an anonymous function,
      and returns a :class:`Term` representing a lambda function:

      >>> @defop
      ... def Lambda(
      ...     var: Annotated[Operation[[], S], Scoped[A]],
      ...     body: Annotated[T, Scoped[A | B]]
      ... ) -> Annotated[Callable[[S], T], Scoped[B]]:
      ...     raise NotImplementedError

    * The :class:`Scoped` annotation is used here to indicate that the argument ``var``
      passed to :func:`Lambda` may appear free in ``body``, but not in the resulting function.
      In other words, it is bound by :func:`Lambda`:

      >>> assert x not in fvsof(Lambda(x, add(x(), 1)))

      However, variables in ``body`` other than ``var`` still appear free in the result:

      >>> assert y in fvsof(Lambda(x, add(x(), y())))

    * :class:`Scoped` can also be used with variadic arguments and keyword arguments.
      For example, we can define a generalized :func:`LambdaN` that takes a variable
      number of arguments and keyword arguments:

      >>> @defop
      ... def LambdaN(
      ...     body: Annotated[T, Scoped[A | B]]
      ...     *args: Annotated[Operation[[], S], Scoped[A]],
      ...     **kwargs: Annotated[Operation[[], S], Scoped[A]]
      ... ) -> Annotated[Callable[..., T], Scoped[B]]:
      ...     raise NotImplementedError

      This is equivalent to the built-in :class:`Operation` :func:`deffn`:

      >>> assert not {x, y} & fvsof(LambdaN(add(x(), y()), x, y))

    * :class:`Scoped` and :func:`defop` can also express more complex scoping semantics.
      For example, we can define a :func:`Let` operation that binds a variable in
      a :class:`Term` ``body`` to a ``value`` that may be another possibly open :class:`Term` :

      >>> @defop
      ... def Let(
      ...     var: Annotated[Operation[[], S], Scoped[A]],
      ...     val: Annotated[S, Scoped[B]],
      ...     body: Annotated[T, Scoped[A | B]]
      ... ) -> Annotated[T, Scoped[B]]:
      ...     raise NotImplementedError

      Here the variable ``var`` is bound by :func:`Let` in `body` but not in ``val`` :

      >>> assert x not in fvsof(Let(x, add(y(), 1), add(x(), y())))
      >>> assert {x, y} in fvsof(Let(x, add(y(), x()), add(x(), y())))

      This is reflected in the free variables of subterms of the result:

      >>> assert x in fvsof(Let(x, add(x(), y()), add(x(), y())).args[1])
      >>> assert x not in fvsof(Let(x, add(y(), 1), add(x(), y())).args[2])
    """

    ordinal: collections.abc.Set

    def __class_getitem__(cls, item: TypeVar | typing._SpecialForm):
        assert not isinstance(item, tuple), "can only be in one scope"
        if isinstance(item, typing.TypeVar):
            return cls(ordinal=frozenset({item}))
        elif typing.get_origin(item) is typing.Union and typing.get_args(item):
            return cls(ordinal=frozenset(typing.get_args(item)))
        else:
            raise TypeError(
                f"expected TypeVar or non-empty Union of TypeVars, but got {item}"
            )

    @staticmethod
    def _param_is_var(param: type | inspect.Parameter) -> bool:
        """
        Helper function that checks if a parameter is annotated as an :class:`Operation` .

        :param param: The parameter to check.
        :returns: ``True`` if the parameter is an :class:`Operation` , ``False`` otherwise.
        """
        if isinstance(param, inspect.Parameter):
            param = param.annotation
        if typing.get_origin(param) is Annotated:
            param = typing.get_args(param)[0]
        if typing.get_origin(param) is not None:
            param = typing.cast(type, typing.get_origin(param))
        return isinstance(param, type) and issubclass(param, Operation)

    @classmethod
    def _get_param_ordinal(cls, param: type | inspect.Parameter) -> collections.abc.Set:
        """
        Given a type or parameter, extracts the ordinal from its :class:`Scoped` annotation.

        :param param: The type or signature parameter to extract the ordinal from.
        :returns: The ordinal typevars.
        """
        if isinstance(param, inspect.Parameter):
            return cls._get_param_ordinal(param.annotation)
        elif typing.get_origin(param) is Annotated:
            for a in typing.get_args(param)[1:]:
                if isinstance(a, cls):
                    return a.ordinal
            return set()
        else:
            return set()

    @classmethod
    def _get_root_ordinal(cls, sig: inspect.Signature) -> collections.abc.Set:
        """
        Given a signature, computes the intersection of all :class:`Scoped` annotations.

        :param sig: The signature to check.
        :returns: The intersection of the `ordinal`s of all :class:`Scoped` annotations.
        """
        return set(cls._get_param_ordinal(sig.return_annotation)).intersection(
            *(cls._get_param_ordinal(p) for p in sig.parameters.values())
        )

    @classmethod
    def _get_fresh_ordinal(cls, *, name: str = "RootScope") -> collections.abc.Set:
        return {TypeVar(name)}

    @classmethod
    def _check_has_single_scope(cls, sig: inspect.Signature) -> bool:
        """
        Checks if each parameter has at most one :class:`Scoped` annotation.

        :param sig: The signature to check.
        :returns: True if each parameter has at most one :class:`Scoped` annotation, False otherwise.
        """
        # invariant: at most one Scope annotation per parameter
        return not any(
            len([a for a in p.annotation.__metadata__ if isinstance(a, cls)]) > 1
            for p in sig.parameters.values()
            if typing.get_origin(p.annotation) is Annotated
        )

    @classmethod
    def _check_no_typevar_overlap(cls, sig: inspect.Signature) -> bool:
        """
        Checks if there is no overlap between ordinal typevars and generic ones.

        :param sig: The signature to check.
        :returns: True if there is no overlap between ordinal typevars and generic ones, False otherwise.
        """

        def _get_free_type_vars(
            tp: type | typing._SpecialForm | inspect.Parameter | tuple | list,
        ) -> collections.abc.Set[TypeVar]:
            if isinstance(tp, TypeVar):
                return {tp}
            elif isinstance(tp, (tuple, list)):
                return set().union(*map(_get_free_type_vars, tp))
            elif isinstance(tp, inspect.Parameter):
                return _get_free_type_vars(tp.annotation)
            elif typing.get_origin(tp) is Annotated:
                return _get_free_type_vars(typing.get_args(tp)[0])
            elif typing.get_origin(tp) is not None:
                return _get_free_type_vars(typing.get_args(tp))
            else:
                return set()

        # invariant: no overlap between ordinal typevars and generic ones
        free_type_vars = _get_free_type_vars(
            (sig.return_annotation, *sig.parameters.values())
        )
        return all(
            free_type_vars.isdisjoint(cls._get_param_ordinal(p))
            for p in (
                sig.return_annotation,
                *sig.parameters.values(),
            )
        )

    @classmethod
    def _check_no_boundvars_in_result(cls, sig: inspect.Signature) -> bool:
        """
        Checks that no bound variables would appear free in the return value.

        :param sig: The signature to check.
        :returns: True if no bound variables would appear free in the return value, False otherwise.

        .. note::

            This is used as a post-condition for :func:`infer_annotations`.
            However, it is not a necessary condition for the correctness of the
            `Scope` annotations of an operation - our current implementation
            merely does not extend to cases where this condition is true.
        """
        root_ordinal = cls._get_root_ordinal(sig)
        return_ordinal = cls._get_param_ordinal(sig.return_annotation)
        return not any(
            root_ordinal < cls._get_param_ordinal(p) <= return_ordinal
            for p in sig.parameters.values()
            if cls._param_is_var(p)
        )

    @classmethod
    def infer_annotations(cls, sig: inspect.Signature) -> inspect.Signature:
        """
        Given a :class:`inspect.Signature` for an :class:`Operation` for which
        only some :class:`inspect.Parameter` s have manual :class:`Scoped` annotations,
        computes a new signature with :class:`Scoped` annotations attached to each parameter,
        including the return type annotation.

        The new annotations are inferred by joining the manual annotations with a
        fresh root scope. The root scope is the intersection of all :class:`Scoped`
        annotations in the resulting :class:`inspect.Signature` object.

        :class`Operation` s in this root scope are free in the result and in all arguments.

        :param sig: The signature of the operation.
        :returns: A new signature with inferred :class:`Scoped` annotations.
        """
        # pre-conditions
        assert cls._check_has_single_scope(sig)
        assert cls._check_no_typevar_overlap(sig)
        assert cls._check_no_boundvars_in_result(sig)

        root_ordinal = cls._get_root_ordinal(sig)
        if not root_ordinal:
            root_ordinal = cls._get_fresh_ordinal()

        # add missing Scoped annotations and join everything with the root scope
        new_annos: list[type | typing._SpecialForm] = []
        for anno in (
            sig.return_annotation,
            *(p.annotation for p in sig.parameters.values()),
        ):
            new_scope = cls(ordinal=cls._get_param_ordinal(anno) | root_ordinal)
            if typing.get_origin(anno) is Annotated:
                new_anno = typing.get_args(anno)[0]
                new_anno = Annotated[new_anno, new_scope]
                for other in typing.get_args(anno)[1:]:
                    if not isinstance(other, cls):
                        new_anno = Annotated[new_anno, other]
            else:
                new_anno = Annotated[anno, new_scope]

            new_annos.append(new_anno)

        # construct a new Signature structure with the inferred annotations
        new_return_anno, new_annos = new_annos[0], new_annos[1:]
        inferred_sig = sig.replace(
            parameters=[
                p.replace(annotation=a)
                for p, a in zip(sig.parameters.values(), new_annos)
            ],
            return_annotation=new_return_anno,
        )

        # post-conditions
        assert cls._get_root_ordinal(inferred_sig) == root_ordinal != set()
        return inferred_sig

    def analyze(self, bound_sig: inspect.BoundArguments) -> frozenset[Operation]:
        """
        Computes a set of bound variables given a signature with bound arguments.

        The :func:`analyze` methods of :class:`Scoped` annotations that appear on
        the signature of an :class:`Operation` are used by :func:`defop` to generate
        implementations of :func:`Operation.__fvs_rule__` underlying alpha-renaming
        in :func:`defterm` and :func:`defdata` and free variable sets in :func:`fvsof` .

        Specifically, the :func:`analyze` method of the :class:`Scoped` annotation
        of a parameter computes the set of bound variables in that parameter's value.
        The :func:`Operation.__fvs_rule__` method generated by :func:`defop` simply
        extracts the annotation of each parameter, calls :func:`analyze` on the value
        given for the corresponding parameter in ``bound_sig`` , and returns the results.

        :param bound_sig: The :class:`inspect.Signature` of an :class:`Operation`
            together with values for all of its arguments.
        :returns: A set of bound variables.
        """
        bound_vars: frozenset[Operation] = frozenset()
        return_ordinal = self._get_param_ordinal(bound_sig.signature.return_annotation)
        for name, param in bound_sig.signature.parameters.items():
            param_ordinal = self._get_param_ordinal(param)
            if (
                self._param_is_var(param)
                and param_ordinal <= self.ordinal
                and not param_ordinal <= return_ordinal
            ):
                if param.kind is inspect.Parameter.VAR_POSITIONAL:
                    # pre-condition: all bound variables should be distinct
                    assert len(bound_sig.arguments[name]) == len(
                        set(bound_sig.arguments[name])
                    )
                    param_bound_vars = {*bound_sig.arguments[name]}
                elif param.kind is inspect.Parameter.VAR_KEYWORD:
                    # pre-condition: all bound variables should be distinct
                    assert len(bound_sig.arguments[name].values()) == len(
                        set(bound_sig.arguments[name].values())
                    )
                    param_bound_vars = {*bound_sig.arguments[name].values()}
                else:
                    param_bound_vars = {bound_sig.arguments[name]}

                # pre-condition: all bound variables should be distinct
                assert not bound_vars & param_bound_vars

                bound_vars |= param_bound_vars

        return bound_vars


@functools.singledispatch
def defop(t: Callable[P, T], *, name: Optional[str] = None) -> Operation[P, T]:
    """Creates a fresh :class:`Operation`.

    :param t: May be a type, callable, or :class:`Operation`. If a type, the
              operation will have no arguments and return the type. If a callable,
              the operation will have the same signature as the callable, but with
              no default rule. If an operation, the operation will be a distinct
              copy of the operation.
    :param name: Optional name for the operation.
    :returns: A fresh operation.

    .. note::

      The result of :func:`defop` is always fresh (i.e. ``defop(f) != defop(f)``).

    **Example usage**:

    * Defining an operation:

      This example defines an operation that selects one of two integers:

      >>> @defop
      ... def select(x: int, y: int) -> int:
      ...     return x

      The operation can be called like a regular function. By default, ``select``
      returns the first argument:

      >>> select(1, 2)
      1

      We can change its behavior by installing a ``select`` handler:

      >>> from effectful.ops.semantics import handler
      >>> with handler({select: lambda x, y: y}):
      ...     print(select(1, 2))
      2

    * Defining an operation with no default rule:

      We can use :func:`defop` and the
      :exc:`NotImplementedError` exception to define an
      operation with no default rule:

      >>> @defop
      ... def add(x: int, y: int) -> int:
      ...     raise NotImplementedError
      >>> add(1, 2)
      add(1, 2)

      When an operation has no default rule, the free rule is used instead, which
      constructs a term of the operation applied to its arguments. This feature
      can be used to conveniently define the syntax of a domain-specific language.

    * Defining free variables:

      Passing :func:`defop` a type is a handy way to create a free variable.

      >>> import effectful.handlers.operator
      >>> from effectful.ops.semantics import evaluate
      >>> x = defop(int, name='x')
      >>> y = x() + 1

      ``y`` is free in ``x``, so it is not fully evaluated:

      >>> y
      add(x(), 1)

      We bind ``x`` by installing a handler for it:

      >>> with handler({x: lambda: 2}):
      ...     print(evaluate(y))
      3

      .. note::

        Because the result of :func:`defop` is always fresh, it's important to
        be careful with variable identity.

        Two variables with the same name are not equal:

        >>> x1 = defop(int, name='x')
        >>> x2 = defop(int, name='x')
        >>> x1 == x2
        False

        This means that to correctly bind a variable, you must use the same
        operation object. In this example, ``scale`` returns a term with a free
        variable ``x``:

        >>> import effectful.handlers.operator
        >>> def scale(a: float) -> float:
        ...     x = defop(float, name='x')
        ...     return x() * a

        Binding the variable ``x`` by creating a fresh operation object does not

        >>> term = scale(3.0)
        >>> x = defop(float, name='x')
        >>> with handler({x: lambda: 2.0}):
        ...     print(evaluate(term))
        mul(x(), 3.0)

        This does:

        >>> from effectful.ops.semantics import fvsof
        >>> correct_x = [v for v in fvsof(term) if str(x) == 'x'][0]
        >>> with handler({correct_x: lambda: 2.0}):
        ...     print(evaluate(term))
        6.0

    * Defining a fresh :class:`Operation`:

      Passing :func:`defop` an :class:`Operation` creates a fresh operation with
      the same name and signature, but no default rule.

      >>> fresh_select = defop(select)
      >>> fresh_select(1, 2)
      select(1, 2)

      The new operation is distinct from the original:

      >>> with handler({select: lambda x, y: y}):
      ...     print(select(1, 2), fresh_select(1, 2))
      2 select(1, 2)

      >>> with handler({fresh_select: lambda x, y: y}):
      ...     print(select(1, 2), fresh_select(1, 2))
      1 2

    """
    raise NotImplementedError(f"expected type or callable, got {t}")


@defop.register(typing.cast(Type[collections.abc.Callable], collections.abc.Callable))
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
        return self is other

    def __hash__(self):
        return hash(self._default)

    def __default_rule__(self, *args: Q.args, **kwargs: Q.kwargs) -> "Expr[V]":
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

        ret = f"{self.__name__}({args_str}"
        if kwargs:
            ret += f"{', ' if args else ''}"
        ret += f"{kwargs_str})"
        return ret

    def __repr__(self):
        return self.__name__


@defop.register(Operation)
def _(t: Operation[P, T], *, name: Optional[str] = None) -> Operation[P, T]:

    @functools.wraps(t)
    def func(*args, **kwargs):
        raise NotImplementedError

    if name is None:
        name = (
            getattr(t, "__name__", str(t))[:10000] + f"__{random.randint(0, 1 << 32)}"
        )
    return defop(func, name=name)


@defop.register(type)
def _(t: Type[T], *, name: Optional[str] = None) -> Operation[[], T]:
    def func() -> t:  # type: ignore
        raise NotImplementedError

    if name is None:
        name = t.__name__ + f"__{random.randint(0, 1 << 32)}"
    return typing.cast(Operation[[], T], defop(func, name=name))


@defop.register(types.BuiltinFunctionType)
def _(t: Callable[P, T], *, name: Optional[str] = None) -> Operation[P, T]:

    @functools.wraps(t)
    def func(*args, **kwargs):
        if not any(isinstance(a, Term) for a in tree.flatten((args, kwargs))):
            return t(*args, **kwargs)
        else:
            raise NotImplementedError

    return defop(func, name=name)


@defop
def deffn(
    body: Annotated[T, Scoped[S]],
    *args: Annotated[Operation, Scoped[S]],
    **kwargs: Annotated[Operation, Scoped[S]],
) -> Callable[..., T]:
    """An operation that represents a lambda function.

    :param body: The body of the function.
    :type body: T
    :param args: Operations representing the positional arguments of the function.
    :type args: Operation
    :param kwargs: Operations representing the keyword arguments of the function.
    :type kwargs: Operation
    :returns: A callable term.
    :rtype: Callable[..., T]

    :func:`deffn` terms are eliminated by the :func:`call` operation, which
    performs beta-reduction.

    **Example usage**:

    Here :func:`deffn` is used to define a term that represents the function
    ``lambda x, y=1: 2 * x + y``:

    >>> import effectful.handlers.operator
    >>> x, y = defop(int, name='x'), defop(int, name='y')
    >>> term = deffn(2 * x() + y(), x, y=y)
    >>> term
    deffn(add(mul(2, x()), y()), x, y=y)
    >>> term(3, y=4)
    10

    .. note::

      In general, avoid using :func:`deffn` directly. Instead, use
      :func:`defterm` to convert a function to a term because it will
      automatically create the right free variables.

    """
    raise NotImplementedError


class _CustomSingleDispatchCallable(Generic[P, Q, S, T]):
    def __init__(
        self, func: Callable[Concatenate[Callable[[type], Callable[Q, S]], P], T]
    ):
        self._func = func
        self._registry = functools.singledispatch(func)
        functools.update_wrapper(self, func)

    @property
    def dispatch(self):
        return self._registry.dispatch

    @property
    def register(self):
        return self._registry.register

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self._func(self.dispatch, *args, **kwargs)


@_CustomSingleDispatchCallable
def defterm(__dispatch: Callable[[type], Callable[[T], Expr[T]]], value: T):
    """Convert a value to a term, using the type of the value to dispatch.

    :param value: The value to convert.
    :type value: T
    :returns: A term.
    :rtype: Expr[T]

    **Example usage**:

    :func:`defterm` can be passed a function, and it will convert that function
    to a term by calling it with appropriately typed free variables:

    >>> def incr(x: int) -> int:
    ...     return x + 1
    >>> term = defterm(incr)
    >>> term
    deffn(add(int(), 1), int)
    >>> term(2)
    3

    """
    if isinstance(value, Term):
        return value
    else:
        return __dispatch(type(value))(value)


@_CustomSingleDispatchCallable
def defdata(
    __dispatch: Callable[[type], Callable[..., Expr[T]]],
    op: Operation[..., T],
    *args,
    **kwargs,
) -> Expr[T]:
    """Constructs a Term that is an instance of its semantic type.

    :returns: An instance of ``T``.
    :rtype: Expr[T]

    This function is the only way to construct a :class:`Term` from an :class:`Operation`.

    .. note::

      This function is not likely to be called by users of the effectful
      library, but they may wish to register implementations for additional
      types.

    **Example usage**:

    This is how callable terms are implemented:

    .. code-block:: python

      class _CallableTerm(Generic[P, T], Term[collections.abc.Callable[P, T]]):
          def __init__(
              self,
              op: Operation[..., T],
              *args: Expr,
              **kwargs: Expr,
          ):
              self._op = op
              self._args = args
              self._kwargs = kwargs

          @property
          def op(self):
              return self._op

          @property
          def args(self):
              return self._args

          @property
          def kwargs(self):
              return self._kwargs

          def __call__(self, *args: Expr, **kwargs: Expr) -> Expr[T]:
              from effectful.ops.semantics import call

              return call(self, *args, **kwargs)

      @defdata.register(collections.abc.Callable)
      def _(op, *args, **kwargs):
          return _CallableTerm(op, *args, **kwargs)

    When an Operation whose return type is `Callable` is passed to :func:`defdata`,
    it is reconstructed as a :class:`_CallableTerm`, which implements the :func:`__call__` method.
    """
    from effectful.ops.semantics import apply, evaluate, typeof

    arg_ctxs, kwarg_ctxs = op.__fvs_rule__(*args, **kwargs)
    renaming = {
        var: defop(var)
        for bound_vars in (*arg_ctxs, *kwarg_ctxs.values())
        for var in bound_vars
    }

    args_, kwargs_ = list(args), dict(kwargs)
    for i, (v, c) in (
        *enumerate(zip(args, arg_ctxs)),
        *{k: (v, kwarg_ctxs[k]) for k, v in kwargs.items()}.items(),
    ):
        if c:
            v = tree.map_structure(
                lambda a: renaming.get(a, a) if isinstance(a, Operation) else a, v
            )
            res = evaluate(
                v,
                intp={
                    apply: lambda _, op, *a, **k: defdata(op, *a, **k),
                    **{op: renaming[op] for op in c},
                },
            )
            if isinstance(i, int):
                args_[i] = res
            elif isinstance(i, str):
                kwargs_[i] = res

    tp: Type[T] = typeof(
        __dispatch(typing.cast(Type[T], object))(op, *args_, **kwargs_)
    )
    return __dispatch(tp)(op, *args_, **kwargs_)


@defterm.register(object)
@defterm.register(Operation)
@defterm.register(Term)
@defterm.register(type)
@defterm.register(types.BuiltinFunctionType)
def _(value: T) -> T:
    return value


@defdata.register(object)
class _BaseTerm(Generic[T], Term[T]):
    _op: Operation[..., T]
    _args: collections.abc.Sequence[Expr]
    _kwargs: collections.abc.Mapping[str, Expr]

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


@defdata.register(collections.abc.Callable)
class _CallableTerm(Generic[P, T], _BaseTerm[collections.abc.Callable[P, T]]):
    def __call__(self, *args: Expr, **kwargs: Expr) -> Expr[T]:
        from effectful.ops.semantics import call

        return call(self, *args, **kwargs)  # type: ignore


@defterm.register(collections.abc.Callable)
def _(value: Callable[P, T]) -> Expr[Callable[P, T]]:
    from effectful.internals.runtime import interpreter
    from effectful.ops.semantics import apply, call

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


def syntactic_eq(x: Expr[T], other: Expr[T]) -> bool:
    """Syntactic equality, ignoring the interpretation of the terms.

    :param x: A term.
    :type x: Expr[T]
    :param other: Another term.
    :type other: Expr[T]
    :returns: ``True`` if the terms are syntactically equal and ``False`` otherwise.
    """
    if isinstance(x, Term) and isinstance(other, Term):
        op, args, kwargs = x.op, x.args, x.kwargs
        op2, args2, kwargs2 = other.op, other.args, other.kwargs
        try:
            tree.assert_same_structure(
                (op, args, kwargs), (op2, args2, kwargs2), check_types=True
            )
        except (TypeError, ValueError):
            return False
        return all(
            tree.flatten(
                tree.map_structure(
                    syntactic_eq, (op, args, kwargs), (op2, args2, kwargs2)
                )
            )
        )
    elif isinstance(x, Term) or isinstance(other, Term):
        return False
    else:
        return x == other


class ObjectInterpretation(Generic[T, V], Interpretation[T, V]):
    """A helper superclass for defining an ``Interpretation`` of many
    :class:`~effectful.ops.types.Operation` instances with shared state or behavior.

    You can mark specific methods in the definition of an
    :class:`ObjectInterpretation` with operations using the :func:`implements`
    decorator. The :class:`ObjectInterpretation` object itself is an
    ``Interpretation`` (mapping from :class:`~effectful.ops.types.Operation` to :class:`~typing.Callable`)

    >>> from effectful.ops.semantics import handler
    >>> @defop
    ... def read_box():
    ...     pass
    ...
    >>> @defop
    ... def write_box(new_value):
    ...     pass
    ...
    >>> class StatefulBox(ObjectInterpretation):
    ...     def __init__(self, init=None):
    ...         super().__init__()
    ...         self.stored = init
    ...     @implements(read_box)
    ...     def whatever(self):
    ...         return self.stored
    ...     @implements(write_box)
    ...     def write_box(self, new_value):
    ...         self.stored = new_value
    ...
    >>> first_box = StatefulBox(init="First Starting Value")
    >>> second_box = StatefulBox(init="Second Starting Value")
    >>> with handler(first_box):
    ...     print(read_box())
    ...     write_box("New Value")
    ...     print(read_box())
    ...
    First Starting Value
    New Value
    >>> with handler(second_box):
    ...     print(read_box())
    Second Starting Value
    >>> with handler(first_box):
    ...     print(read_box())
    New Value

    """

    # This is a weird hack to get around the fact that
    # the default meta-class runs __set_name__ before __init__subclass__.
    # We basically store the implementations here temporarily
    # until __init__subclass__ is called.
    # This dict is shared by all `Implementation`s,
    # so we need to clear it when we're done.
    _temporary_implementations: dict[Operation[..., T], Callable[..., V]] = dict()
    implementations: dict[Operation[..., T], Callable[..., V]] = dict()

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.implementations = ObjectInterpretation._temporary_implementations.copy()

        for sup in cls.mro():
            if issubclass(sup, ObjectInterpretation):
                cls.implementations = {**sup.implementations, **cls.implementations}

        ObjectInterpretation._temporary_implementations.clear()

    def __iter__(self):
        return iter(self.implementations)

    def __len__(self):
        return len(self.implementations)

    def __getitem__(self, item: Operation[..., T]) -> Callable[..., V]:
        return self.implementations[item].__get__(self, type(self))


class _ImplementedOperation(Generic[P, Q, T, V]):
    impl: Optional[Callable[Q, V]]
    op: Operation[P, T]

    def __init__(self, op: Operation[P, T]):
        self.op = op
        self.impl = None

    def __get__(
        self, instance: ObjectInterpretation[T, V], owner: type
    ) -> Callable[..., V]:
        assert self.impl is not None

        return self.impl.__get__(instance, owner)

    def __call__(self, impl: Callable[Q, V]):
        self.impl = impl
        return self

    def __set_name__(self, owner: ObjectInterpretation[T, V], name):
        assert self.impl is not None
        assert self.op is not None
        owner._temporary_implementations[self.op] = self.impl


def implements(op: Operation[P, V]):
    """Marks a method in an :class:`ObjectInterpretation` as the implementation of a
    particular abstract :class:`Operation`.

    When passed an :class:`Operation`, returns a method decorator which installs
    the given method as the implementation of the given :class:`Operation`.

    """
    return _ImplementedOperation(op)
