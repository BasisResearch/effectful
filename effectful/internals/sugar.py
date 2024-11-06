import collections
import collections.abc
import dataclasses
import functools
import inspect
import numbers
import operator
import typing
from typing import (
    Any,
    Callable,
    Generic,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

import tree
from typing_extensions import ParamSpec

from effectful.internals.runtime import interpreter, weak_memoize
from effectful.ops.core import (
    Expr,
    Interpretation,
    Operation,
    Term,
    apply,
    evaluate,
    syntactic_eq,
    typeof,
)

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


class ObjectInterpretation(Generic[T, V], Interpretation[T, V]):
    """
    A helper superclass for defining an :type:`Interpretation`s of many :type:`Operation` instances with shared
    state or behavior.

    You can mark specific methods in the definition of an :class:`ObjectInterpretation` with operations
    using the :func:`implements` decorator. The :class:`ObjectInterpretation` object itself is an :type:`Interpretation`
    (mapping from :type:`Operation` to :type:`Callable`)

    >>> from effectful.ops.handler import handler
    >>> @Operation
    ... def read_box():
    ...     pass
    ...
    >>> @Operation
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


P1 = ParamSpec("P1")
P2 = ParamSpec("P2")


class _ImplementedOperation(Generic[P1, P2, T, V]):
    impl: Optional[Callable[P2, V]]
    op: Operation[P1, T]

    def __init__(self, op: Operation[P1, T]):
        self.op = op
        self.impl = None

    def __get__(
        self, instance: ObjectInterpretation[T, V], owner: type
    ) -> Callable[..., V]:
        assert self.impl is not None

        return self.impl.__get__(instance, owner)

    def __call__(self, impl: Callable[P2, V]):
        self.impl = impl
        return self

    def __set_name__(self, owner: ObjectInterpretation[T, V], name):
        assert self.impl is not None
        assert self.op is not None
        owner._temporary_implementations[self.op] = self.impl


def implements(op: Operation[P, V]):
    """
    Marks a method in an `ObjectInterpretation` as the implementation of a
    particular abstract `Operation`.

    When passed an `Operation`, returns a method decorator which installs the given
    method as the implementation of the given `Operation`.
    """
    return _ImplementedOperation(op)


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

        @Operation
        def op() -> t:  # type: ignore
            raise NoDefaultRule

    else:

        def op(*args, **kwargs):
            raise NoDefaultRule

        op.__signature__ = inspect.signature(t)
        op = Operation(op)

    op.__name__ = name or t.__name__

    if is_type:
        return typing.cast(Operation[[], T], op)
    else:
        return typing.cast(Operation[P, T], op)


class Annotation:
    pass


@dataclasses.dataclass
class Bound(Annotation):
    scope: int = 0


@dataclasses.dataclass
class Scoped(Annotation):
    scope: int = 0


@weak_memoize
def infer_free_rule(op: Operation[P, T]) -> Callable[P, Term[T]]:
    sig = inspect.signature(op.signature)

    def rename(
        subs: Mapping[Operation[..., S], Operation[..., S]],
        leaf_value: V,  # Union[Term[V], Operation[..., V], V],
    ) -> V:  # Union[Term[V], Operation[..., V], V]:
        if isinstance(leaf_value, Operation):
            return subs.get(leaf_value, leaf_value)  # type: ignore
        elif isinstance(leaf_value, Term):
            with interpreter({apply: lambda _, op, *a, **k: op.__free_rule__(*a, **k), **subs}):  # type: ignore
                return evaluate(leaf_value)  # type: ignore
        else:
            return leaf_value

    @functools.wraps(op.signature)
    def _rule(*args: P.args, **kwargs: P.kwargs) -> Term[T]:
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
            renaming_map = {
                var: gensym(var.__type_rule__()) for var in bound_vars[scope]
            }  # TODO support finitary operations
            # get just the arguments that are in the scope
            for name in scoped_args[scope]:
                bound_sig.arguments[name] = tree.map_structure(
                    lambda a: rename(renaming_map, a),
                    bound_sig.arguments[name],
                )

        tm = _embed_registry.dispatch(object)(
            op, tuple(bound_sig.args), tuple(bound_sig.kwargs.items())
        )
        return embed(tm)  # type: ignore

    return _rule


@weak_memoize
def infer_scope_rule(op: Operation[P, T]) -> Callable[P, Interpretation[V, Type[V]]]:
    sig = inspect.signature(op.signature)

    @functools.wraps(op.signature)
    def _rule(*args: P.args, **kwargs: P.kwargs) -> Interpretation[V, Type[V]]:
        bound_sig = sig.bind(*args, **kwargs)
        bound_sig.apply_defaults()

        bound_vars: dict[Operation[..., V], Callable[..., Type[V]]] = {}
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

    return _rule


@weak_memoize
def infer_type_rule(op: Operation[P, T]) -> Callable[P, Type[T]]:
    sig = inspect.signature(op.signature)

    @functools.wraps(op.signature)
    def _rule(*args: P.args, **kwargs: P.kwargs) -> Type[T]:
        bound_sig = sig.bind(*args, **kwargs)
        bound_sig.apply_defaults()

        anno = sig.return_annotation
        if anno is inspect.Signature.empty:
            return typing.cast(Type[T], object)
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
                    tp: Type[T] = type(arg) if not isinstance(arg, type) else arg
                    return tp
            return typing.cast(Type[T], object)
        elif typing.get_origin(anno) is typing.Annotated:
            tp = typing.get_args(anno)[0]
            if not typing.TYPE_CHECKING:
                tp = tp if typing.get_origin(tp) is None else typing.get_origin(tp)
            return tp
        elif typing.get_origin(anno) is not None:
            return typing.get_origin(anno)
        else:
            return anno

    return _rule


class NoDefaultRule(Exception):
    pass


@weak_memoize
def infer_default_rule(op: Operation[P, T]) -> Callable[P, Expr[T]]:

    @functools.wraps(op.signature)
    def _rule(*args: P.args, **kwargs: P.kwargs) -> Expr[T]:
        try:
            return op.signature(*args, **kwargs)
        except NoDefaultRule:
            return op.__free_rule__(*args, **kwargs)

    return _rule


def embed(expr: Expr[T]) -> Expr[T]:
    if isinstance(expr, Term):
        impl: Callable[[Operation[..., T], Sequence, Sequence[tuple]], Term[T]]
        impl = _embed_registry.dispatch(typeof(expr))
        return impl(expr.op, expr.args, expr.kwargs)
    else:
        return expr


_embed_registry = functools.singledispatch(lambda v: v)
embed_register = _embed_registry.register


_as_term_registry = functools.singledispatch(lambda v: v)
as_term_register = _as_term_registry.register


OPERATORS: dict[Callable[..., Any], Operation[..., Any]] = {}


def register_syntax_op(
    syntax_fn: Callable[P, T], syntax_op_fn: Optional[Callable[P, T]] = None
):
    if syntax_op_fn is None:
        return functools.partial(register_syntax_op, syntax_fn)

    OPERATORS[syntax_fn] = Operation(syntax_op_fn)
    return OPERATORS[syntax_fn]


for _arithmetic_binop in (
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    operator.pow,
    operator.matmul,
    operator.lshift,
    operator.rshift,
    operator.and_,
    operator.or_,
    operator.xor,
):

    @register_syntax_op(_arithmetic_binop)
    def _(__x: T, __y: T) -> T:
        raise NoDefaultRule


for _arithmethic_unop in (
    operator.neg,
    operator.pos,
    operator.abs,
    operator.invert,
):

    @register_syntax_op(_arithmethic_unop)
    def _(__x: T) -> T:
        raise NoDefaultRule


for _other_operator_op in (
    operator.not_,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
    operator.is_,
    operator.is_not,
    operator.contains,
    operator.index,
    operator.getitem,
    operator.setitem,
    operator.delitem,
    # TODO handle these builtin functions
    # getattr,
    # setattr,
    # delattr,
    # len,
    # iter,
    # next,
    # reversed,
):

    @register_syntax_op(_other_operator_op)
    @functools.wraps(_other_operator_op)
    def _(*args, **kwargs):
        raise NoDefaultRule


@register_syntax_op(operator.eq)
def _eq_op(a: Expr[T], b: Expr[T]) -> Expr[bool]:
    """Default implementation of equality for terms. As a special case, equality defaults to syntactic equality rather
    than producing a free term.

    """
    return syntactic_eq(a, b)


@register_syntax_op(operator.ne)
def _ne_op(a: T, b: T) -> bool:
    return OPERATORS[operator.not_](OPERATORS[operator.eq](a, b))  # type: ignore


@embed_register(object)
class BaseTerm(Generic[T], Term[T]):
    op: Operation[..., T]
    args: Sequence[Expr]
    kwargs: Sequence[Tuple[str, Expr]]

    def __init__(
        self,
        op: Operation[..., T],
        args: Sequence[Expr],
        kwargs: Sequence[Tuple[str, Expr]],
    ):
        self.op = op
        self.args = args
        self.kwargs = kwargs

    def __str__(self: "Term[T]") -> str:
        params_str = ""
        if len(self.args) > 0:
            params_str += ", ".join(str(x) for x in self.args)
        if len(self.kwargs) > 0:
            params_str += ", " + ", ".join(f"{k}={str(v)}" for (k, v) in self.kwargs)
        return f"{str(self.op)}({params_str})"

    def __eq__(self, other) -> bool:
        return OPERATORS[operator.eq](self, other)  # type: ignore


@as_term_register(object)
def _unembed_literal(value: T) -> T:
    return value


@embed_register(Operation)
def _embed_literal_op(expr: Operation[P, T]) -> Operation[P, T]:
    return expr


@as_term_register(Operation)
def _unembed_literal_op(value: Operation[P, T]) -> Operation[P, T]:
    return value


_T_Number = TypeVar("_T_Number", bound=numbers.Number)


@embed_register(numbers.Number)
class NumberTerm(Generic[_T_Number], BaseTerm[_T_Number]):

    #######################################################################
    # arithmetic binary operators
    #######################################################################
    def __add__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.add](self, other)  # type: ignore

    def __sub__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.sub](self, other)  # type: ignore

    def __mul__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.mul](self, other)  # type: ignore

    def __truediv__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.truediv](self, other)  # type: ignore

    def __floordiv__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.floordiv](self, other)  # type: ignore

    def __mod__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.mod](self, other)  # type: ignore

    def __pow__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.pow](self, other)  # type: ignore

    def __matmul__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.matmul](self, other)  # type: ignore

    #######################################################################
    # unary operators
    #######################################################################
    def __neg__(self) -> Expr[_T_Number]:
        return OPERATORS[operator.neg](self)  # type: ignore

    def __pos__(self) -> Expr[_T_Number]:
        return OPERATORS[operator.pos](self)  # type: ignore

    def __abs__(self) -> Expr[_T_Number]:
        return OPERATORS[operator.abs](self)  # type: ignore

    def __invert__(self) -> Expr[_T_Number]:
        return OPERATORS[operator.invert](self)  # type: ignore

    #######################################################################
    # comparisons
    #######################################################################
    def __ne__(self, other) -> bool:
        return OPERATORS[operator.ne](self, other)  # type: ignore

    def __lt__(self, other: Expr[_T_Number]) -> Expr[bool]:
        return OPERATORS[operator.lt](self, other)  # type: ignore

    def __le__(self, other: Expr[_T_Number]) -> Expr[bool]:
        return OPERATORS[operator.le](self, other)  # type: ignore

    def __gt__(self, other: Expr[_T_Number]) -> Expr[bool]:
        return OPERATORS[operator.gt](self, other)  # type: ignore

    def __ge__(self, other: Expr[_T_Number]) -> Expr[bool]:
        return OPERATORS[operator.ge](self, other)  # type: ignore

    #######################################################################
    # bitwise operators
    #######################################################################
    def __and__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.and_](self, other)  # type: ignore

    def __or__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.or_](self, other)  # type: ignore

    def __xor__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.xor](self, other)  # type: ignore

    def __rshift__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.rshift](self, other)  # type: ignore

    def __lshift__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.lshift](self, other)  # type: ignore

    #######################################################################
    # reflected operators
    #######################################################################
    def __radd__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.add](other, self)  # type: ignore

    def __rsub__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.sub](other, self)  # type: ignore

    def __rmul__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.mul](other, self)  # type: ignore

    def __rtruediv__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.truediv](other, self)  # type: ignore

    def __rfloordiv__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.floordiv](other, self)  # type: ignore

    def __rmod__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.mod](other, self)  # type: ignore

    def __rpow__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.pow](other, self)  # type: ignore

    def __rmatmul__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.matmul](other, self)  # type: ignore

    # bitwise
    def __rand__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.and_](other, self)  # type: ignore

    def __ror__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.or_](other, self)  # type: ignore

    def __rxor__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.xor](other, self)  # type: ignore

    def __rrshift__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.rshift](other, self)  # type: ignore

    def __rlshift__(self, other: Expr[_T_Number]) -> Expr[_T_Number]:
        return OPERATORS[operator.lshift](other, self)  # type: ignore


@embed_register(collections.abc.Callable)  # type: ignore
class CallableTerm(Generic[P, T], BaseTerm[collections.abc.Callable[P, T]]):
    def __call__(self, *args: Expr, **kwargs: Expr) -> Expr[T]:
        from effectful.ops.function import funcall

        return funcall(self, *args, **kwargs)  # type: ignore


@as_term_register(collections.abc.Callable)  # type: ignore
def _unembed_callable(value: Callable[P, T]) -> Expr[Callable[P, T]]:
    from effectful.ops.function import defun, funcall

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
        **{name: gensym(param.annotation) for name, param in sig.parameters.items()}
    )
    bound_sig.apply_defaults()

    with interpreter(
        {
            apply: lambda _, op, *a, **k: op.__free_rule__(*a, **k),
            funcall: funcall.__default_rule__,
        }
    ):
        body = value(
            *[a() for a in bound_sig.args],
            **{k: v() for k, v in bound_sig.kwargs.items()},
        )

    return defun(body, *bound_sig.args, **bound_sig.kwargs)
