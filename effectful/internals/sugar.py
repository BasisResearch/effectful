import collections
import collections.abc
import dataclasses
import functools
import inspect
import numbers
import operator
import typing
from typing import Callable, Generic, Mapping, Optional, Type, TypeVar

from typing_extensions import ParamSpec

from effectful.internals.runtime import interpreter, weak_memoize
from effectful.ops.core import (
    Box,
    Expr,
    Interpretation,
    Neutral,
    Operation,
    Term,
    apply,
    embed,
    evaluate,
    unembed,
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

    >>> from effectful.ops.core import define
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


def gensym(t: Type[T]) -> Operation[[], T]:
    @Operation
    def op() -> t:  # type: ignore
        raise NoDefaultRule

    return typing.cast(Operation[[], T], op)


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
        bound_sig = sig.bind(
            *(unembed(a) for a in args), **{k: unembed(v) for k, v in kwargs.items()}
        )
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
                arg = bound_sig.arguments[name]
                if sig.parameters[name].kind is inspect.Parameter.VAR_POSITIONAL:
                    bound_sig.arguments[name] = tuple(
                        rename(renaming_map, a) for a in arg
                    )
                elif sig.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
                    bound_sig.arguments[name] = {
                        k: rename(renaming_map, v) for k, v in arg.items()
                    }
                else:
                    bound_sig.arguments[name] = rename(renaming_map, arg)

        return Term(op, tuple(bound_sig.args), tuple(bound_sig.kwargs.items()))

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
                    assert not isinstance(arg, Neutral)  # DEBUG
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
def infer_default_rule(op: Operation[P, T]) -> Callable[P, Box[T]]:

    @functools.wraps(op.signature)
    def _rule(*args: P.args, **kwargs: P.kwargs) -> Box[T]:
        try:
            return op.signature(*args, **kwargs)
        except NoDefaultRule:
            return embed(
                op.__free_rule__(
                    *tuple(unembed(a) for a in args),  # type: ignore
                    **{k: unembed(v) for k, v in kwargs.items()},  # type: ignore
                )
            )

    return _rule


_embed_registry = functools.singledispatch(lambda v: v)
embed.dispatch = _embed_registry.dispatch  # type: ignore
embed.register = lambda tp: lambda fn: _embed_registry.register(tp)(fn)  # type: ignore


_unembed_registry = functools.singledispatch(lambda v: v)
unembed.dispatch = _unembed_registry.dispatch  # type: ignore
unembed.register = lambda tp: lambda fn: _unembed_registry.register(tp)(fn)  # type: ignore


OPERATORS = {}


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
    def _fail(__x: T, __y: T) -> T:
        raise NoDefaultRule


for _arithmethic_unop in (
    operator.neg,
    operator.pos,
    operator.abs,
    operator.invert,
):

    @register_syntax_op(_arithmethic_unop)
    def _fail(__x: T) -> T:
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
    def _fail(*args, **kwargs):
        raise NoDefaultRule


@register_syntax_op(operator.eq)
def _eq_op(a: T, b: T) -> bool:
    return embed(operator.eq(unembed(a), unembed(b)))  # type: ignore


@register_syntax_op(operator.ne)
def _ne_op(a: T, b: T) -> bool:
    return OPERATORS[operator.not_](OPERATORS[operator.eq](a, b))  # type: ignore


@embed.register(object)  # type: ignore
class _BaseNeutral(Generic[T], Neutral[T]):
    __stuck_term__: Term[T]

    def __init__(self, term: Term[T]):
        assert isinstance(term, Term)
        self.__stuck_term__ = term

    #######################################################################
    # equality
    #######################################################################
    def __hash__(self):
        return self.__stuck_term__.__hash__()

    def __eq__(self, other) -> bool:
        return OPERATORS[operator.eq](self, embed(other))  # type: ignore


class _BinopResolutionMixin(Generic[T]):
    #######################################################################
    # binary operator method resolution
    #######################################################################
    # arithmetic
    def __radd__(self, other: Box[T]) -> Box[T]:
        emb_other = embed(other)
        if emb_other is not other:
            return operator.add(emb_other, self)
        else:
            return OPERATORS[operator.add](emb_other, self)

    def __rsub__(self, other: Box[T]) -> Box[T]:
        emb_other = embed(other)
        if emb_other is not other:
            return operator.sub(emb_other, self)
        else:
            return OPERATORS[operator.sub](emb_other, self)

    def __rmul__(self, other: Box[T]) -> Box[T]:
        emb_other = embed(other)
        if emb_other is not other:
            return operator.mul(emb_other, self)
        else:
            return OPERATORS[operator.mul](emb_other, self)

    def __rtruediv__(self, other: Box[T]) -> Box[T]:
        emb_other = embed(other)
        if emb_other is not other:
            return operator.truediv(emb_other, self)
        else:
            return OPERATORS[operator.truediv](emb_other, self)

    def __rfloordiv__(self, other: Box[T]) -> Box[T]:
        emb_other = embed(other)
        if emb_other is not other:
            return operator.floordiv(emb_other, self)
        else:
            return OPERATORS[operator.floordiv](emb_other, self)

    def __rmod__(self, other: Box[T]) -> Box[T]:
        emb_other = embed(other)
        if emb_other is not other:
            return operator.mod(emb_other, self)
        else:
            return OPERATORS[operator.mod](emb_other, self)

    def __rpow__(self, other: Box[T]) -> Box[T]:
        emb_other = embed(other)
        if emb_other is not other:
            return operator.pow(emb_other, self)
        else:
            return OPERATORS[operator.pow](emb_other, self)

    def __rmatmul__(self, other: Box[T]) -> Box[T]:
        emb_other = embed(other)
        if emb_other is not other:
            return operator.matmul(emb_other, self)
        else:
            return OPERATORS[operator.matmul](emb_other, self)

    # bitwise
    def __rand__(self, other: Box[T]) -> Box[T]:
        emb_other = embed(other)
        if emb_other is not other:
            return operator.and_(emb_other, self)
        else:
            return OPERATORS[operator.and_](emb_other, self)

    def __ror__(self, other: Box[T]) -> Box[T]:
        emb_other = embed(other)
        if emb_other is not other:
            return operator.or_(emb_other, self)
        else:
            return OPERATORS[operator.or_](emb_other, self)

    def __rxor__(self, other: Box[T]) -> Box[T]:
        emb_other = embed(other)
        if emb_other is not other:
            return operator.xor(emb_other, self)
        else:
            return OPERATORS[operator.xor](emb_other, self)

    def __rrshift__(self, other: Box[T]) -> Box[T]:
        emb_other = embed(other)
        if emb_other is not other:
            return operator.rshift(emb_other, self)
        else:
            return OPERATORS[operator.rshift](emb_other, self)

    def __rlshift__(self, other: Box[T]) -> Box[T]:
        emb_other = embed(other)
        if emb_other is not other:
            return operator.lshift(emb_other, self)
        else:
            return OPERATORS[operator.lshift](emb_other, self)


_T_Number = TypeVar("_T_Number", bound=numbers.Number)


@embed.register(numbers.Number)  # type: ignore
class _NumberNeutral(
    Generic[_T_Number], _BaseNeutral[_T_Number], _BinopResolutionMixin[_T_Number]
):

    #######################################################################
    # arithmetic binary operators
    #######################################################################
    def __add__(self, other: Box[_T_Number]) -> Box[_T_Number]:
        return OPERATORS[operator.add](self, embed(other))  # type: ignore

    def __sub__(self, other: Box[_T_Number]) -> Box[_T_Number]:
        return OPERATORS[operator.sub](self, embed(other))  # type: ignore

    def __mul__(self, other: Box[_T_Number]) -> Box[_T_Number]:
        return OPERATORS[operator.mul](self, embed(other))  # type: ignore

    def __truediv__(self, other: Box[_T_Number]) -> Box[_T_Number]:
        return OPERATORS[operator.truediv](self, embed(other))  # type: ignore

    def __floordiv__(self, other: Box[_T_Number]) -> Box[_T_Number]:
        return OPERATORS[operator.floordiv](self, embed(other))  # type: ignore

    def __mod__(self, other: Box[_T_Number]) -> Box[_T_Number]:
        return OPERATORS[operator.mod](self, embed(other))  # type: ignore

    def __pow__(self, other: Box[_T_Number]) -> Box[_T_Number]:
        return OPERATORS[operator.pow](self, embed(other))  # type: ignore

    def __matmul__(self, other: Box[_T_Number]) -> Box[_T_Number]:
        return OPERATORS[operator.matmul](self, embed(other))  # type: ignore

    #######################################################################
    # unary operators
    #######################################################################
    def __neg__(self) -> Box[_T_Number]:
        return OPERATORS[operator.neg](self)  # type: ignore

    def __pos__(self) -> Box[_T_Number]:
        return OPERATORS[operator.pos](self)  # type: ignore

    def __abs__(self) -> Box[_T_Number]:
        return OPERATORS[operator.abs](self)  # type: ignore

    def __invert__(self) -> Box[_T_Number]:
        return OPERATORS[operator.invert](self)  # type: ignore

    #######################################################################
    # comparisons
    #######################################################################
    def __ne__(self, other) -> bool:
        return OPERATORS[operator.ne](self, embed(other))  # type: ignore

    def __lt__(self, other: Box[_T_Number]) -> Box[bool]:
        return OPERATORS[operator.lt](self, embed(other))  # type: ignore

    def __le__(self, other: Box[_T_Number]) -> Box[bool]:
        return OPERATORS[operator.le](self, embed(other))  # type: ignore

    def __gt__(self, other: Box[_T_Number]) -> Box[bool]:
        return OPERATORS[operator.gt](self, embed(other))  # type: ignore

    def __ge__(self, other: Box[_T_Number]) -> Box[bool]:
        return OPERATORS[operator.ge](self, embed(other))  # type: ignore

    #######################################################################
    # bitwise operators
    #######################################################################
    def __and__(self, other: Box[_T_Number]) -> Box[_T_Number]:
        return OPERATORS[operator.and_](self, embed(other))  # type: ignore

    def __or__(self, other: Box[_T_Number]) -> Box[_T_Number]:
        return OPERATORS[operator.or_](self, embed(other))  # type: ignore

    def __xor__(self, other: Box[_T_Number]) -> Box[_T_Number]:
        return OPERATORS[operator.xor](self, embed(other))  # type: ignore

    def __rshift__(self, other: Box[_T_Number]) -> Box[_T_Number]:
        return OPERATORS[operator.rshift](self, embed(other))  # type: ignore

    def __lshift__(self, other: Box[_T_Number]) -> Box[_T_Number]:
        return OPERATORS[operator.lshift](self, embed(other))  # type: ignore
