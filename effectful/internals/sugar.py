import collections
import collections.abc
import dataclasses
import functools
import inspect
import numbers
import operator
import typing
from types import EllipsisType
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
    Union,
)

import torch
import tree
from typing_extensions import ParamSpec

from effectful.internals.runtime import interpreter, weak_memoize
from effectful.ops.core import (
    Expr,
    Interpretation,
    Operation,
    Term,
    apply,
    ctxof,
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

    elif isinstance(t, collections.abc.Callable):

        def dummy(*args, **kwargs):
            raise NoDefaultRule

        functools.update_wrapper(dummy, t)
        op = Operation(dummy)

    else:
        raise ValueError(f"expected type or callable, got {t}")

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
            renaming_map = {var: gensym(var) for var in bound_vars[scope]}
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


def register_syntax_op(syntax_fn: Callable[P, T]):
    def register_syntax_op_fn(syntax_op_fn: Callable[P, T]):
        OPERATORS[syntax_fn] = Operation(syntax_op_fn)
        return OPERATORS[syntax_fn]

    return register_syntax_op_fn


for arithmetic_binop in (
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

    @register_syntax_op(arithmetic_binop)
    def _(x: T, y: T) -> T:
        if not isinstance(x, Term) and not isinstance(y, Term):
            return arithmetic_binop(x, y)
        raise NoDefaultRule


for arithmetic_unop in (
    operator.neg,
    operator.pos,
    operator.abs,
    operator.invert,
):

    @register_syntax_op(arithmetic_unop)
    def _(x: T) -> T:
        if not isinstance(x, Term):
            return arithmetic_unop(x)  # typing: ignore
        raise NoDefaultRule


for other_operator_op in (
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

    @register_syntax_op(other_operator_op)
    @functools.wraps(other_operator_op)
    def _(*args, **kwargs):
        if not any(isinstance(a, Term) for a in args) and not any(
            isinstance(a, Term) for a in kwargs.values()
        ):
            return other_operator_op(*args, **kwargs)

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


def term_to_str(term: Term[T]) -> str:
    params_str = ""
    if len(term.args) > 0:
        params_str += ", ".join(str(x) for x in term.args)
    if len(term.kwargs) > 0:
        params_str += ", " + ", ".join(f"{k}={str(v)}" for (k, v) in term.kwargs)
    return f"{str(term.op)}({params_str})"


@embed_register(object)
class BaseTerm(Generic[T], Term[T]):
    op: Operation[..., T]
    args: Sequence[Expr[Any]]
    kwargs: Sequence[Tuple[str, Expr[Any]]]

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
        return term_to_str(self)

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


IndexElement = Union[None, int, slice, Sequence[int], EllipsisType, torch.Tensor]


def _desugar_tensor_index(shape, key):
    new_shape = []
    new_key = []

    def extra_dims(key):
        return sum(1 for k in key if k is None)

    # handle any missing dimensions by adding a trailing Ellipsis
    if not any(k is Ellipsis for k in key):
        key = tuple(key) + (...,)

    for i, k in enumerate(key):
        if k is None:  # add a new singleton dimension
            new_shape.append(1)
            new_key.append(slice(None))
        elif k is Ellipsis:
            assert not any(
                k is Ellipsis for k in key[i + 1 :]
            ), "only one Ellipsis allowed"

            # determine which of the original dimensions this ellipsis refers to
            pre_dims = i - extra_dims(key[:i])  # dimensions that precede the ellipsis
            elided_dims = (
                len(shape) - pre_dims - (len(key) - i - 1 - extra_dims(key[i + 1 :]))
            )  #
            new_shape += shape[pre_dims : pre_dims + elided_dims]
            new_key += [slice(None)] * elided_dims
        else:
            new_shape.append(shape[len(new_shape) - extra_dims(key[:i])])
            new_key.append(k)

    return new_shape, new_key


def _getitem_ellipsis_and_none(
    x: torch.Tensor, key: Tuple[IndexElement, ...]
) -> Tuple[torch.Tensor, Tuple[IndexElement, ...]]:
    """Eliminate ellipses and None in an index expression x[key].

    Returns x1, key1 such that x1[key1] == x[key] nand key1 does not contain None or Ellipsis.

    """

    new_shape, new_key = _desugar_tensor_index(x.shape, key)
    return torch.reshape(x, new_shape), new_key


def sizesof(value: Expr) -> Mapping[Operation[[], int], int]:
    sizes: dict[Operation[[], int], int] = {}

    def _torch_getitem_sizeof(
        x: Expr[torch.Tensor], key: Tuple[Expr[IndexElement], ...]
    ) -> Expr[torch.Tensor]:
        if isinstance(x, torch.Tensor):
            shape, key_ = _desugar_tensor_index(x.shape, key)

            for i, k in enumerate(key_):
                if (
                    isinstance(k, Term)
                    and len(k.args) == 0
                    and len(k.kwargs) == 0
                    and issubclass(typeof(k), int)
                ):
                    if k.op in sizes and sizes[k.op] != shape[i]:
                        raise ValueError(
                            f"Named index {k.op} used in incompatible dimensions of size {sizes[k.op]} and {shape[i]}"
                        )
                    sizes[k.op] = shape[i]

        return torch_getitem.__free_rule__(x, key)

    with interpreter(
        {
            torch_getitem: _torch_getitem_sizeof,
            apply: lambda _, op, *a, **k: op.__free_rule__(*a, **k),
        }
    ):
        evaluate(value)

    return sizes


def partial_eval(t: T, order=None) -> T:
    """Partially evaluate a term with respect to its sized free variables.

    Variables in `order` are converted to positional dimensions in the result
    tensor, in the order they appear. All other variables remain free.

    """
    from effectful.ops.function import defun

    if order is None:
        order = []

    sized_fvs = sizesof(t)

    if any(x for x in order if x not in sized_fvs):
        raise ValueError("sized must be a subset of the term's sized free variables")

    # if there are no sized free variables, then nothing to do
    if len(sized_fvs) == 0:
        return t

    order_set = set(order)
    reindex_fvs = [
        (var, size) for var, size in sized_fvs.items() if var not in order_set
    ]
    ordered_sized_fvs = reindex_fvs + [(var, sized_fvs[var]) for var in order]

    tpe_torch_fn = torch.func.vmap(
        defun(t, *[var for (var, _) in ordered_sized_fvs]), randomness="different"
    )

    inds = torch.broadcast_tensors(
        *(
            torch.arange(size)[(...,) + (None,) * (len(ordered_sized_fvs) - i - 1)]
            for i, (_, size) in enumerate(ordered_sized_fvs)
        )
    )

    flat_result = tpe_torch_fn(*[i.reshape(-1) for i in inds])

    def reindex_flat_tensor(t):
        if not isinstance(t, torch.Tensor):
            return t

        result = t.reshape(inds[0].shape + t.shape[1:])
        return torch_getitem(result, tuple(var() for (var, _) in reindex_fvs))

    return tree.map_structure(reindex_flat_tensor, flat_result)


@functools.cache
def _register_torch_op(torch_fn: Callable[P, T]):

    @Operation
    def _torch_op(*args, **kwargs) -> torch.Tensor:

        tm = _torch_op.__free_rule__(*args, **kwargs)
        sized_fvs = sizesof(tm)

        if (
            _torch_op is torch_getitem
            and not isinstance(args[0], Term)
            and sized_fvs
            and args[1]
            and all(isinstance(k, Term) and k.op in sized_fvs for k in args[1])
        ):
            raise NoDefaultRule
        elif sized_fvs and set(sized_fvs.keys()) == set(ctxof(tm).keys()) - {
            torch_getitem,
            _torch_op,
        }:
            # note: this cast is a lie. partial_eval can return non-tensors, as
            # can torch_fn. for example, some torch functions return tuples,
            # which partial_eval handles.
            return typing.cast(torch.Tensor, partial_eval(tm))
        elif not any(
            tree.flatten(
                tree.map_structure(lambda x: isinstance(x, Term), (args, kwargs))
            )
        ):
            return typing.cast(torch.Tensor, torch_fn(*args, **kwargs))
        else:
            raise NoDefaultRule

    return _torch_op


@_register_torch_op
def torch_getitem(
    x: torch.Tensor,
    key: Tuple[IndexElement, ...],
) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected a tensor but got {type(x)}")

    # fast path for simple cases
    if len(key) == 0:
        return x
    elif not any(isinstance(k, torch.Tensor) for k in key):
        return x[tuple(key)]
    elif all(isinstance(k, torch.Tensor) for k in key):
        return torch.ops.aten.index(x, key)

    # handle None, Ellipsis, and missing dimensions
    x, key = _getitem_ellipsis_and_none(x, key)

    # Convert non-tensor args to tensors
    key_l = list(key)
    for i, arg in list(enumerate(key)):
        if isinstance(arg, slice):
            if arg == slice(None):
                key_l[i] = None
            else:
                # Convert slices to torch.arange()s.
                start = arg.start if arg.start is not None else 0
                stop = arg.stop if arg.stop is not None else x.shape[i]
                step = arg.step if arg.step is not None else 1
                flat_arg = torch.arange(
                    start, stop, step, dtype=torch.long, device=x.device
                )
                key_l[i] = flat_arg.reshape((-1,) + (1,) * i)
        elif isinstance(arg, int):
            key_l[i] = torch.tensor(arg, dtype=torch.long, device=x.device)
        elif isinstance(arg, (list, tuple)):
            flat_arg = torch.tensor(arg, dtype=torch.long, device=x.device)
            key_l[i] = flat_arg.reshape(flat_arg.shape + (1,) * i)

    return torch.ops.aten.index(x, tuple(key_l))


@embed_register(torch.Tensor)
def _embed_tensor(op, args, kwargs):
    match op, args, kwargs:
        case torch_getitem_, (torch.Tensor() as x, key), () if (
            torch_getitem_ is torch_getitem
            and len(key) >= 1
            and not isinstance(x, Term)
            and all(
                typeof(k) is int and not k.args and not k.kwargs
                for k in key
                if isinstance(k, Term)
            )
        ):
            return EagerTensorTerm(x, key)
        case _:
            return TensorTerm(op, args, kwargs)


class TensorTerm(BaseTerm[torch.Tensor]):
    def __getitem__(
        self, key: Union[Expr[IndexElement], Tuple[Expr[IndexElement], ...]]
    ) -> Expr[torch.Tensor]:
        return torch_getitem(self, key if isinstance(key, tuple) else (key,))

    @classmethod
    def __torch_function__(
        cls, func: Callable[..., T], types, args=(), kwargs=None
    ) -> Expr[T]:
        return _register_torch_op(func)(*args, **({} if kwargs is None else kwargs))


class EagerTensorTerm(torch.Tensor):

    op: Operation[..., torch.Tensor] = torch_getitem
    args: Tuple[torch.Tensor, Tuple[IndexElement, ...]]
    kwargs: Tuple = ()

    __match_args__ = ("op", "args", "kwargs")

    def __new__(cls, x: torch.Tensor, key: Tuple[IndexElement, ...]):
        assert not isinstance(x, Term)

        for k in key:
            if isinstance(k, Term):
                assert typeof(k) is int and not k.args and not k.kwargs

        x, key = _getitem_ellipsis_and_none(x, key)
        ret = x.as_subclass(cls)
        ret.args = (x, key)
        return ret

    def __repr__(self):
        return f"{self.__class__.__name__}({term_to_str(self)})"

    @classmethod
    def __torch_function__(
        cls, func: Callable[..., T], types, args=(), kwargs=None
    ) -> Expr[T]:
        return _register_torch_op(func)(*args, **({} if kwargs is None else kwargs))

    def __getitem__(self, key) -> torch.Tensor:
        return torch_getitem(self, key if isinstance(key, tuple) else (key,))

    def __format__(self, format_spec: str) -> str:
        return (
            format(torch.Tensor(self), format_spec)
            + "["
            + ", ".join(str(a) for a in self.args[1])
            + "]"
        )

    @property
    def shape(self) -> torch.Size:  # type: ignore
        x, key = self.args
        return torch.Size([s for s, k in zip(x.shape, key) if not isinstance(k, Term)])

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def numel(self) -> int:
        return self.shape.numel()

    def dim(self) -> int:
        return len(self.shape)

    @property
    def ndim(self) -> int:  # type: ignore
        return self.dim()

    def ndimension(self):
        return self.dim()

    def item(self):
        raise ValueError(f"cannot convert {self} to a Python scalar")

    @property
    def dtype(self):
        return self.args[0].dtype

    @property
    def device(self):
        return self.args[0].device

    def new(self, *args, **kwargs):
        return self.args[0].new(*args, **kwargs)

    @property
    def requires_grad(self):
        return self.args[0].requires_grad

    @property
    def grad_fn(self):
        return self.args[0].grad_fn
