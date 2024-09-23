import collections
import collections.abc
import dataclasses
import functools
import inspect
import typing
from typing import Callable, Generic, Mapping, Optional, Type, TypeVar, Union

from typing_extensions import ParamSpec

from effectful.internals.runtime import interpreter
from effectful.ops.core import Interpretation, Operation, Term, apply, evaluate

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
        return Term(op, (), ())

    return op


class Annotation:
    pass


@dataclasses.dataclass
class Bound(Annotation):
    scope: int = 0


@dataclasses.dataclass
class Scoped(Annotation):
    scope: int = 0


def infer_free_rule(op: Operation[P, T]) -> Callable[P, Term[T]]:
    sig = inspect.signature(op.signature)

    def rename(
        subs: Mapping[Operation[..., S], Operation[..., S]],
        leaf_value: Union[Term[V], Operation[..., V], V],
    ) -> Union[Term[V], Operation[..., V], V]:
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

        bound_vars: dict[int, set[Operation]] = collections.defaultdict(set)
        scoped_args: dict[int, set[str]] = collections.defaultdict(set)
        for param_name, param in bound_sig.signature.parameters.items():
            if typing.get_origin(param.annotation) is typing.Annotated:
                for anno in param.annotation.__metadata__:
                    if isinstance(anno, Bound):
                        bound_vars[anno.scope].add(bound_sig.arguments[param_name])
                        scoped_args[anno.scope].add(param_name)
                    elif isinstance(anno, Scoped):
                        scoped_args[anno.scope].add(param_name)
            else:
                scoped_args[0].add(param_name)

        # TODO replace this temporary check with more general scope level propagation
        if bound_vars:
            max_scope = max(bound_vars.keys(), default=0)
            assert all(s in bound_vars or s > max_scope for s in scoped_args.keys())

        # recursively rename bound variables from innermost to outermost scope
        for scope in sorted(bound_vars.keys()):
            # create fresh variables for each bound variable in the scope
            renaming_map = {
                var: gensym(var.__type_rule__()) for var in bound_vars[scope]
            }  # TODO support finitary operations
            # get just the arguments that are in the scope
            arguments_to_rename = {
                name: bound_sig.arguments[name] for name in scoped_args[scope]
            }
            # substitute the fresh names into the arguments
            renamed_arguments = {
                name: rename(renaming_map, arg)
                for name, arg in arguments_to_rename.items()
            }
            # update the arguments with the renamed values
            bound_sig.arguments.update(renamed_arguments)

        return Term(op, tuple(bound_sig.args), tuple(bound_sig.kwargs.items()))

    return _rule


def infer_scope_rule(op: Operation[P, T]) -> Callable[P, Interpretation[V, Type[V]]]:
    sig = inspect.signature(op.signature)

    @functools.wraps(op.signature)
    def _rule(*args: P.args, **kwargs: P.kwargs) -> Interpretation[V, Type[V]]:
        bound_sig = sig.bind(*args, **kwargs)

        bound_vars: dict[Operation[..., V], Callable[..., Type[V]]] = {}
        for param_name, param in bound_sig.signature.parameters.items():
            if typing.get_origin(param.annotation) is typing.Annotated:
                for anno in param.annotation.__metadata__:
                    if isinstance(anno, Bound):
                        bound_var = bound_sig.arguments[param_name]
                        bound_vars[bound_var] = bound_var.__type_rule__

        return bound_vars

    return _rule


def infer_type_rule(op: Operation[P, T]) -> Callable[P, Type[T]]:
    sig = inspect.signature(op.signature)

    @functools.wraps(op.signature)
    def _rule(*args: P.args, **kwargs: P.kwargs) -> Type[T]:
        anno = sig.return_annotation
        if anno is inspect.Signature.empty or isinstance(anno, typing.TypeVar):
            return typing.cast(Type[T], object)
        elif typing.get_origin(anno) is not None:
            return typing.get_origin(anno)
        else:
            return anno

    return _rule
