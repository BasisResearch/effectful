import functools
import inspect
import operator
import typing
from typing import Callable, Generic, Mapping, Optional, ParamSpec, Sequence, Type, TypeVar

import wrapt

from effectful.internals.runtime import get_runtime, interpreter, weak_memoize
from effectful.ops.core import (
    Interpretation,
    Operation,
    Term,
    TypeInContext,
    apply,
    evaluate,
    gensym,
)
from effectful.ops.handler import fwd, handler

P = ParamSpec("P")
V = TypeVar("V")
T = TypeVar("T")


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


class Annotation(Generic[T]):

    @classmethod
    def extract(cls, bound_sig: inspect.BoundArguments) -> Mapping[str, T]:
        annotations = {}
        for name, param in bound_sig.signature.parameters.items():
            if typing.get_origin(param.annotation) is typing.Annotated:
                for anno in param.annotation.__metadata__:
                    if isinstance(anno, cls):
                        annotations[name] = bound_sig.arguments[name]
        return annotations

    @classmethod
    def check(
        cls, bound_sig: inspect.BoundArguments, anno: T, tp: TypeInContext | Operation
    ) -> bool:
        raise NotImplementedError

    @classmethod
    def check_all(cls, bound_sig: inspect.BoundArguments) -> bool:
        return all(
            cls.check(bound_sig, anno, bound_sig.arguments[name])
            for name, anno in cls.extract(bound_sig).items()
        )


class Bound(Annotation[Operation | Sequence[Operation]]):

    @classmethod
    def check(
        cls, bound_sig: inspect.BoundArguments, anno: Operation, tp: Operation
    ) -> bool:
        return anno not in set(Fresh.extract(bound_sig).values())


class Fresh(Annotation[Operation | Sequence[Operation]]):

    @classmethod
    def check(
        cls, bound_sig: inspect.BoundArguments, anno: Operation, tp: Operation
    ) -> bool:
        return anno not in set(Bound.extract(bound_sig).values())


class Has(Annotation[Mapping[Operation, bool]]):
    vars: Mapping[str, bool]

    def __init__(self, **vars: bool):
        self.vars = vars

    @classmethod
    def extract(
        cls, bound_sig: inspect.BoundArguments
    ) -> Mapping[str, Mapping[Operation, bool]]:
        annotations = {}
        for name, param in bound_sig.signature.parameters.items():
            if typing.get_origin(param.annotation) is typing.Annotated:
                for anno in param.annotation.__metadata__:
                    if isinstance(anno, cls):
                        annotations[name] = {
                            bound_sig.arguments[k]: v for k, v in anno.vars.items()
                        }
        return annotations

    @classmethod
    def check(
        cls,
        bound_sig: inspect.BoundArguments,
        anno: Mapping[Operation, bool],
        tp: TypeInContext,
    ) -> bool:
        return all(has_var == (var in tp.context) for var, has_var in anno.items())


@weak_memoize
def defop(fn: Callable[P, T]) -> Operation[P, T]:

    def rename(subs: Mapping[Operation[[], T], Operation[[], T]]):
        free = {apply: lambda op, *a, **k: Term(op, a, tuple(k.items()))}
        return interpreter({**free, **subs})(evaluate)

    sig = inspect.signature(fn)

    def __judgement__(
        sig: inspect.Signature,
        *arg_types: TypeInContext | Operation,
        **kwarg_types: TypeInContext | Operation
    ) -> TypeInContext:
        bound_sig = sig.bind(*arg_types, **kwarg_types)

        bound_vars = Bound.extract(bound_sig)
        fresh_vars = Fresh.extract(bound_sig)

        assert Bound.check_all(bound_sig)
        assert Fresh.check_all(bound_sig)
        assert Has.check_all(bound_sig)

        ctx = {}
        for name, arg_type in bound_sig.arguments.items():
            if isinstance(arg_type, TypeInContext):
                ctx.update(
                    {v: t for v, t in arg_type.context.items() if v not in bound_vars}
                )
            elif isinstance(arg_type, Operation):
                if arg_type in fresh_vars:
                    ctx[arg_type] = typing.get_args(sig.parameters[name].annotation)[0]
            else:
                continue

        return TypeInContext(context=ctx, type=sig.return_annotation)

    def __binding__(sig: inspect.Signature, *args: P.args, **kwargs: P.kwargs) -> T:

        bound_sig = sig.bind(*args, **kwargs)

        bound_vars = Bound.extract(bound_sig)

        renaming_map = {
            var: gensym(get_runtime()._JUDGEMENTS[var]()) for var in bound_vars.values()
        }
        for argname, argval in tuple(bound_sig.arguments.items()):
            if argname in bound_vars:
                bound_sig.arguments[argname] = renaming_map[bound_vars[argname]]
            else:
                bound_sig.arguments[argname] = rename(renaming_map)(argval)

        return fwd(None, *bound_sig.args, **bound_sig.kwargs)

    op = Operation(fn)
    get_runtime()._JUDGEMENTS[op] = functools.partial(__judgement__, sig)
    get_runtime()._BINDINGS[op] = functools.partial(__binding__, sig)
    return op


class Box(Generic[T], wrapt.ObjectProxy):
    __wrapped__: Term[T] | T

    def __add__(self, other: T | Term[T] | "Box[T]") -> "Box[T]":
        return type(self)(defop(operator.__add__)(
            self if not isinstance(self, Box) else self.__wrapped__,
            other if not isinstance(other, Box) else other.__wrapped__,
        ))

    def __radd__(self, other: T | Term[T] | "Box[T]") -> "Box[T]":
        return type(self)(defop(operator.__add__)(
            other if not isinstance(other, Box) else other.__wrapped__,
            self if not isinstance(self, Box) else self.__wrapped__,
        ))
