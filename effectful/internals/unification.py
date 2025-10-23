"""Type unification and inference utilities for Python's generic type system.

This module implements a unification algorithm for type inference over a subset of
Python's generic types. Unification is a fundamental operation in type systems that
finds substitutions for type variables to make two types equivalent.

The module provides four main operations:

1. **unify(typ, subtyp, subs={})**: The core unification algorithm that attempts to
   find a substitution mapping for type variables that makes a pattern type equal to
   a concrete type. It handles TypeVars, generic types (List[T], Dict[K,V]), unions,
   callables, and function signatures with inspect.Signature/BoundArguments.

2. **substitute(typ, subs)**: Applies a substitution mapping to a type expression,
   replacing all TypeVars with their mapped concrete types. This is used to
   instantiate generic types after unification.

3. **freetypevars(typ)**: Extracts all free (unbound) type variables from a type
   expression. Useful for analyzing generic types and ensuring all TypeVars are
   properly bound.

4. **nested_type(value)**: Infers the type of a runtime value, handling nested
   collections by recursively determining element types. For example, [1, 2, 3]
   becomes list[int], and {"key": [1, 2]} becomes dict[str, list[int]].

The unification algorithm uses a single-dispatch pattern to handle different type
combinations:
- TypeVar unification binds variables to concrete types
- Generic type unification matches origins and recursively unifies type arguments
- Structural unification handles sequences and mappings by element
- Union types attempt unification with any matching branch
- Function signatures unify parameter types with bound arguments

Example usage:
    >>> from effectful.internals.unification import unify, substitute, freetypevars
    >>> import typing
    >>> T = typing.TypeVar('T')
    >>> K = typing.TypeVar('K')
    >>> V = typing.TypeVar('V')

    >>> # Find substitution that makes list[T] equal to list[int]
    >>> subs = unify(list[T], list[int])
    >>> subs
    {~T: <class 'int'>}

    >>> # Apply substitution to instantiate a generic type
    >>> substitute(dict[K, list[V]], {K: str, V: int})
    dict[str, list[int]]

    >>> # Find all type variables in a type expression
    >>> freetypevars(dict[str, list[V]])
    {~V}

This module is primarily used internally by effectful for type inference in its
effect system, allowing it to track and propagate type information through
effect handlers and operations.
"""

import abc
import builtins
import collections
import collections.abc
import functools
import inspect
import numbers
import operator
import types
import typing

try:
    from typing import _collect_type_parameters as _freetypevars  # type: ignore
except ImportError:
    from typing import _collect_parameters as _freetypevars  # type: ignore

import effectful.ops.types

if typing.TYPE_CHECKING:
    TypeConstant = type | abc.ABCMeta | types.EllipsisType | None
    GenericAlias = types.GenericAlias
    UnionType = types.UnionType
else:
    TypeConstant = (
        type | abc.ABCMeta | types.EllipsisType | type(None) | type(typing.Any)
    )
    GenericAlias = types.GenericAlias | typing._GenericAlias
    UnionType = types.UnionType | typing._UnionGenericAlias

TypeVariable = typing.TypeVar | typing.TypeVarTuple | typing.ParamSpec
TypeApplication = GenericAlias | UnionType
TypeExpression = TypeVariable | TypeConstant | TypeApplication
TypeExpressions = TypeExpression | collections.abc.Sequence[TypeExpression]

Substitutions = collections.abc.Mapping[TypeVariable, TypeExpressions]


@typing.overload
def unify(
    typ: inspect.Signature,
    subtyp: inspect.BoundArguments,
    subs: Substitutions = {},
) -> Substitutions: ...


@typing.overload
def unify(
    typ: TypeExpressions,
    subtyp: TypeExpressions,
    subs: Substitutions = {},
) -> Substitutions: ...


def unify(typ, subtyp, subs: Substitutions = {}) -> Substitutions:
    """
    Unify a pattern type with a concrete type, returning a substitution map.

    This function attempts to find a substitution of type variables that makes
    the pattern type (typ) equal to the concrete type (subtyp). It updates
    and returns the substitution mapping, or raises TypeError if unification
    is not possible.

    The function handles:
    - TypeVar unification (binding type variables to concrete types)
    - Generic type unification (matching origins and recursively unifying args)
    - Structural unification of sequences and mappings
    - Exact type matching for non-generic types

    Args:
        typ: The pattern type that may contain TypeVars to be unified
        subtyp: The concrete type to unify with the pattern
        subs: Existing substitution mappings to be extended (not modified)

    Returns:
        A new substitution mapping that includes all previous substitutions
        plus any new TypeVar bindings discovered during unification.

    Raises:
        TypeError: If unification is not possible (incompatible types or
                   conflicting TypeVar bindings)

    Examples:
        >>> import typing
        >>> T = typing.TypeVar('T')
        >>> K = typing.TypeVar('K')
        >>> V = typing.TypeVar('V')

        >>> # Simple TypeVar unification
        >>> unify(T, int, {})
        {~T: <class 'int'>}

        >>> # Generic type unification
        >>> unify(list[T], list[int], {})
        {~T: <class 'int'>}

        >>> # Exact type matching
        >>> unify(int, int, {})
        {}

        >>> # Failed unification - incompatible types
        >>> unify(list[T], dict[str, int], {})  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        TypeError: Cannot unify ...

        >>> # Failed unification - conflicting TypeVar binding
        >>> unify(T, str, {T: int})  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        TypeError: Cannot unify ...
    """
    if isinstance(typ, inspect.Signature):
        return _unify_signature(typ, subtyp, subs)

    if typ != canonicalize(typ) or subtyp != canonicalize(subtyp):
        return unify(canonicalize(typ), canonicalize(subtyp), subs)

    if typ is subtyp or typ == subtyp:
        return subs
    elif isinstance(typ, TypeVariable) or isinstance(subtyp, TypeVariable):
        return _unify_typevar(typ, subtyp, subs)
    elif isinstance(typ, collections.abc.Sequence) or isinstance(
        subtyp, collections.abc.Sequence
    ):
        return _unify_sequence(typ, subtyp, subs)
    elif isinstance(typ, UnionType) or isinstance(subtyp, UnionType):
        return _unify_union(typ, subtyp, subs)
    elif isinstance(typ, GenericAlias) or isinstance(subtyp, GenericAlias):
        return _unify_generic(typ, subtyp, subs)
    elif isinstance(typ, type) and isinstance(subtyp, type) and issubclass(subtyp, typ):
        return subs
    elif typ in (typing.Any, ...) or subtyp in (typing.Any, ...):
        return subs
    else:
        raise TypeError(f"Cannot unify type {typ} with {subtyp} given {subs}. ")


@typing.overload
def _unify_typevar(
    typ: TypeVariable, subtyp: TypeExpression, subs: Substitutions
) -> Substitutions: ...


@typing.overload
def _unify_typevar(
    typ: TypeExpression, subtyp: TypeVariable, subs: Substitutions
) -> Substitutions: ...


def _unify_typevar(typ, subtyp, subs: Substitutions) -> Substitutions:
    if isinstance(typ, TypeVariable) and isinstance(subtyp, TypeVariable):
        return subs if typ == subtyp else {typ: subtyp, **subs}
    elif isinstance(typ, TypeVariable) and not isinstance(subtyp, TypeVariable):
        return unify(subs.get(typ, subtyp), subtyp, {typ: subtyp, **subs})
    elif (
        not isinstance(typ, TypeVariable)
        and isinstance(subtyp, TypeVariable)
        and getattr(subtyp, "__bound__", None) is None
    ):
        return unify(typ, subs.get(subtyp, typ), {subtyp: typ, **subs})
    else:
        raise TypeError(f"Cannot unify type variable {typ} with {subtyp} given {subs}.")


@typing.overload
def _unify_sequence(
    typ: collections.abc.Sequence, subtyp: TypeExpressions, subs: Substitutions
) -> Substitutions: ...


@typing.overload
def _unify_sequence(
    typ: TypeExpressions, subtyp: collections.abc.Sequence, subs: Substitutions
) -> Substitutions: ...


def _unify_sequence(typ, subtyp, subs: Substitutions) -> Substitutions:
    if isinstance(typ, types.EllipsisType) or isinstance(subtyp, types.EllipsisType):
        return subs
    if len(typ) != len(subtyp):
        raise TypeError(f"Cannot unify sequence {typ} with {subtyp} given {subs}. ")
    for p_item, c_item in zip(typ, subtyp):
        subs = unify(p_item, c_item, subs)
    return subs


@typing.overload
def _unify_union(
    typ: UnionType, subtyp: TypeExpression, subs: Substitutions
) -> Substitutions: ...


@typing.overload
def _unify_union(
    typ: TypeExpression, subtyp: UnionType, subs: Substitutions
) -> Substitutions: ...


def _unify_union(typ, subtyp, subs: Substitutions) -> Substitutions:
    if typ == subtyp:
        return subs
    elif isinstance(subtyp, UnionType):
        # If subtyp is a union, try to unify with each argument
        for arg in typing.get_args(subtyp):
            subs = unify(typ, arg, subs)
        return subs
    elif isinstance(typ, UnionType):
        unifiers: list[Substitutions] = []
        for arg in typing.get_args(typ):
            try:
                unifiers.append(unify(arg, subtyp, subs))
            except TypeError:  # noqa
                continue
        if len(unifiers) > 0 and all(u == unifiers[0] for u in unifiers):
            return unifiers[0]
    raise TypeError(f"Cannot unify {typ} with {subtyp} given {subs}")


@typing.overload
def _unify_generic(
    typ: GenericAlias, subtyp: type, subs: Substitutions
) -> Substitutions: ...


@typing.overload
def _unify_generic(
    typ: type, subtyp: GenericAlias, subs: Substitutions
) -> Substitutions: ...


@typing.overload
def _unify_generic(
    typ: GenericAlias, subtyp: GenericAlias, subs: Substitutions
) -> Substitutions: ...


def _unify_generic(typ, subtyp, subs: Substitutions) -> Substitutions:
    if (
        isinstance(typ, GenericAlias)
        and isinstance(subtyp, GenericAlias)
        and issubclass(typing.get_origin(subtyp), typing.get_origin(typ))
    ):
        if typing.get_origin(subtyp) is tuple and typing.get_origin(typ) is not tuple:
            for arg in typing.get_args(subtyp):
                subs = unify(typ, tuple[arg, ...], subs)  # type: ignore
            return subs
        elif typing.get_origin(subtyp) is collections.abc.Mapping and not issubclass(
            typing.get_origin(typ), collections.abc.Mapping
        ):
            return unify(typing.get_args(typ)[0], typing.get_args(subtyp)[0], subs)
        elif typing.get_origin(subtyp) is collections.abc.Generator and not issubclass(
            typing.get_origin(typ), collections.abc.Generator
        ):
            return unify(typing.get_args(typ)[0], typing.get_args(subtyp)[0], subs)
        elif typing.get_origin(typ) == typing.get_origin(subtyp):
            return unify(typing.get_args(typ), typing.get_args(subtyp), subs)
        elif types.get_original_bases(typing.get_origin(subtyp)):
            for base in types.get_original_bases(typing.get_origin(subtyp)):
                if isinstance(base, type | GenericAlias) and issubclass(
                    typing.get_origin(base) or base,  # type: ignore
                    typing.get_origin(typ),
                ):
                    return unify(typ, base[typing.get_args(subtyp)], subs)  # type: ignore
    elif isinstance(typ, type) and isinstance(subtyp, GenericAlias):
        return unify(typ, typing.get_origin(subtyp), subs)
    elif (
        isinstance(typ, GenericAlias)
        and isinstance(subtyp, type)
        and issubclass(subtyp, typing.get_origin(typ))
    ):
        return subs  # implicit expansion to subtyp[Any]
    raise TypeError(f"Cannot unify generic type {typ} with {subtyp} given {subs}.")


def _unify_signature(
    typ: inspect.Signature, subtyp: inspect.BoundArguments, subs: Substitutions
) -> Substitutions:
    if typ != subtyp.signature:
        raise TypeError(f"Cannot unify {typ} with {subtyp} given {subs}. ")

    for name, param in typ.parameters.items():
        if param.annotation is inspect.Parameter.empty:
            continue

        if name not in subtyp.arguments:
            assert param.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
            continue

        ptyp, psubtyp = param.annotation, subtyp.arguments[name]
        if param.kind is inspect.Parameter.VAR_POSITIONAL and isinstance(
            psubtyp, collections.abc.Sequence
        ):
            for psubtyp_item in _freshen(psubtyp):
                subs = unify(ptyp, psubtyp_item, subs)
        elif param.kind is inspect.Parameter.VAR_KEYWORD and isinstance(
            psubtyp, collections.abc.Mapping
        ):
            for psubtyp_item in _freshen(tuple(psubtyp.values())):
                subs = unify(ptyp, psubtyp_item, subs)
        elif param.kind not in {
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        } or isinstance(psubtyp, typing.ParamSpecArgs | typing.ParamSpecKwargs):
            subs = unify(ptyp, _freshen(psubtyp), subs)
        else:
            raise TypeError(f"Cannot unify {param} with {psubtyp} given {subs}")
    return subs


def _freshen(tp: typing.Any):
    """
    Return a freshened version of the given type expression.

    This function replaces all TypeVars in the type expression with new TypeVars
    that have unique names, ensuring that the resulting type has no free TypeVars.
    It is useful for creating fresh type variables in generic programming contexts.

    Args:
        tp: The type expression to freshen. Can be a plain type, TypeVar,
            generic alias, or union type.

    Returns:
        A new type expression with all TypeVars replaced by fresh TypeVars.

    Examples:
        >>> import typing
        >>> T = typing.TypeVar('T')
        >>> isinstance(_freshen(T), typing.TypeVar)
        True
        >>> _freshen(T) == T
        False
    """
    assert all(canonicalize(fv) is fv for fv in freetypevars(tp))
    subs: Substitutions = {
        fv: typing.TypeVar(fv.__name__, bound=fv.__bound__)
        if isinstance(fv, typing.TypeVar)
        else typing.ParamSpec(fv.__name__)
        for fv in freetypevars(tp)
        if isinstance(fv, typing.TypeVar | typing.ParamSpec)
    }
    return substitute(tp, subs)


@functools.singledispatch
def canonicalize(typ) -> TypeExpressions:
    """
    Normalize generic types
    """
    raise TypeError(f"Cannot canonicalize type {typ}.")


@canonicalize.register
def _(typ: type | abc.ABCMeta):
    if issubclass(typ, effectful.ops.types.Term):
        return effectful.ops.types.Term
    elif issubclass(typ, effectful.ops.types.Operation):
        return effectful.ops.types.Operation
    elif typ is dict:
        return collections.abc.MutableMapping
    elif typ is list:
        return collections.abc.MutableSequence
    elif typ is set:
        return collections.abc.MutableSet
    elif typ is frozenset:
        return collections.abc.Set
    elif typ is range:
        return collections.abc.Sequence[int]
    elif typ is types.GeneratorType:
        return collections.abc.Generator
    elif typ in {types.FunctionType, types.BuiltinFunctionType, types.LambdaType}:
        return collections.abc.Callable[..., typing.Any]
    elif isinstance(typ, abc.ABCMeta) and (
        typ in collections.abc.__dict__.values() or typ in numbers.__dict__.values()
    ):
        return typ
    elif isinstance(typ, type) and (
        typ in builtins.__dict__.values() or typ in types.__dict__.values()
    ):
        return typ
    elif types.get_original_bases(typ):
        for base in types.get_original_bases(typ):
            if typing.get_origin(base) is not typing.Generic:
                cbase = canonicalize(base)
                if cbase != object:
                    return cbase
        return typ
    else:
        raise TypeError(f"Cannot canonicalize type {typ}.")


@canonicalize.register
def _(typ: types.EllipsisType | None):
    return typ


@canonicalize.register
def _(typ: typing.TypeVar):
    if (
        typ.__constraints__
        or typ.__covariant__
        or typ.__contravariant__
        or getattr(typ, "__default__", None) is not getattr(typing, "NoDefault", None)
    ):
        raise TypeError(f"Cannot canonicalize typevar {typ} with nonempty attributes")
    return typ


@canonicalize.register
def _(typ: typing.ParamSpec):
    if (
        typ.__bound__
        or typ.__covariant__
        or typ.__contravariant__
        or getattr(typ, "__default__", None) is not getattr(typing, "NoDefault", None)
    ):
        raise TypeError(f"Cannot canonicalize typevar {typ} with nonempty attributes")
    return typ


@canonicalize.register
def _(typ: typing.TypeVarTuple):
    if getattr(typ, "__default__", None) is not getattr(typing, "NoDefault", None):
        raise TypeError(f"Cannot canonicalize typevar {typ} with nonempty attributes")
    return typ


@canonicalize.register
def _(typ: UnionType):
    ctyp = canonicalize(typing.get_args(typ)[0])
    for arg in typing.get_args(typ)[1:]:
        ctyp = ctyp | canonicalize(arg)  # type: ignore
    return ctyp


@canonicalize.register
def _(typ: GenericAlias):
    origin, args = typing.get_origin(typ), typing.get_args(typ)
    if origin is tuple and len(args) == 2 and args[-1] is Ellipsis:  # Variadic tuple
        return collections.abc.Sequence[canonicalize(args[0])]  # type: ignore
    elif isinstance(origin, typing._SpecialForm):
        if len(args) == 1:
            return canonicalize(args[0])
        else:
            raise TypeError(f"Cannot canonicalize type {typ}")
    else:
        return canonicalize(origin)[tuple(canonicalize(a) for a in args)]  # type: ignore


@canonicalize.register
def _(typ: list | tuple):
    return type(typ)(canonicalize(item) for item in typ)


@canonicalize.register
def _(typ: effectful.ops.types._InterpretationMeta):
    return typ


@canonicalize.register
def _(typ: typing._AnnotatedAlias):  # type: ignore
    return canonicalize(typing.get_args(typ)[0])


@canonicalize.register
def _(typ: typing._SpecialGenericAlias):  # type: ignore
    assert not typing.get_args(typ), "Should not have type arguments"
    return canonicalize(typing.get_origin(typ))


@canonicalize.register
def _(typ: typing._LiteralGenericAlias):  # type: ignore
    return canonicalize(nested_type(typing.get_args(typ)[0]))


@canonicalize.register
def _(typ: typing.NewType):
    return canonicalize(typ.__supertype__)


@canonicalize.register
def _(typ: typing.TypeAliasType):
    return canonicalize(typ.__value__)


@canonicalize.register
def _(typ: typing._ConcatenateGenericAlias):  # type: ignore
    return Ellipsis


@canonicalize.register
def _(typ: typing._AnyMeta):  # type: ignore
    return typing.Any


@canonicalize.register
def _(typ: typing.ParamSpecArgs | typing.ParamSpecKwargs):
    return typing.Any


@canonicalize.register
def _(typ: typing._SpecialForm):
    return typing.Any


@canonicalize.register
def _(typ: typing._ProtocolMeta):
    return typing.Any


@canonicalize.register
def _(typ: typing._UnpackGenericAlias):  # type: ignore
    raise TypeError(f"Cannot canonicalize type {typ}")


@canonicalize.register
def _(typ: typing.ForwardRef):
    if typ.__forward_value__ is not None:
        return canonicalize(typ.__forward_value__)
    else:
        raise TypeError(f"Cannot canonicalize lazy ForwardRef {typ}.")


@functools.singledispatch
def nested_type(value) -> TypeExpression:
    """
    Infer the type of a value, handling nested collections with generic parameters.

    This function is a singledispatch generic function that determines the type
    of a given value. For collections (mappings, sequences, sets), it recursively
    infers the types of contained elements to produce a properly parameterized
    generic type. For example, a list [1, 2, 3] becomes Sequence[int].

    The function handles:
    - Basic types and type annotations (passed through unchanged)
    - Collections with recursive type inference for elements
    - Special cases like str/bytes (treated as types, not sequences)
    - Tuples (preserving exact element types)
    - Empty collections (returning the collection's type without parameters)

    This is primarily used by canonicalize() to handle cases where values
    are provided instead of type annotations.

    Args:
        value: Any value whose type needs to be inferred. Can be a type,
               a value instance, or a collection containing other values.

    Returns:
        The inferred type, potentially with generic parameters for collections.

    Raises:
        TypeError: If the value is a TypeVar (TypeVars shouldn't appear in values)
                   or if the value is a Term from effectful.ops.types.

    Examples:
        >>> import collections.abc
        >>> import typing
        >>> from effectful.internals.unification import nested_type

        # Basic types are returned as their type
        >>> nested_type(42)
        <class 'int'>
        >>> nested_type("hello")
        <class 'str'>
        >>> nested_type(3.14)
        <class 'float'>
        >>> nested_type(True)
        <class 'bool'>

        # Type objects pass through unchanged
        >>> nested_type(int)
        <class 'int'>
        >>> nested_type(str)
        <class 'str'>
        >>> nested_type(list)
        <class 'list'>

        # Empty collections return their base type
        >>> nested_type([])
        <class 'list'>
        >>> nested_type({})
        <class 'dict'>
        >>> nested_type(set())
        <class 'set'>

        # Sequences become Sequence[element_type]
        >>> nested_type([1, 2, 3])
        collections.abc.MutableSequence[int]
        >>> nested_type(["a", "b", "c"])
        collections.abc.MutableSequence[str]

        # Tuples preserve exact structure
        >>> nested_type((1, "hello", 3.14))
        tuple[int, str, float]
        >>> nested_type(())
        <class 'tuple'>
        >>> nested_type((1,))
        tuple[int]

        # Sets become Set[element_type]
        >>> nested_type({1, 2, 3})
        collections.abc.MutableSet[int]
        >>> nested_type({"a", "b"})
        collections.abc.MutableSet[str]

        # Mappings become Mapping[key_type, value_type]
        >>> nested_type({"key": "value"})
        collections.abc.MutableMapping[str, str]
        >>> nested_type({1: "one", 2: "two"})
        collections.abc.MutableMapping[int, str]

        # Strings and bytes are NOT treated as sequences
        >>> nested_type("hello")
        <class 'str'>
        >>> nested_type(b"bytes")
        <class 'bytes'>

        # Annotated functions return types derived from their annotations
        >>> def annotated_func(x: int) -> str:
        ...     return str(x)
        >>> nested_type(annotated_func)
        collections.abc.Callable[[int], str]

        # Unannotated functions/callables return their type
        >>> def f(): pass
        >>> nested_type(f)
        <class 'function'>
        >>> nested_type(lambda x: x)
        <class 'function'>

        # Generic aliases and union types pass through
        >>> nested_type(list[int])
        list[int]
        >>> nested_type(int | str)
        int | str
    """
    return type(value)


@nested_type.register
def _(value: TypeExpression):
    return value


@nested_type.register
def _(value: effectful.ops.types.Term):
    raise TypeError(f"Terms should not appear in nested_type, but got {value}")


@nested_type.register
def _(value: effectful.ops.types.Operation):
    typ = nested_type.dispatch(collections.abc.Callable)(value)
    (arg_types, return_type) = typing.get_args(typ)
    return effectful.ops.types.Operation[arg_types, return_type]  # type: ignore


@nested_type.register
def _(value: collections.abc.Callable):
    if typing.get_overloads(value):
        return type(value)

    try:
        sig = inspect.signature(value)
    except ValueError:
        return type(value)

    if sig.return_annotation is inspect.Signature.empty:
        return type(value)
    elif any(
        p.annotation is inspect.Parameter.empty
        or p.kind
        in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
        for p in sig.parameters.values()
    ):
        return collections.abc.Callable[..., sig.return_annotation]
    else:
        return collections.abc.Callable[
            [p.annotation for p in sig.parameters.values()], sig.return_annotation
        ]


@nested_type.register
def _(value: collections.abc.Mapping):
    if value and isinstance(value, effectful.ops.types.Interpretation):
        return effectful.ops.types.Interpretation

    if len(value) == 0:
        return type(value)
    elif len(value) == 1:
        ktyp = nested_type(next(iter(value.keys())))
        vtyp = nested_type(next(iter(value.values())))
        return canonicalize(type(value))[ktyp, vtyp]  # type: ignore
    else:
        ktyp = functools.reduce(operator.or_, map(nested_type, value.keys()))
        vtyp = functools.reduce(operator.or_, map(nested_type, value.values()))
        if isinstance(ktyp, UnionType) or isinstance(vtyp, UnionType):
            return type(value)
        else:
            return canonicalize(type(value))[ktyp, vtyp]  # type: ignore


@nested_type.register
def _(value: collections.abc.Collection):
    if len(value) == 0:
        return type(value)
    elif len(value) == 1:
        vtyp = nested_type(next(iter(value)))
        return canonicalize(type(value))[vtyp]  # type: ignore
    else:
        valtyp = functools.reduce(operator.or_, map(nested_type, value))
        if isinstance(valtyp, UnionType):
            return type(value)
        else:
            return canonicalize(type(value))[valtyp]  # type: ignore


@nested_type.register
def _(value: tuple):
    return (
        nested_type.dispatch(collections.abc.Sequence)(value)
        if type(value) != tuple or len(value) == 0
        else tuple[tuple(nested_type(item) for item in value)]  # type: ignore
    )


@nested_type.register
def _(value: str | bytes | range | None):
    return type(value)


def freetypevars(typ) -> collections.abc.Set[TypeVariable]:
    """
    Return a set of free type variables in the given type expression.

    This function recursively traverses a type expression to find all TypeVar
    instances that appear within it. It handles both simple types and generic
    type aliases with nested type arguments. TypeVars are considered "free"
    when they are not bound to a specific concrete type.

    Args:
        typ: The type expression to analyze. Can be a plain type (e.g., int),
             a TypeVar, or a generic type alias (e.g., List[T], Dict[K, V]).

    Returns:
        A set containing all TypeVar instances found in the type expression.
        Returns an empty set if no TypeVars are present.

    Examples:
        >>> T = typing.TypeVar('T')
        >>> K = typing.TypeVar('K')
        >>> V = typing.TypeVar('V')

        >>> # TypeVar returns itself
        >>> freetypevars(T)
        {~T}

        >>> # Generic type with one TypeVar
        >>> freetypevars(list[T])
        {~T}

        >>> # Generic type with multiple TypeVars
        >>> freetypevars(dict[K, V]) == {K, V}
        True

        >>> # Nested generic types
        >>> freetypevars(list[dict[K, V]]) == {K, V}
        True

        >>> # Concrete types have no free TypeVars
        >>> freetypevars(int)
        set()

        >>> # Generic types with concrete arguments have no free TypeVars
        >>> freetypevars(list[int])
        set()

        >>> # Mixed concrete and TypeVar arguments
        >>> freetypevars(dict[str, T])
        {~T}
    """
    return set(_freetypevars((typ,)))


def substitute(typ, subs: Substitutions) -> TypeExpressions:
    """
    Substitute type variables in a type expression with concrete types.

    This function recursively traverses a type expression and replaces any TypeVar
    instances found with their corresponding concrete types from the substitution
    mapping. If a TypeVar is not present in the substitution mapping, it remains
    unchanged. The function handles nested generic types by recursively substituting
    in their type arguments.

    Args:
        typ: The type expression to perform substitution on. Can be a plain type,
             a TypeVar, or a generic type alias (e.g., List[T], Dict[K, V]).
        subs: A mapping from TypeVar instances to concrete types that should
              replace them.

    Returns:
        A new type expression with all mapped TypeVars replaced by their
        corresponding concrete types.

    Examples:
        >>> T = typing.TypeVar('T')
        >>> K = typing.TypeVar('K')
        >>> V = typing.TypeVar('V')

        >>> # Simple TypeVar substitution
        >>> substitute(T, {T: int})
        <class 'int'>

        >>> # Generic type substitution
        >>> substitute(list[T], {T: str})
        list[str]

        >>> # Nested generic substitution
        >>> substitute(dict[K, list[V]], {K: str, V: int})
        dict[str, list[int]]

        >>> # TypeVar not in mapping remains unchanged
        >>> substitute(T, {K: int})
        ~T

        >>> # Non-generic types pass through unchanged
        >>> substitute(int, {T: str})
        <class 'int'>
    """
    if isinstance(typ, typing.TypeVar | typing.ParamSpec | typing.TypeVarTuple):
        return substitute(subs[typ], subs) if typ in subs else typ
    elif isinstance(typ, list | tuple):
        return type(typ)(substitute(item, subs) for item in typ)
    elif any(fv in subs for fv in freetypevars(typ)):
        args = tuple(subs.get(fv, fv) for fv in _freetypevars((typ,)))
        return substitute(typ[args], subs)
    else:
        return typ
