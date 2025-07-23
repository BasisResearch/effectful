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
import random
import types
import typing

import effectful.ops.types

if typing.TYPE_CHECKING:
    GenericAlias = types.GenericAlias
    UnionType = types.UnionType
else:
    GenericAlias = types.GenericAlias | typing._GenericAlias
    UnionType = types.UnionType | typing._UnionGenericAlias


TypeVariable = (
    typing.TypeVar
    | typing.TypeVarTuple
    | typing.ParamSpec
    | typing.ParamSpecArgs
    | typing.ParamSpecKwargs
)

TypeConstant = type | abc.ABCMeta | types.EllipsisType | None

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

    if isinstance(typ, TypeVariable) or isinstance(subtyp, TypeVariable):
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
        return subs if typ == subtyp or subtyp is Ellipsis else {typ: subtyp, **subs}
    elif isinstance(typ, TypeVariable) and not isinstance(subtyp, TypeVariable):
        return unify(subs.get(typ, subtyp), subtyp, {typ: subtyp, **subs})
    else:
        return unify(typ, subs.get(subtyp, typ), {subtyp: typ, **subs})


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
    if isinstance(subtyp, UnionType):
        # If subtyp is a union, try to unify with each argument
        for arg in typing.get_args(subtyp):
            subs = unify(typ, arg, subs)
        return subs
    elif isinstance(typ, UnionType):
        any_succeeded = False
        for arg in typing.get_args(typ):
            try:
                subs = unify(arg, subtyp, subs)
                any_succeeded = True
            except TypeError:  # noqa
                continue
        if any_succeeded:
            return subs
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

    subtyp_arguments = dict(subtyp.arguments)
    for name, param in typ.parameters.items():
        if name not in subtyp_arguments:
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                subtyp_arguments[name] = ()
            elif param.kind is inspect.Parameter.VAR_KEYWORD:
                subtyp_arguments[name] = {}
            elif param.default is not inspect.Parameter.empty:
                subtyp_arguments[name] = nested_type(param.default)
            else:
                subtyp_arguments[name] = inspect.Parameter.empty

        ptyp, psubtyp = param.annotation, subtyp_arguments[name]
        if ptyp is inspect.Parameter.empty or psubtyp is inspect.Parameter.empty:
            continue
        elif param.kind is inspect.Parameter.VAR_POSITIONAL and isinstance(
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
    fvs = freetypevars(tp)
    subs: dict[TypeVariable, TypeExpressions] = {}
    for fv in fvs:
        if isinstance(fv, typing.TypeVar):
            prefix = fv.__name__[:60]
            freshening = random.randint(0, int(1e7))
            name = f"{prefix}_{freshening}"
            bound = fv.__bound__
            subs[fv] = typing.TypeVar(name, bound=bound)
        elif isinstance(fv, typing.ParamSpec):
            prefix = fv.__name__[:60]
            freshening = random.randint(0, int(1e7))
            name = f"{prefix}_{freshening}"
            subs[fv] = typing.ParamSpec(name)
        else:
            continue
    return substitute(tp, subs) if subs else tp


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
    elif typ is effectful.ops.types.Interpretation:
        return effectful.ops.types.Interpretation
    elif typ is dict:
        return collections.abc.MutableMapping
    elif typ is list:
        return collections.abc.MutableSequence
    elif typ is set:
        return collections.abc.MutableSet
    elif typ is frozenset:
        return collections.abc.Set
    elif typ is types.GeneratorType:
        return collections.abc.Generator
    elif typ in {types.FunctionType, types.BuiltinFunctionType, types.LambdaType}:
        return collections.abc.Callable[..., typing.Any]
    elif typ is typing.Any:
        return typing.Any
    elif isinstance(typ, abc.ABCMeta) and (
        typ in collections.abc.__dict__.values() or typ in numbers.__dict__.values()
    ):
        return typ
    elif isinstance(typ, type) and (
        typ in builtins.__dict__.values() or typ in types.__dict__.values()
    ):
        return typ
    elif types.get_original_bases(typ):
        return canonicalize(types.get_original_bases(typ)[0])
    else:
        raise TypeError(f"Cannot canonicalize type {typ}.")


@canonicalize.register
def _(typ: types.EllipsisType | None):
    return typ


@canonicalize.register
def _(typ: typing.TypeVar | typing.TypeVarTuple | typing.ParamSpec):
    return typ


@canonicalize.register
def _(typ: typing.ParamSpecArgs | typing.ParamSpecKwargs):
    return typing.Any


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
    else:
        return canonicalize(origin)[tuple(canonicalize(a) for a in args)]  # type: ignore


@canonicalize.register
def _(typ: list | tuple):
    return type(typ)(canonicalize(item) for item in typ)


@canonicalize.register
def _(typ: typing._AnnotatedAlias):  # type: ignore
    return canonicalize(typing.get_args(typ)[0])


@functools.singledispatch
def nested_type(value) -> TypeConstant | TypeApplication:
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
        list[int]
        >>> nested_type(["a", "b", "c"])
        list[str]

        # Tuples preserve exact structure
        >>> nested_type((1, "hello", 3.14))
        tuple[int, str, float]
        >>> nested_type(())
        <class 'tuple'>
        >>> nested_type((1,))
        tuple[int]

        # Sets become Set[element_type]
        >>> nested_type({1, 2, 3})
        set[int]
        >>> nested_type({"a", "b"})
        set[str]

        # Mappings become Mapping[key_type, value_type]
        >>> nested_type({"key": "value"})
        dict[str, str]
        >>> nested_type({1: "one", 2: "two"})
        dict[int, str]

        # Nested collections work recursively
        >>> nested_type([{1: "one"}, {2: "two"}])
        list[dict[int, str]]
        >>> nested_type({"key": [1, 2, 3]})
        dict[str, list[int]]

        # Strings and bytes are NOT treated as sequences
        >>> nested_type("hello")
        <class 'str'>
        >>> nested_type(b"bytes")
        <class 'bytes'>

        # Functions/callables return their type
        >>> def f(): pass
        >>> nested_type(f)
        <class 'function'>
        >>> nested_type(lambda x: x)
        <class 'function'>

        # TypeVars raise an error
        >>> T = typing.TypeVar('T')
        >>> nested_type(T)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        TypeError: TypeVars should not appear in values, but got ~T

        # None has its own type
        >>> nested_type(None)
        <class 'NoneType'>

        # Generic aliases and union types pass through
        >>> nested_type(list[int])
        list[int]
        >>> nested_type(int | str)
        int | str
    """
    if not isinstance(value, type) and typing.get_origin(value) is None:
        return type(value)
    else:
        return value


@nested_type.register
def _(value: effectful.ops.types.Operation):
    sig: inspect.Signature = value.__signature__
    if sig.return_annotation is inspect.Signature.empty:
        return effectful.ops.types.Operation[..., typing.Any]
    elif any(
        p.annotation is inspect.Parameter.empty
        or p.kind is not inspect.Parameter.POSITIONAL_ONLY
        for p in sig.parameters.values()
    ):
        return effectful.ops.types.Operation[..., sig.return_annotation]  # type: ignore
    else:
        return effectful.ops.types.Operation[  # type: ignore
            [p.annotation for p in sig.parameters.values()], sig.return_annotation
        ]


@nested_type.register
def _(value: effectful.ops.types.Term):
    raise TypeError(f"Terms should not appear in nested_type, but got {value}")


@nested_type.register
def _(value: TypeVariable):
    raise TypeError(f"TypeVars should not appear in values, but got {value}")


@nested_type.register
def _(value: TypeConstant | TypeApplication):
    return value


@nested_type.register(type(None))
def _(value: None):
    return type(None)


@nested_type.register
def _(value: collections.abc.Callable):
    try:
        sig = inspect.signature(value)
    except ValueError:
        return type(value)

    if sig.return_annotation is inspect.Signature.empty:
        return type(value)
    elif any(
        p.annotation is inspect.Parameter.empty
        or p.kind is not inspect.Parameter.POSITIONAL_ONLY
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
    else:
        k, v = next(iter(value.items()))
        return collections.abc.Mapping[nested_type(k), nested_type(v)]  # type: ignore


@nested_type.register
def _(value: collections.abc.MutableMapping):
    if value and isinstance(value, effectful.ops.types.Interpretation):
        return effectful.ops.types.Interpretation

    args = typing.get_args(nested_type.dispatch(collections.abc.Mapping)(value))
    return type(value) if not args else collections.abc.MutableMapping[args]  # type: ignore


@nested_type.register
def _(value: collections.abc.Set):
    if len(value) == 0:
        return type(value)
    return collections.abc.Set[nested_type(next(iter(value)))]  # type: ignore


@nested_type.register
def _(value: collections.abc.MutableSet):
    args = typing.get_args(nested_type.dispatch(collections.abc.Set)(value))
    return type(value) if not args else collections.abc.MutableSet[args]  # type: ignore


@nested_type.register
def _(value: collections.abc.Sequence):
    if len(value) == 0:
        return type(value)
    return collections.abc.Sequence[nested_type(next(iter(value)))]  # type: ignore


@nested_type.register
def _(value: collections.abc.MutableSequence):
    args = typing.get_args(nested_type.dispatch(collections.abc.Sequence)(value))
    return type(value) if not args else collections.abc.MutableSequence[args]  # type: ignore


@nested_type.register
def _(value: tuple):
    return tuple[tuple(nested_type(item) for item in value)]  # type: ignore


@nested_type.register
def _(value: str | bytes | range | numbers.Number):
    return type(value)


@functools.singledispatch
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
        >>> sorted(freetypevars(dict[K, V]), key=lambda x: x.__name__)
        [~K, ~V]

        >>> # Nested generic types
        >>> sorted(freetypevars(list[dict[K, V]]), key=lambda x: x.__name__)
        [~K, ~V]

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
    # Default case for plain types
    return freetypevars(typing.get_args(typ))


@freetypevars.register
def _(typ: TypeVariable):
    if isinstance(typ, typing.ParamSpecArgs | typing.ParamSpecKwargs):
        return {typing.get_origin(typ)}
    else:
        return {typ}


@freetypevars.register
def _(typ: TypeApplication):
    return freetypevars(typing.get_args(typ))


@freetypevars.register
def _(typ: typing._AnnotatedAlias):  # type: ignore
    return freetypevars(typing.get_args(typ)[0])


@freetypevars.register
def _(typ: collections.abc.Sequence):
    return set().union(*(freetypevars(item) for item in typ))


@freetypevars.register
def _(typ: str | bytes):
    return set()


@freetypevars.register
def _(typ: dict):
    assert all(isinstance(k, str) for k in typ.keys()), "Mapping keys must be strings"
    return freetypevars(typ.values())


@functools.singledispatch
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
    # Default case for plain types
    return typ


@substitute.register
def _(typ: typing.TypeVar | typing.ParamSpec, subs: Substitutions):
    return substitute(subs[typ], subs) if typ in subs else typ


@substitute.register
def _(typ: typing.ParamSpecArgs | typing.ParamSpecKwargs, subs: Substitutions):
    res = substitute(typing.get_origin(typ), subs)
    return res.args if isinstance(res, typing.ParamSpec) else res


@substitute.register
def _(typ: GenericAlias, subs: Substitutions):
    if typing.get_args(typ):
        return substitute(typing.get_origin(typ), subs)[
            substitute(typing.get_args(typ), subs)
        ]  # type: ignore
    else:
        return typ


@substitute.register
def _(typ: UnionType, subs: Substitutions):
    ts: tuple = substitute(typing.get_args(typ), subs)  # type: ignore
    tp, ts = ts[0], ts[1:]
    for arg in ts:
        tp = tp | arg
    return tp


@substitute.register
def _(typ: list | tuple, subs: Substitutions):
    return type(typ)(substitute(item, subs) for item in typ)


@substitute.register
def _(typ: dict, subs: Substitutions):
    assert all(isinstance(k, str) for k in typ.keys()), "Mapping keys must be strings"
    return {k: substitute(v, subs) for k, v in typ.items()}
