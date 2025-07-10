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
    >>> freetypevars(dict[K, list[V]])
    {~K, ~V}

This module is primarily used internally by effectful for type inference in its
effect system, allowing it to track and propagate type information through
effect handlers and operations.
"""

import collections.abc
import functools
import inspect
import random
import types
import typing

if typing.TYPE_CHECKING:
    GenericAlias = types.GenericAlias
else:
    GenericAlias = types.GenericAlias | typing._GenericAlias


Substitutions = collections.abc.Mapping[
    typing.TypeVar | typing.ParamSpec,
    type
    | typing.TypeVar
    | typing.ParamSpec
    | collections.abc.Sequence
    | collections.abc.Mapping[str, typing.Any],
]


@functools.singledispatch
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
    raise TypeError(f"Cannot unify {typ} with {subtyp} given {subs}")


@unify.register
def _(
    typ: inspect.Signature, subtyp: inspect.BoundArguments, subs: Substitutions = {}
) -> Substitutions:
    if typ != subtyp.signature:
        raise TypeError(f"Cannot unify {typ} with {subtyp} given {subs}. ")

    subtyp_arguments = dict(subtyp.arguments)
    for name, param in typ.parameters.items():
        if name in subtyp_arguments:
            continue
        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            subtyp_arguments[name] = ()
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            subtyp_arguments[name] = {}
        elif param.default is not inspect.Parameter.empty:
            subtyp_arguments[name] = nested_type(param.default)
        else:
            subtyp_arguments[name] = inspect.Parameter.empty
    return unify(typ.parameters, subtyp_arguments, subs)


@unify.register
def _(
    typ: inspect.Parameter,
    subtyp: collections.abc.Sequence
    | collections.abc.Mapping
    | type
    | typing.ParamSpecArgs
    | typing.ParamSpecKwargs,
    subs: Substitutions = {},
) -> Substitutions:
    if subtyp is inspect.Parameter.empty:
        return subs
    elif typ.kind is inspect.Parameter.VAR_POSITIONAL and isinstance(
        subtyp, collections.abc.Sequence
    ):
        return unify(tuple(typ.annotation for _ in subtyp), _freshen(subtyp), subs)
    elif typ.kind is inspect.Parameter.VAR_KEYWORD and isinstance(
        subtyp, collections.abc.Mapping
    ):
        return unify(
            tuple(typ.annotation for _ in subtyp),
            _freshen(tuple(subtyp.values())),
            subs,
        )
    elif typ.kind not in {
        inspect.Parameter.VAR_KEYWORD,
        inspect.Parameter.VAR_POSITIONAL,
    } or isinstance(subtyp, typing.ParamSpecArgs | typing.ParamSpecKwargs):
        return unify(typ.annotation, _freshen(subtyp), subs)
    else:
        raise TypeError(f"Cannot unify parameter {typ} with {subtyp} given {subs}. ")


@unify.register
def _(
    typ: collections.abc.Mapping,
    subtyp: collections.abc.Mapping[str, typing.Any],
    subs: Substitutions = {},
) -> Substitutions:
    if set(typ.keys()) != set(subtyp.keys()):
        raise TypeError(f"Cannot unify mapping type {typ} with {subtyp} given {subs}. ")
    for k in typ.keys():
        subs = unify(typ[k], subtyp[k], subs)
    return subs


@unify.register
def _(
    typ: collections.abc.Sequence,
    subtyp: collections.abc.Sequence,
    subs: Substitutions = {},
) -> Substitutions:
    if len(typ) != len(subtyp):
        raise TypeError(
            f"Cannot unify sequence type {typ} with {subtyp} given {subs}. "
        )
    for p_item, c_item in zip(typ, subtyp):
        subs = unify(p_item, c_item, subs)
    return subs


@unify.register
def _(
    typ: typing._AnnotatedAlias,  # type: ignore
    subtyp: type,
    subs: Substitutions = {},
) -> Substitutions:
    return unify(typing.get_args(typ)[0], subtyp, subs)


@unify.register
def _(
    typ: types.UnionType,
    subtyp: type,
    subs: Substitutions = {},
) -> Substitutions:
    any_succeeded = False
    for arg in typing.get_args(typ):
        try:
            subs = unify(arg, subtyp, subs)
            any_succeeded = True
        except TypeError:  # noqa
            continue
    if any_succeeded:
        return subs
    else:
        raise TypeError(f"Cannot unify {typ} with {subtyp} given {subs}")


@unify.register
def _(
    typ: GenericAlias,
    subtyp: type | types.GenericAlias | typing.TypeVar | types.UnionType,
    subs: Substitutions = {},
) -> Substitutions:
    if isinstance(subtyp, GenericAlias):
        subs = unify(typing.get_origin(typ), typing.get_origin(subtyp), subs)
        return unify(typing.get_args(typ), typing.get_args(subtyp), subs)
    else:
        return unify.dispatch(type)(typ, subtyp, subs)


@unify.register
def _(
    typ: type,
    subtyp: type | typing.TypeVar | types.UnionType | GenericAlias,
    subs: Substitutions = {},
) -> Substitutions:
    if isinstance(subtyp, typing.TypeVar):
        return unify(subtyp, subs.get(subtyp, typ), {subtyp: typ, **subs})
    elif isinstance(subtyp, types.UnionType):
        for arg in typing.get_args(subtyp):
            subs = unify(typ, arg, subs)
        return subs
    elif isinstance(subtyp, typing._AnnotatedAlias):  # type: ignore
        return unify(typ, typing.get_args(subtyp)[0], subs)
    elif isinstance(subtyp, GenericAlias) and issubclass(
        typing.get_origin(subtyp), typ
    ):
        return subs
    elif isinstance(subtyp, type) and issubclass(subtyp, typing.get_origin(typ) or typ):
        return subs
    else:
        raise TypeError(f"Cannot unify type {typ} with {subtyp} given {subs}. ")


@unify.register
def _(
    typ: typing.TypeVar,
    subtyp: type | typing.TypeVar | types.UnionType | types.GenericAlias,
    subs: Substitutions = {},
) -> Substitutions:
    return (
        subs
        if typ is subtyp
        else unify(subtyp, subs.get(typ, subtyp), {typ: subtyp, **subs})  # type: ignore
    )


@unify.register
def _(
    typ: typing.ParamSpecArgs,
    subtyp: collections.abc.Sequence,
    subs: Substitutions = {},
) -> Substitutions:
    return subs  # {typ: subtyp, **subs}


@unify.register
def _(
    typ: typing.ParamSpecKwargs,
    subtyp: collections.abc.Mapping,
    subs: Substitutions = {},
) -> Substitutions:
    return subs  # {typ: subtyp, **subs}


@unify.register
def _(
    typ: typing.ParamSpec,
    subtyp: typing.ParamSpec | collections.abc.Sequence,
    subs: Substitutions = {},
) -> Substitutions:
    return subs if typ is subtyp else {typ: subtyp, **subs}


@unify.register
def _(
    typ: types.EllipsisType,
    subtyp: types.EllipsisType | collections.abc.Sequence,
    subs: Substitutions = {},
) -> Substitutions:
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
        >>> isinstance(freshen(T), typing.TypeVar)
        True
        >>> freshen(T) == T
        False
    """
    return substitute(
        tp,
        {
            fv: typing.TypeVar(
                name=f"{fv.__name__[:100]}_{random.randint(0, 1 << 32)}",
                bound=fv.__bound__,
                covariant=fv.__covariant__,
                contravariant=fv.__contravariant__,
            )
            if isinstance(fv, typing.TypeVar)
            else typing.ParamSpec(
                name=f"{fv.__name__[:100]}_{random.randint(0, 1 << 32)}"
            )
            for fv in freetypevars(tp)
            if isinstance(fv, typing.TypeVar | typing.ParamSpec)
        },
    )


@functools.singledispatch
def nested_type(
    value,
) -> type | GenericAlias | types.UnionType | types.EllipsisType | None:
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
    from effectful.ops.types import Term

    if isinstance(value, Term):
        raise TypeError(f"Terms should not appear in nested_type, but got {value}")
    elif not isinstance(value, type) and typing.get_origin(value) is None:
        return type(value)
    else:
        return value


@nested_type.register
def _(value: type | types.UnionType | GenericAlias | types.EllipsisType):
    return value


@nested_type.register(type(None))
def _(value: None):
    return type(None)


@nested_type.register
def _(value: typing.TypeVar):
    raise TypeError(f"TypeVars should not appear in values, but got {value}")


@nested_type.register
def _(value: collections.abc.Callable):
    return type(value)


@nested_type.register
def _(value: collections.abc.Mapping):
    from effectful.ops.types import Interpretation

    if type(value) is Interpretation:  # More specific check
        return Interpretation
    elif len(value) == 0:
        return type(value)
    else:
        k, v = next(iter(value.items()))
        return collections.abc.Mapping[nested_type(k), nested_type(v)]  # type: ignore


@nested_type.register
def _(value: collections.abc.Set):
    if len(value) == 0:
        return type(value)
    return collections.abc.Set[nested_type(next(iter(value)))]  # type: ignore


@nested_type.register
def _(value: list):
    if len(value) == 0:
        return list
    return list[nested_type(next(iter(value)))]  # type: ignore


@nested_type.register
def _(value: dict):
    if len(value) == 0:
        return dict
    k, v = next(iter(value.items()))
    return dict[nested_type(k), nested_type(v)]  # type: ignore


@nested_type.register
def _(value: set):
    if len(value) == 0:
        return set
    return set[nested_type(next(iter(value)))]  # type: ignore


@nested_type.register
def _(value: collections.abc.Sequence):
    if len(value) == 0:
        return type(value)
    return collections.abc.Sequence[nested_type(next(iter(value)))]  # type: ignore


@nested_type.register
def _(value: tuple):
    if len(value) == 0:
        return tuple
    return tuple[tuple(nested_type(item) for item in value)]  # type: ignore


@nested_type.register
def _(value: str | bytes):
    return type(value)


@nested_type.register(range)
def _(value: range):
    return type(value)


@functools.singledispatch
def freetypevars(typ) -> collections.abc.Set[typing.TypeVar | typing.ParamSpec]:
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
def _(typ: typing.TypeVar):
    return {typ}


@freetypevars.register
def _(typ: typing.ParamSpec):
    return {typ}


@freetypevars.register
def _(typ: typing.ParamSpecArgs):
    return freetypevars(typing.get_origin(typ))


@freetypevars.register
def _(typ: typing.ParamSpecKwargs):
    return freetypevars(typing.get_origin(typ))


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
def _(typ: collections.abc.Mapping):
    assert all(isinstance(k, str) for k in typ.keys()), "Mapping keys must be strings"
    return freetypevars(typ.values())


@freetypevars.register
def _(typ: GenericAlias):
    return freetypevars(typing.get_args(typ))


@freetypevars.register
def _(typ: types.UnionType):
    return freetypevars(typing.get_args(typ))


@functools.singledispatch
def substitute(
    typ, subs: Substitutions
) -> (
    type
    | types.GenericAlias
    | types.UnionType
    | None
    | typing.TypeVar
    | typing.ParamSpec
    | collections.abc.Sequence
    | collections.abc.Mapping
):
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
def _(typ: typing.TypeVar, subs: Substitutions):
    return substitute(subs[typ], subs) if typ in subs else typ


@substitute.register
def _(typ: typing.ParamSpec, subs: Substitutions):
    return substitute(subs[typ], subs) if typ in subs else typ


@substitute.register
def _(typ: typing.ParamSpecArgs, subs: Substitutions):
    res = substitute(typing.get_origin(typ), subs)
    return res.args if isinstance(res, typing.ParamSpec) else res


@substitute.register
def _(typ: typing.ParamSpecKwargs, subs: Substitutions):
    res = substitute(typing.get_origin(typ), subs)
    return res.kwargs if isinstance(res, typing.ParamSpec) else res


@substitute.register
def _(typ: list, subs: Substitutions):
    return list(substitute(item, subs) for item in typ)


@substitute.register
def _(typ: tuple, subs: Substitutions):
    return tuple(substitute(item, subs) for item in typ)


@substitute.register
def _(typ: collections.abc.Mapping, subs: Substitutions):
    assert all(isinstance(k, str) for k in typ.keys()), "Mapping keys must be strings"
    return {k: substitute(v, subs) for k, v in typ.items()}


@substitute.register
def _(typ: GenericAlias, subs: Substitutions):
    if typing.get_args(typ):
        return substitute(typing.get_origin(typ), subs)[
            substitute(typing.get_args(typ), subs)
        ]  # type: ignore
    else:
        return typ


@substitute.register
def _(typ: types.UnionType, subs: Substitutions):
    ts: tuple = substitute(typing.get_args(typ), subs)  # type: ignore
    tp, ts = ts[0], ts[1:]
    for arg in ts:
        tp = tp | arg
    return tp
