import collections.abc
import inspect
import types
import typing


def infer_return_type(bound_sig: inspect.BoundArguments) -> type | types.GenericAlias | types.UnionType:
    """
    Infer the return type of a function based on its signature and argument types.
    """
    bound_sig.apply_defaults()
    sig: inspect.Signature = bound_sig.signature

    # validate that the function has a signature with well-formed type annotations
    if sig.return_annotation is inspect.Signature.empty:
        raise TypeError("Function must have a return type annotation")

    if any(p.annotation is inspect.Signature.empty for p in sig.parameters.values()):
        raise TypeError("All parameters must have type annotations")

    result_fvs: set[typing.TypeVar] = freetypevars(sig.return_annotation)
    pattern_fvs: set[typing.TypeVar] = set.union(
        *(freetypevars(p.annotation) for p in sig.parameters.values()),
    )
    concrete_fvs: set[typing.TypeVar] = set.union(
        *(freetypevars(arg) for arg in bound_sig.arguments.values()),
    )
    if (result_fvs | pattern_fvs) & concrete_fvs:
        raise TypeError(
            "Cannot unify free type variables in pattern and concrete types"
        )
    if not result_fvs <= pattern_fvs:
        raise TypeError("unbound type variables in return type")

    # Check for variadic parameters and collections - not implemented yet
    for name, param in sig.parameters.items():
        if param.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise NotImplementedError(f"Parameter '{name}' cannot be variadic")

        if isinstance(bound_sig.arguments[name], collections.abc.Collection):
            raise NotImplementedError(f"Parameter '{name}' cannot be a collection type")

        if freetypevars(bound_sig.arguments[name]):
            raise NotImplementedError(
                f"Parameter '{name}' cannot have free type variables"
            )

    # Build substitution map
    subs: collections.abc.Mapping[typing.TypeVar, type] = {}
    for name in sig.parameters:
        subs = unify(sig.parameters[name].annotation, bound_sig.arguments[name], subs)

    # Apply substitutions to return type
    result_type = substitute(sig.return_annotation, subs)
    if freetypevars(result_type):
        raise TypeError(
            "Return type cannot have free type variables after substitution"
        )
    return result_type


def unify(
    pattern: type
    | typing.TypeVar
    | types.GenericAlias
    | types.UnionType
    | collections.abc.Mapping
    | collections.abc.Sequence,
    concrete: type
    | typing.TypeVar
    | types.UnionType
    | types.GenericAlias
    | collections.abc.Mapping
    | collections.abc.Sequence,
    subs: collections.abc.Mapping[typing.TypeVar, type],
) -> collections.abc.Mapping[typing.TypeVar, type]:
    """
    Unify a pattern type with a concrete type, returning a substitution map.
    Raises TypeError if unification is not possible.
    """
    if isinstance(pattern, typing.TypeVar):
        if pattern in subs and subs[pattern] != concrete:
            raise TypeError(
                f"Cannot unify {pattern} with {concrete} (already unified with {subs[pattern]})"
            )
        return {**subs, **{pattern: concrete}}
    elif typing.get_args(pattern) and typing.get_args(concrete):
        if typing.get_origin(pattern) != typing.get_origin(concrete):
            raise TypeError(f"Cannot unify {pattern} with {concrete}")
        return unify(typing.get_args(pattern), typing.get_args(concrete), subs)
    elif isinstance(pattern, collections.abc.Mapping) and isinstance(
        concrete, collections.abc.Mapping
    ):
        if pattern.keys() != concrete.keys():
            raise TypeError(f"Cannot unify {pattern} with {concrete}")
        for key in pattern:
            subs = unify(pattern[key], concrete[key], subs)
        return subs
    elif isinstance(pattern, collections.abc.Sequence) and isinstance(
        concrete, collections.abc.Sequence
    ):
        if len(pattern) != len(concrete):
            raise TypeError(f"Cannot unify {pattern} with {concrete}")
        for p_item, c_item in zip(pattern, concrete):
            subs = unify(p_item, c_item, subs)
        return subs
    else:
        if pattern != concrete:
            raise TypeError(f"Cannot unify {pattern} with {concrete}")
        return subs


def freetypevars(typ: type | typing.TypeVar | types.GenericAlias | types.UnionType) -> set[typing.TypeVar]:
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
    if isinstance(typ, typing.TypeVar):
        return {typ}
    elif typing.get_args(typ):
        return set.union(*(freetypevars(arg) for arg in typing.get_args(typ)))
    else:
        return set()


def substitute(
    typ: type | types.GenericAlias | types.UnionType, subs: collections.abc.Mapping[typing.TypeVar, type]
) -> type | types.GenericAlias | types.UnionType:
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
    if isinstance(typ, typing.TypeVar):
        return subs.get(typ, typ)
    elif typing.get_args(typ):
        origin = typing.get_origin(typ)
        assert origin is not None, "Type must have an origin"
        new_args = tuple(substitute(arg, subs) for arg in typing.get_args(typ))
        # Handle Union types specially
        if origin is types.UnionType:
            return typing.Union[new_args]
        return origin[new_args]
    else:
        return typ
