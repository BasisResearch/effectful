import collections.abc
import inspect
import types
import typing


def infer_return_type(bound_sig: inspect.BoundArguments) -> type | types.GenericAlias:
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
    | types.GenericAlias
    | collections.abc.Mapping
    | collections.abc.Sequence,
    concrete: type
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
        return {**subs, pattern: concrete}
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


def freetypevars(typ: type | types.GenericAlias) -> set[typing.TypeVar]:
    """
    Return a set of free type variables in the given type.
    """
    if isinstance(typ, typing.TypeVar):
        return {typ}
    elif typing.get_args(typ):
        return set.union(*(freetypevars(arg) for arg in typing.get_args(typ)))
    else:
        return set()


def substitute(
    typ: type | types.GenericAlias, subs: collections.abc.Mapping[typing.TypeVar, type]
) -> type | types.GenericAlias:
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
        >>> substitute(typing.List[T], {T: str})
        typing.List[str]

        >>> # Nested generic substitution
        >>> substitute(typing.Dict[K, typing.List[V]], {K: str, V: int})
        typing.Dict[str, typing.List[int]]

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
        return origin[new_args]
    else:
        return typ
