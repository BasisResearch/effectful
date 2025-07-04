import collections.abc
import functools
import inspect
import types
import typing


def infer_return_type(
    bound_sig: inspect.BoundArguments,
) -> type | types.GenericAlias | types.UnionType:
    """
    Infer the return type of a function based on its signature and argument types.

    This function takes a BoundArguments object (created by binding concrete argument
    types to a function signature) and infers what the return type should be by:
    1. Finding all TypeVars in the function's parameter and return annotations
    2. Unifying the parameter type annotations with the concrete argument types
    3. Applying the resulting TypeVar substitutions to the return type annotation

    The function ensures that all type variables in the return type can be inferred
    from the parameter types (no unbound type variables in the return).

    Args:
        bound_sig: A BoundArguments object obtained by calling
                   inspect.signature(func).bind(*arg_types, **kwarg_types)
                   where arg_types and kwarg_types are concrete types

    Returns:
        The inferred return type with all TypeVars substituted with concrete types

    Raises:
        TypeError: If the function lacks required type annotations, has unbound
                   type variables in the return type, or if unification fails
        NotImplementedError: If the function uses variadic parameters (*args, **kwargs),
                             collection types as parameters, or parameters with
                             free type variables

    Examples:
        >>> import inspect
        >>> import typing
        >>> T = typing.TypeVar('T')
        >>> K = typing.TypeVar('K')
        >>> V = typing.TypeVar('V')

        >>> # Simple generic function
        >>> def identity(x: T) -> T: ...
        >>> sig = inspect.signature(identity)
        >>> bound = sig.bind(int)
        >>> infer_return_type(bound)
        <class 'int'>

        >>> # Function with multiple TypeVars
        >>> def make_dict(key: K, value: V) -> dict[K, V]: ...
        >>> sig = inspect.signature(make_dict)
        >>> bound = sig.bind(str, int)
        >>> infer_return_type(bound)
        dict[str, int]

        >>> # Function with nested generics
        >>> def wrap_in_list(x: T) -> list[T]: ...
        >>> sig = inspect.signature(wrap_in_list)
        >>> bound = sig.bind(bool)
        >>> infer_return_type(bound)
        list[bool]

        >>> # Function with no TypeVars
        >>> def get_int() -> int: ...
        >>> sig = inspect.signature(get_int)
        >>> bound = sig.bind()
        >>> infer_return_type(bound)
        <class 'int'>

        >>> # Error: unbound type variable in return
        >>> def bad_func(x: T) -> tuple[T, K]: ...  # K not in parameters
        >>> sig = inspect.signature(bad_func)
        >>> bound = sig.bind(int)
        >>> infer_return_type(bound)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        TypeError: unbound type variables in return type
    """
    sig: inspect.Signature = bound_sig.signature

    # validate that the function has a signature with well-formed type annotations
    if sig.return_annotation is inspect.Signature.empty:
        raise TypeError("Function must have a return type annotation")

    result_fvs: set[typing.TypeVar] = freetypevars(sig.return_annotation)
    pattern_fvs: set[typing.TypeVar] = (
        set.union(*(freetypevars(p.annotation) for p in sig.parameters.values()))
        if sig.parameters
        else set()
    )
    concrete_fvs: set[typing.TypeVar] = (
        set.union(*(freetypevars(arg) for arg in bound_sig.arguments.values()))
        if bound_sig.arguments
        else set()
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
            raise TypeError(f"Parameter '{name}' cannot be variadic")

        if freetypevars(bound_sig.arguments[name]):
            raise TypeError(
                f"Parameter '{name}' cannot have free type variables"
            )

    # Build substitution map
    subs: collections.abc.Mapping[typing.TypeVar, type] = {}
    for name in sig.parameters:
        subs = unify(
            canonicalize(sig.parameters[name].annotation),
            canonicalize(bound_sig.arguments[name]),
            subs,
        )

    # Apply substitutions to return type
    result_type = substitute(canonicalize(sig.return_annotation), subs)
    if freetypevars(result_type) and not issubclass(
        typing.get_origin(result_type), collections.abc.Callable
    ):
        raise TypeError(
            "Return type cannot have free type variables after substitution"
        )
    return result_type


def unify(
    typ: type
    | typing.TypeVar
    | types.GenericAlias
    | types.UnionType
    | collections.abc.Mapping
    | collections.abc.Sequence,
    subtyp: type
    | typing.TypeVar
    | types.UnionType
    | types.GenericAlias
    | collections.abc.Mapping
    | collections.abc.Sequence,
    subs: collections.abc.Mapping[typing.TypeVar, type],
) -> collections.abc.Mapping[typing.TypeVar, type]:
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

        >>> # Multiple TypeVars
        >>> unify(dict[K, V], dict[str, int], {})
        {~K: <class 'str'>, ~V: <class 'int'>}

        >>> # With existing substitutions
        >>> unify(V, bool, {T: int})
        {~T: <class 'int'>, ~V: <class 'bool'>}

        >>> # Nested generic unification
        >>> unify(list[dict[K, V]], list[dict[str, int]], {})
        {~K: <class 'str'>, ~V: <class 'int'>}

        >>> # Exact type matching
        >>> unify(int, int, {})
        {}

        >>> # Failed unification - incompatible types
        >>> unify(list[T], dict[str, int], {})  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        TypeError: Cannot unify list[~T] with dict[str, int]

        >>> # Failed unification - conflicting TypeVar binding
        >>> unify(T, str, {T: int})  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        TypeError: Cannot unify ~T with <class 'str'> (already unified with <class 'int'>)

        >>> # Callable type unification
        >>> unify(collections.abc.Callable[[T], V], collections.abc.Callable[[int], str], {})
        {~T: <class 'int'>, ~V: <class 'str'>}

        >>> # Sequence unification (tuples as sequences)
        >>> unify((T, V), (int, str), {})
        {~T: <class 'int'>, ~V: <class 'str'>}
    """
    if typing.get_origin(typ) is typing.Annotated:
        # Handle Annotated types by extracting the base type
        return unify(typing.get_args(typ)[0], subtyp, subs)
    elif typing.get_origin(subtyp) is typing.Annotated:
        # Handle Annotated types by extracting the base type
        return unify(typ, typing.get_args(subtyp)[0], subs)
    elif isinstance(typ, typing.TypeVar):
        if typ in subs and subs[typ] != subtyp:
            raise TypeError(
                f"Cannot unify {typ} with {subtyp} (already unified with {subs[typ]})"
            )

        return {**subs, **{typ: subtyp}}
    elif typing.get_args(typ) and typing.get_args(subtyp):
        typ_origin = typing.get_origin(typ)
        subtyp_origin = typing.get_origin(subtyp)

        # Handle Union types - both typing.Union and types.UnionType are compatible
        if typ_origin in (typing.Union, types.UnionType) and subtyp_origin in (
            typing.Union,
            types.UnionType,
        ):
            return unify(typing.get_args(typ), typing.get_args(subtyp), subs)

        if typ_origin != subtyp_origin:
            raise TypeError(f"Cannot unify {typ} with {subtyp}")
        return unify(typing.get_args(typ), typing.get_args(subtyp), subs)
    elif isinstance(typ, collections.abc.Mapping) and isinstance(
        subtyp, collections.abc.Mapping
    ):
        if typ.keys() != subtyp.keys():
            raise TypeError(f"Cannot unify {typ} with {subtyp}")
        for key in typ:
            subs = unify(typ[key], subtyp[key], subs)
        return subs
    elif isinstance(typ, collections.abc.Sequence) and isinstance(
        subtyp, collections.abc.Sequence
    ):
        if len(typ) != len(subtyp):
            raise TypeError(f"Cannot unify {typ} with {subtyp}")
        for p_item, c_item in zip(typ, subtyp):
            subs = unify(p_item, c_item, subs)
        return subs
    else:
        subtyp = typing.get_origin(subtyp) or subtyp
        typ = typing.get_origin(typ) or typ
        if not issubclass(subtyp, typ):
            raise TypeError(f"Cannot unify {typ} with {subtyp}")
        return subs


def canonicalize(
    typ: type | typing.TypeVar | types.GenericAlias | types.UnionType,
) -> type:
    """
    Return a canonical form of the given type expression.

    This function normalizes the type by removing Annotated wrappers and
    ensuring that generic types are represented in their canonical form.
    It does not modify TypeVars or Union types, but ensures that generic
    aliases are returned in a consistent format.

    Args:
        typ: The type expression to canonicalize.

    Returns:
        A canonicalized version of the input type expression.

    Examples:
        >>> T = typing.TypeVar('T')
        >>> canonicalize(typing.List[T])
        list[~T]
        >>> canonicalize(typing.Annotated[int, "example"])
        <class 'int'>
    """
    if typing.get_origin(typ) is typing.Annotated:
        return canonicalize(typing.get_args(typ)[0])
    elif typ is inspect.Parameter.empty:
        return canonicalize(typing.Any)
    elif typing.get_origin(typ) in {typing.Union, types.UnionType}:
        t = canonicalize(typing.get_args(typ)[0])
        for arg in typing.get_args(typ)[1:]:
            t = t | canonicalize(arg)
        return t
    elif isinstance(typ, typing.TypeVar):
        return typ
    elif isinstance(typ, typing._GenericAlias | types.GenericAlias) and typing.get_origin(typ) is not typ:  # type: ignore
        # Handle generic types
        origin = canonicalize(typing.get_origin(typ))
        assert origin is not None, "Type must have an origin"
        return origin[tuple(canonicalize(a) for a in typing.get_args(typ))]
    # Handle legacy typing aliases like typing.Callable
    elif typ is typing.Callable:
        return canonicalize(collections.abc.Callable)
    elif typ is typing.Any:
        return canonicalize(object)
    elif typ is typing.List:
        return canonicalize(list)
    elif typ is typing.Dict:
        return canonicalize(dict)
    elif typ is typing.Set:
        return canonicalize(set)
    elif not isinstance(typ, type) and typing.get_origin(typ) is None:
        return canonicalize(_nested_type(typ))
    else:
        return typ


@functools.singledispatch
def _nested_type(value) -> type:
    from effectful.ops.types import Term

    if isinstance(value, Term):
        raise TypeError(f"Terms should not appear in _nested_type, but got {value}")
    elif not isinstance(value, type) and typing.get_origin(value) is None:
        return type(value)
    else:
        return value


@_nested_type.register
def _(value: type | types.UnionType | types.GenericAlias | types.EllipsisType | types.NoneType) -> type:
    return value


@_nested_type.register
def _(value: typing.TypeVar) -> type:
    raise TypeError(f"TypeVars should not appear in values, but got {value}")


@_nested_type.register
def _(value: collections.abc.Callable) -> type:
    return type(value)


@_nested_type.register
def _(value: collections.abc.Mapping) -> type:
    from effectful.ops.types import Interpretation

    if isinstance(value, Interpretation):  # type: ignore
        return Interpretation
    elif len(value) == 0:
        return type(value)
    else:
        k, v = next(iter(value.items()))
        return collections.abc.Mapping[_nested_type(k), _nested_type(v)]


@_nested_type.register
def _(value: collections.abc.Set) -> type:
    if len(value) == 0:
        return type(value)
    return collections.abc.Set[_nested_type(next(iter(value)))]


@_nested_type.register
def _(value: collections.abc.Sequence) -> type:
    if len(value) == 0:
        return type(value)
    return collections.abc.Sequence[_nested_type(next(iter(value)))]


@_nested_type.register
def _(value: tuple) -> type:
    return tuple[tuple(_nested_type(item) for item in value)]


@_nested_type.register
def _(value: str | bytes) -> type:
    # Handle str and bytes as their own types, not collections.abc.Sequence
    return type(value)


def freetypevars(
    typ: type | typing.TypeVar | types.GenericAlias | types.UnionType,
) -> set[typing.TypeVar]:
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
    elif typing.get_origin(typ) is typing.Annotated:
        return freetypevars(typing.get_args(typ)[0])
    elif isinstance(typ, list | tuple):
        # Handle plain lists and tuples (not generic aliases)
        return set.union(*(freetypevars(item) for item in typ)) if typ else set()
    elif typing.get_args(typ):
        # Handle generic aliases
        return set.union(*(freetypevars(arg) for arg in typing.get_args(typ)))
    else:
        return set()


def substitute(
    typ: type | types.GenericAlias | types.UnionType,
    subs: collections.abc.Mapping[typing.TypeVar, type],
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
    elif isinstance(typ, list | tuple):
        # Handle plain lists/sequences (e.g., in Callable's parameter list)
        return type(typ)(substitute(item, subs) for item in typ)
    elif typing.get_args(typ):
        origin = typing.get_origin(typ)
        assert origin is not None, "Type must have an origin"
        new_args = tuple(substitute(arg, subs) for arg in typing.get_args(typ))
        # Handle Union types specially
        if origin is types.UnionType:
            return typing.Union[new_args]  # noqa
        return origin[new_args]
    else:
        return typ
