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
                   type variables in the return type, if unification fails,
                   if the function uses variadic parameters (*args, **kwargs),
                   or if parameters have free type variables.

    Examples:
        >>> import inspect
        >>> import typing
        >>> from effectful.internals.unification import infer_return_type
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

    # Check for type variables in concrete arguments - not implemented yet
    for name, param in sig.parameters.items():
        if freetypevars(bound_sig.arguments.get(name, None)):
            raise TypeError(
                f"Parameter '{name}' cannot have free type variables"
            )

    # Build substitution map
    subs: collections.abc.Mapping[typing.TypeVar, type] = {}
    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            for arg in bound_sig.arguments[name]:
                subs = unify(
                    canonicalize(param.annotation),
                    canonicalize(arg),
                    subs,
                )
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            for arg in bound_sig.arguments[name].values():
                subs = unify(
                    canonicalize(param.annotation),
                    canonicalize(arg),
                    subs,
                )
        else:
            subs = unify(
                canonicalize(param.annotation),
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
    if isinstance(typ, typing.TypeVar):
        if typ in subs and subs[typ] != subtyp:
            raise TypeError(
                f"Cannot unify {typ} with {subtyp} (already unified with {subs[typ]})"
            )

        return {**subs, **{typ: subtyp}}
    elif isinstance(typ, types.UnionType) or isinstance(subtyp, types.UnionType):
        # TODO handle UnionType properly
        return unify(typing.get_args(typ), typing.get_args(subtyp), subs)
    elif typing.get_args(typ) and typing.get_args(subtyp):
        subs = unify(typing.get_origin(typ), typing.get_origin(subtyp), subs)
        return unify(typing.get_args(typ), typing.get_args(subtyp), subs)
    elif isinstance(typ, list | tuple) and isinstance(subtyp, list | tuple):
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

    This function normalizes type expressions by:
    - Removing Annotated wrappers to get the base type
    - Converting legacy typing module aliases (e.g., typing.List) to modern forms (e.g., list)
    - Preserving TypeVars unchanged
    - Recursively canonicalizing type arguments in generic types
    - Converting typing.Any to object
    - Converting inspect.Parameter.empty to typing.Any (then to object)
    - Handling Union types by creating canonical unions with | operator
    - Converting non-type values to their types using _nested_type

    Args:
        typ: The type expression to canonicalize. Can be a plain type, TypeVar,
             generic alias, union type, or even a value that needs type inference.

    Returns:
        A canonicalized version of the input type expression with consistent
        representation and modern syntax.

    Examples:
        >>> import typing
        >>> import inspect
        >>> import collections.abc
        >>> from effectful.internals.unification import canonicalize
        >>> T = typing.TypeVar('T')
        >>> K = typing.TypeVar('K')
        >>> V = typing.TypeVar('V')

        # Legacy typing aliases are converted to modern forms
        >>> canonicalize(typing.List[int])
        list[int]
        >>> canonicalize(typing.Dict[str, int])
        dict[str, int]
        >>> canonicalize(typing.Set[bool])
        set[bool]
        >>> canonicalize(typing.Callable[[int], str])
        collections.abc.Callable[[int], str]

        # TypeVars are preserved unchanged
        >>> canonicalize(T)
        ~T
        >>> canonicalize(list[T])
        list[~T]

        # Annotated types are unwrapped
        >>> canonicalize(typing.Annotated[int, "metadata"])
        <class 'int'>
        >>> canonicalize(typing.Annotated[list[str], "doc string"])
        list[str]

        # Nested generic types are recursively canonicalized
        >>> canonicalize(typing.List[typing.Dict[K, V]])
        list[dict[~K, ~V]]
        >>> canonicalize(typing.Dict[str, typing.List[T]])
        dict[str, list[~T]]

        # Union types are canonicalized with | operator
        >>> result = canonicalize(typing.Union[int, str])
        >>> result == int | str
        True
        >>> result = canonicalize(typing.Union[list[T], dict[K, V]])
        >>> result == list[T] | dict[K, V]
        True

        # typing.Any becomes object
        >>> canonicalize(typing.Any)
        <class 'object'>

        # inspect.Parameter.empty becomes object (via Any)
        >>> canonicalize(inspect.Parameter.empty)
        <class 'object'>

        # Plain types pass through unchanged
        >>> canonicalize(int)
        <class 'int'>
        >>> canonicalize(str)
        <class 'str'>
        >>> canonicalize(list)
        <class 'list'>

        # Values are converted to their types via nested_type
        >>> canonicalize([1, 2, 3])
        collections.abc.Sequence[int]
        >>> canonicalize({"key": "value"})
        collections.abc.Mapping[str, str]
        >>> canonicalize((1, "hello", 3.14))
        tuple[int, str, float]

        # Complex nested canonicalization
        >>> canonicalize(typing.List[typing.Union[typing.Dict[str, T], None]])
        list[dict[str, ~T] | None]
    """
    if typing.get_origin(typ) is typing.Annotated:
        return canonicalize(typing.get_args(typ)[0])
    elif typ is inspect.Parameter.empty:
        return canonicalize(typing.Any)
    elif typ is None:
        return type(None)
    elif typing.get_origin(typ) in {typing.Union, types.UnionType}:
        t = canonicalize(typing.get_args(typ)[0])
        for arg in typing.get_args(typ)[1:]:
            t = t | canonicalize(arg)
        return t
    elif isinstance(typ, typing.TypeVar):
        return typ
    elif isinstance(typ, typing._GenericAlias | types.GenericAlias) and typing.get_origin(typ) is not typ:  # type: ignore
        # Handle generic types
        origin = typing.get_origin(typ)
        args = typing.get_args(typ)
        
        # Special handling for Callable types
        if origin is collections.abc.Callable and args:
            if len(args) == 2 and isinstance(args[0], (list, tuple)):
                # Callable[[arg1, arg2, ...], return_type] format
                param_list = [canonicalize(a) for a in args[0]]
                return_type = canonicalize(args[1])
                return collections.abc.Callable[[*param_list], return_type]
            else:
                # Handle other Callable formats
                return origin[tuple(canonicalize(a) for a in args)]
        else:
            # Regular generic types
            canonical_origin = canonicalize(origin)
            return canonical_origin[tuple(canonicalize(a) for a in args)]
    # Handle legacy typing aliases
    elif hasattr(typing, 'List') and typ is getattr(typing, 'List', None):
        return list
    elif hasattr(typing, 'Dict') and typ is getattr(typing, 'Dict', None):
        return dict
    elif hasattr(typing, 'Set') and typ is getattr(typing, 'Set', None):
        return set
    elif typ is typing.Callable:
        return collections.abc.Callable
    elif typ is typing.Any:
        return object
    elif not isinstance(typ, type) and typing.get_origin(typ) is None:
        return canonicalize(nested_type(typ))
    else:
        return typ


@functools.singledispatch
def nested_type(value) -> type:
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
        collections.abc.Sequence[int]
        >>> nested_type(["a", "b", "c"])
        collections.abc.Sequence[str]

        # Tuples preserve exact structure
        >>> nested_type((1, "hello", 3.14))
        tuple[int, str, float]
        >>> nested_type(())
        <class 'tuple'>
        >>> nested_type((1,))
        tuple[int]

        # Sets become Set[element_type]
        >>> nested_type({1, 2, 3})
        collections.abc.Set[int]
        >>> nested_type({"a", "b"})
        collections.abc.Set[str]

        # Mappings become Mapping[key_type, value_type]
        >>> nested_type({"key": "value"})
        collections.abc.Mapping[str, str]
        >>> nested_type({1: "one", 2: "two"})
        collections.abc.Mapping[int, str]

        # Nested collections work recursively
        >>> nested_type([{1: "one"}, {2: "two"}])
        collections.abc.Sequence[collections.abc.Mapping[int, str]]
        >>> nested_type({"key": [1, 2, 3]})
        collections.abc.Mapping[str, collections.abc.Sequence[int]]

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
    elif value is None:
        return type(None)
    elif not isinstance(value, type) and typing.get_origin(value) is None:
        return type(value)
    else:
        return value


@nested_type.register
def _(value: type | types.UnionType | types.GenericAlias | types.EllipsisType | types.NoneType) -> type:
    return value


@nested_type.register
def _(value: typing._GenericAlias) -> type:  # type: ignore
    # Handle typing module generic aliases
    return value


@nested_type.register
def _(value: types.NoneType) -> type:
    # Handle None specially
    return type(None)


@nested_type.register
def _(value: typing.TypeVar) -> type:
    raise TypeError(f"TypeVars should not appear in values, but got {value}")


@nested_type.register
def _(value: collections.abc.Callable) -> type:
    return type(value)


@nested_type.register
def _(value: collections.abc.Mapping) -> type:
    from effectful.ops.types import Interpretation

    if type(value) is Interpretation:  # More specific check
        return Interpretation
    elif len(value) == 0:
        return type(value)
    else:
        k, v = next(iter(value.items()))
        return collections.abc.Mapping[nested_type(k), nested_type(v)]


@nested_type.register
def _(value: collections.abc.Set) -> type:
    if len(value) == 0:
        return type(value)
    return collections.abc.Set[nested_type(next(iter(value)))]


@nested_type.register
def _(value: collections.abc.Sequence) -> type:
    if len(value) == 0:
        return type(value)
    return collections.abc.Sequence[nested_type(next(iter(value)))]


@nested_type.register
def _(value: tuple) -> type:
    if len(value) == 0:
        return tuple
    return tuple[tuple(nested_type(item) for item in value)]


@nested_type.register
def _(value: str | bytes) -> type:
    # Handle str and bytes as their own types, not collections.abc.Sequence
    return type(value)


@nested_type.register(range)
def _(value: range) -> type:
    # Handle range as its own type, not as a sequence
    return type(value)


def freetypevars(
    typ: type | typing.TypeVar | types.GenericAlias | types.UnionType | types.NoneType,
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
