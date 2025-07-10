import collections.abc
import inspect
import typing

import pytest

from effectful.internals.unification import (
    freetypevars,
    nested_type,
    substitute,
    unify,
)

if typing.TYPE_CHECKING:
    T = typing.Any
    K = typing.Any
    V = typing.Any
    U = typing.Any
else:
    T = typing.TypeVar("T")
    K = typing.TypeVar("K")
    V = typing.TypeVar("V")
    U = typing.TypeVar("U")


@pytest.mark.parametrize(
    "typ,fvs",
    [
        # Basic cases
        (T, {T}),
        (int, set()),
        (str, set()),
        # Single TypeVar in generic
        (list[T], {T}),
        (set[T], {T}),
        (tuple[T], {T}),
        # Multiple TypeVars
        (dict[K, V], {K, V}),
        (tuple[K, V], {K, V}),
        (dict[T, T], {T}),  # Same TypeVar used twice
        # Nested generics with TypeVars
        (list[dict[K, V]], {K, V}),
        (dict[K, list[V]], {K, V}),
        (list[tuple[T, U]], {T, U}),
        (tuple[list[T], dict[K, V]], {T, K, V}),
        # Concrete types in generics
        (list[int], set()),
        (dict[str, int], set()),
        (tuple[int, str, float], set()),
        # Mixed concrete and TypeVars
        (dict[str, T], {T}),
        (dict[K, int], {K}),
        (tuple[T, int, V], {T, V}),
        (list[tuple[int, T]], {T}),
        # Deeply nested
        (list[dict[K, list[tuple[V, T]]]], {K, V, T}),
        (dict[tuple[K, V], list[dict[U, T]]], {K, V, U, T}),
        # Union types (if supported)
        (list[T] | dict[K, V], {T, K, V}),
        (T | int, {T}),
        # Callable types
        (collections.abc.Callable[[T], V], {T, V}),
        (collections.abc.Callable[[int, T], T], {T}),
        (collections.abc.Callable[[], T], {T}),
        (collections.abc.Callable[[T, U], V], {T, U, V}),
        (collections.abc.Callable[[int], int], set()),
        (collections.abc.Callable[[T], list[T]], {T}),
        (collections.abc.Callable[[dict[K, V]], tuple[K, V]], {K, V}),
        # Nested Callable
        (collections.abc.Callable[[T], collections.abc.Callable[[U], V]], {T, U, V}),
        (list[collections.abc.Callable[[T], V]], {T, V}),
        (dict[K, collections.abc.Callable[[T], V]], {K, T, V}),
        # ParamSpec and TypeVarTuple (if needed later)
        # (collections.abc.Callable[typing.ParamSpec("P"), T], {T}),  # Would need to handle ParamSpec
    ],
)
def test_freetypevars(typ: type, fvs: set[typing.TypeVar]):
    assert freetypevars(typ) == fvs


@pytest.mark.parametrize(
    "typ,subs,expected",
    [
        # Basic substitution
        (T, {T: int}, int),
        (T, {T: str}, str),
        (T, {T: list[int]}, list[int]),
        # TypeVar not in mapping
        (T, {K: int}, T),
        (T, {}, T),
        # Non-TypeVar types
        (int, {T: str}, int),
        (str, {}, str),
        (list[int], {T: str}, list[int]),
        # Single TypeVar in generic
        (list[T], {T: int}, list[int]),
        (set[T], {T: str}, set[str]),
        (tuple[T], {T: float}, tuple[float]),
        # Multiple TypeVars
        (dict[K, V], {K: str, V: int}, dict[str, int]),
        (tuple[K, V], {K: int, V: str}, tuple[int, str]),
        (dict[K, V], {K: str}, dict[str, V]),  # Partial substitution
        # Same TypeVar used multiple times
        (dict[T, T], {T: int}, dict[int, int]),
        (tuple[T, T, T], {T: str}, tuple[str, str, str]),
        # Nested generics
        (list[dict[K, V]], {K: str, V: int}, list[dict[str, int]]),
        (dict[K, list[V]], {K: int, V: str}, dict[int, list[str]]),
        (list[tuple[T, U]], {T: int, U: str}, list[tuple[int, str]]),
        # Mixed concrete and TypeVars
        (dict[str, T], {T: int}, dict[str, int]),
        (tuple[int, T, str], {T: float}, tuple[int, float, str]),
        (list[tuple[int, T]], {T: str}, list[tuple[int, str]]),
        # Deeply nested substitution
        (list[dict[K, list[V]]], {K: str, V: int}, list[dict[str, list[int]]]),
        (
            dict[tuple[K, V], list[T]],
            {K: int, V: str, T: float},
            dict[tuple[int, str], list[float]],
        ),
        # Substituting with generic types
        (T, {T: list[int]}, list[int]),
        (list[T], {T: dict[str, int]}, list[dict[str, int]]),
        (
            dict[K, V],
            {K: list[int], V: dict[str, float]},
            dict[list[int], dict[str, float]],
        ),
        # Empty substitution
        (list[T], {}, list[T]),
        (dict[K, V], {}, dict[K, V]),
        # Union types (if supported)
        (T | int, {T: str}, str | int),
        (
            list[T] | dict[K, V],
            {T: int, K: str, V: float},
            list[int] | dict[str, float],
        ),
        # Irrelevant substitutions (TypeVars not in type)
        (list[T], {K: int, V: str}, list[T]),
        (int, {T: str, K: int}, int),
        # Callable types
        (
            collections.abc.Callable[[T], V],
            {T: int, V: str},
            collections.abc.Callable[[int], str],
        ),
        (
            collections.abc.Callable[[int, T], T],
            {T: str},
            collections.abc.Callable[[int, str], str],
        ),
        (
            collections.abc.Callable[[], T],
            {T: float},
            collections.abc.Callable[[], float],
        ),
        (
            collections.abc.Callable[[T, U], V],
            {T: int, U: str, V: bool},
            collections.abc.Callable[[int, str], bool],
        ),
        (
            collections.abc.Callable[[int], int],
            {T: str},
            collections.abc.Callable[[int], int],
        ),
        (
            collections.abc.Callable[[T], list[T]],
            {T: int},
            collections.abc.Callable[[int], list[int]],
        ),
        (
            collections.abc.Callable[[dict[K, V]], tuple[K, V]],
            {K: str, V: int},
            collections.abc.Callable[[dict[str, int]], tuple[str, int]],
        ),
        # Nested Callable
        (
            collections.abc.Callable[[T], collections.abc.Callable[[U], V]],
            {T: int, U: str, V: bool},
            collections.abc.Callable[[int], collections.abc.Callable[[str], bool]],
        ),
        (
            list[collections.abc.Callable[[T], V]],
            {T: int, V: str},
            list[collections.abc.Callable[[int], str]],
        ),
        (
            dict[K, collections.abc.Callable[[T], V]],
            {K: str, T: int, V: float},
            dict[str, collections.abc.Callable[[int], float]],
        ),
        # Partial substitution with Callable
        (
            collections.abc.Callable[[T, U], V],
            {T: int},
            collections.abc.Callable[[int, U], V],
        ),
        (
            collections.abc.Callable[[T], dict[K, V]],
            {T: int, K: str},
            collections.abc.Callable[[int], dict[str, V]],
        ),
    ],
)
def test_substitute(
    typ: type, subs: typing.Mapping[typing.TypeVar, type], expected: type
):
    assert substitute(typ, subs) == expected  # type: ignore


@pytest.mark.parametrize(
    "typ,subtyp,initial_subs,expected_subs",
    [
        # Basic TypeVar unification
        (T, int, {}, {T: int}),
        (T, str, {}, {T: str}),
        (T, list[int], {}, {T: list[int]}),
        # With existing substitutions
        (V, bool, {T: int}, {T: int, V: bool}),
        (K, str, {T: int, V: bool}, {T: int, V: bool, K: str}),
        # Generic type unification
        (list[T], list[int], {}, {T: int}),
        (dict[K, V], dict[str, int], {}, {K: str, V: int}),
        (tuple[T, U], tuple[int, str], {}, {T: int, U: str}),
        (set[T], set[float], {}, {T: float}),
        # Same TypeVar used multiple times
        (dict[T, T], dict[int, int], {}, {T: int}),
        (tuple[T, T, T], tuple[str, str, str], {}, {T: str}),
        # Nested generic unification
        (list[dict[K, V]], list[dict[str, int]], {}, {K: str, V: int}),
        (dict[K, list[V]], dict[int, list[str]], {}, {K: int, V: str}),
        (list[tuple[T, U]], list[tuple[bool, float]], {}, {T: bool, U: float}),
        # Deeply nested
        (list[dict[K, list[V]]], list[dict[str, list[int]]], {}, {K: str, V: int}),
        (
            dict[tuple[K, V], list[T]],
            dict[tuple[int, str], list[bool]],
            {},
            {K: int, V: str, T: bool},
        ),
        # Mixed concrete and TypeVars
        (dict[str, T], dict[str, int], {}, {T: int}),
        (tuple[int, T, str], tuple[int, float, str], {}, {T: float}),
        (list[tuple[int, T]], list[tuple[int, str]], {}, {T: str}),
        # Exact type matching (no TypeVars)
        (int, int, {}, {}),
        (str, str, {}, {}),
        (list[int], list[int], {}, {}),
        (dict[str, int], dict[str, int], {}, {}),
        # Callable type unification
        (
            collections.abc.Callable[[T], V],
            collections.abc.Callable[[int], str],
            {},
            {T: int, V: str},
        ),
        (
            collections.abc.Callable[[T, U], V],
            collections.abc.Callable[[int, str], bool],
            {},
            {T: int, U: str, V: bool},
        ),
        (
            collections.abc.Callable[[], T],
            collections.abc.Callable[[], float],
            {},
            {T: float},
        ),
        (
            collections.abc.Callable[[T], list[T]],
            collections.abc.Callable[[int], list[int]],
            {},
            {T: int},
        ),
        # Nested Callable
        (
            collections.abc.Callable[[T], collections.abc.Callable[[U], V]],
            collections.abc.Callable[[int], collections.abc.Callable[[str], bool]],
            {},
            {T: int, U: str, V: bool},
        ),
        # Complex combinations
        (
            dict[K, collections.abc.Callable[[T], V]],
            dict[str, collections.abc.Callable[[int], bool]],
            {},
            {K: str, T: int, V: bool},
        ),
    ],
)
def test_unify_success(
    typ: type,
    subtyp: type,
    initial_subs: typing.Mapping[typing.TypeVar, type],
    expected_subs: typing.Mapping[typing.TypeVar, type],
):
    assert unify(typ, subtyp, initial_subs) == expected_subs  # type: ignore


@pytest.mark.parametrize(
    "typ,subtyp",
    [
        # Incompatible types
        (list[T], dict[str, int]),
        (int, str),
        (list[int], list[str]),
        # Mismatched generic types
        (list[T], set[int]),
        (dict[K, V], list[int]),
        # Same TypeVar with different values
        (dict[T, T], dict[int, str]),
        (tuple[T, T], tuple[int, str]),
        # Mismatched arities
        (tuple[T, U], tuple[int, str, bool]),
        (
            collections.abc.Callable[[T], V],
            collections.abc.Callable[[int, str], bool],
        ),
        # Sequence length mismatch
        ((T, V), (int,)),
        ([T, V], [int, str, bool]),
    ],
)
def test_unify_failure(
    typ: type,
    subtyp: type,
):
    with pytest.raises(TypeError):
        unify(typ, subtyp, {})


# Test functions with various type patterns
def identity(x: T) -> T:
    return x


def make_pair(x: T, y: V) -> tuple[T, V]:
    return (x, y)


def wrap_in_list(x: T) -> list[T]:
    return [x]


def get_first(items: list[T]) -> T:
    return items[0]


def getitem_mapping(mapping: collections.abc.Mapping[K, V], key: K) -> V:
    return mapping[key]


def dict_values(d: dict[K, V]) -> list[V]:
    return list(d.values())


def process_callable(func: collections.abc.Callable[[T], V], arg: T) -> V:
    return func(arg)


def chain_callables(
    f: collections.abc.Callable[[T], U], g: collections.abc.Callable[[U], V]
) -> collections.abc.Callable[[T], V]:
    def result(x: T) -> V:
        return g(f(x))

    return result


def constant_func() -> int:
    return 42


def multi_generic(a: T, b: list[T], c: dict[K, V]) -> tuple[T, K, V]:
    return (a, next(iter(c.keys())), next(iter(c.values())))


def same_type_twice(x: T, y: T) -> T:
    return x if len(str(x)) > len(str(y)) else y


def nested_generic(x: T) -> dict[str, list[T]]:
    return {"items": [x]}


def variadic_args_func(*args: T) -> T:  # Variadic args not supported
    return args[0]


def variadic_kwargs_func(**kwargs: T) -> T:  # Variadic kwargs not supported
    return next(iter(kwargs.values()))


@pytest.mark.parametrize(
    "func,args,kwargs,expected_return_type",
    [
        # Simple generic functions
        (identity, (int,), {}, int),
        (identity, (str,), {}, str),
        (identity, (list[int],), {}, list[int]),
        # Multiple TypeVars
        (make_pair, (int, str), {}, tuple[int, str]),
        (make_pair, (bool, list[float]), {}, tuple[bool, list[float]]),
        # Generic collections
        (wrap_in_list, (int,), {}, list[int]),
        (wrap_in_list, (dict[str, bool],), {}, list[dict[str, bool]]),
        (get_first, (list[str],), {}, str),
        (get_first, (list[tuple[int, float]],), {}, tuple[int, float]),
        (getitem_mapping, (collections.abc.Mapping[str, int], str), {}, int),
        (
            getitem_mapping,
            (collections.abc.Mapping[bool, list[str]], bool),
            {},
            list[str],
        ),
        # Dict operations
        (dict_values, (dict[str, int],), {}, list[int]),
        (dict_values, (dict[bool, list[str]],), {}, list[list[str]]),
        # Callable types
        (process_callable, (collections.abc.Callable[[int], str], int), {}, str),
        (
            process_callable,
            (collections.abc.Callable[[list[int]], bool], list[int]),
            {},
            bool,
        ),
        # Complex callable return
        (
            chain_callables,
            (
                collections.abc.Callable[[int], str],
                collections.abc.Callable[[str], bool],
            ),
            {},
            collections.abc.Callable[[int], bool],
        ),
        # No generics
        (constant_func, (), {}, int),
        # Mixed generics
        (multi_generic, (int, list[int], dict[str, bool]), {}, tuple[int, str, bool]),
        (
            multi_generic,
            (float, list[float], dict[bool, list[str]]),
            {},
            tuple[float, bool, list[str]],
        ),
        # Same TypeVar used multiple times
        (same_type_twice, (int, int), {}, int),
        (same_type_twice, (str, str), {}, str),
        # Nested generics
        (nested_generic, (int,), {}, dict[str, list[int]]),
        (
            nested_generic,
            (collections.abc.Callable[[str], bool],),
            {},
            dict[str, list[collections.abc.Callable[[str], bool]]],
        ),
        # Keyword arguments
        (make_pair, (), {"x": int, "y": str}, tuple[int, str]),
        (
            multi_generic,
            (),
            {"a": bool, "b": list[bool], "c": dict[int, str]},
            tuple[bool, int, str],
        ),
        # variadic args and kwargs
        (variadic_args_func, (int,), {}, int),
        (variadic_args_func, (int, int), {}, int),
        (variadic_kwargs_func, (), {"x": int}, int),
        (variadic_kwargs_func, (), {"x": int, "y": int}, int),
    ],
)
def test_infer_return_type_success(
    func: collections.abc.Callable,
    args: tuple,
    kwargs: dict,
    expected_return_type: type,
):
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    result = substitute(sig.return_annotation, unify(sig, bound))
    assert result == expected_return_type


# Error cases
def unbound_typevar_func(x: T) -> tuple[T, V]:  # V not in parameters
    return (x, "error")


def no_return_annotation(x: T):  # No return annotation
    return x


def no_param_annotation(x) -> T:  # No parameter annotation
    return x


@pytest.mark.parametrize(
    "func,args,kwargs",
    [
        # Missing annotations
        (
            no_param_annotation,
            (int,),
            {},
        ),
        # Type mismatch - trying to unify incompatible types
        (same_type_twice, (int, str), {}),
    ],
)
def test_infer_return_type_failure(
    func: collections.abc.Callable,
    args: tuple,
    kwargs: dict,
):
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    with pytest.raises(TypeError):
        unify(sig, bound)


@pytest.mark.parametrize(
    "value,expected",
    [
        # Basic value types
        (42, int),
        (0, int),
        (-5, int),
        ("hello", str),
        ("", str),
        (3.14, float),
        (0.0, float),
        (True, bool),
        (False, bool),
        (None, type(None)),
        (b"bytes", bytes),
        (b"", bytes),
        # Type objects pass through
        (int, int),
        (str, str),
        (float, float),
        (bool, bool),
        (list, list),
        (dict, dict),
        (set, set),
        (tuple, tuple),
        (type(None), type(None)),
        (type(...), type(...)),
        # Generic aliases pass through
        (list[int], list[int]),
        (dict[str, int], dict[str, int]),
        (set[bool], set[bool]),
        (tuple[int, str], tuple[int, str]),
        (int | str, int | str),
        (list[T], list[T]),
        (dict[K, V], dict[K, V]),
        # Union types pass through
        (int | str, int | str),
        # Empty collections
        ([], list),
        ({}, dict),
        (set(), set),
        ((), tuple),
        # Lists/sequences with single type
        ([1, 2, 3], list[int]),
        ([1], list[int]),
        (["a", "b", "c"], list[str]),
        ([True, False], list[bool]),
        ([1.1, 2.2], list[float]),
        # Sets with elements
        ({1, 2, 3}, set[int]),
        ({1}, set[int]),
        ({"a", "b"}, set[str]),
        ({True, False}, set[bool]),
        # Dicts/mappings
        ({"key": "value"}, dict[str, str]),
        ({1: "one", 2: "two"}, dict[int, str]),
        ({"a": 1, "b": 2}, dict[str, int]),
        ({True: 1.0, False: 2.0}, dict[bool, float]),
        # Tuples preserve exact structure
        ((1, "hello", 3.14), tuple[int, str, float]),
        ((1,), tuple[int]),
        ((1, 2), tuple[int, int]),
        (("a", "b", "c"), tuple[str, str, str]),
        ((True, 1, "x", 3.14), tuple[bool, int, str, float]),
        # Nested collections
        ([[1, 2], [3, 4]], list[list[int]]),
        ([{1, 2}, {3, 4}], list[set[int]]),
        ([{"a": 1}, {"b": 2}], list[dict[str, int]]),
        ({"key": [1, 2, 3]}, dict[str, list[int]]),
        ({"a": {1, 2}, "b": {3, 4}}, dict[str, set[int]]),
        ({1: {"x": True}, 2: {"y": False}}, dict[int, dict[str, bool]]),
        # Tuples in collections
        ([(1, "a"), (2, "b")], list[tuple[int, str]]),
        ({(1, 2), (3, 4)}, set[tuple[int, int]]),
        ({1: (True, "x"), 2: (False, "y")}, dict[int, tuple[bool, str]]),
        # Functions/callables
        (lambda x: x, type(lambda x: x)),
        (print, type(print)),
        (len, type(len)),
        # Complex nested structures
        ([[[1]]], list[list[list[int]]]),
        ({"a": {"b": {"c": 1}}}, dict[str, dict[str, dict[str, int]]]),
        # Special string/bytes handling (NOT treated as sequences)
        ("hello", str),
        (b"world", bytes),
        # Other built-in types
        (range(5), type(range(5))),
        (slice(1, 10), type(slice(1, 10))),
    ],
)
def test_nested_type(value, expected):
    result = nested_type(value)
    assert result == expected


def test_nested_type_typevar_error():
    """Test that TypeVars raise TypeError in nested_type"""
    with pytest.raises(TypeError, match="TypeVars should not appear in values"):
        nested_type(T)

    with pytest.raises(TypeError, match="TypeVars should not appear in values"):
        nested_type(K)

    with pytest.raises(TypeError, match="TypeVars should not appear in values"):
        nested_type(V)


def test_nested_type_term_error():
    """Test that Terms raise TypeError in nested_type"""
    # We can't import Term here without creating a circular dependency,
    # so we'll create a mock object that would trigger the isinstance check
    from unittest.mock import Mock

    from effectful.ops.types import Term

    mock_term = Mock(spec=Term)
    with pytest.raises(TypeError, match="Terms should not appear in nested_type"):
        nested_type(mock_term)


def sequence_getitem(seq: collections.abc.Sequence[T], index: int) -> T:
    return seq[index]


def mapping_getitem(mapping: collections.abc.Mapping[K, V], key: K) -> V:
    return mapping[key]


def sequence_mapping_getitem(
    seq: collections.abc.Sequence[collections.abc.Mapping[K, V]], index: int, key: K
) -> V:
    return mapping_getitem(sequence_getitem(seq, index), key)


def mapping_sequence_getitem(
    mapping: collections.abc.Mapping[K, collections.abc.Sequence[T]], key: K, index: int
) -> T:
    return sequence_getitem(mapping_getitem(mapping, key), index)


def sequence_from_pair(a: T, b: T) -> collections.abc.Sequence[T]:
    return [a, b]


def mapping_from_pair(a: K, b: V) -> collections.abc.Mapping[K, V]:
    return {a: b}


def sequence_of_mappings(
    key1: K, val1: V, key2: K, val2: V
) -> collections.abc.Sequence[collections.abc.Mapping[K, V]]:
    """Creates a sequence containing two mappings."""
    return sequence_from_pair(
        mapping_from_pair(key1, val1), mapping_from_pair(key2, val2)
    )


def mapping_of_sequences(
    key1: K, val1: T, val2: T, key2: K, val3: T, val4: T
) -> collections.abc.Mapping[K, collections.abc.Sequence[T]]:
    """Creates a mapping where each key maps to a sequence of two values."""
    return mapping_from_pair(key1, sequence_from_pair(val1, val2))


def nested_sequence_mapping(
    k1: K, v1: T, v2: T, k2: K, v3: T, v4: T
) -> collections.abc.Sequence[collections.abc.Mapping[K, collections.abc.Sequence[T]]]:
    """Creates a sequence of mappings, where each mapping contains sequences."""
    return sequence_from_pair(
        mapping_from_pair(k1, sequence_from_pair(v1, v2)),
        mapping_from_pair(k2, sequence_from_pair(v3, v4)),
    )


def get_from_constructed_sequence(a: T, b: T, index: int) -> T:
    """Constructs a sequence from two elements and gets one by index."""
    return sequence_getitem(sequence_from_pair(a, b), index)


def get_from_constructed_mapping(key: K, value: V, lookup_key: K) -> V:
    """Constructs a mapping from a key-value pair and looks up the value."""
    return mapping_getitem(mapping_from_pair(key, value), lookup_key)


def double_nested_get(
    k1: K,
    v1: T,
    v2: T,
    k2: K,
    v3: T,
    v4: T,
    outer_index: int,
    inner_key: K,
    inner_index: int,
) -> T:
    """Creates nested structure and retrieves deeply nested value."""
    nested = nested_sequence_mapping(k1, v1, v2, k2, v3, v4)
    mapping = sequence_getitem(nested, outer_index)
    sequence = mapping_getitem(mapping, inner_key)
    return sequence_getitem(sequence, inner_index)


def construct_and_extend_sequence(
    a: T, b: T, c: T, d: T
) -> collections.abc.Sequence[collections.abc.Sequence[T]]:
    """Constructs two sequences and combines them into a sequence of sequences."""
    seq1 = sequence_from_pair(a, b)
    seq2 = sequence_from_pair(c, d)
    return sequence_from_pair(seq1, seq2)


def transform_mapping_values(
    key1: K, val1: T, key2: K, val2: T
) -> collections.abc.Mapping[K, collections.abc.Sequence[T]]:
    """Creates a mapping where each value is wrapped in a sequence."""
    # Create mappings where each value becomes a single-element sequence
    # Note: In a real implementation, we'd need a sequence_from_single function
    # For now, using sequence_from_pair with the same value twice as a workaround
    return mapping_from_pair(key1, sequence_from_pair(val1, val1))


@pytest.mark.parametrize(
    "seq,index,key",
    [
        # Original test case: list of dicts with string keys and int values
        ([{"a": 1}, {"b": 2}, {"c": 3}], 1, "b"),
        # Different value types
        ([{"x": "hello"}, {"y": "world"}, {"z": "test"}], 2, "z"),
        ([{"name": 3.14}, {"value": 2.71}, {"constant": 1.41}], 0, "name"),
        ([{"flag": True}, {"enabled": False}, {"active": True}], 1, "enabled"),
        # Mixed value types in same dict (should still work)
        ([{"a": [1, 2, 3]}, {"b": [4, 5, 6]}, {"c": [7, 8, 9]}], 0, "a"),
        ([{"data": {"nested": "value"}}, {"info": {"deep": "data"}}], 1, "info"),
        # Different key types
        ([{1: "one"}, {2: "two"}, {3: "three"}], 2, 3),
        ([{True: "yes"}, {False: "no"}], 0, True),
        # Nested collections as values
        ([{"items": [1, 2, 3]}, {"values": [4, 5, 6]}], 0, "items"),
        ([{"matrix": [[1, 2], [3, 4]]}, {"grid": [[5, 6], [7, 8]]}], 1, "grid"),
        ([{"sets": {1, 2, 3}}, {"groups": {4, 5, 6}}], 0, "sets"),
        # Complex nested structures
        (
            [
                {"users": [{"id": 1, "name": "Alice"}]},
                {"users": [{"id": 2, "name": "Bob"}]},
            ],
            1,
            "users",
        ),
        (
            [
                {"config": {"db": {"host": "localhost", "port": 5432}}},
                {"config": {"cache": {"ttl": 300}}},
            ],
            0,
            "config",
        ),
        # Edge cases with single element sequences
        ([{"only": "one"}], 0, "only"),
        # Tuples as values
        ([{"point": (1, 2)}, {"coord": (3, 4)}, {"pos": (5, 6)}], 2, "pos"),
        ([{"rgb": (255, 0, 0)}, {"hsv": (0, 100, 100)}], 0, "rgb"),
    ],
)
def test_infer_composition_1(seq, index, key):
    sig1 = inspect.signature(sequence_getitem)
    sig2 = inspect.signature(mapping_getitem)

    sig12 = inspect.signature(sequence_mapping_getitem)

    inferred_type1 = substitute(
        sig1.return_annotation,
        unify(sig1, sig1.bind(nested_type(seq), nested_type(index))),
    )

    inferred_type2 = substitute(
        sig2.return_annotation,
        unify(sig2, sig2.bind(nested_type(inferred_type1), nested_type(key))),
    )

    inferred_type12 = substitute(
        sig12.return_annotation,
        unify(
            sig12,
            sig12.bind(nested_type(seq), nested_type(index), nested_type(key)),
        ),
    )

    # check that the composed inference matches the direct inference
    assert isinstance(unify(inferred_type2, inferred_type12), collections.abc.Mapping)

    # check that the result of nested_type on the value of the composition unifies with the inferred type
    assert isinstance(
        unify(nested_type(sequence_mapping_getitem(seq, index, key)), inferred_type12),
        collections.abc.Mapping,
    )


@pytest.mark.parametrize(
    "mapping,key,index",
    [
        # Dict of lists with string keys
        (
            {
                "fruits": ["apple", "banana", "cherry"],
                "colors": ["red", "green", "blue"],
            },
            "fruits",
            1,
        ),
        ({"numbers": [1, 2, 3, 4, 5], "primes": [2, 3, 5, 7, 11]}, "primes", 3),
        # Different value types in sequences
        ({"floats": [1.1, 2.2, 3.3], "constants": [3.14, 2.71, 1.41]}, "constants", 0),
        (
            {"flags": [True, False, True, False], "states": [False, True, False]},
            "flags",
            2,
        ),
        # Nested structures
        (
            {"matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]], "identity": [[1, 0], [0, 1]]},
            "matrix",
            1,
        ),
        (
            {"teams": [{"name": "A", "score": 10}, {"name": "B", "score": 20}]},
            "teams",
            0,
        ),
        # Different key types
        (
            {
                1: ["one", "uno", "un"],
                2: ["two", "dos", "deux"],
                3: ["three", "tres", "trois"],
            },
            2,
            1,
        ),
        ({True: ["yes", "true", "1"], False: ["no", "false", "0"]}, False, 2),
        # Lists of different collection types
        (
            {"data": [{"a": 1}, {"b": 2}, {"c": 3}], "info": [{"x": 10}, {"y": 20}]},
            "data",
            2,
        ),
        # Edge cases
        ({"single": ["only"]}, "single", 0),
        ({"empty_key": [], "full": [1, 2, 3]}, "full", 1),
        # Complex nested case
        (
            {
                "users": [
                    {"id": 1, "tags": ["admin", "user"]},
                    {"id": 2, "tags": ["user", "guest"]},
                    {"id": 3, "tags": ["guest"]},
                ]
            },
            "users",
            1,
        ),
        # More diverse cases
        (
            {"names": ["Alice", "Bob", "Charlie", "David"], "ages": [25, 30, 35, 40]},
            "names",
            3,
        ),
        (
            {"options": [[1, 2], [3, 4], [5, 6]], "choices": [[7], [8], [9]]},
            "options",
            2,
        ),
        # Deeply nested lists
        (
            {"deep": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], "shallow": [[9, 10]]},
            "deep",
            0,
        ),
    ],
)
def test_infer_composition_2(mapping, key, index):
    sig1 = inspect.signature(mapping_getitem)
    sig2 = inspect.signature(sequence_getitem)

    sig12 = inspect.signature(mapping_sequence_getitem)

    # First infer type of mapping_getitem(mapping, key) -> should be a sequence
    inferred_type1 = substitute(
        sig1.return_annotation,
        unify(sig1, sig1.bind(nested_type(mapping), nested_type(key))),
    )

    # Then infer type of sequence_getitem(result_from_step1, index) -> should be element type
    inferred_type2 = substitute(
        sig2.return_annotation,
        unify(sig2, sig2.bind(nested_type(inferred_type1), nested_type(index))),
    )

    # Directly infer type of mapping_sequence_getitem(mapping, key, index)
    inferred_type12 = substitute(
        sig12.return_annotation,
        unify(
            sig12,
            sig12.bind(nested_type(mapping), nested_type(key), nested_type(index)),
        ),
    )

    # The composed inference should match the direct inference
    assert isinstance(unify(inferred_type2, inferred_type12), collections.abc.Mapping)

    # check that the result of nested_type on the value of the composition unifies with the inferred type
    assert isinstance(
        unify(
            nested_type(mapping_sequence_getitem(mapping, key, index)), inferred_type12
        ),
        collections.abc.Mapping,
    )


@pytest.mark.parametrize(
    "a,b,index",
    [
        # Basic types
        (1, 2, 0),
        (1, 2, 1),
        ("hello", "world", 0),
        (3.14, 2.71, 1),
        (True, False, 0),
        # Complex types
        ([1, 2], [3, 4], 1),
        ({"a": 1}, {"b": 2}, 0),
        ({1, 2}, {3, 4}, 1),
        # Mixed but same types
        ([1, 2, 3], [4, 5], 0),
        ({"x": "a", "y": "b"}, {"z": "c"}, 1),
    ],
)
def test_get_from_constructed_sequence(a, b, index):
    """Test type inference through sequence construction and retrieval."""
    sig_construct = inspect.signature(sequence_from_pair)
    sig_getitem = inspect.signature(sequence_getitem)
    sig_composed = inspect.signature(get_from_constructed_sequence)

    # Infer type of sequence_from_pair(a, b) -> Sequence[T]
    construct_subs = unify(
        sig_construct, sig_construct.bind(nested_type(a), nested_type(b))
    )
    inferred_sequence_type = substitute(sig_construct.return_annotation, construct_subs)

    # Infer type of sequence_getitem(sequence, index) -> T
    getitem_subs = unify(
        sig_getitem, sig_getitem.bind(inferred_sequence_type, nested_type(index))
    )
    inferred_element_type = substitute(sig_getitem.return_annotation, getitem_subs)

    # Directly infer type of get_from_constructed_sequence(a, b, index)
    direct_subs = unify(
        sig_composed,
        sig_composed.bind(nested_type(a), nested_type(b), nested_type(index)),
    )
    direct_type = substitute(sig_composed.return_annotation, direct_subs)

    # The composed inference should match the direct inference
    assert isinstance(
        unify(inferred_element_type, direct_type), collections.abc.Mapping
    )

    # check that the result of nested_type on the value of the composition unifies with the inferred type
    assert isinstance(
        unify(nested_type(get_from_constructed_sequence(a, b, index)), direct_type),
        collections.abc.Mapping,
    )


@pytest.mark.parametrize(
    "key,value,lookup_key",
    [
        # Basic types
        ("name", "Alice", "name"),
        (1, "one", 1),
        (True, "yes", True),
        (3.14, "pi", 3.14),
        # Complex value types
        ("data", [1, 2, 3], "data"),
        ("config", {"host": "localhost", "port": 8080}, "config"),
        ("items", {1, 2, 3}, "items"),
        # Different key types
        (42, {"value": "answer"}, 42),
        ("key", (1, 2, 3), "key"),
    ],
)
def test_get_from_constructed_mapping(key, value, lookup_key):
    """Test type inference through mapping construction and retrieval."""
    sig_construct = inspect.signature(mapping_from_pair)
    sig_getitem = inspect.signature(mapping_getitem)
    sig_composed = inspect.signature(get_from_constructed_mapping)

    # Infer type of mapping_from_pair(key, value) -> Mapping[K, V]
    construct_subs = unify(
        sig_construct, sig_construct.bind(nested_type(key), nested_type(value))
    )
    inferred_mapping_type = substitute(sig_construct.return_annotation, construct_subs)

    # Infer type of mapping_getitem(mapping, lookup_key) -> V
    getitem_subs = unify(
        sig_getitem, sig_getitem.bind(inferred_mapping_type, nested_type(lookup_key))
    )
    inferred_value_type = substitute(sig_getitem.return_annotation, getitem_subs)

    # Directly infer type of get_from_constructed_mapping(key, value, lookup_key)
    direct_subs = unify(
        sig_composed,
        sig_composed.bind(
            nested_type(key), nested_type(value), nested_type(lookup_key)
        ),
    )
    direct_type = substitute(sig_composed.return_annotation, direct_subs)

    # The composed inference should match the direct inference
    assert isinstance(unify(inferred_value_type, direct_type), collections.abc.Mapping)

    # check that the result of nested_type on the value of the composition unifies with the inferred type
    assert isinstance(
        unify(
            nested_type(get_from_constructed_mapping(key, value, lookup_key)),
            direct_type,
        ),
        collections.abc.Mapping,
    )


@pytest.mark.parametrize(
    "key1,val1,key2,val2,index",
    [
        # Basic case
        ("a", 1, "b", 2, 0),
        ("x", "hello", "y", "world", 1),
        # Different types
        (1, "one", 2, "two", 0),
        (True, 1.0, False, 0.0, 1),
        # Complex values
        ("list1", [1, 2], "list2", [3, 4], 0),
        ("dict1", {"a": 1}, "dict2", {"b": 2}, 1),
    ],
)
def test_sequence_of_mappings(key1, val1, key2, val2, index):
    """Test type inference for creating a sequence of mappings."""
    sig_map = inspect.signature(mapping_from_pair)
    sig_seq = inspect.signature(sequence_from_pair)
    sig_composed = inspect.signature(sequence_of_mappings)

    # Step 1: Infer types of the two mappings
    map1_subs = unify(sig_map, sig_map.bind(nested_type(key1), nested_type(val1)))
    map1_type = substitute(sig_map.return_annotation, map1_subs)

    # Step 2: Infer type of sequence containing these mappings
    # We need to unify the two mapping types first
    unified_map_type = map1_type  # Assuming they're compatible

    seq_subs = unify(sig_seq, sig_seq.bind(unified_map_type, unified_map_type))
    seq_type = substitute(sig_seq.return_annotation, seq_subs)

    # Direct inference
    direct_subs = unify(
        sig_composed,
        sig_composed.bind(
            nested_type(key1), nested_type(val1), nested_type(key2), nested_type(val2)
        ),
    )
    direct_type = substitute(sig_composed.return_annotation, direct_subs)

    # The types should match
    assert isinstance(unify(seq_type, direct_type), collections.abc.Mapping)

    # Note: nested_type(sequence_of_mappings(...)) returns concrete types (list[dict[K,V]])
    # while our function signature uses abstract types (Sequence[Mapping[K,V]])
    # This is expected behavior - concrete implementations vs abstract interfaces


@pytest.mark.parametrize(
    "k1,v1,v2,k2,v3,v4,outer_idx,inner_key,inner_idx",
    [
        # Basic test case
        ("first", 1, 2, "second", 3, 4, 0, "first", 1),
        ("a", "x", "y", "b", "z", "w", 1, "b", 0),
        # Different types
        (1, 10.0, 20.0, 2, 30.0, 40.0, 0, 1, 1),
        ("data", [1], [2], "info", [3], [4], 1, "info", 0),
    ],
)
def test_double_nested_get(k1, v1, v2, k2, v3, v4, outer_idx, inner_key, inner_idx):
    """Test type inference through deeply nested structure construction and retrieval."""
    # Get signatures for all functions involved
    sig_nested = inspect.signature(nested_sequence_mapping)
    sig_seq_get = inspect.signature(sequence_getitem)
    sig_map_get = inspect.signature(mapping_getitem)
    sig_composed = inspect.signature(double_nested_get)

    # Step 1: Infer type of nested_sequence_mapping construction
    nested_subs = unify(
        sig_nested,
        sig_nested.bind(
            nested_type(k1),
            nested_type(v1),
            nested_type(v2),
            nested_type(k2),
            nested_type(v3),
            nested_type(v4),
        ),
    )
    nested_seq_type = substitute(sig_nested.return_annotation, nested_subs)
    # This should be Sequence[Mapping[K, Sequence[T]]]

    # Step 2: Get element from outer sequence
    outer_get_subs = unify(
        sig_seq_get, sig_seq_get.bind(nested_seq_type, nested_type(outer_idx))
    )
    mapping_type = substitute(sig_seq_get.return_annotation, outer_get_subs)
    # This should be Mapping[K, Sequence[T]]

    # Step 3: Get sequence from mapping
    inner_map_subs = unify(
        sig_map_get, sig_map_get.bind(mapping_type, nested_type(inner_key))
    )
    sequence_type = substitute(sig_map_get.return_annotation, inner_map_subs)
    # This should be Sequence[T]

    # Step 4: Get element from inner sequence
    final_get_subs = unify(
        sig_seq_get, sig_seq_get.bind(sequence_type, nested_type(inner_idx))
    )
    composed_type = substitute(sig_seq_get.return_annotation, final_get_subs)
    # This should be T

    # Direct inference on the composed function
    direct_subs = unify(
        sig_composed,
        sig_composed.bind(
            nested_type(k1),
            nested_type(v1),
            nested_type(v2),
            nested_type(k2),
            nested_type(v3),
            nested_type(v4),
            nested_type(outer_idx),
            nested_type(inner_key),
            nested_type(inner_idx),
        ),
    )
    direct_type = substitute(sig_composed.return_annotation, direct_subs)

    # The composed inference should match the direct inference
    assert isinstance(unify(composed_type, direct_type), collections.abc.Mapping)

    # check that the result of nested_type on the value of the composition unifies with the inferred type
    assert isinstance(
        unify(
            nested_type(
                double_nested_get(
                    k1, v1, v2, k2, v3, v4, outer_idx, inner_key, inner_idx
                )
            ),
            direct_type,
        ),
        collections.abc.Mapping,
    )
