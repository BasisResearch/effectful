import collections.abc
import inspect
import typing

import pytest

from effectful.internals.unification import (
    freetypevars,
    infer_return_type,
    substitute,
    unify,
)

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
    ids=str,
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
    ids=str,
)
def test_substitute(
    typ: type, subs: typing.Mapping[typing.TypeVar, type], expected: type
):
    assert substitute(typ, subs) == expected


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
        # Union types - basic element-wise unification (current implementation)
        # Note: Current unify treats union args as sequences, not true union logic
        (
            T | V,
            int | str,
            {},
            {T: int, V: str},
        ),  # Element-wise unification of TypeVars
        (T | V, int | str, {}, {T: int, V: str}),  # typing.Union syntax
        # Simple union compatibility - TypeVar gets unified with itself
        (T | int, T | int, {}, {T: T}),  # Identical unions - T unifies with T
        (T | int, T | int, {}, {T: T}),  # Identical typing.Union
        # Sequence unification (tuples as sequences)
        ((T, V), (int, str), {}, {T: int, V: str}),
        ([T, V], [int, str], {}, {T: int, V: str}),
        # Complex combinations
        (
            dict[K, collections.abc.Callable[[T], V]],
            dict[str, collections.abc.Callable[[int], bool]],
            {},
            {K: str, T: int, V: bool},
        ),
    ],
    ids=str,
)
def test_unify_success(
    typ: type,
    subtyp: type,
    initial_subs: typing.Mapping[typing.TypeVar, type],
    expected_subs: typing.Mapping[typing.TypeVar, type],
):
    assert unify(typ, subtyp, initial_subs) == expected_subs


@pytest.mark.parametrize(
    "typ,subtyp,initial_subs,error_pattern",
    [
        # Incompatible types
        (
            list[T],
            dict[str, int],
            {},
            "Cannot unify list\\[~T\\] with dict\\[str, int\\]",
        ),
        (int, str, {}, "Cannot unify <class 'int'> with <class 'str'>"),
        (list[int], list[str], {}, "Cannot unify <class 'int'> with <class 'str'>"),
        # Conflicting TypeVar bindings
        (
            T,
            str,
            {T: int},
            "Cannot unify ~T with <class 'str'> \\(already unified with <class 'int'>\\)",
        ),
        (
            list[T],
            list[str],
            {T: int},
            "Cannot unify ~T with <class 'str'> \\(already unified with <class 'int'>\\)",
        ),
        # Mismatched generic types
        (list[T], set[int], {}, "Cannot unify list\\[~T\\] with set\\[int\\]"),
        (dict[K, V], list[int], {}, "Cannot unify dict\\[~K, ~V\\] with list\\[int\\]"),
        # Same TypeVar with different values
        (
            dict[T, T],
            dict[int, str],
            {},
            "Cannot unify ~T with <class 'str'> \\(already unified with <class 'int'>\\)",
        ),
        (
            tuple[T, T],
            tuple[int, str],
            {},
            "Cannot unify ~T with <class 'str'> \\(already unified with <class 'int'>\\)",
        ),
        # Mismatched arities
        (tuple[T, U], tuple[int, str, bool], {}, "Cannot unify"),
        (
            collections.abc.Callable[[T], V],
            collections.abc.Callable[[int, str], bool],
            {},
            "Cannot unify",
        ),
        # Sequence length mismatch
        ((T, V), (int,), {}, "Cannot unify"),
        ([T, V], [int, str, bool], {}, "Cannot unify"),
        # Union type failures - element-wise unification failures
        (
            T | V,
            int | str,
            {T: float},
            "Cannot unify ~T with <class 'int'>",
        ),  # TypeVar conflict
        (
            T | int,
            V | str,
            {},
            "Cannot unify <class 'int'> with <class 'str'>",
        ),  # Concrete type mismatch
        (
            T | int,
            V | str,
            {},
            "Cannot unify <class 'int'> with <class 'str'>",
        ),  # typing.Union mismatch
        # Union with different arities
        (T | V, int | str | bool, {}, "Cannot unify"),  # Different union sizes
    ],
    ids=str,
)
def test_unify_failure(
    typ: type,
    subtyp: type,
    initial_subs: typing.Mapping[typing.TypeVar, type],
    error_pattern: str,
):
    with pytest.raises(TypeError, match=error_pattern):
        unify(typ, subtyp, initial_subs)


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
    ],
    ids=str,
)
def test_infer_return_type_success(
    func: collections.abc.Callable,
    args: tuple,
    kwargs: dict,
    expected_return_type: type,
):
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    result = infer_return_type(bound)
    assert result == expected_return_type


# Error cases
def unbound_typevar_func(x: T) -> tuple[T, V]:  # V not in parameters
    return (x, "error")  # type: ignore


def no_return_annotation(x: T):  # No return annotation
    return x


def no_param_annotation(x) -> T:  # No parameter annotation
    return x  # type: ignore


def variadic_args_func(*args: T) -> T:  # Variadic args not supported
    return args[0]


def variadic_kwargs_func(**kwargs: T) -> T:  # Variadic kwargs not supported
    return next(iter(kwargs.values()))


@pytest.mark.parametrize(
    "func,args,kwargs",
    [
        # Unbound type variable in return
        (
            unbound_typevar_func,
            (int,),
            {},
        ),
        # Missing annotations
        (
            no_return_annotation,
            (int,),
            {},
        ),
        (
            no_param_annotation,
            (int,),
            {},
        ),
        # Type mismatch - trying to unify incompatible types
        (same_type_twice, (int, str), {}),
    ],
    ids=str,
)
def test_infer_return_type_failure(
    func: collections.abc.Callable,
    args: tuple,
    kwargs: dict,
):
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    with pytest.raises(TypeError):
        infer_return_type(bound)


# Variadic functions - not implemented yet, marked as expected failures
@pytest.mark.xfail(reason="Variadic args not implemented")
def test_infer_return_type_variadic_args():
    sig = inspect.signature(variadic_args_func)
    bound = sig.bind(int)
    result = infer_return_type(bound)
    assert result == int


@pytest.mark.xfail(reason="Variadic kwargs not implemented")
def test_infer_return_type_variadic_kwargs():
    sig = inspect.signature(variadic_kwargs_func)
    bound = sig.bind(x=int)
    result = infer_return_type(bound)
    assert result == int
