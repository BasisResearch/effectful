import collections.abc
import dataclasses
import functools
import inspect
import typing
from typing import Literal

import pytest

from effectful.internals.unification import (
    Box,
    Substitutions,
    TypeEvaluator,
    TypeExpressions,
    TypeVariable,
    canonicalize,
    freetypevars,
    infer_return_type,
    nested_type,
    substitute,
    unify,
)

if typing.TYPE_CHECKING:
    T = typing.Any
    K = typing.Any
    V = typing.Any
    U = typing.Any
    W = typing.Any
else:
    T = typing.TypeVar("T")
    K = typing.TypeVar("K")
    V = typing.TypeVar("V")
    U = typing.TypeVar("U")
    W = typing.TypeVar("W")


@dataclasses.dataclass
class _Substitute(TypeEvaluator):
    """
    Helper class to perform substitution using TypeEvaluator.
    Used to test TypeEvaluator's traversal logic against the ground-truth ``substitute``
    """

    subs: Substitutions

    @classmethod
    def substitute(cls, typ: TypeExpressions, subs: Substitutions) -> TypeExpressions:
        return cls(subs).evaluate(typ)

    @functools.singledispatchmethod
    def evaluate(self, typ: TypeExpressions) -> TypeExpressions:
        return super().evaluate(typ)

    @evaluate.register
    def _(self, typ: TypeVariable):
        if typ in self.subs:
            return self.evaluate(self.subs[typ])
        else:
            return super().evaluate(typ)


@dataclasses.dataclass
class _FreeTypeVars(TypeEvaluator):
    """
    Helper class to perform free variable collection using TypeEvaluator.
    Used to test TypeEvaluator's traversal logic against the ground-truth ``freetypevars``
    """

    fvs: set[TypeVariable] = dataclasses.field(default_factory=set)

    @classmethod
    def freetypevars(cls, typ: TypeExpressions) -> collections.abc.Set[TypeVariable]:
        evaluator = cls()
        evaluator.evaluate(typ)
        return evaluator.fvs

    @functools.singledispatchmethod
    def evaluate(self, typ: TypeExpressions) -> TypeExpressions:
        return super().evaluate(typ)

    @evaluate.register
    def _(self, typ: TypeVariable):
        self.fvs.add(typ)
        return super().evaluate(typ)


@pytest.mark.parametrize(
    "typ,expected",
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
@pytest.mark.parametrize(
    "freetypevars_impl", [freetypevars, _FreeTypeVars.freetypevars]
)
def test_freetypevars(
    freetypevars_impl: collections.abc.Callable[
        [TypeExpressions], collections.abc.Set[TypeVariable]
    ],
    typ: TypeExpressions,
    expected: collections.abc.Set[typing.TypeVar],
):
    assert freetypevars_impl(typ) == expected


def test_canonicalize_1():
    assert canonicalize(int) == int
    assert canonicalize(list[int]) == collections.abc.MutableSequence[int]
    assert canonicalize(dict[str, int]) == collections.abc.MutableMapping[str, int]
    assert (
        canonicalize(dict[str, set[int]])
        == collections.abc.MutableMapping[str, collections.abc.MutableSet[int]]
    )
    assert canonicalize(tuple[int, ...]) == collections.abc.Sequence[int]
    assert canonicalize(tuple[int, str]) == tuple[int, str]

    class CustomDict[T](dict[T, T]):
        pass

    class ConcreteCustomDict(CustomDict[int]):
        pass

    class CustomDictSet[T](CustomDict[frozenset[T]]):
        pass

    assert canonicalize(CustomDict[int]) == canonicalize(dict[int, int])
    assert canonicalize(ConcreteCustomDict) == canonicalize(CustomDict[int])
    assert canonicalize(CustomDictSet[T]) == canonicalize(
        dict[frozenset[T], frozenset[T]]
    )
    assert canonicalize(CustomDictSet[int]) == canonicalize(
        dict[frozenset[int], frozenset[int]]
    )

    class GenericClass[T]:
        pass

    assert canonicalize(GenericClass[int]) == GenericClass[int]

    assert canonicalize(Literal["read", "write"]) == str
    assert canonicalize(Literal[200, 404]) == int
    assert canonicalize(Literal[True]) == bool
    assert canonicalize(Literal[1, "a"]) == (int | str)


def test_canonicalize_typeddict_class():
    """Canonicalizing a TypedDict class preserves structure with canonical field types."""

    class MyTD(typing.TypedDict):
        name: str
        age: int

    # Fields are already canonical types
    result = canonicalize(MyTD)
    assert typing.is_typeddict(result)

    hints = typing.get_type_hints(result)
    assert hints["name"] is str
    assert hints["age"] is int

    class MyTD2(typing.TypedDict):
        name: str
        items: list[int]

    # list[int] -> MutableSequence[int], so a new TypedDict is created
    result = canonicalize(MyTD2)
    assert typing.is_typeddict(result)
    hints = typing.get_type_hints(result)
    assert hints["name"] is str
    assert hints["items"] == collections.abc.MutableSequence[int]

    # Idempotent: canonicalizing the result returns the same object
    assert canonicalize(result) is result


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
        # typing.Self with no binding passes through unchanged
        (typing.Self, {}, typing.Self),
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
@pytest.mark.parametrize("substitute_impl", [substitute, _Substitute.substitute])
def test_substitute(
    substitute_impl: collections.abc.Callable[
        [TypeExpressions, Substitutions], TypeExpressions
    ],
    typ: type,
    subs: typing.Mapping[typing.TypeVar, type],
    expected: type,
):
    assert substitute_impl(typ, subs) == expected  # type: ignore


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
        # Literal type unification
        (Literal["read", "write"], str, {}, {}),
        (Literal[200, 404], int, {}, {}),
        (Literal[True], bool, {}, {}),
        (T, Literal["read", "write"], {}, {T: str}),
        (T, Literal[1, 2, 3], {}, {T: int}),
        (list[T], list[Literal["a", "b"]], {}, {T: str}),
    ],
)
def test_unify_success(
    typ: type,
    subtyp: type,
    initial_subs: typing.Mapping,
    expected_subs: typing.Mapping,
):
    assert unify(typ, subtyp, initial_subs) == {
        k: canonicalize(v) for k, v in expected_subs.items()
    }


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


def test_unify_union_1():
    assert unify(int | str, int | str) == {}
    assert unify(int | str, str) == {}
    assert unify(int | str, int) == {}

    assert unify(T, int | str) == {T: int | str}


def test_unify_tuple_variadic():
    assert unify(tuple[T, ...], tuple[int, ...]) == {T: int}
    assert unify(tuple[T, ...], tuple[int]) == {T: int}
    assert unify(tuple[T, ...], tuple[int, int]) == {T: int}
    assert unify(collections.abc.Sequence[T], tuple[int, ...]) == {T: int}


def test_unify_tuple_non_variadic():
    assert unify(tuple[T], tuple[int | str]) == {T: int | str}
    assert unify(tuple[T, V], tuple[int, str]) == {T: int, V: str}
    assert unify(tuple[T, T], tuple[int, int]) == {T: int}
    assert unify(tuple[T, T, T], tuple[str, str, str]) == {T: str}
    assert unify(collections.abc.Sequence[T], tuple[int, int]) == {T: int}


def test_unify_both_abstract():
    assert unify(tuple[T, ...], tuple[V, V]) == {T: V}
    assert unify(tuple[T, V], tuple[int, U]) == {T: int, V: U}
    assert unify(list[T], list[tuple[V, V]]) == {T: tuple[V, V]}
    assert unify(
        collections.abc.Callable[[T], T],
        collections.abc.Callable[[tuple[int, V]], tuple[int, V]],
    ) == {T: tuple[int, V]}
    assert unify(
        (tuple[T, int], tuple[int, int]),
        (tuple[int, V], tuple[U, V]),
    ) == {T: int, U: int, V: int}
    assert unify(
        (list[T], T),
        (list[list[V]], list[V]),
    ) == {T: canonicalize(list[V])}


# Test functions with various type patterns
def identity[T](x: T) -> T:
    return x


def make_pair[T, V](x: T, y: V) -> tuple[T, V]:
    return (x, y)


def wrap_in_list[T](x: T) -> list[T]:
    return [x]


def get_first[T](items: list[T]) -> T:
    return items[0]


def getitem_mapping[K, V](mapping: collections.abc.Mapping[K, V], key: K) -> V:
    return mapping[key]


def dict_values[K, V](d: dict[K, V]) -> list[V]:
    return list(d.values())


def process_callable[T, V](func: collections.abc.Callable[[T], V], arg: T) -> V:
    return func(arg)


def chain_callables[T, U, V](
    f: collections.abc.Callable[[T], U], g: collections.abc.Callable[[U], V]
) -> collections.abc.Callable[[T], V]:
    def result(x: T) -> V:
        return g(f(x))

    return result


def constant_func() -> int:
    return 42


def multi_generic[T, K, V](a: T, b: list[T], c: dict[K, V]) -> tuple[T, K, V]:
    return (a, next(iter(c.keys())), next(iter(c.values())))


def same_type_twice[T](x: T, y: T) -> T:
    return x if len(str(x)) > len(str(y)) else y


def nested_generic[T](x: T) -> dict[str, list[T]]:
    return {"items": [x]}


def variadic_args_func[T](*args: T) -> T:  # Variadic args not supported
    return args[0]


def variadic_kwargs_func[T](**kwargs: T) -> T:  # Variadic kwargs not supported
    return next(iter(kwargs.values()))


def variadic_args_const_return(*args: object) -> int:
    return len(args)


def default_param_func[T](x: T, y: str = "ok") -> T:
    return x


def returns_none(x: int) -> None:
    return None


def returns_annotated[T](x: T) -> typing.Annotated[T, "meta"]:
    return x


def no_return_annotation_func(x: int):  # type: ignore[no-untyped-def]
    return x


class _Foo:
    def return_self(self: typing.Self) -> typing.Self:
        return self

    def return_list_self(self: typing.Self) -> list[typing.Self]:
        return [self]

    def return_self_or_none(self: typing.Self) -> typing.Self | None:
        return self

    def annotated_self(self: typing.Self) -> typing.Self:
        return self

    def takes_other(self: typing.Self, other: typing.Self) -> typing.Self:
        return self

    def mixed_with_typevar[T](self: typing.Self, x: T) -> tuple[typing.Self, T]:
        return (self, x)

    def return_dict_self(self: typing.Self) -> dict[str, typing.Self]:
        return {"me": self}

    def return_callable_self(
        self: typing.Self,
    ) -> collections.abc.Callable[[typing.Self], int]:
        return id

    def return_type_self(self: typing.Self) -> type[typing.Self]:
        return type(self)

    @classmethod
    def from_config(cls, config: int) -> typing.Self:
        return cls()


class _Bar:
    def return_self(self: typing.Self) -> typing.Self:
        return self


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
        # zero-arg variadic positional/keyword (return type doesn't depend on T)
        (variadic_args_const_return, (), {}, int),
        # default-valued param: omitted vs provided
        (default_param_func, (int,), {}, int),
        (default_param_func, (str,), {"y": str}, str),
        # special return-annotation forms
        (returns_none, (int,), {}, type(None)),
        (returns_annotated, (int,), {}, int),
        (no_return_annotation_func, (int,), {}, object),
    ],
)
def test_infer_return_type_success(
    func: collections.abc.Callable,
    args: tuple,
    kwargs: dict,
    expected_return_type: type,
):
    # Args here are type expressions, not values; box them so nested_type
    # treats them as already-typed (per Box's intended purpose).
    sig = inspect.signature(func)
    bound = sig.bind(
        *[Box(a) for a in args],
        **{k: Box(v) for k, v in kwargs.items()},
    )
    result = infer_return_type(bound)
    assert canonicalize(result) == canonicalize(expected_return_type)


@pytest.mark.parametrize(
    "func,args,kwargs,expected_return_type",
    [
        # typing.Self return types (methods).  Args are wrapped in Box so
        # nested_type treats them as already-typed values.
        (_Foo.return_self, (Box(int),), {}, int),
        (_Foo.return_self, (Box(str),), {}, str),
        (_Foo.return_self, (Box(_Foo),), {}, _Foo),
        (_Foo.return_self, (Box(_Bar),), {}, _Bar),
        (_Bar.return_self, (Box(_Bar),), {}, _Bar),
        (_Foo.annotated_self, (Box(_Foo),), {}, _Foo),
        (_Foo.return_list_self, (Box(int),), {}, list[int]),
        (_Foo.return_list_self, (Box(_Foo),), {}, list[_Foo]),
        (_Foo.return_self_or_none, (Box(int),), {}, int | None),
        (_Foo.return_self_or_none, (Box(_Foo),), {}, _Foo | None),
        # Self as a non-self parameter
        (_Foo.takes_other, (Box(int), Box(int)), {}, int),
        (_Foo.takes_other, (Box(_Foo), Box(_Foo)), {}, _Foo),
        # Self mixed with other TypeVars
        (_Foo.mixed_with_typevar, (Box(int), Box(str)), {}, tuple[int, str]),
        (
            _Foo.mixed_with_typevar,
            (Box(_Foo), Box(list[int])),
            {},
            tuple[_Foo, list[int]],
        ),
        # Self in dict[str, Self]
        (_Foo.return_dict_self, (Box(int),), {}, dict[str, int]),
        (_Foo.return_dict_self, (Box(_Foo),), {}, dict[str, _Foo]),
        # Self inside Callable[[Self], int]
        (
            _Foo.return_callable_self,
            (Box(int),),
            {},
            collections.abc.Callable[[int], int],
        ),
        (
            _Foo.return_callable_self,
            (Box(_Foo),),
            {},
            collections.abc.Callable[[_Foo], int],
        ),
        # type[Self]
        (_Foo.return_type_self, (Box(int),), {}, type[int]),
        (_Foo.return_type_self, (Box(_Foo),), {}, type[_Foo]),
    ],
)
def test_infer_return_type_self_success(
    func: collections.abc.Callable,
    args: tuple,
    kwargs: dict,
    expected_return_type: type,
):
    """Self resolution goes through infer_return_type (the Self TypeVar is
    internal to the conversion in _sig_to_type and never appears in the result
    subs that the user could pass to substitute)."""
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    result = infer_return_type(bound)
    assert canonicalize(result) == canonicalize(expected_return_type)


# Error cases
def unbound_typevar_func[T](x: T) -> tuple[T, V]:  # V not in parameters
    return (x, "error")


def no_return_annotation[T](x: T):  # No return annotation
    return x


def no_param_annotation[T](x) -> T:  # type: ignore
    return x


@pytest.mark.parametrize(
    "func,args,kwargs",
    [
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
    bound = sig.bind(
        *[Box(a) for a in args],
        **{k: Box(v) for k, v in kwargs.items()},
    )
    with pytest.raises(TypeError):
        infer_return_type(bound)


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
        # Boxed type objects pass through
        (Box(int), int),
        # Boxed generic aliases pass through
        (Box(list[int]), list[int]),
        (int, type),
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
    result = nested_type(value).value
    assert canonicalize(result) == canonicalize(expected)


def test_nested_type_typeddict_str_keys_mixed_values():
    """Dicts with str keys and heterogeneous value types produce a TypedDict."""
    value = {"name": "Alice", "age": 30}
    result = nested_type(value).value
    # Should be a TypedDict, not dict
    assert typing.is_typeddict(result)
    hints = typing.get_type_hints(result)
    assert hints == {"name": str, "age": int}


def test_nested_type_typeddict_multiple_value_types():
    """TypedDict with more than two distinct value types."""
    value = {"label": "x", "count": 5, "flag": True}
    result = nested_type(value).value
    assert typing.is_typeddict(result)
    hints = typing.get_type_hints(result)
    assert hints == {"label": str, "count": int, "flag": bool}


def test_nested_type_typeddict_nested_values():
    """TypedDict with nested collection values."""
    value = {"items": [1, 2, 3], "name": "test"}
    result = nested_type(value).value
    assert typing.is_typeddict(result)
    hints = typing.get_type_hints(result)
    assert canonicalize(hints["items"]) == canonicalize(list[int])
    assert hints["name"] is str


def test_nested_type_typeddict_instance_roundtrip():
    """A TypedDict instance with heterogeneous values produces a TypedDict."""

    class UserTD(typing.TypedDict):
        name: str
        age: int

    value = UserTD(name="a", age=1)
    result = nested_type(value).value
    assert typing.is_typeddict(result)
    hints = typing.get_type_hints(result)
    assert hints == {"name": str, "age": int}


def test_nested_type_typeddict_homogeneous_str_keys():
    """Multi-key str dicts produce TypedDict even with homogeneous value types."""
    result = nested_type({"a": 1, "b": 2}).value
    assert typing.is_typeddict(result)
    hints = typing.get_type_hints(result)
    assert hints == {"a": int, "b": int}

    result = nested_type({"a": {1, 2}, "b": {3, 4}}).value
    assert typing.is_typeddict(result)
    hints = typing.get_type_hints(result)
    assert canonicalize(hints["a"]) == canonicalize(set[int])
    assert canonicalize(hints["b"]) == canonicalize(set[int])


def test_nested_type_non_str_keys_mixed_values_stays_dict():
    """Dicts with non-str keys and mixed value types stay as plain dict."""
    value = {1: "one", 2: True}
    result = nested_type(value).value
    # Should remain a plain dict type, not a TypedDict
    assert not typing.is_typeddict(result)
    assert result is dict


def test_nested_type_term_error():
    """Test that Terms raise TypeError in nested_type"""
    # We can't import Term here without creating a circular dependency,
    # so we'll create a mock object that would trigger the isinstance check
    from unittest.mock import Mock

    from effectful.ops.types import Term

    mock_term = Mock(spec=Term)
    with pytest.raises(TypeError, match="Terms should not appear in nested_type"):
        nested_type(mock_term)


def sequence_getitem[T](seq: collections.abc.Sequence[T], index: int) -> T:
    return seq[index]


def mapping_getitem[K, V](mapping: collections.abc.Mapping[K, V], key: K) -> V:
    return mapping[key]


def sequence_mapping_getitem[K, V](
    seq: collections.abc.Sequence[collections.abc.Mapping[K, V]], index: int, key: K
) -> V:
    return mapping_getitem(sequence_getitem(seq, index), key)


def mapping_sequence_getitem[K, T](
    mapping: collections.abc.Mapping[K, collections.abc.Sequence[T]], key: K, index: int
) -> T:
    return sequence_getitem(mapping_getitem(mapping, key), index)


def sequence_from_pair[T](a: T, b: T) -> collections.abc.Sequence[T]:
    return [a, b]


def mapping_from_pair[K, V](a: K, b: V) -> collections.abc.Mapping[K, V]:
    return {a: b}


def sequence_of_mappings[K, V](
    key1: K, val1: V, key2: K, val2: V
) -> collections.abc.Sequence[collections.abc.Mapping[K, V]]:
    """Creates a sequence containing two mappings."""
    return sequence_from_pair(
        mapping_from_pair(key1, val1), mapping_from_pair(key2, val2)
    )


def mapping_of_sequences[K, T](
    key1: K, val1: T, val2: T, key2: K, val3: T, val4: T
) -> collections.abc.Mapping[K, collections.abc.Sequence[T]]:
    """Creates a mapping where each key maps to a sequence of two values."""
    return mapping_from_pair(key1, sequence_from_pair(val1, val2))


def nested_sequence_mapping[K, T](
    k1: K, v1: T, v2: T, k2: K, v3: T, v4: T
) -> collections.abc.Sequence[collections.abc.Mapping[K, collections.abc.Sequence[T]]]:
    """Creates a sequence of mappings, where each mapping contains sequences."""
    return sequence_from_pair(
        mapping_from_pair(k1, sequence_from_pair(v1, v2)),
        mapping_from_pair(k2, sequence_from_pair(v3, v4)),
    )


def get_from_constructed_sequence[T](a: T, b: T, index: int) -> T:
    """Constructs a sequence from two elements and gets one by index."""
    return sequence_getitem(sequence_from_pair(a, b), index)


def get_from_constructed_mapping[K, V](key: K, value: V, lookup_key: K) -> V:
    """Constructs a mapping from a key-value pair and looks up the value."""
    return mapping_getitem(mapping_from_pair(key, value), lookup_key)


def double_nested_get[K, T](
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


def construct_and_extend_sequence[T](
    a: T, b: T, c: T, d: T
) -> collections.abc.Sequence[collections.abc.Sequence[T]]:
    """Constructs two sequences and combines them into a sequence of sequences."""
    seq1 = sequence_from_pair(a, b)
    seq2 = sequence_from_pair(c, d)
    return sequence_from_pair(seq1, seq2)


def transform_mapping_values[K, T](
    key1: K, val1: T, key2: K, val2: T
) -> collections.abc.Mapping[K, collections.abc.Sequence[T]]:
    """Creates a mapping where each value is wrapped in a sequence."""
    # Create mappings where each value becomes a single-element sequence
    # Note: In a real implementation, we'd need a sequence_from_single function
    # For now, using sequence_from_pair with the same value twice as a workaround
    return mapping_from_pair(key1, sequence_from_pair(val1, val1))


def call_func[T, V](
    func: collections.abc.Callable[[T], V],
    arg: T,
) -> V:
    """Calls a function with a single argument."""
    return func(arg)


def call_binary_func[T, U, V](
    func: collections.abc.Callable[[T, U], V],
    arg1: T,
    arg2: U,
) -> V:
    """Calls a binary function with two arguments."""
    return func(arg1, arg2)


def map_sequence[T, U](
    f: collections.abc.Callable[[T], U],
    seq: collections.abc.Sequence[T],
) -> collections.abc.Sequence[U]:
    """Applies a function to each element in a sequence."""
    return [call_func(f, x) for x in seq]


def compose_mappings[T, U, V](
    f: collections.abc.Callable[[T], U],
    g: collections.abc.Callable[[U], V],
) -> collections.abc.Callable[[T], V]:
    """Composes two functions that operate on mappings."""

    def composed(x: T) -> V:
        return call_func(g, call_func(f, x))

    return composed


def compose_binary[T, U, V](
    f: collections.abc.Callable[[T], U],
    g: collections.abc.Callable[[U, U], V],
) -> collections.abc.Callable[[T], V]:
    """Composes a unary function with a binary function."""

    def composed(x: T) -> V:
        return call_binary_func(g, call_func(f, x), call_func(f, x))

    return composed


def apply_to_sequence_element[T, U](
    f: collections.abc.Callable[[T], U],
    seq: collections.abc.Sequence[T],
    index: int,
) -> U:
    """Gets an element from a sequence and applies a function to it."""
    element = sequence_getitem(seq, index)
    return call_func(f, element)


def map_and_get[T, U](
    f: collections.abc.Callable[[T], U],
    seq: collections.abc.Sequence[T],
    index: int,
) -> U:
    """Maps a function over a sequence and gets element at index."""
    mapped_seq = map_sequence(f, seq)
    return sequence_getitem(mapped_seq, index)


def compose_and_apply[T, U, V](
    f: collections.abc.Callable[[T], U],
    g: collections.abc.Callable[[U], V],
    value: T,
) -> V:
    """Composes two functions and applies the result to a value."""
    composed = compose_mappings(f, g)
    return call_func(composed, value)


def double_compose_apply[T, U, V, W](
    f: collections.abc.Callable[[T], U],
    g: collections.abc.Callable[[U], V],
    h: collections.abc.Callable[[V], W],
    value: T,
) -> W:
    """Composes three functions and applies to a value."""
    fg = compose_mappings(f, g)
    fgh = compose_mappings(fg, h)
    return call_func(fgh, value)


def binary_on_sequence_elements[T, U](
    f: collections.abc.Callable[[T, T], U],
    seq: collections.abc.Sequence[T],
    index1: int,
    index2: int,
) -> U:
    """Gets two elements from a sequence and applies a binary function."""
    elem1 = sequence_getitem(seq, index1)
    elem2 = sequence_getitem(seq, index2)
    return call_binary_func(f, elem1, elem2)


def map_sequence_and_apply_binary[T, U, V](
    f: collections.abc.Callable[[T], U],
    g: collections.abc.Callable[[U, U], V],
    seq: collections.abc.Sequence[T],
    index1: int,
    index2: int,
) -> V:
    """Maps a function over sequence, then applies binary function to two elements."""
    mapped = map_sequence(f, seq)
    elem1 = sequence_getitem(mapped, index1)
    elem2 = sequence_getitem(mapped, index2)
    return call_binary_func(g, elem1, elem2)


def construct_apply_and_get[T, U](
    f: collections.abc.Callable[[T], U],
    a: T,
    b: T,
    index: int,
) -> U:
    """Constructs a sequence, applies function to elements, and gets one."""
    seq = sequence_from_pair(a, b)
    return apply_to_sequence_element(f, seq, index)


def sequence_function_composition[T](
    funcs: collections.abc.Sequence[collections.abc.Callable[[T], T]],
    value: T,
) -> T:
    """Applies a sequence of functions in order to a value."""
    result = value
    for func in funcs:
        result = call_func(func, result)
    return result


def map_with_constructed_function[T, U, V](
    f: collections.abc.Callable[[T], U],
    g: collections.abc.Callable[[U], V],
    seq: collections.abc.Sequence[T],
) -> collections.abc.Sequence[V]:
    """Composes two functions and maps the result over a sequence."""
    composed = compose_mappings(f, g)
    return map_sequence(composed, seq)


def cross_apply_binary[T, U, V](
    f: collections.abc.Callable[[T, U], V],
    seq1: collections.abc.Sequence[T],
    seq2: collections.abc.Sequence[U],
    index1: int,
    index2: int,
) -> V:
    """Gets elements from two sequences and applies a binary function."""
    elem1 = sequence_getitem(seq1, index1)
    elem2 = sequence_getitem(seq2, index2)
    return call_binary_func(f, elem1, elem2)


def nested_function_application[T, U, V](
    outer_f: collections.abc.Callable[[T], collections.abc.Callable[[U], V]],
    inner_arg: U,
    outer_arg: T,
) -> V:
    """Applies a function that returns a function, then applies the result."""
    inner_f = call_func(outer_f, outer_arg)
    return call_func(inner_f, inner_arg)


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

    inferred_type1 = infer_return_type(sig1.bind(seq, index))

    inferred_type2 = infer_return_type(sig2.bind(Box(inferred_type1), key))

    inferred_type12 = infer_return_type(sig12.bind(seq, index, key))

    # check that the composed inference matches the direct inference
    assert isinstance(unify(inferred_type2, inferred_type12), collections.abc.Mapping)

    # check that the result of nested_type on the value of the composition unifies with the inferred type
    assert isinstance(
        unify(
            nested_type(sequence_mapping_getitem(seq, index, key)).value,
            inferred_type12,
        ),
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
        pytest.param(
            {"names": ["Alice", "Bob", "Charlie", "David"], "ages": [25, 30, 35, 40]},
            "names",
            3,
            marks=pytest.mark.xfail(
                reason="Heterogeneous TypedDict values can't bind V in Mapping[K,V]",
                raises=TypeError,
            ),
        ),
        (
            {"options": [[1, 2], [3, 4], [5, 6]], "choices": [[7], [8], [9]]},
            "options",
            2,
        ),
        # Deeply nested lists
        pytest.param(
            {"deep": [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], "shallow": [[9, 10]]},
            "deep",
            0,
            marks=pytest.mark.xfail(
                reason="Heterogeneous TypedDict values can't bind V in Mapping[K,V]",
                raises=TypeError,
            ),
        ),
    ],
)
def test_infer_composition_2(mapping, key, index):
    sig1 = inspect.signature(mapping_getitem)
    sig2 = inspect.signature(sequence_getitem)

    sig12 = inspect.signature(mapping_sequence_getitem)

    # First infer type of mapping_getitem(mapping, key) -> should be a sequence
    inferred_type1 = infer_return_type(sig1.bind(mapping, key))

    # Then infer type of sequence_getitem(result_from_step1, index) -> should be element type
    inferred_type2 = infer_return_type(sig2.bind(Box(inferred_type1), index))

    # Directly infer type of mapping_sequence_getitem(mapping, key, index)
    inferred_type12 = infer_return_type(sig12.bind(mapping, key, index))

    # The composed inference should match the direct inference
    assert isinstance(unify(inferred_type2, inferred_type12), collections.abc.Mapping)

    # check that the result of nested_type on the value of the composition unifies with the inferred type
    assert isinstance(
        unify(
            nested_type(mapping_sequence_getitem(mapping, key, index)).value,
            inferred_type12,
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
        ({1: "a", 2: "b"}, {3: "c"}, 1),
    ],
)
def test_get_from_constructed_sequence(a, b, index):
    """Test type inference through sequence construction and retrieval."""
    sig_construct = inspect.signature(sequence_from_pair)
    sig_getitem = inspect.signature(sequence_getitem)
    sig_composed = inspect.signature(get_from_constructed_sequence)

    # Infer type of sequence_from_pair(a, b) -> Sequence[T]
    inferred_sequence_type = infer_return_type(sig_construct.bind(a, b))

    # Infer type of sequence_getitem(sequence, index) -> T
    inferred_element_type = infer_return_type(
        sig_getitem.bind(Box(inferred_sequence_type), index)
    )

    # Directly infer type of get_from_constructed_sequence(a, b, index)
    direct_type = infer_return_type(sig_composed.bind(a, b, index))

    # The composed inference should match the direct inference
    assert isinstance(
        unify(inferred_element_type, direct_type), collections.abc.Mapping
    )

    # check that the result of nested_type on the value of the composition unifies with the inferred type
    assert isinstance(
        unify(
            nested_type(get_from_constructed_sequence(a, b, index)).value, direct_type
        ),
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
    inferred_mapping_type = infer_return_type(sig_construct.bind(key, value))

    # Infer type of mapping_getitem(mapping, lookup_key) -> V
    inferred_value_type = infer_return_type(
        sig_getitem.bind(Box(inferred_mapping_type), lookup_key)
    )

    # Directly infer type of get_from_constructed_mapping(key, value, lookup_key)
    direct_type = infer_return_type(sig_composed.bind(key, value, lookup_key))

    # The composed inference should match the direct inference
    assert isinstance(unify(inferred_value_type, direct_type), collections.abc.Mapping)

    # check that the result of nested_type on the value of the composition unifies with the inferred type
    assert isinstance(
        unify(
            nested_type(get_from_constructed_mapping(key, value, lookup_key)).value,
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
    map1_type = infer_return_type(sig_map.bind(key1, val1))

    # Step 2: Infer type of sequence containing these mappings
    # We need to unify the two mapping types first
    unified_map_type = map1_type  # Assuming they're compatible

    seq_type = infer_return_type(
        sig_seq.bind(Box(unified_map_type), Box(unified_map_type))
    )

    # Direct inference
    direct_type = infer_return_type(
        sig_composed.bind(
            key1,
            val1,
            key2,
            val2,
        )
    )

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
    nested_seq_type = infer_return_type(sig_nested.bind(k1, v1, v2, k2, v3, v4))
    # This should be Sequence[Mapping[K, Sequence[T]]]

    # Step 2: Get element from outer sequence
    mapping_type = infer_return_type(sig_seq_get.bind(Box(nested_seq_type), outer_idx))
    # This should be Mapping[K, Sequence[T]]

    # Step 3: Get sequence from mapping
    sequence_type = infer_return_type(sig_map_get.bind(Box(mapping_type), inner_key))
    # This should be Sequence[T]

    # Step 4: Get element from inner sequence
    composed_type = infer_return_type(sig_seq_get.bind(Box(sequence_type), inner_idx))
    # This should be T

    # Direct inference on the composed function
    direct_type = infer_return_type(
        sig_composed.bind(k1, v1, v2, k2, v3, v4, outer_idx, inner_key, inner_idx)
    )

    # The composed inference should match the direct inference
    assert isinstance(unify(composed_type, direct_type), collections.abc.Mapping)

    # check that the result of nested_type on the value of the composition unifies with the inferred type
    assert isinstance(
        unify(
            nested_type(
                double_nested_get(
                    k1, v1, v2, k2, v3, v4, outer_idx, inner_key, inner_idx
                )
            ).value,
            direct_type,
        ),
        collections.abc.Mapping,
    )


@pytest.mark.parametrize(
    "f,seq,index",
    [
        # Basic function applications
        (lambda x: x * 2, [1, 2, 3], 0),
        (lambda x: x * 2, [1, 2, 3], 2),
        (lambda x: x.upper(), ["hello", "world"], 1),
        (lambda x: len(x), ["a", "bb", "ccc"], 2),
        (lambda x: x + 1.0, [1.0, 2.0, 3.0], 1),
    ],
)
def test_apply_to_sequence_element(f, seq, index):
    """Test type inference through sequence access and function application."""
    sig_getitem = inspect.signature(sequence_getitem)
    sig_call = inspect.signature(call_func)
    sig_composed = inspect.signature(apply_to_sequence_element)

    # Step 1: Infer type of sequence_getitem(seq, index) -> T
    element_type = infer_return_type(sig_getitem.bind(seq, index))

    # Step 2: Infer type of call_func(f, element) -> U
    composed_type = infer_return_type(sig_call.bind(f, Box(element_type)))

    # Direct inference
    direct_type = infer_return_type(sig_composed.bind(f, seq, index))

    # The composed inference should match the direct inference
    assert isinstance(unify(composed_type, direct_type), collections.abc.Mapping)

    # check that the result of nested_type on the value of the composition unifies with the inferred type
    assert isinstance(
        unify(nested_type(apply_to_sequence_element(f, seq, index)).value, direct_type),
        collections.abc.Mapping,
    )


@pytest.mark.parametrize(
    "f,seq,index",
    [
        # Basic transformations
        (lambda x: x * 2, [1, 2, 3], 1),
        (lambda x: x.upper(), ["hello", "world"], 0),
        (lambda x: len(x), ["a", "bb", "ccc"], 2),
        (lambda x: x + 1, [10, 20, 30], 0),
    ],
)
def test_map_and_get(f, seq, index):
    """Test type inference through mapping and element retrieval."""
    sig_map = inspect.signature(map_sequence)
    sig_getitem = inspect.signature(sequence_getitem)
    sig_composed = inspect.signature(map_and_get)

    # Step 1: Infer type of map_sequence(f, seq) -> Sequence[U]
    mapped_type = infer_return_type(sig_map.bind(f, seq))

    # Step 2: Infer type of sequence_getitem(mapped_seq, index) -> U
    composed_type = infer_return_type(sig_getitem.bind(Box(mapped_type), index))

    # Direct inference
    direct_type = infer_return_type(sig_composed.bind(f, seq, index))

    # The composed inference should match the direct inference
    assert isinstance(unify(composed_type, direct_type), collections.abc.Mapping)

    # check that the result of nested_type on the value of the composition unifies with the inferred type
    assert isinstance(
        unify(nested_type(map_and_get(f, seq, index)).value, direct_type),
        collections.abc.Mapping,
    )


@pytest.mark.parametrize(
    "f,g,value",
    [
        # Basic function compositions
        (lambda x: x * 2, lambda x: x + 1, 5),
        (lambda x: str(x), lambda x: x.upper(), 42),
        (lambda x: len(x), lambda x: x * 2, "hello"),
        (lambda x: [x], lambda x: x[0], 1),
    ],
)
def test_compose_and_apply(f, g, value):
    """Test type inference through function composition and application."""
    sig_compose = inspect.signature(compose_mappings)
    sig_call = inspect.signature(call_func)
    sig_composed = inspect.signature(compose_and_apply)

    # Step 1: Infer type of compose_mappings(f, g) -> Callable[[T], V]
    composed_func_type = infer_return_type(sig_compose.bind(f, g))

    # Step 2: Infer type of call_func(composed, value) -> V
    result_type = infer_return_type(sig_call.bind(Box(composed_func_type), value))

    # Direct inference
    direct_type = infer_return_type(sig_composed.bind(f, g, value))

    # The composed inference should match the direct inference
    assert isinstance(unify(result_type, direct_type), collections.abc.Mapping)

    # check that the result of nested_type on the value of the composition unifies with the inferred type
    assert isinstance(
        unify(nested_type(compose_and_apply(f, g, value)).value, direct_type),
        collections.abc.Mapping,
    )


@pytest.mark.parametrize(
    "f,a,b,index",
    [
        # Basic constructions and applications
        (lambda x: x * 2, 1, 2, 0),
        (lambda x: x * 2, 1, 2, 1),
        (lambda x: x.upper(), "hello", "world", 0),
        (lambda x: len(x), "a", "bb", 1),
    ],
)
def test_construct_apply_and_get(f, a, b, index):
    """Test type inference through construction, application, and retrieval."""
    sig_construct = inspect.signature(sequence_from_pair)
    sig_apply = inspect.signature(apply_to_sequence_element)
    sig_composed = inspect.signature(construct_apply_and_get)

    # Step 1: Infer type of sequence_from_pair(a, b) -> Sequence[T]
    seq_type = infer_return_type(sig_construct.bind(a, b))

    # Step 2: Infer type of apply_to_sequence_element(f, seq, index) -> U
    composed_type = infer_return_type(
        sig_apply.bind(f, Box(seq_type), index),
    )

    # Direct inference
    direct_type = infer_return_type(
        sig_composed.bind(
            f,
            a,
            b,
            index,
        ),
    )

    # The composed inference should match the direct inference
    assert isinstance(unify(composed_type, direct_type), collections.abc.Mapping)

    # check that the result of nested_type on the value of the composition unifies with the inferred type
    assert isinstance(
        unify(nested_type(construct_apply_and_get(f, a, b, index)).value, direct_type),
        collections.abc.Mapping,
    )


@pytest.mark.parametrize(
    "f,seq,index1,index2",
    [
        # Basic binary operations
        (lambda x, y: x + y, [1, 2, 3], 0, 1),
        (lambda x, y: x + y, [1, 2, 3], 1, 2),
        (lambda x, y: x + y, ["hello", "world", "test"], 0, 2),
        (lambda x, y: x * y, [2, 3, 4], 0, 2),
    ],
)
def test_binary_on_sequence_elements(f, seq, index1, index2):
    """Test type inference through sequence access and binary function application."""
    sig_getitem = inspect.signature(sequence_getitem)
    sig_call_binary = inspect.signature(call_binary_func)
    sig_composed = inspect.signature(binary_on_sequence_elements)

    # Step 1: Infer types of sequence_getitem calls
    elem1_type = infer_return_type(sig_getitem.bind(seq, index1))

    elem2_type = infer_return_type(sig_getitem.bind(seq, index2))

    # Step 2: Infer type of call_binary_func(f, elem1, elem2) -> V
    composed_type = infer_return_type(
        sig_call_binary.bind(f, Box(elem1_type), Box(elem2_type)),
    )

    # Direct inference
    direct_type = infer_return_type(
        sig_composed.bind(
            f,
            seq,
            index1,
            index2,
        ),
    )

    # The composed inference should match the direct inference
    assert isinstance(unify(composed_type, direct_type), collections.abc.Mapping)

    # check that the result of nested_type on the value of the composition unifies with the inferred type
    assert isinstance(
        unify(
            nested_type(binary_on_sequence_elements(f, seq, index1, index2)).value,
            direct_type,
        ),
        collections.abc.Mapping,
    )


def test_unify_generic_typeddict_extracts_typevar():
    """unify(Datum[T], nested_type(dict).value) should return {T: int}."""

    class Datum[T](typing.TypedDict):
        name: str
        value: T

    result = unify(Datum[T], nested_type({"name": "a", "value": 1}).value)
    assert result == {T: int}


# --- TypedDict <-> Mapping subtyping ---


def test_unify_mapping_typeddict_subclass():
    """TypedDict unifies with Mapping/MutableMapping via subclass check."""

    class Info(typing.TypedDict):
        name: str
        age: int

    # TypedDict is a subclass of MutableMapping, so this succeeds
    subs = unify(collections.abc.MutableMapping, Info)
    assert subs == {}

    subs = unify(collections.abc.Mapping, Info)
    assert subs == {}


# ============================================================
# typing.Self resolution tests
# ============================================================


# --- SelfTypeReplacer ---


@pytest.mark.parametrize("tv_name", ["Self", "X"])
@pytest.mark.parametrize(
    "typ,expected_factory",
    [
        # bare Self → tv
        (typing.Self, lambda tv: tv),
        # Self nested in generics → tv nested in same generics
        (list[typing.Self], lambda tv: list[tv]),  # type: ignore[misc,valid-type]
        (dict[str, typing.Self], lambda tv: dict[str, tv]),  # type: ignore[misc,valid-type]
        (
            collections.abc.Callable[[typing.Self], int],
            lambda tv: collections.abc.Callable[[tv], int],
        ),
        (type[typing.Self], lambda tv: type[tv]),  # type: ignore[misc,valid-type]
        # Union containing Self
        (typing.Self | None, lambda tv: tv | None),
        # Types without Self pass through unchanged
        (int, lambda _tv: int),
        (list[int], lambda _tv: list[int]),
        (T, lambda _tv: T),
        (list[T], lambda _tv: list[T]),
    ],
)
def test_self_type_replacer(typ, expected_factory, tv_name):
    """SelfTypeReplacer rewrites every typing.Self occurrence to the given TypeVar."""
    from effectful.internals.unification import SelfTypeReplacer

    tv = typing.TypeVar(tv_name)  # type: ignore[misc]
    replacer = SelfTypeReplacer(tv)
    assert replacer.evaluate(typ) == expected_factory(tv)


# --- chaining two signatures with Self from different "classes" ---


def test_chained_self_signatures():
    """Two infer_return_type calls don't conflate Self across classes."""
    sig_a = inspect.signature(_Foo.return_self)
    sig_b = inspect.signature(_Bar.return_self)

    assert canonicalize(infer_return_type(sig_a.bind(Box(_Foo)))) == _Foo
    assert canonicalize(infer_return_type(sig_b.bind(Box(_Bar)))) == _Bar


# --- classmethod with Self: cls is stripped, Self stays unresolved ---


def test_classmethod_self_not_resolved():
    """Classmethod Self stays unresolved when cls is stripped.

    ``inspect.signature`` strips ``cls``, so ``from_config(config: int) -> Self``
    has no parameter that binds Self. SelfTypeReplacer produces a fresh
    TypeVar; nothing in unify binds it; the TypeVar leaks through the result.
    """
    sig = inspect.signature(_Foo.from_config)
    result = infer_return_type(sig.bind(Box(int)))
    assert isinstance(result, typing.TypeVar)
    assert result.__name__ == "Self"


# --- composition tests with Self ---


@pytest.mark.parametrize("obj_type", [_Foo, _Bar, int, str])
def test_infer_self_composition_1(obj_type):
    """Compose return_list_self -> get_first, verify matches return_self.

    Step 1: return_list_self(obj) -> list[Self]  (Self method)
    Step 2: get_first(list[T]) -> T              (generic function)
    Direct: return_self(obj) -> Self
    """
    sig1 = inspect.signature(_Foo.return_list_self)
    sig2 = inspect.signature(get_first)
    sig_direct = inspect.signature(_Foo.return_self)

    # Step 1: infer list[Self] with Self bound to obj_type
    inferred_type1 = infer_return_type(sig1.bind(Box(obj_type)))

    # Step 2: get_first(list[obj_type]) -> obj_type
    inferred_type2 = infer_return_type(sig2.bind(Box(inferred_type1)))

    # Direct: return_self(obj_type) -> obj_type
    inferred_direct = infer_return_type(sig_direct.bind(Box(obj_type)))

    # The composed inference should match the direct inference
    assert isinstance(unify(inferred_type2, inferred_direct), collections.abc.Mapping)


@pytest.mark.parametrize("obj_type", [_Foo, _Bar, int, str])
def test_infer_self_composition_2(obj_type):
    """Compose identity -> return_list_self, verify matches wrap_in_list.

    Step 1: identity(x: T) -> T                  (generic function)
    Step 2: return_list_self(self) -> list[Self]  (Self method)
    Direct: wrap_in_list(x: T) -> list[T]
    """
    sig1 = inspect.signature(identity)
    sig2 = inspect.signature(_Foo.return_list_self)
    sig_direct = inspect.signature(wrap_in_list)

    inferred_type1 = infer_return_type(sig1.bind(Box(obj_type)))
    inferred_type2 = infer_return_type(sig2.bind(Box(inferred_type1)))
    inferred_direct = infer_return_type(sig_direct.bind(Box(obj_type)))

    assert isinstance(unify(inferred_type2, inferred_direct), collections.abc.Mapping)


def test_infer_return_type_direct_simple():
    """infer_return_type produces the substituted return type.

    Inputs are Boxed types per the convention used by ``__type_rule__``.
    """

    def f(x: T) -> list[T]: ...  # type: ignore[valid-type]

    sig = inspect.signature(f)
    result = infer_return_type(sig.bind(Box(int)))
    assert canonicalize(result) == canonicalize(list[int])


def test_infer_return_type_direct_self():
    """infer_return_type resolves typing.Self via the new TypedDict path.

    typing.Self only resolves when ``self`` is annotated explicitly: there is
    no implicit self-injection (it would be unsafe for class/static/free
    functions where the first positional is not the receiver).
    """

    class C:
        def m(self: typing.Self) -> typing.Self: ...

    sig = inspect.signature(C.m)
    result = infer_return_type(sig.bind(C()))
    assert canonicalize(result) == C


def test_infer_return_type_bare_self_unresolved():
    """A method with bare ``self`` does not bind Self — it stays a free TypeVar."""

    class C:
        def m(self) -> typing.Self: ...

    sig = inspect.signature(C.m)
    result = infer_return_type(sig.bind(C()))
    assert isinstance(result, typing.TypeVar)
    assert result.__name__ == "Self"


def test_infer_return_type_direct_default_omitted():
    """infer_return_type with a default-valued param omitted still returns correctly."""

    def f(x: T, y: str = "ok") -> T: ...  # type: ignore[valid-type]

    sig = inspect.signature(f)
    result = infer_return_type(sig.bind(Box(int)))
    assert canonicalize(result) == int


# --- TypedDict-valued parameters in signatures ---


def test_infer_return_type_typeddict_param():
    """A function parameter typed as a TypedDict unifies with a runtime
    str-keyed dict whose nested_type is a structurally-compatible TypedDict.

    Composes the sig→TypedDict conversion with the TypedDict-vs-TypedDict
    unification path added in PR 651.
    """

    class User(typing.TypedDict):
        name: str
        age: int

    def greet(u: User) -> str: ...

    sig = inspect.signature(greet)
    # nested_type({"name": "a", "age": 1}) returns a TypedDict, which is
    # structurally compatible with User → unification should succeed.
    result = infer_return_type(sig.bind({"name": "a", "age": 1}))
    assert canonicalize(result) is str


def test_infer_return_type_generic_typeddict_param():
    """A parameterized TypedDict in a function parameter binds the TypeVar
    from the runtime dict's value types."""

    class Datum[T](typing.TypedDict):
        name: str
        value: T

    def extract[T](d: Datum[T]) -> T: ...

    sig = inspect.signature(extract)
    result = infer_return_type(sig.bind({"name": "a", "value": 42}))
    assert canonicalize(result) is int


# --- Failure cases ---


def test_infer_return_type_conflicting_self_bindings():
    """A method with two parameters both annotated typing.Self must reject
    callers that pass differently-typed receivers for those slots.

    Without explicit Self binding, this would silently widen Self to a common
    supertype; with the SelfTypeReplacer pattern, both Self occurrences alias
    the same fresh TypeVar, so conflicting concrete types must raise.
    """

    class C:
        def f(self: typing.Self, other: typing.Self) -> typing.Self: ...

    sig = inspect.signature(C.f)
    # Same type for both: succeeds.
    assert canonicalize(infer_return_type(sig.bind(Box(_Foo), Box(_Foo)))) == _Foo
    # Mismatched concrete types: must raise.
    with pytest.raises(TypeError):
        infer_return_type(sig.bind(Box(_Foo), Box(_Bar)))


def test_infer_return_type_wrong_arg_type():
    """Passing an arg whose type is incompatible with the parameter
    annotation must raise."""

    def f(x: int) -> int: ...

    sig = inspect.signature(f)
    with pytest.raises(TypeError):
        infer_return_type(sig.bind(Box(str)))


def test_infer_return_type_missing_required_arg():
    """Omitting a required positional arg surfaces as a Signature.bind error
    before we reach infer_return_type — document that contract here."""

    def f(x: int, y: int) -> int: ...

    sig = inspect.signature(f)
    with pytest.raises(TypeError):
        sig.bind(Box(int))  # missing `y`
