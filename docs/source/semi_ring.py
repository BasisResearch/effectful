import collections.abc
import operator
import types
from typing import Annotated, ParamSpec, TypeVar, cast, overload

import effectful.handlers.numbers  # noqa: F401
from effectful.ops.semantics import coproduct, evaluate, fwd, handler
from effectful.ops.syntax import Scoped, defop
from effectful.ops.types import Interpretation, Operation, Term

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
A = TypeVar("A")
B = TypeVar("B")


# https://stackoverflow.com/questions/2703599/what-would-a-frozen-dict-be
class SemiRingDict(collections.abc.Mapping[K, V]):
    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)
        self._hash = None

    def __iter__(self):
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __getitem__(self, key: K) -> V:
        return self._d[key]

    def __hash__(self) -> int:
        if self._hash is None:
            hash_ = 0
            for pair in self.items():
                hash_ ^= hash(pair)
            self._hash = hash_
        return self._hash

    def __add__(self, other: "SemiRingDict[K, V]") -> "SemiRingDict[K, V]":
        new_dict = self._d.copy()
        for key, value in other.items():
            if key in new_dict:
                new_dict[key] += value
            else:
                new_dict[key] = value
        return SemiRingDict(new_dict)


@defop
def Sum(
    e1: SemiRingDict[K, V],
    k: Annotated[Operation[[], K], Scoped[A]],
    v: Annotated[Operation[[], V], Scoped[A]],
    e2: Annotated[SemiRingDict[S, T], Scoped[A]],
) -> SemiRingDict[S, T]:
    raise NotImplementedError


@defop
def Let(
    e1: Annotated[T, Scoped[A]],
    x: Annotated[Operation[[], T], Scoped[B]],
    e2: Annotated[S, Scoped[B]],
) -> Annotated[S, Scoped[A]]:
    raise NotImplementedError


@defop
def Record(**kwargs: T) -> collections.abc.Mapping[str, T]:
    raise NotImplementedError


@defop
def Field(record: collections.abc.Mapping[str, T], key: str) -> T:
    raise NotImplementedError


@defop
def Dict(*contents: tuple[K, V]) -> SemiRingDict[K, V]:
    raise NotImplementedError


@defop
def add(x: T, y: T) -> T:
    if not any(isinstance(a, Term) for a in (x, y)):
        return operator.add(x, y)
    else:
        raise NotImplementedError


ops = types.SimpleNamespace()
ops.Sum = Sum
ops.Let = Let
ops.Record = Record
ops.Dict = Dict
ops.Field = Field


def eager_dict(*contents: tuple[K, V]) -> SemiRingDict[K, V]:
    if not any(isinstance(v, Term) for kv in contents for v in kv):
        return SemiRingDict(list(contents))
    else:
        return fwd()


def eager_record(**kwargs: T) -> collections.abc.Mapping[str, T]:
    if not any(isinstance(v, Term) for v in kwargs.values()):
        return dict(**kwargs)
    else:
        return fwd()


@overload
def eager_add(x: int, y: int) -> int: ...


@overload
def eager_add(x: SemiRingDict[K, V], y: SemiRingDict[K, V]) -> SemiRingDict[K, V]: ...


def eager_add(x, y):
    if isinstance(x, SemiRingDict) and isinstance(y, SemiRingDict):
        new_dict = x._d.copy()
        for key, value in y.items():
            if key in new_dict:
                new_dict[key] += value
            else:
                new_dict[key] = value
        return SemiRingDict(new_dict)
    elif isinstance(x, int) and isinstance(y, int):
        return x + y
    else:
        return fwd()


def eager_field(r: dict[str, T], k: str) -> T:
    match r, k:
        case dict(), str():
            return r[k]
        case SemiRingDict(), _ if not isinstance(k, Term):
            return r[k]
        case _:
            return fwd()


def eager_sum(
    e1: SemiRingDict[K, V],
    k: Operation[[], K],
    v: Operation[[], V],
    e2: SemiRingDict[S, T],
) -> SemiRingDict[S, T]:
    match e1, e2:
        case SemiRingDict(), Term():
            new_d: SemiRingDict[S, T] = SemiRingDict()
            for key, value in e1.items():
                new_d += handler({k: lambda: key, v: lambda: value})(evaluate)(e2)  # type: ignore
            return new_d
        case SemiRingDict(), SemiRingDict():
            new_d = SemiRingDict()
            for _ in e1.items():
                new_d += e2
            return new_d
        case _:
            return fwd()


def eager_let(e1: T, x: Operation[[], T], e2: S) -> S:
    return cast(S, handler({x: lambda: e1})(evaluate)(e2))


def vertical_fusion(e1: T, x: Operation[[], T], e2: S) -> S:
    match e1, e2:
        case (
            Term(ops.Sum, (e_sum, k1, v1, Term(ops.Dict, (Term(k1a), e_lhs)))),
            Term(ops.Sum, (Term(xa), k2, v2, Term(ops.Dict, (Term(k2a), e_rhs)))),
        ) if (
            x == xa and k1 == k1a and k2 == k2a
        ):
            return evaluate(
                Sum(
                    e_sum,  # type: ignore
                    k1,  # type: ignore
                    v1,  # type: ignore
                    Let(
                        e_lhs, v2, Let(k1(), k2, Dict(k2(), Let(e_lhs, k2, e_rhs)))  # type: ignore
                    ),
                )
            )
        case _:
            return fwd()


eager: Interpretation = {
    add: eager_add,
    Dict: eager_dict,
    Record: eager_record,
    Sum: eager_sum,
    Field: eager_field,
    Let: eager_let,
}

opt: Interpretation = {
    Let: vertical_fusion,
}


if __name__ == "__main__":
    x, y, k, v = (
        defop(SemiRingDict[int, int], name="x"),
        defop(SemiRingDict[int, int], name="y"),
        defop(int, name="k"),
        defop(int, name="v"),
    )

    term: SemiRingDict[int, int] = Let(
        Sum(x(), k, v, Dict((k(), v() + 1))), y, Sum(y(), k, v, Dict((k(), v() + 1)))
    )

    print("Without optimization:", term)
    with handler(coproduct(eager, opt)):
        print("With optimization:", evaluate(term))
