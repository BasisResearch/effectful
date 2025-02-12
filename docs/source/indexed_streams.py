import collections.abc
import dataclasses
import functools
import itertools
import operator
import tree
from typing import Annotated, Callable, Concatenate, Generic, Literal, ParamSpec, TypeVar

import effectful.handlers.numbers  # noqa: F401
from effectful.ops.semantics import apply, call, coproduct, evaluate, fwd, handler, product, typeof
from effectful.ops.syntax import Scoped, defop, defdata
from effectful.ops.types import Interpretation, Operation, Term

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
A = TypeVar("A")
B = TypeVar("B")


@defop
def index() -> K:
    raise NotImplementedError


@defop
def value() -> V:
    raise NotImplementedError


@defop
def ready() -> bool:
    raise NotImplementedError


@defop
def skip(ind, is_ready: bool) -> bool:
    raise NotImplementedError


Stream = Interpretation[K, V]


@dataclasses.dataclass
class Semiring(Generic[T]):
    add: Callable[[T, T], T]
    mul: Callable[[T, T], T]
    zero: T
    one: T


def stream_mul(semiring: Semiring[V], s1: Stream[K, V], s2: Stream[K, V]) -> Stream[K, V]:

    # # coproduct version
    # return coproduct({
    #     index: lambda: max(s1[index](), fwd()),
    #     value: lambda: semiring.mul(s1[value](), fwd()),
    #     ready: lambda: s1[ready]() and fwd() and s1[index]() == index(),
    #     skip: lambda ind, is_ready: s1[skip](ind, is_ready) & fwd(),
    # }, s2)
    return {
        index: lambda: max(s1[index](), s2[index]()),
        value: lambda: semiring.mul(s1[value](), s2[value]()),
        ready: lambda: s1[ready]() and s2[ready]() and s1[index]() == s2[index](),
        skip: lambda ind, is_ready: s1[skip](ind, is_ready) & s2[skip](ind, is_ready),
    }


def stream_add(semiring: Semiring[V], s1: Stream[K, V], s2: Stream[K, V]) -> Stream[K, V]:
    return {
        index: lambda: min(s1[index](), s2[index]()),
        value: lambda: semiring.add(s1[value](), s2[value]()),
        ready: lambda: s1[ready]() and s2[ready]() and s1[index]() == s2[index](),
        skip: lambda ind, is_ready: s1[skip](ind, is_ready) & s2[skip](ind, is_ready),
    }


def stream_contract(s: Stream[K, V]) -> Stream[K, V]:
    # TODO support nested streams
    return {
        index: lambda: None,
        value: s[value],
        ready: s[ready],
        skip: lambda _, is_ready: s[skip](s[index](), is_ready),
    }


def stream_expand(s: Stream[K, V]) -> Stream[K, V]:
    # TODO implement
    return {
        index: ...,
        value: ...,
        ready: ...,
        skip: ...,
    }


def stream_map(f: Callable[[T], V], s: Stream[K, T]) -> Stream[K, V]:
    # TODO generalize for nested streams
    return {
        index: s[index],
        value: lambda: f(s[value]()),
        ready: s[ready],
        skip: s[skip],
    }


def stream_eval(semiring: Semiring[V], s: Stream[K, V]) -> V | dict[K, V]:
    # {index(s): value(s) for s in reduce(skip, s) if ready(s)}
    with handler(s):
        res = semiring.zero if index() is None else {}

        while True:
            if ready():
                if index() is None:
                    res = semiring.add(res, value())
                else:
                    res[index()] = semiring.add(res.get(index(), semiring.zero), value())
            if skip(index(), ready()):
                break

    return res


def Dense(vals: collections.abc.Sequence[V]) -> Stream[int, V]:

    i = 0

    def stream_index():
        return i

    def stream_value():
        return vals[i]

    def stream_ready():
        return i < len(vals)

    def stream_skip(ind, is_ready):
        nonlocal i
        i = max(i, ind + (1 if is_ready else 0))
        return i >= len(vals)

    return {
        index: stream_index,
        value: stream_value,
        ready: stream_ready,
        skip: stream_skip,
    }


def Sparse(vals: collections.abc.Mapping[int, V]) -> Stream[int, V]:
    keys = sorted(vals.keys())
    q = 0

    def stream_index():
        return keys[q]

    def stream_value():
        return vals[q]

    def stream_ready():
        return q < len(vals)

    def stream_skip(ind, is_ready):
        nonlocal q
        while q < ind + (1 if is_ready else 0):
            q += 1
        return q >= len(vals)

    return {
        index: stream_index,
        value: stream_value,
        ready: stream_ready,
        skip: stream_skip,
    }


# unfold == product
# filter == fst -> coproduct
# fmap == snd -> coproduct
# foldl == bnf -> product -> reduce

# expr = add(add(x(), 1), y())
#  --> anf_expr = product(
#   {v1: defterm(lambda: add(x(), 1))},
#   {v2: defterm(lambda: add(v1, y()))}
#  )
# handler(product({x: ..., y: ...}, anf_expr))(v2)() == evaluate(expr)
# product(
#   product({x: ..., y: ...}, {v1: defterm(lambda: add(x(), 1))}),
#   product({x: ..., y: ...}, {v2: defterm(lambda: add(v1(), y()))})
#   )
#  ==
# product(
#   product({x: ...}, {v1: defterm(lambda: add(x(), 1))})
#   product({y: ...}, {v2: defterm(lambda: add(v1(), y()))})
#  )


@defop
def unfolds(
    bindings: Interpretation,
    bodies: Interpretation,
) -> Interpretation:    
    return product(bindings, bodies)


@defop
def fmaps(
    mappings: Interpretation,
    bodies: Interpretation,
) -> Interpretation:
    return coproduct(mappings, bodies)


@defop
def folds(
    # semiring: Semiring[T],
    streams: Interpretation,
    value: V,  # Callable[P, T]
    # weight: Callable[[V], T],
    # weight: T | None = None
) -> V:  # tuple[V, T]:  # Callable[P, V]:
    # handler(unfolds(streams, fmaps({weight_op: ...}, {value_op: ...}))(value_op)()
    env, res_op = bnf(value, *streams.keys())
    return handler(unfolds(streams, env))(res_op)()


def bnf(term: Term[T], *ops: Operation) -> tuple[Interpretation, Operation[[], T]]:
    env = {}

    def _apply(_, op, *args, **kwargs):
        tm = defdata(op, *args, **kwargs)
        if op in ops:
            var = defop(typeof(tm))
            env[var] = tm
            return var()
        else:
            return tm

    with handler({apply: _apply}):
        res = evaluate(term)

    res_op = defop(typeof(term))
    env[res_op] = res
    return env, res_op


@defop
def unfold(
    body: T,  # body: Interpretation
    indices: Interpretation,  # [S, collections.abc.Iterable[S]]
) -> T:  # collections.abc.Iterable[T]:
    if isinstance(body, Term) and body.op == unfold:
        return unfold(body.args[0], coproduct(body.args[1], indices))
    elif isinstance(body, Term) and body.op == filter:
        ...
    else:
        raise NotImplementedError


# i = defop(int)
# i_ = 0
# def dense_i():
#   nonlocal i_
#   i_ = skip(i_)
#   return i_
# index_ = {i: dense_i}
# value = unfold(A[i()], index_)

@defop
def filter(
    body: T,
    guard: bool
) -> T:  # collections.abc.Iterable[T]:
    if isinstance(body, Term) and body.op == filter:
        return filter(body.args[0], body.args[1] and guard)
    elif isinstance(body, Term) and body.op == unfold:
        ...
    else:
        raise NotImplementedError


@defop
def fmap(
    fn: Callable[P, T],
    *args: P.args,
    **_: P.kwargs
) -> T:
    # e.g. stream_add == fmap(operator.add, x, y)
    # TODO distribute over unfold
    for i, arg in enumerate(args):
        if isinstance(arg, Term) and arg.op == unfold:
            ...
        elif isinstance(arg, Term) and arg.op == filter:
            ...
        else:
            ...
            continue

    return call(fn, *args, **_)


def ranges(sizes: Mapping[Operation[[], int], int]) -> collections.abc.Iterable[Interpretation[int, int]]:
    range_iters = itertools.product(*(range(sizes[op]) for op in sizes))
    for range_it in range_iters:
        yield {op: functools.partial(lambda x: x, val) for op, val in zip(sizes, range_it)}


Expectation(
    f(x)
    for z1 in sample(z1_dist)
    for z2 in sample(z2_dist(z1))
    for x in sample(x_dist(z1, z2))
)


# unnormalized
Expectation(
    weight * vars[-1]
    for (weight, vars) in Infer(
        (w1(z1) * w2(z1, z2) * w3(z1, z2, x), (z1, z2, x))
        for z1 in sample(z1_dist)
        # if factor(w1(z1)) != 0
        for z2 in sample(z2_dist(z1))
        # if factor(w2(z1, z2)) != 0
        for x in sample(x_dist(z1, z2))
        # if factor(w3(z1, z2, x)) != 0
    )
)


@defop
def foldl(
    # base_index_state: Interpretation[S, collections.abc.Iterable[S]],
    base_index_states: collections.abc.Iterable[Interpretation],
    body: T,
    *,
    kernel: Callable[[V, T], V] = lambda aa, b: (a for a in (*aa, b))
) -> V:

    if isinstance(body, Term):
        if body.op == unfold:
            return foldl(product(base_index_state, body.args[1]), body.args[0], kernel=kernel)

        return functools.reduce(kernel, (
            b
            for (a, k) in (
                tree.unflatten_as((body.args, body.kwargs), it)
                for it in itertools.product(*(
                    i if isinstance(i, collections.abc.Iterable) else (i,) 
                    for i in tree.flatten((body.args, body.kwargs))
                ))
            )
            for b in handler(base_index_state)(body.op)(*a, **k)
        ))
    elif tree.is_nested(body):
        return tree.map_structure(functools.partial(foldl, base_index_state, kernel=kernel), body)
    else:
        return body


def PointwiseFunctionSemiring(semiring: Semiring[T]) -> Semiring[Callable[..., T]]:
    return Semiring(
        add = lambda f, g: lambda *a, **k: semiring.add(f(*a, **k), g(*a, **k)),
        mul = lambda f, g: lambda *a, **k: semiring.mul(f(*a, **k), g(*a, **k)),
        zero = lambda *_, **__: semiring.zero,
        one = lambda *_, **__: semiring.one,
    )


@defop
def Sum(
    semiring: Annotated[Semiring[T], Scoped[A]],
    body: Annotated[T, Scoped[A | B]],
    *vars: Annotated[Operation[[], int], Scoped[B]]
) -> Annotated[T, Scoped[A]]:

    result = semiring.zero
    for vals in itertools.product(*(range(2) for _ in vars)):
        subs = {var: functools.partial(lambda x: x, val) for var, val in zip(vars, vals)}
        result = semiring.add(result, handler(subs)(evaluate)(body))
    return result


if __name__ == "__main__":

    LinAlg: Semiring[float] = Semiring(operator.add, operator.mul, 0., 1.)

    print(stream_eval(LinAlg, stream_add(LinAlg, Dense([1, 2, 3]), Dense([4, 5, 6]))))

    print(stream_eval(LinAlg, stream_contract(stream_add(LinAlg, Dense([1, 2, 3]), Dense([4, 5, 6])))))

    x, y = defop(int), defop(int)
    print(Sum(LinAlg, x() * y() ** 2, x, y))
