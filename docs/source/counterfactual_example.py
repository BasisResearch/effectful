import math
from typing import (
    Callable,
    Concatenate,
    Mapping,
    NamedTuple,
    ParamSpec,
    Protocol,
    TypedDict,
    TypeVar,
)

from effectful.ops.semantics import evaluate, fvsof, fwd, handler, product, typeof
from effectful.ops.syntax import Scoped, deffn, defop
from effectful.ops.types import Interpretation, Operation, Term

P = ParamSpec("P")
Q = ParamSpec("Q")
S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")


class Distribution(Protocol[T]):
    def rand(self) -> T: ...
    def pdf(self, value: T) -> float: ...


class Bernoulli(NamedTuple):
    prob: float

    def rand(self) -> bool:
        raise NotImplementedError

    def pdf(self) -> float:
        raise NotImplementedError


class Normal(NamedTuple):
    loc: float
    scale: float

    def rand(self) -> float:
        raise NotImplementedError

    def pdf(self, value: float) -> float:
        raise NotImplementedError


@defop
def app(fn: Callable[P, T], *args: P.args, **_: P.kwargs) -> T:
    raise NotImplementedError


@defop
def sample(dist: Distribution[T]) -> T:
    raise NotImplementedError


@defop
def factor(weight: float) -> float:
    raise NotImplementedError


@defop
def intervene(obs: T, act: T, name: Operation[[], bool]) -> T:
    raise NotImplementedError


def substitute(subs: Mapping[Operation[[], T], T]) -> Callable[[Term[S]], Term[S]]:

    def thunk(val: T) -> Callable[[], T]:
        return lambda: val

    return handler({var: thunk(subs[var]) for var in subs})(evaluate)


def list_sample(dist: Distribution[T]) -> T:
    intervention_points = fvsof(dist) - {sample, factor, intervene, app}
    if len(intervention_points) > 0:
        fv = list(intervention_points)[0]
        return intervene(
            sample(substitute({fv: 0})(dist)), sample(substitute({fv: 1})(dist)), fv
        )
    else:
        return fwd()


def list_factor(weight: float) -> float:
    intervention_points = fvsof(weight) - {sample, factor, intervene, app}
    if len(intervention_points) > 0:
        fv = list(intervention_points)[0]
        return app(
            lambda a, b: a * b,
            factor(substitute({fv: lambda: 0})(weight)),
            factor(substitute({fv: lambda: 1})(weight)),
        )
    else:
        return fwd()


def list_app(fn: Callable[P, T], *args: P.args, **_: P.kwargs) -> T:
    intervention_points = fvsof(args) - {sample, factor, intervene, app}
    if len(intervention_points) > 0:
        fv = list(intervention_points)[0]
        return intervene(
            app(fn, *substitute({fv: 0})(args)), app(fn, *substitute({fv: 1})(args)), fv
        )
    else:
        return fwd()


NestedList: Interpretation = {
    sample: list_sample,
    factor: list_factor,
    app: list_app,
}


def mwc_intervene(obs: T, act: T, name: Operation[[], bool]) -> T:
    branch: bool | Term[bool] = name()
    if not isinstance(branch, Term):
        return act if branch else obs
    else:
        return fwd()


MultiWorldCounterfactual: Interpretation = {
    intervene: mwc_intervene,
}


def model(i_x: Operation[[], bool], i_y: Operation[[], bool]) -> tuple[float, float]:
    x = sample(Bernoulli(0.5))

    x = intervene(x, False, i_x)

    y_loc = app(lambda x: 0.0 if x else 1.0, x)

    y_loc = intervene(y_loc, 1.5, i_y)

    y = sample(Normal(y_loc, 1.0))

    y_11 = substitute({i_x: 1, i_y: 1})(y)
    y_00 = substitute({i_x: 0, i_y: 0})(y)

    w00 = factor(app(lambda a, b: math.exp(-abs(a - b) / 0.5), y_00, 1.5))

    return w00, app(lambda a, b: a - b, y_11, y_00)


i_x = defop(bool, name="i_x")
i_y = defop(bool, name="i_y")

assert {intervene, sample, factor, app} == fvsof(model(i_x, i_y))

with handler(NestedList), handler(MultiWorldCounterfactual):
    print(model(i_x, i_y))

    assert {sample, factor, app} == fvsof(model(i_x, i_y))

    assert not {intervene, i_x, i_y} <= fvsof(model(i_x, i_y))
