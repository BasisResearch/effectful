from effectful.ops.types import Term
from effectful.ops.weighted.monoid import (
    ArgMaxMonoid,
    ArgMinMonoid,
    JaxCartesianProdMonoid,
    LogSumMonoid,
    MaxMonoid,
    MinMonoid,
    ProdMonoid,
    SumMonoid,
)


def Exp(streams, body):
    raise NotImplementedError


def ArgMin(streams, body):
    result = ArgMinMonoid.reduce(streams, body)
    assert isinstance(result, Term) or (isinstance(result, tuple) and len(result) == 2)
    return result


def ArgMax(streams, body):
    result = ArgMaxMonoid.reduce(streams, body)
    assert isinstance(result, Term) or (isinstance(result, tuple) and len(result) == 2)
    return result


def Min(streams, body):
    return MinMonoid.reduce(streams, body)


def Max(streams, body):
    return MaxMonoid.reduce(streams, body)


def Sum(streams, body):
    return SumMonoid.reduce(streams, body)


def Prod(streams, body):
    return ProdMonoid.reduce(streams, body)


def LogSum(streams, body):
    return LogSumMonoid.reduce(streams, body)


def CartesianProd(streams, body):
    return JaxCartesianProdMonoid.reduce(streams, body)
