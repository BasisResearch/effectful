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
from effectful.ops.weighted.reduce import reduce


def Exp(streams, body):
    raise NotImplementedError


def ArgMin(streams, body):
    result = reduce(ArgMinMonoid, streams, body)
    assert isinstance(result, Term) or (isinstance(result, tuple) and len(result) == 2)
    return result


def ArgMax(streams, body):
    result = reduce(ArgMaxMonoid, streams, body)
    assert isinstance(result, Term) or (isinstance(result, tuple) and len(result) == 2)
    return result


def Min(streams, body):
    return reduce(MinMonoid, streams, body)


def Max(streams, body):
    return reduce(MaxMonoid, streams, body)


def Sum(streams, body):
    return reduce(SumMonoid, streams, body)


def Prod(streams, body):
    return reduce(ProdMonoid, streams, body)


def LogSum(streams, body):
    return reduce(LogSumMonoid, streams, body)


def CartesianProd(streams, body):
    return reduce(JaxCartesianProdMonoid, streams, body)
