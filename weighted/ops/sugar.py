from effectful.ops.types import Term

from weighted.ops.fold import fold
from weighted.ops.monoid import (
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
    # TODO
    raise NotImplementedError()


def ArgMin(streams, body):
    result = fold(ArgMinMonoid, streams, body)
    assert isinstance(result, Term) or (isinstance(result, tuple) and len(result) == 2)
    return result


def ArgMax(streams, body):
    result = fold(ArgMaxMonoid, streams, body)
    assert isinstance(result, Term) or (isinstance(result, tuple) and len(result) == 2)
    return result


def Min(streams, body):
    return fold(MinMonoid, streams, body)


def Max(streams, body):
    return fold(MaxMonoid, streams, body)


def Sum(streams, body):
    return fold(SumMonoid, streams, body)


def Prod(streams, body):
    return fold(ProdMonoid, streams, body)


def LogSum(streams, body):
    return fold(LogSumMonoid, streams, body)


def CartesianProd(streams, body):
    return fold(JaxCartesianProdMonoid, streams, body)
