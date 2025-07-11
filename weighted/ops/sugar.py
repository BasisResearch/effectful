from effectful.ops.types import Term

from weighted.ops.fold import fold
from weighted.ops.semiring import ArgMaxAlg, ArgMinAlg, LinAlg, LogAlg, MaxAlg, MinAlg


def Exp(streams, body):
    # TODO
    raise NotImplementedError()


def ArgMin(streams, body):
    result = fold(ArgMinAlg, streams, body)
    assert isinstance(result, Term) or (isinstance(result, tuple) and len(result) == 2)
    return result


def ArgMax(streams, body):
    result = fold(ArgMaxAlg, streams, body)
    assert isinstance(result, Term) or (isinstance(result, tuple) and len(result) == 2)
    return result


def Min(streams, body):
    return fold(MinAlg, streams, body)


def Max(streams, body):
    return fold(MaxAlg, streams, body)


def Sum(streams, body):
    return fold(LinAlg, streams, body)


def LogSum(streams, body):
    return fold(LogAlg, streams, body)
