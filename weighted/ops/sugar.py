from weighted.ops.fold import fold
from weighted.ops.semiring import ArgMaxAlg, ArgMinAlg, LinAlg, MaxAlg, MinAlg


def Exp(streams, body):
    # TODO
    pass


def ArgMin(streams, body):
    return fold(ArgMinAlg, streams, body)


def ArgMax(streams, body):
    return fold(ArgMaxAlg, streams, body)


def Min(streams, body):
    return fold(MinAlg, streams, body)


def Max(streams, body):
    return fold(MaxAlg, streams, body)


def Sum(streams, body):
    return fold(LinAlg, streams, body)
