import functools
from collections.abc import Callable

from effectful.ops.types import Operation


class _ClassMethodOperation[**P, S, T]:
    def __init__(self, default):
        self._default = default
        self._operation = {}

    def __get__(self, instance, owner: type[S]) -> Callable[P, T]:
        op = self._operation.get(owner, None)
        if op is None:
            func = self._default.__func__
            op = Operation.define(functools.partial(func, owner), name=func.__name__)
            self._operation[owner] = op
        return op
