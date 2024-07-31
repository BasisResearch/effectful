from effectful.internals.sugar import implements
from effectful.ops.core import Operation, define


@define(Operation)
def plus(x: int) -> int:
    raise NotImplementedError


def test_plus_handler_typing():
    @implements(plus)
    def _plus(x: int) -> int:
        return x + 1

    @implements(plus)
    def _bad_args(x: bool) -> int:
        return 0

    @implements(plus)
    def _bad_return(x: int) -> bool:
        return False

    @implements(plus)
    def _too_many_args(x: int, y: int) -> int:
        return 0
