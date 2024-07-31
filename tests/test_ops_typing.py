import pytest

from effectful.internals.sugar import ObjectInterpretation, implements
from effectful.ops.core import Operation


@Operation
def plus(x: int, /) -> int:
    raise NotImplementedError


# we expect this to be a valid implementation
class Plus(ObjectInterpretation):
    @implements(plus)
    def _plus(self, x: int, /) -> int:
        return x + 1


@pytest.mark.mypy_testing
def test_bad_args():
    class BadArgs(ObjectInterpretation):
        @implements(plus)  # E: [arg-type]
        def _plus(self, x: bool, /) -> int:
            return 0


# this should produce a mypy error, but doesn't!
@pytest.mark.mypy_testing
def test_bad_return():
    class BadReturn(ObjectInterpretation):
        @implements(plus)  # E: [arg-type]
        def _plus(self, x: int, /) -> bool:
            return False


@pytest.mark.mypy_testing
def test_too_many_args():
    class TooManyArgs(ObjectInterpretation):
        @implements(plus)  # E: [arg-type]
        def _plus(self, x: int, y: int, /) -> int:
            return 0
