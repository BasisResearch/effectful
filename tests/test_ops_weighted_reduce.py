"""Tests for the reduce operation covering every possible body form.

The Body[T] type admits 5 forms, dispatched in _body_value:
  1. Interpretation  — Mapping[Operation, Callable]
  2. Callable         — function / lambda
  3. Mapping          — dict with non-Operation keys
  4. Generator        — generator object
  5. Raw value (T)    — Term, int, tuple, etc. (else branch)

Forms 2–4 require a promoted monoid (promote(SumMonoid)) because the
standard monoid add only operates on scalar values.
"""

from effectful.ops.syntax import defop
from effectful.ops.weighted.sugar import Sum

# ---------------------------------------------------------------------------
# Raw value body (else branch in _body_value → evaluate(body, intp=intp))
# ---------------------------------------------------------------------------


def test_body_raw_value():
    """Body is a single stream variable reference."""
    x = defop(int, name="x")
    result = Sum({x: [1, 2, 3]}, x())
    assert result == 6


def test_body_raw_value_arithmetic():
    """Body is an arithmetic expression on the stream variable."""
    x = defop(int, name="x")
    result = Sum({x: [1, 2, 3]}, x() + 10)
    assert result == 36  # 11 + 12 + 13


def test_body_raw_value_constant():
    """Body is a constant that does not reference any stream variable."""
    x = defop(int, name="x")
    result = Sum({x: [1, 2, 3]}, 5)
    assert result == 15  # 5 + 5 + 5


def test_body_raw_value_multi_stream():
    """Body references two stream variables."""
    x = defop(int, name="x")
    y = defop(int, name="y")
    result = Sum({x: [1, 2], y: [10, 20]}, x() + y())
    # all pairs: (1+10, 1+20, 2+10, 2+20) = 11+21+12+22 = 66
    assert result == 66


def test_body_raw_value_tuple():
    """Body is a tuple of expressions — evaluated element-wise via evaluate."""
    x = defop(int, name="x")
    result = Sum({x: [1, 2, 3]}, (x(), x() + 1))
    # tuples are added with +, which concatenates: (1, 2) + (2, 3) + (3, 4)
    assert result == (6, 9)


# ---------------------------------------------------------------------------
# Callable body (callable branch in _body_value → handler(intp)(body))
# ---------------------------------------------------------------------------


def test_body_callable_no_args():
    """Body is a zero-arg lambda. Result is a callable returning the sum."""
    x = defop(int, name="x")
    result = Sum({x: [1, 2, 3]}, lambda: x())
    assert callable(result)
    assert result() == 6


def test_body_callable_with_args():
    """Body is a lambda accepting an argument. Result is a callable."""
    x = defop(int, name="x")
    result = Sum({x: [1, 2, 3]}, lambda a: x() + a)
    assert callable(result)
    # Each stream element gives (val + a); sum = (1+a) + (2+a) + (3+a) = 6 + 3a
    assert result(0) == 6
    assert result(10) == 36


# ---------------------------------------------------------------------------
# Mapping body (Mapping branch in _body_value → recursive _body_value)
# ---------------------------------------------------------------------------


def test_body_mapping():
    """Body is a flat dict with string keys."""
    x = defop(int, name="x")
    result = Sum({x: [1, 2, 3]}, {"a": x(), "b": x() + x()})
    assert result == {"a": 6, "b": 12}


def test_body_mapping_nested():
    """Body is a nested dict — _body_value recurses into sub-dicts."""
    x = defop(int, name="x")
    result = Sum({x: [1, 2, 3]}, {"outer": {"inner": x()}})
    assert result == {"outer": {"inner": 6}}


# ---------------------------------------------------------------------------
# Interpretation body (Interpretation branch in _body_value)
# ---------------------------------------------------------------------------


def test_body_interpretation():
    """Body is an Interpretation mapping an operation to a handler that
    references the stream variable."""
    x = defop(int, name="x")
    y = defop(int, name="y")
    result = Sum({x: [1, 2, 3]}, {y: lambda: x()})
    # result should be a mapping with y as key and a callable as value
    assert isinstance(result, dict)
    assert y in result
    assert callable(result[y])
    # The combined handler for y should return 1+2+3 = 6
    assert result[y]() == 6
