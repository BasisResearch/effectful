import ast
import collections.abc
import copy
import typing

import pytest

from effectful.internals.disassembly import (
    CompLambda,
    DummyIterName,
    disassemble,
    ensure_ast,
)


def compile_and_eval(
    node: ast.expr | ast.Expression, globals_dict: dict | None = None
) -> typing.Any:
    """Compile an AST node and evaluate it."""
    if globals_dict is None:
        globals_dict = {}

    # Wrap in an Expression node if needed
    if not isinstance(node, ast.Expression):
        node = ast.Expression(body=node)

    # Fix location info
    ast.fix_missing_locations(node)

    # Compile and evaluate
    code = compile(node, "<ast>", "eval")
    return eval(code, globals_dict)


def materialize[T](genexpr: collections.abc.Generator[T, None, None]) -> list[T]:
    """Materialize a nested generator expression to a nested list."""

    def _materialize(genexpr):
        if isinstance(genexpr, str | bytes):
            return genexpr
        elif isinstance(genexpr, collections.abc.Generator):
            return [_materialize(item) for item in genexpr]
        elif isinstance(genexpr, tuple):
            # Kept as a tuple so that sets of tuples stay hashable
            return tuple(_materialize(item) for item in genexpr)
        elif isinstance(genexpr, collections.abc.Sequence):
            return [_materialize(item) for item in genexpr]
        elif isinstance(genexpr, collections.abc.Set):
            return {_materialize(item) for item in genexpr}
        elif isinstance(genexpr, collections.abc.Mapping):
            return {_materialize(k): _materialize(v) for k, v in genexpr.items()}
        else:
            return genexpr

    return [_materialize(x) for x in genexpr]


def assert_ast_equivalent(
    genexpr: collections.abc.Generator[typing.Any, None, None],
    reconstructed_ast: ast.AST,
    globals_dict: dict | None = None,
):
    """Assert that a reconstructed AST produces the same results as the original generator."""
    # Check AST structure
    assert isinstance(reconstructed_ast, ast.Expression)
    assert hasattr(reconstructed_ast.body, "elt")  # The expression part
    assert hasattr(reconstructed_ast.body, "generators")  # The comprehension part
    assert len(reconstructed_ast.body.generators) > 0
    for comp in reconstructed_ast.body.generators:
        assert hasattr(comp, "target")  # Loop variable
        assert hasattr(comp, "iter")  # Iterator
        assert hasattr(comp, "ifs")  # Conditions

    # Save current globals to restore later
    curr_globals = globals().copy()
    globals().update(globals_dict or {})

    # Materialize original generator to list for comparison
    original_list = materialize(genexpr)

    # Clean up globals to avoid pollution
    for key in globals_dict or {}:
        if key not in curr_globals:
            del globals()[key]
    globals().update(curr_globals)

    # Compile and evaluate the reconstructed AST
    reconstructed_gen = compile_and_eval(reconstructed_ast, globals_dict)
    reconstructed_list = materialize(reconstructed_gen)
    assert reconstructed_list == original_list, (
        f"AST produced {reconstructed_list}, expected {original_list}"
    )


# ============================================================================
# BASIC GENERATOR EXPRESSION TESTS
# ============================================================================


@pytest.mark.parametrize(
    "genexpr",
    [
        # Simple generator expressions
        (x for x in range(5)),
        (y for y in range(10)),
        (item for item in [1, 2, 3]),
        # Edge cases for simple generators
        (i for i in range(0)),  # Empty range
        (n for n in range(1)),  # Single item range
        (val for val in range(100)),  # Large range
        (x for x in range(-5, 5)),  # Negative range
        (step for step in range(0, 10, 2)),  # Step range
        (rev for rev in range(10, 0, -1)),  # Reverse range
    ],
)
def test_simple_generators(genexpr):
    """Test reconstruction of simple generator expressions."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# ARITHMETIC AND EXPRESSION TESTS
# ============================================================================


@pytest.mark.parametrize(
    "genexpr",
    [
        # Basic arithmetic operations
        (x * 2 for x in range(5)),
        (x + 1 for x in range(5)),
        (x - 1 for x in range(5)),
        (x**2 for x in range(5)),
        (x % 2 for x in range(10)),
        (x / 2 for x in range(1, 6)),
        (x // 2 for x in range(10)),
        # Complex expressions
        (x * 2 + 1 for x in range(5)),
        ((x + 1) * (x - 1) for x in range(5)),
        (x**2 + 2 * x + 1 for x in range(5)),
        # Unary operations
        (-x for x in range(5)),
        (+x for x in range(-5, 5)),
        (~x for x in range(5)),
        # More complex arithmetic edge cases
        (x**3 for x in range(1, 5)),  # Higher powers
        (x * x * x for x in range(5)),  # Repeated multiplication
        (x + x + x for x in range(5)),  # Repeated addition
        (x - x + 1 for x in range(5)),  # Operations that might simplify
        (x / x for x in range(1, 5)),  # Division by self
        (x % (x + 1) for x in range(1, 10)),  # Modulo with expression
        # Nested arithmetic expressions
        ((x + 1) ** 2 for x in range(5)),
        ((x * 2 + 3) * (x - 1) for x in range(5)),
        (x * (x + 1) * (x + 2) for x in range(5)),
        # Mixed operations with precedence
        (x + 3 * 2 for x in range(3)),
        (x * 2 + 9 / 3 for x in range(1, 4)),
        ((x + 2) * (x - 2) for x in range(1, 4)),
        # Edge cases with zero and one
        (x * 0 for x in range(5)),
        (x * 1 for x in range(5)),
        (x + 0 for x in range(5)),
        (x**1 for x in range(5)),
        (0 + x for x in range(5)),
        (1 * x for x in range(5)),
    ],
)
def test_arithmetic_expressions(genexpr):
    """Test reconstruction of generators with arithmetic expressions."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# FSTRING EXPRESSIONS
# ============================================================================


@pytest.mark.parametrize(
    "genexpr",
    [
        # Basic f-string cases
        (f"{x}" for x in range(5)),  # Single value, no format
        (f"{x} is {x**2}" for x in range(5)),  # Multiple values
        (f"{x:02d}" for x in range(10)),  # Format spec
        (f"{x:.2f}" for x in [1.2345, 2.3456, 3.4567]),  # Float format spec
        # Conversion specifiers
        (f"{x!r}" for x in ["hello", "world"]),  # repr conversion
        (f"{x!s}" for x in [1, 2, 3]),  # str conversion
        (f"{x!a}" for x in ["hello\n", "world\t"]),  # ascii conversion
        # Conversion with format spec
        (f"{x!r:>10}" for x in ["hello", "world"]),  # repr with alignment
        (f"{x!s:^15}" for x in [1, 2, 3]),  # str with center align
        # Empty and literal f-strings
        ("" for x in range(3)),  # Empty f-string
        ("constant" for x in range(3)),  # No formatting
        (f"x={x}" for x in range(5)),  # Literal prefix
        (f"result: {x * 2}" for x in range(5)),  # Literal with expression
        # Complex expressions in f-strings
        (f"{x + 1}" for x in range(5)),  # Arithmetic
        (f"{x * x}" for x in range(5)),  # Multiplication
        (f"{x % 2}" for x in range(10)),  # Modulo
        (f"{-x}" for x in range(-2, 3)),  # Unary minus
        # Nested formatting
        (f"{x:0{2}d}" for x in range(5)),  # Format spec with expression
        (f"{x:>{3 * 2}}" for x in range(5)),  # Expression in format spec
        # Multiple formatted values
        (f"{x} + {y} = {x + y}" for x in range(3) for y in range(3)),  # Multiple vars
        (f"({x}, {y})" for x in range(2) for y in range(2)),  # Tuple display
        # F-strings with various data types
        (f"{s}" for s in ["hello", "world"]),  # Strings
        (f"{b}" for b in [True, False]),  # Booleans
        (f"{n}" for n in [None, None]),  # None values
        (f"{lst}" for lst in [[1, 2], [3, 4]]),  # Lists
        # Complex format specifications
        (f"{x:+05d}" for x in range(-2, 3)),  # Sign, zero pad, width
        (f"{x:.2%}" for x in [0.1, 0.25, 0.333]),  # Percentage format
        (f"{x:.2e}" for x in [100, 1000, 10000]),  # Scientific notation
        (f"{x:#x}" for x in [10, 15, 255]),  # Hex with prefix
        (f"{x:b}" for x in [2, 7, 15]),  # Binary format
        # Edge cases
        ("{x}" for x in range(3)),  # Escaped braces
        (f"{{x}} = {x}" for x in range(3)),  # Mixed escaped/formatted
        (f"{{{x}}}" for x in range(3)),  # Brace around formatted
    ],
)
def test_fstring_expressions(genexpr):
    """Test reconstruction of generators with f-string expressions."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# COMPARISON OPERATORS
# ============================================================================


@pytest.mark.parametrize(
    "genexpr",
    [
        # All comparison operators
        (x for x in range(10) if x < 5),
        (x for x in range(10) if x <= 5),
        (x for x in range(10) if x > 5),
        (x for x in range(10) if x >= 5),
        (x for x in range(10) if x == 5),
        (x for x in range(10) if x != 5),
        # in/not in operators
        (x for x in range(10) if x in [2, 4, 6, 8]),
        (x for x in range(10) if x not in [2, 4, 6, 8]),
        # is/is not operators (with None)
        (x for x in [1, None, 3, None, 5] if x is not None),
        (x for x in [1, None, 3, None, 5] if x is None),
        # Boolean operations - these are complex cases that might need special handling
        (x for x in range(10) if not x % 2),
        (x for x in range(10) if not (x > 5)),
        (x for x in range(10) if x > 2 and x < 8),
        (x for x in range(10) if x < 3 or x > 7),
        # More complex comparison edge cases
        # Comparisons with expressions
        (x for x in range(10) if x * 2 > 10),
        (x for x in range(10) if x + 1 <= 5),
        (x for x in range(10) if x**2 < 25),
        (x for x in range(10) if (x + 1) * 2 != 6),
        # Complex membership tests
        (x for x in range(20) if x in range(5, 15)),
        (x for x in range(10) if x not in range(3, 7)),
        (x for x in range(10) if x % 2 in [0]),
        (x for x in range(10) if x not in []),  # Empty container
        # Complex boolean combinations
        (x for x in range(20) if not (x < 5 or x > 15)),
        (x for x in range(20) if x > 5 and x < 15 and x % 2 == 0),
        (x for x in range(20) if x < 5 or x > 15 or x == 10),
        (x for x in range(20) if not (x > 5 and x < 15)),
        # Mixed comparison and boolean operations
        (x for x in range(20) if (x > 10 and x % 2 == 0) or (x < 5 and x % 3 == 0)),
        (x for x in range(20) if not (x % 2 == 0 and x % 3 == 0)),
        # Edge cases with identity comparisons
        (x for x in [0, 1, 2, None, 4] if x is not None and x > 1),
        (x for x in [True, False, 1, 0] if x is True),
        (x for x in [True, False, 1, 0] if x is not False),
    ],
)
def test_comparison_operators(genexpr):
    """Test reconstruction of all comparison operators."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# CHAINED COMPARISON TESTS
# ============================================================================


@pytest.mark.parametrize(
    "genexpr",
    [
        # Chained comparisons
        (x for x in range(20) if 5 < x < 15),
        (x for x in range(20) if 0 <= x <= 10),
    ],
)
def test_chained_comparison_operators(genexpr):
    """Test reconstruction of chained (ternary) comparison operators."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# FILTERED GENERATOR TESTS
# ============================================================================


@pytest.mark.parametrize(
    "genexpr",
    [
        # Simple filters
        (x for x in range(10) if x % 2 == 0),
        (x for x in range(10) if x > 5),
        (x for x in range(10) if x < 5),
        (x for x in range(10) if x != 5),
        # Complex filters
        (x for x in range(20) if x % 2 == 0 if x % 3 == 0),
        (x for x in range(100) if x > 10 if x < 90 if x % 5 == 0),
        # Filters with expressions
        (x * 2 for x in range(10) if x % 2 == 0),
        (x**2 for x in range(10) if x > 3),
        # Boolean operations in filters
        (x for x in range(10) if not x % 2),
        (x for x in range(10) if x > 2 and x < 8),
        (x for x in range(10) if x < 3 or x > 7),
        # More complex filter edge cases
        (x for x in range(50) if x % 7 == 0),  # Different modulo
        (x for x in range(10) if x >= 0),  # Always true condition
        (x for x in range(10) if x < 0),  # Always false condition
        (
            x for x in range(20) if x % 2 == 0 and x % 3 == 0
        ),  # Multiple conditions with and
        (
            x for x in range(20) if x % 2 == 0 or x % 3 == 0
        ),  # Multiple conditions with or
        # Nested boolean operations
        (x for x in range(20) if (x > 5 and x < 15) or x == 0),
        (x for x in range(20) if not (x > 10 and x < 15)),
        (x for x in range(50) if x > 10 and (x % 2 == 0 or x % 3 == 0)),
        # Multiple consecutive filters
        (x for x in range(100) if x > 20 if x < 80 if x % 10 == 0),
        (x for x in range(50) if x % 2 == 0 if x % 3 != 0 if x > 10),
        # Filters with complex expressions
        (x + 1 for x in range(20) if (x * 2) % 3 == 0),
        (x**2 for x in range(10) if x * (x + 1) > 10),
        (x / 2 for x in range(1, 20) if x % (x // 2 + 1) == 0),
        # Edge cases with truthiness
        (x for x in range(10) if x),  # Truthy filter
        (x for x in range(-5, 5) if not x),  # Falsy filter
        (x for x in range(10) if bool(x % 2)),  # Explicit bool conversion
    ],
)
def test_filtered_generators(genexpr):
    """Test reconstruction of generators with if conditions."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# NESTED LOOP TESTS
# ============================================================================


@pytest.mark.parametrize(
    "genexpr",
    [
        # Basic nested loops
        ((x, y) for x in range(3) for y in range(3)),
        (x + y for x in range(3) for y in range(3)),
        (x * y for x in range(1, 4) for y in range(1, 4)),
        # Nested with filters
        ((x, y) for x in range(5) for y in range(5) if x < y),
        (x + y for x in range(5) if x % 2 == 0 for y in range(5) if y % 2 == 1),
        # Triple nested
        (x + y + z for x in range(2) for y in range(3) for z in range(4)),
        ((x, y, z) for x in range(2) for y in range(3) for z in range(4)),
        # More complex nested loop edge cases
        # Different sized ranges
        ((x, y) for x in range(2) for y in range(5)),
        ((x, y) for x in range(10) for y in range(2)),
        # Asymmetric operations
        (x - y for x in range(5) for y in range(3)),
        (x / (y + 1) for x in range(1, 6) for y in range(3)),
        (x**y for x in range(1, 4) for y in range(3)),
        # Complex expressions with multiple variables
        (x * y + x for x in range(3) for y in range(3)),
        (x + y + x * y for x in range(1, 4) for y in range(1, 4)),
        ((x + y) ** 2 for x in range(3) for y in range(3)),
        # Filters on different loop levels
        ((x, y) for x in range(10) if x % 2 == 0 for y in range(10) if y % 3 == 0),
        (x * y for x in range(5) for y in range(5) if x != y),
        (x + y for x in range(5) for y in range(5) if x + y < 5),
        # Triple and quadruple nested with various patterns
        (x + y + z for x in range(2) for y in range(2) for z in range(2)),
        (x * y * z for x in range(1, 3) for y in range(1, 3) for z in range(1, 3)),
        (
            (x, y, z, w)
            for x in range(2)
            for y in range(2)
            for z in range(2)
            for w in range(2)
        ),
        # Nested loops with complex filters
        ((x, y) for x in range(5) if x > 1 for y in range(5) if x < y),
        (x + y for x in range(3) if x > 0 for y in range(3)),
        # Mixed range types
        ((x, y) for x in range(-2, 2) for y in range(0, 4, 2)),
        (x * y for x in range(5, 0, -1) for y in range(1, 6)),
        # Dependent nested loops
        ((x, y) for x in range(3) for y in range(x, 3)),
        (x + y for x in range(3) for y in range(x + 1, 3)),
    ],
)
def test_nested_loops(genexpr):
    """Test reconstruction of generators with nested loops."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ===========================================================================
# NESTED COMPREHENSIONS
# ===========================================================================


@pytest.mark.parametrize(
    "genexpr",
    [
        # nested generators
        ((x for x in range(i + 1)) for i in range(5)),
        ((x for j in range(i) for x in range(j)) for i in range(5)),
        (((x for x in range(i + j)) for j in range(i)) for i in range(5)),
        # nested generators with filters
        ((x for x in range(i)) for i in range(5) if i > 0),
        ((x for x in range(i) if x < i) for i in range(5) if i > 0),
        (((x for x in range(i + j) if x < i + j) for j in range(i)) for i in range(5)),
        # aggregation function call
        (sum(x for x in range(i + 1)) for i in range(3)),
        (max(x for x in range(i + 1)) for i in range(3)),
        (dict((x, x + 1) for x in range(i + 1)) for i in range(3)),
        (set(x for x in range(i + 1)) for i in range(3)),
        # map
        (list(map(abs, (x + 1 for x in range(i + 1)))) for i in range(3)),
        (list(enumerate(x + 1 for x in range(i + 1))) for i in range(3)),
        # nesting on both sides
        ((y for y in range(x)) for x in (x_ + 1 for x_ in range(5))),
        ((y for y in range(x)) for x in (x_ + 1 for x_ in range(5))),
    ],
)
def test_nested_comprehensions(genexpr):
    """Test reconstruction of nested comprehensions."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


def test_nested_comprehensions_multiline():
    """The same filter reconstructs the same way however the source is laid out.

    On Python 3.12 these two spellings disassemble to different jump layouts --
    only the one-line form emits POP_JUMP_IF_TRUE -- which used to make the
    multiline form come out negated.
    """
    one_line = (x for x in range(5) if x > 1)
    assert_ast_equivalent(one_line, disassemble(one_line))

    multiline = (
        x
        for x in range(5)  # comment to avoid reformatting
        if x > 1
    )
    assert_ast_equivalent(multiline, disassemble(multiline))

    assert ast.unparse(disassemble(x for x in range(5) if x > 1)) == ast.unparse(
        disassemble(
            x
            for x in range(5)  # comment to avoid reformatting
            if x > 1
        )
    )


# ============================================================================
# DIFFERENT COMPREHENSION TYPES
# ============================================================================


@pytest.mark.parametrize(
    "genexpr",
    [
        # Comprehensions as iterator constants
        (x_ for x_ in [x for x in range(5)]),
        (x_ for x_ in {x for x in range(5)}),
        (x_ for x_ in {x: x**2 for x in range(5)}),
        # Comprehensions as yield expressions
        ([y * 2 for y in range(x + 1)] for x in range(3)),
        ({y + 3 for y in range(x + 1)} for x in range(3)),
        ({y: y**2 for y in range(x + 1)} for x in range(3)),
        # nested non-generators
        ([x for x in range(i)] for i in range(5)),
        ([x for j in range(i) for x in range(j)] for i in range(5)),
        ({x: x**2 for x in range(i)} for i in range(5)),
        ([[x for x in range(i + j)] for j in range(i)] for i in range(5)),
        # Nested comprehensions with filters inside
        ([x for x in range(i)] for i in range(5) if i > 0),
        ([x for x in range(i) if x < i] for i in range(5) if i > 0),
        ([[x for x in range(i + j) if x < i + j] for j in range(i)] for i in range(5)),
    ],
)
def test_different_comprehension_types(genexpr):
    """Test reconstruction of different comprehension types."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# DICT DISPLAYS
#
# A dict display with dynamic keys is built by BUILD_MAP from key/value pairs
# that the compiler pushed in source order. Dicts compare equal whatever order
# they were built in, so these tests pin the order down directly: which of two
# equal keys wins, what `items()` yields, and when each subexpression runs.
# ============================================================================


_EVAL_ORDER: list[str] = []


def _note(tag: str, value: typing.Any) -> typing.Any:
    """Record that this subexpression was evaluated, and pass its value along."""
    _EVAL_ORDER.append(tag)
    return value


def test_dict_display_duplicate_keys():
    """The last of two equal keys wins, so the pairs must keep their order."""
    genexpr = ({x: "first", x: "second"} for x in range(1))  # noqa: F602
    reconstructed = disassemble(genexpr)
    assert ast.unparse(reconstructed) == (
        "({x: 'first', x: 'second'} for x in range(0, 1, 1))"
    )
    assert materialize(genexpr) == [{0: "second"}]
    assert materialize(compile_and_eval(reconstructed)) == [{0: "second"}]


def test_dict_display_insertion_order():
    genexpr = ({x: "a", x + 1: "b", x + 2: "c"} for x in range(1))
    reconstructed = disassemble(genexpr)
    expected = [[(0, "a"), (1, "b"), (2, "c")]]
    assert [list(d.items()) for d in genexpr] == expected
    assert [list(d.items()) for d in compile_and_eval(reconstructed)] == expected


def test_dict_display_evaluation_order():
    """Keys and values run left to right, key before its own value."""
    genexpr = (
        {_note("k1", "a"): _note("v1", 1), _note("k2", "b"): _note("v2", 2)}
        for _ in range(1)
    )
    reconstructed = disassemble(genexpr)

    _EVAL_ORDER.clear()
    assert materialize(genexpr) == [{"a": 1, "b": 2}]
    assert _EVAL_ORDER == ["k1", "v1", "k2", "v2"]

    _EVAL_ORDER.clear()
    assert materialize(compile_and_eval(reconstructed, {"_note": _note})) == [
        {"a": 1, "b": 2}
    ]
    assert _EVAL_ORDER == ["k1", "v1", "k2", "v2"]


# ============================================================================
# CONDITIONAL EXPRESSIONS
# ============================================================================


@pytest.mark.parametrize(
    "genexpr",
    [
        # simple conditional expressions without nesting
        ((lambda x: x if x % 2 == 0 else -x)(xi) for xi in range(5)),
        ((lambda x: (x + 1) if x < 5 else (x - 1))(xi) for xi in range(10)),
        ((lambda x: (x * 2) if x > 0 else (x / 2))(xi) for xi in range(-5, 5)),
        ((lambda x: (x**2) if x != 0 else 1)(xi) for xi in range(-3, 4)),
        # simple conditional expressions with negation
        ((lambda x: (x + 10) if not (x < 5) else (x - 10))(xi) for xi in range(20)),
        ((lambda x: (x * 3) if not (x % 2 == 0) else (x // 3))(xi) for xi in range(10)),
        ((lambda x: (x**3) if not (x < 0) else (x**0.5))(xi) for xi in range(-5, 15)),
        # conditional expressions with lazy test
        (
            (lambda x: (x + 10) if (x > 5 and x < 15) else (x - 10))(xi)
            for xi in range(20)
        ),
        (
            (lambda x: (x * 3) if (x % 2 == 0 or x % 3 == 0) else (x // 3))(xi)
            for xi in range(10)
        ),
        (
            (lambda x: (x**3) if not (x < 0 or x > 10) else (x**0.5))(xi)
            for xi in range(-5, 15)
        ),
    ],
)
def test_conditional_expressions_simple_no_comprehension(genexpr):
    """Test reconstruction of simple conditional expressions isolated from comprehension bodies."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


@pytest.mark.parametrize(
    "genexpr",
    [
        # nested conditional expressions
        (
            (lambda x: (x + 1) if x < 5 else ((x - 1) if x < 10 else (x * 2)))(xi)
            for xi in range(15)
        ),
        (
            (
                lambda x: (
                    (x * 2) if x % 2 == 0 else ((x // 2) if x % 3 == 0 else (x + 2))
                )
            )(xi)
            for xi in range(10)
        ),
        (
            (lambda x: (x**2) if x > 0 else ((-x) ** 2 if x < -5 else 1))(xi)
            for xi in range(-10, 5)
        ),
    ],
)
def test_conditional_expressions_nested_no_comprehension(genexpr):
    """Test reconstruction of nested conditional expressions isolated from comprehension bodies."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


@pytest.mark.parametrize(
    "genexpr",
    [
        # Basic conditional expressions in comprehension bodies
        ((x if x % 2 == 0 else -x) for x in range(5)),
        ((x * 2 if x > 0 else x / 2) for x in range(-3, 4)),
        ((x**2 if x != 0 else 1) for x in range(-2, 3)),
        # Conditional expressions with filters
        ((x if x % 2 == 0 else -x) for x in range(10) if x > 2),
        ((x * 3 if x > 5 else x + 1) for x in range(20) if x % 3 == 0),
        # Nested loops with conditional expressions
        ((x + y if x > y else x - y) for x in range(3) for y in range(3)),
        (
            (x * y if x != 0 and y != 0 else 0)
            for x in range(-2, 3)
            for y in range(-2, 3)
        ),
        # Multiple conditional expressions
        (
            (x if x > 0 else 0) + (y if y > 0 else 0)
            for x in range(-2, 3)
            for y in range(-2, 3)
        ),
        # Conditional expressions in different parts
        ([x if x > 0 else -x for x in range(i)] for i in range(1, 4)),
        ((x if x % 2 == 0 else -x) for x in (y if y > 2 else y + 10 for y in range(5))),
        # Complex nested conditional expressions
        ((x if x > 0 else (x + 5 if x > -3 else x * 2)) for x in range(-5, 5)),
        ((x * 2 if x > 0 else (x / 2 if x < 0 else 1)) for x in range(-3, 4)),
        # Conditional expressions with function calls
        ((abs(x) if x < 0 else x) for x in range(-3, 4)),
        ((max(x, 0) if x is not None else 0) for x in [None, -1, 0, 1, 2]),
        # Mixed with other complex expressions
        ((x + 1 if x % 2 == 0 else x - 1) * 2 for x in range(5)),
        ((x, y, x + y if x > y else x - y) for x in range(3) for y in range(3)),
    ],
)
def test_conditional_expressions_simple_comprehensions(genexpr):
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


@pytest.mark.parametrize(
    "genexpr",
    [
        # `and` chains compile to a run of jumps that each fall through to the
        # loop back-edge, which the disassembler folds into a single filter.
        (x for x in range(10) if x > 2 and x < 8),
        (x for x in range(20) if x > 5 and x % 2 == 0 and x < 15),
        (x for x in range(-10, 10) if abs(x) > 3 and x % 2 == 0),
        (x for x in ["hello", "world", "test"] if len(x) > 3 and x.startswith("h")),
        (x for x in range(20) if x % 2 == 0 and x % 3 == 0 and x > 0 and x < 18),
        ((x, y) for x in range(5) for y in range(5) if x < y and x + y > 2),
        # `or` in filter position: reconstructed incorrectly, see the marker.
        (x for x in range(10) if x < 3 or x > 7),
        (x for x in range(20) if x < 5 or x > 15 or x == 10),
        (x for x in range(20) if (x > 10 and x % 2 == 0) or (x < 5 and x % 3 == 0)),
        (x for x in range(20) if x > 5 and (x < 10 or x > 15)),
        (x for x in range(100) if (x > 10 and x < 50) and (x % 3 == 0 or x % 5 == 0)),
        # `not (a and b)` is compiled exactly like `not a or not b`.
        (x for x in range(100) if not (x > 30 and x < 70)),
        # Chained comparisons in filter position.
        (x for x in range(20) if 5 < x < 15),
        (x for x in range(20) if 0 <= x <= 10),
        (x for x in range(50) if 10 < x < 20 < x * 2),
        (x for x in range(10) if 0 <= x <= 5 <= x + 5),
        (x for x in range(50) if 5 < x < 15 and x % 2 == 0),
        (x for x in range(50) if x > 20 or 5 < x < 15),
    ],
)
def test_lazy_boolean_and_chained_comparisons_in_filters(genexpr):
    """Lazy boolean operators and chained comparisons in *filter* position.

    This is the hard case: a filter's condition is recognised structurally, by
    the jump falling through to the loop back-edge, so any condition CPython
    compiles with an intermediate join point is misread.
    """
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


@pytest.mark.parametrize(
    "genexpr",
    [
        # The same operators in *ternary* position all work: both arms produce a
        # value, so the fork/merge machinery in _symbolic_exec applies directly.
        ((x if x > 5 and x < 15 else 0) for x in range(20)),
        ((x if x < 3 or x > 17 else -x) for x in range(20)),
        ((x if 5 < x < 15 else 0) for x in range(20)),
        ((x * 2 if 0 <= x <= 10 else x / 2) for x in range(-5, 15)),
        ((x if x > 2 and x < 8 else -x) for x in range(10)),
        ((x if x < 2 or x > 8 else -x) for x in range(10)),
        ((x if not (x > 2 and x < 8) else -x) for x in range(10)),
        ((x if 0 <= x <= 5 <= x + 5 else -x) for x in range(10)),
        ((x if x > 1 and x < 9 or x == 0 else -x) for x in range(10)),
        # ... including nested inside another ternary
        ((x if x > 5 or x < 2 else (0 if x == 3 else 1)) for x in range(10)),
        ((x if x > 5 else (0 if 2 < x < 4 else 1)) for x in range(10)),
    ],
)
def test_lazy_boolean_and_chained_comparisons_in_ternaries(genexpr):
    """Lazy boolean operators and chained comparisons in conditional expressions."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


def test_short_circuit_filter_is_a_disjunction_of_paths():
    """A short-circuiting filter is rebuilt as the disjunction over its paths.

    `x < 3 or x > 7` used to be misread as a conditional expression, yielding an
    AST that compiled and ran but computed `[0, 1, 2, False, False, ...]` instead
    of `[0, 1, 2, 8, 9]`. Each path through the condition now contributes one
    disjunct, so the reconstruction is equivalent rather than merely plausible.
    """
    genexpr = (x for x in range(10) if x < 3 or x > 7)
    reconstructed = disassemble(genexpr)

    assert isinstance(reconstructed.body, ast.GeneratorExp)
    filters = reconstructed.body.generators[0].ifs
    assert len(filters) == 1
    assert isinstance(filters[0], ast.BoolOp) and isinstance(filters[0].op, ast.Or)
    assert materialize(compile_and_eval(reconstructed)) == [0, 1, 2, 8, 9]


@pytest.mark.parametrize(
    "genexpr",
    [
        # Simple conditional as function argument
        (max(x if x > 0 else 0, 1) for x in range(-2, 3)),
        (abs(x if x < 0 else -x) for x in range(-3, 3)),
        (len(str(x) if x > 10 else "small") for x in range(15)),
        # Multiple conditional arguments
        (
            max(x if x > 0 else 0, y if y > 0 else 0)
            for x in range(-1, 2)
            for y in range(-1, 2)
        ),
        (
            pow(x if x != 0 else 1, y if y > 0 else 1)
            for x in range(3)
            for y in range(3)
        ),
        # Nested function calls with conditionals
        (max(abs(x if x < 0 else -x), 1) for x in range(-3, 4)),
        (int(str(x if x > 5 else x + 10)) for x in range(10)),
        # Conditionals in keyword arguments (using dict constructor as example)
        (dict(a=x if x > 0 else 0, b=x * 2 if x < 5 else x) for x in range(8)),
        # Method calls with conditional arguments
        ([1, 2, 3].index(x if x in [1, 2, 3] else 1) for x in range(5)),
        ("hello".replace("l", x if isinstance(x, str) else "X") for x in ["a", 1, "b"]),
        # Complex nested case: conditional in function argument, function call in conditional
        (abs(x if len(str(x)) > 1 else x * 10) for x in range(15)),
        # Mixed: conditional in function call within comprehension filter
        (x for x in range(20) if max(x if x > 10 else 0, 5) > 8),
    ],
)
def test_conditional_expressions_function_arguments(genexpr):
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# GENERATOR EXPRESSION WITH GLOBALS
# ============================================================================


@pytest.mark.parametrize(
    "genexpr,globals_dict",
    [
        # Using constants
        ((x + a for x in range(5)), {"a": 10}),  # type: ignore  # noqa: F821
        ((data[i] for i in range(2)), {"data": [3, 4]}),  # type: ignore  # noqa: F821
        # Using global functions
        ((abs(x) for x in range(-5, 5)), {"abs": abs}),
        ((len(s) for s in ["a", "ab", "abc"]), {"len": len}),
        ((max(x, 5) for x in range(10)), {"max": max}),
        ((min(x, 5) for x in range(10)), {"min": min}),
        ((round(x / 3, 2) for x in range(10)), {"round": round}),
    ],
)
def test_variable_lookup(genexpr, globals_dict):
    """Test reconstruction of expressions with globals."""
    ast_node = disassemble(genexpr)

    # Need to provide the same globals for evaluation
    assert_ast_equivalent(genexpr, ast_node, globals_dict)


# ============================================================================
# EDGE CASES AND COMPLEX SCENARIOS
# ============================================================================


@pytest.mark.parametrize(
    "genexpr,globals_dict",
    [
        # Using lambdas and functions
        (((lambda y: y * 2)(x) for x in range(5)), {}),
        (((lambda y: y + 1)(x) for x in range(5)), {}),
        (((lambda y: y**2)(x) for x in range(5)), {}),
        (((lambda a, b: a + b)(x, x) for x in range(5)), {}),
        (((lambda: (x for x in range(i)))() for i in range(3)), {}),
        ((f(x) for x in range(5)), {"f": lambda y: y * 3}),  # type: ignore  # noqa: F821
        # Attribute access
        ((x.real for x in [1 + 2j, 3 + 4j, 5 + 6j]), {}),
        ((x.imag for x in [1 + 2j, 3 + 4j, 5 + 6j]), {}),
        ((x.conjugate() for x in [1 + 2j, 3 + 4j, 5 + 6j]), {}),
        # slicing and indexing
        ((s[:2] for s in ["hello", "world"]), {}),
        ((s[1:3] for s in ["hello", "world"]), {}),
        ((s[-1] for s in ["hello", "world"]), {}),
        ((s[0:3] for s in ["hello", "world"]), {}),
        ((s[::-1] for s in ["hello", "world"]), {}),
        ((s[1:2:] for s in ["hello", "world"]), {}),
        # Method calls
        ((s.upper() for s in ["hello", "world"]), {}),
        ((s.lower() for s in ["HELLO", "WORLD"]), {}),
        ((s.strip() for s in [" hello ", "  world  "]), {}),
        ((x.bit_length() for x in range(1, 10)), {}),
        ((str(x).zfill(3) for x in range(10)), {"str": str}),
        # Subscript operations
        (((10, 20, 30)[i] for i in range(3)), {}),
        (([10, 20, 30][i] for i in range(3)), {}),
        (({"a": 1, "b": 2, "c": 3}[k] for k in ["a", "b", "c"]), {}),
        (("hello"[i] for i in range(5)), {}),
        ((data[i][j] for i in range(2) for j in range(2)), {"data": [[1, 2], [3, 4]]}),  # type: ignore  # noqa: F821
        # # More complex attribute chains
        # ((obj.value.bit_length() for obj in [type('', (), {'value': x})() for x in range(1, 5)]), {}),
        # Multiple function calls
        ((abs(max(x, -x)) for x in range(-3, 4)), {"abs": abs, "max": max}),
        ((len(str(x)) for x in range(100, 110)), {"len": len, "str": str}),
        # Mixed operations
        (
            (abs(x) + len(str(x)) for x in range(-10, 10)),
            {"abs": abs, "len": len, "str": str},
        ),
        ((s.upper().lower() for s in ["Hello", "World"]), {}),
        # Edge cases with complex data structures
        (((1, 2, 3)[x % 3] for x in range(10)), {}),
        (([1, 2, 3][x % 3] for x in range(10)), {}),
        (({1, 2, 3} for x in range(10)), {}),
        # (({"even": x, "odd": x + 1}["even" if x % 2 == 0 else "odd"] for x in range(5)), {}),
        # Function calls with multiple arguments
        ((pow(x, 2, 10) for x in range(5)), {"pow": pow}),
        ((divmod(x, 3) for x in range(10)), {"divmod": divmod}),
    ],
)
def test_complex_scenarios(genexpr, globals_dict):
    """Test reconstruction of complex generator expressions."""
    ast_node = disassemble(genexpr)

    # Need to provide the same globals for evaluation
    assert_ast_equivalent(genexpr, ast_node, globals_dict)


# ============================================================================
# UNPACKING LOOP TARGETS
# ============================================================================


@pytest.mark.parametrize(
    "genexpr",
    [
        # Simple tuple targets
        ((a, b) for a, b in [(1, 2), (3, 4)]),
        (a + b for a, b in [(1, 2), (3, 4)]),
        (a * b for a, b in [(2, 3), (4, 5)]),
        ((b, a) for a, b in [(1, 2), (3, 4)]),
        ((a, b, c) for a, b, c in [(1, 2, 3), (4, 5, 6)]),
        ((a, b, c, d) for a, b, c, d in [(1, 2, 3, 4)]),
        # Nested tuple targets
        ((a, b, c) for a, (b, c) in [(1, (2, 3)), (4, (5, 6))]),
        ((a, b, c) for (a, b), c in [((1, 2), 3)]),
        ((a, b, c, d) for (a, b), (c, d) in [((1, 2), (3, 4))]),
        ((a, b, c) for a, (b, (c,)) in [(1, (2, (3,)))]),
        # Unpacking over dict views
        ((k, v) for k, v in {1: "a", 2: "b"}.items()),
        (v for k, v in {1: "a", 2: "b"}.items()),
        # Unpacking combined with filters
        ((a, b) for a, b in [(1, 2), (3, 1)] if a < b),
        (a + b for a, b in [(1, 2), (3, 4)] if a % 2 == 0),
        ((a, b) for a, b in [(1, 2), (3, 4)] if a > 0 if b > 3),
        # Unpacking in nested loops, in either position
        ((x, a, b) for x in range(2) for a, b in [(1, 2), (3, 4)]),
        ((a, b, y) for a, b in [(1, 2)] for y in range(2)),
        ((a, b, c, d) for a, b in [(1, 2)] for c, d in [(3, 4), (5, 6)]),
        ((a, b, x) for a, b in [(1, 2), (3, 4)] for x in range(a)),
        # Unpacking inside other comprehension types
        ([a for a, b in [(1, 2), (3, 4)]] for _ in range(2)),
        ({a for a, b in [(1, 2), (3, 4)]} for _ in range(2)),
        ({a: b for a, b in [(1, 2), (3, 4)]} for _ in range(2)),
        ((a for a, b in [(1, 2), (3, 4)]) for _ in range(2)),
        # Unpacking with a conditional expression in the body
        ((a if a > b else b) for a, b in [(1, 2), (4, 3)]),
    ],
)
def test_unpacking_targets(genexpr):
    """Test reconstruction of comprehensions that unpack their loop target."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


@pytest.mark.parametrize(
    "genexpr",
    [
        ((x, a, b) for x in range(2) for a, b in [(1, 2)]),
        ((a, b, c, d) for a, b in [(1, 2)] for c, d in [(3, 4)]),
        ((x, a, b) for x in range(2) for a, *b in [(1, 2, 3)]),
        ((x, a, b, c) for x in range(2) for a, (b, c) in [(1, (2, 3))]),
        ((x, a) for x in range(2) for (a,) in [(1,)]),
        ((x, a, b) for x in range(2) for a, *b in [(1,)]),  # type: ignore[var-annotated]
        ((x, y, a) for x in range(2) for y in range(2) for a, b in [(1, 2)]),
    ],
)
def test_unpacking_over_single_element_literal(genexpr):
    """A one-element inner loop over a literal.

    Python 3.14 unrolls this: it assigns the targets outright and emits no
    FOR_ITER, so the loop is not there to be recovered. The names it bound are
    substituted at their uses instead, which reproduces the same elements.
    """
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


@pytest.mark.parametrize(
    "genexpr",
    [
        # Starred last: UNPACK_EX with only a "before" count
        (a for a, *b in [(1, 2, 3), (4, 5, 6)]),
        (b for a, *b in [(1, 2, 3), (4, 5, 6)]),
        ((a, b) for a, *b in [(1, 2, 3)]),
        ((a, b, c) for a, b, *c in [(1, 2, 3, 4)]),
        # Starred first: the "after" count lives in the high byte of the
        # argument, so the instruction is prefixed with EXTENDED_ARG
        (a for *a, b in [(1, 2, 3), (4, 5, 6)]),
        (b for *a, b in [(1, 2, 3), (4, 5, 6)]),
        ((a, b) for *a, b in [(1, 2, 3)]),
        # Starred in the middle
        ((a, b, c) for a, *b, c in [(1, 2, 3, 4)]),
        ((a, b, c) for a, *b, c in [(1, 2, 3, 4, 5)]),
        # Starred target that collects nothing
        ((a, b) for a, *b in [(1,)]),  # type: ignore[var-annotated]
        # Combined with filters, nesting and other comprehension types
        ((a, b) for a, *b in [(1, 2), (3, 4)] if a > 1),
        ((x, a, b) for x in range(2) for a, *b in [(1, 2, 3), (4, 5, 6)]),
        ([a for a, *b in [(1, 2, 3)]] for _ in range(2)),
    ],
)
def test_unpacking_starred_targets(genexpr):
    """Test reconstruction of starred loop targets (UNPACK_EX)."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# OUTERMOST ITERABLE TYPES
#
# The outermost iterable is not part of the comprehension's bytecode: it is a
# live object reachable through `gi_frame.f_locals[".0"]`, so `ensure_ast` has
# to rebuild an expression for it from the object alone.
# ============================================================================


@pytest.mark.parametrize(
    "genexpr",
    [
        # Strings and bytes
        (c for c in "hello"),
        (c.upper() for c in "hello" if c != "l"),
        (c for c in "h\xe9llo"),  # non-ASCII takes a different iterator type
        (b for b in b"abc"),
        (b for b in bytearray(b"abc")),
        # Sequences
        (x for x in [1, 2, 3]),
        (x for x in (1, 2, 3)),
        (x for x in range(3)),
        # Sets and frozensets
        (x for x in {1, 2, 3}),
        (x for x in frozenset({1, 2, 3})),
        # Dict views
        (k for k in {1: "a", 2: "b"}),
        (k for k in {1: "a", 2: "b"}.keys()),
        (v for v in {1: "a", 2: "b"}.values()),
        (kv for kv in {1: "a", 2: "b"}.items()),
        # reversed() over each of the underlying sequence types
        (x for x in reversed([1, 2, 3])),
        (x for x in reversed((1, 2, 3))),
        (c for c in reversed("abc")),
        (x for x in reversed(range(3))),
        # Comprehensions as the outermost iterable
        (x for x in (y for y in range(3))),
        (x for x in [y for y in range(3)]),
        (x for x in {y for y in range(3)}),
        # Nested/structured contents
        (t for t in [(1, 2), (3, 4)]),
        (d for d in [{"a": 1}, {"b": 2}]),
        (x for x in [[1, 2], [3, 4]]),
    ],
)
def test_outermost_iterable_types(genexpr):
    """Test reconstruction of the outermost iterable from the live object."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


@pytest.mark.parametrize(
    "genexpr",
    [
        # zip/enumerate/map/filter wrap other iterators rather than a concrete
        # sequence, but pickle with their constituent parts.
        (x for x in zip([1, 2], [3, 4])),
        ((a, b) for a, b in zip([1, 2], [3, 4])),
        (x for x in zip("ab", range(2), [7, 8])),
        (x for x in enumerate("ab")),
        ((i, c) for i, c in enumerate("abc")),
        (x for x in enumerate("ab", 5)),
        (x for x in map(abs, [-1, 2])),
        (x for x in map(max, [1, 2], [3, 0])),
        (x for x in filter(None, [0, 1, 2])),
        (x for x in filter(bool, [0, 1, 2])),
        # A strict zip over equal-length iterables behaves like a lax one, but
        # its strictness has to survive anyway -- see the ragged case below.
        ((a, b) for a, b in zip([1, 2], [3, 4], strict=True)),
        (x for x in zip("ab", range(2), [7, 8], strict=True)),
        (x for x in zip(range(2), map(abs, [-1, -2]), strict=True)),
        # Nested adaptors, and adaptors over non-sequence iterables
        (x for x in zip(range(2), map(abs, [-1, -2]))),
        (x for x in enumerate(filter(None, [0, 1]))),
        (x for x in map(abs, range(-2, 2))),
        (x for x in zip("ab", (y for y in range(2)))),
        # With a filter and a non-trivial element expression
        (a + b for a, b in zip([1, 2], [3, 4]) if a > 1),
    ],
)
def test_outermost_iterable_adaptors(genexpr):
    """zip/enumerate/map/filter are rebuilt from the parts they pickle with."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


@pytest.mark.parametrize(
    "genexpr",
    [
        # A conditional expression as an inner loop's iterable. Its value is
        # consumed by FOR_ITER rather than by the yield, so the half-built
        # IfExp has to be pulled back out of the element slot.
        (y for x in range(4) for y in (range(x) if x % 2 == 0 else range(x, x + 2))),
        (y for x in range(4) for y in ([x] if x else [0])),
        (y for x in range(4) for y in (range(x) if x > 1 else range(1))),
        # An *empty* list literal as an arm is misread: BUILD_LIST(0) is also
        # how an inlined list comprehension starts, and nothing later
        # disambiguates the two here.
        pytest.param(
            (y for x in range(4) for y in ([x] if x else [])),
            marks=pytest.mark.xfail(
                strict=True,
                reason="an empty list literal is indistinguishable from the start of a list comprehension",
            ),
        ),
        ((x, y) for x in range(3) for y in ([0] if x % 2 else [1, 2])),
        # ... with a filter on the inner loop, and nested two deep
        (y for x in range(4) for y in (range(x) if x % 2 == 0 else [9]) if y > 0),
        pytest.param(
            (
                z
                for x in range(3)
                for y in (range(x) if x else [0])
                for z in ([y] if y else [7])
            ),
            marks=pytest.mark.xfail(
                strict=True,
                reason="two conditional iterables in one comprehension leave paths that do not pairwise merge",
            ),
        ),
        # ... and one whose arms are comprehensions of different kinds
        (y for x in range(3) for y in ([i for i in range(x)] if x else {8})),
    ],
)
def test_conditional_expression_as_iterable(genexpr):
    """Test a conditional expression in the iterable position of a for clause."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


@pytest.mark.parametrize(
    "genexpr",
    [
        # An always-false filter lets the compiler drop the body entirely, so no
        # element expression survives in the bytecode. The loop still runs.
        (x for x in range(6) if False),
        (x for x in range(6) if x and False),
        (y for x in range(6) if False and (y := x)),  # noqa: F821
        ([x] for x in range(4) if False),
        ((x, y) for x in range(4) for y in range(3) if False),
        (x for x in range(6) if False if x > 1),
        ({x for x in range(3)} for _ in range(2) if False),
    ],
)
def test_unreachable_comprehension_body(genexpr):
    """A comprehension whose body the compiler proved unreachable yields nothing."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)
    assert materialize(compile_and_eval(ast_node)) == []


def test_outermost_iterable_strict_zip_stays_strict():
    """A ragged strict zip raises; the reconstruction must raise there too."""
    genexpr = (a + b for a, b in zip([1, 2, 3], [4, 5], strict=True))
    reconstructed = disassemble(genexpr)
    assert ast.unparse(reconstructed) == (
        "(a + b for a, b in zip([1, 2, 3], [4, 5], strict=True))"
    )

    with pytest.raises(ValueError):
        materialize(compile_and_eval(reconstructed))
    with pytest.raises(ValueError):
        materialize(genexpr)


def test_outermost_iterable_lax_zip_stays_lax():
    """A lax zip stops at the shortest iterable and must not become strict."""
    genexpr = (a + b for a, b in zip([1, 2, 3], [4, 5]))
    reconstructed = disassemble(genexpr)
    assert ast.unparse(reconstructed) == "(a + b for a, b in zip([1, 2, 3], [4, 5]))"
    assert materialize(compile_and_eval(reconstructed)) == [5, 7]
    assert materialize(genexpr) == [5, 7]


def test_outermost_iterable_partially_consumed_strict_zip():
    """Strictness survives even once the zip has been partly consumed."""
    zipped = zip([1, 2, 3], [4, 5], strict=True)
    next(zipped)

    genexpr = (a + b for a, b in zipped)
    reconstructed = disassemble(genexpr)  # must precede consuming `genexpr`
    assert "strict=True" in ast.unparse(reconstructed)
    with pytest.raises(ValueError):
        materialize(compile_and_eval(reconstructed))


@pytest.mark.parametrize(
    "genexpr",
    [
        # "dict_item" was once an internal marker in the first slot of a tuple,
        # which is a string a user's data is perfectly entitled to hold.
        (x for x in (("dict_item", 1, 2),)),
        (x for x in [("dict_item", "key", "value")]),
        (x for x in (("dict_item",), ("dict_item", 1), ("dict_item", 1, 2, 3))),
        (("dict_item", x) for x in range(2)),
        (("dict_item", x, x + 1) for x in range(2)),
        (x for x in {("dict_item", 1, 2): "v"}.items()),
    ],
)
def test_tuples_starting_with_dict_item(genexpr):
    """No element of a user's tuple is an internal marker to be stripped."""
    assert_ast_equivalent(genexpr, disassemble(genexpr))


def test_outermost_iterable_partially_consumed_adaptor():
    """A consumed prefix is reflected in the adaptor's inner iterators."""
    zipped = zip([1, 2, 3], [4, 5, 6])
    next(zipped)

    genexpr = (a + b for a, b in zipped)
    reconstructed = disassemble(genexpr)  # must precede consuming `genexpr`
    assert materialize(genexpr) == [7, 9]
    assert materialize(compile_and_eval(reconstructed)) == [7, 9]


def test_outermost_iterable_partially_consumed():
    """Only the *unconsumed* remainder of the outermost iterator belongs in the AST."""
    iterator = iter([10, 20, 30, 40])
    next(iterator)
    next(iterator)

    genexpr = (x + 1 for x in iterator)
    assert ast.unparse(disassemble(genexpr)) == "(x + 1 for x in [30, 40])"
    assert materialize(genexpr) == [31, 41]


def test_outermost_iterable_partially_consumed_str():
    iterator = iter("hello")
    next(iterator)

    genexpr = (c for c in iterator)
    assert ast.unparse(disassemble(genexpr)) == "(c for c in 'ello')"
    assert materialize(genexpr) == ["e", "l", "l", "o"]


# ============================================================================
# BINARY OPERATORS
# ============================================================================


@pytest.mark.parametrize(
    "genexpr",
    [
        # Bitwise operators, which BINARY_OP folds in with the arithmetic ones
        (x & 3 for x in range(8)),
        (x | 3 for x in range(8)),
        (x ^ 3 for x in range(8)),
        (x << 2 for x in range(4)),
        (x >> 1 for x in range(8)),
        (~x & 7 for x in range(8)),
        # Mixed precedence across the whole operator table
        (x & 1 | x >> 2 ^ 3 for x in range(8)),
        ((x | 1) & (x ^ 2) for x in range(8)),
        (x + 1 & x - 1 for x in range(8)),
        (x * 2 % 5 // 2 for x in range(8)),
        (x**2 - x // 2 + x % 3 for x in range(1, 8)),
        # Operators on non-numeric operands
        (s + "!" for s in ["a", "b"]),
        (s * 2 for s in ["a", "b"]),
        (t + (9,) for t in [(1,), (2,)]),
        (frozenset({x}) | frozenset({9}) for x in range(3)),
        (frozenset({x, 1}) & frozenset({1}) for x in range(3)),
        (frozenset({x, 1}) ^ frozenset({1}) for x in range(3)),
        ({"a": x} | {"b": 0} for x in range(3)),
    ],
)
def test_binary_operators(genexpr):
    """Test reconstruction of the full BINARY_OP table."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


def test_matmul_operator():
    """BINARY_OP argument 4 is `@`, which no built-in type implements.

    The comprehension is disassembled but never evaluated, so this checks the
    reconstructed source rather than the reconstructed values.
    """
    genexpr = (a @ b for a in [1, 2])  # noqa: F821
    assert ast.unparse(disassemble(genexpr)) == "(a @ b for a in (1, 2))"


# ============================================================================
# KEYWORD ARGUMENTS AT CALL SITES
#
# Python 3.12 compiles these as KW_NAMES followed by CALL; 3.13 replaced the
# pair with a single CALL_KW instruction.
# ============================================================================


@pytest.mark.parametrize(
    "genexpr,globals_dict",
    [
        ((dict(a=x) for x in range(3)), {}),
        ((dict(a=x, b=x * 2) for x in range(3)), {}),
        ((dict(a=x if x > 0 else 0, b=x * 2 if x < 5 else x) for x in range(8)), {}),
        # Mixed positional and keyword arguments
        ((sorted([3, x], reverse=True) for x in range(3)), {}),
        ((sorted([x, 1], key=abs) for x in range(3)), {}),
        ((sorted([x, 1], key=abs, reverse=True) for x in range(3)), {}),
        ((int(str(x), base=8) for x in range(8)), {}),
        ((round(x / 3, ndigits=2) for x in range(5)), {}),
        # Keyword arguments on a method call
        (("a,b".split(sep=",") for x in range(2)), {}),
        (("a-b".replace("-", "+") for x in range(2)), {}),
        # Nested calls, each with keywords
        ((dict(a=dict(b=x)) for x in range(3)), {}),
        ((sorted(sorted([x, 1]), reverse=True) for x in range(3)), {}),
        # Keyword arguments to a user-supplied callable
        ((f(x, scale=2) for x in range(3)), {"f": lambda v, scale=1: v * scale}),  # type: ignore  # noqa: F821
    ],
)
def test_keyword_arguments(genexpr, globals_dict):
    """Test reconstruction of calls with keyword arguments."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node, globals_dict)


@pytest.mark.parametrize(
    "genexpr",
    [
        # Starred positional arguments
        (max(*[x, 1]) for x in range(3)),
        (max(1, *[x, 2]) for x in range(3)),
        (max(*[x, 1], *[2, 3]) for x in range(3)),
        (max(*(x, 1)) for x in range(3)),
        (sum([x, 1], *[0]) for x in range(3)),
        # Double-starred keyword arguments
        (dict(**{"a": x}) for x in range(3)),
        (dict(a=x, **{"b": 1}) for x in range(3)),
        (dict(**{"a": x}, **{"b": 2}) for x in range(3)),
        (sorted([x, 1], **{"reverse": True}) for x in range(3)),  # type: ignore[call-overload]
        # Both at once
        (max(*[[x, 1]], **{"default": 0}) for x in range(3)),  # type: ignore[call-overload]
        # On a method call, where the callable comes with a `self`
        ("-".join(*[[str(x), "z"]]) for x in range(3)),
        # Unpacking a comprehension, and unpacking into a nested call
        (max(*[y for y in range(x + 2)]) for x in range(3)),
        (max(*[abs(y) for y in range(-x - 1, 1)]) for x in range(3)),
        (dict(**{str(k): k for k in range(x + 1)}) for x in range(3)),
    ],
)
def test_star_argument_calls(genexpr):
    """Test reconstruction of `*args`/`**kwargs` call sites (CALL_FUNCTION_EX).

    The arguments arrive already collected into a sequence and a mapping, so the
    reconstruction spells every argument as unpacked; that evaluates identically
    even where the source passed some of them plainly.
    """
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# LAMBDAS
# ============================================================================


@pytest.mark.parametrize(
    "genexpr",
    [
        ((lambda y: y * 2)(x) for x in range(5)),
        ((lambda y, z: y + z)(x, x) for x in range(5)),
        ((lambda y: y)(x) for x in range(5)),
        # Closing over the loop variable, and over an enclosing comprehension
        ((lambda y: y + x)(x) for x in range(5)),
        ((lambda: x)() for x in range(5)),
        (((lambda y: lambda z: z + y)(x))(1) for x in range(5)),
        # Lambdas whose body is itself a comprehension
        ((lambda: (y for y in range(x)))() for x in range(3)),
        ((lambda: [y for y in range(x)])() for x in range(3)),
        ((lambda n: sum(y for y in range(n)))(x) for x in range(3)),
        # Lambdas as arguments to other calls
        (sorted([x, 1], key=lambda v: -v) for x in range(3)),
        (list(map(lambda v: v * 2, [x, 1])) for x in range(3)),
        # Conditional expressions inside a lambda body
        ((lambda y: y if y % 2 else -y)(x) for x in range(5)),
        ((lambda y: (y if y > 1 else 0) + 1)(x) for x in range(5)),
    ],
)
def test_lambdas(genexpr):
    """Test reconstruction of lambdas appearing inside comprehensions."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


@pytest.mark.parametrize(
    "genexpr",
    [
        # Positional defaults, which attach to the *trailing* parameters
        ((lambda y, z=2: y * z)(x) for x in range(3)),  # type: ignore[assignment]
        ((lambda y=1: y)() for x in range(3)),  # type: ignore[assignment]
        ((lambda y, z=2, w=3: y * z * w)(x) for x in range(3)),  # type: ignore[assignment]
        ((lambda y, z=2: y * z)(x, 5) for x in range(3)),
        # Keyword-only parameters, with and without defaults
        ((lambda y, *, z=1: y + z)(x) for x in range(3)),  # type: ignore[assignment]
        ((lambda y, *, z=1, w=2: y + z + w)(x) for x in range(3)),  # type: ignore[assignment]
        ((lambda y, *, z: y + z)(x, z=4) for x in range(3)),
        # Positional-only parameters
        ((lambda y, /, z=2: y * z)(x) for x in range(3)),  # type: ignore[assignment]
        # *args and **kwargs
        ((lambda *a: sum(a))(x, x) for x in range(3)),
        ((lambda **k: sum(k.values()))(a=x) for x in range(3)),
        ((lambda *a, **k: len(a) + len(k))(x, b=1) for x in range(3)),
        ((lambda y, *a: y + len(a))(x, 1, 2) for x in range(3)),
        (
            (lambda y, *a, z=3, **k: y + len(a) + z + len(k))(x, 1, w=2)
            for x in range(3)
        ),
        # Defaults that are themselves non-trivial expressions
        ((lambda y, z=(1, 2): y + len(z))(x) for x in range(3)),  # type: ignore[assignment]
        ((lambda y, z=[1]: y + len(z))(x) for x in range(3)),  # type: ignore[assignment]
    ],
)
def test_lambda_default_and_variadic_arguments(genexpr):
    """Test reconstruction of lambda defaults and variadic parameters."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


@pytest.mark.parametrize(
    "genexpr",
    [
        # A lambda reached as a live object -- through the outermost iterable --
        # rather than built inside the comprehension. Its defaults are on the
        # function, not in the code object it was compiled to.
        (fn() for fn in [lambda value=1: value]),
        (fn(2) for fn in [lambda a, b=10: a * b]),
        (fn(2) for fn in (lambda a, b=10: a * b,)),
        (fn(1) for fn in [lambda a, *, k=5: a + k]),
        (fn(1) for fn in [lambda a, /, b=2, *, k=5: a * b + k]),
        (fn() for fn in [lambda x=(1, 2): sum(x)]),
        (fn() for fn in [lambda x=[1, 2]: len(x)]),
        (fn(1, 2) for fn in [lambda a, b=0, *rest, k=5, **kw: a + b + k + len(rest)]),
        # ... several of them, and one with no defaults alongside
        (fn() for fn in [lambda: 0, lambda v=1: v, lambda v=2: v]),  # type: ignore[misc]
        # ... and one passed through map rather than iterated directly
        (fn(3) for fn in map(lambda f: f, [lambda a, b=4: a * b])),
    ],
)
def test_lambda_object_defaults(genexpr):
    """A lambda arriving as a live object keeps the defaults attached to it."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# ASSIGNMENT EXPRESSIONS
# ============================================================================


@pytest.mark.parametrize(
    "genexpr",
    [
        # A walrus binds in the *enclosing* scope, so at module level it
        # compiles to COPY + STORE_GLOBAL rather than to a local.
        (y for x in range(4) if (y := x * 2) > 1),  # noqa: F821
        ((y := x) + 1 for x in range(3)),  # noqa: F821
        (y * y for x in range(4) if (y := x + 1) > 2),  # noqa: F821
        ((y := x) if x > 1 else -1 for x in range(4)),  # noqa: F821
        # Bound in one clause and read in a later one
        ((y, z) for x in range(3) if (y := x + 1) for z in range(y)),  # noqa: F821
        # Inside a nested comprehension, and inside a lambda
        ([(z := w) + z for w in range(x)] for x in range(4)),  # noqa: F821
        ((lambda n: [(z := w) + z for w in range(n)])(x) for x in range(4)),  # noqa: F821
        # Combined with a short-circuiting filter. The `or` must not re-evaluate
        # the assignment, which is why the disjunction is absorbed.
        (y for x in range(6) if (y := x * 2) > 6 or y == 0),  # noqa: F821
    ],
)
def test_assignment_expressions(genexpr):
    """A walrus in a comprehension binds in the *enclosing* scope (STORE_GLOBAL here)."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# COMPREHENSIONS NESTED IN EACH SYNTACTIC POSITION
#
# A comprehension can appear in the element, in the iterable, and inside a
# filter, and each of the four comprehension kinds can nest inside any other.
# The filter position is the interesting one: filters are reconstructed from
# control flow, so a comprehension inside a filter has to survive being treated
# as part of a boolean condition.
# ============================================================================


@pytest.mark.parametrize(
    "genexpr",
    [
        # ... in filter position
        (x for x in range(5) if any(y > 2 for y in range(x))),
        (x for x in range(5) if all(y < 3 for y in range(x))),
        (x for x in range(5) if [y for y in range(x)]),
        (x for x in range(5) if {y for y in range(x)}),
        (x for x in range(5) if {y: y for y in range(x)}),
        (x for x in range(6) if len([y for y in range(x) if y % 2]) > 1),
        (x for x in range(5) if any(y for y in range(x) if y % 2)),
        (
            x
            for x in range(6)
            if all(y < x for y in range(2)) and any(z > 1 for z in range(x))
        ),  # noqa: E501  # fmt: skip
        # ... in filter position, inside a short-circuiting condition
        (x for x in range(6) if sum(y for y in range(x)) > 3 or x == 0),
        (x for x in range(6) if x == 0 or any(y > 2 for y in range(x))),
        (x for x in range(6) if x == 0 or len([y for y in range(x)]) > 2),
        (
            x
            for x in range(6)
            if any(y > 1 for y in range(x)) or all(z < 2 for z in range(x))
        ),  # noqa: E501  # fmt: skip
        # ... in iterable position
        (x for x in [y for y in range(5) if y % 2]),
        (x for x in {y for y in range(5) if y % 2}),
        (x for x in {y: y for y in range(3)}),
        (x for x in (y for y in range(5) if y > 1 or y == 0)),
        (x for x in [y for y in [z for z in range(4)] if y % 2]),
        (x for x in [y for y in range(4)] if x > 1 or x == 0),
        ((a, b) for a in range(3) for b in [c for c in range(a)]),
        # ... in element position
        ([y for y in range(x) if y % 2 or y == 0] for x in range(4)),
        ({y for y in range(x) if y > 1} for x in range(5)),
        ({y: [z for z in range(y)] for y in range(x)} for x in range(4)),
        (sum(y for y in range(x) if y % 2) for x in range(5)),
        ([(z for z in range(y)) for y in range(x)] for x in range(3)),
        # ... in several positions at once, with different kinds
        (
            [y for y in {z for z in range(x)}]
            for x in range(4)
            if any(w > 1 for w in range(x))
        ),  # noqa: E501  # fmt: skip
        ({k: [v for v in range(k)] for k in {j for j in range(x)}} for x in range(4)),
        ([y for y in range(x) if y or y == 0] for x in range(5) if x > 1 or x == 0),
        (
            (y for y in range(x) if y % 2 or y == 0)
            for x in (z for z in range(4))
            if x < 3 or x == 3
        ),  # noqa: E501  # fmt: skip
        # ... nesting the four kinds inside one another
        ({k: {v for v in range(k)} for k in [j for j in range(x)]} for x in range(4)),
        ([{y: y} for y in range(x)] for x in range(4)),
        ({(y, y * 2) for y in range(x)} for x in range(4)),
        ({y: y for y in range(x) if y % 2 or y == 0} for x in range(5)),
        ({y for y in range(x) if y % 2 or y == 0} for x in range(5)),
        ([y for y in range(x) if 1 < y < 4] for x in range(6)),
    ],
)
def test_comprehensions_in_every_position(genexpr):
    """Test comprehensions nested in the element, the iterable and the filter."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


@pytest.mark.parametrize(
    "genexpr",
    [
        # A lambda body is a separate code object with no loop of its own, so
        # every branch inside one is a conditional expression rather than a
        # filter -- including the branches of a comprehension nested in it.
        ((lambda n: [y for y in range(n) if y % 2])(x) for x in range(4)),
        ((lambda n: {y for y in range(n)})(x) for x in range(4)),
        ((lambda n: {y: y**2 for y in range(n)})(x) for x in range(4)),
        ((lambda n: sum(y for y in range(n)))(x) for x in range(4)),
        ((lambda n: [y for y in range(n) if y > 1 or y == 0])(x) for x in range(5)),
        ((lambda n: (y for y in range(n) if y % 2 or y == 0))(x) for x in range(4)),
        ((lambda n: [y if y > 1 else -y for y in range(n)])(x) for x in range(4)),
        ((lambda n: [y for y in range(n) if 1 < y < 3])(x) for x in range(5)),
        # Lambdas nested in lambdas, and lambdas inside the comprehension body
        ((lambda n: (lambda m: [y for y in range(m)])(n))(x) for x in range(3)),
        ([(lambda v: v * 2)(y) for y in range(x)] for x in range(4)),
        (list(map(lambda n: [y for y in range(n)], range(x))) for x in range(3)),
        ([(lambda v: v if v > 1 else -v)(y) for y in range(x)] for x in range(4)),
    ],
)
def test_comprehensions_inside_lambdas(genexpr):
    """Test comprehensions nested inside lambda bodies."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# CLOSURES
#
# A comprehension written inside a function reads the function's locals out of
# closure cells, not out of globals. Those cells belong to a scope the
# reconstruction does not reproduce, so leaving a free variable as a bare name
# would silently turn it into a global lookup -- picking up a different value,
# or none at all. The captured value is written into the tree instead. Each
# test below evaluates the reconstruction in a namespace that binds the same
# name to something else, so a lookup that leaked out would be caught.
# ============================================================================


def test_closure_shadowed_by_global():
    def make():
        value = 1
        return (value for _ in range(1))

    genexpr = make()
    reconstructed = disassemble(genexpr)
    assert ast.unparse(reconstructed) == "(1 for _ in range(0, 1, 1))"
    assert materialize(genexpr) == [1]
    assert materialize(compile_and_eval(reconstructed, {"value": 2})) == [1]


@pytest.mark.parametrize(
    "make,shadow,expected",
    [
        # In the element expression, the filter, and an inner iterable
        (lambda: (lambda n: (x * n for x in range(4)))(3), {"n": 100}, [0, 3, 6, 9]),
        (
            lambda: (lambda t: (x for x in range(6) if x > t))(3),
            {"t": -1},
            [4, 5],
        ),
        (
            lambda: (lambda k: (y for x in range(2) for y in range(k)))(2),
            {"k": 5},
            [0, 1, 0, 1],
        ),
        # A captured container, indexed and iterated
        (
            lambda: (lambda d: (d[i] for i in range(2)))([10, 20]),
            {"d": [0, 0]},
            [10, 20],
        ),
        (
            lambda: (lambda d: (v for v in d))({"a": 1, "b": 2}),
            {"d": {}},
            ["a", "b"],
        ),
        # Captured by a lambda nested inside the comprehension
        (
            lambda: (lambda n: ((lambda y: y * n)(x) for x in range(4)))(3),
            {"n": 100},
            [0, 3, 6, 9],
        ),
        (
            lambda: (lambda n: ((lambda: (lambda: n)())() for _ in range(2)))(7),
            {"n": 0},
            [7, 7],
        ),
        # Captured by a comprehension nested inside the comprehension
        (
            lambda: (lambda n: ([y * n for y in range(x)] for x in range(3)))(2),
            {"n": 100},
            [[], [0], [0, 2]],
        ),
        (
            lambda: (lambda n: (sum(y for y in range(x) if y < n) for x in range(4)))(
                2
            ),
            {"n": 100},
            [0, 0, 1, 1],
        ),
        # Two free variables at once
        (
            lambda: (lambda a, b: (x * a + b for x in range(3)))(2, 1),
            {"a": 0, "b": 0},
            [1, 3, 5],
        ),
    ],
)
def test_closure_free_variables(make, shadow, expected):
    genexpr = make()
    reconstructed = disassemble(genexpr)
    assert materialize(genexpr) == expected
    assert materialize(compile_and_eval(reconstructed, dict(shadow))) == expected
    # The name must not survive anywhere in the tree, or the shadowing binding
    # above would have been the one that answered.
    assert not (
        {node.id for node in ast.walk(reconstructed) if isinstance(node, ast.Name)}
        & set(shadow)
    )


def test_closure_target_captured_by_nested_lambda_stays_a_name():
    """A cell this comprehension *creates* is bound in the reconstruction too."""
    genexpr = ((lambda: x)() for x in range(3))
    reconstructed = disassemble(genexpr)
    assert materialize(genexpr) == [0, 1, 2]
    assert materialize(compile_and_eval(reconstructed, {"x": 99})) == [0, 1, 2]


def test_closure_shadowed_by_a_nested_comprehension_target():
    """Only the free `n` is the captured one; the inner comprehension rebinds it."""

    def make():
        n = 5
        return (n + sum(n for n in range(x)) for x in range(3))

    genexpr = make()
    reconstructed = disassemble(genexpr)
    assert materialize(genexpr) == [5, 5, 6]
    assert materialize(compile_and_eval(reconstructed, {"n": 100, "sum": sum})) == [
        5,
        5,
        6,
    ]


def test_closure_shadowed_by_a_nested_lambda_parameter():
    """Only the free `n` is the captured one; the lambda's parameter is its own."""

    def make():
        n = 5
        return ((lambda n: n * 2)(x) + n for x in range(3))

    genexpr = make()
    reconstructed = disassemble(genexpr)
    assert materialize(genexpr) == [5, 7, 9]
    assert materialize(compile_and_eval(reconstructed, {"n": 100})) == [5, 7, 9]


def test_closure_inside_a_live_lambda():
    """A lambda reached as an object closes over cells of its own."""

    def make():
        n = 3
        return (fn(2) for fn in [lambda a, b=10: a * n + b])

    genexpr = make()
    reconstructed = disassemble(genexpr)
    assert materialize(genexpr) == [16]
    assert materialize(compile_and_eval(reconstructed, {"n": 100})) == [16]


def test_closure_value_that_cannot_be_represented():
    """A capture with no AST spelling is refused rather than quietly dropped."""

    class Opaque:
        pass

    def make():
        obj = Opaque()
        return (obj for _ in range(1))

    with pytest.raises(TypeError, match="captured in free variable 'obj'"):
        disassemble(make())


def test_closure_captured_iterator_is_refused():
    """An iterator's remaining elements are not the iterator, so it is refused."""

    def make():
        flags = iter([True, False, True, False])
        return (x for x in range(4) if next(flags))

    with pytest.raises(TypeError, match="captured in free variable 'flags'"):
        disassemble(make())


# ============================================================================
# STATEFUL EXPRESSIONS
#
# What comes back is syntax, so evaluating it runs the comprehension a second
# time. A filter or element expression that depends on state that has since
# moved on answers differently then -- faithfully reconstructed, but no longer
# in agreement with the generator it came from.
# ============================================================================


_FLAGS = iter([True, False, True, False, False, False, True, False])


def test_stateful_filter_is_re_evaluated():
    genexpr = (x for x in range(4) if next(_FLAGS))
    reconstructed = disassemble(genexpr)
    assert ast.unparse(reconstructed) == "(x for x in range(0, 4, 1) if next(_FLAGS))"

    # The reconstruction is the same comprehension, but `_FLAGS` has advanced by
    # the time it runs, so the two do not agree element for element.
    assert materialize(genexpr) == [0, 2]
    assert materialize(
        compile_and_eval(reconstructed, {"_FLAGS": _FLAGS, "next": next})
    ) == [2]


@pytest.mark.parametrize(
    "genexpr",
    [
        # Short-circuiting conditions nested in one another
        (x for x in range(20) if (x > 2 or x < 1) and (x < 10 or x > 15)),
        (x for x in range(20) if ((x > 2 and x < 5) or (x > 10 and x < 15)) or x == 0),
        (x for x in range(30) if not (x % 2 == 0 or x % 3 == 0)),
        (x for x in range(30) if not (not (x > 5) or not (x < 20))),
        (x for x in range(40) if (x > 5 and x < 35) and (x % 3 == 0 or x % 5 == 0)),
        (
            x
            for x in range(40)
            if (x < 5 and x % 2 == 0) or (10 < x < 15) or (x > 35 and x % 3 == 0)
        ),  # noqa: E501  # fmt: skip
        # Short-circuiting conditions spanning several generators
        (
            (x, y)
            for x in range(6)
            if x < 2 or x > 4
            for y in range(6)
            if y < 1 or y > 4
        ),
        (
            (x, y)
            for x in range(5)
            if x % 2 == 0 or x == 1
            for y in range(x)
            if y > 0 and y < 3
        ),  # noqa: E501  # fmt: skip
        # Conditional expressions and filters that are both lazy
        ((x if (x > 2 or x < 1) else -x) for x in range(10) if x % 2 == 0 or x == 1),
        (x for x in range(20) if (x if x > 5 else not x) or x == 3),
        (
            (x if x > 5 or x < 2 else (0 if x % 2 == 0 or x == 3 else 1))
            for x in range(12)
        ),  # noqa: E501  # fmt: skip
        # Chained comparisons combined with lazy operators
        (x for x in range(30) if 5 < x < 15 or 20 < x < 25),
        (x for x in range(30) if 5 < x < 15 and (x % 2 == 0 or x % 3 == 0)),
        ((x if 5 < x < 15 else 0) for x in range(20) if 2 < x < 18),
        # Lazy conditions inside a nested comprehension, and around it
        ([y for y in range(x) if y > 1 or y == 0] for x in range(5) if x > 2 or x == 0),
        ((y for y in range(x) if y % 2 or y == 0) for x in range(4) if x < 3 or x == 3),
    ],
)
def test_nested_lazy_conditions(genexpr):
    """Test short-circuiting conditions nested inside one another."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# STRUCTURAL STRESS CASES
# ============================================================================

# These two must stay on one line: on Python 3.12 `dis` mis-reports jumps for
# multiline comprehensions, which test_multiline_comprehensions covers directly.
_STRESS_MANY_FILTERS = ((x, y) for x in range(10) if x % 2 == 0 if x > 2 for y in range(10) if y % 3 == 0 if y < x)  # fmt: skip
_STRESS_NESTED_TERNARY = ([y if y > 1 else -y for y in range(x)] for x in range(4) if (x if x % 2 == 1 else x % 2 == 0))  # fmt: skip


@pytest.mark.parametrize(
    "genexpr",
    [
        # Deep loop nesting
        (
            a + b + c + d + e
            for a in range(2)
            for b in range(2)
            for c in range(2)
            for d in range(2)
            for e in range(2)
        ),
        (
            (a, b, c, d)
            for a in range(2)
            for b in range(a + 1)
            for c in range(b + 1)
            for d in range(c + 1)
        ),
        # Many filters spread over many loops. Kept on one line: on Python 3.12
        # `dis` mis-reports jumps for multiline comprehensions, which is covered
        # separately by test_multiline_comprehensions below.
        (x for x in range(50) if x > 5 if x < 40 if x % 2 == 0 if x % 3 == 0),
        _STRESS_MANY_FILTERS,
        # Deep comprehension nesting
        (((z for z in range(y)) for y in range(x)) for x in range(3)),
        ([[z for z in range(y)] for y in range(x)] for x in range(3)),
        ({y: [z for z in range(y)] for y in range(x)} for x in range(3)),
        # Structured literals in the element position
        ({x, x + 1} for x in range(3)),
        ({x: x + 1} for x in range(3)),
        (((x, x), x) for x in range(3)),
        ([x, [x, [x]]] for x in range(3)),
        ({"k": [x, {"j": (x,)}]} for x in range(3)),
        # Ternaries interleaved with nesting
        _STRESS_NESTED_TERNARY,
        (((y if y else -1) for y in range(x)) for x in range(3)),
    ],
)
def test_structural_stress(genexpr):
    """Test reconstruction of deeply nested and heavily filtered comprehensions."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# MULTILINE COMPREHENSIONS
#
# On Python 3.12 `dis` reports a different jump layout for a filter whose source
# spans several lines. The reconstruction used to come out negated, because the
# filter/conditional distinction was drawn from the *local* instruction order.
# Classifying branches from the control-flow graph instead is insensitive to
# that, so these now reconstruct identically on 3.12 and 3.13+.
# ============================================================================


@pytest.mark.parametrize(
    "genexpr",
    [
        (
            x
            for x in range(5)  # comment to avoid reformatting
            if x > 1
        ),
        (
            x
            for x in range(10)  # comment to avoid reformatting
            if x % 2 == 0
            if x > 2
        ),
        (
            (x, y)
            for x in range(10)
            if x % 2 == 0
            if x > 2
            for y in range(10)
            if y % 3 == 0
            if y < x
        ),
        (
            [y if y > 1 else -y for y in range(x)]
            for x in range(4)
            if (x if x % 2 == 1 else x % 2 == 0)
        ),
    ],
)
def test_multiline_comprehensions(genexpr):
    """Filters in multiline comprehensions are mis-disassembled on Python 3.12."""
    ast_node = disassemble(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


def test_multiline_comprehensions_same_on_one_line():
    """The same expressions reconstruct correctly when written on one line."""
    one_line = (x for x in range(10) if x % 2 == 0 if x > 2)
    assert_ast_equivalent(one_line, disassemble(one_line))


# ============================================================================
# HELPER FUNCTION TESTS
# ============================================================================


@pytest.mark.parametrize(
    "value,expected_str",
    [
        # AST nodes should be returned as-is
        (ast.Name(id="x", ctx=ast.Load()), "x"),
        (ast.Constant(value=42), "42"),
        (ast.List(elts=[], ctx=ast.Load()), "[]"),
        (
            ast.BinOp(
                left=ast.Constant(value=1), op=ast.Add(), right=ast.Constant(value=2)
            ),
            "1 + 2",
        ),
        # Constants should become ast.Constant nodes
        (42, "42"),
        (3.14, "3.14"),
        (-42, "-42"),
        (-3.14, "-3.14"),
        ("hello", "'hello'"),
        ("", "''"),
        (b"bytes", "b'bytes'"),
        (b"", "b''"),
        (True, "True"),
        (False, "False"),
        (None, "None"),
        # Complex numbers
        (1 + 2j, "(1+2j)"),
        (0 + 1j, "1j"),
        (3 + 0j, "(3+0j)"),
        (-1 - 2j, "(-1-2j)"),
        # Tuples should become ast.Tuple nodes
        ((), "()"),
        ((1,), "(1,)"),
        ((1, 2), "(1, 2)"),
        (("a", "b", "c"), "('a', 'b', 'c')"),
        # A tuple is a tuple: no element of one is a marker to be stripped
        (("dict_item", "key", "value"), "('dict_item', 'key', 'value')"),
        (("dict_item", 42, "answer"), "('dict_item', 42, 'answer')"),
        # Nested tuples
        ((1, (2, 3)), "(1, (2, 3))"),
        (((1, 2), (3, 4)), "((1, 2), (3, 4))"),
        ((1, 2, (3, (4, 5))), "(1, 2, (3, (4, 5)))"),
        # Lists should become ast.List nodes
        ([1, 2, 3], "[1, 2, 3]"),
        (["hello", "world"], "['hello', 'world']"),
        ([True, False, None], "[True, False, None]"),
        # Nested lists
        ([[1, 2], [3, 4]], "[[1, 2], [3, 4]]"),
        ([1, [2, [3, 4]], 5], "[1, [2, [3, 4]], 5]"),
        # Mixed nested structures
        ([(1, 2), (3, 4)], "[(1, 2), (3, 4)]"),
        (([1, 2], [3, 4]), "([1, 2], [3, 4])"),
        # Dicts should become ast.Dict nodes
        ({"a": 1}, "{'a': 1}"),
        ({"x": 10, "y": 20}, "{'x': 10, 'y': 20}"),
        ({1: "one", 2: "two"}, "{1: 'one', 2: 'two'}"),
        # Nested dicts
        ({"a": {"b": 1}}, "{'a': {'b': 1}}"),
        (
            {"nums": [1, 2, 3], "strs": ["a", "b"]},
            "{'nums': [1, 2, 3], 'strs': ['a', 'b']}",
        ),
        # Range objects
        (range(5), "range(0, 5, 1)"),
        (range(1, 10), "range(1, 10, 1)"),
        (range(0, 10, 2), "range(0, 10, 2)"),
        (range(10, 0, -1), "range(10, 0, -1)"),
        (range(-5, 5), "range(-5, 5, 1)"),
        # Empty collections
        ([], "[]"),
        ((), "()"),
        ({}, "{}"),
        # Complex nested structures
        ([1, [2, 3], 4], "[1, [2, 3], 4]"),
        ({"a": [1, 2], "b": {"c": 3}}, "{'a': [1, 2], 'b': {'c': 3}}"),
        ([(1, {"a": [2, 3]}), ({"b": 4}, 5)], "[(1, {'a': [2, 3]}), ({'b': 4}, 5)]"),
        # Edge cases with special values
        ([None, True, False, 0, ""], "[None, True, False, 0, '']"),
        (
            {"": "empty", None: "none", 0: "zero"},
            "{'': 'empty', None: 'none', 0: 'zero'}",
        ),
        # Large numbers
        (999999999999999999999, "999999999999999999999"),
        (1.7976931348623157e308, "1.7976931348623157e+308"),  # Close to float max
        # Sets - note unparse equivalence may fail for unordered collections
        ({1, 2, 3}, "{1, 2, 3}"),
    ],
)
def test_ensure_ast(value, expected_str):
    """Test that ensure_ast correctly converts various values to AST nodes."""

    result = ensure_ast(value)

    # Compare the unparsed strings
    result_str = ast.unparse(result)
    assert result_str == expected_str, (
        f"ensure_ast({repr(value)}) produced '{result_str}', expected '{expected_str}'"
    )


def test_error_handling():
    """Test that appropriate errors are raised for unsupported cases."""
    # Test with non-generator input
    with pytest.raises(ValueError):
        disassemble([1, 2, 3])  # Not a generator

    # Test with consumed generator
    gen = (x for x in range(5))
    list(gen)  # Consume it
    with pytest.raises(ValueError):
        disassemble(gen)

    # Test with a generator that has been started but not consumed
    gen = (x for x in range(5))
    next(gen)
    with pytest.raises(ValueError):
        disassemble(gen)


def test_comp_lambda_copy():
    """Test that CompLambda is compatible with copy.copy and copy.deepcopy."""
    # Create a test generator expression AST
    genexpr_ast = ast.GeneratorExp(
        elt=ast.Name(id="x", ctx=ast.Load()),
        generators=[
            ast.comprehension(
                target=ast.Name(id="x", ctx=ast.Store()),
                iter=DummyIterName(),
                ifs=[],
                is_async=0,
            )
        ],
    )

    # Create a CompLambda instance
    comp_lambda = CompLambda(genexpr_ast)

    # Test copy.copy
    copied = copy.copy(comp_lambda)
    assert isinstance(copied, CompLambda)
    assert ast.unparse(copied.body) == ast.unparse(comp_lambda.body)
    assert copied.body is comp_lambda.body  # Shallow copy shares the body

    # Test copy.deepcopy
    deep_copied = copy.deepcopy(comp_lambda)
    assert isinstance(deep_copied, CompLambda)
    assert ast.unparse(deep_copied.body) == ast.unparse(comp_lambda.body)
    assert deep_copied.body is not comp_lambda.body  # Deep copy creates new body

    # Test that deep copied version works the same way
    iterator = ast.Call(
        func=ast.Name(id="range", ctx=ast.Load()),
        args=[ast.Constant(value=5)],
        keywords=[],
    )

    original_result = comp_lambda.inline(iterator)
    deep_copied_result = deep_copied.inline(iterator)

    assert ast.unparse(original_result) == ast.unparse(deep_copied_result)
    assert type(original_result) == type(deep_copied_result)


# ============================================================================
# AST TRANSFORMER TESTS
# ============================================================================
