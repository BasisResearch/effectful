import ast
from types import GeneratorType
from typing import Any

import pytest

from effectful.internals.disassembler import (
    GeneratorExpToForexpr,
    NameToCall,
    ensure_ast,
    reconstruct,
)


def compile_and_eval(
    node: ast.expr | ast.Expression, globals_dict: dict | None = None
) -> Any:
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


def assert_ast_equivalent(
    genexpr: GeneratorType, reconstructed_ast: ast.AST, globals_dict: dict | None = None
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
    original_list = list(genexpr)

    # Clean up globals to avoid pollution
    for key in globals_dict or {}:
        if key not in curr_globals:
            del globals()[key]
    globals().update(curr_globals)

    # Compile and evaluate the reconstructed AST
    reconstructed_gen = compile_and_eval(reconstructed_ast, globals_dict)
    reconstructed_list = list(reconstructed_gen)
    assert (
        reconstructed_list == original_list
    ), f"AST produced {reconstructed_list}, expected {original_list}"


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
    ast_node = reconstruct(genexpr)
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
        (x + y * 2 for x in range(3) for y in range(3)),
        (x * 2 + y / 3 for x in range(1, 4) for y in range(1, 4)),
        ((x + y) * (x - y) for x in range(1, 4) for y in range(1, 4)),
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
    ast_node = reconstruct(genexpr)
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
        pytest.param(
            (x for x in range(10) if x < 3 or x > 7),
            marks=pytest.mark.xfail(reason="Lambda reconstruction not implemented yet"),
        ),
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
        pytest.param(
            (x for x in range(20) if x < 5 or x > 15 or x == 10),
            marks=pytest.mark.xfail(reason="Lambda reconstruction not implemented yet"),
        ),
        pytest.param(
            (x for x in range(20) if not (x > 5 and x < 15)),
            marks=pytest.mark.xfail(reason="Lambda reconstruction not implemented yet"),
        ),
        # Mixed comparison and boolean operations
        pytest.param(
            (x for x in range(20) if (x > 10 and x % 2 == 0) or (x < 5 and x % 3 == 0)),
            marks=pytest.mark.xfail(reason="Lambda reconstruction not implemented yet"),
        ),
        pytest.param(
            (x for x in range(20) if not (x % 2 == 0 and x % 3 == 0)),
            marks=pytest.mark.xfail(reason="Lambda reconstruction not implemented yet"),
        ),
        # Edge cases with identity comparisons
        (x for x in [0, 1, 2, None, 4] if x is not None and x > 1),
        (x for x in [True, False, 1, 0] if x is True),
        (x for x in [True, False, 1, 0] if x is not False),
    ],
)
def test_comparison_operators(genexpr):
    """Test reconstruction of all comparison operators."""
    ast_node = reconstruct(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# CHAINED COMPARISON TESTS
# ============================================================================


@pytest.mark.xfail(reason="Chained comparisons not yet fully supported")
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
    ast_node = reconstruct(genexpr)
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
        pytest.param(
            (x for x in range(10) if x < 3 or x > 7),
            marks=pytest.mark.xfail(reason="Lazy conjunctions not implemented yet"),
        ),
        # More complex filter edge cases
        (x for x in range(50) if x % 7 == 0),  # Different modulo
        (x for x in range(10) if x >= 0),  # Always true condition
        (x for x in range(10) if x < 0),  # Always false condition
        (
            x for x in range(20) if x % 2 == 0 and x % 3 == 0
        ),  # Multiple conditions with and
        pytest.param(
            (x for x in range(20) if x % 2 == 0 or x % 3 == 0),
            marks=pytest.mark.xfail(reason="Lazy conjunctions not implemented yet"),
        ),  # Multiple conditions with or
        # Nested boolean operations
        pytest.param(
            (x for x in range(20) if (x > 5 and x < 15) or x == 0),
            marks=pytest.mark.xfail(reason="Lazy conjunctions not implemented yet"),
        ),
        pytest.param(
            (x for x in range(20) if not (x > 10 and x < 15)),
            marks=pytest.mark.xfail(reason="Lazy conjunctions not implemented yet"),
        ),
        pytest.param(
            (x for x in range(50) if x > 10 and (x % 2 == 0 or x % 3 == 0)),
            marks=pytest.mark.xfail(reason="Lazy conjunctions not implemented yet"),
        ),
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
    ast_node = reconstruct(genexpr)
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
        ((x, y, z) for x in range(2) for y in range(2) for z in range(2)),
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
        (
            (x, y, z)
            for x in range(5)
            for y in range(5)
            for z in range(5)
            if x < y and y < z
        ),
        (x + y for x in range(3) if x > 0 for y in range(3)),
        # Mixed range types
        ((x, y) for x in range(-2, 2) for y in range(0, 4, 2)),
        (x * y for x in range(5, 0, -1) for y in range(1, 6)),
        # Dependent nested loops
        ((x, y) for x in range(3) for y in range(x, 3)),
        (x + y for x in range(3) for y in range(x + 1, 3)),
        (
            x * y * z
            for x in range(3)
            for y in range(x + 1, x + 3)
            for z in range(y, y + 3)
        ),
    ],
)
def test_nested_loops(genexpr):
    """Test reconstruction of generators with nested loops."""
    ast_node = reconstruct(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ===========================================================================
# NESTED COMPREHENSIONS
# ===========================================================================


@pytest.mark.parametrize(
    "genexpr",
    [
        ([x for x in range(i)] for i in range(5)),
        ({x: x**2 for x in range(i)} for i in range(5)),
        ([[x for x in range(i + j)] for j in range(i)] for i in range(5)),
        # aggregation function call
        (sum(x for x in range(i + 1)) for i in range(3)),
        (max(x for x in range(i + 1)) for i in range(3)),
        # map
        (list(map(abs, (x + 1 for x in range(i + 1)))) for i in range(3)),
        (list(enumerate(x + 1 for x in range(i + 1))) for i in range(3)),
        # Nested comprehensions with filters inside
        ([x for x in range(i)] for i in range(5) if i > 0),
        ([x for x in range(i) if x < i] for i in range(5) if i > 0),
        ([[x for x in range(i + j) if x < i + j] for j in range(i)] for i in range(5)),
        (
            [[x for x in range(i + j) if x < i + j] for j in range(i)]
            for i in range(5)
            if i > 0
        ),
        # nesting on both sides
        ([y for y in range(x)] for x in (x_ + 1 for x_ in range(5))),
        ([y for y in range(x)] for x in (x_ + 1 for x_ in range(5))),
    ],
)
def test_nested_comprehensions(genexpr):
    """Test reconstruction of nested comprehensions."""
    ast_node = reconstruct(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


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
        ([y for y in range(x + 1)] for x in range(3)),
        ({y for y in range(x + 1)} for x in range(3)),
        ({y: y**2 for y in range(x + 1)} for x in range(3)),
    ],
)
def test_different_comprehension_types(genexpr):
    """Test reconstruction of different comprehension types."""
    ast_node = reconstruct(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# GENERATOR EXPRESSION WITH GLOBALS
# ============================================================================


@pytest.mark.parametrize(
    "genexpr,globals_dict",
    [
        # Using constants
        ((x + a for x in range(5)), {"a": 10}),  # noqa: F821
        ((data[i] for i in range(2)), {"data": [3, 4]}),  # noqa: F821
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
    ast_node = reconstruct(genexpr)

    # Need to provide the same globals for evaluation
    assert_ast_equivalent(genexpr, ast_node, globals_dict)


# ============================================================================
# EDGE CASES AND COMPLEX SCENARIOS
# ============================================================================


@pytest.mark.parametrize(
    "genexpr,globals_dict",
    [
        # Using lambdas and functions
        pytest.param(
            ((lambda y: y * 2)(x) for x in range(5)),
            {},
            marks=pytest.mark.xfail(reason="Lambda reconstruction not implemented yet"),
        ),
        pytest.param(
            ((lambda y: y + 1)(x) for x in range(5)),
            {},
            marks=pytest.mark.xfail(reason="Lambda reconstruction not implemented yet"),
        ),
        pytest.param(
            ((lambda y: y**2)(x) for x in range(5)),
            {},
            marks=pytest.mark.xfail(reason="Lambda reconstruction not implemented yet"),
        ),
        # More complex lambdas
        # (((lambda a, b: a + b)(x, x) for x in range(5)), {}),
        ((f(x) for x in range(5)), {"f": lambda y: y * 3}),  # noqa: F821
        # Attribute access
        ((x.real for x in [1 + 2j, 3 + 4j, 5 + 6j]), {}),
        ((x.imag for x in [1 + 2j, 3 + 4j, 5 + 6j]), {}),
        ((x.conjugate() for x in [1 + 2j, 3 + 4j, 5 + 6j]), {}),
        # Method calls
        ((s.upper() for s in ["hello", "world"]), {}),
        ((s.lower() for s in ["HELLO", "WORLD"]), {}),
        ((s.strip() for s in [" hello ", "  world  "]), {}),
        ((x.bit_length() for x in range(1, 10)), {}),
        ((str(x).zfill(3) for x in range(10)), {"str": str}),
        # Subscript operations
        (([10, 20, 30][i] for i in range(3)), {}),
        (({"a": 1, "b": 2, "c": 3}[k] for k in ["a", "b", "c"]), {}),
        (("hello"[i] for i in range(5)), {}),
        ((data[i][j] for i in range(2) for j in range(2)), {"data": [[1, 2], [3, 4]]}),  # noqa: F821
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
        (([1, 2, 3][x % 3] for x in range(10)), {}),
        # (({"even": x, "odd": x + 1}["even" if x % 2 == 0 else "odd"] for x in range(5)), {}),
        # Function calls with multiple arguments
        ((pow(x, 2, 10) for x in range(5)), {"pow": pow}),
        ((divmod(x, 3) for x in range(10)), {"divmod": divmod}),
    ],
)
def test_complex_scenarios(genexpr, globals_dict):
    """Test reconstruction of complex generator expressions."""
    ast_node = reconstruct(genexpr)

    # Need to provide the same globals for evaluation
    assert_ast_equivalent(genexpr, ast_node, globals_dict)


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
        # Special dict_item tuples
        (("dict_item", "key", "value"), "('key', 'value')"),
        (("dict_item", 42, "answer"), "(42, 'answer')"),
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
    assert (
        result_str == expected_str
    ), f"ensure_ast({repr(value)}) produced '{result_str}', expected '{expected_str}'"


def test_error_handling():
    """Test that appropriate errors are raised for unsupported cases."""
    # Test with non-generator input
    with pytest.raises(AssertionError):
        reconstruct([1, 2, 3])  # Not a generator

    # Test with consumed generator
    gen = (x for x in range(5))
    list(gen)  # Consume it
    with pytest.raises(AssertionError):
        reconstruct(gen)


# ============================================================================
# AST TRANSFORMER TESTS
# ============================================================================


@pytest.mark.parametrize(
    "source_src,varnames,expected_src",
    [
        # Simple name replacement
        ("x", {"x"}, "x()"),
        ("x + y", {"x"}, "x() + y"),
        ("x + y", {"y"}, "x + y()"),
        ("x + y", {"x", "y"}, "x() + y()"),
        # Names in different contexts (Note: Store/Del contexts work differently)
        # These would need to be parsed as statements, not expressions
        # Skipping assignment and del tests as they can't be parsed in eval mode
        # Complex expressions
        ("x * 2 + y / 3", {"x", "y"}, "x() * 2 + y() / 3"),
        ("func(x, y, z)", {"x", "z"}, "func(x(), y, z())"),
        # Nested expressions
        ("x + (y * z)", {"x", "y", "z"}, "x() + (y() * z())"),
        ("[x, y, z]", {"x", "y"}, "[x(), y(), z]"),
        ("{'a': x, 'b': y}", {"x", "y"}, "{'a': x(), 'b': y()}"),
        # Attribute access
        ("x.attr", {"x"}, "x().attr"),
        ("x.method()", {"x"}, "x().method()"),
        ("obj.x", {"x"}, "obj.x"),  # x is an attribute, not a variable
        # Subscript operations
        ("x[0]", {"x"}, "x()[0]"),
        ("arr[x]", {"x"}, "arr[x()]"),
        ("x[y]", {"x", "y"}, "x()[y()]"),
        # Function calls
        ("f(x)", {"f"}, "f()(x)"),
        ("f(x)", {"x"}, "f(x())"),
        ("f(x, y)", {"f", "x", "y"}, "f()(x(), y())"),
        # Lambda expressions
        ("lambda a: x + a", {"x"}, "lambda a: x() + a"),
        (
            "lambda x: x + y",
            {"x", "y"},
            "lambda x: x() + y()",
        ),  # Transformer doesn't check scope
        # Comprehensions
        ("[x for i in range(3)]", {"x"}, "[x() for i in range(3)]"),
        ("[i for i in x]", {"x"}, "[i for i in x()]"),
        (
            "[x for x in range(3)]",
            {"x"},
            "[x() for x in range(3)]",
        ),  # Transformer doesn't check scope
        ("(x + i for i in range(3))", {"x"}, "(x() + i for i in range(3))"),
        # Multiple occurrences
        ("x + x * x", {"x"}, "x() + x() * x()"),
        # Names not in the set should not be transformed
        ("x + y + z", {"x"}, "x() + y + z"),
        ("a + b + c", {"x", "y", "z"}, "a + b + c"),
    ],
)
def test_name_to_call_transformer(source_src, varnames, expected_src):
    """Test NameToCall transformer converts specified names to function calls."""
    # Parse the source code
    source_ast = ast.parse(source_src, mode="eval")
    expected_ast = ast.parse(expected_src, mode="eval")

    # Apply the transformer
    transformer = NameToCall(varnames)
    transformed_ast = transformer.visit(source_ast)

    # Compare the unparsed strings
    transformed_src = ast.unparse(transformed_ast)
    expected_unparsed = ast.unparse(expected_ast)

    assert transformed_src == expected_unparsed, (
        f"Transformation failed:\n"
        f"Input: {source_src}\n"
        f"Varnames: {varnames}\n"
        f"Expected: {expected_unparsed}\n"
        f"Got: {transformed_src}"
    )


@pytest.mark.parametrize(
    "source_src,varnames,expected_src",
    [
        # Nested comprehensions in body
        ("[x for x in range(i)]", {"i"}, "[x for x in range(i())]"),
        (
            "[[x for x in range(j)] for j in range(i)]",
            {"i"},
            "[[x for x in range(j)] for j in range(i())]",
        ),
        (
            "[[x for x in range(j)] for j in range(i)]",
            {"i", "j"},
            "[[x for x in range(j())] for j in range(i())]",
        ),
        # Complex nested comprehensions
        ("sum(x for x in range(i))", {"i"}, "sum((x for x in range(i())))"),
        ("sum(x for x in range(i))", {"x"}, "sum((x() for x in range(i)))"),
        ("sum(x for x in range(i))", {"i", "x"}, "sum((x() for x in range(i())))"),
        # Names in nested comprehensions
        ("[y for y in range(x)]", {"x"}, "[y for y in range(x())]"),
        ("[y for y in range(x)]", {"y"}, "[y() for y in range(x)]"),
        (
            "[[y + z for y in range(x)] for z in range(x)]",
            {"x"},
            "[[y + z for y in range(x())] for z in range(x())]",
        ),
        (
            "[[y + z for y in range(x)] for z in range(x)]",
            {"x", "y", "z"},
            "[[y() + z() for y in range(x())] for z in range(x())]",
        ),
        # Mixed comprehension types with names
        (
            "{x: [y for y in range(x)] for x in range(n)}",
            {"n"},
            "{x: [y for y in range(x)] for x in range(n())}",
        ),
        (
            "{i: [j for j in range(i)] for i in range(n)}",
            {"i", "j", "n"},
            "{i(): [j() for j in range(i())] for i in range(n())}",
        ),
        # Names in different parts of comprehensions
        ("[f(x) for x in data]", {"f"}, "[f()(x) for x in data]"),
        ("[f(x) for x in data]", {"data"}, "[f(x) for x in data()]"),
        ("[f(x) for x in data]", {"f", "data"}, "[f()(x) for x in data()]"),
        (
            "[f(x) for x in data if pred(x)]",
            {"f", "pred", "data"},
            "[f()(x) for x in data() if pred()(x)]",
        ),
        # Nested function calls
        ("f(g(x))", {"f"}, "f()(g(x))"),
        ("f(g(x))", {"g"}, "f(g()(x))"),
        ("f(g(x))", {"x"}, "f(g(x()))"),
        ("f(g(h(x)))", {"f", "g", "h", "x"}, "f()(g()(h()(x())))"),
        # Complex expressions with comprehensions
        (
            "sum([x * y for x in range(a) for y in range(b)])",
            {"a", "b"},
            "sum([x * y for x in range(a()) for y in range(b())])",
        ),
        (
            "max(x + y for x in items1 for y in items2)",
            {"items1", "items2"},
            "max((x + y for x in items1() for y in items2()))",
        ),
        # Boolean operations with names
        ("x and y", {"x"}, "x() and y"),
        ("x and y", {"x", "y"}, "x() and y()"),
        ("x or y or z", {"x", "z"}, "x() or y or z()"),
        ("not x", {"x"}, "not x()"),
        # Ternary expressions
        ("x if cond else y", {"cond"}, "x if cond() else y"),
        ("x if cond else y", {"x", "y", "cond"}, "x() if cond() else y()"),
        # Names in slice operations
        ("arr[start:end]", {"start", "end"}, "arr[start():end()]"),
        ("arr[i:j:k]", {"i", "j", "k"}, "arr[i():j():k()]"),
        ("matrix[i][j]", {"i", "j"}, "matrix[i()][j()]"),
        # Dict/set comprehensions
        ("{x: y for x, y in pairs}", {"pairs"}, "{x: y for (x, y) in pairs()}"),
        ("{f(x) for x in items}", {"f", "items"}, "{f()(x) for x in items()}"),
        # Nested comprehensions with filters
        (
            "[x for x in [y for y in range(n) if y > m] if x < k]",
            {"n", "m", "k"},
            "[x for x in [y for y in range(n()) if y > m()] if x < k()]",
        ),
    ],
)
def test_name_to_call_nested_comprehensions(source_src, varnames, expected_src):
    """Test NameToCall transformer with nested comprehensions and complex expressions."""
    # Parse the source code
    source_ast = ast.parse(source_src, mode="eval")
    expected_ast = ast.parse(expected_src, mode="eval")

    # Apply the transformer
    transformer = NameToCall(varnames)
    transformed_ast = transformer.visit(source_ast)

    # Compare the unparsed strings
    transformed_src = ast.unparse(transformed_ast)
    expected_unparsed = ast.unparse(expected_ast)

    assert transformed_src == expected_unparsed, (
        f"Transformation failed:\n"
        f"Input: {source_src}\n"
        f"Varnames: {varnames}\n"
        f"Expected: {expected_unparsed}\n"
        f"Got: {transformed_src}"
    )


@pytest.mark.parametrize(
    "genexpr_src,expected_src",
    [
        # Simple generator expressions
        (
            "(x for x in range(10))",
            "forexpr(x(), {x: lambda: range(10)})",
        ),
        (
            "(x * 2 for x in range(10))",
            "forexpr(x() * 2, {x: lambda: range(10)})",
        ),
        (
            "(x + 1 for x in items)",
            "forexpr(x() + 1, {x: lambda: items})",
        ),
        # Complex expressions
        (
            "(x ** 2 + 2 * x + 1 for x in range(5))",
            "forexpr(x() ** 2 + 2 * x() + 1, {x: lambda: range(5)})",
        ),
        (
            "(f(x) for x in data)",
            "forexpr(f(x()), {x: lambda: data})",
        ),
        # Multiple nested loops
        (
            "(x + y for x in range(3) for y in range(4))",
            "forexpr(x() + y(), {x: lambda: range(3), y: lambda: range(4)})",
        ),
        (
            "(x * y for x in items1 for y in items2)",
            "forexpr(x() * y(), {x: lambda: items1, y: lambda: items2})",
        ),
        # Nested loops with dependencies
        (
            "(x + y for x in range(3) for y in range(x))",
            "forexpr(x() + y(), {x: lambda: range(3), y: lambda: range(x())})",
        ),
        (
            "((x, y) for x in range(3) for y in range(x, 5))",
            "forexpr((x(), y()), {x: lambda: range(3), y: lambda: range(x(), 5)})",
        ),
        # Triple nested loops
        (
            "(x + y + z for x in range(2) for y in range(2) for z in range(2))",
            "forexpr(x() + y() + z(), {x: lambda: range(2), y: lambda: range(2), z: lambda: range(2)})",
        ),
        # Complex iterators
        (
            "(x for x in [1, 2, 3])",
            "forexpr(x(), {x: lambda: [1, 2, 3]})",
        ),
        (
            "(x for x in list(range(5)))",
            "forexpr(x(), {x: lambda: list(range(5))})",
        ),
        # Expressions with function calls on iterators
        (
            "(x for x in sorted(items))",
            "forexpr(x(), {x: lambda: sorted(items)})",
        ),
        # Generator expressions with filters
        (
            "(x for x in range(10) if x % 2 == 0)",
            "forexpr(x(), {x: (x for x in range(10) if x % 2 == 0)})",
        ),
        (
            "(x + y for x in range(3) if x > 0 for y in range(3))",
            "forexpr(x() + y(), {x: (x for x in range(3) if x > 0), y: lambda: range(3)})",
        ),
        (
            "(x * 2 for x in items if x > 5)",
            "forexpr(x() * 2, {x: (x for x in items if x > 5)})",
        ),
        (
            "(x + y for x in range(5) for y in range(5) if x < y)",
            "forexpr(x() + y(), {x: lambda: range(5), y: (y for y in range(5) if x() < y)})",
        ),
        (
            "(x for x in range(20) if x % 2 == 0 if x % 3 == 0)",
            "forexpr(x(), {x: (x for x in range(20) if x % 2 == 0 if x % 3 == 0)})",
        ),
        # Generator expressions with unpacking
        (
            "((x, y) for x, y in pairs)",
            "forexpr((x(), y()), {(x, y): lambda: pairs})",
        ),
        (
            "(a + b for a, b in zip(list1, list2))",
            "forexpr(a() + b(), {(a, b): lambda: zip(list1, list2)})",
        ),
        (
            "(x + y + z for x, (y, z) in nested_pairs)",
            "forexpr(x() + y() + z(), {(x, (y, z)): lambda: nested_pairs})",
        ),
    ],
)
def test_generator_exp_to_forexpr_transformer(genexpr_src, expected_src):
    """Test GeneratorExpToForexpr transformer converts generator expressions to forexpr calls."""
    # Parse the source code
    source_ast = ast.parse(genexpr_src, mode="eval")

    # Apply the transformer
    transformer = GeneratorExpToForexpr()

    # For test cases that should raise NotImplementedError
    if expected_src == "NOT_APPLICABLE":
        # The xfail marker will handle the exception
        transformed_ast = transformer.visit(source_ast)
        return

    transformed_ast = transformer.visit(source_ast)
    expected_ast = ast.parse(expected_src, mode="eval")

    # Compare the unparsed strings
    transformed_src = ast.unparse(transformed_ast)
    expected_unparsed = ast.unparse(expected_ast)

    assert transformed_src == expected_unparsed, (
        f"Transformation failed:\n"
        f"Input: {genexpr_src}\n"
        f"Expected: {expected_unparsed}\n"
        f"Got: {transformed_src}"
    )


@pytest.mark.parametrize(
    "genexpr_src,expected_src",
    [
        # Generator expressions yielding comprehensions
        (
            "([x for x in range(i)] for i in range(5))",
            "forexpr([x for x in range(i())], {i: lambda: range(5)})",
        ),
        (
            "({x: x**2 for x in range(i)} for i in range(5))",
            "forexpr({x: x**2 for x in range(i())}, {i: lambda: range(5)})",
        ),
        (
            "({x for x in range(i) if x > 2} for i in range(10))",
            "forexpr({x for x in range(i()) if x > 2}, {i: lambda: range(10)})",
        ),
        # Double nested comprehensions
        (
            "([[x for x in range(j)] for j in range(i)] for i in range(3))",
            "forexpr([[x for x in range(j)] for j in range(i())], {i: lambda: range(3)})",
        ),
        (
            "([[x + y for y in range(j)] for j in range(i)] for i in range(3))",
            "forexpr([[x + y for y in range(j)] for j in range(i())], {i: lambda: range(3)})",
        ),
        # Generator with comprehension in iterator
        (
            "(x * 2 for x in [y**2 for y in range(5)])",
            "forexpr(x() * 2, {x: lambda: [y**2 for y in range(5)]})",
        ),
        (
            "(x for x in {y: y**2 for y in range(3)}.values())",
            "forexpr(x(), {x: lambda: {y: y**2 for y in range(3)}.values()})",
        ),
        # Complex expressions with nested calls
        (
            "(sum([x for x in range(i)]) for i in range(5))",
            "forexpr(sum([x for x in range(i())]), {i: lambda: range(5)})",
        ),
        (
            "(max(x for x in range(i + 1)) for i in range(3))",
            "forexpr(max(forexpr(x(), {x: lambda: range(i() + 1)})), {i: lambda: range(3)})",
        ),
        (
            "(list(enumerate(x + 1 for x in range(i + 1))) for i in range(3))",
            "forexpr(list(enumerate(forexpr(x() + 1, {x: lambda: range(i() + 1)}))), {i: lambda: range(3)})",
        ),
        # Nested generators in iterator
        (
            "([y for y in range(x)] for x in (z + 1 for z in range(5)))",
            "forexpr([y for y in range(x())], {x: lambda: forexpr(z() + 1, {z: lambda: range(5)})})",
        ),
        # Complex filters with nested comprehensions
        (
            "(x for x in range(10) if x in [y**2 for y in range(5)])",
            "forexpr(x(), {x: (x for x in range(10) if x in [y**2 for y in range(5)])})",
        ),
        (
            "([x, y] for x in range(3) if x > 0 for y in range(3) if y in [z for z in range(x)])",
            "forexpr([x(), y()], {x: (x for x in range(3) if x > 0), y: (y for y in range(3) if y in [z for z in range(x())])})",
        ),
        # Multiple filters with dependencies
        (
            "(x + y for x in range(5) if x > 1 for y in range(x) if y < x - 1)",
            "forexpr(x() + y(), {x: (x for x in range(5) if x > 1), y: (y for y in range(x()) if y < x() - 1)})",
        ),
        (
            "((x, y, z) for x in range(3) for y in range(x, 5) if y > x for z in range(y) if z < y)",
            "forexpr((x(), y(), z()), {x: lambda: range(3), y: (y for y in range(x(), 5) if y > x()), z: (z for z in range(y()) if z < y())})",
        ),
        # Unpacking with nested structures
        (
            "(a + b + c for (a, b), c in [((1, 2), 3), ((4, 5), 6)])",
            "forexpr(a() + b() + c(), {((a, b), c): lambda: [((1, 2), 3), ((4, 5), 6)]})",
        ),
        (
            "(x + sum(lst) for x, lst in [(1, [2, 3]), (4, [5, 6])])",
            "forexpr(x() + sum(lst()), {(x, lst): lambda: [(1, [2, 3]), (4, [5, 6])]})",
        ),
        # Complex iterators
        (
            "(x for x in sorted([y**2 for y in range(5)]))",
            "forexpr(x(), {x: lambda: sorted([y**2 for y in range(5)])})",
        ),
        (
            "(item for sublist in [[1, 2], [3, 4], [5, 6]] for item in sublist)",
            "forexpr(item(), {sublist: lambda: [[1, 2], [3, 4], [5, 6]], item: lambda: sublist()})",
        ),
        # Expressions with method calls
        (
            "(s.upper() for s in ['hello', 'world'] if s.startswith('h'))",
            "forexpr(s().upper(), {s: (s for s in ['hello', 'world'] if s.startswith('h'))})",
        ),
        (
            "(obj.value for obj in objects if hasattr(obj, 'value'))",
            "forexpr(obj().value, {obj: (obj for obj in objects if hasattr(obj, 'value'))})",
        ),
        # CRITICAL: Generator expressions yielding generator expressions
        (
            "((x for x in range(i)) for i in range(5))",
            "forexpr(forexpr(x(), {x: lambda: range(i())}), {i: lambda: range(5)})",
        ),
        (
            "((x * 2 for x in range(i)) for i in range(3))",
            "forexpr(forexpr(x() * 2, {x: lambda: range(i())}), {i: lambda: range(3)})",
        ),
        (
            "((x + y for x in range(3) for y in range(x)) for i in range(2))",
            "forexpr(forexpr(x() + y(), {x: lambda: range(3), y: lambda: range(x())}), {i: lambda: range(2)})",
        ),
        # Generator yielding filtered generator
        (
            "((x for x in range(10) if x % 2 == 0) for i in range(3))",
            "forexpr(forexpr(x(), {x: (x for x in range(10) if x % 2 == 0)}), {i: lambda: range(3)})",
        ),
        (
            "((x for x in range(i) if x > 0) for i in range(5))",
            "forexpr(forexpr(x(), {x: (x for x in range(i()) if x > 0)}), {i: lambda: range(5)})",
        ),
        # Nested generators with multiple levels
        (
            "((y for y in (x for x in range(i))) for i in range(3))",
            "forexpr(forexpr(y(), {y: lambda: forexpr(x(), {x: lambda: range(i())})}), {i: lambda: range(3)})",
        ),
        (
            "(((x + y for x in range(2)) for y in range(3)) for z in range(4))",
            "forexpr(forexpr(forexpr(x() + y(), {x: lambda: range(2)}), {y: lambda: range(3)}), {z: lambda: range(4)})",
        ),
        # Generator with unpacking yielding generator
        (
            "((x + b for x in range(a)) for a, b in [(2, 3), (4, 5)])",
            "forexpr(forexpr(x() + b(), {x: lambda: range(a())}), {(a, b): lambda: [(2, 3), (4, 5)]})",
        ),
        # Complex case: generator yielding generator with filters and dependencies
        (
            "((x + y for x in range(i) if x > 0 for y in range(x)) for i in range(5) if i > 2)",
            "forexpr(forexpr(x() + y(), {x: (x for x in range(i()) if x > 0), y: lambda: range(x())}), {i: (i for i in range(5) if i > 2)})",
        ),
        # Generator expression yielding sum of generator expression
        (
            "(sum(x for x in range(i)) for i in range(5))",
            "forexpr(sum(forexpr(x(), {x: lambda: range(i())})), {i: lambda: range(5)})",
        ),
        (
            "(max(x * 2 for x in range(i) if x > 0) for i in range(10))",
            "forexpr(max(forexpr(x() * 2, {x: (x for x in range(i()) if x > 0)})), {i: lambda: range(10)})",
        ),
    ],
)
def test_generator_exp_to_forexpr_nested_comprehensions(genexpr_src, expected_src):
    """Test GeneratorExpToForexpr transformer with nested comprehensions and complex expressions."""
    # Parse the source code
    source_ast = ast.parse(genexpr_src, mode="eval")

    # Apply the transformer
    transformer = GeneratorExpToForexpr()
    transformed_ast = transformer.visit(source_ast)
    expected_ast = ast.parse(expected_src, mode="eval")

    # Compare the unparsed strings
    transformed_src = ast.unparse(transformed_ast)
    expected_unparsed = ast.unparse(expected_ast)

    assert transformed_src == expected_unparsed, (
        f"Transformation failed:\n"
        f"Input: {genexpr_src}\n"
        f"Expected: {expected_unparsed}\n"
        f"Got: {transformed_src}"
    )
