import ast
import pytest
import dis
import inspect
from types import GeneratorType
from typing import Any, Union

from effectful.internals.genexpr import reconstruct


def compile_and_eval(node: ast.AST, globals_dict: dict = None) -> Any:
    """Compile an AST node and evaluate it."""
    if globals_dict is None:
        globals_dict = {}
    
    # Wrap in an Expression node if needed
    if not isinstance(node, ast.Expression):
        node = ast.Expression(body=node)
    
    # Fix location info
    ast.fix_missing_locations(node)
    
    # Compile and evaluate
    code = compile(node, '<ast>', 'eval')
    return eval(code, globals_dict)


def assert_ast_equivalent(genexpr: GeneratorType, reconstructed_ast: ast.AST, globals_dict: dict = None):
    """Assert that a reconstructed AST produces the same results as the original generator."""
    assert inspect.isgenerator(genexpr), "Input must be a generator"
    assert inspect.getgeneratorstate(genexpr) == 'GEN_CREATED', "Generator must not be consumed"

    # Check AST structure
    assert isinstance(reconstructed_ast, ast.GeneratorExp)
    assert hasattr(reconstructed_ast, 'elt')  # The expression part
    assert hasattr(reconstructed_ast, 'generators')  # The comprehension part
    assert len(reconstructed_ast.generators) > 0
    for comp in reconstructed_ast.generators:
        assert hasattr(comp, 'target')  # Loop variable
        assert hasattr(comp, 'iter')  # Iterator
        assert hasattr(comp, 'ifs')  # Conditions

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
    
    assert reconstructed_list == original_list, \
        f"AST produced {reconstructed_list}, expected {original_list}"


# ============================================================================
# BASIC GENERATOR EXPRESSION TESTS
# ============================================================================

@pytest.mark.parametrize("genexpr", [
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
])
def test_simple_generators(genexpr):
    """Test reconstruction of simple generator expressions."""
    ast_node = reconstruct(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# ARITHMETIC AND EXPRESSION TESTS
# ============================================================================

@pytest.mark.parametrize("genexpr", [
    # Basic arithmetic operations
    (x * 2 for x in range(5)),
    (x + 1 for x in range(5)),
    (x - 1 for x in range(5)),
    (x ** 2 for x in range(5)),
    (x % 2 for x in range(10)),
    (x / 2 for x in range(1, 6)),
    (x // 2 for x in range(10)),
    
    # Complex expressions
    (x * 2 + 1 for x in range(5)),
    ((x + 1) * (x - 1) for x in range(5)),
    (x ** 2 + 2 * x + 1 for x in range(5)),
    
    # Unary operations
    (-x for x in range(5)),
    (+x for x in range(-5, 5)),
    (~x for x in range(5)),
    
    # More complex arithmetic edge cases
    (x ** 3 for x in range(1, 5)),  # Higher powers
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
    (x ** 1 for x in range(5)),
    (0 + x for x in range(5)),
    (1 * x for x in range(5)),
])
def test_arithmetic_expressions(genexpr):
    """Test reconstruction of generators with arithmetic expressions."""
    ast_node = reconstruct(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# COMPARISON OPERATORS
# ============================================================================

@pytest.mark.parametrize("genexpr", [
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
    (x for x in range(10) if x > 2 and x < 8),
    (x for x in range(10) if x < 3 or x > 7),
    (x for x in range(10) if not x % 2),
    (x for x in range(10) if not (x > 5)),
    
    # More complex comparison edge cases
    # Comparisons with expressions
    (x for x in range(10) if x * 2 > 10),
    (x for x in range(10) if x + 1 <= 5),
    (x for x in range(10) if x ** 2 < 25),
    (x for x in range(10) if (x + 1) * 2 != 6),
    
    # Complex membership tests
    (x for x in range(20) if x in range(5, 15)),
    (x for x in range(10) if x not in range(3, 7)),
    (x for x in range(10) if x % 2 in [0]),
    (x for x in range(10) if x not in []),  # Empty container
    
    # Complex boolean combinations
    (x for x in range(20) if x > 5 and x < 15 and x % 2 == 0),
    (x for x in range(20) if x < 5 or x > 15 or x == 10),
    (x for x in range(20) if not (x > 5 and x < 15)),  # FIXME
    (x for x in range(20) if not (x < 5 or x > 15)),
    
    # Mixed comparison and boolean operations
    (x for x in range(20) if (x > 10 and x % 2 == 0) or (x < 5 and x % 3 == 0)),  # FIXME
    (x for x in range(20) if not (x % 2 == 0 and x % 3 == 0)),  # FIXME
    
    # Edge cases with identity comparisons
    (x for x in [0, 1, 2, None, 4] if x is not None and x > 1),
    (x for x in [True, False, 1, 0] if x is True),
    (x for x in [True, False, 1, 0] if x is not False),
])
def test_comparison_operators(genexpr):
    """Test reconstruction of all comparison operators."""
    ast_node = reconstruct(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# CHAINED COMPARISON TESTS
# ============================================================================

@pytest.mark.xfail(reason="Chained comparisons not yet fully supported")
@pytest.mark.parametrize("genexpr", [
    # Chained comparisons
    (x for x in range(20) if 5 < x < 15),
    (x for x in range(20) if 0 <= x <= 10),
    (x for x in range(20) if x >= 5 and x <= 15),
])
def test_chained_comparison_operators(genexpr):
    """Test reconstruction of chained (ternary) comparison operators."""
    ast_node = reconstruct(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# FILTERED GENERATOR TESTS
# ============================================================================

@pytest.mark.parametrize("genexpr", [
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
    (x ** 2 for x in range(10) if x > 3),
    
    # Boolean operations in filters
    (x for x in range(10) if x > 2 and x < 8),
    (x for x in range(10) if x < 3 or x > 7),
    (x for x in range(10) if not x % 2),
    
    # More complex filter edge cases
    (x for x in range(50) if x % 7 == 0),  # Different modulo
    (x for x in range(10) if x >= 0),  # Always true condition
    (x for x in range(10) if x < 0),  # Always false condition
    (x for x in range(20) if x % 2 == 0 and x % 3 == 0),  # Multiple conditions with and
    (x for x in range(20) if x % 2 == 0 or x % 3 == 0),  # Multiple conditions with or
    
    # Nested boolean operations
    (x for x in range(20) if (x > 5 and x < 15) or x == 0),  # FIXME
    (x for x in range(20) if not (x > 10 and x < 15)),  # FIXME
    (x for x in range(50) if x > 10 and (x % 2 == 0 or x % 3 == 0)),
    
    # Multiple consecutive filters
    (x for x in range(100) if x > 20 if x < 80 if x % 10 == 0),
    (x for x in range(50) if x % 2 == 0 if x % 3 != 0 if x > 10),
    
    # Filters with complex expressions
    (x + 1 for x in range(20) if (x * 2) % 3 == 0),
    (x ** 2 for x in range(10) if x * (x + 1) > 10),
    (x / 2 for x in range(1, 20) if x % (x // 2 + 1) == 0),
    
    # Edge cases with truthiness
    (x for x in range(10) if x),  # Truthy filter
    (x for x in range(-5, 5) if not x),  # Falsy filter
    (x for x in range(10) if bool(x % 2)),  # Explicit bool conversion
])
def test_filtered_generators(genexpr):
    """Test reconstruction of generators with if conditions."""
    ast_node = reconstruct(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# NESTED LOOP TESTS
# ============================================================================

@pytest.mark.parametrize("genexpr", [
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
    (x ** y for x in range(1, 4) for y in range(3)),
    
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
    ((x, y, z, w) for x in range(2) for y in range(2) for z in range(2) for w in range(2)),
    
    # Nested loops with complex filters
    ((x, y, z) for x in range(5) for y in range(5) for z in range(5) if x < y and y < z),
    (x + y + z for x in range(3) if x > 0 for y in range(3) if y != x for z in range(3) if z != x and z != y),
    
    # Mixed range types
    ((x, y) for x in range(-2, 2) for y in range(0, 4, 2)),
    (x * y for x in range(5, 0, -1) for y in range(1, 6)),

    # Dependent nested loops
    ((x, y) for x in range(3) for y in range(x, 3)),
    (x + y for x in range(3) for y in range(x + 1, 3)),
    (x * y * z for x in range(3) for y in range(x + 1, x + 3) for z in range(y, y + 3)),
])
def test_nested_loops(genexpr):
    """Test reconstruction of generators with nested loops."""
    ast_node = reconstruct(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ===========================================================================
# NESTED COMPREHENSIONS
# ===========================================================================

@pytest.mark.parametrize("genexpr", [
    ([x for x in range(i)] for i in range(5)),
    ({x: x**2 for x in range(i)} for i in range(5)),
    ([[x for x in range(i + j)] for j in range(i)] for i in range(5)),

    # function call
    (sum(x for x in range(i + 1)) for i in range(3)),

    # Nested comprehensions with filters inside
    ([x for x in range(i)] for i in range(5) if i > 0),
    ([x for x in range(i) if x < i] for i in range(5) if i > 0),
    ([[x for x in range(i + j) if x < i + j] for j in range(i)] for i in range(5)),
    ([[x for x in range(i + j) if x < i + j] for j in range(i)] for i in range(5) if i > 0),

    # nesting on both sides
    ([y for y in range(x)] for x in (x_ + 1 for x_ in range(5))),
    ([y for y in range(x)] for x in (x_ + 1 for x_ in range(5))),
])
def test_nested_comprehensions(genexpr):
    """Test reconstruction of nested comprehensions."""
    ast_node = reconstruct(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# DIFFERENT COMPREHENSION TYPES
# ============================================================================

@pytest.mark.parametrize("genexpr", [
    # Comprehensions as iterator constants
    (x_ for x_ in [x for x in range(5)]),
    (x_ for x_ in {x for x in range(5)}),
    (x_ for x_ in {x: x**2 for x in range(5)}),

    # Comprehensions as yield expressions
    ([y for y in range(x + 1)] for x in range(3)),
    ({y for y in range(x + 1)} for x in range(3)),
    ({y: y**2 for y in range(x + 1)} for x in range(3)),
])
def test_different_comprehension_types(genexpr):
    """Test reconstruction of different comprehension types."""
    ast_node = reconstruct(genexpr)
    assert_ast_equivalent(genexpr, ast_node)


# ============================================================================
# GENERATOR EXPRESSION WITH GLOBALS
# ============================================================================

@pytest.mark.parametrize("genexpr,globals_dict", [
    # Using constants
    ((x + a for x in range(5)), {'a': 10}),
    ((data[i] for i in range(2)), {'data': [3, 4]}),

    # Using global functions
    ((abs(x) for x in range(-5, 5)), {'abs': abs}),
    ((len(s) for s in ["a", "ab", "abc"]), {'len': len}),
    ((max(x, 5) for x in range(10)), {'max': max}),
    ((min(x, 5) for x in range(10)), {'min': min}),
    ((round(x / 3, 2) for x in range(10)), {'round': round}),
])
def test_variable_lookup(genexpr, globals_dict):
    """Test reconstruction of expressions with globals."""
    ast_node = reconstruct(genexpr)
    
    # Need to provide the same globals for evaluation
    assert_ast_equivalent(genexpr, ast_node, globals_dict)


# ============================================================================
# EDGE CASES AND COMPLEX SCENARIOS
# ============================================================================

@pytest.mark.parametrize("genexpr,globals_dict", [
    # Using lambdas and functions
    (((lambda y: y * 2)(x) for x in range(5)), {}),
    (((lambda y: y + 1)(x) for x in range(5)), {}),
    (((lambda y: y ** 2)(x) for x in range(5)), {}),
    
    # More complex lambdas
    # (((lambda a, b: a + b)(x, x) for x in range(5)), {}),
    ((f(x) for x in range(5)), {'f': lambda y: y * 3}),
    
    # Attribute access
    ((x.real for x in [1+2j, 3+4j, 5+6j]), {}),
    ((x.imag for x in [1+2j, 3+4j, 5+6j]), {}),
    ((x.conjugate() for x in [1+2j, 3+4j, 5+6j]), {}),
    
    # Method calls
    ((s.upper() for s in ["hello", "world"]), {}),
    ((s.lower() for s in ["HELLO", "WORLD"]), {}),
    ((s.strip() for s in [" hello ", "  world  "]), {}),
    ((x.bit_length() for x in range(1, 10)), {}),
    ((str(x).zfill(3) for x in range(10)), {'str': str}),
    
    # Subscript operations
    (([10, 20, 30][i] for i in range(3)), {}),
    (({'a': 1, 'b': 2, 'c': 3}[k] for k in ['a', 'b', 'c']), {}),
    (("hello"[i] for i in range(5)), {}),
    ((data[i][j] for i in range(2) for j in range(2)), {'data': [[1, 2], [3, 4]]}),
    
    # # More complex attribute chains
    # ((obj.value.bit_length() for obj in [type('', (), {'value': x})() for x in range(1, 5)]), {}),
    
    # Multiple function calls
    ((abs(max(x, -x)) for x in range(-3, 4)), {'abs': abs, 'max': max}),
    ((len(str(x)) for x in range(100, 110)), {'len': len, 'str': str}),
    
    # Mixed operations
    ((abs(x) + len(str(x)) for x in range(-10, 10)), {'abs': abs, 'len': len, 'str': str}),
    ((s.upper().lower() for s in ["Hello", "World"]), {}),
    
    # Edge cases with complex data structures
    (([1, 2, 3][x % 3] for x in range(10)), {}),
    # (({"even": x, "odd": x + 1}["even" if x % 2 == 0 else "odd"] for x in range(5)), {}),
    
    # Function calls with multiple arguments
    ((pow(x, 2, 10) for x in range(5)), {'pow': pow}),
    ((divmod(x, 3) for x in range(10)), {'divmod': divmod}),
])
def test_complex_scenarios(genexpr, globals_dict):
    """Test reconstruction of complex generator expressions."""
    ast_node = reconstruct(genexpr)

    # Need to provide the same globals for evaluation
    assert_ast_equivalent(genexpr, ast_node, globals_dict)


# ============================================================================
# HELPER FUNCTION TESTS
# ============================================================================

@pytest.mark.parametrize("value,expected_str", [
    # AST nodes should be returned as-is
    (ast.Name(id='x', ctx=ast.Load()), 'x'),
    (ast.Constant(value=42), '42'),
    (ast.List(elts=[], ctx=ast.Load()), '[]'),
    (ast.BinOp(left=ast.Constant(value=1), op=ast.Add(), right=ast.Constant(value=2)), '1 + 2'),
    
    # Constants should become ast.Constant nodes
    (42, '42'),
    (3.14, '3.14'),
    (-42, '-42'),
    (-3.14, '-3.14'),
    ('hello', "'hello'"),
    ("", "''"),
    (b'bytes', "b'bytes'"),
    (b'', "b''"),
    (True, 'True'),
    (False, 'False'),
    (None, 'None'),
    
    # Complex numbers
    (1+2j, '(1+2j)'),
    (0+1j, '1j'),
    (3+0j, '(3+0j)'),
    (-1-2j, '(-1-2j)'),
    
    # Tuples should become ast.Tuple nodes
    ((), '()'),
    ((1,), '(1,)'),
    ((1, 2), '(1, 2)'),
    (('a', 'b', 'c'), "('a', 'b', 'c')"),
    
    # Special dict_item tuples
    (('dict_item', 'key', 'value'), "('key', 'value')"),
    (('dict_item', 42, 'answer'), "(42, 'answer')"),
    
    # Nested tuples
    ((1, (2, 3)), '(1, (2, 3))'),
    (((1, 2), (3, 4)), '((1, 2), (3, 4))'),
    ((1, 2, (3, (4, 5))), '(1, 2, (3, (4, 5)))'),
    
    # Lists should become ast.List nodes
    ([1, 2, 3], '[1, 2, 3]'),
    (['hello', 'world'], "['hello', 'world']"),
    ([True, False, None], '[True, False, None]'),
    
    # Nested lists
    ([[1, 2], [3, 4]], '[[1, 2], [3, 4]]'),
    ([1, [2, [3, 4]], 5], '[1, [2, [3, 4]], 5]'),
    
    # Mixed nested structures
    ([(1, 2), (3, 4)], '[(1, 2), (3, 4)]'),
    (([1, 2], [3, 4]), '([1, 2], [3, 4])'),
    
    # Dicts should become ast.Dict nodes
    ({'a': 1}, "{'a': 1}"),
    ({'x': 10, 'y': 20}, "{'x': 10, 'y': 20}"),
    ({1: 'one', 2: 'two'}, "{1: 'one', 2: 'two'}"),
    
    # Nested dicts
    ({'a': {'b': 1}}, "{'a': {'b': 1}}"),
    ({'nums': [1, 2, 3], 'strs': ['a', 'b']}, "{'nums': [1, 2, 3], 'strs': ['a', 'b']}"),
    
    # Range objects
    (range(5), 'range(0, 5, 1)'),
    (range(1, 10), 'range(1, 10, 1)'),
    (range(0, 10, 2), 'range(0, 10, 2)'),
    (range(10, 0, -1), 'range(10, 0, -1)'),
    (range(-5, 5), 'range(-5, 5, 1)'),
    
    # Empty collections
    ([], '[]'),
    ((), '()'),
    ({}, '{}'),
    
    # Complex nested structures
    ([1, [2, 3], 4], '[1, [2, 3], 4]'),
    ({'a': [1, 2], 'b': {'c': 3}}, "{'a': [1, 2], 'b': {'c': 3}}"),
    ([(1, {'a': [2, 3]}), ({'b': 4}, 5)], "[(1, {'a': [2, 3]}), ({'b': 4}, 5)]"),
    
    # Edge cases with special values
    ([None, True, False, 0, ''], "[None, True, False, 0, '']"),
    ({'': 'empty', None: 'none', 0: 'zero'}, "{'': 'empty', None: 'none', 0: 'zero'}"),
    
    # Large numbers
    (999999999999999999999, '999999999999999999999'),
    (1.7976931348623157e+308, '1.7976931348623157e+308'),  # Close to float max
    
    # Sets - note unparse equivalence may fail for unordered collections
    ({1, 2, 3}, '{1, 2, 3}'),
])
def test_ensure_ast(value, expected_str):
    """Test that ensure_ast correctly converts various values to AST nodes."""
    from effectful.internals.genexpr import ensure_ast
    
    result = ensure_ast(value)

    # Compare the unparsed strings
    result_str = ast.unparse(result)
    assert result_str == expected_str, \
        f"ensure_ast({repr(value)}) produced '{result_str}', expected '{expected_str}'"


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
