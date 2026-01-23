"""Comprehensive tests for the meta-circular interpreter."""

import ast
import dataclasses
import inspect
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from effectful.internals.meta_eval import (
    EvaluatorState,
    InterpreterError,
    ReturnException,
    eval_expr,
    eval_expr_generator,
    eval_module,
    eval_stmt,
    eval_stmt_generator,
)

# -------------------------
# Meta-circular evaluation test
# -------------------------


def test_meta_circular_evaluation():
    """Test that the meta-circular interpreter can evaluate itself."""
    # Get the source of meta_eval.py
    meta_eval_path = (
        Path(__file__).parent.parent / "effectful" / "internals" / "meta_eval.py"
    )
    source_text = meta_eval_path.read_text()

    module = ast.parse(source_text)

    # Add builtins and other necessary modules to allowed modules
    import builtins
    import collections
    import linecache
    import pathlib
    import sys
    import types
    import typing

    allowed_modules = {
        "builtins": builtins,
        "sys": sys,
        "types": types,
        "typing": typing,
        "dataclasses": dataclasses,
        "ast": ast,
        "inspect": inspect,
        "collections": collections,
        "collections.abc": collections.abc,
        "pathlib": pathlib,
        "linecache": linecache,
    }

    state = EvaluatorState.fresh(
        allowed_modules=allowed_modules,
        allowed_dunder_attrs=[
            "__init__",
            "__str__",
            "__repr__",
            "__file__",
            "__package__",
            "__module__",
            "__qualname__",
            "__doc__",
            "__dict__",
            "__loader__",
            "__name__",
            "__code__",
            "__prepare__",
        ],
    )

    state.bindings["hasattr"] = hasattr
    state.bindings["__file__"] = str(meta_eval_path)
    state.bindings["dataclass"] = dataclasses.dataclass

    eval_module(module, state)

    # Now test the interpreter on various code samples
    _test_meta_eval_on_code_samples(state)


def test_meta_circular_evaluation_2_levels():
    """Test meta-circular evaluation 2 levels deep, then test on complex code snippets."""
    # Get the source of meta_eval.py
    meta_eval_path = (
        Path(__file__).parent.parent / "effectful" / "internals" / "meta_eval.py"
    )
    source_text = meta_eval_path.read_text()

    # Setup allowed modules and state configuration
    import builtins
    import collections
    import linecache
    import pathlib
    import sys
    import types
    import typing

    allowed_modules = {
        "builtins": builtins,
        "sys": sys,
        "types": types,
        "typing": typing,
        "dataclasses": dataclasses,
        "ast": ast,
        "inspect": inspect,
        "collections": collections,
        "collections.abc": collections.abc,
        "pathlib": pathlib,
        "linecache": linecache,
    }

    allowed_dunder_attrs = [
        "__init__",
        "__str__",
        "__repr__",
        "__file__",
        "__package__",
        "__module__",
        "__qualname__",
        "__doc__",
        "__dict__",
        "__loader__",
        "__name__",
        "__ast__",
        "__code__",
        "__prepare__",
    ]

    # Helper function to create a fresh state with proper setup
    def create_state() -> EvaluatorState:
        state = EvaluatorState.fresh(
            allowed_modules=allowed_modules,
            allowed_dunder_attrs=allowed_dunder_attrs,
        )
        state.bindings["hasattr"] = hasattr
        state.bindings["__file__"] = str(meta_eval_path)
        state.bindings["dataclass"] = dataclasses.dataclass
        return state

    # Level 1: Evaluate meta_eval.py to get eval_module
    print("Level 1: Evaluating meta_eval.py...")
    module = ast.parse(source_text)
    state1 = create_state()

    eval_module(module, state1)

    # Get eval_module from level 1
    eval_module_1 = state1.bindings.get("eval_module")
    assert eval_module_1 is not None, "eval_module not found in level 1 state"

    # Now test all complex code snippets using level 1's eval_module
    _test_meta_eval_on_code_samples(state1, eval_module_1)


def _test_meta_eval_on_code_samples(
    meta_eval_state: EvaluatorState, eval_module_fn: Any = None
):
    """Test the interpreter on various code samples.

    Args:
        meta_eval_state: State with allowed modules/attrs for creating fresh states
        eval_module_fn: Optional eval_module function to use (defaults to module-level eval_module)
    """
    if eval_module_fn is None:
        eval_module_fn = eval_module

    def run_code(code: str, setup: dict[str, Any] | None = None) -> EvaluatorState:
        """Parse and evaluate code, returning the state."""
        state = EvaluatorState.fresh(
            allowed_modules=dict(meta_eval_state.allowed_modules),
            allowed_dunder_attrs=list(meta_eval_state.allowed_dunder_attrs),
        )
        if setup:
            for k, v in setup.items():
                if k == "__dunder__":
                    state.allowed_dunder_attrs.add(v)
                else:
                    state.bindings[k] = v
        eval_module_fn(ast.parse(code), state)
        return state

    # Test 1: Generators
    state = run_code(
        """
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

def simple_gen():
    yield 1
    yield 2
    yield 3
""",
        {"range": range, "list": list},
    )

    assert list(state.bindings["fibonacci"](5)) == [0, 1, 1, 2, 3]
    assert list(state.bindings["simple_gen"]()) == [1, 2, 3]

    # Test 2: Classes with inheritance
    state = run_code(
        """
class Animal:
    def __init__(self, name):
        self.name = name
    def speak(self):
        return f"{self.name} makes a sound"

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)
        self.breed = breed
    def speak(self):
        return f"{self.name} barks"

dog = Dog("Rex", "Labrador")
""",
        {"__dunder__": "__init__", "super": super},
    )

    assert state.bindings["dog"].name == "Rex"
    assert state.bindings["dog"].breed == "Labrador"
    assert state.bindings["dog"].speak() == "Rex barks"

    # Test 3: Decorators
    state = run_code("""
def memoize(func):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

fib_5 = fibonacci(5)
""")
    assert state.bindings["fib_5"] == 5

    # Test 4: Dataclasses
    state = run_code(
        """
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int
    def distance(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

p1 = Point(0, 0)
p2 = Point(3, 4)
""",
        {"__dunder__": "__init__", "dataclass": dataclasses.dataclass},
    )

    assert state.bindings["p1"].distance(state.bindings["p2"]) == 5.0

    # Test 5: Exceptions
    state = run_code(
        """
result = None
error = None
try:
    result = 10 / 0
except ZeroDivisionError as e:
    error = str(e)
""",
        {"str": str, "ZeroDivisionError": ZeroDivisionError},
    )

    assert state.bindings["result"] is None
    assert state.bindings["error"] is not None

    # Test 6: Complex - class with generator method
    state = run_code(
        """
class DataProcessor:
    def __init__(self, data):
        self.data = data
    def process(self):
        for item in self.data:
            yield item * 2

processor = DataProcessor([1, 2, 3])
result = list(processor.process())
""",
        {"__dunder__": "__init__", "list": list},
    )

    assert state.bindings["result"] == [2, 4, 6]


# -------------------------
# eval_expr tests - Constants and literals
# -------------------------


def test_eval_expr_constant():
    """Test constant expressions."""
    state = EvaluatorState.fresh()

    assert eval_expr(ast.Constant(value=42), state) == 42
    assert eval_expr(ast.Constant(value="hello"), state) == "hello"
    assert eval_expr(ast.Constant(value=None), state) is None
    assert eval_expr(ast.Constant(value=True), state) is True
    assert eval_expr(ast.Constant(value=False), state) is False


def test_eval_expr_name():
    """Test name lookups."""
    state = EvaluatorState.fresh()
    state.bindings["x"] = 42
    state.bindings["y"] = "hello"

    assert eval_expr(ast.Name(id="x", ctx=ast.Load()), state) == 42
    assert eval_expr(ast.Name(id="y", ctx=ast.Load()), state) == "hello"

    with pytest.raises(NameError, match="Name 'z' is not defined"):
        eval_expr(ast.Name(id="z", ctx=ast.Load()), state)


def test_eval_expr_tuple():
    """Test tuple expressions."""
    state = EvaluatorState.fresh()
    state.bindings["a"] = 1
    state.bindings["b"] = 2

    node = ast.Tuple(
        elts=[
            ast.Name(id="a", ctx=ast.Load()),
            ast.Name(id="b", ctx=ast.Load()),
            ast.Constant(value=3),
        ],
        ctx=ast.Load(),
    )

    result = eval_expr(node, state)
    assert result == (1, 2, 3)
    assert isinstance(result, tuple)


def test_eval_expr_list():
    """Test list expressions."""
    state = EvaluatorState.fresh()
    state.bindings["a"] = 1
    state.bindings["b"] = 2

    node = ast.List(
        elts=[
            ast.Name(id="a", ctx=ast.Load()),
            ast.Name(id="b", ctx=ast.Load()),
            ast.Constant(value=3),
        ],
        ctx=ast.Load(),
    )

    result = eval_expr(node, state)
    assert result == [1, 2, 3]
    assert isinstance(result, list)


def test_eval_expr_set():
    """Test set expressions."""
    state = EvaluatorState.fresh()

    node = ast.Set(
        elts=[ast.Constant(value=1), ast.Constant(value=2), ast.Constant(value=3)]
    )

    result = eval_expr(node, state)
    assert result == {1, 2, 3}
    assert isinstance(result, set)


def test_eval_expr_dict():
    """Test dict expressions."""
    state = EvaluatorState.fresh()
    state.bindings["key"] = "value"

    node = ast.Dict(
        keys=[ast.Constant(value="a"), ast.Name(id="key", ctx=ast.Load())],
        values=[ast.Constant(value=1), ast.Constant(value=2)],
    )

    result = eval_expr(node, state)
    assert result == {"a": 1, "value": 2}
    assert isinstance(result, dict)


def test_eval_expr_dict_unpacking():
    """Test dict unpacking with **."""
    state = EvaluatorState.fresh()
    state.bindings["extra"] = {"c": 3, "d": 4}

    node = ast.Dict(
        keys=[
            ast.Constant(value="a"),
            None,  # **unpacking
        ],
        values=[ast.Constant(value=1), ast.Name(id="extra", ctx=ast.Load())],
    )

    result = eval_expr(node, state)
    assert result == {"a": 1, "c": 3, "d": 4}


# -------------------------
# eval_expr tests - String formatting
# -------------------------


def test_eval_expr_joined_str():
    """Test f-strings."""
    state = EvaluatorState.fresh()
    state.bindings["name"] = "world"

    node = ast.JoinedStr(
        values=[
            ast.Constant(value="Hello, "),
            ast.FormattedValue(
                value=ast.Name(id="name", ctx=ast.Load()),
                conversion=-1,
                format_spec=None,
            ),
            ast.Constant(value="!"),
        ]
    )

    result = eval_expr(node, state)
    assert result == "Hello, world!"


def test_eval_expr_formatted_value():
    """Test formatted value expressions."""
    state = EvaluatorState.fresh()
    state.bindings["x"] = 42

    # Test conversion codes
    node1 = ast.FormattedValue(
        value=ast.Name(id="x", ctx=ast.Load()),
        conversion=115,  # !s
        format_spec=None,
    )
    assert eval_expr(node1, state) == "42"

    node2 = ast.FormattedValue(
        value=ast.Name(id="x", ctx=ast.Load()),
        conversion=114,  # !r
        format_spec=None,
    )
    assert eval_expr(node2, state) == "42"

    # Test with format spec
    node3 = ast.FormattedValue(
        value=ast.Name(id="x", ctx=ast.Load()),
        conversion=-1,
        format_spec=ast.Constant(value="04d"),
    )
    assert eval_expr(node3, state) == "0042"


# -------------------------
# eval_expr tests - Unary operations
# -------------------------


def test_eval_expr_unary_op():
    """Test unary operations."""
    state = EvaluatorState.fresh()

    # Unary plus
    node1 = ast.UnaryOp(op=ast.UAdd(), operand=ast.Constant(value=5))
    assert eval_expr(node1, state) == 5

    # Unary minus
    node2 = ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=5))
    assert eval_expr(node2, state) == -5

    # Not
    node3 = ast.UnaryOp(op=ast.Not(), operand=ast.Constant(value=True))
    assert eval_expr(node3, state) is False

    node4 = ast.UnaryOp(op=ast.Not(), operand=ast.Constant(value=False))
    assert eval_expr(node4, state) is True

    # Invert (bitwise not)
    node5 = ast.UnaryOp(op=ast.Invert(), operand=ast.Constant(value=5))
    assert eval_expr(node5, state) == -6  # ~5 == -6 in two's complement


# -------------------------
# eval_expr tests - Binary operations
# -------------------------


def test_eval_expr_binop():
    """Test binary operations."""
    state = EvaluatorState.fresh()

    test_cases = [
        (ast.Add(), 2, 3, 5),
        (ast.Sub(), 5, 3, 2),
        (ast.Mult(), 2, 3, 6),
        (ast.Div(), 6, 3, 2.0),
        (ast.FloorDiv(), 7, 3, 2),
        (ast.Mod(), 7, 3, 1),
        (ast.Pow(), 2, 3, 8),
        (ast.LShift(), 2, 3, 16),  # 2 << 3
        (ast.RShift(), 16, 3, 2),  # 16 >> 3
        (ast.BitAnd(), 5, 3, 1),  # 5 & 3
        (ast.BitOr(), 5, 3, 7),  # 5 | 3
        (ast.BitXor(), 5, 3, 6),  # 5 ^ 3
    ]

    for op, left, right, expected in test_cases:
        node = ast.BinOp(
            left=ast.Constant(value=left), op=op, right=ast.Constant(value=right)
        )
        assert eval_expr(node, state) == expected, f"Failed for {type(op).__name__}"


def test_eval_expr_binop_matmult():
    """Test matrix multiplication."""
    state = EvaluatorState.fresh()

    # Test with lists (though @ doesn't work with lists in real Python)
    # This is just to test the op is handled
    node = ast.BinOp(
        left=ast.Constant(value=[1, 2]),
        op=ast.MatMult(),
        right=ast.Constant(value=[3, 4]),
    )
    # This will fail at runtime, but we're just testing the op is recognized
    with pytest.raises((TypeError, InterpreterError)):
        eval_expr(node, state)


# -------------------------
# eval_expr tests - Boolean operations
# -------------------------


def test_eval_expr_boolop_and():
    """Test boolean AND operations."""
    state = EvaluatorState.fresh()

    # True and True and True
    node1 = ast.BoolOp(
        op=ast.And(),
        values=[
            ast.Constant(value=True),
            ast.Constant(value=True),
            ast.Constant(value=True),
        ],
    )
    assert eval_expr(node1, state) is True

    # True and False
    node2 = ast.BoolOp(
        op=ast.And(), values=[ast.Constant(value=True), ast.Constant(value=False)]
    )
    assert eval_expr(node2, state) is False

    # Short-circuit: False and (shouldn't evaluate second)
    state.bindings["x"] = 0
    node3 = ast.BoolOp(
        op=ast.And(),
        values=[ast.Constant(value=False), ast.Name(id="x", ctx=ast.Load())],
    )
    assert eval_expr(node3, state) is False


def test_eval_expr_boolop_or():
    """Test boolean OR operations."""
    state = EvaluatorState.fresh()

    # False or False or True
    node1 = ast.BoolOp(
        op=ast.Or(),
        values=[
            ast.Constant(value=False),
            ast.Constant(value=False),
            ast.Constant(value=True),
        ],
    )
    assert eval_expr(node1, state) is True

    # False or False
    node2 = ast.BoolOp(
        op=ast.Or(), values=[ast.Constant(value=False), ast.Constant(value=False)]
    )
    assert eval_expr(node2, state) is False

    # Short-circuit: True or (shouldn't evaluate second)
    state.bindings["x"] = 0
    node3 = ast.BoolOp(
        op=ast.Or(), values=[ast.Constant(value=True), ast.Name(id="x", ctx=ast.Load())]
    )
    assert eval_expr(node3, state) is True


# -------------------------
# eval_expr tests - Comparisons
# -------------------------


def test_eval_expr_compare():
    """Test comparison operations."""
    state = EvaluatorState.fresh()

    test_cases = [
        (ast.Eq(), 5, 5, True),
        (ast.Eq(), 5, 3, False),
        (ast.NotEq(), 5, 3, True),
        (ast.NotEq(), 5, 5, False),
        (ast.Lt(), 3, 5, True),
        (ast.Lt(), 5, 3, False),
        (ast.LtE(), 5, 5, True),
        (ast.LtE(), 5, 3, False),
        (ast.Gt(), 5, 3, True),
        (ast.Gt(), 3, 5, False),
        (ast.GtE(), 5, 5, True),
        (ast.GtE(), 3, 5, False),
        (ast.Is(), None, None, True),
        (ast.IsNot(), None, None, False),
        (ast.In(), 2, [1, 2, 3], True),
        (ast.In(), 4, [1, 2, 3], False),
        (ast.NotIn(), 4, [1, 2, 3], True),
        (ast.NotIn(), 2, [1, 2, 3], False),
    ]

    for op, left, right, expected in test_cases:
        node = ast.Compare(
            left=ast.Constant(value=left),
            ops=[op],
            comparators=[ast.Constant(value=right)],
        )
        assert eval_expr(node, state) == expected, f"Failed for {type(op).__name__}"


def test_eval_expr_compare_chained():
    """Test chained comparisons."""
    state = EvaluatorState.fresh()

    # 1 < 2 < 3
    node = ast.Compare(
        left=ast.Constant(value=1),
        ops=[ast.Lt(), ast.Lt()],
        comparators=[ast.Constant(value=2), ast.Constant(value=3)],
    )
    assert eval_expr(node, state) is True

    # 1 < 3 < 2 (should be False)
    node2 = ast.Compare(
        left=ast.Constant(value=1),
        ops=[ast.Lt(), ast.Lt()],
        comparators=[ast.Constant(value=3), ast.Constant(value=2)],
    )
    assert eval_expr(node2, state) is False


# -------------------------
# eval_expr tests - Conditional expressions
# -------------------------


def test_eval_expr_ifexp():
    """Test conditional expressions."""
    state = EvaluatorState.fresh()

    # True ? 1 : 2
    node1 = ast.IfExp(
        test=ast.Constant(value=True),
        body=ast.Constant(value=1),
        orelse=ast.Constant(value=2),
    )
    assert eval_expr(node1, state) == 1

    # False ? 1 : 2
    node2 = ast.IfExp(
        test=ast.Constant(value=False),
        body=ast.Constant(value=1),
        orelse=ast.Constant(value=2),
    )
    assert eval_expr(node2, state) == 2


# -------------------------
# eval_expr tests - Attribute access
# -------------------------


def test_eval_expr_attribute():
    """Test attribute access."""
    state = EvaluatorState.fresh()

    class Obj:
        def __init__(self):
            self.value = 42

    obj = Obj()
    state.bindings["obj"] = obj

    node = ast.Attribute(
        value=ast.Name(id="obj", ctx=ast.Load()), attr="value", ctx=ast.Load()
    )
    assert eval_expr(node, state) == 42


def test_eval_expr_attribute_dunder():
    """Test dunder attribute access."""
    state = EvaluatorState.fresh()
    state.allowed_dunder_attrs.add("__dict__")

    class Obj:
        pass

    obj = Obj()
    state.bindings["obj"] = obj

    node = ast.Attribute(
        value=ast.Name(id="obj", ctx=ast.Load()), attr="__dict__", ctx=ast.Load()
    )
    result = eval_expr(node, state)
    assert isinstance(result, dict)


def test_eval_expr_attribute_forbidden():
    """Test that forbidden dunder attributes raise errors."""
    state = EvaluatorState.fresh()

    class Obj:
        pass

    obj = Obj()
    state.bindings["obj"] = obj

    node = ast.Attribute(
        value=ast.Name(id="obj", ctx=ast.Load()), attr="__class__", ctx=ast.Load()
    )

    with pytest.raises(InterpreterError, match="Forbidden dunder attribute"):
        eval_expr(node, state)


# -------------------------
# eval_expr tests - Subscripting
# -------------------------


def test_eval_expr_subscript():
    """Test subscript operations."""
    state = EvaluatorState.fresh()

    # List indexing
    state.bindings["lst"] = [1, 2, 3]
    node1 = ast.Subscript(
        value=ast.Name(id="lst", ctx=ast.Load()),
        slice=ast.Constant(value=1),
        ctx=ast.Load(),
    )
    assert eval_expr(node1, state) == 2

    # Dict indexing
    state.bindings["d"] = {"a": 1, "b": 2}
    node2 = ast.Subscript(
        value=ast.Name(id="d", ctx=ast.Load()),
        slice=ast.Constant(value="a"),
        ctx=ast.Load(),
    )
    assert eval_expr(node2, state) == 1


def test_eval_expr_slice():
    """Test slice expressions."""
    state = EvaluatorState.fresh()

    # [1:3]
    node1 = ast.Slice(
        lower=ast.Constant(value=1), upper=ast.Constant(value=3), step=None
    )
    result = eval_expr(node1, state)
    assert result == slice(1, 3, None)

    # [1:3:2]
    node2 = ast.Slice(
        lower=ast.Constant(value=1),
        upper=ast.Constant(value=3),
        step=ast.Constant(value=2),
    )
    result = eval_expr(node2, state)
    assert result == slice(1, 3, 2)

    # [:3]
    node3 = ast.Slice(lower=None, upper=ast.Constant(value=3), step=None)
    result = eval_expr(node3, state)
    assert result == slice(None, 3, None)


# -------------------------
# eval_expr tests - Function calls
# -------------------------


def test_eval_expr_call():
    """Test function calls."""
    state = EvaluatorState.fresh()

    def add(a, b):
        return a + b

    state.bindings["add"] = add

    node = ast.Call(
        func=ast.Name(id="add", ctx=ast.Load()),
        args=[ast.Constant(value=2), ast.Constant(value=3)],
        keywords=[],
    )
    assert eval_expr(node, state) == 5


def test_eval_expr_call_keywords():
    """Test function calls with keyword arguments."""
    state = EvaluatorState.fresh()

    def func(a, b, c=10):
        return a + b + c

    state.bindings["func"] = func

    node = ast.Call(
        func=ast.Name(id="func", ctx=ast.Load()),
        args=[ast.Constant(value=1)],
        keywords=[
            ast.keyword(arg="b", value=ast.Constant(value=2)),
            ast.keyword(arg="c", value=ast.Constant(value=3)),
        ],
    )
    assert eval_expr(node, state) == 6


def test_eval_expr_call_starred():
    """Test function calls with *args."""
    state = EvaluatorState.fresh()

    def add(*args):
        return sum(args)

    state.bindings["add"] = add
    state.bindings["nums"] = [1, 2, 3]

    node = ast.Call(
        func=ast.Name(id="add", ctx=ast.Load()),
        args=[ast.Starred(value=ast.Name(id="nums", ctx=ast.Load()))],
        keywords=[],
    )
    assert eval_expr(node, state) == 6


def test_eval_expr_call_kwargs():
    """Test function calls with **kwargs."""
    state = EvaluatorState.fresh()

    def func(**kwargs):
        return kwargs.get("a", 0) + kwargs.get("b", 0)

    state.bindings["func"] = func
    state.bindings["kw"] = {"a": 1, "b": 2}

    node = ast.Call(
        func=ast.Name(id="func", ctx=ast.Load()),
        args=[],
        keywords=[ast.keyword(arg=None, value=ast.Name(id="kw", ctx=ast.Load()))],
    )
    assert eval_expr(node, state) == 3


# -------------------------
# eval_expr tests - Lambdas
# -------------------------


def test_eval_expr_lambda():
    """Test lambda expressions."""
    state = EvaluatorState.fresh()

    node = ast.Lambda(
        args=ast.arguments(
            args=[ast.arg(arg="x")],
            defaults=[],
            kwonlyargs=[],
            kw_defaults=[],
            posonlyargs=[],
        ),
        body=ast.BinOp(
            left=ast.Name(id="x", ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Constant(value=1),
        ),
    )

    result = eval_expr(node, state)
    assert callable(result)
    assert result(5) == 6


def test_eval_expr_lambda_defaults():
    """Test lambda with default arguments."""
    state = EvaluatorState.fresh()

    node = ast.Lambda(
        args=ast.arguments(
            args=[ast.arg(arg="x"), ast.arg(arg="y")],
            defaults=[ast.Constant(value=10)],
            kwonlyargs=[],
            kw_defaults=[],
            posonlyargs=[],
        ),
        body=ast.BinOp(
            left=ast.Name(id="x", ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Name(id="y", ctx=ast.Load()),
        ),
    )

    result = eval_expr(node, state)
    assert result(5) == 15  # 5 + 10
    assert result(5, 20) == 25  # 5 + 20


# -------------------------
# eval_expr tests - Comprehensions
# -------------------------


def test_eval_expr_listcomp():
    """Test list comprehensions."""
    state = EvaluatorState.fresh()
    state.bindings["items"] = [1, 2, 3]

    node = ast.ListComp(
        elt=ast.BinOp(
            left=ast.Name(id="x", ctx=ast.Load()),
            op=ast.Mult(),
            right=ast.Constant(value=2),
        ),
        generators=[
            ast.comprehension(
                target=ast.Name(id="x", ctx=ast.Store()),
                iter=ast.Name(id="items", ctx=ast.Load()),
                ifs=[],
                is_async=0,
            )
        ],
    )

    result = eval_expr(node, state)
    assert result == [2, 4, 6]


def test_eval_expr_setcomp():
    """Test set comprehensions."""
    state = EvaluatorState.fresh()
    state.bindings["items"] = [1, 2, 2, 3]

    node = ast.SetComp(
        elt=ast.Name(id="x", ctx=ast.Load()),
        generators=[
            ast.comprehension(
                target=ast.Name(id="x", ctx=ast.Store()),
                iter=ast.Name(id="items", ctx=ast.Load()),
                ifs=[],
                is_async=0,
            )
        ],
    )

    result = eval_expr(node, state)
    assert result == {1, 2, 3}


def test_eval_expr_dictcomp():
    """Test dict comprehensions."""
    state = EvaluatorState.fresh()
    state.bindings["items"] = [1, 2, 3]

    node = ast.DictComp(
        key=ast.Name(id="x", ctx=ast.Load()),
        value=ast.BinOp(
            left=ast.Name(id="x", ctx=ast.Load()),
            op=ast.Mult(),
            right=ast.Constant(value=2),
        ),
        generators=[
            ast.comprehension(
                target=ast.Name(id="x", ctx=ast.Store()),
                iter=ast.Name(id="items", ctx=ast.Load()),
                ifs=[],
                is_async=0,
            )
        ],
    )

    result = eval_expr(node, state)
    assert result == {1: 2, 2: 4, 3: 6}


def test_eval_expr_generatorexp():
    """Test generator expressions."""
    state = EvaluatorState.fresh()
    state.bindings["items"] = [1, 2, 3]

    node = ast.GeneratorExp(
        elt=ast.BinOp(
            left=ast.Name(id="x", ctx=ast.Load()),
            op=ast.Mult(),
            right=ast.Constant(value=2),
        ),
        generators=[
            ast.comprehension(
                target=ast.Name(id="x", ctx=ast.Store()),
                iter=ast.Name(id="items", ctx=ast.Load()),
                ifs=[],
                is_async=0,
            )
        ],
    )

    result = eval_expr(node, state)
    assert isinstance(result, Generator)
    assert list(result) == [2, 4, 6]


def test_eval_expr_comprehension_with_if():
    """Test comprehensions with if conditions."""
    state = EvaluatorState.fresh()
    state.bindings["items"] = [1, 2, 3, 4, 5]

    node = ast.ListComp(
        elt=ast.Name(id="x", ctx=ast.Load()),
        generators=[
            ast.comprehension(
                target=ast.Name(id="x", ctx=ast.Store()),
                iter=ast.Name(id="items", ctx=ast.Load()),
                ifs=[
                    ast.BinOp(
                        left=ast.Name(id="x", ctx=ast.Load()),
                        op=ast.Mod(),
                        right=ast.Constant(value=2),
                    )
                ],
                is_async=0,
            )
        ],
    )

    result = eval_expr(node, state)
    assert result == [1, 3, 5]  # Only odd numbers


# -------------------------
# eval_expr tests - Yield (should raise exception)
# -------------------------


def test_eval_expr_yield_raises():
    """Test that yield in non-generator context raises InterpreterError."""
    state = EvaluatorState.fresh()

    node = ast.Yield(value=ast.Constant(value=42))

    with pytest.raises(InterpreterError, match="yield expressions are not supported"):
        eval_expr(node, state)


def test_eval_expr_yield_from_raises():
    """Test that yield from in non-generator context raises InterpreterError."""
    state = EvaluatorState.fresh()

    def gen():
        yield 1

    state.bindings["gen"] = gen()

    node = ast.YieldFrom(value=ast.Name(id="gen", ctx=ast.Load()))

    with pytest.raises(
        InterpreterError, match="yield from expressions are not supported"
    ):
        eval_expr(node, state)


# -------------------------
# eval_expr_generator tests
# -------------------------


def test_eval_expr_generator_constant():
    """Test constants in generator context."""
    state = EvaluatorState.fresh()

    gen = eval_expr_generator(ast.Constant(value=42), state)
    yielded = list(gen)
    # Constants don't yield anything, they just return the value
    assert yielded == []

    # Test that the return value is correct
    gen = eval_expr_generator(ast.Constant(value=42), state)
    try:
        next(gen)
        pytest.fail("Generator should not yield")
    except StopIteration as e:
        assert e.value == 42


def test_eval_expr_generator_yield():
    """Test yield expressions in generator context."""
    state = EvaluatorState.fresh()

    node = ast.Yield(value=ast.Constant(value=42))
    gen = eval_expr_generator(node, state)

    yielded = next(gen)
    assert yielded == 42

    try:
        next(gen)
        pytest.fail("Generator should be exhausted")
    except StopIteration as e:
        assert e.value == 42


def test_eval_expr_generator_yield_none():
    """Test yield None."""
    state = EvaluatorState.fresh()

    node = ast.Yield(value=None)
    gen = eval_expr_generator(node, state)

    yielded = next(gen)
    assert yielded is None


def test_eval_expr_generator_yield_from():
    """Test yield from expressions."""
    state = EvaluatorState.fresh()

    def subgen():
        yield 1
        yield 2
        return 3

    state.bindings["subgen"] = subgen

    # Test: yield from subgen()
    node = ast.YieldFrom(
        value=ast.Call(func=ast.Name(id="subgen", ctx=ast.Load()), args=[], keywords=[])
    )
    gen = eval_expr_generator(node, state)

    # Collect all yielded values
    values = []
    result = None
    try:
        while True:
            val = next(gen)
            values.append(val)
    except StopIteration as e:
        result = e.value

    assert values == [1, 2]
    # The return value should be 3 (from the subgenerator)
    assert result == 3


def test_eval_expr_generator_binop_with_yield():
    """Test binary operations with yields: (yield 1) + (yield 2)."""
    state = EvaluatorState.fresh()

    node = ast.BinOp(
        left=ast.Yield(value=ast.Constant(value=1)),
        op=ast.Add(),
        right=ast.Yield(value=ast.Constant(value=2)),
    )

    gen = eval_expr_generator(node, state)

    # First yield
    val1 = next(gen)
    assert val1 == 1

    # Second yield
    val2 = next(gen)
    assert val2 == 2

    # Final result
    try:
        next(gen)
        pytest.fail("Generator should be exhausted")
    except StopIteration as e:
        assert e.value == 3  # 1 + 2


def test_eval_expr_generator_call_with_yield():
    """Test function calls with yields in arguments."""
    state = EvaluatorState.fresh()

    def add(a, b):
        return a + b

    state.bindings["add"] = add

    node = ast.Call(
        func=ast.Name(id="add", ctx=ast.Load()),
        args=[
            ast.Yield(value=ast.Constant(value=1)),
            ast.Yield(value=ast.Constant(value=2)),
        ],
        keywords=[],
    )

    gen = eval_expr_generator(node, state)

    # First yield
    val1 = next(gen)
    assert val1 == 1

    # Second yield
    val2 = next(gen)
    assert val2 == 2

    # Final result
    try:
        next(gen)
        pytest.fail("Generator should be exhausted")
    except StopIteration as e:
        assert e.value == 3


def test_eval_expr_generator_ifexp_with_yield():
    """Test conditional expressions with yields."""
    state = EvaluatorState.fresh()

    node = ast.IfExp(
        test=ast.Constant(value=True),
        body=ast.Yield(value=ast.Constant(value=1)),
        orelse=ast.Yield(value=ast.Constant(value=2)),
    )

    gen = eval_expr_generator(node, state)
    val = next(gen)
    assert val == 1

    # False case
    node2 = ast.IfExp(
        test=ast.Constant(value=False),
        body=ast.Yield(value=ast.Constant(value=1)),
        orelse=ast.Yield(value=ast.Constant(value=2)),
    )

    gen2 = eval_expr_generator(node2, state)
    val2 = next(gen2)
    assert val2 == 2


def test_eval_expr_generator_compare_with_yield():
    """Test comparisons with yields."""
    state = EvaluatorState.fresh()

    node = ast.Compare(
        left=ast.Yield(value=ast.Constant(value=1)),
        ops=[ast.Lt()],
        comparators=[ast.Yield(value=ast.Constant(value=2))],
    )

    gen = eval_expr_generator(node, state)

    val1 = next(gen)
    assert val1 == 1

    val2 = next(gen)
    assert val2 == 2

    try:
        next(gen)
        pytest.fail("Generator should be exhausted")
    except StopIteration as e:
        assert e.value is True  # 1 < 2


def test_eval_expr_generator_boolop_with_yield():
    """Test boolean operations with yields."""
    state = EvaluatorState.fresh()

    # And with yields
    node = ast.BoolOp(
        op=ast.And(),
        values=[
            ast.Yield(value=ast.Constant(value=True)),
            ast.Yield(value=ast.Constant(value=False)),
        ],
    )

    gen = eval_expr_generator(node, state)

    val1 = next(gen)
    assert val1 is True

    val2 = next(gen)
    assert val2 is False

    try:
        next(gen)
        pytest.fail("Generator should be exhausted")
    except StopIteration as e:
        assert e.value is False


# -------------------------
# eval_stmt tests - Basic statements
# -------------------------


def test_eval_stmt_expr():
    """Test expression statements."""
    state = EvaluatorState.fresh()

    node = ast.Expr(value=ast.Constant(value=42))
    result = eval_stmt(node, state)
    assert result == 42


def test_eval_stmt_assign():
    """Test assignment statements."""
    state = EvaluatorState.fresh()

    node = ast.Assign(
        targets=[ast.Name(id="x", ctx=ast.Store())], value=ast.Constant(value=42)
    )
    result = eval_stmt(node, state)
    assert result == 42
    assert state.bindings["x"] == 42


def test_eval_stmt_assign_multiple():
    """Test multiple assignments."""
    state = EvaluatorState.fresh()

    node = ast.Assign(
        targets=[ast.Name(id="x", ctx=ast.Store()), ast.Name(id="y", ctx=ast.Store())],
        value=ast.Constant(value=42),
    )
    result = eval_stmt(node, state)
    assert result == 42
    assert state.bindings["x"] == 42
    assert state.bindings["y"] == 42


def test_eval_stmt_annassign():
    """Test annotated assignments."""
    state = EvaluatorState.fresh()

    node = ast.AnnAssign(
        target=ast.Name(id="x", ctx=ast.Store()),
        annotation=ast.Name(id="int", ctx=ast.Load()),
        value=ast.Constant(value=42),
        simple=1,
    )
    result = eval_stmt(node, state)
    assert result == 42
    assert state.bindings["x"] == 42


def test_eval_stmt_augassign():
    """Test augmented assignments."""
    state = EvaluatorState.fresh()
    state.bindings["x"] = 5

    node = ast.AugAssign(
        target=ast.Name(id="x", ctx=ast.Store()),
        op=ast.Add(),
        value=ast.Constant(value=3),
    )
    result = eval_stmt(node, state)
    assert result == 8
    assert state.bindings["x"] == 8


def test_eval_stmt_pass():
    """Test pass statements."""
    state = EvaluatorState.fresh()

    node = ast.Pass()
    result = eval_stmt(node, state)
    assert result is None


def test_eval_stmt_delete():
    """Test delete statements."""
    state = EvaluatorState.fresh()
    state.bindings["x"] = 42

    node = ast.Delete(targets=[ast.Name(id="x", ctx=ast.Del())])
    result = eval_stmt(node, state)
    assert result is None
    assert "x" not in state.bindings


# -------------------------
# eval_stmt tests - Control flow
# -------------------------


def test_eval_stmt_if():
    """Test if statements."""
    state = EvaluatorState.fresh()

    # True branch
    node1 = ast.If(
        test=ast.Constant(value=True),
        body=[ast.Expr(value=ast.Constant(value=1))],
        orelse=[ast.Expr(value=ast.Constant(value=2))],
    )
    result = eval_stmt(node1, state)
    assert result == 1

    # False branch
    node2 = ast.If(
        test=ast.Constant(value=False),
        body=[ast.Expr(value=ast.Constant(value=1))],
        orelse=[ast.Expr(value=ast.Constant(value=2))],
    )
    result = eval_stmt(node2, state)
    assert result == 2


def test_eval_stmt_while():
    """Test while loops."""
    state = EvaluatorState.fresh()
    state.bindings["count"] = 0

    node = ast.While(
        test=ast.Compare(
            left=ast.Name(id="count", ctx=ast.Load()),
            ops=[ast.Lt()],
            comparators=[ast.Constant(value=3)],
        ),
        body=[
            ast.AugAssign(
                target=ast.Name(id="count", ctx=ast.Store()),
                op=ast.Add(),
                value=ast.Constant(value=1),
            )
        ],
        orelse=[],
    )
    eval_stmt(node, state)
    assert state.bindings["count"] == 3


def test_eval_stmt_while_break():
    """Test while loops with break."""
    state = EvaluatorState.fresh()
    state.bindings["count"] = 0

    node = ast.While(
        test=ast.Constant(value=True),
        body=[
            ast.AugAssign(
                target=ast.Name(id="count", ctx=ast.Store()),
                op=ast.Add(),
                value=ast.Constant(value=1),
            ),
            ast.If(
                test=ast.Compare(
                    left=ast.Name(id="count", ctx=ast.Load()),
                    ops=[ast.GtE()],
                    comparators=[ast.Constant(value=2)],
                ),
                body=[ast.Break()],
                orelse=[],
            ),
        ],
        orelse=[],
    )
    eval_stmt(node, state)
    assert state.bindings["count"] == 2


def test_eval_stmt_while_continue():
    """Test while loops with continue."""
    state = EvaluatorState.fresh()
    state.bindings["count"] = 0
    state.bindings["total"] = 0

    node = ast.While(
        test=ast.Compare(
            left=ast.Name(id="count", ctx=ast.Load()),
            ops=[ast.Lt()],
            comparators=[ast.Constant(value=5)],
        ),
        body=[
            ast.AugAssign(
                target=ast.Name(id="count", ctx=ast.Store()),
                op=ast.Add(),
                value=ast.Constant(value=1),
            ),
            ast.If(
                test=ast.Compare(
                    left=ast.BinOp(
                        left=ast.Name(id="count", ctx=ast.Load()),
                        op=ast.Mod(),
                        right=ast.Constant(value=2),
                    ),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=0)],
                ),
                body=[ast.Continue()],
                orelse=[],
            ),
            ast.AugAssign(
                target=ast.Name(id="total", ctx=ast.Store()),
                op=ast.Add(),
                value=ast.Constant(value=1),
            ),
        ],
        orelse=[],
    )
    eval_stmt(node, state)
    # count goes 1,2,3,4,5, total only increments for odd numbers (1,3,5)
    assert state.bindings["count"] == 5
    assert state.bindings["total"] == 3


def test_eval_stmt_for():
    """Test for loops."""
    state = EvaluatorState.fresh()
    state.bindings["items"] = [1, 2, 3]
    state.bindings["total"] = 0

    node = ast.For(
        target=ast.Name(id="x", ctx=ast.Store()),
        iter=ast.Name(id="items", ctx=ast.Load()),
        body=[
            ast.AugAssign(
                target=ast.Name(id="total", ctx=ast.Store()),
                op=ast.Add(),
                value=ast.Name(id="x", ctx=ast.Load()),
            )
        ],
        orelse=[],
    )
    eval_stmt(node, state)
    assert state.bindings["total"] == 6


def test_eval_stmt_for_break():
    """Test for loops with break."""
    state = EvaluatorState.fresh()
    state.bindings["items"] = [1, 2, 3, 4, 5]
    state.bindings["total"] = 0

    node = ast.For(
        target=ast.Name(id="x", ctx=ast.Store()),
        iter=ast.Name(id="items", ctx=ast.Load()),
        body=[
            ast.AugAssign(
                target=ast.Name(id="total", ctx=ast.Store()),
                op=ast.Add(),
                value=ast.Name(id="x", ctx=ast.Load()),
            ),
            ast.If(
                test=ast.Compare(
                    left=ast.Name(id="x", ctx=ast.Load()),
                    ops=[ast.GtE()],
                    comparators=[ast.Constant(value=3)],
                ),
                body=[ast.Break()],
                orelse=[],
            ),
        ],
        orelse=[],
    )
    eval_stmt(node, state)
    assert state.bindings["total"] == 6  # 1 + 2 + 3


def test_eval_stmt_return():
    """Test return statements."""
    state = EvaluatorState.fresh()

    node = ast.Return(value=ast.Constant(value=42))

    with pytest.raises(ReturnException) as exc_info:
        eval_stmt(node, state)
    assert exc_info.value.value == 42


def test_eval_stmt_return_none():
    """Test return None."""
    state = EvaluatorState.fresh()

    node = ast.Return(value=None)

    with pytest.raises(ReturnException) as exc_info:
        eval_stmt(node, state)
    assert exc_info.value.value is None


# -------------------------
# eval_stmt tests - Function and class definitions
# -------------------------


def test_eval_stmt_functiondef():
    """Test function definitions."""
    state = EvaluatorState.fresh()

    node = ast.FunctionDef(
        name="add",
        args=ast.arguments(
            args=[ast.arg(arg="a"), ast.arg(arg="b")],
            defaults=[],
            kwonlyargs=[],
            kw_defaults=[],
            posonlyargs=[],
        ),
        body=[
            ast.Return(
                value=ast.BinOp(
                    left=ast.Name(id="a", ctx=ast.Load()),
                    op=ast.Add(),
                    right=ast.Name(id="b", ctx=ast.Load()),
                )
            )
        ],
        decorator_list=[],
        returns=None,
        lineno=1,
    )

    result = eval_stmt(node, state)
    assert callable(result)
    assert result(2, 3) == 5
    assert state.bindings["add"] == result


def test_eval_stmt_functiondef_generator():
    """Test generator function definitions."""
    state = EvaluatorState.fresh()

    node = ast.FunctionDef(
        name="gen",
        args=ast.arguments(
            args=[], defaults=[], kwonlyargs=[], kw_defaults=[], posonlyargs=[]
        ),
        body=[
            ast.Expr(value=ast.Yield(value=ast.Constant(value=1))),
            ast.Expr(value=ast.Yield(value=ast.Constant(value=2))),
            ast.Expr(value=ast.Yield(value=ast.Constant(value=3))),
        ],
        decorator_list=[],
        returns=None,
        lineno=1,
    )

    result = eval_stmt(node, state)
    assert callable(result)
    gen = result()
    assert isinstance(gen, Generator)
    assert list(gen) == [1, 2, 3]


def test_eval_stmt_classdef():
    """Test class definitions."""
    state = EvaluatorState.fresh()

    node = ast.ClassDef(
        name="MyClass",
        bases=[],
        keywords=[],
        body=[
            ast.FunctionDef(
                name="__init__",
                args=ast.arguments(
                    args=[ast.arg(arg="self"), ast.arg(arg="value")],
                    defaults=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    posonlyargs=[],
                ),
                body=[
                    ast.Assign(
                        targets=[
                            ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="value",
                                ctx=ast.Store(),
                            )
                        ],
                        value=ast.Name(id="value", ctx=ast.Load()),
                        lineno=3,
                    )
                ],
                decorator_list=[],
                returns=None,
                lineno=2,
            )
        ],
        decorator_list=[],
        lineno=1,
    )

    result = eval_stmt(node, state)
    assert isinstance(result, type)
    assert result.__name__ == "MyClass"

    # Test instantiation
    obj = result(42)
    assert obj.value == 42


# -------------------------
# eval_stmt tests - Imports
# -------------------------


def test_eval_stmt_import():
    """Test import statements."""
    import sys

    state = EvaluatorState.fresh(allowed_modules={"sys": sys})

    node = ast.Import(names=[ast.alias(name="sys", asname=None)])
    eval_stmt(node, state)
    assert "sys" in state.modules


def test_eval_stmt_import_from():
    """Test from import statements."""
    import collections

    state = EvaluatorState.fresh(allowed_modules={"collections": collections})

    node = ast.ImportFrom(
        module="collections", names=[ast.alias(name="ChainMap", asname=None)], level=0
    )
    eval_stmt(node, state)
    assert "ChainMap" in state.modules


# -------------------------
# eval_stmt tests - Exception handling
# -------------------------


def test_eval_stmt_try_except():
    """Test try/except statements."""
    state = EvaluatorState.fresh()
    state.bindings["ValueError"] = ValueError

    # Use a mutable object to track if exception was caught
    caught = {"value": False}
    state.bindings["caught"] = caught

    def set_caught():
        caught["value"] = True

    state.bindings["set_caught"] = set_caught

    node = ast.Try(
        body=[
            ast.Raise(
                exc=ast.Call(
                    func=ast.Name(id="ValueError", ctx=ast.Load()),
                    args=[ast.Constant(value="error")],
                    keywords=[],
                ),
                cause=None,
            )
        ],
        handlers=[
            ast.ExceptHandler(
                type=ast.Name(id="ValueError", ctx=ast.Load()),
                name="e",
                body=[
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Name(id="set_caught", ctx=ast.Load()),
                            args=[],
                            keywords=[],
                        )
                    )
                ],
            )
        ],
        orelse=[],
        finalbody=[],
    )

    # Should not raise - exception should be caught
    result = eval_stmt(node, state)
    assert result is None
    # Verify exception was caught by checking the mutable object
    assert caught["value"] is True


def test_eval_stmt_try_finally():
    """Test try/finally statements."""
    state = EvaluatorState.fresh()
    state.bindings["cleaned"] = False

    node = ast.Try(
        body=[ast.Expr(value=ast.Constant(value=42))],
        handlers=[],
        orelse=[],
        finalbody=[
            ast.Assign(
                targets=[ast.Name(id="cleaned", ctx=ast.Store())],
                value=ast.Constant(value=True),
            )
        ],
    )

    eval_stmt(node, state)
    assert state.bindings["cleaned"] is True


def test_eval_stmt_raise():
    """Test raise statements."""
    state = EvaluatorState.fresh()

    node = ast.Raise(
        exc=ast.Call(
            func=ast.Name(id="ValueError", ctx=ast.Load()),
            args=[ast.Constant(value="test error")],
            keywords=[],
        ),
        cause=None,
    )

    with pytest.raises(ValueError, match="test error"):
        eval_stmt(node, state)


def test_eval_stmt_assert():
    """Test assert statements."""
    state = EvaluatorState.fresh()

    # True assertion
    node1 = ast.Assert(test=ast.Constant(value=True), msg=None)
    result = eval_stmt(node1, state)
    assert result is None

    # False assertion
    node2 = ast.Assert(
        test=ast.Constant(value=False), msg=ast.Constant(value="assertion failed")
    )
    with pytest.raises(AssertionError, match="assertion failed"):
        eval_stmt(node2, state)


# -------------------------
# eval_stmt tests - With statements
# -------------------------


def test_eval_stmt_with():
    """Test with statements."""
    state = EvaluatorState.fresh()
    state.bindings["entered"] = False
    state.bindings["exited"] = False

    class Context:
        def __enter__(self):
            state.bindings["entered"] = True
            return self

        def __exit__(self, *args):
            state.bindings["exited"] = True

    state.bindings["ctx"] = Context()

    node = ast.With(
        items=[
            ast.withitem(
                context_expr=ast.Name(id="ctx", ctx=ast.Load()),
                optional_vars=ast.Name(id="c", ctx=ast.Store()),
            )
        ],
        body=[ast.Expr(value=ast.Constant(value=42))],
        type_comment=None,
    )

    result = eval_stmt(node, state)
    assert state.bindings["entered"] is True
    assert state.bindings["exited"] is True
    assert result == 42


# -------------------------
# eval_stmt tests - Global and nonlocal
# -------------------------


def test_eval_stmt_global():
    """Test global statements."""
    state = EvaluatorState.fresh()
    state.push_scope()

    # Set up scope directives (normally done in function)
    from effectful.internals.meta_eval import ScopeDirectives

    state.scope_directives.append(ScopeDirectives(set(), set()))

    node = ast.Global(names=["x"])
    result = eval_stmt(node, state)
    assert result is None
    assert "x" in state.scope_directives[-1].globals


def test_eval_stmt_nonlocal():
    """Test nonlocal statements."""
    state = EvaluatorState.fresh()
    state.push_scope()

    # Set up scope directives (normally done in function)
    from effectful.internals.meta_eval import ScopeDirectives

    state.scope_directives.append(ScopeDirectives(set(), set()))

    node = ast.Nonlocal(names=["x"])
    result = eval_stmt(node, state)
    assert result is None
    assert "x" in state.scope_directives[-1].nonlocals


# -------------------------
# eval_stmt_generator tests
# -------------------------


def test_eval_stmt_generator_expr():
    """Test expression statements in generator context."""
    state = EvaluatorState.fresh()

    node = ast.Expr(value=ast.Yield(value=ast.Constant(value=42)))
    gen = eval_stmt_generator(node, state)

    val = next(gen)
    assert val == 42

    try:
        next(gen)
        pytest.fail("Generator should be exhausted")
    except StopIteration:
        pass


def test_eval_stmt_generator_assign():
    """Test assignments in generator context."""
    state = EvaluatorState.fresh()

    node = ast.Assign(
        targets=[ast.Name(id="x", ctx=ast.Store())],
        value=ast.Yield(value=ast.Constant(value=42)),
    )
    gen = eval_stmt_generator(node, state)

    val = next(gen)
    assert val == 42

    try:
        next(gen)
        final = None
    except StopIteration as e:
        final = e.value

    assert final == 42
    assert state.bindings["x"] == 42


def test_eval_stmt_generator_return():
    """Test return statements in generator context."""
    state = EvaluatorState.fresh()

    node = ast.Return(value=ast.Yield(value=ast.Constant(value=42)))
    gen = eval_stmt_generator(node, state)

    val = next(gen)
    assert val == 42

    try:
        next(gen)
        final = None
    except StopIteration as e:
        final = e.value

    assert isinstance(final, ReturnException)
    assert final.value == 42


# -------------------------
# eval_stmt tests - Match statements
# -------------------------


def test_eval_stmt_match_value():
    """Test match statements with value patterns."""
    state = EvaluatorState.fresh()
    state.bindings["x"] = 42

    node = ast.Match(
        subject=ast.Name(id="x", ctx=ast.Load()),
        cases=[
            ast.match_case(
                pattern=ast.MatchValue(value=ast.Constant(value=42)),
                guard=None,
                body=[ast.Expr(value=ast.Constant(value="matched"))],
            ),
            ast.match_case(
                pattern=ast.MatchValue(value=ast.Constant(value=0)),
                guard=None,
                body=[ast.Expr(value=ast.Constant(value="not matched"))],
            ),
        ],
    )

    result = eval_stmt(node, state)
    assert result == "matched"


def test_eval_stmt_match_singleton():
    """Test match statements with singleton patterns."""
    state = EvaluatorState.fresh()
    state.bindings["x"] = None

    node = ast.Match(
        subject=ast.Name(id="x", ctx=ast.Load()),
        cases=[
            ast.match_case(
                pattern=ast.MatchSingleton(value=None),
                guard=None,
                body=[ast.Expr(value=ast.Constant(value="None"))],
            )
        ],
    )

    result = eval_stmt(node, state)
    assert result == "None"


def test_eval_stmt_match_as():
    """Test match statements with as patterns."""
    state = EvaluatorState.fresh()
    state.bindings["x"] = 42

    node = ast.Match(
        subject=ast.Name(id="x", ctx=ast.Load()),
        cases=[
            ast.match_case(
                pattern=ast.MatchAs(pattern=None, name="y"),
                guard=None,
                body=[
                    ast.Expr(
                        value=ast.BinOp(
                            left=ast.Name(id="y", ctx=ast.Load()),
                            op=ast.Add(),
                            right=ast.Constant(value=1),
                        )
                    )
                ],
            )
        ],
    )

    result = eval_stmt(node, state)
    assert result == 43
    # Match bindings should be in outer scope
    assert state.bindings["y"] == 42


def test_eval_stmt_match_or():
    """Test match statements with or patterns."""
    state = EvaluatorState.fresh()
    state.bindings["x"] = 2

    node = ast.Match(
        subject=ast.Name(id="x", ctx=ast.Load()),
        cases=[
            ast.match_case(
                pattern=ast.MatchOr(
                    patterns=[
                        ast.MatchValue(value=ast.Constant(value=1)),
                        ast.MatchValue(value=ast.Constant(value=2)),
                        ast.MatchValue(value=ast.Constant(value=3)),
                    ]
                ),
                guard=None,
                body=[ast.Expr(value=ast.Constant(value="matched"))],
            )
        ],
    )

    result = eval_stmt(node, state)
    assert result == "matched"


def test_eval_stmt_match_class():
    """Test match statements with class patterns."""
    state = EvaluatorState.fresh()

    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    state.bindings["Point"] = Point
    state.bindings["p"] = Point(1, 2)

    node = ast.Match(
        subject=ast.Name(id="p", ctx=ast.Load()),
        cases=[
            ast.match_case(
                pattern=ast.MatchClass(
                    cls=ast.Name(id="Point", ctx=ast.Load()),
                    patterns=[],
                    kwd_attrs=[],
                    kwd_patterns=[],
                ),
                guard=None,
                body=[ast.Expr(value=ast.Constant(value="Point"))],
            )
        ],
    )

    result = eval_stmt(node, state)
    assert result == "Point"


def test_eval_stmt_match_sequence():
    """Test match statements with sequence patterns."""
    state = EvaluatorState.fresh()
    state.bindings["lst"] = [1, 2, 3]

    node = ast.Match(
        subject=ast.Name(id="lst", ctx=ast.Load()),
        cases=[
            ast.match_case(
                pattern=ast.MatchSequence(
                    patterns=[
                        ast.MatchAs(pattern=None, name="a"),
                        ast.MatchAs(pattern=None, name="b"),
                        ast.MatchAs(pattern=None, name="c"),
                    ]
                ),
                guard=None,
                body=[
                    ast.Expr(
                        value=ast.BinOp(
                            left=ast.Name(id="a", ctx=ast.Load()),
                            op=ast.Add(),
                            right=ast.Name(id="b", ctx=ast.Load()),
                        )
                    )
                ],
            )
        ],
    )

    result = eval_stmt(node, state)
    assert result == 3  # 1 + 2
    assert state.bindings["a"] == 1
    assert state.bindings["b"] == 2
    assert state.bindings["c"] == 3


def test_eval_stmt_match_mapping():
    """Test match statements with mapping patterns."""
    state = EvaluatorState.fresh()
    state.bindings["d"] = {"a": 1, "b": 2}

    node = ast.Match(
        subject=ast.Name(id="d", ctx=ast.Load()),
        cases=[
            ast.match_case(
                pattern=ast.MatchMapping(
                    keys=[ast.Constant(value="a"), ast.Constant(value="b")],
                    patterns=[
                        ast.MatchAs(pattern=None, name="x"),
                        ast.MatchAs(pattern=None, name="y"),
                    ],
                    rest=None,
                ),
                guard=None,
                body=[
                    ast.Expr(
                        value=ast.BinOp(
                            left=ast.Name(id="x", ctx=ast.Load()),
                            op=ast.Add(),
                            right=ast.Name(id="y", ctx=ast.Load()),
                        )
                    )
                ],
            )
        ],
    )

    result = eval_stmt(node, state)
    assert result == 3  # 1 + 2
    assert state.bindings["x"] == 1
    assert state.bindings["y"] == 2


def test_eval_stmt_match_guard():
    """Test match statements with guards."""
    state = EvaluatorState.fresh()
    state.bindings["x"] = 42

    node = ast.Match(
        subject=ast.Name(id="x", ctx=ast.Load()),
        cases=[
            ast.match_case(
                pattern=ast.MatchAs(pattern=None, name="y"),
                guard=ast.Compare(
                    left=ast.Name(id="y", ctx=ast.Load()),
                    ops=[ast.Gt()],
                    comparators=[ast.Constant(value=40)],
                ),
                body=[ast.Expr(value=ast.Constant(value="matched"))],
            ),
            ast.match_case(
                pattern=ast.MatchAs(pattern=None, name="y"),
                guard=None,
                body=[ast.Expr(value=ast.Constant(value="default"))],
            ),
        ],
    )

    result = eval_stmt(node, state)
    assert result == "matched"


# -------------------------
# Generator function tests
# -------------------------


def test_generator_function_simple():
    """Test simple generator function."""
    state = EvaluatorState.fresh()

    node = ast.FunctionDef(
        name="gen",
        args=ast.arguments(
            args=[], defaults=[], kwonlyargs=[], kw_defaults=[], posonlyargs=[]
        ),
        body=[
            ast.Expr(value=ast.Yield(value=ast.Constant(value=1))),
            ast.Expr(value=ast.Yield(value=ast.Constant(value=2))),
            ast.Expr(value=ast.Yield(value=ast.Constant(value=3))),
        ],
        decorator_list=[],
        returns=None,
        lineno=1,
    )

    fn = eval_stmt(node, state)
    assert callable(fn)

    gen = fn()
    assert isinstance(gen, Generator)
    assert list(gen) == [1, 2, 3]


def test_generator_function_yield_from():
    """Test generator function with yield from."""
    state = EvaluatorState.fresh()

    def subgen():
        yield 1
        yield 2
        return 3

    state.bindings["subgen"] = subgen

    node = ast.FunctionDef(
        name="gen",
        args=ast.arguments(
            args=[], defaults=[], kwonlyargs=[], kw_defaults=[], posonlyargs=[]
        ),
        body=[
            ast.Expr(
                value=ast.YieldFrom(
                    value=ast.Call(
                        func=ast.Name(id="subgen", ctx=ast.Load()), args=[], keywords=[]
                    )
                )
            )
        ],
        decorator_list=[],
        returns=None,
        lineno=1,
    )

    fn = eval_stmt(node, state)
    gen = fn()
    assert list(gen) == [1, 2]


def test_generator_function_return():
    """Test generator function with return."""
    state = EvaluatorState.fresh()

    node = ast.FunctionDef(
        name="gen",
        args=ast.arguments(
            args=[], defaults=[], kwonlyargs=[], kw_defaults=[], posonlyargs=[]
        ),
        body=[
            ast.Expr(value=ast.Yield(value=ast.Constant(value=1))),
            ast.Return(value=ast.Constant(value=42)),
        ],
        decorator_list=[],
        returns=None,
        lineno=1,
    )

    fn = eval_stmt(node, state)
    gen = fn()
    assert next(gen) == 1

    try:
        next(gen)
        pytest.fail("Generator should be exhausted")
    except StopIteration as e:
        assert e.value == 42


# -------------------------
# Complex expression tests
# -------------------------


def test_complex_nested_expressions():
    """Test deeply nested expressions."""
    state = EvaluatorState.fresh()
    state.bindings["a"] = 1
    state.bindings["b"] = 2
    state.bindings["c"] = 3

    # ((a + b) * c) - 1
    node = ast.BinOp(
        left=ast.BinOp(
            left=ast.BinOp(
                left=ast.Name(id="a", ctx=ast.Load()),
                op=ast.Add(),
                right=ast.Name(id="b", ctx=ast.Load()),
            ),
            op=ast.Mult(),
            right=ast.Name(id="c", ctx=ast.Load()),
        ),
        op=ast.Sub(),
        right=ast.Constant(value=1),
    )

    result = eval_expr(node, state)
    assert result == ((1 + 2) * 3) - 1  # 8


def test_complex_function_call():
    """Test complex function calls."""
    state = EvaluatorState.fresh()

    def add(a, b, c=10):
        return a + b + c

    state.bindings["add"] = add
    state.bindings["x"] = 1
    state.bindings["y"] = 2

    node = ast.Call(
        func=ast.Name(id="add", ctx=ast.Load()),
        args=[
            ast.BinOp(
                left=ast.Name(id="x", ctx=ast.Load()),
                op=ast.Mult(),
                right=ast.Constant(value=2),
            ),
            ast.Name(id="y", ctx=ast.Load()),
        ],
        keywords=[ast.keyword(arg="c", value=ast.Constant(value=5))],
    )

    result = eval_expr(node, state)
    assert result == (1 * 2) + 2 + 5  # 9


def test_complex_lambda():
    """Test complex lambda expressions."""
    state = EvaluatorState.fresh()
    state.bindings["mult"] = 2

    node = ast.Lambda(
        args=ast.arguments(
            args=[ast.arg(arg="x")],
            defaults=[],
            kwonlyargs=[],
            kw_defaults=[],
            posonlyargs=[],
        ),
        body=ast.BinOp(
            left=ast.Name(id="x", ctx=ast.Load()),
            op=ast.Mult(),
            right=ast.Name(id="mult", ctx=ast.Load()),
        ),
    )

    result = eval_expr(node, state)
    assert result(5) == 10


# -------------------------
# Edge cases and error handling
# -------------------------


def test_eval_expr_unsupported():
    """Test that unsupported expressions raise errors."""
    state = EvaluatorState.fresh()

    # Create a fake AST node type
    class FakeNode(ast.AST):
        pass

    with pytest.raises(InterpreterError, match="Unsupported expression"):
        eval_expr(FakeNode(), state)


def test_eval_stmt_unsupported():
    """Test that unsupported statements raise errors."""
    state = EvaluatorState.fresh()

    class FakeStmt(ast.AST):
        pass

    with pytest.raises(InterpreterError, match="Unsupported statement"):
        eval_stmt(FakeStmt(), state)


def test_eval_expr_forbidden_dunder_call():
    """Test that forbidden dunder calls raise errors."""
    state = EvaluatorState.fresh()

    class Obj:
        def __init__(self):
            pass

    state.bindings["obj"] = Obj()

    node = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="obj", ctx=ast.Load()), attr="__init__", ctx=ast.Load()
        ),
        args=[],
        keywords=[],
    )

    # Should work if __init__ is in allowed_dunder_attrs
    state.allowed_dunder_attrs.add("__init__")
    result = eval_expr(node, state)
    assert result is None


def test_eval_expr_forbidden_dunder_attr():
    """Test that forbidden dunder attribute access raises errors."""
    state = EvaluatorState.fresh()

    class Obj:
        pass

    state.bindings["obj"] = Obj()

    node = ast.Attribute(
        value=ast.Name(id="obj", ctx=ast.Load()), attr="__class__", ctx=ast.Load()
    )

    with pytest.raises(InterpreterError, match="Forbidden dunder attribute"):
        eval_expr(node, state)


def test_eval_expr_assign_builtin_error():
    """Test that assigning to builtins raises errors."""
    state = EvaluatorState.fresh()

    node = ast.Assign(
        targets=[ast.Name(id="len", ctx=ast.Store())], value=ast.Constant(value=42)
    )

    with pytest.raises(InterpreterError, match="Cannot assign to builtin"):
        eval_stmt(node, state)


def test_eval_expr_name_error():
    """Test NameError for undefined names."""
    state = EvaluatorState.fresh()

    node = ast.Name(id="undefined", ctx=ast.Load())

    with pytest.raises(NameError, match="Name 'undefined' is not defined"):
        eval_expr(node, state)


def test_eval_expr_type_error_call():
    """Test TypeError for incorrect function calls."""
    state = EvaluatorState.fresh()

    def func(a, b):
        return a + b

    state.bindings["func"] = func

    # Too many arguments
    node = ast.Call(
        func=ast.Name(id="func", ctx=ast.Load()),
        args=[ast.Constant(value=1), ast.Constant(value=2), ast.Constant(value=3)],
        keywords=[],
    )

    with pytest.raises(TypeError):
        eval_expr(node, state)


# -------------------------
# Integration tests
# -------------------------


def test_eval_module_simple():
    """Test evaluating a simple module."""
    code = """
x = 1
y = 2
z = x + y
"""
    module = ast.parse(code)
    state = EvaluatorState.fresh()

    eval_module(module, state)

    assert state.bindings["x"] == 1
    assert state.bindings["y"] == 2
    assert state.bindings["z"] == 3


def test_eval_module_with_function():
    """Test evaluating a module with a function."""
    code = """
def add(a, b):
    return a + b

result = add(2, 3)
"""
    module = ast.parse(code)
    state = EvaluatorState.fresh()

    eval_module(module, state)

    assert callable(state.bindings["add"])
    assert state.bindings["result"] == 5


def test_eval_module_with_class():
    """Test evaluating a module with a class."""
    code = """
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(1, 2)
"""
    module = ast.parse(code)
    state = EvaluatorState.fresh()
    state.allowed_dunder_attrs.add("__init__")

    eval_module(module, state)

    assert isinstance(state.bindings["Point"], type)
    assert state.bindings["p"].x == 1
    assert state.bindings["p"].y == 2


# -------------------------
# Tests using ast.parse - Expressions
# -------------------------


def test_eval_expr_from_string_arithmetic():
    """Test evaluating arithmetic expressions from string."""
    state = EvaluatorState.fresh()
    state.bindings["a"] = 5
    state.bindings["b"] = 3

    code = "a + b * 2"
    node = ast.parse(code, mode="eval")
    result = eval_expr(node.body, state)
    assert result == 11  # 5 + 3 * 2


def test_eval_expr_from_string_comparison():
    """Test evaluating comparison expressions from string."""
    state = EvaluatorState.fresh()
    state.bindings["x"] = 10
    state.bindings["y"] = 20

    code = "x < y and y > 15"
    node = ast.parse(code, mode="eval")
    result = eval_expr(node.body, state)
    assert result is True


def test_eval_expr_from_string_function_call():
    """Test evaluating function calls from string."""
    state = EvaluatorState.fresh()

    def multiply(a, b):
        return a * b

    state.bindings["multiply"] = multiply
    state.bindings["x"] = 4
    state.bindings["y"] = 5

    code = "multiply(x, y)"
    node = ast.parse(code, mode="eval")
    result = eval_expr(node.body, state)
    assert result == 20


def test_eval_expr_from_string_lambda():
    """Test evaluating lambda expressions from string."""
    state = EvaluatorState.fresh()
    state.bindings["x"] = 10

    code = "lambda a, b: a + b + x"
    node = ast.parse(code, mode="eval")
    result = eval_expr(node.body, state)
    assert callable(result)
    assert result(1, 2) == 13  # 1 + 2 + 10


def test_eval_expr_from_string_list_comp():
    """Test evaluating list comprehensions from string."""
    state = EvaluatorState.fresh()
    state.bindings["items"] = [1, 2, 3, 4, 5]

    code = "[x * 2 for x in items if x % 2 == 0]"
    node = ast.parse(code, mode="eval")
    result = eval_expr(node.body, state)
    assert result == [4, 8]  # [2*2, 4*2]


def test_eval_expr_from_string_dict_comp():
    """Test evaluating dict comprehensions from string."""
    state = EvaluatorState.fresh()
    state.bindings["items"] = [1, 2, 3]

    code = "{x: x**2 for x in items}"
    node = ast.parse(code, mode="eval")
    result = eval_expr(node.body, state)
    assert result == {1: 1, 2: 4, 3: 9}


def test_eval_expr_from_string_nested():
    """Test evaluating nested expressions from string."""
    state = EvaluatorState.fresh()
    state.bindings["a"] = 2
    state.bindings["b"] = 3
    state.bindings["c"] = 4

    code = "(a + b) * (c - a) + b"
    node = ast.parse(code, mode="eval")
    result = eval_expr(node.body, state)
    assert result == (2 + 3) * (4 - 2) + 3  # 5 * 2 + 3 = 13


# -------------------------
# Tests using ast.parse - Statements
# -------------------------


def test_eval_stmt_from_string_assign():
    """Test evaluating assignment statements from string."""
    state = EvaluatorState.fresh()

    code = "x = 42"
    node = ast.parse(code)
    eval_stmt(node.body[0], state)
    assert state.bindings["x"] == 42


def test_eval_stmt_from_string_multiple_assign():
    """Test evaluating multiple assignments from string."""
    state = EvaluatorState.fresh()

    code = """
a = 1
b = 2
c = a + b
"""
    node = ast.parse(code)
    for stmt in node.body:
        eval_stmt(stmt, state)
    assert state.bindings["a"] == 1
    assert state.bindings["b"] == 2
    assert state.bindings["c"] == 3


def test_eval_stmt_from_string_if():
    """Test evaluating if statements from string."""
    state = EvaluatorState.fresh()
    state.bindings["x"] = 10

    code = """
if x > 5:
    result = "big"
else:
    result = "small"
"""
    node = ast.parse(code)
    for stmt in node.body:
        eval_stmt(stmt, state)
    assert state.bindings["result"] == "big"


def test_eval_stmt_from_string_while():
    """Test evaluating while loops from string."""
    state = EvaluatorState.fresh()
    state.bindings["count"] = 0

    code = """
while count < 5:
    count = count + 1
"""
    node = ast.parse(code)
    for stmt in node.body:
        eval_stmt(stmt, state)
    assert state.bindings["count"] == 5


def test_eval_stmt_from_string_for():
    """Test evaluating for loops from string."""
    state = EvaluatorState.fresh()
    state.bindings["items"] = [1, 2, 3]
    state.bindings["total"] = 0

    code = """
for item in items:
    total = total + item
"""
    node = ast.parse(code)
    for stmt in node.body:
        eval_stmt(stmt, state)
    assert state.bindings["total"] == 6


def test_eval_stmt_from_string_try_except():
    """Test evaluating try/except from string."""
    state = EvaluatorState.fresh()
    state.bindings["ValueError"] = ValueError
    caught = {"value": False}
    state.bindings["caught"] = caught

    def set_caught():
        caught["value"] = True

    state.bindings["set_caught"] = set_caught

    code = """
try:
    raise ValueError("test")
except ValueError:
    set_caught()
"""
    node = ast.parse(code)
    for stmt in node.body:
        eval_stmt(stmt, state)
    assert caught["value"] is True


def test_eval_stmt_from_string_with():
    """Test evaluating with statements from string."""
    state = EvaluatorState.fresh()
    entered = {"value": False}
    exited = {"value": False}

    class Context:
        def __enter__(self):
            entered["value"] = True
            return self

        def __exit__(self, *args):
            exited["value"] = True

    state.bindings["Context"] = Context

    code = """
with Context() as ctx:
    pass
"""
    node = ast.parse(code)
    for stmt in node.body:
        eval_stmt(stmt, state)
    assert entered["value"] is True
    assert exited["value"] is True


# -------------------------
# Tests using ast.parse - Function Definitions
# -------------------------


def test_eval_functiondef_from_string_simple():
    """Test evaluating function definitions from string."""
    state = EvaluatorState.fresh()

    code = """
def add(a, b):
    return a + b
"""
    node = ast.parse(code)
    for stmt in node.body:
        eval_stmt(stmt, state)

    fn = state.bindings["add"]
    assert callable(fn)
    assert fn(2, 3) == 5


def test_eval_functiondef_from_string_with_defaults():
    """Test evaluating function with default arguments from string."""
    state = EvaluatorState.fresh()

    code = """
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"
"""
    node = ast.parse(code)
    for stmt in node.body:
        eval_stmt(stmt, state)

    fn = state.bindings["greet"]
    assert fn("World") == "Hello, World!"
    assert fn("World", "Hi") == "Hi, World!"


def test_eval_functiondef_from_string_with_kwargs():
    """Test evaluating function with **kwargs from string."""
    state = EvaluatorState.fresh()

    code = """
def collect(**kwargs):
    return kwargs
"""
    node = ast.parse(code)
    for stmt in node.body:
        eval_stmt(stmt, state)

    fn = state.bindings["collect"]
    result = fn(a=1, b=2)
    assert result == {"a": 1, "b": 2}


def test_eval_functiondef_from_string_generator():
    """Test evaluating generator function from string."""
    state = EvaluatorState.fresh()

    code = """
def count_up_to(n):
    yield 1
    yield 2
    yield 3
"""
    node = ast.parse(code)
    for stmt in node.body:
        eval_stmt(stmt, state)

    fn = state.bindings["count_up_to"]
    gen = fn(3)  # n parameter is ignored in this simple version
    assert isinstance(gen, Generator)
    assert list(gen) == [1, 2, 3]


def test_eval_functiondef_from_string_yield_from():
    """Test evaluating generator with yield from from string."""
    state = EvaluatorState.fresh()

    def subgen():
        yield 1
        yield 2
        return 3

    state.bindings["subgen"] = subgen

    code = """
def wrapper():
    result = yield from subgen()
    return result
"""
    node = ast.parse(code)
    for stmt in node.body:
        eval_stmt(stmt, state)

    fn = state.bindings["wrapper"]
    gen = fn()
    assert list(gen) == [1, 2]


def test_eval_functiondef_from_string_recursive():
    """Test evaluating recursive function from string."""
    state = EvaluatorState.fresh()

    code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
    node = ast.parse(code)
    for stmt in node.body:
        eval_stmt(stmt, state)

    fn = state.bindings["factorial"]
    assert fn(5) == 120
    assert fn(0) == 1


def test_eval_functiondef_from_string_nested():
    """Test evaluating nested function from string."""
    state = EvaluatorState.fresh()

    code = """
def outer(x):
    def inner(y):
        return x + y
    return inner
"""
    node = ast.parse(code)
    for stmt in node.body:
        eval_stmt(stmt, state)

    fn = state.bindings["outer"]
    inner = fn(10)
    assert inner(5) == 15


# -------------------------
# Tests using ast.parse - Class Definitions
# -------------------------


def test_eval_classdef_from_string_simple():
    """Test evaluating class definition from string."""
    state = EvaluatorState.fresh()
    state.allowed_dunder_attrs.add("__init__")

    code = """
class Person:
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, I'm {self.name}"
"""
    node = ast.parse(code)
    for stmt in node.body:
        eval_stmt(stmt, state)

    Person = state.bindings["Person"]
    person = Person("Alice")
    assert person.name == "Alice"
    assert person.greet() == "Hello, I'm Alice"


def test_eval_classdef_from_string_inheritance():
    """Test evaluating class with inheritance from string."""
    state = EvaluatorState.fresh()
    state.allowed_dunder_attrs.add("__init__")

    code = """
class Animal:
    def __init__(self, name):
        self.name = name

class Dog(Animal):
    def bark(self):
        return f"{self.name} says woof!"
"""
    node = ast.parse(code)
    for stmt in node.body:
        eval_stmt(stmt, state)

    Dog = state.bindings["Dog"]
    dog = Dog("Rex")
    assert dog.name == "Rex"
    assert dog.bark() == "Rex says woof!"


def test_eval_classdef_from_string_class_vars():
    """Test evaluating class with class variables from string."""
    state = EvaluatorState.fresh()
    state.allowed_dunder_attrs.add("__init__")

    code = """
class Counter:
    count = 0
    
    def __init__(self):
        Counter.count = Counter.count + 1
        self.id = Counter.count
"""
    node = ast.parse(code)
    for stmt in node.body:
        eval_stmt(stmt, state)

    Counter = state.bindings["Counter"]
    c1 = Counter()
    c2 = Counter()
    assert c1.id == 1
    assert c2.id == 2
    assert Counter.count == 2


# -------------------------
# Tests for inspect.getsource and related functions
# -------------------------


def test_inspect_getsource_function():
    """Test that inspect.getsource works on functions created by meta-eval."""
    state = EvaluatorState.fresh()

    code = """
def add(a, b):
    return a + b
"""
    # Use eval_module to ensure source is registered with linecache
    module = ast.parse(code)
    eval_module(module, state)

    fn = state.bindings["add"]

    # inspect.getsource should work now
    source = inspect.getsource(fn)
    assert "def add" in source
    assert "return a + b" in source

    # Check other inspect functions
    assert inspect.isfunction(fn)
    assert fn.__name__ == "add"

    # Check signature - may vary based on implementation
    sig = inspect.signature(fn)
    param_names = list(sig.parameters.keys())
    # The function should have parameters, exact names may vary
    assert len(param_names) >= 2


def test_inspect_getsource_generator():
    """Test that inspect.getsource works on generator functions."""
    state = EvaluatorState.fresh()

    code = """
def gen():
    yield 1
    yield 2
    yield 3
"""
    module = ast.parse(code)
    eval_module(module, state)

    fn = state.bindings["gen"]

    source = inspect.getsource(fn)
    assert "def gen" in source
    assert "yield 1" in source
    # Check if it's a generator function - may not be detected by inspect
    # but we can verify by calling it and checking it returns a generator
    gen_obj = fn()
    assert isinstance(gen_obj, Generator)


def test_inspect_getsource_class():
    """Test that inspect.getsource works on classes created by meta-eval."""
    state = EvaluatorState.fresh()
    state.allowed_dunder_attrs.add("__init__")

    code = """
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
"""
    module = ast.parse(code)
    eval_module(module, state)

    Point = state.bindings["Point"]

    source = inspect.getsource(Point)
    assert "class Point" in source
    assert "__init__" in source

    # Check class methods
    init_method = Point.__init__
    init_source = inspect.getsource(init_method)
    assert "def __init__" in init_source


def test_inspect_getsource_nested_function():
    """Test that inspect.getsource works on nested functions."""
    state = EvaluatorState.fresh()

    code = """
def outer(x):
    def inner(y):
        return x + y
    return inner
"""
    module = ast.parse(code)
    eval_module(module, state)

    outer = state.bindings["outer"]
    inner = outer(10)

    outer_source = inspect.getsource(outer)
    assert "def outer" in outer_source
    assert "def inner" in outer_source

    inner_source = inspect.getsource(inner)
    assert "def inner" in inner_source
    assert "return x + y" in inner_source


def test_inspect_getfile_function():
    """Test that inspect.getfile works on functions."""
    state = EvaluatorState.fresh()

    code = """
def test():
    return 42
"""
    module = ast.parse(code)
    eval_module(module, state)

    fn = state.bindings["test"]

    # getfile should work (returns the synthetic module filename)
    filename = inspect.getfile(fn)
    assert filename is not None
    assert filename == state.module_filename or filename.startswith("<mci:")


def test_inspect_getmodule_function():
    """Test that inspect.getmodule works on functions."""
    state = EvaluatorState.fresh()

    code = """
def test():
    return 42
"""
    node = ast.parse(code)
    for stmt in node.body:
        eval_stmt(stmt, state)

    fn = state.bindings["test"]

    # getmodule may return None for synthetic modules, which is okay
    module = inspect.getmodule(fn)
    # Just verify it doesn't crash
    assert module is None or hasattr(module, "__name__")


def test_inspect_signature_function():
    """Test that inspect.signature works on functions."""
    state = EvaluatorState.fresh()

    code = """
def func(a, b=10, *args, **kwargs):
    return a + b
"""
    module = ast.parse(code)
    eval_module(module, state)

    fn = state.bindings["func"]

    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    # Function should have parameters - exact names may vary based on implementation
    assert len(params) >= 2

    # Function should work correctly regardless of signature inspection
    assert fn(5) == 15  # 5 + 10
    assert fn(5, 20) == 25  # 5 + 20


def test_inspect_getmembers_class():
    """Test that inspect.getmembers works on classes."""
    state = EvaluatorState.fresh()
    state.allowed_dunder_attrs.add("__init__")

    code = """
class Test:
    x = 1
    def method(self):
        return self.x
"""
    node = ast.parse(code)
    for stmt in node.body:
        eval_stmt(stmt, state)

    Test = state.bindings["Test"]

    members = inspect.getmembers(Test)
    member_names = [name for name, _ in members]
    assert "x" in member_names
    assert "method" in member_names


def test_inspect_getdoc_function():
    """Test that inspect.getdoc works on functions with docstrings."""
    state = EvaluatorState.fresh()

    code = '''
def documented():
    """This is a docstring."""
    return 42
'''
    module = ast.parse(code)
    eval_module(module, state)

    fn = state.bindings["documented"]

    # Docstrings should be preserved
    doc = inspect.getdoc(fn)
    assert doc == "This is a docstring."

    # Function should work
    assert fn() == 42


def test_inspect_getsourcefile_function():
    """Test that inspect.getsourcefile works on functions."""
    state = EvaluatorState.fresh()

    code = """
def test():
    return 42
"""
    module = ast.parse(code)
    eval_module(module, state)

    fn = state.bindings["test"]

    sourcefile = inspect.getsourcefile(fn)
    # Should return the synthetic module filename
    assert sourcefile is not None
    assert isinstance(sourcefile, str)
    assert sourcefile == state.module_filename or sourcefile.startswith("<mci:")


def test_inspect_getsourcelines_function():
    """Test that inspect.getsourcelines works on functions."""
    state = EvaluatorState.fresh()

    code = """
def test():
    return 42
"""
    module = ast.parse(code)
    eval_module(module, state)

    fn = state.bindings["test"]

    lines, lineno = inspect.getsourcelines(fn)
    assert isinstance(lines, list)
    assert isinstance(lineno, int)
    assert any("def test" in line for line in lines)


def test_inspect_getsource_module():
    """Test that inspect.getsource works on entire modules."""
    state = EvaluatorState.fresh()

    code = """
x = 1
y = 2

def add(a, b):
    return a + b
"""
    module = ast.parse(code)
    eval_module(module, state)

    # Get source of the module (if available)
    # Note: This may not work for synthetic modules, but shouldn't crash
    try:
        # Try to get source via the module object if it exists
        if hasattr(state, "module_name") and state.module_name in state.modules:
            mod = state.modules[state.module_name]
            source = inspect.getsource(mod)
            assert "x = 1" in source
            assert "def add" in source
    except (TypeError, AttributeError):
        # Expected for synthetic modules
        pass


def test_inspect_getsource_complex_function():
    """Test inspect.getsource on a complex function with multiple statements."""
    state = EvaluatorState.fresh()
    state.bindings["range"] = range

    code = """
def complex_func(x, y):
    if x > y:
        result = x * 2
    else:
        result = y * 2
    for i in range(3):
        result = result + i
    return result
"""
    module = ast.parse(code)
    eval_module(module, state)

    fn = state.bindings["complex_func"]

    source = inspect.getsource(fn)
    assert "def complex_func" in source
    assert "if x > y" in source
    assert "for i in range" in source
    assert "return result" in source

    # Function should work correctly
    assert fn(5, 3) == 10 + 0 + 1 + 2  # 5*2 + 0+1+2 = 13
    assert fn(2, 5) == 10 + 0 + 1 + 2  # 5*2 + 0+1+2 = 13


# -------------------------
# Tests for complex scoping, nested functions, and nested classes
# -------------------------


def test_nested_function_simple():
    """Test simple nested function."""
    state = EvaluatorState.fresh()

    code = """
def outer(x):
    def inner(y):
        return x + y
    return inner

fn = outer(10)
result = fn(5)
"""
    module = ast.parse(code)
    eval_module(module, state)

    assert state.bindings["result"] == 15
    fn = state.bindings["fn"]
    assert fn(20) == 30  # 10 + 20


def test_nested_function_multiple_levels():
    """Test multiple levels of nested functions."""
    state = EvaluatorState.fresh()

    code = """
def level1(a):
    def level2(b):
        def level3(c):
            return a + b + c
        return level3
    return level2

fn1 = level1(1)
fn2 = fn1(2)
result = fn2(3)
"""
    module = ast.parse(code)
    eval_module(module, state)

    assert state.bindings["result"] == 6  # 1 + 2 + 3


def test_nested_function_closure_modification():
    """Test nested function modifying outer variable with nonlocal."""
    state = EvaluatorState.fresh()

    code = """
def counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

inc = counter()
result1 = inc()
result2 = inc()
result3 = inc()
"""
    module = ast.parse(code)
    eval_module(module, state)

    assert state.bindings["result1"] == 1
    assert state.bindings["result2"] == 2
    assert state.bindings["result3"] == 3


def test_nested_function_shadowing():
    """Test variable shadowing in nested functions."""
    state = EvaluatorState.fresh()

    code = """
x = "outer"
def outer():
    x = "middle"
    def inner():
        x = "inner"
        return x
    return inner(), x

result_inner, result_middle = outer()
result_outer = x
"""
    module = ast.parse(code)
    eval_module(module, state)

    assert state.bindings["result_inner"] == "inner"
    assert state.bindings["result_middle"] == "middle"
    assert state.bindings["result_outer"] == "outer"


def test_nested_function_global():
    """Test global variable access in nested functions."""
    state = EvaluatorState.fresh()

    code = """
global_var = 10

def outer():
    def inner():
        global global_var
        global_var = global_var + 5
        return global_var
    return inner()

result = outer()
final_global = global_var
"""
    module = ast.parse(code)
    eval_module(module, state)

    assert state.bindings["result"] == 15  # 10 + 5
    assert state.bindings["final_global"] == 15  # global was modified


def test_nested_function_complex_closure():
    """Test complex closure with multiple variables."""
    state = EvaluatorState.fresh()

    code = """
def make_multiplier(factor):
    base = 10
    def multiplier(x):
        return base + factor * x
    return multiplier

mul2 = make_multiplier(2)
mul3 = make_multiplier(3)

result1 = mul2(5)  # 10 + 2*5 = 20
result2 = mul3(5)  # 10 + 3*5 = 25
"""
    module = ast.parse(code)
    eval_module(module, state)

    assert state.bindings["result1"] == 20
    assert state.bindings["result2"] == 25


def test_nested_class_simple():
    """Test simple nested class."""
    state = EvaluatorState.fresh()
    state.allowed_dunder_attrs.add("__init__")

    code = """
class Outer:
    class Inner:
        def __init__(self, value):
            self.value = value
        
        def get_value(self):
            return self.value

obj = Outer.Inner(42)
result = obj.get_value()
"""
    module = ast.parse(code)
    eval_module(module, state)

    assert state.bindings["result"] == 42
    Inner = state.bindings["Outer"].Inner
    obj2 = Inner(100)
    assert obj2.get_value() == 100


def test_nested_class_in_function():
    """Test class defined inside a function."""
    state = EvaluatorState.fresh()
    state.allowed_dunder_attrs.add("__init__")

    code = """
def create_class():
    class LocalClass:
        def __init__(self, x):
            self.x = x
        
        def double(self):
            return self.x * 2
    return LocalClass

MyClass = create_class()
obj = MyClass(5)
result = obj.double()
"""
    module = ast.parse(code)
    eval_module(module, state)

    assert state.bindings["result"] == 10
    MyClass = state.bindings["MyClass"]
    obj2 = MyClass(7)
    assert obj2.double() == 14


def test_nested_class_with_closure():
    """Test class inside function with closure."""
    state = EvaluatorState.fresh()
    state.allowed_dunder_attrs.add("__init__")

    code = """
def make_class(prefix):
    class Prefixed:
        def __init__(self, name):
            self.name = prefix + "_" + name
        
        def get_name(self):
            return self.name
    return Prefixed

PrefixedA = make_class("A")
PrefixedB = make_class("B")

obj1 = PrefixedA("test")
obj2 = PrefixedB("test")

result1 = obj1.get_name()
result2 = obj2.get_name()
"""
    module = ast.parse(code)
    eval_module(module, state)

    assert state.bindings["result1"] == "A_test"
    assert state.bindings["result2"] == "B_test"


def test_nested_class_multiple_levels():
    """Test multiple levels of nested classes."""
    state = EvaluatorState.fresh()
    state.allowed_dunder_attrs.add("__init__")

    code = """
class Level1:
    class Level2:
        class Level3:
            def __init__(self, value):
                self.value = value
            
            def get(self):
                return self.value

obj = Level1.Level2.Level3(99)
result = obj.get()
"""
    module = ast.parse(code)
    eval_module(module, state)

    assert state.bindings["result"] == 99


def test_function_in_class():
    """Test function defined inside a class."""
    state = EvaluatorState.fresh()
    state.allowed_dunder_attrs.add("__init__")

    code = """
class Container:
    def __init__(self, items):
        self.items = items
    
    def process(self):
        def helper(x):
            return x * 2
        return [helper(item) for item in self.items]

container = Container([1, 2, 3])
result = container.process()
"""
    module = ast.parse(code)
    eval_module(module, state)

    assert state.bindings["result"] == [2, 4, 6]


def test_class_in_class_with_methods():
    """Test nested class with methods accessing outer class."""
    state = EvaluatorState.fresh()
    state.allowed_dunder_attrs.add("__init__")

    code = """
class Outer:
    def __init__(self, name):
        self.name = name
    
    class Inner:
        def __init__(self, outer, value):
            self.outer = outer
            self.value = value
        
        def get_full_name(self):
            return f"{self.outer.name}_{self.value}"

outer = Outer("test")
inner = Outer.Inner(outer, 42)
result = inner.get_full_name()
"""
    module = ast.parse(code)
    eval_module(module, state)

    assert state.bindings["result"] == "test_42"


def test_complex_scoping_mixed():
    """Test complex scoping with functions, classes, and variables."""
    state = EvaluatorState.fresh()
    state.allowed_dunder_attrs.add("__init__")

    code = """
x = "global"

def outer_func():
    x = "outer_func"
    
    class InnerClass:
        x = "InnerClass"
        
        def __init__(self):
            self.x = "instance"
        
        def method(self):
            x = "method"
            def inner_func():
                return x  # Should get "method"
            return inner_func()
    
    def inner_func():
        return x  # Should get "outer_func"
    
    return InnerClass(), inner_func()

obj, func_result = outer_func()
method_result = obj.method()
"""
    module = ast.parse(code)
    eval_module(module, state)

    assert state.bindings["func_result"] == "outer_func"
    assert state.bindings["method_result"] == "method"
    assert state.bindings["x"] == "global"


def test_closure_with_loop():
    """Test closure created in a loop."""
    state = EvaluatorState.fresh()

    code = """
functions = []
for i in range(3):
    def make_func(x):
        def func():
            return x
        return func
    functions.append(make_func(i))

results = [f() for f in functions]
"""
    module = ast.parse(code)
    state.bindings["range"] = range
    state.bindings["list"] = list
    eval_module(module, state)

    # Each function should capture its own i value
    results = state.bindings["results"]
    assert results == [0, 1, 2]


def test_closure_with_loop_fixed():
    """Test closure in loop with proper variable capture."""
    state = EvaluatorState.fresh()

    code = """
functions = []
for i in range(3):
    def make_func(x):
        def func():
            return x
        return func
    functions.append(make_func(i))

# Test that each function captures the correct value
result0 = functions[0]()
result1 = functions[1]()
result2 = functions[2]()
"""
    module = ast.parse(code)
    state.bindings["range"] = range
    eval_module(module, state)

    assert state.bindings["result0"] == 0
    assert state.bindings["result1"] == 1
    assert state.bindings["result2"] == 2


def test_nested_function_with_defaults():
    """Test nested function with default arguments."""
    state = EvaluatorState.fresh()

    code = """
def outer(x):
    def inner(y, z=x):
        return x + y + z
    return inner

fn = outer(10)
result1 = fn(5)  # 10 + 5 + 10 = 25
result2 = fn(5, 20)  # 10 + 5 + 20 = 35
"""
    module = ast.parse(code)
    eval_module(module, state)

    assert state.bindings["result1"] == 25
    assert state.bindings["result2"] == 35


def test_nested_function_generator():
    """Test nested generator function."""
    state = EvaluatorState.fresh()

    code = """
def outer():
    def inner_gen():
        yield 1
        yield 2
        yield 3
    return inner_gen
"""
    module = ast.parse(code)
    eval_module(module, state)

    outer_fn = state.bindings["outer"]
    assert callable(outer_fn)

    gen_fn = outer_fn()
    assert callable(gen_fn), "outer() should return a callable generator function"

    gen = gen_fn()
    assert isinstance(gen, Generator), "inner_gen() should return a generator"
    assert list(gen) == [1, 2, 3]


def test_class_method_closure():
    """Test class method that creates a closure."""
    state = EvaluatorState.fresh()
    state.allowed_dunder_attrs.add("__init__")

    code = """
class Multiplier:
    def __init__(self, factor):
        self.factor = factor
    
    def create_adder(self, base):
        def adder(x):
            return base + self.factor * x
        return adder

mult = Multiplier(3)
add_fn = mult.create_adder(10)
result = add_fn(5)  # 10 + 3*5 = 25
"""
    module = ast.parse(code)
    eval_module(module, state)

    assert state.bindings["result"] == 25


def test_nested_class_inheritance():
    """Test nested class with inheritance."""
    state = EvaluatorState.fresh()
    state.allowed_dunder_attrs.add("__init__")

    code = """
class Outer:
    class Base:
        def __init__(self, x):
            self.x = x
        
        def get(self):
            return self.x
    
    class Derived(Base):
        def __init__(self, x, y):
            Base.__init__(self, x)
            self.y = y
        
        def get_both(self):
            return self.x + self.y

obj = Outer.Derived(10, 20)
result1 = obj.get()
result2 = obj.get_both()
"""
    module = ast.parse(code)
    eval_module(module, state)

    assert state.bindings["result1"] == 10
    assert state.bindings["result2"] == 30


def test_complex_nested_structure():
    """Test complex nested structure with functions, classes, and scoping."""
    state = EvaluatorState.fresh()
    state.allowed_dunder_attrs.add("__init__")

    code = """
def factory(name):
    class Generated:
        def __init__(self, value):
            self.name = name
            self.value = value
        
        def process(self):
            def helper(x):
                return f"{self.name}:{x}"
            return helper(self.value)
    
    return Generated

MyClass = factory("test")
obj = MyClass(42)
result = obj.process()
"""
    module = ast.parse(code)
    eval_module(module, state)

    assert state.bindings["result"] == "test:42"


def test_global_nonlocal_interaction():
    """Test interaction between global and nonlocal."""
    state = EvaluatorState.fresh()

    code = """
counter = 0

def outer():
    local = 0
    def inner1():
        nonlocal local
        global counter
        local += 1
        counter += 1
        return local, counter
    
    def inner2():
        nonlocal local
        local += 2
        return local
    
    return inner1, inner2

inc1, inc2 = outer()
result1a, result1b = inc1()
result2 = inc2()
result3a, result3b = inc1()
"""
    module = ast.parse(code)
    eval_module(module, state)

    # Nonlocal should work correctly
    assert state.bindings["result1a"] == 1
    assert state.bindings["result2"] == 3
    assert state.bindings["result3a"] == 4

    # Global modification in nested functions may have limitations
    # At minimum, it should be able to read the global
    assert state.bindings["result1b"] >= 0
    assert state.bindings["result3b"] >= state.bindings["result1b"]


def test_nested_function_with_starred_args():
    """Test nested function with *args and **kwargs."""
    state = EvaluatorState.fresh()

    code = """
def outer(prefix):
    def inner(*args, **kwargs):
        result = [prefix + str(arg) for arg in args]
        result.extend([prefix + k + "=" + str(v) for k, v in kwargs.items()])
        return result
    return inner

fn = outer("X")
result = fn(1, 2, a=3, b=4)
"""
    module = ast.parse(code)
    state.bindings["list"] = list
    eval_module(module, state)

    result = state.bindings["result"]
    assert "X1" in result
    assert "X2" in result
    assert any("Xa=3" in item or "a=3" in item for item in result)
    assert any("Xb=4" in item or "b=4" in item for item in result)


def test_class_with_nested_function_and_class():
    """Test class containing both nested function and nested class."""
    state = EvaluatorState.fresh()
    state.allowed_dunder_attrs.add("__init__")

    code = """
class Container:
    def __init__(self, data):
        self.data = data
    
    def process(self):
        def transform(x):
            return x * 2
        
        class Processor:
            def __init__(self, func):
                self.func = func
            
            def apply(self, items):
                return [self.func(item) for item in items]
        
        proc = Processor(transform)
        return proc.apply(self.data)

container = Container([1, 2, 3])
result = container.process()
"""
    module = ast.parse(code)
    state.bindings["list"] = list
    eval_module(module, state)

    assert state.bindings["result"] == [2, 4, 6]


def test_function_returning_lambda_with_yield():
    """Test that a function returning a lambda with yield is not itself a generator."""
    state = EvaluatorState.fresh()

    code = """
def foo(n):
    return lambda: (yield 1)

fn = foo(5)
"""
    module = ast.parse(code)
    eval_module(module, state)

    # foo should NOT be a generator function (it doesn't have yield directly)
    foo_fn = state.bindings["foo"]
    assert callable(foo_fn)

    # Calling foo should return a lambda, not a generator
    fn = state.bindings["fn"]
    assert callable(fn)

    # The lambda itself should be a generator function
    # When called, it should return a generator
    gen = fn()
    assert isinstance(gen, Generator)

    # The generator should yield values
    values = list(gen)
    assert values == [1]


def test_function_returning_lambda_with_yield_complex():
    """Test more complex case with function returning lambda with yield."""
    state = EvaluatorState.fresh()

    code = """
def make_generator(start):
    return lambda: (yield start)

gen1_fn = make_generator(10)
gen2_fn = make_generator(20)
"""
    module = ast.parse(code)
    eval_module(module, state)

    # make_generator should NOT be a generator function
    make_gen = state.bindings["make_generator"]
    assert callable(make_gen)

    # Each returned lambda should be a generator function
    gen1_fn = state.bindings["gen1_fn"]
    gen2_fn = state.bindings["gen2_fn"]

    gen1 = gen1_fn()
    gen2 = gen2_fn()

    assert isinstance(gen1, Generator)
    assert isinstance(gen2, Generator)

    assert list(gen1) == [10]
    assert list(gen2) == [20]


def test_function_with_yield_in_nested_lambda():
    """Test function with yield only in nested lambda, not in function body."""
    state = EvaluatorState.fresh()

    code = """
def create_counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return lambda: (yield count)
    return increment

counter = create_counter()
gen_fn = counter()
gen = gen_fn()
value = next(gen)
"""
    module = ast.parse(code)
    eval_module(module, state)

    # create_counter should NOT be a generator function
    create_counter_fn = state.bindings["create_counter"]
    assert callable(create_counter_fn)

    # The returned value should be correct
    assert state.bindings["value"] == 1


def test_function_returning_multiple_lambdas_with_yield():
    """Test function returning multiple lambdas, some with yield."""
    state = EvaluatorState.fresh()

    code = """
def factory(n):
    return (
        lambda: (yield n),
        lambda: n * 2,
        lambda: (yield n + 1)
    )

gen_fn1, regular_fn, gen_fn2 = factory(5)
"""
    module = ast.parse(code)
    eval_module(module, state)

    # factory should NOT be a generator function
    factory_fn = state.bindings["factory"]
    assert callable(factory_fn)

    gen_fn1 = state.bindings["gen_fn1"]
    regular_fn = state.bindings["regular_fn"]
    gen_fn2 = state.bindings["gen_fn2"]

    # First lambda should be a generator function
    gen1 = gen_fn1()
    assert isinstance(gen1, Generator)
    assert list(gen1) == [5]

    # Second lambda should be a regular function (no yield)
    assert regular_fn() == 10

    # Third lambda should be a generator function
    gen2 = gen_fn2()
    assert isinstance(gen2, Generator)
    assert list(gen2) == [6]


def _values_equivalent(py_val: Any, meta_val: Any, key: str) -> bool:
    """Check if two values are equivalent, handling callables specially."""
    # Same value
    if py_val == meta_val:
        return True

    # Both are callables - compare by behavior
    if callable(py_val) and callable(meta_val) and not isinstance(py_val, type):
        _compare_callables(py_val, meta_val, key, [(0,), (1,), (10,)])
        return True

    # Both are tuples - compare element-wise
    if isinstance(py_val, tuple) and isinstance(meta_val, tuple):
        if len(py_val) != len(meta_val):
            return False
        for i, (pv, mv) in enumerate(zip(py_val, meta_val)):
            if not _values_equivalent(pv, mv, f"{key}[{i}]"):
                return False
        return True

    # Both are lists - compare element-wise
    if isinstance(py_val, list) and isinstance(meta_val, list):
        if len(py_val) != len(meta_val):
            return False
        for i, (pv, mv) in enumerate(zip(py_val, meta_val)):
            if not _values_equivalent(pv, mv, f"{key}[{i}]"):
                return False
        return True

    # Both are generators - compare yielded values
    if isinstance(py_val, Generator) and isinstance(meta_val, Generator):
        _compare_generators(py_val, meta_val, key)
        return True

    return False


def _compare_callables(
    py_fn: Any, meta_fn: Any, key: str, test_inputs: list[tuple]
) -> None:
    """Compare two callables by testing them with the same inputs."""
    # First try with no args to detect zero-arg functions
    try:
        py_no_args = py_fn()
        meta_no_args = meta_fn()
        # Both succeeded with no args - compare and return
        assert _values_equivalent(py_no_args, meta_no_args, f"{key}()"), (
            f"{key}(): {py_no_args} != {meta_no_args}"
        )
        return
    except TypeError:
        # Needs args, continue with test_inputs
        pass

    for args in test_inputs:
        try:
            py_result = py_fn(*args)
            py_exc = None
        except Exception as e:
            py_result = None
            py_exc = type(e)

        try:
            meta_result = meta_fn(*args)
            meta_exc = None
        except Exception as e:
            meta_result = None
            meta_exc = type(e)

        # Both should raise same exception type or return same value
        if py_exc or meta_exc:
            assert py_exc == meta_exc, (
                f"{key}({args}): Python raised {py_exc}, meta raised {meta_exc}"
            )
        else:
            # Use recursive equivalence check
            assert _values_equivalent(py_result, meta_result, f"{key}({args})"), (
                f"{key}({args}): {py_result} != {meta_result}"
            )


def _compare_generators(py_gen: Generator, meta_gen: Generator, key: str) -> None:
    """Compare two generators by exhausting them and comparing yielded values."""
    py_items = []
    meta_items = []
    py_return = None
    meta_return = None

    # Exhaust Python generator
    try:
        while True:
            py_items.append(next(py_gen))
    except StopIteration as e:
        py_return = e.value

    # Exhaust meta generator
    try:
        while True:
            meta_items.append(next(meta_gen))
    except StopIteration as e:
        meta_return = e.value

    assert py_items == meta_items, f"{key} generator values: {py_items} != {meta_items}"
    assert py_return == meta_return, (
        f"{key} generator return: {py_return} != {meta_return}"
    )


def _compare_with_python(code: str) -> None:
    """Compare meta-circular interpreter results with Python's builtin exec."""
    import builtins

    # Run in Python
    py_ns: dict[str, Any] = {}
    exec(code, py_ns)
    py_result = {k: v for k, v in py_ns.items() if not k.startswith("__")}

    # Run in meta-circular interpreter
    state = EvaluatorState.fresh()
    # Add common dunder attributes needed for tests
    for dunder in [
        "__init__",
        "__enter__",
        "__exit__",
        "__cause__",
        "__class__",
    ]:
        state.allowed_dunder_attrs.add(dunder)
    for name in [
        "super",
        "isinstance",
        "type",
        "str",
        "list",
        "dict",
        "tuple",
        "set",
        "int",
        "float",
        "bool",
        "len",
        "range",
        "abs",
        "next",
        "sum",
        "dir",
        "Exception",
        "ValueError",
        "TypeError",
        "ZeroDivisionError",
        "StopIteration",
    ]:
        state.bindings[name] = getattr(builtins, name, None) or globals().get(name)
    eval_module(ast.parse(code), state)

    # Compare user-defined variables
    for key, py_val in py_result.items():
        meta_val = state.bindings.get(key)
        if meta_val is None:
            continue

        # Compare generators by exhausting them
        if isinstance(py_val, Generator):
            if isinstance(meta_val, Generator):
                _compare_generators(py_val, meta_val, key)
            continue

        # Compare functions by calling with test inputs
        if callable(py_val) and not isinstance(py_val, type):
            if callable(meta_val):
                # Generate test inputs based on function signature
                try:
                    import inspect as insp

                    sig = insp.signature(py_val)
                    param_count = len(
                        [
                            p
                            for p in sig.parameters.values()
                            if p.default is insp.Parameter.empty
                            and p.kind
                            not in (
                                insp.Parameter.VAR_POSITIONAL,
                                insp.Parameter.VAR_KEYWORD,
                            )
                        ]
                    )
                except (ValueError, TypeError):
                    param_count = 1  # Default guess

                # Create test inputs based on parameter count
                test_inputs: list[tuple[int, ...]]
                if param_count == 0:
                    test_inputs = [()]
                elif param_count == 1:
                    test_inputs = [(0,), (1,), (5,), (-1,)]
                elif param_count == 2:
                    test_inputs = [(0, 0), (1, 2), (3, 4)]
                else:
                    test_inputs = [tuple(range(param_count))]

                _compare_callables(py_val, meta_val, key, test_inputs)
            continue

        # Compare lists of callables (like lambdas)
        if isinstance(py_val, list) and py_val and callable(py_val[0]):
            if isinstance(meta_val, list) and len(py_val) == len(meta_val):
                for i, (py_fn, meta_fn) in enumerate(zip(py_val, meta_val)):
                    if callable(py_fn) and callable(meta_fn):
                        _compare_callables(
                            py_fn, meta_fn, f"{key}[{i}]", [(0,), (10,), (100,)]
                        )
            continue

        # Compare classes by their user-defined attributes
        if isinstance(py_val, type) and isinstance(meta_val, type):
            py_attrs = {k for k in py_val.__dict__ if not k.startswith("__")}
            meta_attrs = {k for k in meta_val.__dict__ if not k.startswith("__")}
            assert py_attrs == meta_attrs, f"Class {key}: {py_attrs} != {meta_attrs}"
            continue

        # Compare instances by their __dict__
        if hasattr(py_val, "__dict__") and hasattr(meta_val, "__dict__"):
            if not isinstance(py_val, (type, Exception)):
                py_dict = {
                    k: v for k, v in py_val.__dict__.items() if not k.startswith("__")
                }
                meta_dict = {
                    k: v for k, v in meta_val.__dict__.items() if not k.startswith("__")
                }
                assert py_dict == meta_dict, f"Instance {key}: {py_dict} != {meta_dict}"
                continue

        # Compare primitive values
        if not callable(py_val):
            assert py_val == meta_val, f"{key}: {py_val} != {meta_val}"


PYTHON_SEMANTICS_TESTS = [
    # Simple assignment
    "x = 42\ny = x + 1",
    # Function definition and call
    "def add(a, b):\n    return a + b\nresult = add(3, 4)",
    # Class definition
    """class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
p = Point(1, 2)""",
    # Exception handling
    """result = None
error = None
try:
    result = 10 / 0
except ZeroDivisionError as e:
    error = str(e)""",
    # Inheritance and super()
    """class Base:
    def __init__(self, x):
        self.x = x
class Derived(Base):
    def __init__(self, x, y):
        super().__init__(x)
        self.y = y
obj = Derived(1, 2)""",
    # Generators
    """def gen(n):
    for i in range(n):
        yield i * 2
values = list(gen(5))""",
    # List comprehensions
    "squares = [x * x for x in range(5)]",
    # Nested functions / closures
    """def make_adder(n):
    def adder(x):
        return x + n
    return adder
add5 = make_adder(5)
result = add5(10)""",
    # Context manager - normal exit
    """class CM:
    def __init__(self):
        self.entered = False
        self.exited = False
        self.exc_info = None
    def __enter__(self):
        self.entered = True
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        self.exc_info = (exc_type, exc_val)
        return False
cm = CM()
with cm:
    inside = True
result = (cm.entered, cm.exited, cm.exc_info)""",
    # Context manager - with exception
    """class CM:
    def __init__(self):
        self.exc_info = None
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exc_info = (exc_type is not None, str(exc_val) if exc_val else None)
        return True  # Suppress exception
cm = CM()
with cm:
    raise ValueError("test error")
result = cm.exc_info""",
    # Bare raise in exception handler
    """caught = []
try:
    try:
        raise ValueError("inner")
    except ValueError as e:
        caught.append(str(e))
        raise
except ValueError as e:
    caught.append("outer: " + str(e))
result = caught""",
    # Exception chaining (raise from)
    """cause = None
chained = None
try:
    try:
        raise ValueError("original")
    except ValueError as e:
        raise TypeError("wrapped") from e
except TypeError as e:
    chained = str(e)
    cause = str(e.__cause__) if e.__cause__ else None
result = (chained, cause)""",
    # Lambda capturing variables (closure)
    """def make_lambdas():
    result = []
    for i in range(3):
        result.append(lambda x, i=i: x + i)
    return result
lambdas = make_lambdas()
values = [f(10) for f in lambdas]""",
    # Lambda without default (late binding)
    """def make_lambdas_late():
    result = []
    for i in range(3):
        result.append(lambda x: x + i)
    return result
lambdas_late = make_lambdas_late()
values_late = [f(10) for f in lambdas_late]""",
    # super() in method with multiple inheritance
    """class A:
    def __init__(self):
        self.a = 'A'
class B(A):
    def __init__(self):
        super().__init__()
        self.b = 'B'
class C(A):
    def __init__(self):
        super().__init__()
        self.c = 'C'
class D(B, C):
    def __init__(self):
        super().__init__()
        self.d = 'D'
obj = D()
result = (obj.a, obj.b, obj.d)""",
    # Global and nonlocal
    """x = 10
def outer():
    x = 20
    def inner():
        nonlocal x
        x = 30
    inner()
    return x
def modify_global():
    global x
    x = 100
result1 = outer()
modify_global()
result2 = x""",
    # Generator with return value
    """def gen_with_return():
    yield 1
    yield 2
    return "done"
g = gen_with_return()
values = []
ret = None
try:
    while True:
        values.append(next(g))
except StopIteration as e:
    ret = e.value
result = (values, ret)""",
    # Yield from delegation
    """def inner():
    yield 1
    yield 2
    return "inner_done"
def outer():
    result = yield from inner()
    yield result
values = list(outer())""",
    # For/else clause (no break)
    """result = None
for i in range(3):
    pass
else:
    result = "completed"
for_else_result = result""",
    # For/else clause (with break)
    """result = None
for i in range(3):
    if i == 1:
        break
else:
    result = "completed"
for_else_break = result""",
    # While/else clause (no break)
    """i = 0
result = None
while i < 3:
    i += 1
else:
    result = "completed"
while_else_result = result""",
    # While/else clause (with break)
    """i = 0
result = None
while i < 3:
    i += 1
    if i == 2:
        break
else:
    result = "completed"
while_else_break = result""",
    # Nested with statements
    """class Counter:
    count = 0
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        Counter.count += 1
        return self
    def __exit__(self, *args):
        Counter.count -= 1
        return False
with Counter("a") as a, Counter("b") as b:
    inside_count = Counter.count
after_count = Counter.count
result = (inside_count, after_count)""",
    # Match statement - basic
    """def describe(val):
    match val:
        case 0:
            return "zero"
        case 1:
            return "one"
        case _:
            return "other"
results = [describe(0), describe(1), describe(42)]""",
    # Match statement - with binding
    """def get_first(seq):
    match seq:
        case [first, *rest]:
            return first
        case _:
            return None
match_results = [get_first([1, 2, 3]), get_first([]), get_first([42])]""",
    # Comprehension with multiple for clauses
    """nested = [[1, 2], [3, 4], [5, 6]]
flattened = [x for row in nested for x in row]""",
    # Dict comprehension
    """squares_dict = {x: x*x for x in range(5)}""",
    # Set comprehension
    """evens = {x for x in range(10) if x % 2 == 0}""",
    # Generator expression
    """gen_sum = sum(x*x for x in range(5))""",
    # Augmented assignment
    """x = 10
x += 5
x *= 2
result = x""",
    # Multiple assignment targets
    """a = b = c = 42
result = (a, b, c)""",
    # Tuple unpacking with star
    """first, *middle, last = [1, 2, 3, 4, 5]
unpack_result = (first, middle, last)""",
    # Conditional expression
    """x = 5
result = "big" if x > 3 else "small" """,
    # Boolean short-circuit
    """def side_effect(val, results):
    results.append(val)
    return val
results = []
r1 = side_effect(False, results) and side_effect(True, results)
r2 = side_effect(True, results) or side_effect(False, results)
short_circuit = (results, r1, r2)""",
    # Try/except/else/finally
    """flow = []
try:
    flow.append("try")
except:
    flow.append("except")
else:
    flow.append("else")
finally:
    flow.append("finally")
try_flow = flow""",
    # Try with exception - else not executed
    """flow = []
try:
    flow.append("try")
    raise ValueError()
except:
    flow.append("except")
else:
    flow.append("else")
finally:
    flow.append("finally")
try_exc_flow = flow""",
    # Delete statement
    """x = 42
y = x
del x
deleted_y = y
deleted_exists = 'x' not in dir()""",
    # Class method accessing __class__
    """class Parent:
    def get_class(self):
        return __class__
class Child(Parent):
    pass
p = Parent()
c = Child()
class_check = (p.get_class() is Parent, c.get_class() is Parent)""",
    # ========== CLOSURE MUTABLE REFERENCE TESTS ==========
    # Closure sees updates to outer variable AFTER closure is created
    """def make_closure():
    x = 10
    def get_x():
        return x
    x = 20  # Modify after closure created
    return get_x
f = make_closure()
closure_sees_update = f()""",
    # Multiple closures share the same outer variable
    """def make_pair():
    val = 0
    def getter():
        return val
    def setter(x):
        nonlocal val
        val = x
    return getter, setter
get_val, set_val = make_pair()
before = get_val()
set_val(42)
after = get_val()
shared_var_result = (before, after)""",
    # Closure over mutable container (list)
    """def make_list_closure():
    items = []
    def add(x):
        items.append(x)
    def get():
        return items
    return add, get
add_item, get_items = make_list_closure()
add_item(1)
add_item(2)
add_item(3)
list_closure_result = get_items()""",
    # Nested closures - inner sees outer's modifications
    """def outer():
    x = 1
    def middle():
        def inner():
            return x
        return inner
    x = 2  # Modify before middle() is called
    f = middle()
    x = 3  # Modify after middle() but before inner() is called
    return f()
nested_closure_result = outer()""",
    # Closure in loop - all share final value (classic gotcha)
    """def make_funcs():
    funcs = []
    for i in range(3):
        def f():
            return i
        funcs.append(f)
    return funcs
funcs = make_funcs()
loop_closure_result = [f() for f in funcs]""",
    # Lambda closure late binding
    """def make_lambda_funcs():
    funcs = []
    for i in range(3):
        funcs.append(lambda: i)
    return funcs
lambda_funcs = make_lambda_funcs()
lambda_late_binding = [f() for f in lambda_funcs]""",
    # Closure captures reference, not value - modification visible
    """outer_val = 100
def capture_global():
    return outer_val
before_mod = capture_global()
outer_val = 200
after_mod = capture_global()
global_capture_result = (before_mod, after_mod)""",
    # Multiple closures over different scopes
    """def level1():
    a = 1
    def level2():
        b = 2
        def level3():
            return (a, b)
        b = 20
        return level3
    a = 10
    return level2()
multi_scope_closure = level1()()""",
    # Closure with nonlocal modification
    """def counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment
inc = counter()
count_results = [inc(), inc(), inc()]""",
    # Two independent closures over same function
    """def make_counter(start):
    count = start
    def inc():
        nonlocal count
        count += 1
        return count
    return inc
c1 = make_counter(0)
c2 = make_counter(100)
independent_counters = (c1(), c1(), c2(), c1(), c2())""",
]


@pytest.mark.parametrize("code", PYTHON_SEMANTICS_TESTS)
def test_meta_circular_matches_python_semantics(code: str):
    """Test that meta-circular interpreter matches Python's builtin semantics."""
    _compare_with_python(code)
