"""Tests for effectful.handlers.llm.type_checking (toplevel typecheck_source interface)."""

import ast
import typing

import pytest

from effectful.handlers.llm import type_checking

# --- typecheck_source: success cases ---


class TestTypecheckSourceSuccess:
    def test_simple_function_int_to_int(self) -> None:
        mod = ast.parse("def f(x: int) -> int:\n    return x")
        type_checking.typecheck_source(mod, {}, [int], int)

    def test_empty_ctx(self) -> None:
        mod = ast.parse("def identity(x: int) -> int:\n    return x")
        type_checking.typecheck_source(mod, {}, [int], int)

    def test_none_ctx(self) -> None:
        mod = ast.parse("def g() -> str:\n    return 'hi'")
        type_checking.typecheck_source(
            mod, typing.cast(typing.Mapping[str, typing.Any], None), [], str
        )

    def test_ctx_with_value_stub(self) -> None:
        mod = ast.parse("def use_x(x: int) -> int:\n    return x")
        type_checking.typecheck_source(mod, {"x": 42}, [int], int)

    def test_ctx_multiple_stubs(self) -> None:
        mod = ast.parse("def add(a: int, b: int) -> int:\n    return a + b")
        type_checking.typecheck_source(
            mod,
            {"a": 1, "b": 2},
            [int, int],
            int,
        )

    def test_ellipsis_params_expected_return_only(self) -> None:
        mod = ast.parse("def get_one() -> int:\n    return 1")
        type_checking.typecheck_source(mod, {}, None, int)

    def test_function_with_list_param(self) -> None:
        mod = ast.parse("def sum_list(x: list[int]) -> int:\n    return sum(x)")
        type_checking.typecheck_source(mod, {}, [list[int]], int)

    def test_function_with_tuple_return(self) -> None:
        mod = ast.parse("def pair() -> tuple[int, str]:\n    return (1, 'a')")
        type_checking.typecheck_source(mod, {}, [], tuple[int, str])

    def test_sync_function_return_none(self) -> None:
        mod = ast.parse("def noop() -> None:\n    pass")
        type_checking.typecheck_source(mod, {}, [], type(None))  # type: ignore[arg-type]

    def test_module_import_in_ctx(self) -> None:
        import math

        mod = ast.parse("def use_pi(pi: float) -> float:\n    return pi")
        type_checking.typecheck_source(
            mod,
            {"math": math},
            [float],
            float,
        )


# --- typecheck_source: source accesses ctx (globals and imports) ---


class TestTypecheckSourceAccessesCtx:
    """Tests that synthesized code can access ctx as globals/imports and mypy validates usage."""

    def test_function_uses_global_from_ctx(self) -> None:
        # Function body references a name from ctx; prelude has scale: int
        mod = ast.parse("def scale_by(x: int) -> int:\n    return x * scale")
        type_checking.typecheck_source(mod, {"scale": 2}, [int], int)

    def test_function_uses_multiple_globals_from_ctx(self) -> None:
        mod = ast.parse("def compute(x: int) -> int:\n    return x * factor + offset")
        type_checking.typecheck_source(
            mod,
            {"factor": 10, "offset": 1},
            [int],
            int,
        )

    def test_function_uses_module_from_ctx_math(self) -> None:
        import math

        mod = ast.parse(
            "def sqrt_sum(a: float, b: float) -> float:\n    return math.sqrt(a * a + b * b)"
        )
        type_checking.typecheck_source(
            mod,
            {"math": math},
            [float, float],
            float,
        )

    def test_function_uses_module_from_ctx_math_pi(self):
        import math

        mod = ast.parse(
            "def circle_area(r: float) -> float:\n    return math.pi * r * r"
        )
        type_checking.typecheck_source(
            mod,
            {"math": math},
            [float],
            float,
        )

    def test_function_imports_effectful_from_ctx(self):
        import effectful

        mod = ast.parse(
            "def get_effectful_version() -> str:\n    return getattr(effectful, '__version__', '')"
        )
        type_checking.typecheck_source(
            mod,
            {"effectful": effectful},
            [],
            str,
        )

    def test_function_uses_effectful_submodule_from_ctx(self):
        import effectful.handlers.llm as llm_mod

        mod = ast.parse(
            "def has_type_checking() -> bool:\n    return hasattr(llm_mod, 'type_checking')"
        )
        type_checking.typecheck_source(
            mod,
            {"llm_mod": llm_mod},
            [],
            bool,
        )

    def test_function_uses_imported_function_from_ctx(self):
        import math

        mod = ast.parse("def apply_sqrt(x: float) -> float:\n    return math.sqrt(x)")
        type_checking.typecheck_source(
            mod,
            {"math": math},
            [float],
            float,
        )

    def test_global_type_mismatch_raises(self):
        # Function uses global as int but we stub it as str -> mypy should fail
        mod = ast.parse("def use_global(x: int) -> int:\n    return x + global_val")
        with pytest.raises(ValueError, match="Type check failed"):
            type_checking.typecheck_source(
                mod,
                {"global_val": "not an int"},
                [int],
                int,
            )


# --- typecheck_source: module with multiple functions ---


class TestTypecheckSourceModuleWithMultipleFunctions:
    """Only the last function is checked; it may call helpers defined above."""

    def test_last_function_calls_helper(self):
        src = """
def double(x: int) -> int:
    return x * 2

def main(n: int) -> int:
    return double(n)
"""
        mod = ast.parse(src)
        type_checking.typecheck_source(mod, {}, [int], int)

    def test_last_function_uses_several_helpers(self):
        src = """
def add_one(x: int) -> int:
    return x + 1

def double(x: int) -> int:
    return x * 2

def pipeline(x: int) -> int:
    return double(add_one(x))
"""
        mod = ast.parse(src)
        type_checking.typecheck_source(mod, {}, [int], int)

    def test_helpers_may_have_different_signatures(self):
        src = """
def greet(name: str) -> str:
    return "hello, " + name

def length(s: str) -> int:
    return len(s)

def main(s: str) -> int:
    return length(greet(s))
"""
        mod = ast.parse(src)
        type_checking.typecheck_source(mod, {}, [str], int)

    def test_last_function_uses_helper_and_global_from_ctx(self):
        src = """
def scale(x: int) -> int:
    return x * factor

def main(x: int) -> int:
    return scale(x) + offset
"""
        mod = ast.parse(src)
        type_checking.typecheck_source(
            mod,
            {"factor": 2, "offset": 1},
            [int],
            int,
        )

    def test_multiple_functions_last_has_optional_return(self) -> None:
        src = """
def safe_head(x: list[int]) -> typing.Optional[int]:
    return x[0] if x else None

def main(x: list[int]) -> typing.Optional[int]:
    return safe_head(x)
"""
        mod = ast.parse(src)
        type_checking.typecheck_source(mod, {}, [list[int]], typing.Optional[int])  # type: ignore[arg-type]  # noqa: UP045


# --- typecheck_source: Optional and annotation variation (qualname robustness) ---


class TestTypecheckSourceOptionalAndAnnotationVariations:
    """Optional matching and annotation style variations: if the code would work at runtime, typecheck passes."""

    def test_optional_param_and_return_typing_optional(self) -> None:
        # Source uses typing.Optional[int]; we expect Optional[int]. Prelude has "import typing".
        mod = ast.parse(
            "def identity(x: typing.Optional[int]) -> typing.Optional[int]:\n    return x"
        )
        type_checking.typecheck_source(
            mod,
            {},
            [typing.Optional[int]],  # type: ignore[list-item]  # noqa: UP045
            typing.Optional[int],  # type: ignore[arg-type]  # noqa: UP045
        )

    def test_optional_param_and_return_pipe_syntax(self) -> None:
        # Source uses int | None; we expect Optional[int] (emitted as int | None in postlude).
        mod = ast.parse("def identity(x: int | None) -> int | None:\n    return x")
        type_checking.typecheck_source(
            mod,
            {},
            [typing.Optional[int]],  # type: ignore[list-item]  # noqa: UP045
            typing.Optional[int],  # type: ignore[arg-type]  # noqa: UP045
        )

    def test_optional_return_only(self) -> None:
        mod = ast.parse("def maybe_one() -> typing.Optional[int]:\n    return 1")
        type_checking.typecheck_source(mod, {}, [], typing.Optional[int])  # type: ignore[arg-type]  # noqa: UP045

    def test_optional_global_from_ctx(self) -> None:
        # Global in ctx is Optional[int]; function uses it and returns int.
        mod = ast.parse(
            "def with_default(x: int) -> int:\n    return (default if default is not None else 0) + x"
        )
        type_checking.typecheck_source(
            mod,
            {"default": 10},
            [int],
            int,
        )

    def test_list_int_builtin_vs_typing_list(self) -> None:
        # Source uses list[int]; we pass list[int]. Equivalent.
        mod = ast.parse("def first(x: list[int]) -> int:\n    return x[0]")
        type_checking.typecheck_source(mod, {}, [list[int]], int)

    def test_list_int_source_typing_list(self):
        # Source uses typing.List[int]; we expect list[int]. Same type for mypy.
        mod = ast.parse("def first(x: typing.List[int]) -> int:\n    return x[0]")
        type_checking.typecheck_source(mod, {}, [list[int]], int)

    def test_dict_annotation_variation(self):
        # Source uses typing.Dict[str, int]; we expect dict[str, int].
        mod = ast.parse(
            "def get_key(d: typing.Dict[str, int], k: str) -> int:\n    return d[k]"
        )
        type_checking.typecheck_source(mod, {}, [dict[str, int], str], int)

    def test_tuple_annotation_variation(self):
        # Source uses typing.Tuple[int, str]; we expect tuple[int, str].
        mod = ast.parse("def pair() -> typing.Tuple[int, str]:\n    return (1, 'a')")
        type_checking.typecheck_source(mod, {}, [], tuple[int, str])

    def test_mixed_annotations_builtin_and_typing(self):
        # Mix: param with list[int], return with typing.Optional[list[int]].
        mod = ast.parse(
            "def head(x: list[int]) -> typing.Optional[int]:\n    return x[0] if x else None"
        )
        type_checking.typecheck_source(mod, {}, [list[int]], typing.Optional[int])  # type: ignore  # noqa: UP045


# --- typecheck_source: signature mismatch causes typecheck to fail ---


class TestTypecheckSourceSignatureMismatchFails:
    """When the last function's signature does not match the expected Callable type, typecheck_source raises."""

    def test_return_type_mismatch_fails(self):
        # Expected Callable[[], int], function has -> str
        mod = ast.parse("def f() -> str:\n    return 'x'")
        with pytest.raises(ValueError, match="Type check failed"):
            type_checking.typecheck_source(mod, {}, [], int)

    def test_return_type_mismatch_multiple_functions_last_wrong(self):
        # Only the last function is checked; it has -> str but we expect int
        src = """
def helper() -> int:
    return 1

def main() -> str:
    return "wrong"
"""
        mod = ast.parse(src)
        with pytest.raises(ValueError, match="Type check failed"):
            type_checking.typecheck_source(mod, {}, [], int)

    def test_param_count_mismatch_fails(self):
        # Expected Callable[[int], int], function has (x: int, y: int) -> int
        mod = ast.parse("def f(x: int, y: int) -> int:\n    return x + y")
        with pytest.raises(ValueError, match="Type check failed"):
            type_checking.typecheck_source(mod, {}, [int], int)

    def test_param_count_mismatch_too_few_fails(self):
        # Expected Callable[[int, int], int], function has (x: int) -> int
        mod = ast.parse("def f(x: int) -> int:\n    return x")
        with pytest.raises(ValueError, match="Type check failed"):
            type_checking.typecheck_source(mod, {}, [int, int], int)

    def test_param_type_mismatch_fails(self):
        # Expected Callable[[int], int], function has (x: str) -> int
        mod = ast.parse("def f(x: str) -> int:\n    return len(x)")
        with pytest.raises(ValueError, match="Type check failed"):
            type_checking.typecheck_source(mod, {}, [int], int)

    def test_zero_arg_wrong_return_fails(self):
        # Zero-arg function: _return_check line forces return type; str != int
        mod = ast.parse("def get() -> str:\n    return 'nope'")
        with pytest.raises(ValueError, match="Type check failed"):
            type_checking.typecheck_source(mod, {}, [], int)

    def test_expected_optional_return_but_got_str_fails(self):
        # Expected Callable[[], Optional[int]], function has -> str
        mod = ast.parse("def f() -> str:\n    return 'x'")
        with pytest.raises(ValueError, match="Type check failed"):
            type_checking.typecheck_source(mod, {}, [], typing.Optional[int])  # type: ignore  # noqa: UP045


# --- typecheck_source: failure cases (wrong types) ---


class TestTypecheckSourceFailureWrongTypes:
    def test_wrong_return_type_raises(self):
        mod = ast.parse("def f() -> str:\n    return 'x'")
        with pytest.raises(ValueError, match="Type check failed"):
            type_checking.typecheck_source(mod, {}, [], int)

    def test_wrong_param_count_raises(self):
        mod = ast.parse("def f(x: int, y: int) -> int:\n    return x + y")
        with pytest.raises(ValueError, match="Type check failed"):
            type_checking.typecheck_source(mod, {}, [int], int)

    def test_wrong_param_type_raises(self):
        mod = ast.parse("def f(x: str) -> int:\n    return len(x)")
        with pytest.raises(ValueError, match="Type check failed"):
            type_checking.typecheck_source(mod, {}, [int], int)

    def test_return_type_incompatible_with_body_raises(self):
        mod = ast.parse("def f() -> int:\n    return 'wrong'")
        with pytest.raises(ValueError, match="Type check failed"):
            type_checking.typecheck_source(mod, {}, [], int)

    def test_return_type_mismatch_raises(self):
        mod = ast.parse("def f() -> str:\n    return 42")
        with pytest.raises(ValueError, match="Type check failed"):
            type_checking.typecheck_source(mod, {}, [], str)


# --- typecheck_source: invalid AST / preconditions ---


class TestTypecheckSourceInvalidInput:
    def test_not_module_raises(self):
        node = ast.FunctionDef(
            "f",
            ast.arguments([], None, [], [], None, []),
            [ast.Return(ast.Constant(1))],
            [],
        )
        with pytest.raises(ValueError, match="Module AST"):
            type_checking.typecheck_source(node, {}, [], int)

    def test_empty_body_raises(self):
        mod = ast.Module(body=[], type_ignores=[])
        with pytest.raises(ValueError, match="at least one statement"):
            type_checking.typecheck_source(mod, {}, [], int)

    def test_last_statement_not_function_raises(self):
        mod = ast.parse("def f() -> int:\n    return 1\nx = 2")
        with pytest.raises(ValueError, match="last statement to be a function"):
            type_checking.typecheck_source(mod, {}, [], int)

    def test_multiple_statements_uses_last_as_function(self):
        mod = ast.parse("a = 1\ndef f() -> int:\n    return 0")
        type_checking.typecheck_source(mod, {}, [], int)


# --- typecheck_source: ctx with type aliases (PEP 695) ---


@pytest.mark.skipif(
    not hasattr(typing, "TypeAliasType"),
    reason="PEP 695 type alias requires Python 3.12+",
)
class TestTypecheckSourceCtxTypeAlias:
    def test_ctx_value_type_alias_emits_definition(self):
        type MyInt = int
        mod = ast.parse("def use_my(x: MyInt) -> MyInt:\n    return x")
        type_checking.typecheck_source(mod, {"MyInt": MyInt}, [int], int)

    def test_expected_return_type_alias(self):
        type Id = int
        mod = ast.parse("def one() -> int:\n    return 1")
        type_checking.typecheck_source(mod, {}, [], Id)


# --- typecheck_source: edge cases ---


class TestTypecheckSourceEdgeCases:
    def test_ctx_value_inference_skipped_on_error(self):
        mod = ast.parse("def f() -> int:\n    return 1")
        type_checking.typecheck_source(mod, {"weird": object()}, [], int)

    def test_expected_params_empty_list(self):
        mod = ast.parse("def f() -> float:\n    return 3.14")
        type_checking.typecheck_source(mod, {}, [], float)

    def test_function_name_preserved_in_postlude(self):
        mod = ast.parse("def my_custom_name(x: int) -> int:\n    return x")
        type_checking.typecheck_source(mod, {}, [int], int)


# --- typecheck_source: report contains mypy output ---


class TestTypecheckSourceReportContent:
    def test_value_error_contains_mypy_output(self):
        mod = ast.parse("def f() -> str:\n    return 123")
        with pytest.raises(ValueError) as exc_info:
            type_checking.typecheck_source(mod, {}, [], str)
        msg = str(exc_info.value)
        assert "Type check failed" in msg
        assert "return" in msg or "int" in msg or "str" in msg or "error" in msg.lower()
