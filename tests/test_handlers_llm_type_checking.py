"""Tests for effectful.handlers.llm.type_checking."""

import ast
import inspect
import textwrap
import types
import typing
from collections import ChainMap
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import pydantic
import pytest

from effectful.handlers.llm.evaluation import (
    collect_imports,
    collect_runtime_type_stubs,
    collect_variable_declarations,
    mypy_type_check,
    type_to_ast,
)
from effectful.internals.unification import nested_type
from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled


def get_context() -> Mapping[str, Any]:
    """Get the lexical context at the callsite.

    Returns a ChainMap containing locals and globals from the calling context.
    """
    frame = inspect.currentframe()
    assert frame is not None
    frame = frame.f_back
    assert frame is not None

    # Check if we're in a class definition by looking for __qualname__
    qualname = frame.f_locals.get("__qualname__")
    n_frames = 1
    if qualname is not None:
        name_components = qualname.split(".")
        for name in reversed(name_components):
            if name == "<locals>":
                break
            n_frames += 1

    contexts = []
    for offset in range(n_frames):
        assert frame is not None
        locals_proxy: types.MappingProxyType[str, Any] = types.MappingProxyType(
            frame.f_locals
        )
        globals_proxy: types.MappingProxyType[str, Any] = types.MappingProxyType(
            frame.f_globals
        )
        contexts.append(locals_proxy)
        frame = frame.f_back

    contexts.append(globals_proxy)
    context: Mapping[str, Any] = {
        k: v
        for context in contexts
        for k, v in context.items()
        if not (
            (
                isinstance(v, types.ModuleType)
                and v.__name__.startswith("tests.test_handlers_llm_type_checking")
            )
            or (
                hasattr(v, "__module__")
                and (
                    v.__module__.startswith("tests.test_handlers_llm_type_checking")
                    or v.__module__.startswith("_pytest")
                )
                and inspect.isclass(v)
                and k.startswith("Test")
            )
            or k == "self"
            or k == "__loader__"
        )
    }
    return context


class TestTypeToAstBasicTypes:
    """Test type_to_ast with basic Python types."""

    @pytest.mark.parametrize(
        "typ,expected",
        [
            (int, "int"),
            (str, "str"),
            (float, "float"),
            (bool, "bool"),
            (type(None), "None"),
        ],
    )
    def test_basic_types(self, typ, expected):
        result = type_to_ast(typ)
        assert ast.unparse(result) == expected


class TestTypeToAstTypingTypes:
    """Test type_to_ast with typing module types."""

    def test_typing_any(self):
        import typing

        result = type_to_ast(typing.Any)
        assert ast.unparse(result) == "typing.Any"


class TestCollectImports:
    """Test collect_imports function."""

    def test_collects_module_imports(self):
        """Test that modules in context are collected as imports."""
        import math
        import os

        ctx = {"math": math, "os": os, "x": 42}
        result = collect_imports(ctx)
        unparsed = [ast.unparse(stmt) for stmt in result]
        assert any("math" in s and "import" in s for s in unparsed)
        assert any("os" in s and "import" in s for s in unparsed)


class TestCollectImportsStress:
    """Stress test collect_imports with get_context: imports, aliases, external symbols."""

    def test_from_import_symbol_no_module_in_context(self):
        """Context has symbol (e.g. Any) but no module (typing); we still emit from-import."""
        from typing import Any  # noqa: F401 - captured by get_context

        ctx = get_context()
        result = collect_imports(ctx)
        unparsed = [ast.unparse(stmt) for stmt in result]
        assert unparsed[-1].startswith("from typing import") and "Any" in unparsed[-1]

    def test_plain_import_via_get_context(self):
        """Plain imports (import math) show up in import statements."""
        import math  # noqa: F401 - captured by get_context

        ctx = get_context()
        result = collect_imports(ctx)
        unparsed = [ast.unparse(stmt) for stmt in result]
        assert any(s.startswith("import ") and " math, " in s for s in unparsed)

    def test_import_alias_via_get_context(self):
        """Import aliases (import os as myos) show up as import os as myos."""
        import os as myos  # noqa: F401 - captured by get_context

        ctx = get_context()
        result = collect_imports(ctx)
        unparsed = [ast.unparse(stmt) for stmt in result]
        assert any("os as myos" in s for s in unparsed)

    def test_from_import_alias(self):
        """from typing import List as L: mapping has L, we emit from typing import List as L."""
        from typing import List as L  # noqa: F401, UP035 - captured by get_context

        ctx = get_context()
        assert "L" in ctx
        result = collect_imports(ctx)
        unparsed = [ast.unparse(stmt) for stmt in result]
        assert any("List as L" in s for s in unparsed)

    def test_mixed_imports_and_symbols(self):
        """Mixed: from typing import Any, import math, import collections as coll."""
        import collections as coll  # noqa: F401
        import math  # noqa: F401
        from typing import Any  # noqa: F401

        ctx = get_context()
        result = collect_imports(ctx)
        unparsed = [ast.unparse(stmt) for stmt in result]
        assert any(s.startswith("from") and "Any" in s for s in unparsed), unparsed
        assert any("math" in s and "import" in s for s in unparsed), unparsed
        assert any("collections" in s and "coll" in s for s in unparsed), unparsed


class TestCollectVariableDeclarations:
    """Test collect_variable_declarations end-to-end."""

    def test_basic_context(self):
        ctx = {"x": 42, "y": "hello"}
        result = collect_variable_declarations(ctx)
        # Should produce type-annotated variable declarations
        unparsed = [ast.unparse(stmt) for stmt in result]
        assert "x: int" in unparsed
        assert "y: str" in unparsed


class TestTypeToAstFunctionTypes:
    """Test type_to_ast with function types."""

    def test_function_type_becomes_callable(self):
        """types.FunctionType should become Callable (mypy-compatible)."""
        result = type_to_ast(types.FunctionType)
        assert ast.unparse(result) == "collections.abc.Callable"


class TestTypeToAstGenericTypes:
    """Test type_to_ast with generic types."""

    def test_callable_with_args(self):
        """Callable[[int], str] should be rendered correctly."""
        import collections.abc

        typ = collections.abc.Callable[[int], str]
        result = type_to_ast(typ)
        assert ast.unparse(result) == "collections.abc.Callable[[int], str]"

    def test_callable_varargs(self):
        """Callable[..., str] (varargs) should be rendered correctly."""
        import collections.abc

        typ = collections.abc.Callable[..., str]
        result = type_to_ast(typ)
        assert ast.unparse(result) == "collections.abc.Callable[..., str]"

    def test_callable_multiple_args(self):
        """Callable[[int, str], bool] should be rendered correctly."""
        import collections.abc

        typ = collections.abc.Callable[[int, str], bool]
        result = type_to_ast(typ)
        assert ast.unparse(result) == "collections.abc.Callable[[int, str], bool]"

    def test_callable_no_args(self):
        """Callable[[], int] (no args) should be rendered correctly."""
        import collections.abc

        typ = collections.abc.Callable[[], int]
        result = type_to_ast(typ)
        assert ast.unparse(result) == "collections.abc.Callable[[], int]"

    def test_mutable_sequence_int(self):
        """MutableSequence[int] from nested_type([1,2,3])."""
        typ = nested_type([1, 2, 3]).value
        result = type_to_ast(typ)
        assert ast.unparse(result) == "collections.abc.MutableSequence[int]"

    def test_mutable_mapping_str_int(self):
        """MutableMapping[str, int] from nested_type({'a': 1})."""
        typ = nested_type({"a": 1}).value
        result = type_to_ast(typ)
        assert ast.unparse(result) == "collections.abc.MutableMapping[str, int]"

    def test_mutable_set_int(self):
        """MutableSet[int] from nested_type({1, 2, 3})."""
        typ = nested_type({1, 2, 3}).value
        result = type_to_ast(typ)
        assert ast.unparse(result) == "collections.abc.MutableSet[int]"


class TestTypeToAstUnionTypes:
    """Test type_to_ast with union types."""

    def test_union_int_str(self):
        """int | str union type."""
        typ = int | str
        result = type_to_ast(typ)
        assert ast.unparse(result) == "int | str"

    def test_union_with_none(self):
        """int | None (Optional[int])."""
        typ = int | None
        result = type_to_ast(typ)
        assert ast.unparse(result) == "int | None"


class TestTypeToAstTypingAnnotations:
    """Test type_to_ast with common typing module annotations."""

    def test_optional_int(self):
        """typing.Optional[int] renders as typing.Union[int, None]."""
        import typing

        typ = typing.Optional[int]  # noqa: UP045 - intentionally testing old syntax
        result = type_to_ast(typ)
        assert ast.unparse(result) == "typing.Union[int, None]"

    def test_typing_dict(self):
        """typing.Dict[str, int]."""
        import typing

        typ = typing.Dict[str, int]  # noqa: UP006 - intentionally testing old syntax
        result = type_to_ast(typ)
        assert ast.unparse(result) == "dict[str, int]"

    def test_typing_list(self):
        """typing.List[int]."""
        import typing

        typ = typing.List[int]  # noqa: UP006 - intentionally testing old syntax
        result = type_to_ast(typ)
        assert ast.unparse(result) == "list[int]"

    def test_typing_set(self):
        """typing.Set[int]."""
        import typing

        typ = typing.Set[int]  # noqa: UP006 - intentionally testing old syntax
        result = type_to_ast(typ)
        assert ast.unparse(result) == "set[int]"

    def test_typing_mapping(self):
        """typing.Mapping[str, int]."""
        import typing

        typ = typing.Mapping[str, int]
        result = type_to_ast(typ)
        assert ast.unparse(result) == "collections.abc.Mapping[str, int]"

    def test_typing_sequence(self):
        """typing.Sequence[int]."""
        import typing

        typ = typing.Sequence[int]
        result = type_to_ast(typ)
        assert ast.unparse(result) == "collections.abc.Sequence[int]"


class TestTypeToAstAnnotated:
    """Test type_to_ast with typing.Annotated (strips to inner type for typecheck stubs)."""

    def test_annotated_strips_to_inner_type(self):
        """Annotated[int, "meta"] renders as int."""
        typ = Annotated[int, "meta"]
        result = type_to_ast(typ)
        assert ast.unparse(result) == "int"

    def test_annotated_with_multiple_metadata(self):
        """Annotated[str, "a", "b"] strips to str."""
        typ = Annotated[str, "a", "b"]
        result = type_to_ast(typ)
        assert ast.unparse(result) == "str"

    def test_annotated_generic(self):
        """Annotated[list[int], "tag"] strips to list[int]."""
        typ = Annotated[list[int], "tag"]
        result = type_to_ast(typ)
        assert ast.unparse(result) == "list[int]"


class TestTypeToAstBuiltinAnnotations:
    """Test type_to_ast with builtin generic annotations (Python 3.9+)."""

    def test_builtin_list(self):
        """list[int] builtin annotation."""
        typ = list[int]
        result = type_to_ast(typ)
        assert ast.unparse(result) == "list[int]"

    def test_builtin_dict(self):
        """dict[str, int] builtin annotation."""
        typ = dict[str, int]
        result = type_to_ast(typ)
        assert ast.unparse(result) == "dict[str, int]"

    def test_builtin_set(self):
        """set[int] builtin annotation."""
        typ = set[int]
        result = type_to_ast(typ)
        assert ast.unparse(result) == "set[int]"

    def test_builtin_tuple(self):
        """tuple[int, str] builtin annotation."""
        typ = tuple[int, str]
        result = type_to_ast(typ)
        assert ast.unparse(result) == "tuple[int, str]"


class TestGetContextStress:
    """Stress test with get_context - exercises real lexical contexts."""

    def test_get_context_with_local_variables(self):
        """Test that we can handle variables from get_context."""
        x = 42  # noqa: F841 - intentionally captured by get_context
        y = "hello"  # noqa: F841 - intentionally captured by get_context
        z = [1, 2, 3]  # noqa: F841 - intentionally captured by get_context
        ctx = get_context()
        result = collect_variable_declarations(ctx)
        unparsed = [ast.unparse(stmt) for stmt in result]
        print("\n".join(unparsed))
        assert "x: int" in unparsed
        assert "y: str" in unparsed

    def test_functions_with_annotations(self):
        """Test that annotated functions use Callable[[...], ...] from __annotations__."""

        def annotated_func(x: int) -> str:
            return str(x)

        ctx = get_context()
        result = collect_variable_declarations(ctx)
        unparsed = [ast.unparse(stmt) for stmt in result]
        # annotated_func should have Callable[[int], str]
        assert "annotated_func: collections.abc.Callable[[int], str]" in unparsed

    def test_functions_without_annotations(self):
        """Test that unannotated functions get Callable type."""

        def unannotated_func(x):
            return x

        ctx = get_context()
        result = collect_variable_declarations(ctx)
        unparsed = [ast.unparse(stmt) for stmt in result]
        # unannotated_func should get Callable type
        assert "unannotated_func: collections.abc.Callable" in unparsed

    def test_operations_in_context(self):
        """Test that Operations get correct type annotations."""

        @defop
        def my_op(x: int) -> str:  # noqa: F841 - captured by get_context
            raise NotHandled

        ctx = get_context()
        result = collect_variable_declarations(ctx)
        unparsed = [ast.unparse(stmt) for stmt in result]
        # Operation should have exact annotation
        assert "my_op: effectful.ops.types.Operation[[int], str]" in unparsed

    def test_custom_exception_in_context(self):
        """Test that custom exceptions get correct type annotations without __main__."""

        class MyError(Exception):  # noqa: F841 - captured by get_context
            pass

        err = MyError("test")  # noqa: F841 - captured by get_context
        ctx = get_context()
        result = collect_variable_declarations(ctx)
        unparsed = [ast.unparse(stmt) for stmt in result]
        print("\n".join(unparsed))
        # The exception instance should have the exception class type without __main__
        assert "err: MyError" in unparsed


class TestTypeToAstOperations:
    """Test type_to_ast with effectful Operation types."""

    def test_operation_type(self):
        """Operation type from nested_type."""

        @defop
        def my_op(x: int) -> str:
            raise NotHandled

        typ = nested_type(my_op).value
        result = type_to_ast(typ)
        unparsed = ast.unparse(result)
        # Check exact structure
        assert unparsed == "effectful.ops.types.Operation[[int], str]"


class TestTypeToAstPolymorphicTypes:
    """Test type_to_ast with polymorphic (generic) types."""

    def test_callable_with_typevar(self):
        """Callable with TypeVar: Callable[[T], T]."""
        import typing

        T = typing.TypeVar("T")  # noqa: PLC0132
        typ = typing.Callable[[T], T]
        result = type_to_ast(typ)
        unparsed = ast.unparse(result)
        # Check exact structure: collections.abc.Callable[[T], T]
        assert unparsed == "collections.abc.Callable[[T], T]"

    def test_callable_with_bounded_typevar(self):
        """Callable with bounded TypeVar: Callable[[T], T] where T: int."""
        import typing

        T_bounded = typing.TypeVar("T_bounded", bound=int)  # noqa: PLC0132
        typ = typing.Callable[[T_bounded], T_bounded]
        result = type_to_ast(typ)
        unparsed = ast.unparse(result)
        # Bounded TypeVar still uses its name
        assert unparsed == "collections.abc.Callable[[T_bounded], T_bounded]"

    def test_operation_with_typevar(self):
        """Operation with TypeVar: Operation[[T], T]."""

        @defop
        def identity[T](x: T) -> T:
            raise NotHandled

        typ = nested_type(identity).value
        result = type_to_ast(typ)
        unparsed = ast.unparse(result)
        # Check exact structure
        assert unparsed == "effectful.ops.types.Operation[[T], T]"

    def test_operation_with_bounded_typevar_312_syntax(self):
        """Operation with bounded TypeVar using Python 3.12+ [T: int] syntax."""

        @defop
        def bounded_op[T: int](x: T) -> T:
            raise NotHandled

        typ = nested_type(bounded_op).value
        result = type_to_ast(typ)
        unparsed = ast.unparse(result)
        # Check that the bounded TypeVar is preserved
        assert unparsed == "effectful.ops.types.Operation[[T], T]"

    def test_generic_class(self):
        """Generic class: list[T]."""
        import typing

        T = typing.TypeVar("T")  # noqa: PLC0132
        typ = list[T]
        result = type_to_ast(typ)
        unparsed = ast.unparse(result)
        # Check exact structure
        assert unparsed == "list[T]"

    def test_callable_with_paramspec(self):
        """Callable with ParamSpec: Callable[P, int]."""
        import typing

        P = typing.ParamSpec("P")  # noqa: PLC0132
        typ = typing.Callable[P, int]
        result = type_to_ast(typ)
        unparsed = ast.unparse(result)
        # ParamSpec is rendered by name
        assert unparsed == "collections.abc.Callable[P, int]"

    def test_tuple_with_typevartuple(self):
        """tuple with TypeVarTuple: tuple[*Ts]."""

        Ts = typing.TypeVarTuple("Ts")  # noqa: PLC0132
        typ = tuple[*Ts]
        result = type_to_ast(typ)
        unparsed = ast.unparse(result)
        # TypeVarTuple with Unpack
        assert "tuple" in unparsed
        assert "Ts" in unparsed


class TestCollectRuntimeTypeStubs:
    """Test collect_runtime_type_stubs for generating class stubs."""

    def test_exception_subclass_stub(self):
        """Test that runtime exception classes get proper stubs with inheritance."""

        class MyError(Exception):
            pass

        ctx = {"MyError": MyError}
        result = collect_runtime_type_stubs(ctx)
        unparsed = "\n".join(ast.unparse(stmt) for stmt in result)
        assert "class MyError(Exception):" in unparsed

    def test_class_with_callable_method(self):
        """Test that classes with callable methods get typed stubs."""

        class MyClass:
            def my_method(self, x: int) -> str:
                return str(x)

        ctx = {"MyClass": MyClass}
        result = collect_runtime_type_stubs(ctx)
        unparsed = "\n".join(ast.unparse(stmt) for stmt in result)
        assert "class MyClass:" in unparsed
        assert "def my_method(self, x: int) -> str:" in unparsed

    def test_class_with_typed_attribute(self):
        """Test that classes with __annotations__ get typed stubs."""

        class MyClass:
            x: int
            y: str

        ctx = {"MyClass": MyClass}
        result = collect_runtime_type_stubs(ctx)
        unparsed = "\n".join(ast.unparse(stmt) for stmt in result)
        assert "class MyClass:" in unparsed
        assert "x: int" in unparsed
        assert "y: str" in unparsed

    def test_exception_chain_inheritance(self):
        """Test exception with multiple levels of inheritance."""

        class BaseError(Exception):
            pass

        class SpecificError(BaseError):
            pass

        ctx = {"BaseError": BaseError, "SpecificError": SpecificError}
        result = collect_runtime_type_stubs(ctx)
        unparsed = "\n".join(ast.unparse(stmt) for stmt in result)
        assert "class BaseError(Exception):" in unparsed
        assert "class SpecificError(BaseError):" in unparsed

    def test_exception_with_attributes(self):
        """Test exception class with typed attributes."""

        class ValidationError(Exception):
            field: str
            message: str

        ctx = {"ValidationError": ValidationError}
        result = collect_runtime_type_stubs(ctx)
        unparsed = "\n".join(ast.unparse(stmt) for stmt in result)
        assert "class ValidationError(Exception):" in unparsed
        assert "field: str" in unparsed
        assert "message: str" in unparsed

    def test_exception_with_init(self):
        """Test exception class with __init__ method."""

        class CustomError(Exception):
            def __init__(self, code: int, message: str) -> None:
                super().__init__(message)
                self.code = code

        ctx = {"CustomError": CustomError}
        result = collect_runtime_type_stubs(ctx)
        unparsed = "\n".join(ast.unparse(stmt) for stmt in result)
        assert "class CustomError(Exception):" in unparsed
        assert "def __init__(self, code: int, message: str):" in unparsed

    def test_dataclass_like_with_fields(self):
        """Test class with multiple typed fields like a dataclass."""

        class Person:
            name: str
            age: int
            email: str | None

        ctx = {"Person": Person}
        result = collect_runtime_type_stubs(ctx)
        unparsed = "\n".join(ast.unparse(stmt) for stmt in result)
        assert "class Person:" in unparsed
        assert "name: str" in unparsed
        assert "age: int" in unparsed
        assert "email: str | None" in unparsed

    def test_method_with_generic_types(self):
        """Test method with generic type annotations."""

        class Container:
            def get_items(self) -> list[str]:
                return []

            def set_mapping(self, data: dict[str, int]) -> None:
                pass

        ctx = {"Container": Container}
        result = collect_runtime_type_stubs(ctx)
        unparsed = "\n".join(ast.unparse(stmt) for stmt in result)
        assert "def get_items(self) -> list[str]:" in unparsed
        assert "def set_mapping(self, data: dict[str, int]) -> None:" in unparsed

    def test_field_with_generic_types(self):
        """Test fields with generic type annotations."""

        class DataStore:
            items: list[int]
            cache: dict[str, Any]

        ctx = {"DataStore": DataStore}
        result = collect_runtime_type_stubs(ctx)
        unparsed = "\n".join(ast.unparse(stmt) for stmt in result)
        assert "items: list[int]" in unparsed
        assert "cache: dict[str, typing.Any]" in unparsed

    def test_generic_superclass(self):
        """Test class inheriting from generic type."""

        class StringList(list[str]):
            pass

        ctx = {"StringList": StringList}
        result = collect_runtime_type_stubs(ctx)
        unparsed = "\n".join(ast.unparse(stmt) for stmt in result)
        assert "class StringList(list[str]):" in unparsed

    def test_generic_class_with_type_params(self):
        """Test generic class with its own type parameters: class Foo[T](list[T])."""

        class Container[T](list[T]):
            def get_first(self) -> T:
                return self[0]

        ctx = {"Container": Container}
        result = collect_runtime_type_stubs(ctx)
        unparsed = "\n".join(ast.unparse(stmt) for stmt in result)
        print(unparsed)
        # Should include generic base class
        assert "class Container(list[T], typing.Generic[T]):" in unparsed, unparsed
        assert "def get_first(self) -> T:" in unparsed

    def test_multiple_inheritance(self):
        """Test class with multiple base classes."""

        class Mixin:
            pass

        class Base:
            pass

        class Combined(Base, Mixin):
            pass

        ctx = {"Base": Base, "Mixin": Mixin, "Combined": Combined}
        result = collect_runtime_type_stubs(ctx)
        unparsed = "\n".join(ast.unparse(stmt) for stmt in result)
        assert "class Base:" in unparsed
        assert "class Mixin:" in unparsed
        assert "class Combined(Base, Mixin):" in unparsed

    def test_method_with_optional_params(self):
        """Test method with optional/default parameters."""

        class Config:
            def setup(self, name: str, debug: bool = False) -> None:
                pass

        ctx = {"Config": Config}
        result = collect_runtime_type_stubs(ctx)
        unparsed = "\n".join(ast.unparse(stmt) for stmt in result)
        assert "def setup(self, name: str, debug: bool) -> None:" in unparsed

    def test_class_with_classmethod_and_staticmethod(self):
        """Test class with classmethod and staticmethod (should not appear as regular methods)."""

        class Factory:
            @classmethod
            def create(cls) -> "Factory":
                return cls()

            @staticmethod
            def validate(x: int) -> bool:
                return x > 0

            def instance_method(self) -> str:
                return "hello"

        ctx = {"Factory": Factory}
        result = collect_runtime_type_stubs(ctx)
        unparsed = "\n".join(ast.unparse(stmt) for stmt in result)
        # instance_method should be present
        assert "def instance_method(self) -> str:" in unparsed

    def test_method_with_callable_param(self):
        """Test method with Callable parameter type."""

        class Handler:
            def register(self, callback: Callable[[int], str]) -> None:
                pass

        ctx = {"Handler": Handler}
        result = collect_runtime_type_stubs(ctx)
        unparsed = "\n".join(ast.unparse(stmt) for stmt in result)
        assert (
            "def register(self, callback: collections.abc.Callable[[int], str]) -> None:"
            in unparsed
        )

    def test_nested_generic_types(self):
        """Test deeply nested generic types."""

        class NestedData:
            matrix: list[list[int]]
            lookup: dict[str, list[tuple[int, str]]]

        ctx = {"NestedData": NestedData}
        result = collect_runtime_type_stubs(ctx)
        unparsed = "\n".join(ast.unparse(stmt) for stmt in result)
        assert "matrix: list[list[int]]" in unparsed
        assert "lookup: dict[str, list[tuple[int, str]]]" in unparsed

    def test_union_types_in_method(self):
        """Test method with union type annotations."""

        class Parser:
            def parse(self, data: str | bytes) -> int | None:
                return None

        ctx = {"Parser": Parser}
        result = collect_runtime_type_stubs(ctx)
        unparsed = "\n".join(ast.unparse(stmt) for stmt in result)
        assert "def parse(self, data: str | bytes) -> int | None:" in unparsed

    def test_dataclasses(self):
        """Test method with dataclasses annotations."""

        @dataclass
        class Point:
            x: int
            y: int

        ctx = {"Point": Point}
        result = collect_runtime_type_stubs(ctx)
        unparsed = "\n".join(ast.unparse(stmt) for stmt in result)
        assert "class Point:\n    x: int\n    y: int" in unparsed

    def test_typed_dicts(self):
        """Test method subclassing typed dict."""

        class TypedPoint(TypedDict):
            x: int
            y: int

        ctx = {"TypedPoint": TypedPoint}
        result = collect_runtime_type_stubs(ctx)
        unparsed = "\n".join(ast.unparse(stmt) for stmt in result)
        assert "class TypedPoint:\n    x: int\n    y: int" in unparsed

    def test_pydantic_base_models(self):
        """Test method subclassing pydantic base models."""

        class BasePoint(pydantic.BaseModel):
            x: int
            y: int

        ctx = {"Point": BasePoint}
        result = collect_runtime_type_stubs(ctx)
        unparsed = "\n".join(ast.unparse(stmt) for stmt in result)
        assert (
            "class Point(pydantic.main.BaseModel):\n    x: int\n    y: int" in unparsed
        )


class TestTypeAliases:
    """Test type_to_ast with type aliases."""

    def test_simple_type_alias(self):
        """Test simple type alias."""

        type IntList = list[int]
        result = type_to_ast(IntList)
        unparsed = ast.unparse(result)
        assert unparsed == "IntList"

    def test_generic_type_alias(self):
        """Test generic type alias with TypeVar using Python 3.12+ syntax."""
        type MyList[T] = list[T]  # noqa: PLC0132 - T is defined by the type syntax
        result = type_to_ast(MyList)
        unparsed = ast.unparse(result)
        assert unparsed == "MyList"


class TestMypyTypeCheckE2E:
    """End-to-end stress tests for mypy_type_check with get_context and ast.parse. Never empty context."""

    def test_simple_function_with_get_context(self):
        """One function; ctx from get_context(); typecheck passes."""
        _ = 1  # noqa: F841 - in context
        source = "def f(x: int, s: str) -> bool:\n    return len(s) > x"
        module = ast.parse(source)
        mypy_type_check(module, get_context(), [int, str], bool)

    def test_simple_function_no_params_with_get_context(self):
        """Function with no params, returns int; get_context()."""
        _ = 1  # noqa: F841
        source = "def g() -> int:\n    return 42"
        module = ast.parse(source)
        mypy_type_check(module, get_context(), None, int)

    def test_module_with_multiple_statements_then_function(self):
        """Module has assignments then the function we check; get_context()."""
        _ = 1  # noqa: F841
        source = """
a = 1
b = "x"
def h(n: int) -> str:
    return str(n)
"""
        module = ast.parse(source)
        mypy_type_check(module, get_context(), [int], str)

    def test_with_get_context_and_typed_values(self):
        """Use get_context(); function signature matches expected; passes."""
        x = 42  # noqa: F841
        s = "hello"  # noqa: F841

        ctx = get_context()
        source = "def add_one(n: int) -> int:\n    return n + 1"
        module = ast.parse(source)
        mypy_type_check(module, ctx, [int], int)

    def test_nested_function_module_last_is_outer(self):
        """Module body has def outer with nested def inner; we check outer; get_context()."""
        _ = 1  # noqa: F841
        source = """
def outer(x: int) -> str:
    def inner(y: int) -> int:
        return y + 1
    return str(inner(x))
"""
        module = ast.parse(source)
        mypy_type_check(module, get_context(), [int], str)

    def test_local_class_in_context_and_module(self):
        """Runtime-only class in ctx; module has class + function using it; stubs generated."""

        class LocalKlass:
            def method(self) -> int:
                return 0

        source = """
def use_it(obj: LocalKlass) -> int:
    return obj.method()
"""
        module = ast.parse(source)
        ctx = ChainMap({"LocalKlass": LocalKlass}, get_context())
        mypy_type_check(module, ctx, [LocalKlass], int)

    def test_decorated_function(self):
        """Function has a decorator; last stmt is still the function; get_context()."""
        _ = 1  # noqa: F841
        source = """
def dec(f):
    return f
@dec
def decorated(x: int) -> bool:
    return x > 0
"""
        module = ast.parse(source)
        mypy_type_check(module, get_context(), [int], bool)

    def test_module_uses_typing_from_context(self):
        """Context provides List from get_context; module uses list[int]; typecheck passes."""

        _ = list  # noqa: F841 - in context
        source = """
def sum_list(nums: list[int]) -> int:
    return sum(nums)
"""
        module = ast.parse(source)
        mypy_type_check(module, get_context(), [list], int)

    def test_typecheck_passes_with_fully_annotated_function(self):
        """Typechecking passes when the module has full type annotations (params + return) matching expected."""
        _ = 1  # noqa: F841
        source = """
def add(x: int, y: int) -> int:
    return x + y
"""
        module = ast.parse(source)
        mypy_type_check(module, get_context(), [int, int], int)

    def test_typecheck_passes_with_annotated_generics(self):
        """Typechecking passes when the function uses generic annotations (list, dict) in params."""
        _ = list  # noqa: F841
        _ = dict  # noqa: F841
        source = """
def process(nums: list[int], mapping: dict[str, int]) -> int:
    return len(nums) + len(mapping)
"""
        module = ast.parse(source)
        mypy_type_check(module, get_context(), [list, dict], int)

    def test_typecheck_passes_when_module_uses_typing_annotated(self):
        """Typechecking passes when the function uses typing.Annotated in params and return."""
        from typing import Annotated  # noqa: F401 - in context for get_context

        source = """
def f(x: Annotated[int, "positive"], y: Annotated[str, "name"]) -> Annotated[bool, "ok"]:
    return len(y) > x
"""
        module = ast.parse(source)
        mypy_type_check(module, get_context(), [int, str], bool)

    def test_typecheck_passes_with_expected_annotated(self):
        """Typechecking passes when expected params/return use Annotated (stripped for stub)."""
        _ = 1  # noqa: F841
        source = "def g(x: int) -> int:\n    return x"
        module = ast.parse(source)
        mypy_type_check(
            module,
            get_context(),
            [Annotated[int, "value"]],
            Annotated[int, "result"],
        )

    def test_typecheck_symbol_in_annotated_metadata_does_not_crash_mypy(self):
        """A symbol (type/class) in Annotated metadata is resolved from context and does not crash mypy."""

        class Tag:
            """Metadata marker class in context."""

        source = """
def f(x: Annotated[int, Tag]) -> int:
    return x
"""
        module = ast.parse(source)
        ctx = get_context()
        mypy_type_check(module, ctx, [int], int)


class TestMypyTypeCheckFailures:
    """Failure cases: mypy_type_check must raise TypeError with mypy report. All use get_context()."""

    def test_wrong_return_type_raises(self):
        """Function returns int but expected return is str; mypy fails."""
        _ = 1  # noqa: F841
        source = "def f(x: int) -> str:\n    return x"
        module = ast.parse(source)
        with pytest.raises(TypeError) as exc_info:
            mypy_type_check(module, get_context(), [int], str)
        assert (
            "mypy" in str(exc_info.value).lower()
            or "error" in str(exc_info.value).lower()
        )

    def test_wrong_param_count_raises(self):
        """Expected (int, str) but function takes (int); fails."""
        _ = 1  # noqa: F841
        source = "def g(x: int) -> bool:\n    return True"
        module = ast.parse(source)
        with pytest.raises(TypeError) as exc_info:
            mypy_type_check(module, get_context(), [int, str], bool)
        assert exc_info.type is TypeError

    def test_incompatible_param_type_raises(self):
        """Function annotates param as str, we expect int; mismatch."""
        _ = 1  # noqa: F841
        source = "def h(s: str) -> int:\n    return len(s)"
        module = ast.parse(source)
        with pytest.raises(TypeError):
            mypy_type_check(module, get_context(), [int], int)

    def test_empty_module_raises(self):
        """Empty module.body raises TypeError before mypy."""
        _ = 1  # noqa: F841
        module = ast.Module(body=[], type_ignores=[])
        with pytest.raises(TypeError, match="empty"):
            mypy_type_check(module, get_context(), None, int)

    def test_last_statement_not_function_raises(self):
        """Last stmt is an expression, not a function def."""
        _ = 1  # noqa: F841
        source = "x = 1"
        module = ast.parse(source)
        with pytest.raises(TypeError, match="function"):
            mypy_type_check(module, get_context(), [], int)

    def test_assign_then_expr_no_function_raises(self):
        """Module has only assignments/expressions; no function def."""
        _ = 1  # noqa: F841
        source = "a = 1\na + 1"
        module = ast.parse(source)
        with pytest.raises(TypeError, match="function"):
            mypy_type_check(module, get_context(), [], int)

    def test_failure_dataclass_wrong_return(self):
        """Dataclass in context; function returns wrong type; get_context()."""

        @dataclass
        class Box:
            value: int

        source = """
def bad(b: Box) -> Box:
    return b.value
"""
        module = ast.parse(source)
        ctx = ChainMap({"Box": Box}, get_context())
        with pytest.raises(TypeError):
            mypy_type_check(module, ctx, [Box], Box)

    def test_failure_dataclass_wrong_param_type(self):
        """Dataclass; expected param int, function takes Box; get_context()."""

        @dataclass
        class Box:
            value: int

        source = "def use(b: Box) -> int:\n    return b.value"
        module = ast.parse(source)
        ctx = ChainMap({"Box": Box}, get_context())
        with pytest.raises(TypeError):
            mypy_type_check(module, ctx, [int], int)

    def test_failure_plain_class_wrong_return(self):
        """Plain class in context; function returns wrong type; get_context()."""

        class Node:
            def __init__(self, x: int) -> None:
                self.x = x

        source = "def make() -> Node:\n    return 42"
        module = ast.parse(source)
        ctx = ChainMap({"Node": Node}, get_context())
        with pytest.raises(TypeError):
            mypy_type_check(module, ctx, [], Node)

    def test_failure_plain_class_param_mismatch(self):
        """Plain class; expected (Node, Node), function takes (Node); get_context()."""

        class Node:
            pass

        source = "def add(a: Node) -> Node:\n    return a"
        module = ast.parse(source)
        ctx = ChainMap({"Node": Node}, get_context())
        with pytest.raises(TypeError):
            mypy_type_check(module, ctx, [Node, Node], Node)

    def test_failure_local_class_wrong_return(self):
        """Runtime local class; function annotated to return class but returns int; get_context()."""

        class LocalKlass:
            pass

        source = """
class LocalKlass:
    pass
def f() -> LocalKlass:
    return 1
"""
        module = ast.parse(source)
        ctx = ChainMap({"LocalKlass": LocalKlass}, get_context())
        with pytest.raises(TypeError):
            mypy_type_check(module, ctx, [], LocalKlass)

    def test_failure_runtime_local_class_param_wrong_type(self):
        """Runtime local class; we expect (int,), function takes (LocalKlass,); get_context()."""

        class LocalKlass:
            pass

        source = """
class LocalKlass:
    pass
def g(obj: LocalKlass) -> int:
    return 0
"""
        module = ast.parse(source)
        ctx = ChainMap({"LocalKlass": LocalKlass}, get_context())
        with pytest.raises(TypeError):
            mypy_type_check(module, ctx, [int], int)

    def test_failure_nested_function_outer_wrong_return(self):
        """Nested function: outer declared -> str but returns int; get_context()."""
        _ = 1  # noqa: F841
        source = """
def outer(x: int) -> str:
    def inner() -> int:
        return 1
    return inner()
"""
        module = ast.parse(source)
        with pytest.raises(TypeError):
            mypy_type_check(module, get_context(), [int], str)

    def test_failure_nested_function_expected_wrong_param_count(self):
        """Nested: we expect (int, str), outer takes (int); get_context()."""
        _ = 1  # noqa: F841
        source = """
def outer(x: int) -> bool:
    def inner() -> bool:
        return True
    return inner()
"""
        module = ast.parse(source)
        with pytest.raises(TypeError):
            mypy_type_check(module, get_context(), [int, str], bool)

    def test_failure_typeddict_wrong_return(self):
        """TypedDict in context; function returns wrong type; get_context()."""

        class Point(TypedDict):
            x: int
            y: int

        source = "def origin() -> Point:\n    return 0"
        module = ast.parse(source)
        ctx = ChainMap({"Point": Point}, get_context())
        with pytest.raises(TypeError):
            mypy_type_check(module, ctx, [], Point)

    def test_failure_typeddict_param_mismatch(self):
        """TypedDict; we expect (Point, int), function takes (Point,); get_context()."""

        class Point(TypedDict):
            x: int
            y: int

        source = "def get_x(p: Point) -> int:\n    return p['x']"
        module = ast.parse(source)
        ctx = ChainMap({"Point": Point}, get_context())
        with pytest.raises(TypeError):
            mypy_type_check(module, ctx, [Point, int], int)

    def test_failure_pydantic_model_wrong_return(self):
        """Pydantic BaseModel in context; function returns wrong type; get_context()."""

        class Item(pydantic.BaseModel):
            name: str

        source = "def make() -> Item:\n    return 1"
        module = ast.parse(source)
        ctx = ChainMap({"Item": Item}, get_context())
        with pytest.raises(TypeError):
            mypy_type_check(module, ctx, [], Item)

    def test_failure_pydantic_model_param_wrong(self):
        """Pydantic model; we expect (str,), function takes (Item,); get_context()."""

        class Item(pydantic.BaseModel):
            name: str

        source = "def get_name(i: Item) -> str:\n    return i.name"
        module = ast.parse(source)
        ctx = ChainMap({"Item": Item}, get_context())
        with pytest.raises(TypeError):
            mypy_type_check(module, ctx, [str], str)

    def test_failure_custom_class_wrong_return(self):
        """Custom class; function declares return type but returns other; get_context()."""

        class Custom:
            def run(self) -> int:
                return 0

        source = "def get_custom() -> Custom:\n    return 1"
        module = ast.parse(source)
        ctx = ChainMap({"Custom": Custom}, get_context())
        with pytest.raises(TypeError):
            mypy_type_check(module, ctx, [], Custom)

    def test_failure_custom_class_param_expected_mismatch(self):
        """Custom class; expected (Custom, str), function (Custom,) -> str; get_context()."""

        class Custom:
            pass

        source = "def greet(c: Custom) -> str:\n    return 'hi'"
        module = ast.parse(source)
        ctx = ChainMap({"Custom": Custom}, get_context())
        with pytest.raises(TypeError):
            mypy_type_check(module, ctx, [Custom, str], str)

    def test_failure_callable_wrong_return(self):
        """Callable type: we expect Callable[[int], str], function returns Callable[[int], int]; get_context()."""
        _ = 1  # noqa: F841
        source = "def f() -> Callable[[int], int]:\n    return lambda x: x"
        module = ast.parse(source)
        with pytest.raises(TypeError):
            mypy_type_check(module, get_context(), [], Callable[[int], str])

    def test_failure_optional_return_wrong(self):
        """Optional/union: function returns int, we expect str | None; get_context()."""
        _ = 1  # noqa: F841
        source = "def f(x: int) -> int:\n    return x"
        module = ast.parse(source)
        with pytest.raises(TypeError):
            mypy_type_check(module, get_context(), [int], str | None)

    def test_failure_exception_class_wrong_base(self):
        """Custom exception class; function returns int but expected return MyErr; get_context()."""

        class MyErr(Exception):
            pass

        source = "def raise_it() -> MyErr:\n    return 1"
        module = ast.parse(source)
        ctx = ChainMap({"MyErr": MyErr}, get_context())
        with pytest.raises(TypeError):
            mypy_type_check(module, ctx, [], MyErr)


class TestMypyTypeCheckNameCollision:
    """Tests that mypy_type_check renames synthesized functions whose names
    collide with variable declarations or class stubs from the context."""

    def test_single_function_collides_with_variable(self):
        """Function name matches a variable in context; should still pass type-check."""
        count_char = lambda s: s.count("a")  # noqa: E731, F841

        source = textwrap.dedent("""\
            def count_char(s: str) -> int:
                return s.count('a')
        """)
        module = ast.parse(source)
        ctx = get_context()
        # Should NOT raise — the collision is handled by renaming
        mypy_type_check(module, ctx, [str], int)

    def test_colliding_function_still_detects_type_errors(self):
        """Even after renaming, real type errors are still caught."""
        count_char = lambda s: s.count("a")  # noqa: E731, F841

        source = textwrap.dedent("""\
            def count_char(s: str) -> int:
                return s  # wrong return type
        """)
        module = ast.parse(source)
        ctx = get_context()
        with pytest.raises(TypeError):
            mypy_type_check(module, ctx, [str], int)

    def test_no_collision_passes_normally(self):
        """No name collision — normal type-check should work as before."""
        x = 42  # noqa: F841

        source = textwrap.dedent("""\
            def some_unique_func(s: str) -> int:
                return len(s)
        """)
        module = ast.parse(source)
        ctx = get_context()
        mypy_type_check(module, ctx, [str], int)

    def test_multiple_functions_one_collides(self):
        """Module has helper + main function; only main collides with context."""
        process = "some_value"  # noqa: F841

        source = textwrap.dedent("""\
            def helper(x: int) -> str:
                return str(x)
            def process(items: list[int]) -> list[str]:
                return [helper(i) for i in items]
        """)
        module = ast.parse(source)
        ctx = get_context()
        mypy_type_check(module, ctx, [list[int]], list[str])

    def test_multiple_functions_both_collide(self):
        """Both helper and main function names collide with context variables."""
        helper = lambda: None  # noqa: E731, F841
        compute = 123  # noqa: F841

        source = textwrap.dedent("""\
            def helper(x: int) -> str:
                return str(x)
            def compute(n: int) -> str:
                return helper(n)
        """)
        module = ast.parse(source)
        ctx = get_context()
        mypy_type_check(module, ctx, [int], str)

    def test_collision_with_class_stub(self):
        """Function name collides with a runtime class stub in context."""

        class MyModel:
            value: int

        # Also define a function named MyModel in synthesized code
        source = textwrap.dedent("""\
            def MyModel(x: int) -> int:
                return x * 2
        """)
        module = ast.parse(source)
        ctx = ChainMap({"MyModel": MyModel}, get_context())
        mypy_type_check(module, ctx, [int], int)

    def test_collision_does_not_mutate_original_ast(self):
        """Renaming should not modify the original module AST."""
        count_char = lambda s: s.count("a")  # noqa: E731, F841

        source = textwrap.dedent("""\
            def count_char(s: str) -> int:
                return s.count('a')
        """)
        module = ast.parse(source)
        original_name = module.body[-1].name

        ctx = get_context()
        mypy_type_check(module, ctx, [str], int)

        # Original AST must be untouched
        assert module.body[-1].name == original_name

    def test_helper_reference_updated_after_rename(self):
        """When a helper function is renamed, calls to it inside other
        functions are also updated so mypy still sees valid code."""
        validate = True  # noqa: F841 — collides with helper name

        source = textwrap.dedent("""\
            def validate(x: int) -> bool:
                return x > 0
            def run(x: int) -> bool:
                return validate(x)
        """)
        module = ast.parse(source)
        ctx = get_context()
        mypy_type_check(module, ctx, [int], bool)
