import ast
import builtins
import linecache
import typing
from types import CodeType
from typing import Any

from RestrictedPython import (
    Eval,
    Guards,
    RestrictingNodeTransformer,
    compile_restricted,
    safe_globals,
)

from effectful.ops.syntax import ObjectInterpretation, defop, implements


@defop
def parse(source: str, filename: str) -> ast.Module:
    """
    Parse source text into an AST.

    source: The Python source code to parse.
    filename: The filename recorded in the resulting AST for tracebacks and tooling.

    Returns the parsed AST.
    """
    raise NotImplementedError(
        "An eval provider must be installed in order to parse code."
    )


@defop
def compile(module: ast.Module, filename: str) -> CodeType:
    """
    Compile an AST into a Python code object.

    module: The AST to compile (typically produced by parse()).
    filename: The filename recorded in the resulting code object (CodeType.co_filename), used in tracebacks and by inspect.getsource().

    Returns the compiled code object.
    """
    raise NotImplementedError(
        "An eval provider must be installed in order to compile code."
    )


@defop
def exec(
    bytecode: CodeType,
    env: dict[str, Any],
) -> None:
    """
    Execute a compiled code object.

    bytecode: A code object to execute (typically produced by compile()).
    env: The namespace mapping used during execution.
    """
    raise NotImplementedError(
        "An eval provider must be installed in order to execute code."
    )


class UnsafeEvalProvider(ObjectInterpretation):
    """UNSAFE provider that handles parse, comple and exec operations
    by shelling out to python *without* any further checks. Only use for testing."""

    @implements(parse)
    def parse(self, source: str, filename: str) -> ast.Module:
        # Cache source under `filename` so inspect.getsource() can retrieve it later.
        # inspect uses f.__code__.co_filename -> linecache.getlines(filename)
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(True),
            filename,
        )

        return ast.parse(source, filename=filename, mode="exec")

    @implements(compile)
    def compile(self, module: ast.AST, filename: str) -> CodeType:
        return builtins.compile(typing.cast(typing.Any, module), filename, "exec")

    @implements(exec)
    def exec(
        self,
        bytecode: CodeType,
        env: dict[str, Any],
    ) -> None:
        # Ensure builtins exist in the execution environment.
        env.setdefault("__builtins__", __builtins__)

        # Execute module-style so top-level defs land in `env`.
        builtins.exec(bytecode, env, env)


class RestrictedEvalProvider(ObjectInterpretation):
    """
    Safer provider using RestrictedPython.

    RestrictedPython is not a complete sandbox, but it enforces a restricted
    language subset and expects you to provide a constrained exec environment.

    policy : dict[str, Any], optional
        RestrictedPython compile_restricted policy for compilation
    """

    policy: type[RestrictingNodeTransformer] | None = None

    def __init__(
        self,
        *,
        policy: type[RestrictingNodeTransformer] | None = None,
    ):
        self.policy = policy

    @implements(parse)
    def parse(self, source: str, filename: str) -> ast.Module:
        # Keep inspect.getsource() working for dynamically-defined objects.
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(True),
            filename,
        )
        return ast.parse(source, filename=filename, mode="exec")

    @implements(compile)
    def compile(self, module: ast.Module, filename: str) -> CodeType:
        # RestrictedPython can compile from an AST directly.
        return compile_restricted(
            module,
            filename=filename,
            mode="exec",
            policy=self.policy or RestrictingNodeTransformer,
        )

    @implements(exec)
    def exec(
        self,
        bytecode: CodeType,
        env: dict[str, Any],
    ) -> None:
        # Build restricted globals from RestrictedPython's defaults
        rglobals: dict[str, Any] = safe_globals.copy()

        # Enable class definitions (required for Python 3)
        rglobals["__metaclass__"] = type
        rglobals["__name__"] = "restricted"

        # Layer `env` on top (without letting callers replace the restricted builtins).
        rglobals.update({k: v for k, v in env.items() if k != "__builtins__"})

        # Enable for loops and comprehensions
        rglobals["_getiter_"] = Eval.default_guarded_getiter
        # Enable sequence unpacking in comprehensions and for loops
        rglobals["_iter_unpack_sequence_"] = Guards.guarded_iter_unpack_sequence

        rglobals["getattr"] = Guards.safer_getattr
        rglobals["setattr"] = Guards.guarded_setattr
        rglobals["_write_"] = lambda x: x

        # Track keys before execution to identify new definitions
        keys_before = set(rglobals.keys())

        builtins.exec(bytecode, rglobals, rglobals)

        # Copy newly defined items back to env so caller can access them
        for key in rglobals:
            if key not in keys_before:
                env[key] = rglobals[key]
