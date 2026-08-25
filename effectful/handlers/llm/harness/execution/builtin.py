import ast
import builtins
import linecache
import types
import typing

from effectful.handlers.llm.harness.execution.hooks import (
    compile,
    eval,
    exec,
    parse,
)
from effectful.handlers.llm.harness.hooks import PromptInjectingInterpretation
from effectful.ops.syntax import implements


class BuiltinExecutor(PromptInjectingInterpretation):
    """UNSAFE provider that handles the parse, compile and exec operations with
    the interpreter's own builtins, in this process, *without* any further
    checks. Only use for testing.

    Runs whatever it is given: type checking is a separate handler
    (`~effectful.handlers.llm.harness.validation.mypy.MypyTypeChecker` or
    `~effectful.handlers.llm.harness.validation.ty.TyTypeChecker`), installed
    alongside this one when generated code should be checked before it runs."""

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
    def compile(
        self,
        source: str | ast.AST,
        filename: str,
        mode: str = "exec",
        flags: int = 0,
        dont_inherit: bool = False,
        optimize: int = -1,
    ) -> types.CodeType:
        return builtins.compile(
            typing.cast(typing.Any, source),
            filename,
            mode,
            flags,
            dont_inherit,
            optimize,
        )

    @implements(eval)
    def eval(
        self,
        bytecode: types.CodeType,
        env: dict[str, typing.Any],
    ) -> typing.Any:
        # Evaluate in a copy (with builtins ensured): the op's contract is that
        # binding effects are discarded, so a walrus -- or the builtins seeded
        # here -- must not leak into the caller's env.
        g = dict(env)
        g.setdefault("__builtins__", __builtins__)
        return builtins.eval(bytecode, g, g)

    @implements(exec)
    def exec(
        self,
        bytecode: types.CodeType,
        env: dict[str, typing.Any],
    ) -> None:
        # Ensure builtins exist in the execution environment.
        env.setdefault("__builtins__", __builtins__)

        # Execute module-style so top-level defs land in `env`.
        builtins.exec(bytecode, env, env)
