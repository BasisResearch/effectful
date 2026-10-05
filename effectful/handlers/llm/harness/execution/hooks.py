"""The four operations an eval provider implements: `parse`, `compile`, `eval`, `exec`.

Every piece of model-authored Python the harness runs -- REPL snippets,
synthesized bodies and returned callables, tool-call expressions, and the
doctests of all of these -- goes through these operations rather than the
builtins, so one installed provider decides how all of it is parsed, compiled
and run. They have no default rule: each raises ``NotImplementedError`` until
:mod:`~effectful.handlers.llm.harness.execution.builtin` or :mod:`~effectful.handlers.llm.harness.execution.restricted` is installed.
Each operation's docstring states its signature contract; `compile` explains why
it alone mirrors its builtin.
"""

import ast
import types
import typing

from effectful.ops.types import Operation


@Operation.define
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


@Operation.define
def compile(
    source: str | ast.AST,
    filename: str,
    mode: str = "exec",
    flags: int = 0,
    dont_inherit: bool = False,
    optimize: int = -1,
) -> types.CodeType:
    """
    Compile source text or an AST into a Python code object.

    Takes `builtins.compile`'s signature, so it can stand in for the builtin
    wherever one is called positionally -- notably inside `doctest`'s runner, which
    `run_doctests` redirects here. Only ``mode`` differs, defaulting to ``"exec"``
    (the module compile that synthesis does) rather than being required.

    source: The source to compile: an AST (typically produced by parse()) or the
        source text of one.
    filename: The filename recorded in the resulting code object (CodeType.co_filename), used in tracebacks and by inspect.getsource().
    mode: ``"exec"``, ``"eval"`` or ``"single"``, as for `builtins.compile`.
    flags, dont_inherit, optimize: as for `builtins.compile`.

    Returns the compiled code object.
    """
    raise NotImplementedError(
        "An eval provider must be installed in order to compile code."
    )


@Operation.define
def eval(
    bytecode: types.CodeType,
    env: dict[str, typing.Any],
) -> typing.Any:
    """
    Evaluate a compiled expression code object and return its value.

    bytecode: A code object compiled in ``"eval"`` mode (typically produced by
        compile(..., mode="eval")).
    env: The namespace mapping used during evaluation.

    Returns the expression's value. Binding effects are discarded: unlike
    `exec`, ``env`` is not updated after evaluation -- the only construct that
    could bind a name from eval-mode code is a scope-escaping walrus, and
    callers that must not observe one reject it before compiling.

    Deliberately ``(bytecode, env)``, symmetric with the sibling `exec`
    operation, rather than `builtins.eval`'s ``(source, globals, locals)``.
    Nothing stands this operation in for the builtin (`compile` explains why it
    alone mirrors one), ``globals=None`` ("use the caller's frame") is meaningless
    as an effect operation, and accepting ``str`` source would collapse the
    parse -> compile -> eval separation the operations are built on.
    """
    raise NotImplementedError(
        "An eval provider must be installed in order to evaluate code."
    )


@Operation.define
def exec(
    bytecode: types.CodeType,
    env: dict[str, typing.Any],
) -> None:
    """
    Execute a compiled code object.

    bytecode: A code object to execute (typically produced by compile()).
    env: The namespace mapping used during execution.

    After ``exec(bytecode, env)`` returns, ``env`` reflects all top-level
    binding effects of the executed code (new names and rebindings alike).
    """
    raise NotImplementedError(
        "An eval provider must be installed in order to execute code."
    )
