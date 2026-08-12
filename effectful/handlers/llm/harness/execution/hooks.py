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
