"""An UNSAFE eval provider built on the interpreter's own builtins.

`BuiltinExecutor` implements the parse/compile/eval/exec operations by calling
`ast.parse`, `builtins.compile`, `builtins.eval` and `builtins.exec` directly, in
this process, *without* any further checks -- generated code gets the full
authority of the interpreter running the harness. Only use it for testing, or
where the code being run is trusted for some other reason;
`~effectful.handlers.llm.harness.execution.restricted.RestrictedPythonExecutor`
is the provider for anything else.

It runs whatever it is given: type checking is a separate handler
(`~effectful.handlers.llm.harness.validation.mypy.MypyTypeChecker` or
`~effectful.handlers.llm.harness.validation.ty.TyTypeChecker`), installed
alongside this one when generated code should be checked before it runs.
"""

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
    """Code you write runs as ordinary Python in this process, with nothing
    restricting it. The whole standard library is available, any installed
    third-party package is importable, and the filesystem, the network and the
    process itself are all reachable. If an import would work in a normal Python
    session, it works here.

    So write straightforward code and import what you need instead of working
    around a sandbox that is not there. The corresponding responsibility is
    yours: the same lack of restriction means a stray `open(..., "w")` or a
    `subprocess` call really does touch the machine. Do the work the request
    asks for and nothing else with side effects beyond it.
    """

    @implements(parse)
    def parse(self, source: str, filename: str) -> ast.Module:
        """Parse `source`, registering it under `filename` so it stays readable.

        Generated source has no file behind it, so `inspect.getsource` on a
        function defined here would otherwise fail. Seeding `linecache` under
        the same name the code object will carry (`inspect` goes from
        ``f.__code__.co_filename`` to ``linecache.getlines(filename)``) makes
        the synthesized code introspectable like any other -- which is what lets
        a traceback show real lines, and a later turn read back what it wrote.
        """
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
        """Compile `source` -- text or AST -- with `builtins.compile`.

        A straight pass-through: unlike
        `~effectful.handlers.llm.harness.execution.restricted.RestrictedPythonExecutor.compile`,
        there is no policy to apply and nothing to rewrite, so the flags the
        caller passes are the flags CPython sees.
        """
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
        """Evaluate `bytecode` for its value, discarding its binding effects.

        The evaluation happens in a *copy* of `env`, because the operation's
        contract is that `eval` yields a value and changes nothing: a walrus in
        the expression -- or the ``__builtins__`` entry seeded here, which the
        caller never asked for -- must not leak back into the caller's
        environment. `exec` is the operation that does bind.
        """
        g = dict(env)
        g.setdefault("__builtins__", __builtins__)
        return builtins.eval(bytecode, g, g)

    @implements(exec)
    def exec(
        self,
        bytecode: types.CodeType,
        env: dict[str, typing.Any],
    ) -> None:
        """Execute `bytecode` in `env`, keeping whatever it binds.

        Unlike `eval`, this runs against `env` itself, and passes it as both
        globals and locals so execution is module-style: a top-level ``def`` or
        assignment lands in `env` and is visible to the next statement executed
        there, which is what makes a sequence of snippets behave like a session
        rather than a series of unrelated fragments.
        """
        env.setdefault("__builtins__", __builtins__)
        builtins.exec(bytecode, env, env)
