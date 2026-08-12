import ast
import code
import codeop
import collections
import collections.abc
import contextlib
import functools
import inspect
import io
import linecache
import types
import typing
import uuid

import pydantic

import effectful.handlers.llm.harness.execution.hooks
import effectful.handlers.llm.harness.validation.hooks
from effectful.handlers.llm.harness.execution.hooks import compile, exec, parse
from effectful.handlers.llm.harness.hooks import (
    AssistantResult,
    Message,
    call_assistant,
    call_system,
)
from effectful.handlers.llm.harness.serialization import (
    _TYPE_CHECK_ANCHOR_KEY,
    PromptSection,
    TypeToPydanticType,
    to_content_blocks,
)
from effectful.handlers.llm.harness.synthesis.function import (
    SplicedRegion,
    _def_nodes,
    _recover_skill_def,
)
from effectful.handlers.llm.types import Skill, Tool
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Operation


class _OpCommandCompiler(codeop.CommandCompiler):
    """A `codeop.CommandCompiler` that routes compilation through the
    `parse`/`compile` effect operations (so the installed eval provider owns it
    and `parse` populates `linecache`), replacing the native single-mode
    compiler that `code.InteractiveInterpreter` installs.
    """

    def __call__(
        self, source: str, filename: str = "<input>", symbol: str = "single"
    ) -> types.CodeType:
        # `runsource` passes symbol="single"; we ignore it and compile in the
        # exec mode the ops produce, so a complete multi-statement block runs in
        # one shot.  Incomplete/invalid input raises SyntaxError, which
        # `runsource` routes to `showsyntaxerror` (we do not buffer partial input
        # -- there is no line-at-a-time protocol).
        return compile(parse(source, filename), filename)


class ReplSession(code.InteractiveInterpreter):
    """A persistent, output-capturing Python session seeded from a lexical
    context.

    `exec_code(source)` runs a pre-compiled code object in `self.locals` through
    the `exec` effect operation.  Both bindings and captured stdout/stderr
    persist across calls -- variables, imports and definitions accumulate exactly
    like a REPL -- and the session (with its buffer) is discarded as a whole when
    it goes out of scope.  Each call returns only the output it produced; a
    snippet that raises has its traceback appended to that output rather than
    propagating -- mirroring `code.InteractiveInterpreter`, only `SystemExit`
    propagates -- so failures are surfaced as text.  There is no bare-expression
    auto-echo, so use `print()` to surface values.

    Compilation -- and therefore syntax checking -- happens earlier, at the
    `Encodable[CodeType]` boundary; this session only executes.
    """

    locals: dict[str, typing.Any]

    # The session's captured output, accumulated across calls and exposed for
    # introspection.  stdout (`print` output) and stderr (writes plus tracebacks)
    # are kept separate; `exec_code` returns each call's slice of both.
    stdout: io.StringIO
    stderr: io.StringIO

    def __init__(self, env: collections.abc.MutableMapping[str, typing.Any]):
        super().__init__(dict(env))
        # Route `runsource`'s compilation through the `parse`/`compile` ops too, so
        # it stays consistent with our `runcode` (which execs through the `exec`
        # op) rather than the native single-mode compiler the base installed.
        self.compile = _OpCommandCompiler()
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self._prior_snippets: list[str] = []

    @property
    def prior_snippets(self) -> list[str]:
        """Sources of the actual error-free executed snippets, in order -- the type-check
        context the `Encodable[CodeType]` decoder splices before the current snippet."""
        return self._prior_snippets

    def runcode(self, code: types.CodeType) -> None:
        # Mirrors `InteractiveInterpreter.runcode` exactly; the only difference
        # is that `exec` here is the effect operation, so execution routes
        # through the installed eval provider.  `showtraceback` reports failures
        # via `self.write`, which `exec_code` has redirected into `self.stderr`.
        try:
            exec(code, self.locals)
        except SystemExit:
            raise
        except:
            self.showtraceback()

    def exec_code(self, code: types.CodeType) -> str:
        """Run Python in a persistent, stateful session and return its output.

        This is a long-lived REPL, not a one-shot sandbox: every call runs in the
        SAME namespace, so names you bind in one call stay available in later
        calls within the same task.  Imports, function/class definitions and
        variable assignments all accumulate during the session of this skill.
        The namespace starts seeded with the in-scope variables of the surrounding context, which you may read and
        rebind.

        Output: returns this call's output -- its stdout (what `print` wrote)
        followed by its stderr (which includes the traceback if the code raised).
        There is NO automatic echoing of results -- a bare expression on its own
        line (e.g. `1 + 1`) displays nothing, so call `print(...)` for anything
        you want to see.  A snippet that raises has its traceback returned and the
        session survives, so you can read the error and continue in the next call
        (only `SystemExit` aborts).

        Provide `code` as a string of Python source.  It must be a complete,
        compilable snippet -- incomplete or invalid source is rejected before it
        runs.
        """
        out_start = self.stdout.tell()
        err_start = self.stderr.tell()
        # Record this snippet's source so the *next* snippet's decode-time type check can
        # splice the accumulated session code into the Skill body. The type check itself
        # lives in the `Encodable[CodeType]` decoder (as it does for synthesized Callables),
        # not here -- this session only runs code.
        self._prior_snippets.append("".join(linecache.getlines(code.co_filename)))
        with (
            contextlib.redirect_stdout(self.stdout),
            contextlib.redirect_stderr(self.stderr),
        ):
            self.runcode(code)
        return self.stdout.getvalue()[out_start:] + self.stderr.getvalue()[err_start:]


_CODE_FILENAME_PREFIX = "<exec_code-"


def _scan_non_nestable(generated: ast.Module) -> None:
    """Reject constructs legal at module level but illegal once nested in a function.

    ``from ... import *`` and ``from __future__ import ...`` are both ``SyntaxError``s
    inside a function body, but mypy *accepts* a nested star import silently, so the
    splice would slip an illegal construct past the type check and fail later at
    ``compile``/``exec``. Detect them explicitly and raise before splicing. Raises
    ``ValueError`` (this is rejecting invalid generated *source*, not signaling a type
    error), so a decoder can catch it alongside ``SyntaxError`` without swallowing a real
    ``TypeError`` from a broken provider.
    """
    if not generated.body:
        raise ValueError("generated code has empty or trivial body AST")
    for stmt in generated.body:
        if isinstance(stmt, ast.ImportFrom):
            if stmt.module == "__future__":
                raise ValueError(
                    "generated code uses `from __future__ import ...`, which is "
                    "illegal once spliced into a function body"
                )
            if any(alias.name == "*" for alias in stmt.names):
                raise ValueError(
                    "generated code uses a star import (`from ... import *`), which "
                    "is illegal once spliced into a function body"
                )


def _splice_snippet(
    generated: ast.Module,
    module_ast: ast.Module,
    skill_def: ast.FunctionDef | ast.AsyncFunctionDef,
) -> SplicedRegion:
    """Splice REPL code -- ``generated`` -- into the anchor Skill's body, in its
    real module source, and return the modified source with the ``[lo, hi]`` line
    span of the spliced statements.

    ``generated`` is the cumulative session code (any already-run snippets followed
    by the current one; the caller prepends them). It becomes the Skill function's
    body at its real (possibly nested) position, so the Skill's parameters and
    enclosing scope -- i.e. the session's seed env -- are in scope and each statement
    sees the ones before it (they are function locals). No ``return`` is appended;
    the REPL code doesn't produce the Skill's declared type, and that contract is
    waived by ``lenient`` type checking. The whole spliced body is reported, but the
    already-run snippets are type-clean (they passed this same check when *they* were
    the current one), so only the new statements can raise.

    Example. For the Skill ::

        @Skill.define
        def analyze(data: list[int]) -> str:
            '''Analyze {data}.'''

    a ``generated`` module of accumulated session statements ::

        total = sum(data)
        print(total / len(data))

    becomes the Skill's body ::

        @Skill.define
        def analyze(data: list[int]) -> str:
            total = sum(data)
            print(total / len(data))

    so each statement sees the Skill's ``data`` and the earlier statements'
    bindings (here ``total``).

    Returns ``None`` when ``generated`` has no statements to check, or when the
    Skill's source can't be recovered -- a Skill defined at a REPL, in a notebook, or
    via ``exec()`` is sourceless, so we skip the check and run the code unchecked, exactly
    as ``splice_into_source`` does for a sourceless Callable anchor. Raises ``RuntimeError``
    only on source *drift* (source recovered but the def no longer sits where it was
    compiled from), which ``_recover_skill_def`` surfaces.
    """
    skill_def.body = list(generated.body)

    # `skill_def` is still a node in `module_ast` (only its body changed), so its
    # walk-order index is stable across the unparse round-trip.
    def_index = _def_nodes(module_ast).index(skill_def)
    checked_source = ast.unparse(ast.fix_missing_locations(module_ast))
    spliced = _def_nodes(ast.parse(checked_source))[def_index]
    lo = spliced.body[0].lineno
    hi = spliced.body[-1].end_lineno or lo
    return checked_source, lo, hi


@TypeToPydanticType.register(types.CodeType)
def _pydantic_type_code(ty):
    """Encode a `types.CodeType` as a JSON string of Python source.

    This is the internal `Encodable` implementation for code objects -- the
    public type is `types.CodeType`, with no separate model (analogous to
    `_ComplexModel`).  Decoding compiles the source through the `parse`/`compile`
    effect operations under a unique per-snippet filename, so invalid source is
    rejected here rather than at run time and the snippet's source lands in
    `linecache` (keeping each snippet's tracebacks resolvable).  A decoded value
    is therefore a ready-to-run code object; re-encoding recovers its source from
    `linecache`, which carries everything the source string did.
    """

    def validate(
        value: types.CodeType | str, info: pydantic.ValidationInfo
    ) -> types.CodeType:
        if isinstance(value, types.CodeType):
            return value
        if not isinstance(value, str):
            raise ValueError(
                f"expected Python source as a string, got {type(value).__name__}"
            )

        ctx = info.context or {}
        anchor = ctx.get(_TYPE_CHECK_ANCHOR_KEY)

        filename = f"{_CODE_FILENAME_PREFIX}{uuid.uuid4()}>"
        module = effectful.handlers.llm.harness.execution.hooks.parse(value, filename)

        # Reject `__future__`/star imports: both are `SyntaxError` once nested in a
        # function body, so such a snippet can't be spliced into the Skill for
        # type checking.
        _scan_non_nestable(module)

        # Type-check the snippet in its execution context, exactly as a synthesized
        # `Callable` is (see `_pydantic_callable`): when the enclosing Skill is the
        # type-check anchor in the decode context, splice the accumulated REPL session
        # (`PythonRepl.repl_history` returns the prior snippets of the session in scope)
        # plus this snippet into the Skill body and check it. A type error raises here
        # -> the tool-call decode fails -> `RetryLLMHandler` retries, so ill-typed code
        # never reaches `runcode`.
        if anchor is not None and _recover_skill_def(anchor) is not None:
            # Prepend the already-run (type-clean) session snippets so their bindings
            # resolve; `value` is the current snippet. The whole cumulative body is
            # spliced and checked.
            anchor_asts = _recover_skill_def(anchor)
            assert anchor_asts is not None
            module_ast, skill_def = anchor_asts
            prior = StatefulReplSynthesizer.repl_history()
            prior_src = "".join(s if s.endswith("\n") else s + "\n" for s in prior)
            session = ast.parse(prior_src + value)
            checked = _splice_snippet(session, module_ast, skill_def)
            effectful.handlers.llm.harness.validation.hooks.type_check(
                *checked, lenient=True
            )

        return effectful.handlers.llm.harness.execution.hooks.compile(module, filename)

    return typing.Annotated[
        ty,
        pydantic.PlainValidator(validate),
        pydantic.PlainSerializer(
            lambda value: "".join(linecache.getlines(value.co_filename))
        ),
        pydantic.WithJsonSchema({"type": "string"}),
    ]


class StatefulReplSynthesizer(ObjectInterpretation):
    """You may run arbitrary Python code in a persistent session. The code is
    executed in the context of this Skill's lexical scope (see the *Lexical
    scope* table for the available names and their types). The session persists
    across turns, so you may define variables, functions, and classes that are
    used in later turns. The return value of the code is returned to you as the
    result of the tool call.

    Use the REPL only when running code actually helps — computing or verifying
    a result, exploring data, or calling a tool. If you can answer directly, just
    answer; do not route a plain text answer through `print(...)`.
    """

    # Off by default; install it where the LLM should be able to run code whose
    # state (variables, imports, definitions) survives across tool calls within a
    # single Skill invocation.  The docstring above is model-facing: it is the
    # `Harness` section this handler adds to the system prompt (see
    # `_call_system`), so implementation notes belong in comments like this one.
    #
    # Scoping mirrors how `__history__` is managed for Skill calls: `_apply`
    # handles `Skill.__apply__` to introduce fresh session-bound handlers
    # (`exec_code`, `read_lexical_variable`, `repl_history`) for the duration of
    # the call, and `_call_assistant` injects an `exec_code` Tool routed to that
    # session.  The session is therefore introduced and eliminated by its own
    # handler, bounded to the Skill call by construction -- there is no global
    # registry of sessions, and nested Skill calls get their own isolated ones.
    #
    # The session is seeded from the Skill's lexical context and routes
    # execution through the `parse`/`compile`/`exec` effect operations, so it
    # works under any installed eval provider (`BuiltinExecutor` or
    # `RestrictedPythonExecutor`).

    @typing.final
    @Tool.define
    @classmethod
    @functools.wraps(ReplSession.exec_code)
    def exec_code(cls, code: types.CodeType) -> str:
        raise NotImplementedError("No handler")

    @typing.final
    @Tool.define
    @classmethod
    def read_lexical_variable(cls, name: str) -> typing.Any:
        """
        Read the value of lexical variable ``name`` into the LLM context.
        """
        raise NotImplementedError("No handler")

    @typing.final
    @Operation.define
    @classmethod
    def repl_history(cls) -> list[str]:
        """This REPL session's error-free executed snippets, in order.

        Empty by default: unlike the tool operations above, this one is asked for
        by a *decoder* (`Encodable[CodeType]`, to type-check a snippet against the
        session it will run in), which can be reached with no REPL in scope at all
        -- decoding a code object outside a managed `StatefulReplSynthesizer` call. "No session"
        is a meaningful answer there (no prior snippets), not a missing handler.
        """
        return []

    @typing.final
    @Operation.define
    @classmethod
    def repl_env(cls) -> dict[str, typing.Any]:
        """The REPL session's current namespace, as a flat dict of name -> value"""
        return {}

    @implements(call_system)
    def _call_system(
        self, harness_prompt: PromptSection, agent_prompt: PromptSection
    ) -> typing.Any:
        return fwd(
            PromptSection(
                type="prompt_section",
                title=harness_prompt["title"],
                content=[
                    *harness_prompt["content"],
                    PromptSection(
                        type="prompt_section",
                        title=type(self).__name__,
                        content=to_content_blocks(inspect.getdoc(type(self)) or ""),
                    ),
                ],
            ),
            agent_prompt,
        )

    @implements(Skill.__apply__)
    def _apply[**P, T](
        self, skill: Skill[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        bound_args = skill.__signature__.bind(*args, **kwargs)
        bound_args.apply_defaults()
        env = collections.ChainMap(bound_args.arguments, skill.__context__)
        session = ReplSession(env=env)
        with handler(
            {
                self.exec_code: session.exec_code,
                self.read_lexical_variable: env.get,
                self.repl_history: lambda: session.prior_snippets,
                self.repl_env: lambda: session.locals,
            }
        ):
            return fwd()

    @implements(call_assistant)
    def _call_assistant[T](
        self,
        messages: collections.abc.Sequence[Message],
        response_type: type[T],
        env: collections.abc.Mapping[str, typing.Any],
        tools: collections.abc.Set[Tool] = frozenset(),
    ) -> AssistantResult[T]:
        return fwd(
            messages,
            response_type,
            {**env, **self.repl_env()},
            tools | {self.exec_code, self.read_lexical_variable},
        )
