"""Run model-authored Python in a REPL for one Skill turn.

:class:`StatefulReplSynthesizer` offers the model an ``exec_code`` Tool. Its
session starts with the Skill's arguments and lexical scope. Imports and
bindings from one snippet remain available to later snippets in that turn;
they are not automatically copied into later Skill turns. The installed eval
provider controls execution. See :class:`ReplSession` for what each call keeps
and returns.

``exec_code`` is a model-facing Tool, not an application entry point. See
:mod:`effectful.handlers.llm.examples.reasoning.continual` for a workflow that
uses it.
"""

import ast
import code
import codeop
import collections
import collections.abc
import contextlib
import io
import linecache
import textwrap
import types
import typing
import uuid

import pydantic

import effectful.handlers.llm.harness.execution.hooks
import effectful.handlers.llm.harness.validation.hooks
from effectful.handlers.llm.harness.durability.compaction import (
    CompactionScope,
    compact_,
)
from effectful.handlers.llm.harness.durability.transaction import HistoryBuilder
from effectful.handlers.llm.harness.execution.hooks import compile, exec, parse
from effectful.handlers.llm.harness.hooks import (
    AssistantResult,
    Message,
    PromptInjectingInterpretation,
    ToolCallExecutionError,
    ToolResult,
    call_agent,
    call_assistant,
    call_tool,
    call_user,
)
from effectful.handlers.llm.harness.serialization import (
    _TYPE_CHECK_ANCHOR_KEY,
    DecodedToolCall,
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
from effectful.ops.syntax import implements
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
    """A :class:`code.InteractiveInterpreter` seeded from a lexical context, whose
    compilation and execution go through the eval-provider operations
    (`_OpCommandCompiler`, `runcode`), capturing stdout and stderr per snippet.
    What a snippet keeps and returns is on `exec_code`."""

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
        """Sources of the snippets that reached execution, in order, including ones
        that raised -- the type-check context the `Encodable[CodeType]` decoder
        splices before the current snippet."""
        return self._prior_snippets

    def runcode(self, code: types.CodeType) -> None:
        # Mirrors `InteractiveInterpreter.runcode` closely; the only differences
        # are that `exec` here is the effect operation, so execution routes
        # through the installed eval provider, and errors propagate after writing.
        try:
            exec(code, self.locals)
        except:
            self.showtraceback()
            raise

    def exec_code(self, code: types.CodeType) -> str:
        """Run `code` in this session's namespace and return what it printed.

        Every snippet runs in the SAME namespace, so imports, definitions and
        assignments accumulate: this is a REPL, not a one-shot sandbox. The
        namespace starts seeded from the enclosing Skill call's scope (its bound
        arguments over the Skill's lexical context), which a snippet may read and
        rebind. Its lifetime is that call's: see `StatefulReplSynthesizer.call_agent`,
        which creates and discards it, and `StatefulReplSynthesizer.call_user` for
        why the model is told so twice.

        Returns this snippet's own slice of the session's output -- stdout (what
        `print` wrote) then stderr. There is no bare-expression auto-echo, so a
        snippet that prints nothing returns the empty string.

        A snippet that raises propagates and the session survives, so the next
        snippet can repair it. Whether bindings made before the raise survive is
        the executor's: the builtin executor keeps them, the restricted one copies
        bindings back only when a snippet returns. Output printed before the raise
        is *not* returned, since the raise reaches the caller in its place.
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
    first_new_stmt: int = 0,
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
    waived by ``lenient`` type checking.

    Only the current snippet is *reported*: ``first_new_stmt`` is its index in
    ``generated.body``, and the region starts there rather than at the body's first
    statement. The earlier snippets are still spliced, because their bindings are
    what the current one reads -- but they must not be diagnosed again, and not
    merely because they already passed. A function body is a stricter scope than the
    session it stands in for: the session is one namespace, seeded with the enclosing
    scope, where rebinding a seeded name is an ordinary assignment. Spliced into a
    function, that same assignment makes the name *local to the whole body*, so a
    snippet that read the seeded value before it -- and ran fine -- becomes a
    use-before-assignment the moment a later snippet writes to it. Reporting only
    the new statements keeps that artifact of the model out of the diagnostics.
    (It survives in one narrow form: a single snippet that reads a seeded name and
    then rebinds it is diagnosed, though the session would run it.)

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

    The caller decides whether there is anything to splice at all: it skips this
    when the Skill's source can't be recovered -- a Skill defined at a REPL, in a
    notebook, or via ``exec()`` is sourceless, so the code runs unchecked, exactly
    as `_splice_function` does for a sourceless Callable anchor -- and when the
    snippet contributes no statements to report on. Raises ``RuntimeError`` only on
    source *drift* (source recovered but the def no longer sits where it was
    compiled from), which ``_recover_skill_def`` surfaces.
    """
    assert 0 <= first_new_stmt < len(generated.body)
    skill_def.body = list(generated.body)

    # `skill_def` is still a node in `module_ast` (only its body changed), so its
    # walk-order index is stable across the unparse round-trip.
    def_index = _def_nodes(module_ast).index(skill_def)
    checked_source = ast.unparse(ast.fix_missing_locations(module_ast))
    spliced = _def_nodes(ast.parse(checked_source))[def_index]
    lo = spliced.body[first_new_stmt].lineno
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
        # (`StatefulReplSynthesizer.repl_history` returns the prior snippets of the session in scope)
        # plus this snippet into the Skill body and check it. A type error raises here
        # -> the tool-call decode fails -> `TenacityRetryer` retries, so ill-typed code
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
            # Where the current snippet starts in the cumulative body -- the prior
            # snippets are spliced for their bindings but reported on separately
            # when they were current, so only this snippet's span is diagnosed.
            # A snippet of only comments contributes no statements, and there is
            # then nothing to splice or report.
            first_new_stmt = len(ast.parse(prior_src).body)
            if first_new_stmt < len(session.body):
                effectful.handlers.llm.harness.validation.hooks.type_check(
                    *_splice_snippet(session, module_ast, skill_def, first_new_stmt),
                    lenient=True,
                )

        return effectful.handlers.llm.harness.execution.hooks.compile(module, filename)

    return typing.Annotated[
        ty,
        pydantic.InstanceOf,
        pydantic.BeforeValidator(validate),
        pydantic.PlainSerializer(
            lambda value: "".join(linecache.getlines(value.co_filename))
        ),
        pydantic.WithJsonSchema({"type": "string"}),
    ]


class StatefulReplSynthesizer(PromptInjectingInterpretation):
    """A REPL is a temporary Python session for the current Skill turn. Use
    ``exec_code`` to inspect values or run code in that session; print values
    you want to see. Imports and bindings survive between snippets in this
    turn. They do not automatically appear in a later Skill call. Store a
    result on ``self`` or another program-owned object when it must outlive
    this session. For a receiver with a declared ``codes: dict[str, str]``
    field, a snippet can save data and an explicit Tool::

        from effectful.handlers.llm import Tool

        @Tool.define
        def next_guess(prefix: str) -> str:
            \"\"\"Append a trial letter to a prefix.\"\"\"
            return prefix + "A"

        self.next_guess = next_guess
        self.codes["room0"] = "BBA"

    In a later Skill turn, a new REPL session can read the saved data::

        print(self.codes["room0"])
    """

    @typing.final
    @Tool.define
    @classmethod
    def exec_code(
        cls, code: types.CodeType, compact: CompactionScope = CompactionScope.NONE
    ) -> str:
        """Run Python in the current Skill turn's REPL and return its output.

        The session starts with the Skill's arguments and lexical scope. Imports,
        definitions, and assignments remain available to later snippets in this
        turn, but not automatically to later Skill calls. Print a value to see it;
        a bare expression produces no output::

            print(1 + 1)  # returns "2\\n"

        The result contains stdout followed by stderr.

        If code raises, the Tool reports the traceback and output printed before
        the error is lost. With the builtin executor, bindings made before the
        error remain in this session; the restricted executor keeps bindings
        only after a successful snippet. Where source and a type checker are
        available, the snippet is checked before it runs.

        ``compact`` shortens conversation history after a successful snippet. The
        current request, snippet, and result remain; Python objects are not reset.
        """
        raise NotImplementedError("No handler")

    @typing.final
    @Operation.define
    @classmethod
    def repl_history(cls) -> list[str]:
        """This REPL session's executed snippets, in order, including ones that raised.

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

    @implements(call_agent)
    def call_agent[**P, T](
        self, skill: Skill[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """Open a REPL session for the duration of this call.

        The session's namespace is seeded from a `collections.ChainMap` of the
        bound arguments over the Skill's lexical context, so this call's own
        arguments shadow same-named module globals -- which is the scope the
        model is shown, and the scope a snippet executes against.

        Every session-scoped operation is bound here, inside the `handler`
        block, so the session is created and discarded with the call. A nested
        Skill call runs this rule again and gets a namespace of its own; nothing
        outlives the ``with``.
        """
        bound_args = skill.__signature__.bind(*args, **kwargs)
        bound_args.apply_defaults()
        env = collections.ChainMap(bound_args.arguments, skill.__context__)
        session = ReplSession(env=env)
        with handler(
            {
                self.exec_code: lambda code, compact=CompactionScope.NONE: (
                    session.exec_code(code)
                ),
                self.repl_history: lambda: session.prior_snippets,
                self.repl_env: lambda: session.locals,
            }
        ):
            return fwd()

    @implements(call_user)
    def call_user(self, user_prompt: PromptSection) -> typing.Any:
        """Tell each request which session it opens, and what is bound in it.

        Attached to the request rather than to the system message because both
        facts are per-call: the session boundary *is* this message, and the
        arguments named by its heading are this call's.

        The section is the same text every time, so this rule
        needs nothing from the call it decorates -- which is why it can be a rule
        at all, rather than something `call_agent` closes over the `Skill` to
        build.
        """
        session_note = textwrap.dedent("""
        A fresh, empty Python session opens with this request and is discarded when you
        answer it. Names you bound in an earlier request's session are gone; `self` and
        the lexical scope are not, because they belong to the calling program rather
        than to the session.

        The parameters of the signature heading this request are bound in that session,
        under those names and with those types -- their values are already written out
        above, so read the names rather than retyping what you were given. The rest of
        the namespace is in the system message's *Modules already in scope that do
        not need to be re-imported* and *Variables already in scope that do not
        need to be re-defined* tables.
        """)
        return fwd(
            PromptSection(
                type="prompt_section",
                title=user_prompt["title"],
                content=[
                    *user_prompt["content"],
                    PromptSection(
                        type="prompt_section",
                        title="REPL session",
                        content=to_content_blocks(session_note),
                    ),
                ],
            )
        )

    @implements(call_tool)
    def call_tool[T](self, tool_call: DecodedToolCall[T]) -> ToolResult[T]:
        """Compact the conversation after a successful ``exec_code(compact=...)``.

        Only on success: a snippet that raised clears nothing, since the model
        has not yet had the chance to record what mattered, and the traceback it
        needs to repair the snippet is in the very round a clear would drop.

        Forwarding first is what makes this safe to do inline. By the time
        control returns, `HistoryBuilder` has appended this call's tool message,
        so the round `compact` keeps is complete; and because compaction only
        removes messages *ahead* of the assistant message that requested this
        call, a sibling tool call answered later still finds that message and
        cannot be orphaned.
        """
        message, result, is_final = fwd(tool_call)
        if tool_call.tool is self.exec_code and not isinstance(
            result, ToolCallExecutionError
        ):
            compact_(
                HistoryBuilder.get_history(),
                tool_call.id,
                tool_call.bound_args.arguments.get("compact", CompactionScope.NONE),
            )
        return message, result, is_final

    @implements(call_assistant)
    def call_assistant[T](
        self,
        messages: collections.abc.Sequence[Message],
        response_type: type[T],
        env: collections.abc.Mapping[str, typing.Any],
        tools: collections.abc.Set[Tool] = frozenset(),
    ) -> AssistantResult[T]:
        """Offer the REPL tools, over an env that includes the session namespace.

        The session's bindings are merged *over* the request env, so a name the
        model defined in an earlier snippet is visible to whatever the next
        request decodes -- notably the type check a new snippet is held to,
        which would otherwise see every such name as undefined.
        """
        return fwd(
            messages,
            response_type,
            {**env, **self.repl_env()},
            tools
            | {
                self.exec_code,
            },
        )
