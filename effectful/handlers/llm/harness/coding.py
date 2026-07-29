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

from effectful.handlers.llm.harness.completions import (
    AssistantResult,
    call_assistant,
    call_system,
)
from effectful.handlers.llm.harness.encoding import (
    MethodTemplateBody,
    TemplateBody,
    _callable_type_from_signature,
)
from effectful.handlers.llm.harness.execution import compile, exec, parse
from effectful.handlers.llm.types import FinalTool, Template, Tool
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

    # The session's captured output, accumulated across calls and exposed for
    # introspection.  stdout (`print` output) and stderr (writes plus tracebacks)
    # are kept separate; `exec_code` returns each call's slice of both.
    stdout: io.StringIO
    stderr: io.StringIO

    def __init__(self, env: collections.abc.MutableMapping[str, typing.Any]):
        # Run in a fresh writable dict seeded with a flat view of `env`.  This is
        # forced by `exec`: its globals must be one real dict (a ChainMap is
        # rejected), and a REPL needs a single persistent namespace so a function
        # defined in one snippet sees a name a later snippet binds.  Seeding a flat
        # copy also leaves the lexical seed untouched, so REPL assignments never
        # leak into the surrounding scope.
        scope: dict[str, typing.Any] = dict(env)
        # When `env` is the per-call `ChainMap` (its outer layers are read-only
        # frame proxies), splice this dict in as an extra shadowing first layer so
        # the bindings are *also* visible to the rest of the Template call
        # (mirroring `exec`) -- still scoped to the call, since that ChainMap is.
        if isinstance(env, collections.ChainMap):
            env.maps.insert(0, scope)
        # `InteractiveInterpreter.__init__` stores it as `self.locals`, so we reuse
        # the base's runcode/showtraceback/write machinery.
        super().__init__(scope)
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
        variable assignments all accumulate during the session of this template.
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
        # splice the accumulated session code into the Template body. The type check itself
        # lives in the `Encodable[CodeType]` decoder (as it does for synthesized Callables),
        # not here -- this session only runs code.
        self._prior_snippets.append("".join(linecache.getlines(code.co_filename)))
        with (
            contextlib.redirect_stdout(self.stdout),
            contextlib.redirect_stderr(self.stderr),
        ):
            self.runcode(code)
        return self.stdout.getvalue()[out_start:] + self.stderr.getvalue()[err_start:]


class StatefulReplSynthesizer(ObjectInterpretation):
    """Expose a persistent Python session to the LLM as an `exec_code` Tool.

    Off by default; install it where the LLM should be able to run code whose
    state (variables, imports, definitions) survives across tool calls within a
    single Template invocation.

    Scoping mirrors how `__history__` is managed for Template calls: `PythonRepl`
    handles `Template.__apply__` to introduce fresh session-bound handlers (`exec_code`,
    `read_lexical_variable`, `repl_history`) for the duration of the call, and intercepts
    `call_assistant` to inject an `exec_code` Tool routed to that session.  The session is
    therefore introduced and
    eliminated by its own handler, bounded to the Template call by construction --
    there is no global registry of sessions, and nested Template calls get their
    own isolated sessions.

    The session is seeded from the Template's lexical context and routes execution
    through the `parse`/`compile`/`exec` effect operations, so it works under any
    installed eval provider (`UnsafeEvalProvider` or `RestrictedEvalProvider`).
    """

    @typing.final
    class _ReplInteractionTool[**P, T](Tool[P, T]):
        """## Python REPL

        You may run arbitrary Python code in a persistent session. The code is
        executed in the context of this Template's lexical scope (see the *Lexical
        scope* table for the available names and their types). The session persists
        across turns, so you may define variables, functions, and classes that are
        used in later turns. The return value of the code is returned to you as the
        result of the tool call.

        Use the REPL only when running code actually helps — computing or verifying
        a result, exploring data, or calling a tool. If you can answer directly, just
        answer; do not route a plain text answer through `print(...)`.
        """

    @typing.final
    @_ReplInteractionTool.define
    @classmethod
    @functools.wraps(ReplSession.exec_code)
    def exec_code(cls, code: types.CodeType) -> str:
        raise NotImplementedError("No handler")

    @typing.final
    @_ReplInteractionTool.define
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

    @implements(call_system)
    def _call_system(self, template, tool_types=frozenset()):
        return fwd(template, tool_types=tool_types | {self._ReplInteractionTool})

    @implements(Template.__apply__)
    def _apply[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        bound_args = template.__signature__.bind(*args, **kwargs)
        bound_args.apply_defaults()
        env = collections.ChainMap(bound_args.arguments, template.__context__)
        session = ReplSession(env=env)
        with handler(
            {
                self.exec_code: session.exec_code,
                self.read_lexical_variable: env.get,
                self.repl_history: lambda: session.prior_snippets,
            }
        ):
            return fwd()

    @implements(call_assistant)
    def _call_assistant[T](
        self,
        env: collections.abc.Mapping[str, typing.Any],
        response_type: type[T],
        tools: collections.abc.Set[Tool] = frozenset(),
        anchor: "Template | None" = None,
        force_tool: bool = False,
    ) -> AssistantResult[T]:
        return fwd(
            env,
            response_type,
            tools | {self.exec_code, self.read_lexical_variable},
            anchor=anchor,
            force_tool=force_tool,
        )


class FinalBodySynthesizer(ObjectInterpretation):
    """Answer a Template by synthesizing a function and calling it.

    Instead of asking the LLM to generate an instance of the Template's return
    type directly, this handler exposes a :class:`FinalTool` that lets the model
    "answer" by writing a Python function with the Template's signature.  The
    harness applies that function to the original arguments and its return value
    becomes the Template's result.  This is the declarative "CodeAdapt" workflow:
    the LLM writes code implementing the body of the Template rather than
    reasoning out the answer itself.

    The synthesis tool is offered *alongside* the Template's normal completion
    paths rather than replacing them: across turns the model may freely call any
    other tool in scope (their results are fed back as usual), and it may still
    answer the return type directly via structured output.  The loop terminates
    when it either answers directly or calls the synthesis :class:`FinalTool`.
    To force the synthesis path, pass ``tool_choice="required"`` (handler config
    is forwarded to the model request).  The function is synthesized by reusing
    the existing ``Callable`` synthesis machinery: the tool's argument is typed
    as ``Callable[[params], ret]``, so :func:`call_assistant`'s tool-call
    decoding parses, type-checks, compiles and executes the model's code into a
    real function before it is applied.

    Failures compose with :class:`RetryLLMHandler`: a function that fails to
    synthesize surfaces as a :class:`ToolCallDecodingError`, and one that raises
    when applied to the inputs as a :class:`ToolCallExecutionError`; both are fed
    back to the model as a tool message and the loop continues so it can revise::

        with (
            handler(LiteLLMProvider(model="gpt-5-mini")),
            handler(SynthesizeAndCall()),
            handler(RetryLLMHandler()),
        ):
            ...

    Requires an eval provider (e.g. :class:`UnsafeEvalProvider` or
    :class:`RestrictedEvalProvider`) to be installed so the synthesized code can
    be compiled and executed.
    """

    @typing.final
    class _SynthesisFinalTool[T](FinalTool[[collections.abc.Callable[..., T]], T]):
        """## Code synthesis

        You may "answer" a Template by writing code instead of producing the value
        directly. A final tool (typically `submit_solution`) accepts a single
        argument: a Python function whose signature matches the Template's signature
        (see its spec below). The harness applies that function to the original
        inputs and its return value becomes the answer, so write the function body
        as a drop-in implementation of the Template. The function may reference
        names from the lexical scope (see the *Lexical scope* table).

        You do not need to write a docstring or doctests: on submission the harness
        attaches the Template's own docstring to your function and runs *its*
        doctests (with recursive calls to the Template routed to your
        implementation). A solution whose doctests fail — or that errors when
        applied — is rejected and fed back to you to revise, so the answer only
        stands once the Template's doctests pass. Write just the implementation;
        any docstring you add is replaced and ignored. Calling this tool terminates
        the completion.

        This answers the *current* call only. Each call is a fresh, independent
        task: even if you already submitted a working solution earlier in this
        conversation, a prior submission is not a standing answer — you must call
        `submit_solution` again to answer the current call. Never end a turn with
        a prose summary in place of the answer; a plain message is not a valid
        response and will be rejected.
        """

        __toolname__: typing.ClassVar[typing.Literal["submit_solution"]] = (
            "submit_solution"
        )

        @classmethod
        def define(
            cls,
            template: Template[..., T],
            bound_args: inspect.BoundArguments,
        ) -> FinalTool[[collections.abc.Callable[..., T]], T]:
            if isinstance(template.__default__, types.MethodType):
                signature = inspect.signature(template.__default__.__func__)
                args, kwargs = (
                    (template.__default__.__self__,) + bound_args.args,
                    bound_args.kwargs,
                )
                body_type = MethodTemplateBody[  # type: ignore
                    typing.get_args(_callable_type_from_signature(signature))
                ]
                return_type = signature.return_annotation
            else:
                signature = inspect.signature(template)
                args, kwargs = bound_args.args, bound_args.kwargs
                body_type = TemplateBody[  # type: ignore
                    typing.get_args(_callable_type_from_signature(signature))
                ]
                return_type = signature.return_annotation

            def submit_solution(implementation: body_type) -> return_type:  # type: ignore
                """
                Answer this Template by submitting a Python function that implements
                it (see the "Code synthesis" section); its return value on the
                original arguments becomes the answer.
                """
                return implementation(*args, **kwargs)  # type: ignore

            return super().define(submit_solution, name=cls.__toolname__)

    @implements(call_system)
    def _call_system(self, template, tool_types=frozenset()):
        return fwd(template, tool_types=tool_types | {self._SynthesisFinalTool})

    @implements(Template.__apply__)
    def _apply[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        bound_args = template.__signature__.bind(*args, **kwargs)
        bound_args.apply_defaults()
        tool = self._SynthesisFinalTool.define(template, bound_args)

        def _add_synthesis_tool(
            env, response_type, tools=frozenset(), anchor=None, force_tool=False
        ):
            if any(isinstance(t, self._SynthesisFinalTool) for t in tools):
                return fwd()
            return fwd(
                env, response_type, tools | {tool}, anchor=anchor, force_tool=force_tool
            )

        with handler({call_assistant: _add_synthesis_tool}):
            return fwd()
