"""The operations of the agent loop.

These are the extension points that every other handler in
:mod:`effectful.handlers.llm.harness` implements or intercepts.
"""

import abc
import collections
import collections.abc
import dataclasses
import functools
import inspect
import json
import traceback
import typing

import litellm
import pydantic

from effectful.handlers.llm.harness.serialization import (
    _IS_FINAL_KEY,
    _NAME2TOOL_KEY,
    _TYPE_CHECK_ANCHOR_KEY,
    DecodedToolCall,
    PromptSection,
    _advertised_names,
    _BoxedResponse,
    _NameAndTool,
    _render_prompt_section,
    _UndecodableReturn,
    format_as_content_blocks,
    to_content_blocks,
)
from effectful.handlers.llm.types import (
    Encodable,
    Skill,
    Tool,
)
from effectful.internals.unification import (
    freetypevars,
    nested_type,
    substitute,
    unify,
)
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Operation

Message = (
    litellm.ChatCompletionAssistantMessage
    | litellm.ChatCompletionToolMessage
    | litellm.ChatCompletionSystemMessage
    | litellm.ChatCompletionUserMessage
)


class DecodingError[E: Exception](abc.ABC, Exception):
    """Base class for decoding errors that can occur during LLM response processing."""

    original_error: E

    @abc.abstractmethod
    def to_feedback_message(self, *, include_traceback: bool = True) -> Message:
        """Convert the decoding error into a feedback message to be sent back to the LLM."""
        raise NotImplementedError


@dataclasses.dataclass
class ToolCallDecodingError[E: Exception](DecodingError[E]):
    """Error raised when decoding a tool call fails."""

    original_error: E
    raw_message: Message
    raw_tool_call: litellm.ChatCompletionMessageToolCall

    def __str__(self) -> str:
        return f"Error decoding tool call '{self.raw_tool_call.function.name}': {self.original_error}. Please provide a valid response and try again."

    def to_feedback_message(
        self, *, include_traceback: bool = True
    ) -> litellm.ChatCompletionToolMessage:
        error_message = f"{self}"
        if include_traceback:
            tb = traceback.format_exc()
            error_message = f"{error_message}\n\nTraceback:\n```\n{tb}```"
        return {
            "role": "tool",
            "tool_call_id": self.raw_tool_call.id,
            "content": error_message,
        }


@dataclasses.dataclass
class ResultDecodingError[E: Exception](DecodingError[E]):
    """Error raised when decoding the LLM response result fails."""

    original_error: E
    raw_message: Message

    def __str__(self) -> str:
        return f"Error decoding response: {self.original_error}. Please provide a valid response and try again."

    def to_feedback_message(
        self, *, include_traceback: bool = True
    ) -> litellm.ChatCompletionUserMessage:
        error_message = f"{self}"
        if include_traceback:
            tb = traceback.format_exc()
            error_message = f"{error_message}\n\nTraceback:\n```\n{tb}```"
        return {"role": "user", "content": error_message}


@dataclasses.dataclass
class ToolCallExecutionError[E: Exception, T](DecodingError[E]):
    """Error raised when a tool execution fails at runtime."""

    original_error: E
    raw_tool_call: DecodedToolCall[T]

    def __str__(self) -> str:
        return f"Tool execution failed: Error executing tool '{self.raw_tool_call.name}': {self.original_error}"

    def to_feedback_message(
        self, *, include_traceback: bool = True
    ) -> litellm.ChatCompletionToolMessage:
        error_message = f"{self}"
        if include_traceback:
            tb = traceback.format_exc()
            error_message = f"{error_message}\n\nTraceback:\n```\n{tb}```"
        return {
            "role": "tool",
            "tool_call_id": self.raw_tool_call.id,
            "content": error_message,
        }


@Operation.define
@functools.wraps(litellm.completion, assigned=(), updated=())
def completion(*args, **kwargs) -> typing.Any:
    """Low-level LLM request. Handlers may log/modify requests and delegate via fwd().

    This effect is emitted for model request/response rounds so handlers can
    observe/log requests.

    """
    return litellm.completion(*args, **kwargs)


type AssistantResult[T] = tuple[
    litellm.ChatCompletionAssistantMessage,
    typing.Sequence[DecodedToolCall],
    T | None,
]


@Operation.define
def call_assistant[T](
    messages: collections.abc.Sequence[Message],
    response_type: type[T],
    env: collections.abc.Mapping[str, typing.Any],
    tools: collections.abc.Set[Tool] = frozenset(),
) -> AssistantResult[T]:
    """Low-level LLM request. Handlers may log/modify requests and delegate via fwd().

    This effect is emitted for model request/response rounds so handlers can
    observe/log requests.

    The request is fully determined by the arguments: `messages` is the
    conversation sent to the model, so the rule reads no ambient history and a
    caller (or an intercepting handler) decides exactly what the model sees.

    The available `tools` are passed explicitly as a set; handlers that expose
    additional tools (synthetic readers, REPL access, synthesis) intercept this
    operation and union them into `tools` before forwarding.

    Raises:
        ToolCallDecodingError: If a tool call cannot be decoded. The error
            includes the raw assistant message for retry handling.
        ResultDecodingError: If the result cannot be decoded. The error
            includes the raw assistant message for retry handling.
    """

    if _TYPE_CHECK_ANCHOR_KEY in env:
        tools = tools - {env[_TYPE_CHECK_ANCHOR_KEY]}

    if response_type is _UndecodableReturn and not tools:
        raise TypeError(
            f"{inspect.getdoc(_UndecodableReturn)} -- but this request offers "
            f"no tools, so no reply could ever decode. Install a handler that "
            f"offers a final-answer tool (e.g. `FinalBodySynthesizer`'s "
            f"``submit_solution``), or give the skill's signature an "
            f"instantiation channel such as a ``type[T]`` parameter."
        )

    name2tool = _advertised_names(tools)
    env = {_NAME2TOOL_KEY: name2tool, **env}
    spec_encoding: pydantic.TypeAdapter[_NameAndTool] = pydantic.TypeAdapter(
        Encodable[_NameAndTool]
    )
    tool_specs = [
        spec_encoding.dump_python(_NameAndTool(name, t), mode="json")
        for name, t in sorted(name2tool.items())
    ]

    # The OpenAI API requires a wrapper object for non-object structured output types,
    # so we create one on the fly here. Using a Pydantic model offloads JSON schema
    # generation and validation logic to litellm, and offers better error messages.
    response_format: type[_BoxedResponse[T]] = pydantic.create_model(
        "BoxedResponse",
        value=Encodable[response_type],  # type: ignore[valid-type]
        __base__=_BoxedResponse,
    )

    response: litellm.types.utils.ModelResponse = completion(
        messages=list(messages),
        response_format=None if response_type is str else response_format,
        tools=tool_specs,
    )
    choice = response.choices[0]
    assert isinstance(choice, litellm.types.utils.Choices)

    message: litellm.Message = choice.message

    raw_message = typing.cast(
        litellm.ChatCompletionAssistantMessage, message.model_dump(mode="json")
    )
    assert raw_message["role"] == "assistant"

    raw_tool_calls = message.get("tool_calls") or []
    tool_calls: list[DecodedToolCall] = []
    encoding: pydantic.TypeAdapter[DecodedToolCall] = pydantic.TypeAdapter(
        Encodable[DecodedToolCall]
    )
    for raw_tool_call in raw_tool_calls:
        try:
            tool_calls += [encoding.validate_python(raw_tool_call, context=env)]
        except Exception as e:
            raise ToolCallDecodingError(
                raw_tool_call=raw_tool_call,
                original_error=e,
                raw_message=raw_message,
            ) from e

    result = None
    if not tool_calls:
        serialized_result = message.get("content") or message.get("reasoning_content")
        assert isinstance(serialized_result, str)
        try:
            if response_type is str:
                result = typing.cast(T, serialized_result)
            else:
                result = response_format.model_validate(
                    json.loads(serialized_result),
                    context={**env, _IS_FINAL_KEY: True},
                ).value
        except Exception as e:
            raise ResultDecodingError(e, raw_message=raw_message) from e

    return (raw_message, tool_calls, result)


type ToolResult[T] = tuple[
    litellm.ChatCompletionToolMessage, T | ToolCallExecutionError, bool
]


@Operation.define
def call_tool[T](tool_call: DecodedToolCall[T]) -> ToolResult[T]:
    """Implements a roundtrip call to a python function. Input is a json
    string representing an LLM tool call request parameters. The output is
    the serialised response to the model.

    Returns the appended tool message, the tool's return value, and whether the
    call finalizes the Skill -- always ``False`` here. Finalization is a policy
    a handler of this operation applies to its own tools, not a property of the
    tool's type: see
    `effectful.handlers.llm.harness.synthesis.body.FinalBodySynthesizer`, which
    marks its ``submit_solution`` call final so that value becomes the Skill's
    result and the completion loop stops.

    The returned value is a :class:`ToolCallExecutionError` rather than the tool's
    result when a handler captured a failed call (see
    `effectful.handlers.llm.harness.durability.TenacityRetryer`); this rule itself
    raises instead.
    """
    # call tool with python types
    try:
        result = tool_call.tool(
            *tool_call.bound_args.args, **tool_call.bound_args.kwargs
        )
    except Exception as e:
        raise ToolCallExecutionError(raw_tool_call=tool_call, original_error=e) from e

    return_type: pydantic.TypeAdapter[typing.Any] = pydantic.TypeAdapter(
        Encodable[nested_type(result).value]  # type: ignore[misc]
    )
    encoded_result = to_content_blocks(
        return_type.dump_python(result, mode="json", context={})
    )
    message = litellm.ChatCompletionToolMessage(
        role="tool",
        content=encoded_result,  # type: ignore
        tool_call_id=tool_call.id,
    )
    return (message, result, False)


@Operation.define
def call_user(user_prompt: PromptSection) -> litellm.ChatCompletionUserMessage:
    """
    Format a `Skill`'s prompt applied to arguments into a user message.

    `user_prompt` is wrapped in an enclosing document for the same reason
    `call_system` assembles one: `_render_prompt_section` treats level 0 as the
    document itself and does not render its title, so a section handed straight
    to it would lose its heading.  Wrapped, `user_prompt` is a child and its
    title becomes the message's ``#`` heading.
    """
    document = PromptSection(
        type="prompt_section",
        title=f"User Prompt: {user_prompt['title']}",
        content=[user_prompt],
    )
    return litellm.ChatCompletionUserMessage(
        role="user", content=list(_render_prompt_section(document))
    )


@Operation.define
def call_system(
    harness_prompt: PromptSection, agent_prompt: PromptSection
) -> litellm.ChatCompletionSystemMessage:
    """Assemble the system message from the two halves of the system prompt.

    `agent_prompt` describes the task: the caller (`AgentLoop._call`)
    introspects it from the `Skill` being called.  `harness_prompt` describes
    the machinery the task runs under, and arrives empty: each installed handler
    with something to say about the harness intercepts this operation and adds the
    section documenting the capability it provides.  This rule only has to put the
    two together and flatten the result.

    Handing the handlers their own argument is what keeps them independent.  A
    handler appends to `harness_prompt` and forwards; it never looks a section up
    by title, never has to create one that isn't there, and cannot disturb the
    other half of the document.  Installation order therefore decides only the
    order the sections appear in -- innermost first, since it intercepts first --
    and never whether one of them lands.
    """
    document = PromptSection(
        type="prompt_section",
        title=f"System Prompt: {agent_prompt['title']}",
        content=[harness_prompt, agent_prompt],
    )
    return litellm.ChatCompletionSystemMessage(
        role="system", content=list(_render_prompt_section(document))
    )


def _instantiate_return_type(
    skill: Skill, bound_args: inspect.BoundArguments
) -> typing.Any:
    """The skill's return annotation, with type parameters instantiated from
    the actual arguments of one call.

    A generic skill -- ``def make_fn[T](typ: type[T]) -> Callable[[T], T]`` --
    declares its return type in terms of variables the *caller* fixes, so the
    response schema for a particular call must be built from the
    instantiation, not the variable. The whole signature is unified against
    the type-level image of the call's arguments, through two channels:

    * A parameter annotated ``type[T]`` binds ``T`` to the very class (or
      type alias) the caller passed -- a value-level fact, read off directly.
      This is the channel a generic skill should prefer to declare.
    * Every other TypeVar-carrying parameter unifies against the *inferred*
      type of its argument (`nested_type`). Type reconstruction from values
      is only reliable for collections and callables, so this channel is
      strictly best-effort: any failure anywhere falls back to the original
      annotation, leaving every type parameter unbound -- refusal, never a
      wrong binding.

    A type parameter that no channel could bind is finally bound to
    `~effectful.handlers.llm.harness.serialization._UndecodableReturn`
    (unless it carries a bound or constraints, which already produce a real
    schema): there is no sound direct decoding for such a return, so its
    response format is strict-legal but refuses every reply, with feedback
    redirecting the model to answer through a final-answer tool (canonically
    ``submit_solution``, whose synthesized implementation is checked against
    the real generic signature and applied to the real arguments).
    A binding that doesn't fit the model's reply fails response validation
    loudly, and the retry loop reports it.
    """
    sig = inspect.signature(skill)
    return_annotation = sig.return_annotation
    if not freetypevars(return_annotation):
        return return_annotation

    try:
        typed = sig.bind_partial()
        for name, value in bound_args.arguments.items():
            param = sig.parameters[name]
            if param.annotation is inspect.Parameter.empty:
                continue
            if not freetypevars(param.annotation):
                typed.arguments[name] = param.annotation
            elif param.kind is inspect.Parameter.VAR_POSITIONAL:
                typed.arguments[name] = tuple(nested_type(v).value for v in value)
            elif param.kind is inspect.Parameter.VAR_KEYWORD:
                typed.arguments[name] = {
                    k: nested_type(v).value for k, v in value.items()
                }
            elif typing.get_origin(param.annotation) is type:
                typed.arguments[name] = type[value]
            else:
                typed.arguments[name] = nested_type(value).value
        subs = unify(sig, typed, {})
        instantiated = (
            substitute(return_annotation, subs) if subs else return_annotation
        )
    except Exception:
        instantiated = return_annotation

    try:
        leftover = {
            tv: _UndecodableReturn
            for tv in freetypevars(instantiated)
            if getattr(tv, "__bound__", None) is None
            and not getattr(tv, "__constraints__", ())
        }
        return substitute(instantiated, leftover) if leftover else instantiated
    except Exception:
        return return_annotation


call_agent = Skill.__apply__
"""Alias for `Skill.__apply__`: the operation invoked when a `Skill` is called.

Handlers install against this to intercept an agent call, alongside the other
`call_*` hooks in this module.
"""


class AgentLoop(ObjectInterpretation):
    def _skill_system_prompt(self, skill: Skill) -> PromptSection:
        """The half of the system prompt describing a call to `skill`.

        Everything here is introspected from the `Skill`, and its subsections
        are laid out most-constant-first so that the document caches well as
        the conversation grows: the module, then the agent and its skills, then
        the names in scope.  `call_system` puts this after the harness half,
        which is constant over the whole process.

        A plain method, not an operation: a handler that wants to say something
        here intercepts `call_system` and adds to the *harness* half, which is
        the argument it is handed for exactly that purpose.
        """
        from effectful.handlers.llm.harness.legibility.lexical import (
            _agent_section,
            _imports_section,
            _module_section,
            _vars_section,
        )

        sections = (
            _module_section(inspect.getmodule(skill)),
            _agent_section(skill),
            _imports_section(skill.__context__),
            _vars_section(skill.__context__),
        )
        return PromptSection(
            type="prompt_section",
            title=f"`{skill.__name__}{skill.__signature__}`",
            content=[s for s in sections if s is not None],
        )

    def _skill_user_prompt(
        self, skill: Skill, env: collections.abc.Mapping[str, typing.Any]
    ) -> PromptSection:
        """The `Skill`'s prompt -- its docstring -- applied to `env`.

        This is the request itself, as opposed to the standing description of
        the task that `_skill_system_prompt` assembles: the docstring's
        ``{placeholders}`` are filled from the bound arguments, so it is the one
        part of the exchange that differs per call.

        Only the docstring is interpolated.  The title is a heading rendered as
        written, so a signature carrying braces -- a ``{}`` default, say --
        needs no escaping and must not get any.
        """
        assert skill.__doc__ is not None
        return PromptSection(
            type="prompt_section",
            title=f"{skill.__name__}{skill.__signature__}",
            content=format_as_content_blocks(skill.__doc__, env),
        )

    @implements(call_agent)
    def _call[**P, T](self, skill: Skill[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        from effectful.handlers.llm.harness.durability.transaction import HistoryBuilder

        # The harness half starts empty and is filled in by whichever handlers
        # are installed; with none of them it renders as nothing at all.
        message: Message = call_system(
            PromptSection(type="prompt_section", title="Harness", content=[]),
            self._skill_system_prompt(skill),
        )

        bound_args = inspect.signature(skill).bind(*args, **kwargs)
        bound_args.apply_defaults()
        env = skill.__context__.new_child(
            bound_args.arguments | {_TYPE_CHECK_ANCHOR_KEY: skill}
        )

        message = call_user(self._skill_user_prompt(skill, env))

        result: T | None = None
        is_final: bool = False
        response_type = _instantiate_return_type(skill, bound_args)
        while not is_final:
            message, tool_calls, result = call_assistant(
                list(HistoryBuilder.get_history()),
                response_type,
                env,
            )
            if tool_calls:
                for tool_call in tool_calls:
                    message, result, is_final = call_tool(tool_call)
                    if is_final:
                        assert len(tool_calls) == 1, (
                            f"a finalizing tool call must be the only call in its "
                            f"turn, but {len(tool_calls)} were requested"
                        )
                        break
            else:
                is_final = True

        return typing.cast(T, result)
