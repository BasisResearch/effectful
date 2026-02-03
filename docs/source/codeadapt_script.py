# %%
from collections.abc import Callable

from effectful.handlers.llm.encoding_template import Encodable, TemplateEncodable

from effectful.handlers.llm.template import Template
from effectful.ops.types import NotHandled


# %%
@Encodable.define.register(Template)
def _encodable_template(ty, ctx):
    return TemplateEncodable(ty, str, ctx or {})

# %%
@Template.define
def generate_paragraph() -> str:
    """Please generate a paragraph: with exactly 4 sentences ending with 'walk', 'tumbling', 'another', and 'lunatic'.
    """
    raise NotHandled

# %%
@Template.define
def codeact(
    template_name: str,
    args_json: str = "[]",
    kwargs_json: str = "{}",
) -> Callable[[], str]:
    """Generate a code that solve the following problem:
    {template_name}
    Args/kwargs are provided as JSON strings (args_json, kwargs_json).
    DO NOT USE codeadapt tool.
    """
    raise NotHandled


@Template.define
def codeadapt(
    template_name: str,
    args_json: str = "[]",
    kwargs_json: str = "{}",
) -> str:
    """Reason about the template, uses the codeact tool to generate a code that solve the problem.
    The template:
    {template_name}
    Args/kwargs are provided as JSON strings (args_json, kwargs_json).
    """
    raise NotHandled



# %%
import inspect
import json

from effectful.handlers.llm.completions import (
    DecodedToolCall,
    Encodable,
    LiteLLMProvider,
    Message,
    RetryLLMHandler,
    call_assistant,
    call_system,
    call_tool,
    call_user,
    get_message_sequence,
    handler,
    implements,
)


class ToolNotUsedError(Exception):
    """Exception raised when a tool is not used in the code."""

    tool_name: str


class CodeAdapt(LiteLLMProvider):
    def __init__(self, model: str = "gpt-4o"):
        super().__init__(model=model)

    @implements(Template.__apply__)
    def _call[**P, T](self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        tool_called = False

        # Avoid recursive handling when codeadapt is invoked on a Template argument.
        if template is codeadapt and args and isinstance(args[0], Template):
            args = (args[0].__name__, *args[1:])

        message_sequence = get_message_sequence()
        with handler({get_message_sequence: lambda: message_sequence}), handler(
            RetryLLMHandler()
        ):
            # encode arguments
            bound_args = inspect.signature(template).bind(*args, **kwargs)
            bound_args.apply_defaults()
            env = template.__context__.new_child(bound_args.arguments)

            # Create response_model with env so tools passed as arguments are available
            response_model = Encodable.define(
                template.__signature__.return_annotation, env
            )

            call_system(template)
            message: Message = call_user(template.__prompt_template__, env)

            # loop based on: https://cookbook.openai.com/examples/reasoning_function_calls
            tool_calls: list[DecodedToolCall] = []
            result: T | None = None
            while message["role"] != "assistant" or tool_calls:
                print(json.dumps(message_sequence, indent=2))
                message, tool_calls, result = call_assistant(
                    template.tools, response_model, **self.config
                )
                for tool_call in tool_calls:
                    message = call_tool(tool_call)
                    tool_called = True

            assert result is not None, (
                "call_assistant did not produce a result nor tool_calls"
            )
            assert tool_called, "No tool was called"
        return result


# %%
import litellm

from effectful.handlers.llm.evaluation import UnsafeEvalProvider

litellm._turn_on_debug()

code_adapt = CodeAdapt(model="gpt-4o")
with handler(LiteLLMProvider(model="gpt-4o")), handler(UnsafeEvalProvider()):
    res = codeadapt("generate_paragraph")
    print(res)

# %%



