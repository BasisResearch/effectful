import dataclasses
import inspect

import langfuse
import pydantic

from effectful.handlers.llm.harness.hooks import call_tool, completion
from effectful.handlers.llm.harness.serialization import DecodedToolCall
from effectful.handlers.llm.types import Encodable, Template
from effectful.internals.unification import nested_type
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


@dataclasses.dataclass(frozen=True)
class LangfuseTracer(ObjectInterpretation):
    """Traces Tool, Template, and completion calls with Langfuse.

    Compose with a provider via :func:`~effectful.ops.semantics.handler`
    to add tracing::

        with handler(provider), handler(LangfuseTracer()):
            print(limerick(theme))
    """

    client: langfuse.Langfuse = dataclasses.field(default_factory=langfuse.get_client)

    @implements(completion)
    def completion(self, *args, **kwargs):
        messages = kwargs.get("messages")
        if kwargs.get("tools") is not None:
            gen_input = {"tools": kwargs["tools"], "messages": messages}
        else:
            gen_input = messages

        model_parameters = {
            k: kwargs[k]
            for k in ("tool_choice", "temperature", "max_tokens", "top_p")
            if kwargs.get(k) is not None
        }
        metadata = {}
        response_format = kwargs.get("response_format")
        if response_format is not None:
            metadata["response_format"] = (
                response_format.model_json_schema()
                if isinstance(response_format, type)
                and issubclass(response_format, pydantic.BaseModel)
                else response_format
            )
        with self.client.start_as_current_observation(
            as_type="generation",
            name="completion",
            input=gen_input,
            model_parameters=model_parameters or None,
            metadata=metadata or None,
        ) as gen:
            response = fwd()
            gen.update(model=response.model)
            usage = getattr(response, "usage", None)
            if usage is not None:
                gen.update(
                    usage_details={
                        "input": usage.prompt_tokens,
                        "output": usage.completion_tokens,
                        "total": usage.total_tokens,
                    }
                )
            gen.update(output=response.choices[0].message)
            return response

    @implements(call_tool)
    def call_tool(self, tool_call: DecodedToolCall):
        input = {
            name: pydantic.TypeAdapter(
                Encodable[nested_type(value).value]  # type: ignore[misc]
            ).dump_python(value, mode="json", context={})
            for name, value in tool_call.bound_args.arguments.items()
        }
        with self.client.start_as_current_observation(
            as_type="tool",
            name=tool_call.name,
            input=input,
            metadata={"tool_call_id": tool_call.id},
        ) as obs:
            message, result, is_final = fwd()
            obs.update(output=message["content"], metadata={"is_final": is_final})
            return message, result, is_final

    @implements(Template.__apply__)
    def call_template(self, template: Template, *args, **kwargs):
        bound = inspect.signature(template).bind(*args, **kwargs)
        bound.apply_defaults()
        agent_input = {
            name: pydantic.TypeAdapter(
                Encodable[nested_type(value).value]  # type: ignore[misc]
            ).dump_python(value, mode="json", context={})
            for name, value in bound.arguments.items()
        }
        with self.client.start_as_current_observation(
            as_type="agent", name=template.__name__, input=agent_input
        ) as obs:
            result = fwd()
            obs.update(
                output=pydantic.TypeAdapter(
                    Encodable[nested_type(result).value]  # type: ignore[misc]
                ).dump_python(result, mode="json", context={})
            )
            return result
