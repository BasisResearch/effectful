import functools
from typing import Optional

from effectful.handlers.llm import Template
from effectful.handlers.llm.providers import compute_response, format_model_input
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import defop


class Agent:
    '''When inheriting from Agent, Template-valued methods will have the
    previous history of the conversation injected prior to their prompts.

    Example:

    >>> class ConversationAgent(Agent):
    ...     @Template.define
    ...     def respond(self, message: str) -> str:
    ...         """Continue the conversation in response to the message '{message}'"""
    ...         raise NotImplementedError

    Any calls to `agent.format` will have the previous conversation history in their context.

    '''

    def __init__(self):
        self.state = []

    @defop
    @staticmethod
    def current_agent() -> Optional["Agent"]:
        return None

    def __init_subclass__(cls):
        for method_name in dir(cls):
            template = getattr(cls, method_name)
            if not isinstance(template, Template):
                continue

            @functools.wraps(template)
            def wrapper(self, *args, **kwargs):
                with handler(
                    {
                        Agent.current_agent: lambda: self,
                        format_model_input: self._format_model_input,
                        compute_response: self._compute_response,
                    }
                ):
                    return template(self, *args, **kwargs)

            setattr(cls, method_name, wrapper)

    def _format_model_input(self, template, other, *args, **kwargs):
        prompt = fwd()
        if Agent.current_agent() is self:
            assert self is other
            prompt = self.state + prompt
        return prompt

    def _compute_response(self, *args, **kwargs):
        response = fwd()
        if Agent.current_agent() is self:
            self.state += response.output
        return response
