import functools

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import (
    LiteLLMProvider,
    call_user,
    compute_response,
)
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled


class Agent:
    def __init__(self):
        self.state = []  # persist the list of messages

    @defop
    @staticmethod
    def current_agent() -> "Agent | None":
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
                        call_user: self._format_model_input,
                        compute_response: self._compute_response,
                    }
                ):
                    return template(self, *args, **kwargs)

            setattr(cls, method_name, wrapper)

    def _format_model_input(self, template, other, *args, **kwargs):
        # update prompt with previous list of messages
        prompt = fwd()
        if Agent.current_agent() is self:
            assert self is other
            self.state.extend(prompt)
            prompt = self.state
        return prompt

    def _compute_response(self, *args, **kwargs):
        # save response into persisted state
        response = fwd()
        if Agent.current_agent() is self:
            self.state.append(response.choices[0].message.model_dump())
        return response


if __name__ == "__main__":

    class ChatBot(Agent):
        @Template.define
        def send(self, user_input: str) -> str:
            """User writes: {user_input}"""
            raise NotHandled

    provider = LiteLLMProvider()
    chatbot = ChatBot()

    with handler(provider):
        print(chatbot.send("Hi!, how are you? I am in france."))
        print(chatbot.send("Remind me again, where am I?"))
