import functools
from collections import OrderedDict

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import (
    LiteLLMProvider,
    Message,
    MessageSequence,
)
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled


class Agent:
    __history__: OrderedDict[str, Message]

    def __init__(self):
        self.__history__ = OrderedDict()  # persist the list of messages

    def __init_subclass__(cls):
        for method_name in dir(cls):
            template = getattr(cls, method_name)
            if not isinstance(template, Template):
                continue

            @functools.wraps(template)
            def wrapper(self, *args, **kwargs):
                with handler(MessageSequence(self.state)):
                    return template(self, *args, **kwargs)

            setattr(cls, method_name, wrapper)


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
