import functools
from collections import OrderedDict

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import (
    LiteLLMProvider,
    Message,
<<<<<<< HEAD
    MessageSequence,
)
from effectful.ops.semantics import fwd, handler
=======
    get_message_sequence,
)
from effectful.ops.semantics import handler
>>>>>>> 419935553d10eb94c016a13652329f855579008e
from effectful.ops.types import NotHandled


class Agent:
<<<<<<< HEAD
    state: OrderedDict[str, Message]

    def __init__(self):
        self.state = OrderedDict()  # persist the list of messages
=======
    __history__: OrderedDict[str, Message]

    def __init__(self):
        self.__history__ = OrderedDict()  # persist the list of messages
>>>>>>> 419935553d10eb94c016a13652329f855579008e

    def __init_subclass__(cls):
        for method_name in dir(cls):
            template = getattr(cls, method_name)
            if not isinstance(template, Template):
                continue

            @functools.wraps(template)
            def wrapper(self, *args, **kwargs):
<<<<<<< HEAD
                with handler(MessageSequence(self.state)):
=======
                with handler({get_message_sequence: lambda: self.__history__}):
>>>>>>> 419935553d10eb94c016a13652329f855579008e
                    return template(self, *args, **kwargs)

            setattr(cls, method_name, wrapper)

<<<<<<< HEAD
    def _call_assistant(self, messages: list[Message], *args, **kwargs):
        for message in messages:
            self.state[message["id"]] = message

        # update state with message sequence
        response, tool_calls, result = fwd(list(self.state.values()), *args, **kwargs)

        self.state[response["id"]] = response

        return response, tool_calls, result

=======
>>>>>>> 419935553d10eb94c016a13652329f855579008e

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
