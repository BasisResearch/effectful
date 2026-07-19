"""Conversational chat agent with persistent history.

Demonstrates:
- An Agent subclass with automatic conversation history (Agent.__history__)
- Instance attributes available in prompts via {self.bot_name}
- Follow-up questions resolved from earlier turns via accumulated context
- An optional interactive REPL mode
"""

import argparse
import dataclasses

from effectful.handlers.llm import Agent, Template


@dataclasses.dataclass
class ChatBot(Agent):
    """Conversational agent that remembers the conversation so far."""

    bot_name: str = dataclasses.field(default="ChatBot")

    @Template.define
    def send(self, user_input: str) -> str:
        """
        You are a friendly and helpful AI assistant named {self.bot_name}.

        The user writes:
        {user_input}
        """


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        type=str,
        default="Chatty McChatface",
        help="The name of the chatbot",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode, allowing multiple back-and-forth messages",
    )
    parser.add_argument(
        "--messages",
        type=str,
        nargs="+",
        metavar="MESSAGE",
        default=[
            "Hi! Can you tell me about the Statue of Liberty?",
            "Who designed it?",
            "What about the speed of light? How fast is it?",
        ],
        help="The sequence of user messages to send in non-interactive mode",
    )
    args = parser.parse_args()

    chatbot = ChatBot(bot_name=args.name)

    if args.interactive:
        while True:
            print(chatbot.send(input("You: ")))
    else:
        for message in args.messages:
            print(chatbot.send(message))


if __name__ == "__main__":
    main()
