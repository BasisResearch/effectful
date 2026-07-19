import argparse
import dataclasses
import os
import urllib.parse

import requests
from tenacity import stop_after_attempt

from effectful.handlers.llm import Agent, Template, Tool
from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled


@Tool.define
def search_web(query: str) -> str:
    """Search Wikipedia for a topic and return a summary. The query can be a topic name or a natural language question."""
    search_url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "format": "json",
        }
    )
    search_data = requests.get(
        search_url, headers={"User-Agent": "effectful-example/1.0"}
    ).json()
    results = search_data.get("query", {}).get("search", [])
    if not results:
        raise ValueError(f"No results found for: {query}")
    title = results[0]["title"]

    summary_url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "format": "json",
        }
    )
    summary_data = requests.get(
        summary_url, headers={"User-Agent": "effectful-example/1.0"}
    ).json()
    page = next(iter(summary_data["query"]["pages"].values()))
    extract = page.get("extract", "No summary available.")
    url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"

    return f"# {title}\n\n{extract}\n\nSource: {url}"


@dataclasses.dataclass
class ChatBot(Agent):
    """Simple chat agent for testing history accumulation."""

    bot_name: str = dataclasses.field(default="ChatBot")

    @Template.define
    def send(self, user_input: str) -> str:
        """
        You are a friendly and helpful AI assistant named {self.bot_name}.
        If user input contains a question that you're not sure how to answer,
        consider using the web search tool to find the answer and include it in your response.

        The user writes:
        {user_input}
        """
        raise NotHandled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM-guided research agent with web search"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("EFFECTFUL_LLM_MODEL", ""),
        help="LLM model to use",
    )
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
        "--num-retries",
        type=int,
        default=4,
        help="Number of retries for malformed LLM output",
    )
    args = parser.parse_args()

    chatbot = ChatBot(bot_name=args.name)
    provider = LiteLLMProvider(model=args.model)

    with (
        handler(provider),
        handler(RetryLLMHandler(stop=stop_after_attempt(args.num_retries))),
    ):
        if args.interactive:
            while True:
                print(chatbot.send(input("You: ")))
        else:
            print(chatbot.send("Hi! Can you tell me about the Statue of Liberty?"))
            print(chatbot.send("Who designed it?"))
            print(chatbot.send("What about the speed of light? How fast is it?"))
