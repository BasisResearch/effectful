"""Think-Act-Observe chain-of-thought agent.

Demonstrates:
- ``Agent`` mixin for persistent conversation history
- Structured output with Pydantic models (``AgentThought``)
- A think → act → observe reasoning loop
- Pattern matching for action dispatch
"""

import argparse
import dataclasses
import enum
import os
import urllib.parse

import requests

from effectful.handlers.llm import Agent, Template, Tool
from effectful.handlers.llm.completions import (
    LiteLLMProvider,
    RetryLLMHandler,
)
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Search tool
# ---------------------------------------------------------------------------


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
        return f"No results found for: {query}"
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


# ---------------------------------------------------------------------------
# Structured output types
# ---------------------------------------------------------------------------


class AgentAction(enum.StrEnum):
    search_the_web = "search_the_web"
    calculate = "calculate"
    answer = "answer"


@dataclasses.dataclass(frozen=True)
class AgentThought:
    thinking: str
    action: AgentAction
    action_input: str
    is_final: bool


# ---------------------------------------------------------------------------
# TAO Agent
# ---------------------------------------------------------------------------


class TAOAgent(Agent):
    """Think-Act-Observe agent that reasons step by step."""

    @Template.define
    def think(self, query: str) -> AgentThought:
        """You are an AI assistant solving a problem. Based on the user's query
        ({query}) and prior conversation context, think about what action to
        take next.
        """
        raise NotHandled

    @Template.define
    def observe(self, action: str, action_input: str, action_result: str) -> str:
        """You are an observer. Provide a concise, objective observation of this result.

        Action: {action}
        Action input: {action_input}
        Action result: {action_result}

        <instructions>
        Do not make decisions, just describe what you see.
        </instructions>
        """
        raise NotHandled

    def run(self, query: str, max_steps: int = 5) -> str:
        result = ""
        for _ in range(max_steps):
            thought = self.think(query)
            result = self._act(thought.action, thought.action_input)
            self.observe(str(thought.action), thought.action_input, result)
            if thought.is_final:
                break
        return result

    def _act(self, action: AgentAction, action_input: str) -> str:
        match action:
            case AgentAction.search_the_web:
                return search_web(action_input)
            case AgentAction.calculate:
                try:
                    return action_input  # eval(action_input))  # noqa: S307
                except Exception as e:
                    return str(e)
            case AgentAction.answer:
                return action_input


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TAO chain-of-thought agent")
    parser.add_argument(
        "--model",
        type=str,
        default="lm_studio/zai-org/glm-4.7-flash",
        help="LLM model to use",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum number of steps before giving up",
    )
    parser.add_argument(
        "--num-retries",
        type=int,
        default=5,
        help="Number of retries for malformed LLM output",
    )
    args = parser.parse_args()

    if args.model.startswith("lm_studio/"):
        assert os.environ.get("LM_STUDIO_API_BASE")
    elif args.model.startswith("gpt-"):
        assert os.environ.get("OPENAI_API_KEY")
    elif args.model.startswith("claude-"):
        assert os.environ.get("ANTHROPIC_API_KEY")

    provider = LiteLLMProvider(model=args.model)

    agent = TAOAgent()

    with handler(provider), handler(RetryLLMHandler(num_retries=args.num_retries)):
        answer = agent.run(
            "How many tennis balls would fill an Olympic swimming pool?",
            max_steps=args.max_steps,
        )
        print("Answer:", answer)
