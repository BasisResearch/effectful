"""Think-Act-Observe agent: structured chain-of-thought with optional tool use.

Demonstrates:
- Agent mixin for persistent conversation history (the LLM sees its own prior
  reasoning across steps)
- Structured output with an AgentThought dataclass carrying an is_final flag
- A think -> act -> observe loop that continues until the agent is done
- Pattern-matching action dispatch: reason straight to an answer, or call a
  web-search tool when a fact is missing
"""

import argparse
import dataclasses
import enum
import urllib.parse

import requests

from effectful.handlers.llm import Agent, Template, Tool

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
        ({query}) and your own prior reasoning in the conversation history, think
        about what to do next: either `search_the_web` for a fact you are missing,
        or `answer` once you can conclude. Set is_final=true when your action is
        the final answer.
        """

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

    def run(self, query: str, max_steps: int = 5) -> str:
        result = ""
        for i in range(max_steps):
            thought = self.think(query)
            print(f"  [step {i + 1}] {thought.thinking}")
            result = self._act(thought.action, thought.action_input)
            self.observe(str(thought.action), thought.action_input, result)
            if thought.is_final:
                break
        return result

    def _act(self, action: AgentAction, action_input: str) -> str:
        match action:
            case AgentAction.search_the_web:
                return search_web(action_input)
            case AgentAction.answer:
                return action_input


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum think-act-observe steps per problem",
    )
    parser.add_argument(
        "--problem",
        dest="problems",
        metavar="PROBLEM",
        nargs="+",
        default=[
            (
                "A farmer has 17 sheep. All but 9 run away. "
                "Then he buys 5 more. How many sheep does he have now?"
            ),
            "What year was the Eiffel Tower completed, and how tall is it?",
        ],
        help=(
            "One or more problems to solve (a pure-reasoning puzzle and a "
            "web-lookup question by default)"
        ),
    )
    args = parser.parse_args()

    # By default, one puzzle the agent can reason through with no tools, and one
    # that needs a web lookup -- the same loop handles both via its action
    # dispatch.
    for problem in args.problems:
        agent = TAOAgent()
        print(f"\nProblem: {problem}")
        answer = agent.run(problem, max_steps=args.max_steps)
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
