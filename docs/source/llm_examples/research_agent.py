"""Research agent with web search.

Demonstrates:
- ``@defop`` + ``ObjectInterpretation`` to define a pluggable web search effect
- ``@Template.define`` for LLM-implemented answer/refine/judge templates
- Handler composition: stacking a search provider alongside an LLM provider
- Iterative refinement loop: answer → judge → refine → judge → ...
"""

import argparse
import os
import urllib.parse

import requests

from effectful.handlers.llm import Template, Tool
from effectful.handlers.llm.completions import (
    LiteLLMProvider,
)
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Search effect + handler
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
# Templates (auto-capture `search_web` from lexical scope)
# ---------------------------------------------------------------------------


@Template.define
def answer_question(question: str) -> str:
    """Acting as a research assistant that can search the web,
    construct an answer to the user's question: {question}."""
    raise NotHandled


@Template.define
def refine_answer(question: str, answer: str) -> str:
    """Acting as a research assistant that can search the web,
    given the user's original question ({question}),
    refine this previous answer: {answer}."""
    raise NotHandled


@Template.define
def is_question_answered(question: str, answer: str) -> bool:
    """Acting as a research assistant, decide if the user's question
    ({question}) is appropriately answered by: {answer}.
    Respond only true or false."""
    raise NotHandled


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def research_agent(question: str, max_attempts: int = 3) -> str:
    """Answer a question, iteratively refining until satisfactory."""
    answer = answer_question(question)
    for _ in range(max_attempts):
        if is_question_answered(question, answer):
            break
        answer = refine_answer(question, answer)
    return answer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM-guided research agent with web search"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lm_studio/zai-org/glm-4.7-flash",
        help="LLM model to use",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What is the meaning of life?",
        help="The question to research",
    )
    args = parser.parse_args()

    if args.model.startswith("lm_studio/"):
        assert os.environ.get("LM_STUDIO_API_BASE")
    elif args.model.startswith("gpt-"):
        assert os.environ.get("OPENAI_API_KEY")
    elif args.model.startswith("claude-"):
        assert os.environ.get("ANTHROPIC_API_KEY")

    provider = LiteLLMProvider(model=args.model)

    with handler(provider):
        result = research_agent(args.question)
        print(result)
