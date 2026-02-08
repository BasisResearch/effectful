"""Supervisor quality-control wrapper.

Demonstrates:
- Wrapping an agent's output with a quality-control check
- Using one ``Template`` to judge another's output
- Retry loop driven by LLM-based evaluation
"""

import argparse
import dataclasses
import os
import urllib.parse

import requests

from effectful.handlers.llm import Agent, Template, Tool
from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
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
# Structured output for quality judgment
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class QualityJudgment:
    is_acceptable: bool
    feedback: str


# ---------------------------------------------------------------------------
# Research agent
# ---------------------------------------------------------------------------


class Researcher(Agent):
    """Agent that answers research questions using web search."""

    @Template.define
    def answer(self, question: str) -> str:
        """You are a research assistant. Answer the following question using
        the search tool to find accurate information.

        Question: {question}
        """
        raise NotHandled


# ---------------------------------------------------------------------------
# Supervisor (quality judge)
# ---------------------------------------------------------------------------


@Template.define
def judge_quality(question: str, answer: str) -> QualityJudgment:
    """You are a strict quality reviewer. Evaluate whether this answer
    adequately addresses the question with accurate, specific information.

    Question: {question}
    Answer: {answer}

    An answer is acceptable if it contains specific facts (names, dates,
    numbers) relevant to the question. Vague or generic answers should
    be rejected.
    """
    raise NotHandled


# ---------------------------------------------------------------------------
# Supervised agent loop
# ---------------------------------------------------------------------------


def supervised_research(question: str, max_retries: int = 3) -> str:
    """Answer a question with quality-control supervision.

    The researcher agent answers, the supervisor judges quality,
    and if rejected the researcher tries again with feedback.
    """
    researcher = Researcher()

    for attempt in range(max_retries + 1):
        answer = researcher.answer(question)
        judgment = judge_quality(question, answer)

        if judgment.is_acceptable:
            print(f"[supervisor] Accepted on attempt {attempt + 1}")
            return answer

        print(f"[supervisor] Rejected attempt {attempt + 1}: {judgment.feedback}")

    print("[supervisor] Returning best effort after max retries")
    return answer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supervised research agent with quality control"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lm_studio/zai-org/glm-4.7-flash",
        help="LLM model to use",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of supervisor rejections before accepting",
    )
    args = parser.parse_args()

    if args.model.startswith("lm_studio/"):
        assert os.environ.get("LM_STUDIO_API_BASE")
    elif args.model.startswith("gpt-"):
        assert os.environ.get("OPENAI_API_KEY")
    elif args.model.startswith("claude-"):
        assert os.environ.get("ANTHROPIC_API_KEY")

    provider = LiteLLMProvider(model=args.model)

    with handler(provider), handler(RetryLLMHandler(num_retries=3)):
        result = supervised_research(
            "What year was the Eiffel Tower completed and how tall is it?",
            max_retries=args.max_retries,
        )
        print(f"\nFinal answer: {result}")
