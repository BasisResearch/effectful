"""Research agent with web search and LLM quality control.

Demonstrates:
- @Tool.define web-search tool, auto-captured into templates from lexical scope
- An Agent subclass with persistent conversation history
- One Template judging another's output, returning a structured QualityJudgment
  (a bool plus written feedback)
- A feedback-driven refinement loop: answer -> judge -> refine -> judge -> ...
"""

import argparse
import dataclasses
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
# Structured output for quality judgment
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class QualityJudgment:
    is_acceptable: bool
    feedback: str


# ---------------------------------------------------------------------------
# Research agent (persistent history; search_web auto-captured from scope)
# ---------------------------------------------------------------------------


class Researcher(Agent):
    """Agent that answers research questions using web search, refining on feedback."""

    @Template.define
    def answer(self, question: str) -> str:
        """You are a research assistant. Use the search tool to find accurate,
        specific information, then answer the question: {question}"""

    @Template.define
    def refine(self, question: str, feedback: str) -> str:
        """A reviewer rejected your previous answer to the question ({question})
        with this feedback: {feedback}. Use the search tool as needed and provide
        an improved answer that addresses the feedback."""


# ---------------------------------------------------------------------------
# Supervisor (quality judge)
# ---------------------------------------------------------------------------


@Template.define
def judge_quality(question: str, answer: str) -> QualityJudgment:
    """You are a strict quality reviewer. Evaluate whether this answer adequately
    addresses the question with accurate, specific information.

    Question: {question}
    Answer: {answer}

    An answer is acceptable if it contains specific facts (names, dates, numbers)
    relevant to the question. Vague or generic answers should be rejected; when
    rejecting, explain in the feedback what is missing.
    """


# ---------------------------------------------------------------------------
# Supervised agent loop
# ---------------------------------------------------------------------------


def research_agent(question: str, max_retries: int = 3) -> str:
    """Answer a question, refining on supervisor feedback until it is acceptable."""
    researcher = Researcher()
    answer = researcher.answer(question)

    for attempt in range(1, max_retries + 1):
        judgment = judge_quality(question, answer)
        if judgment.is_acceptable:
            print(f"[supervisor] Accepted on attempt {attempt}")
            return answer
        print(f"[supervisor] Rejected attempt {attempt}: {judgment.feedback}")
        answer = researcher.refine(question, judgment.feedback)

    print("[supervisor] Returning best effort after max retries")
    return answer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        type=str,
        default="What year was the Eiffel Tower completed and how tall is it?",
        help="The question to research",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of supervisor rejections before returning best effort",
    )
    args = parser.parse_args()

    result = research_agent(args.question, max_retries=args.max_retries)
    print(f"\nFinal answer: {result}")


if __name__ == "__main__":
    main()
