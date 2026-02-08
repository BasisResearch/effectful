"""Chat agent with embedding-based memory.

Demonstrates:
- A stateful chat agent that maintains conversation history
- Embedding-based retrieval of relevant past context
- Simple in-memory vector store with L2 distance
"""

import argparse
import dataclasses
import os

import litellm
import numpy as np

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import LiteLLMProvider
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def get_embedding(text: str) -> np.ndarray:
    """Get an embedding vector for the given text using litellm."""
    response = litellm.embedding(model="text-embedding-ada-002", input=text)
    return np.array(response.data[0]["embedding"], dtype=np.float32)


def find_closest(
    index: list[tuple[str, np.ndarray]], phrase: str
) -> tuple[str, float] | None:
    """Find the closest entry in the index to the given phrase."""
    if not index:
        return None
    phrase_embedding = get_embedding(phrase)

    def dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(((a - b) ** 2).sum())

    return min(
        ((msg, dist(embedding, phrase_embedding)) for msg, embedding in index),
        key=lambda elt: elt[1],
    )


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------


@Template.define
def respond_to_user(
    user_message: str, relevant_context: str, prev_messages: str
) -> str:
    """Given the user wrote: {user_message}
    Continue the conversation.
    The last few messages were: {prev_messages}
    Older relevant context: {relevant_context}"""
    raise NotHandled


# ---------------------------------------------------------------------------
# Chat agent
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ChatAgent:
    """A chat agent that compresses old messages into an embedding index."""

    history: list[dict[str, str]] = dataclasses.field(default_factory=list)
    index: list[tuple[str, np.ndarray]] = dataclasses.field(default_factory=list)

    def _compress(self):
        """Move the oldest pair of messages into the embedding index."""
        oldest_pair, self.history = self.history[:2], self.history[2:]
        text = "\n".join(m["content"] for m in oldest_pair)
        self.index.append((text, get_embedding(text)))

    def _find_relevant(self, query: str) -> str:
        result = find_closest(self.index, query)
        return result[0] if result else "No relevant context."

    def chat(self, user_input: str):
        relevant = self._find_relevant(user_input)
        prev_messages = "\n".join(
            f"{m['author']}: {m['content']}" for m in self.history
        )
        response = respond_to_user(user_input, relevant, prev_messages)
        self.history.append({"author": "user", "content": user_input})
        self.history.append({"author": "agent", "content": response})
        if len(self.history) > 6:
            self._compress()
        print(f"user: {user_input}")
        print(f"agent: {response}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chat agent with embedding-based memory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lm_studio/zai-org/glm-4.7-flash",
        help="LLM model to use",
    )
    args = parser.parse_args()

    if args.model.startswith("lm_studio/"):
        assert os.environ.get("LM_STUDIO_API_BASE")
    elif args.model.startswith("gpt-"):
        assert os.environ.get("OPENAI_API_KEY")
    elif args.model.startswith("claude-"):
        assert os.environ.get("ANTHROPIC_API_KEY")

    agent = ChatAgent()

    provider = LiteLLMProvider(model=args.model)
    with handler(provider):
        agent.chat("Hello! How are you doing?")
        agent.chat("Lovely! I'm having a great day.")
        agent.chat("What is the capital of France?")
        agent.chat("I didn't know that! That's amazing!")
