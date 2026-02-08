"""Multi-agent Taboo word guessing game.

Demonstrates:
- Two ``Agent`` instances with independent conversation histories
- Inter-agent communication via plain function calls
- Each agent has a different persona and goal
- ``Agent.__history__`` keeps each agent's context isolated
"""

import argparse
import dataclasses
import enum
import os

from effectful.handlers.llm import Agent, Template, Tool
from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


class Confidence(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclasses.dataclass(frozen=True)
class Guess:
    guess: str
    confidence: Confidence


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Hinter(Agent):
    """Agent that gives hints about a secret word without saying it."""

    secret_word: str = dataclasses.field(default="")
    taboo_words: list[str] = dataclasses.field(default_factory=list)

    @Tool.define
    def is_taboo(self, hint: str) -> bool:
        """Check if the given hint contains any taboo words or the secret word."""
        lowered_hint = hint.lower()
        if self.secret_word.lower() in lowered_hint:
            return True
        for taboo in self.taboo_words:
            if taboo.lower() in lowered_hint:
                return True
        return False

    @Template.define
    def give_hint(self, guesser_response: str) -> str:
        """You are playing a word guessing game. You must help the guesser
        figure out the secret word by giving creative hints.

        RULES:
        - You MUST NOT say the secret word: {self.secret_word}
        - You MUST NOT use any of these taboo words: {self.taboo_words}
        - Give a single, concise hint (one sentence)
        - Review conversation history to avoid repeating hints
        - Use the is_taboo tool to check if your hint is valid

        The guesser's last response was: {guesser_response}
        """
        raise NotHandled


class Guesser(Agent):
    """Agent that tries to guess the secret word from hints."""

    @Template.define
    def make_guess(self, hint: str) -> Guess:
        """You are playing a word guessing game. Based on the hints you've
        received, guess the secret word.

        Latest hint: {hint}

        Review the conversation history for all previous hints.
        Make your best guess.
        """
        raise NotHandled


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------


def play_taboo(
    secret_word: str,
    taboo_words: list[str],
    max_rounds: int = 5,
) -> bool:
    """Play a round of Taboo between a hinter and a guesser."""
    hinter = Hinter(secret_word=secret_word, taboo_words=taboo_words)
    guesser = Guesser()

    guesser_response = "I'm ready to guess!"

    for round_num in range(max_rounds):
        # Hinter gives a hint
        hint = hinter.give_hint(guesser_response)
        print(f"  [round {round_num}] Hinter: {hint}")

        # Guesser tries to guess
        guess = guesser.make_guess(hint)
        guesser_response = f"I guessed '{guess.guess}' ({guess.confidence})"
        print(f"  [round {round_num}] Guesser: {guess.guess} ({guess.confidence})")

        if guess.guess.lower().strip() == secret_word.lower():
            print(f"  Correct! Guessed in {round_num} round(s).")
            return True

    print(f"  Failed to guess '{secret_word}' in {max_rounds} rounds.")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-agent Taboo word guessing game")
    parser.add_argument(
        "--model",
        type=str,
        default="lm_studio/zai-org/glm-4.7-flash",
        help="LLM model to use",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum rounds per game",
    )
    args = parser.parse_args()

    if args.model.startswith("lm_studio/"):
        assert os.environ.get("LM_STUDIO_API_BASE")
    elif args.model.startswith("gpt-"):
        assert os.environ.get("OPENAI_API_KEY")
    elif args.model.startswith("claude-"):
        assert os.environ.get("ANTHROPIC_API_KEY")

    games = [
        ("piano", ["music", "keys", "instrument", "play"]),
        ("volcano", ["lava", "eruption", "mountain", "hot"]),
    ]

    provider = LiteLLMProvider(model=args.model)

    with handler(provider), handler(RetryLLMHandler(num_retries=3)):
        for secret, taboo in games:
            print(f"\nGame: '{secret}' (taboo: {taboo})")
            play_taboo(secret, taboo, max_rounds=args.max_rounds)
