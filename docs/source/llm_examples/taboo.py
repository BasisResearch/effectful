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

from effectful.handlers.llm import Agent, Template, Tool

# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


class Confidence(enum.StrEnum):
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

    secret_word: str
    taboo_words: list[str]

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

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum rounds per game",
    )
    parser.add_argument(
        "--secret-word",
        type=str,
        default=None,
        metavar="WORD",
        help="Secret word to guess (used with --taboo-words for a single custom game)",
    )
    parser.add_argument(
        "--taboo-words",
        nargs="+",
        type=str,
        default=None,
        metavar="WORD",
        help="Taboo words the hinter may not say (used with --secret-word)",
    )
    args = parser.parse_args()

    if (args.secret_word is None) != (args.taboo_words is None):
        parser.error("--secret-word and --taboo-words must be given together")

    if args.secret_word is not None:
        games = [(args.secret_word, args.taboo_words)]
    else:
        games = [
            ("piano", ["music", "keys", "instrument", "play"]),
            ("volcano", ["lava", "eruption", "mountain", "hot"]),
        ]

    for secret, taboo in games:
        print(f"\nGame: '{secret}' (taboo: {taboo})")
        play_taboo(secret, taboo, max_rounds=args.max_rounds)


if __name__ == "__main__":
    main()
