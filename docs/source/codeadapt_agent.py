"""CodeAdapt as an Agent: in-context learning across a conversation.

This is the Agent-method variant of ``codeadapt.py``.  Instead of a free
function, the task is a :class:`~effectful.handlers.llm.Template` method on an
:class:`~effectful.handlers.llm.Agent` subclass, so each call accumulates
message history on the instance and the model can take advantage of in-context
learning across calls.

The synthesized function is a drop-in syntactic replacement for the method body
-- it keeps ``self`` in its signature -- and the worked examples in the method's
docstring are run as doctests against that synthesized function (calls on
freshly constructed agents are rerouted to it rather than re-invoking the model).
"""

import argparse
import contextlib
import os

import tenacity

from effectful.handlers.llm import Agent, Template
from effectful.handlers.llm.completions import (
    LangfuseTracer,
    LexicalReaders,
    LiteLLMProvider,
    PythonRepl,
    RetryLLMHandler,
    SynthesizeAndCall,
    TerminalRenderer,
)
from effectful.handlers.llm.evaluation import UnsafeEvalProvider
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled


class CodeAdaptAgent(Agent):
    """
    You are a careful problem solver and an expert Python programmer. You answer by
    writing code, not by reasoning in prose alone: problems that are error-prone to
    work out by hand are often easy to brute-force or verify with a short program.

    You MUST use the ``submit_solution`` tool to give your final answer,
    the harness will not accept a final answer in direct text.

    You can use whatever other tools are available to develop your solution,
    and refine incorrect attempts given feedback from failures of ``submit_solution``.
    """

    @Template.define
    def countdown_reachable(self, numbers: list[int], target: int) -> bool:
        """In the Countdown numbers game, decide whether {target} can be made from
        {numbers}, using each number exactly once and combining them with + - * /
        (every intermediate division must come out exact).

        >>> agent = CodeAdaptAgent()
        >>> agent.countdown_reachable([2, 3, 5], 11)
        True
        >>> agent.countdown_reachable([1, 1], 5)
        False
        >>> agent.countdown_reachable([4, 7, 8, 9], 100)
        True
        >>> agent.countdown_reachable([5, 5, 5], 3)
        False
        """
        raise NotHandled


def main(args: argparse.Namespace) -> None:
    if args.task == "countdown":
        agent = CodeAdaptAgent()
        # Fresh examples (none appear in the docstring doctests), each paired with its
        # known-correct answer so we can validate the agent's output.
        test_examples: list[tuple[list[int], int, bool]] = [
            ([3, 6, 25, 50], 147, True),  # (50 - 25) * 6 - 3
            ([1, 2, 3, 4], 24, True),  # 1 * 2 * 3 * 4
            ([2, 4, 8], 9, False),  # all-even operands can never reach an odd target
        ]
        for numbers, target, expected in test_examples:
            print(f"Testing countdown_reachable({numbers}, {target})...")
            answer = agent.countdown_reachable(numbers, target)
            status = "OK" if answer == expected else "WRONG"
            print(
                f"[{status}] reach {target} from {numbers}: {answer} (expected {expected})"
            )
            assert answer == expected, (
                f"countdown_reachable({numbers}, {target}) = {answer}, expected {expected}"
            )
    else:
        raise ValueError(f"Unknown task {args.task}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CodeAdapt agent: solve reasoning tasks by writing code"
    )
    parser.add_argument(
        "--task",
        choices=("countdown",),
        default="countdown",
        help="Which problem to solve",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("EFFECTFUL_LLM_MODEL", ""),
        help="LLM model to use",
    )
    parser.add_argument(
        "--num-retries",
        type=int,
        default=5,
        help="Number of retries for malformed/failing LLM output",
    )
    parser.add_argument(
        "--langfuse",
        action="store_true",
        help="Whether to log LLM calls and metadata to Langfuse",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Live-render the streaming message history in the terminal",
    )
    args = parser.parse_args()
    with (
        handler(LiteLLMProvider(model=args.model, tool_choice="required")),
        handler(TerminalRenderer()) if args.render else contextlib.nullcontext(),
        handler(UnsafeEvalProvider()),
        handler(PythonRepl()),
        handler(SynthesizeAndCall()),
        handler(RetryLLMHandler(stop=tenacity.stop_after_attempt(args.num_retries))),
        handler(LexicalReaders()),
        handler(LangfuseTracer()) if args.langfuse else contextlib.nullcontext(),
    ):
        main(args)
