"""Learn writing guidelines from one piece of feedback, by textual gradients.

Two jokes are written under a shared ``joke_guidelines`` parameter and composed
into an email under a ``formatting_guidelines`` parameter; a single sentence of
feedback on the *email* is then backpropagated by `textgrad.TextGradOptimizer`:
an internal skill splits the feedback between the email's inputs (the two joke
results and the formatting parameter), the jokes' shares are refined onward to
the joke parameter, and accumulation rewrites both parameters in place -- any
later call passing the same boxes writes under the improved guidelines. (The
boxes live in memory here; to persist them across runs, host them as dataclass
fields of a persistent `~effectful.handlers.llm.types.Agent` and the harness's
``SQLitePersister`` checkpoints them with no extra code.)

Demonstrates:
- `textgrad.Parameter`: a mutable box passed directly as a skill argument; the
  recording handler notes the use and splices the wrapped value into the prompt
- one `textgrad.TextGradOptimizer` in both autograd roles: installed as a
  handler it is the tape recording plain skill calls (dataflow edges recovered
  from call nesting and argument identity); afterwards, *outside* its own
  handler scope, ``step`` backpropagates and accumulates, returning the
  walked graph and the routed per-node feedback for inspection
- gradients accumulating on one box (``joke_guidelines``) from two uses

Run with::

    python -m effectful.handlers.llm.harness \\
        docs/source/llm_examples/optimization/guidelines.py \\
        --model gpt-4o-mini --tool-choice none

``--tool-choice none`` is cosmetic but keeps the traced graph minimal: the two
skills below are module-level, so each sees the other in lexical scope and a
small model sometimes calls it as a tool mid-joke despite being told not to.
The recording handler notes such calls faithfully as extra child nodes (and the backward
pass prunes them -- they reach no parameter), but the demo reads better without
them.
"""

import argparse

from docs.source.llm_examples.optimization.textgrad import (
    Parameter,
    TextGradOptimizer,
)
from effectful.handlers.llm import Skill
from effectful.ops.semantics import handler

joke_guidelines = Parameter(
    "No specific guidelines yet.",
    description="Standing guidelines for writing a good joke.",
)
formatting_guidelines = Parameter(
    "No specific guidelines yet.",
    description="Standing guidelines for the layout and typography of an email.",
)


# The skills declare plain ``str`` parameters: the recording handler unwraps a
# `Parameter` box to its value before the prompt is built, so the model sees
# only the text.
@Skill.define
def joke_writer(topic: str, guidelines: str) -> str:
    """Write a short joke about the following topic: "{topic}".

    Follow these guidelines:
    <guidelines>
    {guidelines}
    </guidelines>

    Do not use any tools. Return only the joke.
    """


@Skill.define
def email_writer(joke_1: str, joke_2: str, guidelines: str) -> str:
    """Write a short email to Jane Doe containing the following two jokes, verbatim:

    Joke 1: {joke_1}

    Joke 2: {joke_2}

    Follow these email formatting guidelines:
    <guidelines>
    {guidelines}
    </guidelines>

    Do not use any tools. Return only the email.
    """


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feedback",
        type=str,
        default=(
            "Jokes about cats should always be about Siamese cats. "
            "Jokes about programmers should be about coffee. "
            "The email should include a title for each joke."
        ),
        help="Natural-language feedback on the final email",
    )
    args = parser.parse_args()

    print(f"joke guidelines (before):       {joke_guidelines.value}")
    print(f"formatting guidelines (before): {formatting_guidelines.value}\n")

    # Forward pass: ordinary calls recorded by the optimizer-as-handler.
    # Passing the boxes (not their .value) records the parameter uses; passing
    # the jokes' returned strings onward is what wires the joke calls in as
    # the email's children.
    optimizer = TextGradOptimizer()
    with handler(optimizer):
        cat_joke = joke_writer(topic="cats", guidelines=joke_guidelines)
        prog_joke = joke_writer(topic="programmers", guidelines=joke_guidelines)
        email = email_writer(
            joke_1=cat_joke, joke_2=prog_joke, guidelines=formatting_guidelines
        )

    print(f"cat joke:  {cat_joke}\n")
    print(f"prog joke: {prog_joke}\n")
    print(f"email:\n{email}\n")
    print(f"feedback:  {args.feedback}\n")

    # Backward + accumulate, outside the optimizer's own handler scope so its
    # internal skill calls do not join the graph they are optimizing. `grads`
    # is the backward pass's ephemeral per-node routed feedback, returned for
    # display; the persistent gradients live on the Parameter boxes.
    graph, grads = optimizer.step(args.feedback)

    def show(node, depth=0):
        pad = "  " * depth
        print(f"{pad}{node.skill_name}: feedback={grads.get(node, [])}")
        for name, box in node.parameters:
            print(f"{pad}  param {name}: gradients={box.gradients}")
        for child in node.children:
            show(child, depth + 1)

    print("optimizer graph:")
    show(graph)

    print(f"\njoke guidelines (after):       {joke_guidelines.value}")
    print(f"formatting guidelines (after): {formatting_guidelines.value}")


if __name__ == "__main__":
    main()
