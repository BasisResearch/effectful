"""Prompt optimization: generalization mode (optimize_anything A.3).

Optimize a system prompt so it works on instances the search never saw. Both a training
set and a validation set are supplied, which is what selects the paper's generalization
mode: search takes its feedback from the training instances, and the artifact that
survives has to carry to held-out ones. This is the mode GEPA and MIPROv2 operate in,
and the one the paper extends beyond prompts.

The artifact is a plain ``str``, the evaluator is itself a model call, and -- as in the
paper -- the prompt is optimized *for a cheaper model* (``--worker-model``) than the one
proposing it, here GPT-4.1-mini, which is A.3's target model too. In effectful "run this
call on a different model" is a scoped handler, so `library.worker` is the entire
mechanism.

The task is constrained writing -- an exact word count, an initial-letter rule, a banned
letter -- scored by deterministic Python, and that substitution needs stating plainly
rather than defending. AIME itself would have been the faithful choice and has ample
headroom: the paper measures GPT-4.1-mini at 46.67% on AIME 2025 from a generic prompt.
What ruled it out is that the problems cannot be embedded here -- the set is large, and
writing a dozen substitutes is not AIME. A substitute set also saturates: GPT-4.1-mini
scores 12/12 on hand-written counting and number-theory problems from the bare seed
prompt, which leaves the search nothing to climb. Constraint tracking is a task
these models do fail, the checker is exact, and the lever a better prompt supplies is
method -- count before answering, verify each constraint separately, revise once.
Nothing measured here transfers to a claim about AIME.

Demonstrates:
- Generalization mode: per-example Pareto objectives over the training instances, so a
  prompt that is best at *something* survives, and selection on a held-out set
- Side Information following the paper's design for this domain as far as the task
  allows -- the instance, the model's reasoning, what it produced, and a per-constraint
  account of what went wrong. A.3 also returns the ground-truth answer, which has no
  analogue here: constrained writing has no reference sentence, only constraints
- The proposer/target-model split as a scoped handler nested inside the harness's stack
- Partial credit as a search gradient: a 0/1 verdict would make most of the run invisible

Measured on 2026-07-29 with gpt-5.5 proposing and gpt-4.1-mini writing, 8 iterations:
training score 0.733 -> 0.867, held-out 0.800 -> 0.800. The search improved the prompt
on the instances it saw and none of that carried, which is left standing rather than
tuned away -- but it is a weak observation, not a negative result. Five validation
instances scored in thirds resolve nothing below about 0.07, one run gives no variance
estimate, and the held-out score is also the set the winner was selected on. It says
this run did not show transfer, and no more than that.
"""

# Simplifications vs. the source:
# - The task is constrained writing, not AIME 2022-2025, for the reason above.
# - Five training and five validation instances. The paper trains on AIME 2022-2024
#   and tests on AIME 2025 -- on the order of a hundred problems and thirty, with
#   Figure 7's 57.78% validation score implying a 45-problem validation split.
# - The winner is selected on the same held-out set this script then reports, where the
#   paper keeps a third split and reports the test score. Read the val number as
#   selection-biased.
# - Budget is counted in optimizer iterations, not metric calls or dollars: about 60
#   evaluator calls here against the paper's ~350 and $6.44. `report` prints the count.
# - There is no baseline optimizer. A.3's actual claim is 60.0% against MIPROv2's
#   51.33% on the same benchmark; nothing here compares against any other optimizer,
#   or even against best-of-N hand-written prompts, so that claim is untouched.
# - The candidate is spliced ahead of the question in a user message rather than being
#   a system prompt, and the answer comes back as a typed ``Answer(reasoning, final)``.
#   The type therefore supplies two of the things the paper's evolved prompt had to
#   learn -- an explicit reasoning step, and isolating the final answer (its rule 6) --
#   so roughly a third of Appendix J's content is unreachable as a lever here.
# - One run, one sample per instance, frozen thereafter by the evaluation cache: no
#   repeats, no seed sweep, no variance estimate.
# - Every score is for the prompt *plus the harness's retry loop*, not for the prompt
#   alone. ``worker(...)`` scopes the model but does not shadow the ``TenacityRetryer``
#   above it, so an answer that fails to decode is fed its own error and asked again;
#   only exhausting the retries reaches the ``except`` here and scores zero. A prompt
#   whose answers are borderline-undecodable is flattered by that.
# - The winner is a maximum over the frontier's validation scores while the seed is a
#   single validation measurement, so the reported delta is biased upward. Only
#   frontier candidates are validated at all, so a candidate that the training set
#   dominates can never be selected however well it generalizes.
# - One proposer model; the paper also reports a weaker-proposer arm (its Table 8).

import argparse
import random
import traceback

import pydantic.dataclasses

from docs.source.llm_examples.optimization.library import (
    WORKER_MODEL,
    Diagnostic,
    Evaluation,
    Result,
    Rollout,
    optimize_anything,
    report,
    worker,
)
from effectful.handlers.llm import Agent, Template


@pydantic.dataclasses.dataclass(frozen=True)
class Writing:
    """One constrained-writing instance. The constraints are in the question, so
    nothing is hidden from the answering model: what a better prompt supplies is the
    *method* for satisfying them, which is exactly what the paper's optimized prompts
    encode."""

    name: str
    topic: str
    words: int
    initial: str
    banned: str

    @property
    def question(self) -> str:
        return (
            f"Write a single sentence about {self.topic}. It must contain exactly "
            f"{self.words} words, every word must begin with the letter "
            f"'{self.initial}', and the letter '{self.banned}' must not appear "
            f"anywhere in the sentence."
        )


WRITINGS: list[Writing] = [
    Writing("sailing", "sailing", 6, "s", "e"),
    Writing("markets", "morning markets", 7, "m", "a"),
    Writing("cats", "curious cats", 5, "c", "i"),
    Writing("planets", "distant planets", 8, "p", "o"),
    Writing("bridges", "old bridges", 6, "b", "u"),
    Writing("trains", "night trains", 7, "t", "e"),
    Writing("gardens", "walled gardens", 5, "g", "s"),
    Writing("harbours", "quiet harbours", 6, "h", "i"),
    Writing("lanterns", "paper lanterns", 7, "l", "o"),
    Writing("rivers", "wide rivers", 5, "r", "a"),
]

TRAIN = [
    w
    for w in WRITINGS
    if w.name in {"sailing", "markets", "cats", "planets", "bridges"}
]
VAL = [w for w in WRITINGS if w not in TRAIN]

SEED_PROMPT = "Answer the question."


@pydantic.dataclasses.dataclass(frozen=True)
class Answer:
    """What the answering model returns: its reasoning and its final answer."""

    reasoning: str
    final: str


@Template.define
def answer_question(instructions: str, question: str) -> Answer:
    """{instructions}

    <question>
    {question}
    </question>
    """


def words_of(sentence: str) -> list[str]:
    """The sentence's words, stripped of punctuation and lowercased.

    >>> words_of("Silent ships sail; south, softly.")
    ['silent', 'ships', 'sail', 'south', 'softly']
    """
    cleaned = "".join(
        c if c.isalpha() or c.isspace() or c == "'" else " " for c in sentence
    )
    return [w for w in cleaned.lower().split() if w]


def score_writing(sentence: str, task: Writing) -> tuple[float, list[Diagnostic]]:
    """Check the three constraints and explain every miss.

    Partial credit on purpose: a 0/1 verdict would make most of the search invisible,
    while per-constraint credit is a gradient the proposer can climb -- and the
    per-constraint breakdown *is* the side information.

    A sentence that satisfies all three constraints of the first instance (six words,
    every word starting with 's', no letter 'e' anywhere) scores 1.0:

    >>> score, _ = score_writing("Ships sail south, ships spin swiftly.", WRITINGS[0])
    >>> score
    1.0

    while the near-miss "Silent ships sail south, softly singing." -- same six words,
    same initial, but 'silent' smuggles in an 'e' -- loses exactly one third:

    >>> score, _ = score_writing("Silent ships sail south, softly singing.", WRITINGS[0])
    >>> round(score, 4)
    0.6667
    """
    words = words_of(sentence)
    count_ok = len(words) == task.words
    starting = [w for w in words if w.startswith(task.initial)]
    initial_ratio = len(starting) / len(words) if words else 0.0
    banned_hits = sentence.lower().count(task.banned)

    diagnostics = [
        Diagnostic("sentence", repr(sentence)),
        Diagnostic(
            "word count",
            f"{len(words)} words {words}, needed exactly {task.words}"
            if not count_ok
            else f"exactly {task.words} words, as required",
        ),
        Diagnostic(
            "initial letter",
            f"{len(starting)}/{len(words)} words begin with '{task.initial}'"
            + (
                ""
                if initial_ratio == 1.0
                else f"; offending words: {[w for w in words if not w.startswith(task.initial)]}"
            ),
        ),
        Diagnostic(
            "banned letter",
            f"the letter '{task.banned}' appears {banned_hits} time(s) and must not appear"
            if banned_hits
            else f"the letter '{task.banned}' does not appear, as required",
        ),
    ]
    score = (
        (1.0 if count_ok else 0.0) + initial_ratio + (1.0 if banned_hits == 0 else 0.0)
    ) / 3.0
    return score, diagnostics


def evaluate_prompt(prompt: str, task: Writing | None, model: str) -> Evaluation:
    """Run one writing instance under the candidate prompt and score it.

    The Side Information follows the paper's design for this domain: the instance, the
    model's reasoning, what it produced, and a per-constraint account of what went
    wrong -- not merely that something did.
    """
    assert task is not None, "the prompt domain always has a dataset"
    try:
        with worker(model):
            produced = answer_question(prompt, task.question)
    except Exception:
        return Evaluation(
            score=0.0,
            diagnostics=[
                Diagnostic("task", task.question),
                Diagnostic("crash", traceback.format_exc(limit=2).strip()),
            ],
        )
    score, diagnostics = score_writing(produced.final, task)
    return Evaluation(
        score=score,
        diagnostics=[
            Diagnostic("task", task.question),
            Diagnostic("reasoning", produced.reasoning),
            *diagnostics,
            Diagnostic(
                "verdict",
                f"scored {score:.2f} of 1.00 -- one third for the exact word count, one "
                f"third for the fraction of words with the right initial, one third for "
                f"avoiding the banned letter",
            ),
        ],
    )


class Proposer(Agent):
    """You are a reflective optimizer. You are shown the current prompt, the score it
    achieved, and diagnostic side information explaining *why* it scored that way, and
    you return a better prompt. You do not mutate blindly: you first read the
    diagnostics to decide which failure mode is costing the most, then you write the
    instruction that addresses it."""

    @Template.define
    def propose_prompt(self, current: str, feedback: list[Rollout]) -> str:
        """You are optimizing the SYSTEM PROMPT given to a small model that writes
        sentences under hard constraints -- an exact word count, a required initial
        letter for every word, and a letter that must not appear. The prompt below is
        the artifact; return an improved one.

        <current_prompt>
        {current}
        </current_prompt>

        Here is how it did on a few instances, with the model's own reasoning, the
        sentence it produced, and a per-constraint account of what went wrong:

        <feedback>
        {feedback}
        </feedback>

        Diagnose before you rewrite. The constraints are always stated in the task
        itself, so the prompt's job is not to repeat them but to supply a *method*
        that makes them stick: how to construct the sentence so the count is right by
        construction, how to check each constraint separately rather than trusting a
        glance, what to do on finding a violation, and which failure the feedback
        shows is currently costing the most. Encode that as durable, general
        instructions -- the prompt is scored on instances you have not seen, with
        different topics, counts, letters and banned letters, so never mention a
        specific instance, and never write a sentence yourself.

        Return the improved prompt as plain text, nothing else.
        """


# ---------------------------------------------------------------------------
# Wiring and main
# ---------------------------------------------------------------------------


def run_prompt(args: argparse.Namespace, rng: random.Random) -> Result:
    return optimize_anything(
        evaluator=lambda prompt, task: evaluate_prompt(prompt, task, args.worker_model),
        proposer=lambda prompt, feedback: Proposer().propose_prompt(prompt, feedback),
        seed=SEED_PROMPT,
        dataset=TRAIN,
        valset=VAL,
        budget=args.budget,
        minibatch_size=args.minibatch,
        selection=args.selection,
        use_side_info=not args.no_side_info,
        rng=rng,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=8, help="Optimizer iterations")
    parser.add_argument(
        "--minibatch", type=int, default=2, help="Instances per reflection step"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for selection and minibatches"
    )
    parser.add_argument(
        "--worker-model",
        default=WORKER_MODEL,
        help="Model the prompt is optimized FOR; the harness's --model is the proposer, "
        "as in the paper's proposer/worker split",
    )
    parser.add_argument(
        "--selection",
        choices=["pareto", "best"],
        default="pareto",
        help="Candidate selection; 'best' mutates the best average instead, which is "
        "the naive alternative the paper's 4.3 argues against rather than an ablation "
        "it runs",
    )
    parser.add_argument(
        "--no-side-info",
        action="store_true",
        help="Score-only feedback: the paper's SI ablation",
    )
    args = parser.parse_args()

    result = run_prompt(args, random.Random(args.seed))
    report(result, selection=args.selection, side_info=not args.no_side_info)
    # No assertion that the score improved. In generalization mode the seed can be
    # pruned as training-dominated while every surviving candidate is worse on the
    # held-out set, and that is a legitimate outcome of a search this small -- the
    # result to report, not a failure to raise on.
    if result.best_score <= result.seed_score:
        print(
            "\nThe search did not improve the held-out score. On five validation "
            "instances that is as likely to be the budget as the method."
        )


if __name__ == "__main__":
    main()
