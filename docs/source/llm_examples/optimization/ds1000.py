"""Learn scipy from execution feedback: textual-gradient training on DS-1000.

The training-loop face of `textgrad.py`, ported from the flagship demo of
`strands-labs/ai-functions` (``memory_backprop_scipy``), with the same eight
real DS-1000 scipy training problems and held-out test problem (see
`ds1000_data.py` for provenance). Where `guidelines.py` applies one human
sentence of feedback once, here the feedback is an *oracle's*: every candidate
solution is executed against the benchmark's own tests, and the pass/fail
verdict -- with the error, test input, and expected-vs-actual values on
failure -- is what backpropagates.

Three steps:

1. **Direct test** -- solve the held-out problem with empty memory.
2. **Training** -- solve the eight training problems, backpropagate each
   problem's verdict, then fold every parameter's merged gradients in a single
   accumulation (``scipy_716`` in the training set teaches the ``minimize()``
   contract the held-out problem needs).
3. **Trained test** -- re-solve the held-out problem under the learned
   ``coding_patterns`` / ``common_pitfalls``.

Demonstrates:
- oracle feedback: `ds1000_data.build_feedback` strings from *executing* model
  code -- the loop closes programmatically, no human in it
- batch training: eight recorded roots under one optimizer; per-problem
  ``backward(optimizer.graph(output_i), feedback_i)`` accumulating on the two
  shared boxes; one ``accumulate`` over a synthetic round node gathering the
  roots (the flagship's single-consolidate shape)
- detached evaluation: the two test solves pass ``box.value``, so they are
  gradient-free by construction -- the held-out problem can never train itself
- the data/skill module split that keeps the held-out test honest: each
  problem's ``code_context`` contains its reference solution, and lives in a
  module the solver's system prompt only names

Run with::

    python -m effectful.handlers.llm.harness \\
        docs/source/llm_examples/optimization/ds1000.py \\
        --model gpt-5-mini --reasoning-effort low --tool-choice none

Expect a few minutes: ~10 solver calls plus a backward call per training
problem and one accumulation per parameter. Results are stochastic run to run,
and the model matters: the run above flipped the held-out problem FAIL -> PASS
(the direct attempt returned the whole ``OptimizeResult`` where the test wants
``res.x``; training on ``scipy_716`` taught exactly that contract). A weaker
model (gpt-4o-mini) visibly *learns* -- its trained attempts fix the
L-BFGS-B bounds format its direct attempt crashed on -- but tends to keep
failing the held-out problem on some other of its four simultaneous
requirements.
"""

import argparse
import textwrap

from docs.source.llm_examples.optimization.ds1000_data import (
    TEST_PROBLEMS,
    TRAIN_PROBLEMS,
    ExecutionResult,
    build_feedback,
    execute_and_test,
    extract_solution_code,
)
from docs.source.llm_examples.optimization.textgrad import (
    CallNode,
    Parameter,
    TextGradOptimizer,
)
from effectful.handlers.llm import Skill
from effectful.ops.semantics import handler

coding_patterns = Parameter(
    "No learned patterns yet.",
    description=(
        "Concise bullet-point list (MAX 15 items) of general, reusable coding "
        "patterns and idioms for scipy/data science. Each bullet should be one "
        "sentence. Merge similar patterns into a single bullet. Do not include "
        "problem-specific details."
    ),
)
common_pitfalls = Parameter(
    "No known pitfalls yet.",
    description=(
        "Concise bullet-point list (MAX 15 items) of common, reusable pitfalls "
        "and mistakes to avoid. Each bullet should be one sentence. Merge "
        "similar pitfalls into a single bullet. Do not include problem-specific "
        "details."
    ),
)


@Skill.define
def solve(
    problem: str, library: str, coding_patterns: str, common_pitfalls: str
) -> str:
    """Solve the data science problem below by generating Python code.

    Output ONLY the Python code -- no explanations, no markdown fences. The
    code will be inserted directly into an execution environment where
    {library}, numpy, and the input variables shown in the problem's <code>
    block are already defined: use them as they are, never redefine or
    re-create them, and assign the answer to exactly the variable the problem
    names (the one marked ``# put solution in this variable``).

    <learned_patterns>
    {coding_patterns}
    </learned_patterns>

    <pitfalls_to_avoid>
    {common_pitfalls}
    </pitfalls_to_avoid>

    <problem>
    {problem}
    </problem>

    Do not use any tools.
    """


def attempt(problem: dict, patterns, pitfalls) -> tuple[str, str, ExecutionResult]:
    """One solve of ``problem``, scored by the DS-1000 oracle.

    ``patterns`` / ``pitfalls`` are the `Parameter` boxes during training (the
    recording handler notes the use and unwraps them) and plain ``.value``
    strings during evaluation (detached: no edge, no gradient). Returns the raw
    model output -- the graph key -- alongside the extracted solution and its
    verdict.
    """
    raw = solve(
        problem=problem["prompt"],
        library=problem["library"],
        coding_patterns=patterns,
        common_pitfalls=pitfalls,
    )
    solution = extract_solution_code(raw)
    return raw, solution, execute_and_test(solution, problem["code_context"])


def show(tag: str, problem: dict, solution: str, result: ExecutionResult) -> None:
    print(f"[{tag}] {problem['id']}: {'PASS' if result.passed else 'FAIL'}")
    print(textwrap.indent(solution.strip(), "    "))
    if not result.passed and result.error:
        print(textwrap.indent(f"error: {result.error.strip()}", "    ! "))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    test = TEST_PROBLEMS[0]

    # Step 1 -- direct test with empty memory, detached (.value): nothing is
    # recorded and no gradient can flow from the held-out problem.
    print("== Step 1: direct test (empty memory) ==\n")
    _, before_solution, before = attempt(
        test, coding_patterns.value, common_pitfalls.value
    )
    show("direct", test, before_solution, before)

    # Step 2 -- training: eight independent solves recorded as eight roots.
    print("== Step 2: training ==\n")
    optimizer = TextGradOptimizer()
    attempts: list[tuple[dict, str, str, ExecutionResult]] = []
    with handler(optimizer):
        for problem in TRAIN_PROBLEMS:
            raw, solution, result = attempt(problem, coding_patterns, common_pitfalls)
            attempts.append((problem, raw, solution, result))

    # Backpropagate each problem's oracle verdict; gradients accumulate on the
    # two shared boxes across all eight backward passes.
    for problem, raw, solution, result in attempts:
        print(f"[train] {problem['id']}: {'PASS' if result.passed else 'FAIL'}")
        optimizer.backward(
            optimizer.graph(raw), build_feedback(problem, solution, result)
        )
    print(
        f"\naccumulated gradients: {len(coding_patterns.gradients)} on "
        f"coding_patterns, {len(common_pitfalls.gradients)} on common_pitfalls"
    )

    # One merged update per parameter: a synthetic round node gathers the
    # eight roots so a single accumulate folds each box's gradients at once.
    training_round = CallNode(
        skill_name="training round",
        children=[optimizer.graph(raw) for _, raw, _, _ in attempts],
    )
    optimizer.accumulate(training_round)

    print(f"\nlearned coding_patterns:\n{coding_patterns.value}\n")
    print(f"learned common_pitfalls:\n{common_pitfalls.value}\n")

    # Step 3 -- re-test with the learned memory, detached again.
    print("== Step 3: trained test ==\n")
    _, after_solution, after = attempt(
        test, coding_patterns.value, common_pitfalls.value
    )
    show("trained", test, after_solution, after)

    verdict = {
        (False, True): "FAIL -> PASS (memory-driven improvement)",
        (True, True): "PASS -> PASS",
        (False, False): "FAIL -> FAIL",
        (True, False): "PASS -> FAIL",
    }[(before.passed, after.passed)]
    print(f"{test['id']}: {verdict}")


if __name__ == "__main__":
    main()
