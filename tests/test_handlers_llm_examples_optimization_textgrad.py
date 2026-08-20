"""Tests for ``docs/source/llm_examples/optimization/textgrad.py`` and its demos.

Two mocking levels, mirroring the other example tests:

- `MockLLM` implements `call_agent` directly, so graph capture and
  backward/consolidate routing are tested without any completion machinery.
- `MockCompletionHandler` (from ``conftest``) scripts `completion` under the
  real ``AgentLoop`` + ``TenacityRetryer`` stack, so the decode-time
  certification of `Feedback.target` -- an invalid name raising and being
  retried -- is exercised end to end.

The final section covers the model-free half of the DS-1000 demo
(``ds1000_data.py``): the problem data is intact, and the execution oracle
passes correct solutions, fails incorrect ones with actionable feedback, and
cleans model output. The demo's learning loop itself is the machinery covered
above (see the batch-training test).
"""

import inspect
import json

import pytest

from docs.source.llm_examples.optimization.textgrad import (
    CallNode,
    Feedback,
    Parameter,
    TextGradOptimizer,
    Updated,
)
from effectful.handlers.llm import Skill
from effectful.handlers.llm.harness.durability.retrying import TenacityRetryer
from effectful.handlers.llm.harness.durability.transaction import HistoryBuilder
from effectful.handlers.llm.harness.hooks import (
    AgentLoop,
    ResultDecodingError,
    call_agent,
)
from effectful.handlers.llm.harness.legibility.lexical import LexicalToolExtractor
from effectful.handlers.llm.harness.provision.litellm import LiteLLMConfigurer
from effectful.ops.semantics import handler
from effectful.ops.syntax import ObjectInterpretation, implements
from tests.conftest import MockCompletionHandler, make_text_response

# ── Skills under test (stateless; parameters are per-test boxes) ─────────────


@Skill.define
def write(topic: str, guidelines: str) -> str:
    """Write a short joke about {topic}, following {guidelines}."""


@Skill.define
def compose(joke_1: str, joke_2: str, guidelines: str) -> str:
    """Write an email with {joke_1} and {joke_2}, following {guidelines}."""


class MockLLM(ObjectInterpretation):
    """Answer `call_agent` from a mapping of skill name to a canned rule.

    A rule is a callable applied to the call's bound arguments (so tests can
    return per-call-distinct objects; identity matters for sibling wiring).
    Every intercepted call is recorded as ``(skill_name, bound_arguments)``.
    """

    def __init__(self, rules):
        self.rules = rules
        self.calls: list[tuple[str, dict]] = []

    @implements(call_agent)
    def _call(self, skill, *args, **kwargs):
        bound = inspect.signature(skill).bind(*args, **kwargs)
        bound.apply_defaults()
        self.calls.append((skill.__name__, dict(bound.arguments)))
        return self.rules[skill.__name__](**bound.arguments)


def forward_rules():
    """Rules for the two-jokes-into-an-email forward pass."""
    return {
        "write": lambda topic, guidelines: f"a joke about {topic}",
        "compose": lambda joke_1, joke_2, guidelines: f"email: {joke_1} / {joke_2}",
    }


def run_forward(mock, joke_box, email_box):
    """The demo's forward shape: two writes feeding one compose, recorded."""
    optimizer = TextGradOptimizer()
    with handler(mock), handler(optimizer):
        j1 = write(topic="cats", guidelines=joke_box)
        j2 = write(topic="dogs", guidelines=joke_box)
        email = compose(joke_1=j1, joke_2=j2, guidelines=email_box)
    return optimizer, j1, j2, email


# ── Graph capture ────────────────────────────────────────────────────────────


def test_records_parameters_and_sibling_edges():
    joke_box = Parameter("joke guide")
    email_box = Parameter("email guide")
    mock = MockLLM(forward_rules())
    optimizer, j1, j2, email = run_forward(mock, joke_box, email_box)

    # Sibling wiring reparents the consumed writes: one root, the email node.
    assert len(optimizer._roots) == 1
    root = optimizer._roots[0]
    assert root.skill_name == "compose" and root.value == email
    assert [c.skill_name for c in root.children] == ["write", "write"]
    assert [c.value for c in root.children] == [j1, j2]

    # Boxes are recorded under their argument names, on the right nodes...
    assert root.parameters == [("guidelines", email_box)]
    for child in root.children:
        assert child.parameters == [("guidelines", joke_box)]

    # ...and unwrapped before the call proceeds: the model side saw plain str.
    for _, arguments in mock.calls:
        assert isinstance(arguments["guidelines"], str)


def test_passing_value_detaches():
    joke_box = Parameter("joke guide")
    mock = MockLLM(forward_rules())
    optimizer = TextGradOptimizer()
    with handler(mock), handler(optimizer):
        write(topic="cats", guidelines=joke_box.value)
    assert optimizer._roots[0].parameters == []


def test_parameter_found_inside_container():
    box = Parameter("guide")

    @Skill.define
    def summarize(notes: list[str]) -> str:
        """Summarize {notes}."""

    mock = MockLLM({"summarize": lambda notes: "summary"})
    optimizer = TextGradOptimizer()
    with handler(mock), handler(optimizer):
        summarize(notes=["a", box])
    assert optimizer._roots[0].parameters == [("notes", box)]
    # The container reaching the call has the box replaced by its value.
    assert mock.calls[0][1]["notes"] == ["a", "guide"]


def test_nested_call_becomes_child():
    box = Parameter("guide")

    @Skill.define
    def inner(guidelines: str) -> str:
        """Use {guidelines}."""

    @Skill.define
    def outer(topic: str) -> str:
        """Write about {topic}."""

    # The outer rule calls `inner` while the outer node is on the stack, the
    # way a model-driven tool call or synthesized code would.
    mock = MockLLM(
        {
            "outer": lambda topic: f"outer({inner(guidelines=box)})",
            "inner": lambda guidelines: "inner-result",
        }
    )
    optimizer = TextGradOptimizer()
    with handler(mock), handler(optimizer):
        outer(topic="cats")

    assert len(optimizer._roots) == 1
    root = optimizer._roots[0]
    assert root.skill_name == "outer"
    assert [c.skill_name for c in root.children] == ["inner"]
    assert root.children[0].parameters == [("guidelines", box)]


def test_parameter_free_child_is_not_a_target():
    """A subtree that reaches no `Parameter` is pruned from the backward pass:
    it is offered to the backward model as no target, so no gradient (and no
    model call) can be spent where none could land."""
    box = Parameter("guide")
    seen_targets: list[list] = []
    rules = {
        "write": lambda topic, guidelines: f"a joke about {topic}",
        "compose": lambda joke_1, joke_2, guidelines: "email",
        "_compute_gradients": lambda targets, trace, output, feedback: (
            seen_targets.append(targets)
            or [Feedback(target=t.name, feedback=f"improve {t.name}") for t in targets]
        ),
        "_accumulate": lambda parameter: Updated(str(parameter.value)),
    }
    mock = MockLLM(rules)
    optimizer = TextGradOptimizer()
    with handler(mock), handler(optimizer):
        j1 = write(topic="cats", guidelines=box)
        j2 = write(topic="dogs", guidelines=box.value)  # detached: dead end
        compose(joke_1=j1, joke_2=j2, guidelines="be brief")
    with handler(mock):
        optimizer.step("funnier")

    # The root offered only the live write as a target -- not the detached one,
    # whose subtree no gradient can reach -- and only two nodes distributed.
    assert [t.name for t in seen_targets[0]] == ["write"]
    assert len(seen_targets) == 2
    assert box.gradients == ["improve guidelines"]


# ── Backward + consolidate (call_agent-level mock) ───────────────────────────


def optimizer_rules(updated="{value} [updated]"):
    """Rules for the optimizer's internal skills: feed back to every target,
    and accumulate by tagging the value."""
    return {
        "_compute_gradients": lambda targets, trace, output, feedback: [
            Feedback(target=t.name, feedback=f"improve {t.name}") for t in targets
        ],
        "_accumulate": lambda parameter: Updated(updated.format(value=parameter.value)),
    }


def test_backward_routes_and_consolidate_dedupes():
    joke_box = Parameter("joke guide", description="joke instructions")
    email_box = Parameter("email guide")
    mock = MockLLM(forward_rules() | optimizer_rules())
    optimizer, *_ = run_forward(mock, joke_box, email_box)

    with handler(mock):  # outside the recording scope, like the demo
        graph, grads = optimizer.step("make it funnier")

    # step returns the walked graph (the email node) plus the ephemeral
    # per-node routed feedback; the graph itself carries no backward state.
    assert graph is optimizer._roots[0]
    assert grads[graph] == ["make it funnier"]

    # The email node distributed to its box and both children; each child then
    # distributed its refined share to the shared joke box.
    assert email_box.gradients == ["improve guidelines"]
    assert grads[graph.children[0]] == ["improve write"]
    assert grads[graph.children[1]] == ["improve write_2"]
    assert joke_box.gradients == ["improve guidelines", "improve guidelines"]

    # One accumulation per box (identity dedup), mutating values in place.
    assert joke_box.value == "joke guide [updated]"
    assert email_box.value == "email guide [updated]"
    accumulations = [c for c in mock.calls if c[0] == "_accumulate"]
    assert len(accumulations) == 2
    # The box itself is the skill's one argument, carrying value, description
    # and gradients together.
    assert {id(c[1]["parameter"]) for c in accumulations} == {
        id(joke_box),
        id(email_box),
    }

    # zero_grad clears the parameters -- the only persistent gradient state.
    optimizer.zero_grad(graph)
    assert not joke_box.gradients and not email_box.gradients


def test_batch_training_backward_per_root_accumulate_once():
    """The training-loop shape (the DS-1000 demo): several independent roots
    recorded under one optimizer, per-root `backward` via `graph(output)`
    accumulating on a shared box, then one `accumulate` over a synthetic node
    gathering the roots -- a single merged update per parameter."""
    box = Parameter("guide")
    mock = MockLLM(forward_rules() | optimizer_rules())
    optimizer = TextGradOptimizer()
    with handler(mock), handler(optimizer):
        outs = [write(topic=t, guidelines=box) for t in ("cats", "dogs", "fish")]

    with handler(mock):
        for i, out in enumerate(outs):
            root = optimizer.graph(out)
            assert root.value == out  # graph() resolves each root by identity
            optimizer.backward(root, f"feedback {i}")
        assert box.gradients == ["improve guidelines"] * 3

        training_round = CallNode(
            skill_name="round", children=[optimizer.graph(o) for o in outs]
        )
        optimizer.accumulate(training_round)

    assert box.value == "guide [updated]"
    assert len([c for c in mock.calls if c[0] == "_accumulate"]) == 1


def test_non_str_parameter_roundtrips_through_json():
    box = Parameter(["short jokes", "no puns"], description="joke rules")

    @Skill.define
    def apply_rules(topic: str, rules: list[str]) -> str:
        """Write about {topic} following {rules}."""

    rules = {
        "apply_rules": lambda topic, rules: "a joke",
        "_compute_gradients": lambda targets, trace, output, feedback: [
            Feedback(target="rules", feedback="add: mention coffee")
        ],
        # The reply boundary is str: for a list-valued parameter the returned
        # text is validated back into the value's type from JSON.
        "_accumulate": lambda parameter: Updated(
            json.dumps(parameter.value + ["mention coffee"])
        ),
    }
    mock = MockLLM(rules)
    optimizer = TextGradOptimizer()
    with handler(mock), handler(optimizer):
        apply_rules(topic="cats", rules=box)
    with handler(mock):
        optimizer.step("needs coffee")

    accumulation = next(c[1] for c in mock.calls if c[0] == "_accumulate")
    assert accumulation["parameter"] is box
    assert box.value == ["short jokes", "no puns", "mention coffee"]


def test_step_requires_unambiguous_root():
    box = Parameter("guide")
    mock = MockLLM(forward_rules() | optimizer_rules())
    optimizer = TextGradOptimizer()
    with handler(mock), handler(optimizer):
        write(topic="cats", guidelines=box)
        j2 = write(topic="dogs", guidelines=box)

    with handler(mock):
        with pytest.raises(ValueError, match="2 roots"):
            optimizer.step("feedback")
        with pytest.raises(ValueError, match="not produced"):
            optimizer.step("feedback", output="never seen")

        graph, _ = optimizer.step("feedback", output=j2)
    assert graph.value == j2
    assert box.gradients == ["improve guidelines"]


# ── Decode-time certification (completion-level mock) ────────────────────────


def make_boxed_response(value):
    """A completion whose content is the ``BoxedResponse`` JSON for `value`."""
    return make_text_response(json.dumps({"value": value}))


def full_stack(mock):
    """The offline harness stack around a scripted `completion` handler."""
    return (
        handler(mock),
        handler(AgentLoop()),
        handler(LexicalToolExtractor()),
        handler(LiteLLMConfigurer(model="test-model")),
        handler(HistoryBuilder()),
        handler(TenacityRetryer()),
    )


def test_invalid_target_is_retried():
    box = Parameter("joke guide")
    mock = MockCompletionHandler(
        [
            # forward: `write` returns str, sent as plain text
            make_text_response("a joke"),
            # backward, attempt 1: hallucinated target -> __post_init__ raises,
            # TenacityRetryer feeds the error back and retries
            make_boxed_response([{"target": "bogus", "feedback": "x"}]),
            # backward, attempt 2: valid
            make_boxed_response([{"target": "guidelines", "feedback": "add: cats"}]),
            # accumulation: structured `Updated` output
            make_boxed_response({"value": "joke guide + cats"}),
        ]
    )
    h1, h2, h3, h4, h5, h6 = full_stack(mock)
    with h1, h2, h3, h4, h5, h6:
        optimizer = TextGradOptimizer()
        with handler(optimizer):
            write(topic="cats", guidelines=box)
        optimizer.step("more cats")

    assert box.gradients == ["add: cats"]
    assert box.value == "joke guide + cats"
    assert mock.call_count == 4

    # The retry really happened: the second backward request carries the failed
    # attempt and the certification error naming the valid targets.
    retry_request = mock.received_messages[2]
    assert "bogus" in json.dumps(retry_request)
    assert "guidelines" in str(retry_request[-1])


def test_exhausted_retries_raise():
    box = Parameter("joke guide")
    mock = MockCompletionHandler(
        [
            make_text_response("a joke"),
            # Every backward attempt names an unknown target (the mock repeats
            # its last response); the retryer gives up and the decoding error
            # propagates, leaving the parameter untouched.
            make_boxed_response([{"target": "bogus", "feedback": "x"}]),
        ]
    )
    h1, h2, h3, h4, h5, h6 = full_stack(mock)
    with h1, h2, h3, h4, h5, h6:
        optimizer = TextGradOptimizer()
        with handler(optimizer):
            write(topic="cats", guidelines=box)
        with pytest.raises(ResultDecodingError):
            optimizer.step("more cats")

    assert box.gradients == []
    assert box.value == "joke guide"  # nothing was consolidated


# ── The DS-1000 demo's execution oracle (model-free) ─────────────────────────


@pytest.fixture(scope="module")
def ds1000():
    """The demo's data module, skipping (only) these tests without scipy."""
    pytest.importorskip("scipy")
    from docs.source.llm_examples.optimization import ds1000_data

    return ds1000_data


@pytest.fixture(scope="module")
def zscores_problem(ds1000):
    """scipy_717: convert z-scores to left-tailed p-values."""
    return ds1000.TRAIN_PROBLEMS[6]


def test_ds1000_problem_data_is_intact(ds1000):
    assert len(ds1000.TRAIN_PROBLEMS) == 8 and len(ds1000.TEST_PROBLEMS) == 1
    for problem in ds1000.TRAIN_PROBLEMS + ds1000.TEST_PROBLEMS:
        assert problem["library"] == "Scipy"
        assert "BEGIN SOLUTION" in problem["prompt"]
        assert "test_execution" in problem["code_context"]


def test_ds1000_oracle_passes_a_correct_solution(ds1000, zscores_problem):
    result = ds1000.execute_and_test(
        "p_values = scipy.stats.norm.cdf(z_scores)", zscores_problem["code_context"]
    )
    assert result.passed and result.error is None


def test_ds1000_oracle_fails_a_wrong_solution_with_detail(ds1000, zscores_problem):
    result = ds1000.execute_and_test(
        "p_values = z_scores", zscores_problem["code_context"]
    )
    assert not result.passed
    assert "assertion failed" in result.error.lower()
    # The assertion re-run captures the concrete disagreement for feedback.
    assert result.test_input and result.expected_output and result.actual_output

    feedback = ds1000.build_feedback(zscores_problem, "p_values = z_scores", result)
    assert "scipy_717 FAILED" in feedback
    assert "p_values = z_scores" in feedback
    assert "Expected output" in feedback


def test_ds1000_oracle_reports_exceptions_as_failures(ds1000, zscores_problem):
    result = ds1000.execute_and_test(
        "p_values = undefined_name", zscores_problem["code_context"]
    )
    assert not result.passed
    assert "NameError" in result.error


def test_ds1000_passing_feedback_reinforces(ds1000, zscores_problem):
    solution = "p_values = scipy.stats.norm.cdf(z_scores)"
    good = ds1000.execute_and_test(solution, zscores_problem["code_context"])
    feedback = ds1000.build_feedback(zscores_problem, solution, good)
    assert "SOLVED" in feedback and "Remember this pattern" in feedback


def test_ds1000_extract_solution_code_strips_fences_and_markers(ds1000):
    raw = "```python\nBEGIN SOLUTION\nresult = 1\nEND SOLUTION\n```"
    assert ds1000.extract_solution_code(raw).strip() == "result = 1"
    assert ds1000.extract_solution_code("result = 2").strip() == "result = 2"
