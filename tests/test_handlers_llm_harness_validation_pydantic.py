"""Tests for the contracts a caller writes into a `Skill`'s own annotations.

A parameter annotated with pydantic metadata carries a *pre-condition*, run by
`PydanticSkillArgValidator` before the model is asked anything; a return
annotation carries a *post-condition*, run by `call_assistant` as it decodes
the answer, where a rejection becomes retry feedback. Both are the caller's,
and neither is part of the encoding -- which is what these tests are about: the
contracts have to hold across every way a skill can be written and every way an
answer can arrive, and to change nothing at all for a skill that declares none.

The axes, and where each is covered:

* signature shape -- positional, keyword-only, defaulted, variadic, generic,
  `Agent` method, multiply-annotated, context-reading (`Pre-conditions`)
* tool-calling mode -- ``mixed``, ``code``, ``json`` (`Every tool-calling mode`)
* answer path -- structured output, or a synthesized body through
  ``write_and_run_body`` (`Post-conditions`)
* the `str` case, where an answer is prose rather than JSON (`Text answers`)
* orthogonality -- an unannotated skill is untouched (`Additivity`)

All of it is offline except the two `requires_llm` tests at the end: the model
is a `MockCompletionHandler` scripted with the exact replies each case needs.
"""

import contextlib
import importlib.util
import json
import sys
import typing

import annotated_types
import pydantic
import pytest
from litellm.files.main import ModelResponse

from effectful.handlers.llm import Skill, Tool
from effectful.handlers.llm.harness.durability.retrying import TenacityRetryer
from effectful.handlers.llm.harness.durability.transaction import HistoryBuilder
from effectful.handlers.llm.harness.execution.builtin import BuiltinExecutor
from effectful.handlers.llm.harness.hooks import AgentLoop, call_assistant, completion
from effectful.handlers.llm.harness.legibility.lexical import LexicalToolExtractor
from effectful.handlers.llm.harness.provision.litellm import LiteLLMConfigurer
from effectful.handlers.llm.harness.synthesis.body import FinalBodySynthesizer
from effectful.handlers.llm.harness.synthesis.toolcall import (
    ExpressionToolCaller,
    MixedToolCaller,
)
from effectful.handlers.llm.harness.validation.pydantic import (
    PydanticSkillArgValidator,
)
from effectful.handlers.llm.harness.validation.ty import TyTypeChecker
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from tests.conftest import (
    MockCompletionHandler,
    make_text_response,
    make_tool_call_response,
    requires_llm,
)

# ============================================================================
# Fixture module: one skill per signature shape, imported from a real file so
# `_recover_skill_def` can splice into its source on the synthesis path.
# ============================================================================

_CONTRACTS_SRC = '''
import dataclasses
import typing
from collections.abc import Sequence

import annotated_types
import pydantic

from effectful.handlers.llm import Agent, Skill, Tool

# What each validator saw, in the order it saw it.
calls: list[tuple] = []


def _positive(x: int) -> int:
    calls.append(("positive", x))
    if x <= 0:
        raise ValueError(f"{x} is not positive")
    return x


def _shout(text: str) -> str:
    """A pre-condition that *normalizes* rather than merely checking."""
    return text.upper()


def _first(x: int) -> int:
    calls.append(("first", x))
    return x


def _second(x: int) -> int:
    calls.append(("second", x))
    return x


def _above_lo(hi: int, info: pydantic.ValidationInfo) -> int:
    """A pre-condition relative to another argument of the same call."""
    lo = (info.context or {})["lo"]
    calls.append(("above_lo", hi, lo))
    if hi <= lo:
        raise ValueError(f"hi must exceed lo={lo}, got {hi}")
    return hi


def _at_least_lo(answer: int, info: pydantic.ValidationInfo) -> int:
    """A post-condition relative to the call's arguments."""
    lo = (info.context or {})["lo"]
    calls.append(("at_least_lo", answer, lo))
    if answer < lo:
        raise ValueError(f"answer must be at least lo={lo}, got {answer}")
    return answer


def _concise(text: str) -> str:
    calls.append(("concise", text))
    if len(text.split()) > 5:
        raise ValueError(f"{len(text.split())} words is too many; keep it under 5")
    return text


Pos = typing.Annotated[int, pydantic.AfterValidator(_positive)]


@dataclasses.dataclass
class Payload:
    """An argument with no metadata on its annotation."""

    text: str


@Skill.define
def scale(x: Pos) -> int:
    """Double {x}."""


@Skill.define
def announce(text: typing.Annotated[str, pydantic.AfterValidator(_shout)]) -> int:
    """Count the characters in {text}."""


@Skill.define
def offset(x: int, *, by: Pos = 1) -> int:
    """Add {by} to {x}."""


@Skill.define
def vsum(*xs: Pos) -> int:
    """Sum {xs}."""


@Skill.define
def kfmt(**parts: Pos) -> int:
    """Sum the values of {parts}."""


@Skill.define
def pick[T](items: typing.Annotated[Sequence[T], annotated_types.MinLen(1)]) -> int:
    """Pick an index into {items}."""


@Skill.define
def twice(
    x: typing.Annotated[
        int, pydantic.AfterValidator(_first), pydantic.AfterValidator(_second)
    ],
) -> int:
    """Double {x}."""


@Skill.define
def bounded(lo: int, hi: typing.Annotated[int, pydantic.AfterValidator(_above_lo)]) -> int:
    """Pick a number between {lo} and {hi}."""


@Skill.define
def echo(payload: Payload) -> int:
    """Count the characters in {payload}."""


@dataclasses.dataclass
class Scaler(Agent):
    """An agent whose skill carries a pre-condition."""

    factor: int

    @Skill.define
    def scale_by(self, x: Pos) -> int:
        """Multiply {x} by {self.factor}."""


@Skill.define
def choose(lo: int) -> typing.Annotated[int, pydantic.AfterValidator(_at_least_lo)]:
    """Pick a number no smaller than {lo}."""


@Skill.define
def blurb(topic: str) -> typing.Annotated[str, pydantic.AfterValidator(_concise)]:
    """Say something brief about {topic}."""


@Skill.define
def plain_blurb(topic: str) -> str:
    """Say something brief about {topic}."""


@Skill.define
def driver(n: int) -> str:
    """Use the tools on {n}, then summarize."""


@Tool.define
def bump(x: Pos) -> int:
    """Increase x by one."""
    calls.append(("bump", x))
    return x + 1
'''


def _import_fixture(tmp_path, source: str, modname: str):
    """Import `source` as a real module, so its source can be recovered."""
    path = tmp_path / f"{modname}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(modname, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod(tmp_path, request):
    modname = f"_contracts_fixture_{request.node.name}".replace("[", "_").replace(
        "]", ""
    )
    module = _import_fixture(tmp_path, _CONTRACTS_SRC, modname)
    yield module
    sys.modules.pop(modname, None)


class ScriptedModel(MockCompletionHandler):
    """A `MockCompletionHandler` that also records what was *asked* of it.

    The response format is the point of the text-answer cases below, and it is
    a request parameter rather than anything observable in the reply.
    """

    def __init__(self, responses):
        super().__init__(responses)
        self.response_formats: list = []
        self.tools: list = []

    @implements(completion)
    def _completion(self, messages=None, **kwargs):
        self.response_formats.append(kwargs.get("response_format"))
        self.tools.append([t["function"]["name"] for t in (kwargs.get("tools") or [])])
        return super()._completion(messages=messages, **kwargs)


def _run(call, responses, *extra_handlers, caller=MixedToolCaller):
    """Run `call` against a scripted model, returning ``(result, model)``.

    Handler order mirrors `harness`: the extras (a retryer, a synthesizer) go
    between `HistoryBuilder` and the type checker, where they intercept
    `call_agent` before `AgentLoop`'s terminal rule.
    """
    model = ScriptedModel(responses)
    stack = [
        handler(AgentLoop()),
        handler(caller()),
        handler(LiteLLMConfigurer(model="test-model")),
        handler(HistoryBuilder()),
        *(handler(h) for h in extra_handlers),
        # Last of the `call_agent` handlers, so it intercepts first -- the order
        # `harness` installs it in, after `FinalBodySynthesizer`.
        handler(PydanticSkillArgValidator()),
        handler(TyTypeChecker()),
        handler(BuiltinExecutor()),
        handler(model),
    ]
    with contextlib.ExitStack() as es:
        for h in stack:
            es.enter_context(h)
        return call(), model


def _value(payload) -> ModelResponse:
    """A structured answer, as the model would send it."""
    return make_text_response(json.dumps({"value": payload}))


def _prompt_text(model: ScriptedModel) -> str:
    """Every message of the last request, flattened to searchable text.

    Content arrives either as a string or as a list of blocks, and a value
    spliced into a prompt may be either -- so both are flattened rather than
    dumped, to keep the assertions below about the text the model reads.
    """
    parts: list[str] = []
    for message in model.received_messages[-1]:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts += [b["text"] for b in content if isinstance(b, dict) and "text" in b]
    return "\n".join(parts)


class _EnvCapture(ObjectInterpretation):
    """Records the environment `call_assistant` is given, per turn."""

    def __init__(self):
        self.envs: list = []

    @implements(call_assistant)
    def _call_assistant(self, messages, response_type, env, tools=frozenset()):
        self.envs.append(env)
        return fwd(messages, response_type, env, tools)


# ============================================================================
# Pre-conditions: the caller's metadata on a parameter, across signature shapes
# ============================================================================


def test_precondition_runs_on_a_direct_python_call(mod):
    """The point of enforcing them in `call_agent`: Python itself would not."""
    result, model = _run(lambda: mod.scale(3), [_value(6)])
    assert result == 6
    assert mod.calls == [("positive", 3)]
    assert model.call_count == 1


def test_precondition_rejects_before_the_model_is_asked(mod):
    """A rejected argument costs no model call: it is a *pre*-condition."""
    with pytest.raises(pydantic.ValidationError, match="-1 is not positive"):
        _run(lambda: mod.scale(-1), [_value(6)])


def test_rejection_is_a_value_error(mod):
    """`pydantic.ValidationError` subclasses `ValueError`, so a caller that
    handles rejections the obvious way keeps working."""
    with pytest.raises(ValueError):
        _run(lambda: mod.scale(-1), [_value(6)])


def test_no_model_call_is_made_when_a_precondition_fails(mod):
    """The minimum stack that enforces a contract: the checker and a provider."""
    model = ScriptedModel([_value(6)])
    with (
        handler(AgentLoop()),
        handler(LiteLLMConfigurer(model="test-model")),
        handler(HistoryBuilder()),
        handler(PydanticSkillArgValidator()),
        handler(model),
    ):
        with pytest.raises(pydantic.ValidationError):
            mod.scale(-1)
    assert model.call_count == 0


def test_without_the_checker_a_direct_call_is_unvalidated(mod):
    """The handler is what makes a pre-condition hold on a Python call: without
    it the annotation is inert, which is the state `check_contracts=False`
    leaves a stack in."""
    model = ScriptedModel([_value(6)])
    with (
        handler(AgentLoop()),
        handler(LiteLLMConfigurer(model="test-model")),
        handler(HistoryBuilder()),
        handler(model),
    ):
        assert mod.scale(-1) == 6
    assert mod.calls == []


def test_a_normalizing_precondition_is_applied_not_just_consulted(mod):
    """A validator that transforms its argument replaces it.

    Asserted on the environment the call runs under -- which is what the prompt
    is then rendered from -- rather than on the rendered text, so the test says
    nothing about how a value is spelled once encoded.
    """
    capture = _EnvCapture()
    _, _ = _run(lambda: mod.announce("hello"), [_value(5)], capture)
    assert capture.envs[0]["text"] == "HELLO"


def test_precondition_on_keyword_only_parameter(mod):
    _, _ = _run(lambda: mod.offset(1, by=2), [_value(3)])
    assert mod.calls == [("positive", 2)]
    with pytest.raises(pydantic.ValidationError):
        _run(lambda: mod.offset(1, by=-2), [_value(3)])


def test_precondition_runs_on_an_applied_default(mod):
    """`apply_defaults` puts the default in `arguments`, so it is checked too --
    a default that violates the contract is a defect worth surfacing."""
    _, _ = _run(lambda: mod.offset(1), [_value(2)])
    assert mod.calls == [("positive", 1)]


def test_precondition_on_variadic_positional_runs_per_element(mod):
    """``*xs: Pos`` annotates each item, not the tuple `bind` collects."""
    result, _ = _run(lambda: mod.vsum(1, 2, 3), [_value(6)])
    assert result == 6
    assert mod.calls == [("positive", 1), ("positive", 2), ("positive", 3)]


def test_precondition_on_variadic_positional_rejects_a_bad_element(mod):
    with pytest.raises(pydantic.ValidationError, match="-2 is not positive"):
        _run(lambda: mod.vsum(1, -2), [_value(0)])


def test_precondition_on_variadic_keyword_runs_per_value(mod):
    result, model = _run(lambda: mod.kfmt(a=1, b=2), [_value(3)])
    assert result == 3
    assert mod.calls == [("positive", 1), ("positive", 2)]
    # The parameter is still the mapping it was bound as, with each value
    # validated in place -- not replaced by one of them.
    capture = _EnvCapture()
    _run(lambda: mod.kfmt(a=1, b=2), [_value(3)], capture)
    assert capture.envs[0]["parts"] == {"a": 1, "b": 2}


def test_precondition_on_variadic_keyword_rejects_a_bad_value(mod):
    with pytest.raises(pydantic.ValidationError, match="-1 is not positive"):
        _run(lambda: mod.kfmt(a=-1), [_value(0)])


def test_precondition_on_a_generic_parameter(mod):
    """A free type variable under the annotation is no obstacle: the metadata
    is applied to whatever the parameter's type resolves to."""
    result, _ = _run(lambda: mod.pick([1, 2, 3]), [_value(0)])
    assert result == 0
    with pytest.raises(pydantic.ValidationError):
        _run(lambda: mod.pick([]), [_value(0)])


def test_multiple_validators_run_in_annotation_order(mod):
    _, _ = _run(lambda: mod.twice(2), [_value(4)])
    assert mod.calls == [("first", 2), ("second", 2)]


def test_precondition_reads_the_calls_other_arguments(mod):
    """The validation context is the call environment, so a pre-condition can
    be stated relative to the rest of the call."""
    result, _ = _run(lambda: mod.bounded(1, 5), [_value(3)])
    assert result == 3
    assert mod.calls == [("above_lo", 5, 1)]

    with pytest.raises(pydantic.ValidationError, match="hi must exceed lo=10"):
        _run(lambda: mod.bounded(10, 5), [_value(3)])


def test_precondition_on_an_agent_method(mod):
    """A method skill binds `self` through the lexical half of the environment
    and its arguments through the other; the pre-condition sees both."""
    scaler = mod.Scaler(factor=2)
    result, _ = _run(lambda: scaler.scale_by(3), [_value(6)])
    assert result == 6
    assert mod.calls == [("positive", 3)]

    with pytest.raises(pydantic.ValidationError):
        _run(lambda: scaler.scale_by(-3), [_value(6)])


# ============================================================================
# Every tool-calling mode: a guarded skill invoked by the model, not by Python
# ============================================================================


CALLERS = [
    pytest.param(MixedToolCaller, id="mixed"),
    pytest.param(ExpressionToolCaller, id="code"),
    pytest.param(LexicalToolExtractor, id="json"),
]


@pytest.mark.parametrize("caller", CALLERS)
def test_precondition_holds_however_the_model_calls_the_skill(mod, caller):
    """A model-issued call to `scale` is rejected in every mode.

    Which layer rejects it differs -- the decoder in the JSON pathway, the
    pre-condition in the code pathway -- but the contract does not, and either
    way the failure is fed back and the conversation continues rather than the
    bad argument reaching a model call. Three completions would mean `scale`
    got its own turn; two means it never did.
    """
    call = (
        json.dumps({"call": "scale(-2)"})
        if caller is ExpressionToolCaller
        else json.dumps({"x": -2})
    )
    result, model = _run(
        lambda: mod.driver(1),
        [make_tool_call_response("scale", call), make_text_response("done")],
        TenacityRetryer(),
        caller=caller,
    )
    assert result == "done"
    assert model.call_count == 2
    assert ("positive", -2) in mod.calls


@pytest.mark.parametrize("caller", CALLERS)
def test_a_satisfied_precondition_lets_the_model_call_through(mod, caller):
    """The mirror image: a good argument is not disturbed by any mode."""
    call = (
        json.dumps({"call": "scale(2)"})
        if caller is ExpressionToolCaller
        else json.dumps({"x": 2})
    )
    result, model = _run(
        lambda: mod.driver(1),
        [
            make_tool_call_response("scale", call),
            _value(4),  # `scale`'s own answer
            make_text_response("done"),
        ],
        TenacityRetryer(),
        caller=caller,
    )
    assert result == "done"
    assert ("positive", 2) in mod.calls
    assert model.call_count == 3


def test_plain_tool_metadata_is_enforced_in_the_json_pathway(mod):
    """A `Tool` is not a `Skill`: it never reaches `call_agent`, so its
    parameter metadata is applied by the tool-call decoder instead."""
    result, model = _run(
        lambda: mod.driver(1),
        [
            make_tool_call_response("bump", json.dumps({"x": -2})),
            make_text_response("done"),
        ],
        TenacityRetryer(),
        caller=LexicalToolExtractor,
    )
    assert result == "done"
    assert not any(c[0] == "bump" for c in mod.calls)


@pytest.mark.xfail(
    reason="the expression pathway evaluates arguments without applying the "
    "parameter's own metadata, so a plain Tool's constraints are enforced in "
    "the JSON pathway only",
)
def test_plain_tool_metadata_is_enforced_in_the_code_pathway(mod):
    result, model = _run(
        lambda: mod.driver(1),
        [
            make_tool_call_response("bump", json.dumps({"call": "bump(-2)"})),
            make_text_response("done"),
        ],
        TenacityRetryer(),
        caller=ExpressionToolCaller,
    )
    assert result == "done"
    assert not any(c[0] == "bump" for c in mod.calls)


# ============================================================================
# Post-conditions: the same annotation, on the way back, by either answer path
# ============================================================================


def test_post_condition_rejects_and_the_model_answers_again(mod):
    result, model = _run(
        lambda: mod.choose(10),
        [_value(3), _value(11)],
        TenacityRetryer(),
    )
    assert result == 11
    assert model.call_count == 2
    assert mod.calls == [("at_least_lo", 3, 10), ("at_least_lo", 11, 10)]


def test_post_condition_message_is_the_feedback_the_model_gets(mod):
    """The message is not incidental -- it is the whole repair instruction."""
    _, model = _run(lambda: mod.choose(10), [_value(3), _value(11)], TenacityRetryer())
    assert "answer must be at least lo=10, got 3" in _prompt_text(model)


def test_post_condition_reads_the_calls_arguments_on_the_direct_path(mod):
    _, _ = _run(lambda: mod.choose(10), [_value(11)])
    assert mod.calls == [("at_least_lo", 11, 10)]


def test_post_condition_reads_the_calls_arguments_on_the_synthesis_path(mod):
    """`write_and_run_body` validates the value its function returned under the
    same environment, so a context-reading post-condition behaves identically
    whichever way the model chose to answer."""
    result, _ = _run(
        lambda: mod.choose(10),
        [
            make_tool_call_response(
                "write_and_run_body",
                json.dumps({"implementation": "def choose(lo):\n    return lo + 1\n"}),
            )
        ],
        FinalBodySynthesizer(),
    )
    assert result == 11
    assert mod.calls == [("at_least_lo", 11, 10)]


def test_post_condition_rejection_on_the_synthesis_path_is_fed_back(mod):
    """A synthesized body that violates the contract is not the answer: the
    failure returns as a tool message and the model submits again."""
    result, model = _run(
        lambda: mod.choose(10),
        [
            make_tool_call_response(
                "write_and_run_body",
                json.dumps({"implementation": "def choose(lo):\n    return 0\n"}),
                tool_call_id="call_1",
            ),
            make_tool_call_response(
                "write_and_run_body",
                json.dumps({"implementation": "def choose(lo):\n    return lo + 5\n"}),
                tool_call_id="call_2",
            ),
        ],
        FinalBodySynthesizer(),
        TenacityRetryer(),
    )
    assert result == 15
    assert model.call_count == 2
    assert ("at_least_lo", 0, 10) in mod.calls


# ============================================================================
# Text answers: a `str` return is prose, annotated or not
# ============================================================================


def test_a_plain_text_return_is_not_boxed(mod):
    _, model = _run(lambda: mod.plain_blurb("otters"), [make_text_response("Otters!")])
    assert model.response_formats == [None]


def test_an_annotated_text_return_is_not_boxed_either(mod):
    """Metadata is the caller's business and says nothing about the wire: a
    response format here would have the model escape its prose into a JSON
    string literal."""
    _, model = _run(lambda: mod.blurb("otters"), [make_text_response("Otters swim.")])
    assert model.response_formats == [None]


def test_a_non_text_return_is_still_boxed(mod):
    _, model = _run(lambda: mod.choose(1), [_value(2)])
    assert model.response_formats[0] is not None


def test_an_annotated_text_answer_comes_back_verbatim(mod):
    """Quotes, punctuation and newlines survive, because nothing re-encoded
    them on the way through."""
    prose = 'Otters "raft"\ntogether.'
    result, _ = _run(lambda: mod.blurb("otters"), [make_text_response(prose)])
    assert result == prose


def test_a_text_post_condition_runs_and_retries(mod):
    result, model = _run(
        lambda: mod.blurb("otters"),
        [
            make_text_response("one two three four five six"),
            make_text_response("Brief."),
        ],
        TenacityRetryer(),
    )
    assert result == "Brief."
    assert model.call_count == 2
    assert "6 words is too many" in _prompt_text(model)


# ============================================================================
# Every kind of metadata: pydantic's functional validators, and the
# `annotated_types` vocabulary pydantic lowers into core-schema constraints
# ============================================================================


def _positive(x: int) -> int:
    """`ValueError` (or `AssertionError`) is what pydantic turns into a
    `ValidationError`; any other exception propagates raw."""
    if x <= 0:
        raise ValueError(f"{x} is not positive")
    return x


def _even(x: int) -> bool:
    return x % 2 == 0


def _thirteen(x: int) -> bool:
    return x == 13


def _wrap(x: typing.Any, handler_) -> int:
    """A wrap validator: it runs around the inner validation, not after it."""
    return handler_(x)


VALIDATOR_KINDS = [
    pytest.param(pydantic.AfterValidator(_positive), 5, 0, id="after"),
    pytest.param(pydantic.BeforeValidator(int), "7", "seven", id="before"),
    pytest.param(pydantic.WrapValidator(_wrap), 5, "seven", id="wrap"),
    pytest.param(pydantic.Field(gt=0), 5, 0, id="field-gt"),
    pytest.param(annotated_types.Gt(0), 5, 0, id="gt"),
    pytest.param(annotated_types.Interval(ge=1, le=9), 5, 10, id="interval"),
    pytest.param(annotated_types.MultipleOf(5), 10, 7, id="multiple-of"),
    pytest.param(annotated_types.Predicate(_even), 4, 3, id="predicate"),
    pytest.param(annotated_types.Not(_thirteen), 5, 13, id="not"),
]

# Metadata that pydantic turns into a JSON-schema *constraint* is also an
# instruction to the model; metadata that becomes a validator function is
# invisible until it rejects. Both are enforced -- only one is advertised.
CONSTRAINING_KINDS = {"field-gt", "gt", "interval", "multiple-of"}


def _skill_with(metadata):
    """A skill whose parameter and return both carry `metadata`."""

    @Skill.define
    def constrained(
        x: typing.Annotated[int, metadata],
    ) -> typing.Annotated[int, metadata]:
        """Echo {x}. Do not use any tools."""

    return constrained


def _unconstrained():
    """The same skill with nothing annotated, as a baseline to compare against."""

    @Skill.define
    def constrained(x: int) -> int:
        """Echo {x}. Do not use any tools."""

    return constrained


def _answer_schema(skill, value):
    """The schema of the response format `skill` is asked to answer in."""
    _, model = _run(lambda: skill(value), [_value(value)])
    return model.response_formats[0].model_json_schema()["properties"]["value"]


def _validated(metadata, value):
    """`value` as the metadata itself would have it -- the same skill carries
    the contract on both ends, so the answer is held to it too."""
    return pydantic.TypeAdapter(typing.Annotated[int, metadata]).validate_python(value)


@pytest.mark.parametrize("metadata,good,bad", VALIDATOR_KINDS)
def test_every_kind_of_metadata_works_as_a_precondition(metadata, good, bad):
    """Each kind reaches the argument, whichever pydantic pathway it takes."""
    skill = _skill_with(metadata)
    result, model = _run(lambda: skill(good), [_value(good)])
    assert result == _validated(metadata, good)
    assert model.call_count == 1

    with pytest.raises(pydantic.ValidationError):
        _run(lambda: skill(bad), [_value(good)])


@pytest.mark.parametrize("metadata,good,bad", VALIDATOR_KINDS)
def test_every_kind_of_metadata_works_as_a_post_condition(metadata, good, bad):
    """And on the way back, where a rejection is retry feedback rather than an
    exception -- so a bad answer costs a turn and a good one is returned."""
    skill = _skill_with(metadata)
    result, model = _run(
        lambda: skill(good), [_value(bad), _value(good)], TenacityRetryer()
    )
    assert result == _validated(metadata, good)
    assert model.call_count == 2


@pytest.mark.parametrize("metadata,good,bad", VALIDATOR_KINDS)
def test_a_constraint_is_advertised_and_a_validator_is_not(
    metadata, good, bad, request
):
    """The one place the kinds visibly differ.

    A constraint pydantic can express in JSON schema is shown to the model as
    part of the response format it is held to, while a functional validator is
    enforced without ever being shown.

    Compared against the same skill with a bare ``int``, rather than against a
    list of schema keywords: what matters is whether the metadata changed what
    the model is asked for, not which keyword pydantic chose to say it with.
    """
    kind = request.node.callspec.id
    annotated = _answer_schema(_skill_with(metadata), good)
    plain = _answer_schema(_unconstrained(), good)
    assert (annotated != plain) is (kind in CONSTRAINING_KINDS), annotated


def test_a_constrained_tool_parameter_stays_strict_schema_legal(mod):
    """A constraint on a *tool*'s parameter rides in the advertised schema
    without costing the strict form the provider requires.

    The strict form is the part worth naming outright -- ``required``,
    ``additionalProperties`` and the ``strict`` flag are the provider's
    contract, not ours. That the constraint reached the schema is again a
    comparison against the unconstrained tool.
    """
    from effectful.handlers.llm.harness.serialization import (
        _NameAndTool,
        _serialize_name_and_tool,
    )

    @Tool.define
    def bounded_bump(x: typing.Annotated[int, pydantic.Field(gt=0)]) -> int:
        """Bump x, which must be positive."""
        return x + 1

    @Tool.define
    def plain_bump(x: int) -> int:
        """Bump x."""
        return x + 1

    def _params(name, tool):
        return _serialize_name_and_tool(_NameAndTool(name, tool))["function"]

    bounded = _params("bounded_bump", bounded_bump)
    plain = _params("plain_bump", plain_bump)

    assert (
        bounded["parameters"]["properties"]["x"]
        != plain["parameters"]["properties"]["x"]
    )
    assert bounded["parameters"]["additionalProperties"] is False
    assert bounded["parameters"]["required"] == ["x"]
    assert bounded["strict"] is True


# ============================================================================
# Additivity: a skill that declares no contracts is untouched
# ============================================================================


def test_an_unannotated_parameter_is_not_validated(mod):
    """No validator runs for a plain annotation, so an ordinary skill pays
    nothing for this feature."""
    _, _ = _run(lambda: mod.echo(mod.Payload(text="hi")), [_value(2)])
    assert mod.calls == []


def test_an_unannotated_argument_reaches_the_skill_unchanged(mod):
    """Identity, not just equality: an unannotated argument is never round
    tripped through the encoding, so nothing is copied or coerced."""
    payload = mod.Payload(text="hi")
    capture = _EnvCapture()
    _, _ = _run(lambda: mod.echo(payload), [_value(2)], capture)
    assert capture.envs[0]["payload"] is payload


def test_an_annotated_argument_is_replaced_by_the_validated_one(mod):
    """The counterpart of the test above, so the two cannot drift: where a
    contract *is* declared, what reaches the model is what the validator
    returned."""
    capture = _EnvCapture()
    _, _ = _run(lambda: mod.announce("hello"), [_value(5)], capture)
    assert capture.envs[0]["text"] == "HELLO"


# ============================================================================
# Live: the same two contracts against a real model
# ============================================================================


@requires_llm
def test_live_post_condition_is_repaired_on_retry():
    """A real model, told nothing about the rule except through the rejection
    message, answers again and satisfies it."""

    def _mentions_otters(answer: str) -> str:
        if "otter" not in answer.lower():
            raise ValueError("the answer must mention otters")
        return answer

    @Skill.define
    def describe(
        animal: str,
    ) -> typing.Annotated[str, pydantic.AfterValidator(_mentions_otters)]:
        """Describe {animal} in one sentence. Do not use any tools."""

    from effectful.handlers.llm.harness import harness

    with handler(harness(num_retries=4, tool_calling="json", eval_provider="none")):
        assert "otter" in describe("a river mammal").lower()


@requires_llm
def test_live_precondition_guard_rejects_off_topic_input():
    """The guardrail shape from ``docs/source/llm_examples/basics``: a
    bool-returning skill, used as the predicate on another skill's parameter."""

    @Skill.define
    def is_about_animals(query: str) -> bool:
        """Is {query} asking about animals? Do not use any tools."""

    @Skill.define
    def answer(
        query: typing.Annotated[str, annotated_types.Predicate(is_about_animals)],
    ) -> str:
        """Answer {query} in one sentence. Do not use any tools."""

    from effectful.handlers.llm.harness import harness

    with handler(harness(num_retries=4, tool_calling="json", eval_provider="none")):
        assert answer("what do otters eat?")
        with pytest.raises(pydantic.ValidationError):
            answer("what is the capital of France?")
