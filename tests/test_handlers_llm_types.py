"""Tests for Agent mixin message sequence semantics."""

import collections.abc
import dataclasses
import inspect
import re
import typing
from dataclasses import dataclass
from pathlib import Path
from types import CodeType

import pydantic
import pytest
from litellm import ModelResponse

from effectful.handlers.llm import Agent, Encodable, Skill, Tool
from effectful.handlers.llm.harness.durability.retrying import TenacityRetryer
from effectful.handlers.llm.harness.durability.transaction import HistoryBuilder
from effectful.handlers.llm.harness.execution.builtin import BuiltinExecutor
from effectful.handlers.llm.harness.hooks import (
    AgentLoop,
    call_agent,
    call_system,
    call_user,
    completion,
)
from effectful.handlers.llm.harness.legibility.lexical import (
    LexicalReaders,
    LexicalToolExtractor,
)
from effectful.handlers.llm.harness.observability.rendering import _message_text
from effectful.handlers.llm.harness.provision.litellm import LiteLLMConfigurer
from effectful.handlers.llm.harness.serialization import (
    _NAME2TOOL_KEY,
    DecodedToolCall,
    PromptSection,
    _advertised_names,
    _NameAndTool,
    _rebase_headings,
    _render_prompt_section,
    format_as_content_blocks,
    to_content_blocks,
)
from effectful.handlers.llm.harness.synthesis.snippet import StatefulReplSynthesizer
from effectful.handlers.llm.harness.validation.mypy import MypyTypeChecker
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled
from tests.conftest import offered_tools, skill_tools


class SkillStringIntp(ObjectInterpretation):
    """Returns the result of skill formatting as a string. Only supports
    skills that produce string prompts.

    """

    @implements(call_agent)
    def _[**P, T](self, skill: Skill[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        bound_args = inspect.signature(skill).bind(*args, **kwargs)
        bound_args.apply_defaults()
        env = skill.__context__.new_child(bound_args.arguments)
        model_input = call_user(
            PromptSection(
                type="prompt_section",
                title=skill.__name__,
                content=format_as_content_blocks(skill.__doc__, env),
            )
        )
        skill_result = model_input["content"]
        assert len(skill_result) == 1
        return skill_result[0]["text"]


def test_skill_formatting_simple():
    @Skill.define
    @staticmethod
    def rhyme(a: str, b: str) -> str:
        """The {a} sat in the {b}."""
        raise NotHandled

    with handler(SkillStringIntp()):
        assert rhyme("cat", "hat").endswith("The cat sat in the hat.")


def test_skill_formatting_method():
    @dataclass
    class User:
        name: str

        @Skill.define
        def greet(self, day: str) -> float:
            """Greet the user '{self.name}' and wish them a good {day}."""
            raise NotHandled

    with handler(SkillStringIntp()):
        user = User("Bob")
        assert user.greet("Monday").endswith(
            "Greet the user 'Bob' and wish them a good Monday."
        )


def _make_skill_in_own_scope():
    """Module-level helper: the skill's lexical scope is this function,
    NOT whatever dynamic caller invokes it."""

    @Skill.define
    def t() -> str:
        """test"""
        raise NotHandled

    return t


class _ModuleLevelA:
    @Skill.define
    def f(self) -> str:
        """Do stuff"""
        raise NotImplementedError


def _define_scoped_skills():
    @Tool.define
    def shown(self) -> int:
        """Should be able to see this tool."""
        return 0

    class A:
        @Skill.define
        def f(self) -> str:
            """test"""
            return ""

    @Skill.define
    def g() -> int:
        """test"""
        return 0

    def _nested():
        nonlocal shown

        @Skill.define
        def h() -> int:
            """test"""
            return 0

        return h

    class B:
        @Skill.define
        def i(self) -> str:
            """test"""
            return ""

        class C:
            @Skill.define
            def j(self) -> str:
                """test"""
                return ""

    return [A().f, g, _nested(), B().i, B.C().j]


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_handlers_llm_provider.py)
# ---------------------------------------------------------------------------


def make_text_response(content: str) -> ModelResponse:
    return ModelResponse(
        id="test",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        model="test-model",
    )


def make_tool_call_response(
    tool_name: str, tool_args: str, tool_call_id: str = "call_1"
) -> ModelResponse:
    return ModelResponse(
        id="test",
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tool_args},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        model="test-model",
    )


class MockCompletionHandler(ObjectInterpretation):
    """Returns pre-configured responses and captures messages sent to the LLM."""

    def __init__(self, responses: list[ModelResponse]):
        self.responses = responses
        self.call_count = 0
        self.received_messages: list[list] = []

    @implements(completion)
    def _completion(self, messages=None, **kwargs):
        self.received_messages.append(list(messages) if messages else [])
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


def _document_headings(md: str) -> list[str]:
    """The ATX headings of `md`, ignoring fenced code blocks.

    The system prompt embeds the Skill's module -- this very file, for a
    Skill defined in a test -- as a fenced block, so a plain substring check
    for a heading matches the assertion literal in that embedded source rather
    than the assembled document.  Skipping fences asserts on the real outline.
    """
    headings: list[str] = []
    fence: str | None = None
    for line in md.splitlines():
        stripped = line.lstrip()
        if fence is None and (stripped.startswith("```") or stripped.startswith("~~~")):
            fence = stripped[:3]
        elif fence is not None and stripped.startswith(fence):
            fence = None
        elif fence is None and re.match(r"^#{1,6}\s", line):
            headings.append(line.rstrip())
    return headings


def assert_single_system_message_first(messages):
    roles = [m["role"] for m in messages]
    assert roles.count("system") == 1
    assert roles[0] == "system"


# ---------------------------------------------------------------------------
# Agent subclass used by most tests
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ChatBot(Agent):
    """You are a chat agent for history-accumulation tests.
    Your goal is to respond to `send` calls consistently across turns.
    """

    bot_name: str = dataclasses.field(default="ChatBot")

    @Skill.define
    def send(self, user_input: str) -> str:
        """A friendly bot named {self.bot_name}. User writes: {user_input}"""
        raise NotHandled


class _DesignerAgent(Agent):
    """You are an agent for nested-skill regression tests.
    Your goal is to call nested tools/skills and return a final response.
    """

    @Skill.define
    def nested_check(self, payload: str) -> str:
        """Check: {payload}. Do not use tools."""
        raise NotHandled

    @Tool.define
    def nested_tool(self, payload: str) -> str:
        """Check payload by calling a nested LLM skill."""
        return self.nested_check(payload)

    @Skill.define
    def outer(self, payload: str) -> str:
        """Call `nested_tool` for: {payload}, then return final answer."""
        raise NotHandled


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _decode_call(name2tool, advertised, arguments='{"payload": "x"}'):
    """Decode a tool call naming `advertised`, against the map that advertised it."""
    return pydantic.TypeAdapter(Encodable[DecodedToolCall]).validate_python(
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": advertised, "arguments": arguments},
        },
        context={_NAME2TOOL_KEY: name2tool},
    )


def test_agent_method_tool_advertised_name_matches_decode_key():
    """A tool is advertised to the LLM under the same name that tool-call decoding
    resolves it by, so the serialize->advertise->decode round-trip is self-consistent.
    `call_assistant` assigns the names (`_advertised_names`) and decodes through the
    same map; this checks an Agent method tool makes that round trip, no LLM required."""
    agent = _DesignerAgent()
    # The map `call_assistant` builds from the tools in scope.
    name2tool = _advertised_names(offered_tools({"self": agent}))
    tool = name2tool["nested_tool"]

    # Serialize exactly as `call_assistant` advertises it.
    spec = pydantic.TypeAdapter(Encodable[_NameAndTool]).dump_python(
        _NameAndTool("nested_tool", tool), mode="json"
    )
    advertised = spec["function"]["name"]
    assert advertised == "nested_tool"  # advertised under the key decode resolves by

    # A tool call using the advertised name decodes back to the same tool object.
    assert _decode_call(name2tool, advertised).tool is tool


def test_same_agent_method_on_two_instances_gets_two_names():
    """Two instances of an Agent contribute the same method under one `__name__`.
    `_advertised_names` still gives each a name of its own, and each decodes back to
    the tool bound to *its* instance -- the case the old ``__name__``-keyed map could
    only assert its way out of."""
    first, second = _DesignerAgent(), _DesignerAgent()
    name2tool = _advertised_names(
        offered_tools({"first": first, "second": second}),
    )

    names = sorted(n for n, t in name2tool.items() if t.__name__ == "nested_tool")
    assert names == ["nested_tool", "nested_tool_2"]

    bound = {_decode_call(name2tool, n).tool for n in names}
    assert bound == {first.nested_tool, second.nested_tool}


class TestAgentHistoryAccumulation:
    """History accumulates across sequential calls on the same instance."""

    def test_second_call_sees_prior_messages(self):
        mock = MockCompletionHandler(
            [
                make_text_response("hi"),
                make_text_response("good"),
            ]
        )
        bot = ChatBot()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(mock),
        ):
            bot.send("hello")
            bot.send("how are you")

        # First call: system + user → 2 messages
        assert len(mock.received_messages[0]) > 0

        # Second call: previous system + user + assistant, PLUS new system + user → 5
        assert len(mock.received_messages[1]) > len(mock.received_messages[0])

        # Verify roles in second call
        roles = [m["role"] for m in mock.received_messages[1]]
        assert roles.count("assistant") >= 1
        assert roles.count("user") >= 2

    def test_history_contains_all_messages_after_two_calls(self):
        mock = MockCompletionHandler(
            [
                make_text_response("r1"),
                make_text_response("r2"),
            ]
        )
        bot = ChatBot()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(mock),
        ):
            bot.send("a")
            bot.send("b")

        # After two complete calls the history should have:
        #   call 1: system, user, assistant  (3)
        #   call 2: system, user, assistant  (3)
        assert len(bot.__history__) >= 4


class TestAgentIsolation:
    """Each agent instance has independent history; non-agent skills are unaffected."""

    def test_two_agents_have_independent_histories(self):
        mock = MockCompletionHandler(
            [
                make_text_response("from bot1"),
                make_text_response("from bot2"),
            ]
        )
        bot1 = ChatBot()
        bot2 = ChatBot()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(mock),
        ):
            bot1.send("msg for bot1")
            bot2.send("msg for bot2")

        # bot2's call should NOT contain bot1's messages — only system + user
        assert len(mock.received_messages[1]) == len(mock.received_messages[0])

        # Each bot made exactly one call, so their histories should be equal in size
        assert len(bot1.__history__) == len(bot2.__history__)

    def test_non_agent_skill_gets_fresh_sequence(self):
        @Skill.define
        def standalone(topic: str) -> str:
            """Write about {topic}."""
            raise NotHandled

        mock = MockCompletionHandler(
            [
                make_text_response("agent reply"),
                make_text_response("standalone reply"),
                make_text_response("agent reply 2"),
            ]
        )
        bot = ChatBot()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(mock),
        ):
            bot.send("hello")
            standalone("fish")
            bot.send("bye")

        # standalone (call index 1) should see only system + user (fresh sequence)
        assert len(mock.received_messages[1]) >= 1

        # bot's third call (call index 2) should see its accumulated history
        # but NOT the standalone messages
        assert len(mock.received_messages[2]) >= 3


class TestSystemPromptInvariant:
    """Exactly one system message is sent and it appears first."""

    def test_agent_first_call_has_one_system_message(self):
        mock = MockCompletionHandler([make_text_response("hi")])
        bot = ChatBot()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(mock),
        ):
            bot.send("hello")

        assert_single_system_message_first(mock.received_messages[0])

    def test_agent_second_call_has_one_system_message(self):
        mock = MockCompletionHandler(
            [
                make_text_response("r1"),
                make_text_response("r2"),
            ]
        )
        bot = ChatBot()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(mock),
        ):
            bot.send("first")
            bot.send("second")

        assert_single_system_message_first(mock.received_messages[0])
        assert_single_system_message_first(mock.received_messages[1])

    def test_nested_agent_flow_has_one_system_message_per_round(self):
        mock = MockCompletionHandler(
            [
                make_tool_call_response("nested_tool", '{"payload": "demo"}'),
                make_text_response("inner"),
                make_text_response("outer"),
            ]
        )
        agent = _DesignerAgent()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(mock),
        ):
            agent.outer("demo")

        for messages in mock.received_messages:
            assert_single_system_message_first(messages)

    def test_retry_flow_has_one_system_message_per_attempt(self):
        class RetryAgent(Agent):
            """You are a retry-flow test agent.
            Your goal is to produce an integer response after retry feedback.
            """

            @Skill.define
            def pick_number(self) -> int:
                """Pick a number."""
                raise NotHandled

        mock = MockCompletionHandler(
            [
                make_text_response('"not_an_int"'),
                make_text_response('{"value": 7}'),
            ]
        )

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(TenacityRetryer()),
            handler(mock),
        ):
            assert RetryAgent().pick_number() == 7

        assert len(mock.received_messages) == 2
        assert_single_system_message_first(mock.received_messages[0])
        assert_single_system_message_first(mock.received_messages[1])

    def test_non_agent_skill_calls_have_one_system_message(self):
        @Skill.define
        def standalone(topic: str) -> str:
            """Write about {topic}."""
            raise NotHandled

        mock = MockCompletionHandler(
            [
                make_text_response("a"),
                make_text_response("b"),
            ]
        )

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(mock),
        ):
            standalone("fish")
            standalone("birds")

        assert_single_system_message_first(mock.received_messages[0])
        assert_single_system_message_first(mock.received_messages[1])

    def test_system_message_assembled_from_introspection(self):
        @Skill.define
        def standalone(topic: str) -> str:
            """Write about {topic}."""
            raise NotHandled

        mock = MockCompletionHandler([make_text_response("ok")])
        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(mock),
        ):
            standalone("fish")

        assert_single_system_message_first(mock.received_messages[0])
        content = _message_text(mock.received_messages[0][0]["content"])
        # The content is a Markdown document introspected from the Skill and
        # rendered from the assembled prompt, not a stored attribute: the task
        # half is a `#` section and the Skill's spec a `###` subsection of it.
        headings = _document_headings(content)
        assert "# `standalone(topic: str) -> str`" in headings
        assert "### `standalone(topic: str) -> str`" in headings
        assert "Write about {topic}." in content


class TestSystemPromptDocument:
    """The system prompt is a `PromptSection` document that handlers rewrite on
    the way down, and that `call_system` renders to content blocks."""

    def _prompt(self, *content) -> PromptSection:
        return self._section("doc", list(content))

    def _section(self, title, content) -> PromptSection:
        return PromptSection(
            type="prompt_section",
            title=title,
            content=to_content_blocks(content) if isinstance(content, str) else content,
        )

    def test_sections_nest_and_text_rebases_below_its_section(self):
        """A docstring authored with its own `##`-rooted headings nests beneath
        whatever section ends up carrying it, so the document has one outline."""
        inner = self._section("Inner", "## Own heading\n\nbody")
        prompt = self._prompt(
            self._section("Outer", [*to_content_blocks("intro"), inner])
        )
        assert (
            _message_text(_render_prompt_section(prompt))
            == "# Outer\n\nintro\n\n## Inner\n\n### Own heading\n\nbody"
        )

    def test_empty_sections_are_omitted(self):
        """An unfilled section -- `Harness` under a stack that installs no
        capability handlers -- leaves no stray heading behind."""
        prompt = self._prompt(
            self._section("Harness", []),
            self._section("Blank", ""),
            self._section("Kept", "body"),
        )
        assert _message_text(_render_prompt_section(prompt)) == "# Kept\n\nbody"

    def test_non_text_blocks_reach_the_system_prompt(self):
        """The document is a block list, not a Markdown string, so a section may
        carry an image."""
        image = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA"},
        }
        prompt = self._prompt(
            self._section(
                "Figure",
                [*to_content_blocks("before"), image, *to_content_blocks("after")],
            )
        )
        blocks = _render_prompt_section(prompt)
        assert [b["type"] for b in blocks] == ["text", "image_url", "text"]
        assert blocks[0]["text"] == "# Figure\n\nbefore"
        assert blocks[1] == image
        assert blocks[2]["text"] == "after"

    def test_the_user_message_carries_the_skill_header(self):
        """`call_user` renders its argument as a *child* of an enclosing
        document, for the same reason `call_system` assembles one: a section
        handed to `_render_prompt_section` at level 0 is the document itself,
        whose title labels it rather than appearing in it. Passed straight
        through, the Skill's header would be computed and then dropped."""

        @Skill.define
        def rhyme(a: str, b: str) -> str:
            """The {a} sat in the {b}."""
            raise NotHandled

        bound = inspect.signature(rhyme).bind("cat", "hat")
        bound.apply_defaults()
        env = rhyme.__context__.new_child(bound.arguments)

        message = call_user(AgentLoop()._skill_user_prompt(rhyme, env))
        assert (
            _message_text(message["content"])
            == "# rhyme(a: str, b: str) -> str\n\nThe cat sat in the hat."
        )

    def test_braces_in_a_signature_survive_the_header(self):
        """Only the docstring is interpolated, so a `{}` default in the
        signature reaches the heading as written -- escaping it here would
        show up literally."""

        @Skill.define
        def defaulted(a: str, opts: dict = {}) -> str:
            """Answer about {a}."""
            raise NotHandled

        bound = inspect.signature(defaulted).bind("x")
        bound.apply_defaults()
        env = defaulted.__context__.new_child(bound.arguments)

        message = call_user(AgentLoop()._skill_user_prompt(defaulted, env))
        assert _message_text(message["content"]).startswith(
            "# defaulted(a: str, opts: dict = {}) -> str"
        )

    def test_a_handler_documents_itself_with_its_own_docstring(self):
        """A handler adds the section describing itself to `harness_prompt` and
        forwards, leaving the section it was handed untouched."""

        @Skill.define
        def standalone(topic: str) -> str:
            """Write about {topic}."""
            raise NotHandled

        class Documented(ObjectInterpretation):
            """A capability of the harness, described for the model."""

            @implements(call_system)
            def _call_system(self, harness_prompt, agent_prompt):
                return fwd(
                    PromptSection(
                        type="prompt_section",
                        title=harness_prompt["title"],
                        content=[
                            *harness_prompt["content"],
                            PromptSection(
                                type="prompt_section",
                                title="Documented",
                                content=to_content_blocks(
                                    inspect.getdoc(type(self)) or ""
                                ),
                            ),
                        ],
                    ),
                    agent_prompt,
                )

        harness_prompt = self._section("Harness", [])
        with handler(Documented()):
            content = _message_text(
                call_system(
                    harness_prompt, AgentLoop()._skill_system_prompt(standalone)
                )["content"]
            )

        assert "# Harness\n\n## Documented" in content
        assert "A capability of the harness, described for the model." in content
        assert harness_prompt["content"] == []


class TestAgentDocstringFallback:
    """Agent subclasses' class docstrings flow into the assembled system message."""

    def _system_content(self, skill):
        message = call_system(
            PromptSection(type="prompt_section", title="Harness", content=[]),
            AgentLoop()._skill_system_prompt(skill),
        )
        return _message_text(message["content"])

    def test_missing_docstring_uses_inherited_doc(self):
        class MissingDocAgent(Agent):
            @Skill.define
            def act(self) -> str:
                """Do something."""
                raise NotHandled

        assert MissingDocAgent.__doc__ is None
        content = self._system_content(MissingDocAgent().act)
        # No subclass docstring -> the Agent base-class docstring is used as the
        # "## Agent" section's prose (inspect.getdoc walks the MRO).
        agent_doc = inspect.getdoc(Agent)
        assert agent_doc is not None
        assert "## Agent `MissingDocAgent`" in content
        # It arrives whole, but with its own headings rebased to sit below that
        # section -- which is at `##`, so its subsections land at `###`.
        assert _rebase_headings(agent_doc, 3) in content

    def test_non_empty_docstring_overrides_inherited_doc(self):
        class ValidDocAgent(Agent):
            """You are a valid-docstring test agent.
            Your goal is to satisfy the explicit Agent docstring requirement.
            """

            @Skill.define
            def act(self) -> str:
                """Do something."""
                raise NotHandled

        assert ValidDocAgent.__doc__ is not None
        content = self._system_content(ValidDocAgent().act)
        assert "You are a valid-docstring test agent." in content


class TestAgentCachedProperty:
    """__history__ is lazily created per instance without requiring __init__."""

    def test_no_init_required(self):
        class MinimalAgent(Agent):
            """You are a minimal cached-property test agent.
            Your goal is to expose lazily initialized Agent state.
            """

            @Skill.define
            def greet(self, name: str) -> str:
                """Hello {name}."""
                raise NotHandled

        agent = MinimalAgent()
        # Should be an empty message sequence, created on first access
        assert isinstance(agent.__history__, collections.abc.MutableSequence)
        assert len(agent.__history__) == 0

    def test_subclass_with_own_init(self):
        class CustomAgent(Agent):
            """You are a custom-init test agent.
            Your goal is to ensure Agent mixin behavior survives custom `__init__`.
            """

            def __init__(self, name: str):
                self.name = name

            @Skill.define
            def greet(self) -> str:
                """Say hello."""
                raise NotHandled

        agent = CustomAgent("Alice")
        assert agent.name == "Alice"
        assert isinstance(agent.__history__, collections.abc.MutableSequence)

    def test_history_is_per_instance(self):
        a = ChatBot()
        b = ChatBot()
        a.__history__.append({"role": "user", "content": "x"})
        assert len(b.__history__) == 0


class TestAgentWithToolCalls:
    """Agent methods that trigger tool calls maintain correct history."""

    def test_tool_call_results_appear_in_history(self):
        @Tool.define
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        class MathAgent(Agent):
            """You are a math-tool test agent.
            Your goal is to call arithmetic tools and return a textual answer.
            """

            @Skill.define
            def compute(self, question: str) -> str:
                """Answer: {question}"""
                raise NotHandled

        mock = MockCompletionHandler(
            [
                make_tool_call_response("add", '{"a": 2, "b": 3}'),
                make_text_response("The answer is 5"),
            ]
        )
        agent = MathAgent()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(mock),
        ):
            result = agent.compute("what is 2+3?")

        assert result == "The answer is 5"

        # History should contain: system, user, assistant (tool_call),
        # tool (result), assistant (final)
        roles = [m["role"] for m in agent.__history__]
        assert "tool" in roles
        assert roles.count("assistant") == 2


class TestAgentWithRetryHandler:
    """RetryLLMHandler composes correctly with Agent history."""

    def test_failed_retries_dont_pollute_history(self):
        mock = MockCompletionHandler(
            [
                # First attempt: invalid result for int
                make_text_response('"not_an_int"'),
                # Retry: valid
                make_text_response('{"value": 42}'),
            ]
        )

        class NumberAgent(Agent):
            """You are a numeric retry test agent.
            Your goal is to return an integer after potential retry corrections.
            """

            @Skill.define
            def pick_number(self) -> int:
                """Pick a number."""
                raise NotHandled

        agent = NumberAgent()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(TenacityRetryer()),
            handler(mock),
        ):
            result = agent.pick_number()

        assert result == 42

        # The malformed assistant message and error feedback from the retry
        # should NOT appear in the agent's history. Only the final successful
        # assistant message should be there.
        roles = {m["role"] for m in agent.__history__}
        assert {"user", "assistant"} == roles - {"system"}


class TestNestedSkillCalling:
    """Issue #560: nested Skill invocation via tool on the same Agent.

    When a Skill triggers a tool call whose implementation invokes
    another Skill on the same Agent, the inner call must:
    - work on a fresh copy of the agent's history
    - NOT write its messages back to agent.__history__
    - return its result correctly so the outer skill can continue
    """

    def test_same_agent_nested_skill_via_tool(self):
        """The scenario from issue #560 completes without error."""
        mock = MockCompletionHandler(
            [
                make_tool_call_response("nested_tool", '{"payload": "demo"}'),
                make_text_response("check passed"),
                make_text_response("all good"),
            ]
        )
        agent = _DesignerAgent()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(mock),
        ):
            result = agent.outer("demo")

        assert result == "all good"

    def test_only_outermost_writes_to_history(self):
        """Inner skill's messages are absent from agent.__history__."""
        mock = MockCompletionHandler(
            [
                make_tool_call_response("nested_tool", '{"payload": "demo"}'),
                make_text_response("inner"),
                make_text_response("outer"),
            ]
        )
        agent = _DesignerAgent()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(mock),
        ):
            agent.outer("demo")

        roles = [m["role"] for m in agent.__history__]
        # Outer call produces: user, assistant(tool_call), tool, assistant(final)
        # Inner call's user + assistant are NOT written back
        assert set(roles) <= {"system", "user", "assistant", "tool"}
        assert roles.count("system") == 1
        assert roles.count("user") == 1
        assert roles.count("assistant") == 2  # tool_call + final
        assert roles.count("tool") == 1

    def test_inner_skill_gets_fresh_messages(self):
        """The nested skill's LLM call sees only its own system + user,
        not the outer skill's in-flight messages."""
        mock = MockCompletionHandler(
            [
                make_tool_call_response("nested_tool", '{"payload": "demo"}'),
                make_text_response("inner"),
                make_text_response("outer"),
            ]
        )
        agent = _DesignerAgent()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(mock),
        ):
            agent.outer("demo")

        # Call 0: outer's first call_assistant → [user]
        # Call 1: inner's call_assistant → [user] (fresh, from empty history)
        # Call 2: outer's second call_assistant → [user, assistant(tc), tool]
        inner_roles = [m["role"] for m in mock.received_messages[1]]
        assert {"user"} <= set(inner_roles) <= {"system", "user"}

    def test_inner_skill_sees_prior_completed_history(self):
        """After a previous top-level call, the nested inner skill sees
        the completed history but NOT the current outer call's in-flight messages."""
        mock = MockCompletionHandler(
            [
                # First call: direct answer (no tool call)
                make_text_response("first"),
                # Second call: tool → nested → final
                make_tool_call_response("nested_tool", '{"payload": "demo"}'),
                make_text_response("inner"),
                make_text_response("second"),
            ]
        )
        agent = _DesignerAgent()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(mock),
        ):
            agent.outer("first")
            agent.outer("second")

        # After first call, agent.__history__ has 2 messages (user + assistant).
        # Second outer call (call 1): starts from history(2) + own user = 3.
        # Inner call (call 2): starts from history(2) + own user = 3.
        # Both see the same base history. If inner saw the outer's in-flight
        # messages (user, assistant(tc)), it would have more.
        assert len(mock.received_messages[1]) == len(mock.received_messages[2])

        # Inner call sees more than just its own user message (it has history)
        assert len(mock.received_messages[2]) > 1

    def test_sequential_call_after_nested_sees_history(self):
        """A follow-up top-level call sees the first call's full history."""
        mock = MockCompletionHandler(
            [
                # First call: tool → nested → final
                make_tool_call_response("nested_tool", '{"payload": "demo"}'),
                make_text_response("inner"),
                make_text_response("first"),
                # Second call: direct answer
                make_text_response("second"),
            ]
        )
        agent = _DesignerAgent()

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(mock),
        ):
            r1 = agent.outer("first")
            r2 = agent.outer("second")

        assert r1 == "first"
        assert r2 == "second"

        # Second call (mock index 3) should see the full history from the first
        # call (4 messages: user+assistant(tc)+tool+assistant) plus its own
        # user = 5 total.
        assert len(mock.received_messages[3]) > len(mock.received_messages[0])
        second_call_roles = [m["role"] for m in mock.received_messages[3]]
        assert second_call_roles.count("assistant") >= 2  # from first call's history


class _HelperAgent(Agent):
    """You are a helper agent for cross-agent nesting regression tests."""

    @Skill.define
    def answer(self, q: str) -> str:
        """Answer: {q}. Do not use tools."""
        raise NotHandled


class _OrchestratorAgent(Agent):
    """You orchestrate work by delegating to a helper agent."""

    def __init__(self, helper: "_HelperAgent"):
        self._helper = helper

    @Tool.define
    def ask_helper(self, q: str) -> str:
        """Ask the helper agent a question."""
        return self._helper.answer(q)

    @Skill.define
    def run(self, task: str) -> str:
        """Task: {task}"""
        raise NotHandled


class TestCrossAgentNestedSkillCalling:
    """A tool call that delegates to a *different* Agent must write back that
    agent's own history, not be mistaken for a same-agent nested call.

    `HistoryBuilder.call_skill` keys its outermost-call detection on the
    identity of each agent's `__history__` (`agents_called`), so being inside
    *some* transaction is not enough to make a call look nested: only a second
    call against the same agent's history is, and a different agent invoked
    mid-call still writes back.
    """

    def test_delegated_agent_history_is_written_back(self):
        """The helper's own history is populated after being called via a tool."""
        mock = MockCompletionHandler(
            [
                make_tool_call_response("ask_helper", '{"q": "demo"}'),
                make_text_response("helper's answer"),
                make_text_response("final answer"),
            ]
        )
        helper = _HelperAgent()
        orch = _OrchestratorAgent(helper)

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(mock),
        ):
            result = orch.run("do the thing")

        assert result == "final answer"
        assert len(helper.__history__) > 0
        helper_roles = [m["role"] for m in helper.__history__]
        assert helper_roles.count("user") == 1
        assert helper_roles.count("assistant") == 1

    def test_followup_call_on_delegate_sees_only_its_own_history(self):
        """A later top-level call on the helper sees its own accumulated
        history, not the orchestrator's."""
        mock = MockCompletionHandler(
            [
                make_tool_call_response("ask_helper", '{"q": "first"}'),
                make_text_response("helper's first answer"),
                make_text_response("first orchestrator answer"),
                make_text_response("helper's second answer"),
            ]
        )
        helper = _HelperAgent()
        orch = _OrchestratorAgent(helper)

        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(mock),
        ):
            orch.run("first task")
            helper.answer("second question")

        # helper's second (direct) call should see its own prior history
        # (system + user + assistant from the delegated call) plus its own
        # new user message.
        followup_roles = [m["role"] for m in mock.received_messages[3]]
        assert followup_roles.count("assistant") >= 1


# ---------------------------------------------------------------------------
# Skill method and scoping tests (moved from test_handlers_llm_types.py)
# ---------------------------------------------------------------------------


def test_skill_method():
    """Test that methods can be used as skills."""
    local_variable = None  # noqa: F841

    @dataclass
    class A(Agent):
        """You are a skill-method test agent.
        Your goal is to expose method tools and method skills correctly.
        """

        x: int

        @Tool.define
        def random(self) -> int:
            """Returns a random number, chosen by fair dice roll."""
            return 4

        @Skill.define
        def f(self) -> int:
            """What is the number after 3?"""
            raise NotHandled

    a = A(0)
    assert isinstance(a.f, Skill)
    assert a.random in skill_tools(a.f)
    # f is the skill itself — found via self but correctly removed (non-recursive)
    assert a.f not in skill_tools(a.f)
    assert any(t() == 4 for t in skill_tools(a.f) if t is a.random)

    class B(A):
        """You are a derived skill-method test agent.
        Your goal is to add inherited-tool coverage for method-skill tests.
        """

        @Tool.define
        def reverse(self, s: str) -> str:
            """Reverses a string."""
            return str(reversed(s))

    b = B(1)
    assert isinstance(b.f, Skill)
    assert b.random in skill_tools(b.f)
    assert b.reverse in skill_tools(b.f)


def test_skill_method_nested_class():
    """Test that skill methods work on nested classes."""
    local_variable = "test"  # noqa: F841

    @Tool.define
    def random() -> int:
        """Returns a random number, chosen by fair dice roll."""
        return 4

    @dataclass
    class A:
        x: int

        @dataclass
        class B:
            y: bool

            @Skill.define
            def f(self) -> int:
                """What is the number after 3?"""
                raise NotHandled

    a = A.B(True)
    assert isinstance(a.f, Skill)
    tools = skill_tools(a.f)
    # random is found via the enclosing function scope
    assert random in tools
    # f is the skill itself — found via self but correctly removed (non-recursive)
    assert a.f not in tools
    assert random() == 4


def test_skill_method_module():
    """Test that skill methods work when defined on module-level classes."""
    a = _ModuleLevelA()
    assert isinstance(a.f, Skill)


def test_skill_method_scoping():
    @Tool.define
    def hidden(self) -> int:
        """Shouldn't be able to see this tool."""
        return 0

    skills = _define_scoped_skills()
    for t in skills:
        assert isinstance(t, Skill)
        assert "shown" in t.__context__
        assert "hidden" not in t.__context__


# ---------------------------------------------------------------------------
# Lexical scope collection
# ---------------------------------------------------------------------------


class TestLexicalScopeCollection:
    """Tests that Skill.define follows Python's lexical scope rules."""

    def test_class_body_locals_excluded_from_context(self):
        """Class body variables (like __qualname__, field defaults) should not
        appear as tools, matching Python's rule that class bodies are not
        lexical scopes for methods."""

        @dataclass
        class Foo:
            x: int

            @Tool.define
            def helper(self) -> int:
                """A tool."""
                return 42

            @Skill.define
            def ask(self) -> str:
                """Ask something."""
                raise NotHandled

        foo = Foo(0)
        # Class body metadata should not leak into context
        assert "__qualname__" not in foo.ask.__context__
        assert "__firstlineno__" not in foo.ask.__context__
        # But the enclosing function scope is visible
        assert "Foo" in foo.ask.__context__

    def test_enclosing_function_scope_visible(self):
        """Tools defined in the enclosing function are visible to skills
        defined inside a class in that function."""

        @Tool.define
        def helper() -> int:
            """A helper tool."""
            return 99

        class Bar:
            @Skill.define
            def ask(self) -> str:
                """Ask something."""
                raise NotHandled

        bar = Bar()
        assert helper in skill_tools(bar.ask)

    def test_dynamic_caller_not_leaked(self):
        """Variables from a dynamic caller (not lexical enclosure) should not
        appear in the skill's context."""
        leaked = False  # noqa: F841

        # _make_skill_in_own_scope is defined at module level, so
        # this test method is a dynamic caller, not a lexical encloser.
        t = _make_skill_in_own_scope()
        assert "leaked" not in t.__context__

    def test_class_method_tools_discovered_via_self(self):
        """After skipping the class body, tools on an Agent are still
        discoverable through the bound `self` instance."""

        @dataclass
        class Widget(Agent):
            """You are a class-body discovery test agent.
            Your goal is to expose tools discovered via bound `self`.
            """

            @Tool.define
            def measure(self) -> int:
                """Measure the widget."""
                return 10

            @Skill.define
            def describe(self) -> str:
                """Describe this widget."""
                raise NotHandled

        w = Widget()
        assert w.measure in skill_tools(w.describe)
        # The skill itself is not in tools (non-recursive)
        assert w.describe not in skill_tools(w.describe)

    def test_inherited_tools_visible(self):
        """Tools from a base Agent class are visible through the instance."""

        class Base(Agent):
            """You are a base-class tool test agent.
            Your goal is to provide a tool inherited by derived agents.
            """

            @Tool.define
            def base_tool(self) -> int:
                """A base tool."""
                return 1

        class Derived(Base):
            """You are a derived-class tool test agent.
            Your goal is to consume tools inherited from a base agent class.
            """

            @Skill.define
            def ask(self) -> str:
                """Ask something."""
                raise NotHandled

        d = Derived()
        assert d.base_tool in skill_tools(d.ask)

    def test_tool_in_enclosing_function_visible_through_class(self):
        """function -> class -> Skill.define: tool in the function is visible."""

        @Tool.define
        def outer_tool() -> int:
            """Outer tool."""
            return 1

        class Inner:
            @Skill.define
            def ask(self) -> str:
                """Ask something."""
                raise NotHandled

        assert outer_tool in skill_tools(Inner().ask)

    def test_tool_in_enclosing_function_visible_through_nested_classes(self):
        """function -> class -> class -> Skill.define: tool in the function
        is still visible after skipping multiple class body frames."""

        @Tool.define
        def outer_tool() -> int:
            """Outer tool."""
            return 1

        class Outer:
            class Inner:
                @Skill.define
                def ask(self) -> str:
                    """Ask something."""
                    raise NotHandled

        assert outer_tool in skill_tools(Outer.Inner().ask)

    def test_nested_function_then_class(self):
        """function -> function -> class -> Skill.define: all enclosing
        function scopes are visible, matching Python's lexical scope rules."""

        def _make():
            @Tool.define
            def inner_tool() -> int:
                """Inner tool."""
                return 2

            class MyClass:
                @Skill.define
                def ask(self) -> str:
                    """Ask."""
                    raise NotHandled

            return MyClass, inner_tool

        outer_var = True  # noqa: F841
        cls, inner_tool = _make()
        assert inner_tool in skill_tools(cls().ask)
        # The test method is a lexical encloser of _make, so its locals
        # are visible — matching Python's actual scoping rules.
        assert "outer_var" in cls().ask.__context__

    def test_nested_function_scopes_skill_at_inner(self):
        """function -> function -> Skill.define: skill sees all
        enclosing function scopes, matching Python's lexical scope rules."""

        def _outer():
            outer_var = "outer"  # noqa: F841

            def _inner():
                inner_var = "inner"  # noqa: F841

                @Skill.define
                def t() -> str:
                    """test"""
                    raise NotHandled

                return t

            return _inner()

        t = _outer()
        assert "inner_var" in t.__context__
        assert "outer_var" in t.__context__


# ---------------------------------------------------------------------------
# staticmethod / classmethod Skills
# ---------------------------------------------------------------------------


class TestStaticAndClassMethodSkills:
    """Tests for @Skill.define applied to staticmethod and classmethod descriptors."""

    def test_staticmethod_skill_in_class(self):
        """@Skill.define @staticmethod in a class body produces a Skill
        accessible as a class attribute."""

        class MyClass:
            @Skill.define
            @staticmethod
            def ask(question: str) -> str:
                """Answer: {question}"""
                raise NotHandled

        assert isinstance(MyClass.ask, Skill)
        assert isinstance(MyClass().ask, Skill)

    def test_staticmethod_skill_callable(self):
        """Staticmethod Skills can be called through a handler."""

        class MyClass:
            @Skill.define
            @staticmethod
            def ask(question: str) -> str:
                """Answer: {question}"""
                raise NotHandled

        mock = MockCompletionHandler([make_text_response("42")])
        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(mock),
        ):
            result = MyClass.ask("what is 6*7?")
        assert result == "42"

    def test_staticmethod_skill_captures_enclosing_scope(self):
        """A staticmethod Skill captures the enclosing function scope,
        even through the re-entrant _define_staticmethod call."""

        @Tool.define
        def helper() -> int:
            """A helper tool."""
            return 99

        class MyClass:
            @Skill.define
            @staticmethod
            def ask(x: int) -> int:
                """Compute {x}."""
                raise NotHandled

        assert helper in skill_tools(MyClass.ask)

    def test_staticmethod_skill_excludes_class_body(self):
        """A staticmethod Skill does not capture class body locals."""

        class MyClass:
            class_var = 42  # noqa: F841

            @Skill.define
            @staticmethod
            def ask() -> str:
                """Ask."""
                raise NotHandled

        assert "class_var" not in MyClass.ask.__context__

    def test_classmethod_skill_in_class(self):
        """@Skill.define @classmethod in a class body produces a Skill
        accessible as a class attribute (lazily via _ClassMethodOpDescriptor)."""

        class MyClass:
            @Skill.define
            @classmethod
            def ask(cls, question: str) -> str:
                """Answer: {question}"""
                raise NotHandled

        assert isinstance(MyClass.ask, Skill)

    def test_classmethod_skill_callable(self):
        """Classmethod Skills can be called through a handler."""

        class MyClass:
            @Skill.define
            @classmethod
            def ask(cls, question: str) -> str:
                """Answer: {question}"""
                raise NotHandled

        mock = MockCompletionHandler([make_text_response("yes")])
        with (
            handler(AgentLoop()),
            handler(LexicalToolExtractor()),
            handler(LiteLLMConfigurer()),
            handler(HistoryBuilder()),
            handler(mock),
        ):
            result = MyClass.ask("is the sky blue?")
        assert result == "yes"

    def test_classmethod_skill_signature_excludes_cls(self):
        """The classmethod Skill's signature does not include cls,
        since the classmethod descriptor binds it automatically."""

        class MyClass:
            @Skill.define
            @classmethod
            def ask(cls, question: str) -> str:
                """Answer: {question}"""
                raise NotHandled

        sig = inspect.signature(MyClass.ask)
        assert "cls" not in sig.parameters
        assert "question" in sig.parameters

    def test_agent_skips_staticmethod_skill(self):
        """Agent.__init_subclass__ does not wrap staticmethod Skills
        in cached_property — they remain accessible as plain Skills."""

        class MyAgent(Agent):
            """You are a staticmethod-skill test agent.
            Your goal is to verify Agent wrapping does not alter static skills.
            """

            @Skill.define
            def instance_method(self) -> str:
                """Say hello."""
                raise NotHandled

            @Skill.define
            @staticmethod
            def static_method(x: int) -> int:
                """Double {x}."""
                raise NotHandled

        agent = MyAgent()
        # instance_method is wrapped by Agent into a cached_property
        assert isinstance(agent.instance_method, Skill)
        # static_method remains a plain Skill accessible on class and instance
        assert isinstance(MyAgent.static_method, Skill)
        assert isinstance(agent.static_method, Skill)
        # static_method should NOT have __history__ set
        assert not hasattr(MyAgent.static_method, "__history__")

    def test_agent_skips_classmethod_skill(self):
        """Agent.__init_subclass__ does not wrap classmethod Skills
        in cached_property — they remain class-level operations."""

        class MyAgent(Agent):
            """You are a classmethod-skill test agent.
            Your goal is to verify Agent wrapping does not alter class skills.
            """

            @Skill.define
            def instance_method(self) -> str:
                """Say hello."""
                raise NotHandled

            @Skill.define
            @classmethod
            def class_method(cls) -> str:
                """Do something."""
                raise NotHandled

        agent = MyAgent()
        assert isinstance(agent.instance_method, Skill)
        assert isinstance(MyAgent.class_method, Skill)
        # class_method should NOT have __history__ set
        assert not hasattr(MyAgent.class_method, "__history__")


def test_skill_formatting_scoped():
    feet_per_mile = 5280  # noqa: F841

    @Skill.define
    def convert(feet: int) -> float:
        """How many miles is {feet} feet? There are {feet_per_mile} feet per mile."""
        raise NotHandled

    with handler(SkillStringIntp()):
        assert convert(7920).endswith(
            "How many miles is 7920 feet? There are 5280 feet per mile."
        )


def test_validate_params_valid():
    """All format vars match signature params -- should succeed."""

    @Skill.define
    def poem(topic: str, style: str) -> str:
        """Write a {style} poem about {topic}."""
        raise NotHandled

    assert poem.__default__.__doc__.endswith("Write a {style} poem about {topic}.")


def test_validate_no_vars():
    """No format vars -- should succeed."""

    @Skill.define
    def simple() -> str:
        """Just a plain prompt with no variables."""
        raise NotHandled

    assert simple.__default__.__doc__.endswith("Just a plain prompt with no variables.")


def test_validate_undefined_var():
    """Referencing a variable not in params or lexical scope raises at define time."""
    with pytest.raises(TypeError, match="author"):

        @Skill.define
        def write_poem(topic: str) -> str:
            """Write a poem about {topic} by {author}."""
            raise NotHandled


def test_validate_multiple_undefined_vars():
    """Multiple undefined variables should all appear in the error."""
    with pytest.raises(TypeError, match="author") as exc_info:

        @Skill.define
        def write_poem(topic: str) -> str:
            """Write a poem about {topic} by {author} in {language}."""
            raise NotHandled

    assert "language" in str(exc_info.value)


def test_validate_compound_field_name():
    """Compound field name like {self.name} passes when root is a param."""

    @dataclass
    class Agent:
        name: str

        @Skill.define
        def greet(self, day: str) -> str:
            """Agent '{self.name}' says hello on {day}."""
            raise NotHandled

    assert Agent.greet.__default__.__doc__.endswith(
        "Agent '{self.name}' says hello on {day}."
    )


def test_validate_staticmethod():
    """Staticmethod skills should also be validated."""

    @Skill.define
    @staticmethod
    def ok(a: str, b: str) -> str:
        """Combine {a} and {b}."""
        raise NotHandled

    # The underlying Skill should exist
    assert ok.__func__.__default__.__doc__.endswith("Combine {a} and {b}.")


def test_validate_staticmethod_undefined():
    """Staticmethod skills with undefined vars should raise."""
    with pytest.raises(TypeError, match="missing"):

        @Skill.define
        @staticmethod
        def bad(a: str) -> str:
            """Combine {a} and {missing}."""
            raise NotHandled


def test_validate_staticmethod_lexical_scope():
    """Staticmethod skills should capture lexical scope variables."""
    feet_per_mile = 5280  # noqa: F841

    @Skill.define
    @staticmethod
    def convert(feet: int) -> str:
        """How many miles is {feet} feet? There are {feet_per_mile} feet per mile."""
        raise NotHandled

    # The inner skill should have the correct lexical context
    inner = convert.__func__
    assert "feet_per_mile" in inner.__context__


def test_staticmethod_lexical_scope_formatting():
    """Staticmethod skills should format lexical scope variables at runtime."""
    feet_per_mile = 5280  # noqa: F841

    @Skill.define
    @staticmethod
    def convert(feet: int) -> str:
        """How many miles is {feet} feet? There are {feet_per_mile} feet per mile."""
        raise NotHandled

    with handler(SkillStringIntp()):
        assert convert(7920).endswith(
            "How many miles is 7920 feet? There are 5280 feet per mile."
        )


def test_validate_lexical_var():
    """Lexical scope variables are allowed in skill format strings."""
    feet_per_mile = 5280  # noqa: F841

    @Skill.define
    def convert(feet: int) -> float:
        """How many miles is {feet} feet? There are {feet_per_mile} feet per mile."""
        raise NotHandled

    assert "feet_per_mile" in convert.__default__.__doc__


def test_validate_both_params_and_lexical():
    """Both params and lexical scope vars are allowed."""
    author = "Shakespeare"  # noqa: F841

    @Skill.define
    def write_poem(topic: str) -> str:
        """Write a poem about {topic} by {author}."""
        raise NotHandled

    assert write_poem.__default__.__doc__.endswith(
        "Write a poem about {topic} by {author}."
    )


def test_validate_undefined_with_lexical_still_fails():
    """Variables not in params or lexical scope still raise."""
    author = "Shakespeare"  # noqa: F841

    with pytest.raises(TypeError, match="nonexistent"):

        @Skill.define
        def bad(topic: str) -> str:
            """Write about {topic} by {author} using {nonexistent}."""
            raise NotHandled


def test_validate_field_name_identifier():
    """arg_name as identifier: {name}."""

    @Skill.define
    def fmt(price: float, name: str) -> str:
        """Buy {name} for {price}."""
        raise NotHandled


def test_validate_field_name_attribute_access():
    """field_name with attribute access: {self.name}."""

    @dataclass
    class Agent:
        name: str

        @Skill.define
        def greet(self, day: str) -> str:
            """{self.name} says hello on {day}."""
            raise NotHandled


def test_validate_field_name_index_access():
    """field_name with index access: {items[0]}."""

    @Skill.define
    def fmt(items: list) -> str:
        """First item is {items[0]}."""
        raise NotHandled


def test_validate_field_name_chained_access():
    """field_name with chained attribute and index: {obj.items[0].name}."""

    @Skill.define
    def fmt(obj: object) -> str:
        """Name: {obj.items[0].name}."""
        raise NotHandled


def test_validate_field_name_positional_digit():
    """arg_name as digit+ (positional): {0} is not supported in skills."""
    with pytest.raises(TypeError, match="0"):

        @Skill.define
        def bad(x: str) -> str:
            """Value: {0}."""
            raise NotHandled


def test_validate_field_name_empty():
    """Empty arg_name (auto-numbering): {} is not supported in skills."""
    with pytest.raises(TypeError):

        @Skill.define
        def bad(x: str) -> str:
            """Value: {}."""
            raise NotHandled


def test_validate_conversion_r():
    """Conversion !r should not affect variable resolution."""

    @Skill.define
    def fmt(value: str) -> str:
        """The value is {value!r}."""
        raise NotHandled


def test_validate_conversion_s():
    """Conversion !s should not affect variable resolution."""

    @Skill.define
    def fmt(value: str) -> str:
        """The value is {value!s}."""
        raise NotHandled


def test_validate_conversion_a():
    """Conversion !a should not affect variable resolution."""

    @Skill.define
    def fmt(value: str) -> str:
        """The value is {value!a}."""
        raise NotHandled


def test_validate_string_format_spec_width_align():
    """String-safe format specs (width, alignment, fill) work at runtime."""

    @Skill.define
    def fmt(label: str) -> str:
        """Label: {label:>20} or {label:*^30}"""
        raise NotHandled


def test_validate_string_format_spec_truncation():
    """String-safe precision (truncation) works at runtime."""

    @Skill.define
    def fmt(val: str) -> str:
        """Truncated: {val!s:.10}"""
        raise NotHandled


def test_validate_numeric_format_spec_passes_validation():
    """Numeric specs like .2f pass *validation* even though they would
    fail at runtime (applied to serialised str, not float).
    """

    @Skill.define
    def fmt(price: float, count: int) -> str:
        """Price: ${price:.2f}, count: {count:d}."""
        raise NotHandled


def test_validate_compound_field_with_spec():
    """Compound field with a spec: root name must resolve."""

    @dataclass
    class Calc:
        precision: int

        @Skill.define
        def compute(self, value: float) -> str:
            """Compute {value} with precision {self.precision:d}."""
            raise NotHandled


def test_validate_format_spec_on_undefined_var():
    """Undefined variable with a format spec should still raise."""
    with pytest.raises(TypeError, match="missing"):

        @Skill.define
        def bad(x: int) -> str:
            """Value: {x} and {missing:.2f}."""
            raise NotHandled


# ---------------------------------------------------------------------------
# Doctests in a Skill docstring must be constant (no spliced arguments).
# Skills are defined *inside* each test so pytest's --doctest-modules does
# not try to collect/run these docstring examples.
# ---------------------------------------------------------------------------


def test_validate_constant_doctest_ok():
    """A doctest with no format fields is accepted."""

    @Skill.define
    def dbl(x: int) -> int:
        """Double {x}.

        >>> dbl(2)
        4
        """
        raise NotHandled

    assert "dbl(2)" in dbl.__default__.__doc__


def test_validate_param_spliced_into_doctest_source_rejected():
    """A parameter spliced into the doctest source is rejected at define time."""
    with pytest.raises(TypeError, match="constant") as exc:

        @Skill.define
        def dbl(x: int) -> int:
            """Double {x}.

            >>> dbl({x})
            4
            """
            raise NotHandled

    assert "'x'" in str(exc.value)


def test_validate_field_spliced_into_doctest_want_rejected():
    """A field spliced into the expected output is rejected."""
    with pytest.raises(TypeError, match="constant"):

        @Skill.define
        def dbl(x: int) -> int:
            """Double {x}.

            >>> dbl(2)
            {x}
            """
            raise NotHandled


def test_validate_bare_braces_in_doctest_rejected():
    """A bare ``{}`` in a doctest is non-constant (str.format treats it as a
    positional field) and is rejected."""
    with pytest.raises(TypeError, match="constant"):

        @Skill.define
        def dbl(x: int) -> int:
            """Double {x}.

            >>> d = {}
            >>> dbl(2)
            4
            """
            raise NotHandled


def test_validate_escaped_braces_in_doctest_ok():
    """Escaped braces ``{{``/``}}`` format to literal braces, so they are
    constant and accepted."""

    @Skill.define
    def make_dict(x: int) -> dict:
        """Build a dict from {x}.

        >>> d = {{}}
        >>> make_dict(2)
        {{'k': 2}}
        """
        raise NotHandled

    assert "make_dict(2)" in make_dict.__default__.__doc__


def test_validate_field_in_prose_with_constant_doctest_ok():
    """Format fields are still allowed in the prose around constant doctests."""

    @Skill.define
    def about(theme: str) -> int:
        """Count words about {theme}.

        >>> about("cats")
        1
        """
        raise NotHandled

    assert "{theme}" in about.__default__.__doc__


# Forward ref through Tool subclass of Operation.
# Use types Pydantic can serialize (not arbitrary classes) to avoid
# PydanticSchemaGenerationError when other tests build tool schemas.
@Tool.define
def _tool_forward_ref(x: "int") -> "str":
    """A tool with forward-referenced parameter and return types."""
    raise NotHandled


def test_tool_forward_ref():
    sig = inspect.signature(_tool_forward_ref)
    assert sig.parameters["x"].annotation is int
    assert sig.return_annotation is str


# ---------------------------------------------------------------------------
# Synthetic readers for lexical context (PR #545 finish-up)
# ---------------------------------------------------------------------------


# Helpers for the test matrix
@dataclasses.dataclass
class _SimpleDataclass:
    x: int
    y: str


class _SimpleModel(pydantic.BaseModel):
    """Pydantic model used in encodable-probe matrix tests."""

    x: int
    y: str


class _OpaqueNoEncoder:
    """Plain user class with no Pydantic encoder."""

    pass


def _example_unannotated(x):
    """Unannotated function for the skip-via-catch tests."""
    return x


def _example_annotated(x: int) -> int:
    """Annotated function returning x."""
    return x


def test_synthetic_reader_returns_captured_value():
    """The reader closes over the value snapshot taken at construction
    time.  In-place mutation of a mutable captured value is visible
    (same object reference); rebinding the source name is not."""
    captured: list[int] = [1, 2, 3]
    tool = LexicalReaders._LexicalVariableTool.define(captured, name="x")
    assert tool() == [1, 2, 3]
    captured.append(4)
    assert tool() == [1, 2, 3, 4]


def test_synthetic_reader_snapshot_survives_rebind():
    """Tools are constructed fresh each `call_assistant` invocation,
    so rebinding the source name between construction and invocation
    has no effect on the captured value."""
    env: dict = {"x": 42}
    tool = LexicalReaders._LexicalVariableTool.define(env["x"], name="x")
    env["x"] = 99
    assert tool() == 42


def test_synthetic_reader_snapshot_survives_deletion():
    """The closure holds the value directly, so deleting the source
    name does not invalidate the reader."""
    env: dict = {"x": 42}
    tool = LexicalReaders._LexicalVariableTool.define(env["x"], name="x")
    del env["x"]
    assert tool() == 42


_PROBE_OK_CASES: list[tuple[str, typing.Any]] = [
    ("primitive_int", 42),
    ("primitive_str", "hello"),
    ("list_of_int", [1, 2, 3]),
    ("dict_value", {"a": 1}),
    ("dataclass_simple", _SimpleDataclass(x=1, y="hello")),
    ("pydantic_model", _SimpleModel(x=1, y="hello")),
    # `re.Pattern` and `pathlib.PosixPath` are encodable in the matrix.
    ("re_pattern", re.compile(r"x")),
    ("pathlib_path", Path("/tmp")),
]


@pytest.mark.parametrize(
    "name,value", _PROBE_OK_CASES, ids=lambda x: x[0] if isinstance(x, tuple) else None
)
def test_lexical_variable_tool_returns_value(name, value):
    """`_LexicalVariableTool` builds a Tool when `Encodable[T]`
    schema generates; calling it returns the captured value."""
    tool = LexicalReaders._LexicalVariableTool.define(value, name=name)
    assert tool() is value


# ---- Encodable-passthrough exposure (annotated callables, classes,
# builtins, methods) ----


def _example_method_owner_unannotated():
    class _C:
        def m(self):
            return 1

    return _C().m


def _example_method_owner_annotated():
    class _C:
        def m(self) -> int:
            return 1

    return _C().m


_EXPOSED_THROUGH_ENCODABLE: list[tuple[str, typing.Callable[[], typing.Any]]] = [
    # Annotated function: nested_type → Callable[[int], int]; _pydantic_callable schema.
    ("annotated_fn", lambda: _example_annotated),
    # Unannotated function: nested_type → function; _pydantic_callable schema.
    ("unannotated_fn", lambda: _example_unannotated),
    # Plain class: nested_type → type; _pydantic_callable schema.
    ("plain_class", lambda: type("Plain", (), {})),
    # Builtin function.
    ("builtin_fn", lambda: len),
    # Bound method, annotated and unannotated.
    ("annotated_method", _example_method_owner_annotated),
    ("unannotated_method", _example_method_owner_unannotated),
]


@pytest.mark.parametrize(
    "name,make_value",
    _EXPOSED_THROUGH_ENCODABLE,
    ids=lambda x: x[0] if isinstance(x, tuple) else None,
)
def test_collect_tools_exposes_callable_shaped_values(name, make_value):
    """With `LexicalReaders` installed, annotated callables, classes,
    builtins, and methods flow through Encodable's broad Callable
    handler and become synthesis-shaped tools."""
    value = make_value()
    env = {name: value}
    # The reader for `value` returns it verbatim; identify it by that value.
    assert any(t() is value for t in offered_tools(env, LexicalReaders()))


def test_lexical_reader_exposes_data_values():
    """Data-shaped values are exposed as readers (positive contract).
    Readers snapshot the value, so calling one returns the *same*
    object that was in env at construction time."""
    env = {
        "x": 1,
        "s": "hello",
        "lst": [1, 2, 3],
        "d": {"k": 1},
        "model": _SimpleModel(x=1, y="hi"),
    }
    tools = offered_tools(env, LexicalReaders())
    # Each value is exposed as a reader that returns the very same object.
    for v in env.values():
        assert any(t() is v for t in tools)


def test_skill_tools_includes_synthetic_readers_for_locals():
    """A skill offers synthetic readers for plain values in lexical scope
    when `LexicalReaders` is installed."""
    _test_data = [10, 20, 30]

    @Skill.define
    def t() -> int:
        """Doc."""
        raise NotHandled

    # Restrict to reader tools (safe to call) rather than other in-scope tools.
    readers = [
        tool
        for tool in skill_tools(t, LexicalReaders())
        if isinstance(tool, LexicalReaders._LexicalVariableTool)
    ]
    assert any(reader() == [10, 20, 30] for reader in readers)


def test_lexical_readers_handler_enables_collection():
    """Installing `LexicalReaders` flips the gate; the same values are
    exposed as zero-arg reader tools."""
    env = {"x": 42, "s": "hello"}
    tools = offered_tools(env, LexicalReaders())
    assert {t() for t in tools} == {42, "hello"}


# ---------------------------------------------------------------------------
# PythonRepl handler (#678)
# ---------------------------------------------------------------------------


def test_python_repl_off_by_default():
    """Without `PythonRepl`, `exec_code` is not collected."""
    assert StatefulReplSynthesizer().exec_code not in offered_tools({"x": 1})


def test_python_repl_exposes_exec_code():
    """With `PythonRepl` installed, `exec_code` is collected alongside the
    base tools."""
    repl = StatefulReplSynthesizer()
    assert repl.exec_code in offered_tools({"x": 1}, repl)


def test_python_repl_composes_with_lexical_readers():
    """Readers and the REPL tool coexist when both handlers are installed."""
    repl = StatefulReplSynthesizer()
    tools = offered_tools({"data": [1, 2, 3]}, LexicalReaders(), repl)
    assert repl.exec_code in tools  # the REPL tool
    readers = [t for t in tools if isinstance(t, LexicalReaders._LexicalVariableTool)]
    assert any(reader() == [1, 2, 3] for reader in readers)  # the data reader


def _drive_repl(body):
    """Run ``body(exec_code)`` inside one `PythonRepl`-scoped Skill call.

    A tiny `call_agent` handler stands in for the LLM loop: it collects
    the tools for a single call and hands `body` that call's `exec_code` tool, so
    `body` sees exactly one REPL session for its duration.  This is the supported
    way to reach a session -- `PythonRepl` introduces it per call, mirroring
    `__history__`.  Returns `body`'s result.
    """
    box = []
    repl = StatefulReplSynthesizer()

    class _Loop(ObjectInterpretation):
        @implements(call_agent)
        def _call(self, *_a, **_k):
            exec_code = repl.exec_code
            # Bodies pass source strings; decode them to code objects as the LLM
            # tool boundary would (`Encodable[CodeType]` compiles the source).
            decode = pydantic.TypeAdapter(Encodable[CodeType]).validate_python
            box.append(body(lambda src: exec_code(decode(src))))
            return None

    @Skill.define
    def _t() -> None:
        """Drive one REPL-scoped call."""
        raise NotImplementedError

    with (
        handler(MypyTypeChecker()),
        handler(_Loop()),
        handler(BuiltinExecutor()),
        handler(repl),
    ):
        _t()
    return box[0]


def test_python_repl_state_persists_within_one_call():
    """Within one Skill call, `exec_code` state carries across calls: a
    binding made in one snippet is visible in the next."""

    def body(exec_code):
        exec_code("kept = 5")
        return exec_code("print(kept)")

    assert _drive_repl(body) == "5\n"


def test_python_repl_distinct_calls_get_isolated_sessions():
    """Each Skill call gets its own session: a binding made in one call is
    not visible in a separate call."""
    _drive_repl(lambda exec_code: exec_code("leaked = 1"))
    assert (
        _drive_repl(lambda exec_code: exec_code("print('leaked' in dir())"))
        == "False\n"
    )


def test_python_repl_nested_call_is_isolated_and_outer_survives():
    """A nested Skill call introduces its own session by construction: the
    inner body cannot see the outer's bindings, and the outer session keeps its
    state across the nested call."""

    def outer(exec_code):
        exec_code("outer_var = 1")
        inner_sees_outer = _drive_repl(  # a fully nested Skill call
            lambda inner: inner("print('outer_var' in dir())")
        )
        outer_after = exec_code("print(outer_var, 'outer_var' in dir())")
        return inner_sees_outer, outer_after

    inner_sees_outer, outer_after = _drive_repl(outer)
    assert inner_sees_outer == "False\n"  # the nested session is isolated
    assert outer_after == "1 True\n"  # the outer session survived the nested call


# ---------------------------------------------------------------------------
# Lexical-context capture and decoding (consolidated from test_handlers_llm.py)
# ---------------------------------------------------------------------------


@Skill.define
def primes(first_digit: int) -> int:
    """Give exactly one prime number with {first_digit} as the first digit. Respond with only the number."""
    raise NotHandled


# Mutually recursive skills (module-level so globals are live for each other).
@Skill.define
def mutual_a() -> str:
    """Use mutual_a and mutual_b as tools to do task A."""
    raise NotHandled


@Skill.define
def mutual_b() -> str:
    """Use mutual_a and mutual_b as tools to do task B."""
    raise NotHandled


# Module-level variable for the shadowing tests below.
shadow_test_value = "global"


def test_primes_decode_int():
    """A non-string return type is decoded from the model's structured output."""
    mock = MockCompletionHandler([make_text_response('{"value": 61}')])

    with (
        handler(AgentLoop()),
        handler(LexicalToolExtractor()),
        handler(LiteLLMConfigurer()),
        handler(HistoryBuilder()),
        handler(mock),
    ):
        result = primes(6)

    assert result == 61
    assert isinstance(result, int)


def test_skill_captures_other_skills_in_lexical_context():
    """Skills defined in lexical scope are captured and offered as tools."""

    @Skill.define
    def story_with_moral(topic: str) -> str:
        """Write a story about {topic} with a moral lesson."""
        raise NotHandled

    @Skill.define
    def story_funny(topic: str) -> str:
        """Write a funny story about {topic}."""
        raise NotHandled

    @Skill.define
    def write_story(topic: str, style: str) -> str:
        """Write a story about {topic} in style {style}."""
        raise NotHandled

    # __context__ is a ChainMap(locals, globals) - sub-skills are visible.
    assert write_story.__context__["story_with_moral"] is story_with_moral
    assert write_story.__context__["story_funny"] is story_funny

    # Skills in lexical context are exposed as callable tools.
    assert story_with_moral in skill_tools(write_story)
    assert story_funny in skill_tools(write_story)


def test_mutually_recursive_skills():
    """Module-level skills see each other (mutual recursion) via globals."""
    assert "mutual_a" in mutual_a.__context__
    assert "mutual_b" in mutual_a.__context__
    assert "mutual_a" in mutual_b.__context__
    assert "mutual_b" in mutual_b.__context__

    # Each sees the other as a callable tool.
    assert mutual_a in skill_tools(mutual_b)
    assert mutual_b in skill_tools(mutual_a)
    # A skill is always dropped from its own toolset.
    assert mutual_a not in skill_tools(mutual_a)
    assert mutual_b not in skill_tools(mutual_b)


def test_lexical_context_shadowing():
    """Local variables shadow global variables in lexical context."""
    shadow_test_value = "local"  # noqa: F841 - intentional shadowing

    @Skill.define
    def skill_with_shadowed_var() -> str:
        """Test skill."""
        raise NotHandled

    # The lexical context should see the LOCAL value, not global.
    assert skill_with_shadowed_var.__context__["shadow_test_value"] == "local"


def test_lexical_context_sees_globals_when_no_local():
    """Globals are visible when there's no local shadow."""

    @Skill.define
    def skill_sees_global() -> str:
        """Test skill."""
        raise NotHandled

    assert skill_sees_global.__context__["shadow_test_value"] == "global"
