"""Tests for Agent mixin message sequence semantics."""

import collections
import dataclasses
import inspect
from dataclasses import dataclass

import pytest
from litellm import ModelResponse

from effectful.handlers.llm import Agent, Template, Tool
from effectful.handlers.llm.completions import (
    LiteLLMProvider,
    RetryLLMHandler,
    call_user,
    completion,
)
from effectful.ops.semantics import handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled


class TemplateStringIntp(ObjectInterpretation):
    """Returns the result of template formatting as a string. Only supports
    templates that produce string prompts.

    """

    @implements(Template.__apply__)
    def _[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        bound_args = inspect.signature(template).bind(*args, **kwargs)
        bound_args.apply_defaults()
        env = template.__context__.new_child(bound_args.arguments)
        model_input = call_user(template.__prompt_template__, env)
        template_result = model_input["content"]
        assert len(template_result) == 1
        return template_result[0]["text"]


def test_template_formatting_simple():
    @Template.define
    @staticmethod
    def rhyme(a: str, b: str) -> str:
        """The {a} sat in the {b}."""
        raise NotHandled

    with handler(TemplateStringIntp()):
        assert rhyme("cat", "hat") == "The cat sat in the hat."


def test_template_formatting_method():
    @dataclass
    class User:
        name: str

        @Template.define
        def greet(self, day: str) -> float:
            """Greet the user '{self.name}' and wish them a good {day}."""
            raise NotHandled

    with handler(TemplateStringIntp()):
        user = User("Bob")
        assert (
            user.greet("Monday") == "Greet the user 'Bob' and wish them a good Monday."
        )


def _make_template_in_own_scope():
    """Module-level helper: the template's lexical scope is this function,
    NOT whatever dynamic caller invokes it."""

    @Template.define
    def t() -> str:
        """test"""
        raise NotHandled

    return t


class _ModuleLevelA:
    @Template.define
    def f(self) -> str:
        """Do stuff"""
        raise NotImplementedError


def _define_scoped_templates():
    @Tool.define
    def shown(self) -> int:
        """Should be able to see this tool."""
        return 0

    class A:
        @Template.define
        def f(self) -> str:
            """test"""
            return ""

    @Template.define
    def g() -> int:
        """test"""
        return 0

    def _nested():
        nonlocal shown

        @Template.define
        def h() -> int:
            """test"""
            return 0

        return h

    class B:
        @Template.define
        def i(self) -> str:
            """test"""
            return ""

        class C:
            @Template.define
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
    def _completion(self, model, messages=None, **kwargs):
        self.received_messages.append(list(messages) if messages else [])
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


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

    @Template.define
    def send(self, user_input: str) -> str:
        """A friendly bot named {self.bot_name}. User writes: {user_input}"""
        raise NotHandled


class _DesignerAgent(Agent):
    """You are an agent for nested-template regression tests.
    Your goal is to call nested tools/templates and return a final response.
    """

    @Template.define
    def nested_check(self, payload: str) -> str:
        """Check: {payload}. Do not use tools."""
        raise NotHandled

    @Tool.define
    def nested_tool(self, payload: str) -> str:
        """Check payload by calling a nested LLM template."""
        return self.nested_check(payload)

    @Template.define
    def outer(self, payload: str) -> str:
        """Call `nested_tool` for: {payload}, then return final answer."""
        raise NotHandled


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


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

        with handler(LiteLLMProvider()), handler(mock):
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

        with handler(LiteLLMProvider()), handler(mock):
            bot.send("a")
            bot.send("b")

        # After two complete calls the history should have:
        #   call 1: system, user, assistant  (3)
        #   call 2: system, user, assistant  (3)
        assert len(bot.__history__) >= 4

    def test_message_ids_are_unique(self):
        mock = MockCompletionHandler(
            [
                make_text_response("r1"),
                make_text_response("r2"),
            ]
        )
        bot = ChatBot()

        with handler(LiteLLMProvider()), handler(mock):
            bot.send("a")
            bot.send("b")

        ids = list(bot.__history__.keys())
        assert len(ids) == len(set(ids)), "message IDs must be unique"


class TestAgentIsolation:
    """Each agent instance has independent history; non-agent templates are unaffected."""

    def test_two_agents_have_independent_histories(self):
        mock = MockCompletionHandler(
            [
                make_text_response("from bot1"),
                make_text_response("from bot2"),
            ]
        )
        bot1 = ChatBot()
        bot2 = ChatBot()

        with handler(LiteLLMProvider()), handler(mock):
            bot1.send("msg for bot1")
            bot2.send("msg for bot2")

        # bot2's call should NOT contain bot1's messages — only system + user
        assert len(mock.received_messages[1]) == len(mock.received_messages[0])

        # Each bot made exactly one call, so their histories should be equal in size
        assert len(bot1.__history__) == len(bot2.__history__)

        # Histories share no message IDs
        assert set(bot1.__history__.keys()).isdisjoint(set(bot2.__history__.keys()))

    def test_non_agent_template_gets_fresh_sequence(self):
        @Template.define
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

        with handler(LiteLLMProvider()), handler(mock):
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

        with handler(LiteLLMProvider()), handler(mock):
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

        with handler(LiteLLMProvider()), handler(mock):
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

        with handler(LiteLLMProvider()), handler(mock):
            agent.outer("demo")

        for messages in mock.received_messages:
            assert_single_system_message_first(messages)

    def test_retry_flow_has_one_system_message_per_attempt(self):
        class RetryAgent(Agent):
            """You are a retry-flow test agent.
            Your goal is to produce an integer response after retry feedback.
            """

            @Template.define
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
            handler(LiteLLMProvider()),
            handler(RetryLLMHandler()),
            handler(mock),
        ):
            assert RetryAgent().pick_number() == 7

        assert len(mock.received_messages) == 2
        assert_single_system_message_first(mock.received_messages[0])
        assert_single_system_message_first(mock.received_messages[1])

    def test_non_agent_template_calls_have_one_system_message(self):
        @Template.define
        def standalone(topic: str) -> str:
            """Write about {topic}."""
            raise NotHandled

        mock = MockCompletionHandler(
            [
                make_text_response("a"),
                make_text_response("b"),
            ]
        )

        with handler(LiteLLMProvider()), handler(mock):
            standalone("fish")
            standalone("birds")

        assert_single_system_message_first(mock.received_messages[0])
        assert_single_system_message_first(mock.received_messages[1])


class TestAgentDocstringEnforcement:
    """Agent subclasses must define explicit non-empty docstrings."""

    def test_missing_docstring_raises(self):
        with pytest.raises(
            ValueError,
            match="Agent subclasses must define a non-empty class docstring.",
        ):

            class MissingDocAgent(Agent):
                pass

    def test_blank_docstring_raises(self):
        with pytest.raises(
            ValueError,
            match="Agent subclasses must define a non-empty class docstring.",
        ):

            class BlankDocAgent(Agent):
                """ """

    def test_non_empty_docstring_succeeds(self):
        class ValidDocAgent(Agent):
            """You are a valid-docstring test agent.
            Your goal is to satisfy the explicit Agent docstring requirement.
            """

        assert ValidDocAgent.__doc__ is not None
        assert "You are a valid-docstring test agent." in ValidDocAgent.__doc__


class TestAgentCachedProperty:
    """__history__ is lazily created per instance without requiring __init__."""

    def test_no_init_required(self):
        class MinimalAgent(Agent):
            """You are a minimal cached-property test agent.
            Your goal is to expose lazily initialized Agent state.
            """

            @Template.define
            def greet(self, name: str) -> str:
                """Hello {name}."""
                raise NotHandled

        agent = MinimalAgent()
        # Should be an OrderedDict, created on first access
        assert isinstance(agent.__history__, collections.OrderedDict)
        assert len(agent.__history__) == 0

    def test_subclass_with_own_init(self):
        class CustomAgent(Agent):
            """You are a custom-init test agent.
            Your goal is to ensure Agent mixin behavior survives custom `__init__`.
            """

            def __init__(self, name: str):
                self.name = name

            @Template.define
            def greet(self) -> str:
                """Say hello."""
                raise NotHandled

        agent = CustomAgent("Alice")
        assert agent.name == "Alice"
        assert isinstance(agent.__history__, collections.OrderedDict)

    def test_history_is_per_instance(self):
        a = ChatBot()
        b = ChatBot()
        a.__history__["fake"] = {"id": "fake", "role": "user", "content": "x"}
        assert "fake" not in b.__history__


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

            @Template.define
            def compute(self, question: str) -> str:
                """Answer: {question}"""
                raise NotHandled

        mock = MockCompletionHandler(
            [
                make_tool_call_response(
                    "add", '{"a": {"value": 2}, "b": {"value": 3}}'
                ),
                make_text_response("The answer is 5"),
            ]
        )
        agent = MathAgent()

        with handler(LiteLLMProvider()), handler(mock):
            result = agent.compute("what is 2+3?")

        assert result == "The answer is 5"

        # History should contain: system, user, assistant (tool_call),
        # tool (result), assistant (final)
        roles = [m["role"] for m in agent.__history__.values()]
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

            @Template.define
            def pick_number(self) -> int:
                """Pick a number."""
                raise NotHandled

        agent = NumberAgent()

        with (
            handler(LiteLLMProvider()),
            handler(RetryLLMHandler()),
            handler(mock),
        ):
            result = agent.pick_number()

        assert result == 42

        # The malformed assistant message and error feedback from the retry
        # should NOT appear in the agent's history. Only the final successful
        # assistant message should be there.
        roles = {m["role"] for m in agent.__history__.values()}
        assert {"user", "assistant"} == roles - {"system"}


class TestNestedTemplateCalling:
    """Issue #560: nested Template invocation via tool on the same Agent.

    When a Template triggers a tool call whose implementation invokes
    another Template on the same Agent, the inner call must:
    - work on a fresh copy of the agent's history
    - NOT write its messages back to agent.__history__
    - return its result correctly so the outer template can continue
    """

    def test_same_agent_nested_template_via_tool(self):
        """The scenario from issue #560 completes without error."""
        mock = MockCompletionHandler(
            [
                make_tool_call_response("nested_tool", '{"payload": "demo"}'),
                make_text_response("check passed"),
                make_text_response("all good"),
            ]
        )
        agent = _DesignerAgent()

        with handler(LiteLLMProvider()), handler(mock):
            result = agent.outer("demo")

        assert result == "all good"

    def test_only_outermost_writes_to_history(self):
        """Inner template's messages are absent from agent.__history__."""
        mock = MockCompletionHandler(
            [
                make_tool_call_response("nested_tool", '{"payload": "demo"}'),
                make_text_response("inner"),
                make_text_response("outer"),
            ]
        )
        agent = _DesignerAgent()

        with handler(LiteLLMProvider()), handler(mock):
            agent.outer("demo")

        roles = [m["role"] for m in agent.__history__.values()]
        # Outer call produces: user, assistant(tool_call), tool, assistant(final)
        # Inner call's user + assistant are NOT written back
        assert set(roles) <= {"system", "user", "assistant", "tool"}
        assert roles.count("system") == 1
        assert roles.count("user") == 1
        assert roles.count("assistant") == 2  # tool_call + final
        assert roles.count("tool") == 1

    def test_inner_template_gets_fresh_messages(self):
        """The nested template's LLM call sees only its own system + user,
        not the outer template's in-flight messages."""
        mock = MockCompletionHandler(
            [
                make_tool_call_response("nested_tool", '{"payload": "demo"}'),
                make_text_response("inner"),
                make_text_response("outer"),
            ]
        )
        agent = _DesignerAgent()

        with handler(LiteLLMProvider()), handler(mock):
            agent.outer("demo")

        # Call 0: outer's first call_assistant → [user]
        # Call 1: inner's call_assistant → [user] (fresh, from empty history)
        # Call 2: outer's second call_assistant → [user, assistant(tc), tool]
        inner_roles = [m["role"] for m in mock.received_messages[1]]
        assert {"user"} <= set(inner_roles) <= {"system", "user"}

    def test_inner_template_sees_prior_completed_history(self):
        """After a previous top-level call, the nested inner template sees
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

        with handler(LiteLLMProvider()), handler(mock):
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

        with handler(LiteLLMProvider()), handler(mock):
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


# ---------------------------------------------------------------------------
# Template method and scoping tests (moved from test_handlers_llm_template.py)
# ---------------------------------------------------------------------------


def test_template_method():
    """Test that methods can be used as templates."""
    local_variable = None  # noqa: F841

    @dataclass
    class A(Agent):
        """You are a template-method test agent.
        Your goal is to expose method tools and method templates correctly.
        """

        x: int

        @Tool.define
        def random(self) -> int:
            """Returns a random number, chosen by fair dice roll."""
            return 4

        @Template.define
        def f(self) -> int:
            """What is the number after 3?"""
            raise NotHandled

    a = A(0)
    assert isinstance(a.f, Template)
    assert "random" in a.f.tools
    # f is the template itself — found via self but correctly removed (non-recursive)
    assert "f" not in a.f.tools
    assert "local_variable" in a.f.__context__ and "local_variable" not in a.f.tools
    assert a.f.tools["random"]() == 4

    class B(A):
        """You are a derived template-method test agent.
        Your goal is to add inherited-tool coverage for method-template tests.
        """

        @Tool.define
        def reverse(self, s: str) -> str:
            """Reverses a string."""
            return str(reversed(s))

    b = B(1)
    assert isinstance(b.f, Template)
    assert "random" in b.f.tools
    assert "reverse" in b.f.tools
    assert "local_variable" in b.f.__context__ and "local_variable" not in a.f.tools


def test_template_method_nested_class():
    """Test that template methods work on nested classes."""
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

            @Template.define
            def f(self) -> int:
                """What is the number after 3?"""
                raise NotHandled

    a = A.B(True)
    assert isinstance(a.f, Template)
    # random is found via the enclosing function scope
    assert "random" in a.f.tools
    # f is the template itself — found via self but correctly removed (non-recursive)
    assert "f" not in a.f.tools
    assert "local_variable" in a.f.__context__ and "local_variable" not in a.f.tools
    assert a.f.tools["random"]() == 4


def test_template_method_module():
    """Test that template methods work when defined on module-level classes."""
    a = _ModuleLevelA()
    assert isinstance(a.f, Template)


def test_template_method_scoping():
    @Tool.define
    def hidden(self) -> int:
        """Shouldn't be able to see this tool."""
        return 0

    templates = _define_scoped_templates()
    for t in templates:
        assert isinstance(t, Template)
        assert "shown" in t.__context__
        assert "hidden" not in t.__context__


# ---------------------------------------------------------------------------
# Lexical scope collection
# ---------------------------------------------------------------------------


class TestLexicalScopeCollection:
    """Tests that Template.define follows Python's lexical scope rules."""

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

            @Template.define
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
        """Tools defined in the enclosing function are visible to templates
        defined inside a class in that function."""

        @Tool.define
        def helper() -> int:
            """A helper tool."""
            return 99

        class Bar:
            @Template.define
            def ask(self) -> str:
                """Ask something."""
                raise NotHandled

        bar = Bar()
        assert "helper" in bar.ask.tools

    def test_dynamic_caller_not_leaked(self):
        """Variables from a dynamic caller (not lexical enclosure) should not
        appear in the template's context."""
        leaked = False  # noqa: F841

        # _make_template_in_own_scope is defined at module level, so
        # this test method is a dynamic caller, not a lexical encloser.
        t = _make_template_in_own_scope()
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

            @Template.define
            def describe(self) -> str:
                """Describe this widget."""
                raise NotHandled

        w = Widget()
        assert "measure" in w.describe.tools
        # The template itself is not in tools (non-recursive)
        assert "describe" not in w.describe.tools

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

            @Template.define
            def ask(self) -> str:
                """Ask something."""
                raise NotHandled

        d = Derived()
        assert "base_tool" in d.ask.tools

    def test_tool_in_enclosing_function_visible_through_class(self):
        """function -> class -> Template.define: tool in the function is visible."""

        @Tool.define
        def outer_tool() -> int:
            """Outer tool."""
            return 1

        class Inner:
            @Template.define
            def ask(self) -> str:
                """Ask something."""
                raise NotHandled

        assert "outer_tool" in Inner().ask.tools

    def test_tool_in_enclosing_function_visible_through_nested_classes(self):
        """function -> class -> class -> Template.define: tool in the function
        is still visible after skipping multiple class body frames."""

        @Tool.define
        def outer_tool() -> int:
            """Outer tool."""
            return 1

        class Outer:
            class Inner:
                @Template.define
                def ask(self) -> str:
                    """Ask something."""
                    raise NotHandled

        assert "outer_tool" in Outer.Inner().ask.tools

    def test_nested_function_then_class(self):
        """function -> function -> class -> Template.define: all enclosing
        function scopes are visible, matching Python's lexical scope rules."""

        def _make():
            @Tool.define
            def inner_tool() -> int:
                """Inner tool."""
                return 2

            class MyClass:
                @Template.define
                def ask(self) -> str:
                    """Ask."""
                    raise NotHandled

            return MyClass

        outer_var = True  # noqa: F841
        cls = _make()
        assert "inner_tool" in cls().ask.tools
        # The test method is a lexical encloser of _make, so its locals
        # are visible — matching Python's actual scoping rules.
        assert "outer_var" in cls().ask.__context__

    def test_nested_function_scopes_template_at_inner(self):
        """function -> function -> Template.define: template sees all
        enclosing function scopes, matching Python's lexical scope rules."""

        def _outer():
            outer_var = "outer"  # noqa: F841

            def _inner():
                inner_var = "inner"  # noqa: F841

                @Template.define
                def t() -> str:
                    """test"""
                    raise NotHandled

                return t

            return _inner()

        t = _outer()
        assert "inner_var" in t.__context__
        assert "outer_var" in t.__context__


# ---------------------------------------------------------------------------
# staticmethod / classmethod Templates
# ---------------------------------------------------------------------------


class TestStaticAndClassMethodTemplates:
    """Tests for @Template.define applied to staticmethod and classmethod descriptors."""

    def test_staticmethod_template_in_class(self):
        """@Template.define @staticmethod in a class body produces a Template
        accessible as a class attribute."""

        class MyClass:
            @Template.define
            @staticmethod
            def ask(question: str) -> str:
                """Answer: {question}"""
                raise NotHandled

        assert isinstance(MyClass.ask, Template)
        assert isinstance(MyClass().ask, Template)

    def test_staticmethod_template_callable(self):
        """Staticmethod Templates can be called through a handler."""

        class MyClass:
            @Template.define
            @staticmethod
            def ask(question: str) -> str:
                """Answer: {question}"""
                raise NotHandled

        mock = MockCompletionHandler([make_text_response("42")])
        with handler(LiteLLMProvider()), handler(mock):
            result = MyClass.ask("what is 6*7?")
        assert result == "42"

    def test_staticmethod_template_captures_enclosing_scope(self):
        """A staticmethod Template captures the enclosing function scope,
        even through the re-entrant _define_staticmethod call."""

        @Tool.define
        def helper() -> int:
            """A helper tool."""
            return 99

        class MyClass:
            @Template.define
            @staticmethod
            def ask(x: int) -> int:
                """Compute {x}."""
                raise NotHandled

        assert "helper" in MyClass.ask.tools

    def test_staticmethod_template_excludes_class_body(self):
        """A staticmethod Template does not capture class body locals."""

        class MyClass:
            class_var = 42  # noqa: F841

            @Template.define
            @staticmethod
            def ask() -> str:
                """Ask."""
                raise NotHandled

        assert "class_var" not in MyClass.ask.__context__

    def test_classmethod_template_in_class(self):
        """@Template.define @classmethod in a class body produces a Template
        accessible as a class attribute (lazily via _ClassMethodOpDescriptor)."""

        class MyClass:
            @Template.define
            @classmethod
            def ask(cls, question: str) -> str:
                """Answer: {question}"""
                raise NotHandled

        assert isinstance(MyClass.ask, Template)

    def test_classmethod_template_callable(self):
        """Classmethod Templates can be called through a handler."""

        class MyClass:
            @Template.define
            @classmethod
            def ask(cls, question: str) -> str:
                """Answer: {question}"""
                raise NotHandled

        mock = MockCompletionHandler([make_text_response("yes")])
        with handler(LiteLLMProvider()), handler(mock):
            result = MyClass.ask("is the sky blue?")
        assert result == "yes"

    def test_classmethod_template_signature_excludes_cls(self):
        """The classmethod Template's signature does not include cls,
        since the classmethod descriptor binds it automatically."""

        class MyClass:
            @Template.define
            @classmethod
            def ask(cls, question: str) -> str:
                """Answer: {question}"""
                raise NotHandled

        sig = inspect.signature(MyClass.ask)
        assert "cls" not in sig.parameters
        assert "question" in sig.parameters

    def test_agent_skips_staticmethod_template(self):
        """Agent.__init_subclass__ does not wrap staticmethod Templates
        in cached_property — they remain accessible as plain Templates."""

        class MyAgent(Agent):
            """You are a staticmethod-template test agent.
            Your goal is to verify Agent wrapping does not alter static templates.
            """

            @Template.define
            def instance_method(self) -> str:
                """Say hello."""
                raise NotHandled

            @Template.define
            @staticmethod
            def static_method(x: int) -> int:
                """Double {x}."""
                raise NotHandled

        agent = MyAgent()
        # instance_method is wrapped by Agent into a cached_property
        assert isinstance(agent.instance_method, Template)
        # static_method remains a plain Template accessible on class and instance
        assert isinstance(MyAgent.static_method, Template)
        assert isinstance(agent.static_method, Template)
        # static_method should NOT have __history__ set
        assert not hasattr(MyAgent.static_method, "__history__")

    def test_agent_skips_classmethod_template(self):
        """Agent.__init_subclass__ does not wrap classmethod Templates
        in cached_property — they remain class-level operations."""

        class MyAgent(Agent):
            """You are a classmethod-template test agent.
            Your goal is to verify Agent wrapping does not alter class templates.
            """

            @Template.define
            def instance_method(self) -> str:
                """Say hello."""
                raise NotHandled

            @Template.define
            @classmethod
            def class_method(cls) -> str:
                """Do something."""
                raise NotHandled

        agent = MyAgent()
        assert isinstance(agent.instance_method, Template)
        assert isinstance(MyAgent.class_method, Template)
        # class_method should NOT have __history__ set
        assert not hasattr(MyAgent.class_method, "__history__")


def test_template_formatting_scoped():
    feet_per_mile = 5280  # noqa: F841

    @Template.define
    def convert(feet: int) -> float:
        """How many miles is {feet} feet? There are {feet_per_mile} feet per mile."""
        raise NotHandled

    with handler(TemplateStringIntp()):
        assert (
            convert(7920)
            == 'How many miles is {"value":7920} feet? There are {"value":5280} feet per mile.'
        )


def test_validate_params_valid():
    """All format vars match signature params -- should succeed."""

    @Template.define
    def poem(topic: str, style: str) -> str:
        """Write a {style} poem about {topic}."""
        raise NotHandled

    assert poem.__prompt_template__ == "Write a {style} poem about {topic}."


def test_validate_no_vars():
    """No format vars -- should succeed."""

    @Template.define
    def simple() -> str:
        """Just a plain prompt with no variables."""
        raise NotHandled

    assert simple.__prompt_template__ == "Just a plain prompt with no variables."


def test_validate_undefined_var():
    """Referencing a variable not in params or lexical scope raises at define time."""
    with pytest.raises(TypeError, match="author"):

        @Template.define
        def write_poem(topic: str) -> str:
            """Write a poem about {topic} by {author}."""
            raise NotHandled


def test_validate_multiple_undefined_vars():
    """Multiple undefined variables should all appear in the error."""
    with pytest.raises(TypeError, match="author") as exc_info:

        @Template.define
        def write_poem(topic: str) -> str:
            """Write a poem about {topic} by {author} in {language}."""
            raise NotHandled

    assert "language" in str(exc_info.value)


def test_validate_compound_field_name():
    """Compound field name like {self.name} passes when root is a param."""

    @dataclass
    class Agent:
        name: str

        @Template.define
        def greet(self, day: str) -> str:
            """Agent '{self.name}' says hello on {day}."""
            raise NotHandled

    assert Agent.greet.__prompt_template__ == "Agent '{self.name}' says hello on {day}."


def test_validate_staticmethod():
    """Staticmethod templates should also be validated."""

    @Template.define
    @staticmethod
    def ok(a: str, b: str) -> str:
        """Combine {a} and {b}."""
        raise NotHandled

    # The underlying Template should exist
    assert ok.__func__.__prompt_template__ == "Combine {a} and {b}."


def test_validate_staticmethod_undefined():
    """Staticmethod templates with undefined vars should raise."""
    with pytest.raises(TypeError, match="missing"):

        @Template.define
        @staticmethod
        def bad(a: str) -> str:
            """Combine {a} and {missing}."""
            raise NotHandled


def test_validate_staticmethod_lexical_scope():
    """Staticmethod templates should capture lexical scope variables."""
    feet_per_mile = 5280  # noqa: F841

    @Template.define
    @staticmethod
    def convert(feet: int) -> str:
        """How many miles is {feet} feet? There are {feet_per_mile} feet per mile."""
        raise NotHandled

    # The inner template should have the correct lexical context
    inner = convert.__func__
    assert "feet_per_mile" in inner.__context__


def test_staticmethod_lexical_scope_formatting():
    """Staticmethod templates should format lexical scope variables at runtime."""
    feet_per_mile = 5280  # noqa: F841

    @Template.define
    @staticmethod
    def convert(feet: int) -> str:
        """How many miles is {feet} feet? There are {feet_per_mile} feet per mile."""
        raise NotHandled

    with handler(TemplateStringIntp()):
        assert (
            convert(7920)
            == 'How many miles is {"value":7920} feet? There are {"value":5280} feet per mile.'
        )


def test_validate_lexical_var():
    """Lexical scope variables are allowed in template format strings."""
    feet_per_mile = 5280  # noqa: F841

    @Template.define
    def convert(feet: int) -> float:
        """How many miles is {feet} feet? There are {feet_per_mile} feet per mile."""
        raise NotHandled

    assert "feet_per_mile" in convert.__prompt_template__


def test_validate_both_params_and_lexical():
    """Both params and lexical scope vars are allowed."""
    author = "Shakespeare"  # noqa: F841

    @Template.define
    def write_poem(topic: str) -> str:
        """Write a poem about {topic} by {author}."""
        raise NotHandled

    assert write_poem.__prompt_template__ == "Write a poem about {topic} by {author}."


def test_validate_undefined_with_lexical_still_fails():
    """Variables not in params or lexical scope still raise."""
    author = "Shakespeare"  # noqa: F841

    with pytest.raises(TypeError, match="nonexistent"):

        @Template.define
        def bad(topic: str) -> str:
            """Write about {topic} by {author} using {nonexistent}."""
            raise NotHandled


def test_validate_field_name_identifier():
    """arg_name as identifier: {name}."""

    @Template.define
    def fmt(price: float, name: str) -> str:
        """Buy {name} for {price}."""
        raise NotHandled


def test_validate_field_name_attribute_access():
    """field_name with attribute access: {self.name}."""

    @dataclass
    class Agent:
        name: str

        @Template.define
        def greet(self, day: str) -> str:
            """{self.name} says hello on {day}."""
            raise NotHandled


def test_validate_field_name_index_access():
    """field_name with index access: {items[0]}."""

    @Template.define
    def fmt(items: list) -> str:
        """First item is {items[0]}."""
        raise NotHandled


def test_validate_field_name_chained_access():
    """field_name with chained attribute and index: {obj.items[0].name}."""

    @Template.define
    def fmt(obj: object) -> str:
        """Name: {obj.items[0].name}."""
        raise NotHandled


def test_validate_field_name_positional_digit():
    """arg_name as digit+ (positional): {0} is not supported in templates."""
    with pytest.raises(TypeError, match="0"):

        @Template.define
        def bad(x: str) -> str:
            """Value: {0}."""
            raise NotHandled


def test_validate_field_name_empty():
    """Empty arg_name (auto-numbering): {} is not supported in templates."""
    with pytest.raises(TypeError):

        @Template.define
        def bad(x: str) -> str:
            """Value: {}."""
            raise NotHandled


def test_validate_conversion_r():
    """Conversion !r should not affect variable resolution."""

    @Template.define
    def fmt(value: str) -> str:
        """The value is {value!r}."""
        raise NotHandled


def test_validate_conversion_s():
    """Conversion !s should not affect variable resolution."""

    @Template.define
    def fmt(value: str) -> str:
        """The value is {value!s}."""
        raise NotHandled


def test_validate_conversion_a():
    """Conversion !a should not affect variable resolution."""

    @Template.define
    def fmt(value: str) -> str:
        """The value is {value!a}."""
        raise NotHandled


def test_validate_string_format_spec_width_align():
    """String-safe format specs (width, alignment, fill) work at runtime."""

    @Template.define
    def fmt(label: str) -> str:
        """Label: {label:>20} or {label:*^30}"""
        raise NotHandled


def test_validate_string_format_spec_truncation():
    """String-safe precision (truncation) works at runtime."""

    @Template.define
    def fmt(val: str) -> str:
        """Truncated: {val!s:.10}"""
        raise NotHandled


def test_validate_numeric_format_spec_passes_validation():
    """Numeric specs like .2f pass *validation* even though they would
    fail at runtime (applied to serialised str, not float).
    """

    @Template.define
    def fmt(price: float, count: int) -> str:
        """Price: ${price:.2f}, count: {count:d}."""
        raise NotHandled


def test_validate_compound_field_with_spec():
    """Compound field with a spec: root name must resolve."""

    @dataclass
    class Calc:
        precision: int

        @Template.define
        def compute(self, value: float) -> str:
            """Compute {value} with precision {self.precision:d}."""
            raise NotHandled


def test_validate_format_spec_on_undefined_var():
    """Undefined variable with a format spec should still raise."""
    with pytest.raises(TypeError, match="missing"):

        @Template.define
        def bad(x: int) -> str:
            """Value: {x} and {missing:.2f}."""
            raise NotHandled
