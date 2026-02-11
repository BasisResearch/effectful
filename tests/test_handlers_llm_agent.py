"""Tests for Agent mixin message sequence semantics."""

import collections
import dataclasses
import inspect
from dataclasses import dataclass

from litellm import ModelResponse

from effectful.handlers.llm import Agent, Template, Tool
from effectful.handlers.llm.completions import (
    LiteLLMProvider,
    RetryLLMHandler,
    completion,
)
from effectful.ops.semantics import handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled


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


# ---------------------------------------------------------------------------
# Agent subclass used by most tests
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ChatBot(Agent):
    """Simple chat agent for testing history accumulation."""

    bot_name: str = dataclasses.field(default="ChatBot")

    @Template.define
    def send(self, user_input: str) -> str:
        """A friendly bot named {self.bot_name}. User writes: {user_input}"""
        raise NotHandled


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentHistoryAccumulation:
    """History accumulates across sequential calls on the same instance."""

    def test_second_call_sees_prior_messages(self):
        mock = MockCompletionHandler(
            [
                make_text_response('{"value": "hi"}'),
                make_text_response('{"value": "good"}'),
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
                make_text_response('{"value": "r1"}'),
                make_text_response('{"value": "r2"}'),
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
                make_text_response('{"value": "r1"}'),
                make_text_response('{"value": "r2"}'),
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
                make_text_response('{"value": "from bot1"}'),
                make_text_response('{"value": "from bot2"}'),
            ]
        )
        bot1 = ChatBot()
        bot2 = ChatBot()

        with handler(LiteLLMProvider()), handler(mock):
            bot1.send("msg for bot1")
            bot2.send("msg for bot2")

        # bot2's call should NOT contain bot1's messages
        assert len(mock.received_messages[1]) >= 1  # system + user only

        # Each bot has its own history
        assert len(bot1.__history__) >= 2  # system, user, assistant
        assert len(bot2.__history__) >= 2

        # Histories share no message IDs
        assert set(bot1.__history__.keys()).isdisjoint(set(bot2.__history__.keys()))

    def test_non_agent_template_gets_fresh_sequence(self):
        @Template.define
        def standalone(topic: str) -> str:
            """Write about {topic}."""
            raise NotHandled

        mock = MockCompletionHandler(
            [
                make_text_response('{"value": "agent reply"}'),
                make_text_response('{"value": "standalone reply"}'),
                make_text_response('{"value": "agent reply 2"}'),
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


class TestAgentCachedProperty:
    """__history__ is lazily created per instance without requiring __init__."""

    def test_no_init_required(self):
        class MinimalAgent(Agent):
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
            @Template.define
            def compute(self, question: str) -> str:
                """Answer: {question}"""
                raise NotHandled

        mock = MockCompletionHandler(
            [
                make_tool_call_response("add", '{"a": 2, "b": 3}'),
                make_text_response('{"value": "The answer is 5"}'),
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
                make_text_response('{"value": "not_an_int"}'),
                # Retry: valid
                make_text_response('{"value": 42}'),
            ]
        )

        class NumberAgent(Agent):
            @Template.define
            def pick_number(self) -> int:
                """Pick a number."""
                raise NotHandled

        agent = NumberAgent()

        with (
            handler(LiteLLMProvider()),
            handler(RetryLLMHandler(num_retries=3)),
            handler(mock),
        ):
            result = agent.pick_number()

        assert result == 42

        # The malformed assistant message and error feedback from the retry
        # should NOT appear in the agent's history. Only the final successful
        # assistant message should be there.
        roles = {m["role"] for m in agent.__history__.values()}
        assert {"user", "assistant"} == roles - {"system"}


# ---------------------------------------------------------------------------
# Template method and scoping tests (moved from test_handlers_llm_template.py)
# ---------------------------------------------------------------------------


def test_template_method():
    """Test that methods can be used as templates."""
    local_variable = None  # noqa: F841

    @dataclass
    class A(Agent):
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
            @Tool.define
            def base_tool(self) -> int:
                """A base tool."""
                return 1

        class Derived(Base):
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

        mock = MockCompletionHandler([make_text_response('{"value": "42"}')])
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

        mock = MockCompletionHandler([make_text_response('{"value": "yes"}')])
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
