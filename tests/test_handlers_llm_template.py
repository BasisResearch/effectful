import inspect
from dataclasses import dataclass

import pytest

from effectful.handlers.llm import Template, Tool
from effectful.handlers.llm.completions import call_user
from effectful.ops.semantics import NotHandled, handler
from effectful.ops.syntax import ObjectInterpretation, implements


def test_template_method():
    """Test that methods can be used as templates."""
    local_variable = None  # noqa: F841

    @dataclass
    class A:
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


class A:
    @Template.define
    def f(self) -> str:
        """Do stuff"""
        raise NotImplementedError


def test_template_method_module():
    """Test that template methods work when defined on module-level classes."""
    a = A()
    assert isinstance(a.f, Template)


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

        def _make():
            @Template.define
            def t() -> str:
                """test"""
                raise NotHandled

            return t

        t = _make()
        assert "leaked" not in t.__context__

    def test_class_method_tools_discovered_via_self(self):
        """After skipping the class body, tools on the same class are still
        discoverable through the bound `self` instance."""

        @dataclass
        class Widget:
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
        """Tools from a base class are visible through the instance."""

        class Base:
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


def test_template_formatting_scoped():
    feet_per_mile = 5280  # noqa: F841

    @Template.define
    def convert(feet: int) -> float:
        """How many miles is {feet} feet? There are {feet_per_mile} feet per mile."""
        raise NotHandled

    with handler(TemplateStringIntp()):
        assert (
            convert(7920)
            == "How many miles is 7920 feet? There are 5280 feet per mile."
        )


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
