from dataclasses import dataclass

import pytest

from effectful.handlers.llm import Template, Tool
from effectful.handlers.llm.completions import format_model_input
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
    assert "f" in a.f.tools
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

    @dataclass
    class A:
        x: int

        @Tool.define
        @staticmethod
        def random() -> int:
            """Returns a random number, chosen by fair dice roll."""
            return 4

        @dataclass
        class B:
            y: bool

            @Template.define
            def f(self) -> int:
                """What is the number after 3?"""
                raise NotHandled

    a = A.B(True)
    assert isinstance(a.f, Template)
    assert "random" in a.f.tools
    assert "f" in a.f.tools
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


class TemplateStringIntp(ObjectInterpretation):
    """Returns the result of template formatting as a string. Only supports
    templates that produce string prompts.

    """

    @implements(Template.__apply__)
    def _[**P, T](
        self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        model_input = format_model_input(template, *args, **kwargs)
        template_result = model_input[0]["content"]
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


@pytest.mark.xfail
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


@pytest.mark.xfail
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
