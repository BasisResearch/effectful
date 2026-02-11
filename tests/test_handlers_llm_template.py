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


@pytest.mark.xfail(reason="Runtime formatting of self.attr not yet supported")
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
    """Staticmethod templates should resolve lexical scope variables."""
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
            == "How many miles is 7920 feet? There are 5280 feet per mile."
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
