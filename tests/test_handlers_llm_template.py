import inspect
from dataclasses import dataclass

from effectful.handlers.llm import Template
from effectful.handlers.llm.completions import call_user
from effectful.ops.semantics import NotHandled, handler
from effectful.ops.syntax import ObjectInterpretation, implements


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
