from dataclasses import dataclass

from effectful.handlers.llm import Template, Tool
from effectful.ops.semantics import NotHandled


def test_template_method():
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
            """What is the number after {self.x}?"""
            raise NotHandled

    a = A(0)
    assert isinstance(a.f, Template)
    assert "random" in a.f.__context__
    assert "f" in a.f.__context__
    assert "local_variable" in a.f.__context__
    assert a.f.__context__["random"]() == 4

    class B(A):
        @Tool.define
        def reverse(self, s: str) -> str:
            """Reverses a string."""
            return str(reversed(s))

    b = B(1)
    assert isinstance(b.f, Template)
    assert "random" in b.f.__context__
    assert "reverse" in b.f.__context__
    assert "local_variable" in b.f.__context__
