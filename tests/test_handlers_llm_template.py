from dataclasses import dataclass

from effectful.handlers.llm import Template, Tool
from effectful.ops.semantics import NotHandled


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
            """What is the number after {self.x}?"""
            raise NotHandled

    a = A(0)
    assert isinstance(a.f, Template)
    assert "random" in a.f.tools
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
                """What is the number after {self.x}?"""
                raise NotHandled

    a = A.B(True)
    assert isinstance(a.f, Template)
    assert "random" in a.f.__context__
    assert "f" in a.f.__context__
    assert "local_variable" in a.f.__context__
    assert a.f.__context__["random"]() == 4


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
