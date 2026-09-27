"""Regression tests for edges of the launcher's ``--autoreload`` found against a live server."""

import pytest

pytest.importorskip("reactivity.hmr")

from effectful.handlers.llm.harness import autoreload  # noqa: E402
from effectful.internals.runtime import interpreter  # noqa: E402
from tests.test_handlers_llm_harness_autoreload import (  # noqa: E402, F401
    _edit,
    helper_stack,
    reloaded,
)


def test_a_build_that_fails_keeps_the_stack_and_is_retried(request):
    """A handler whose constructor raises leaves the running stack, and the next edit rebuilds."""
    reloader, _, root = request.getfixturevalue("helper_stack")
    stack = reloader.snapshot()

    _edit(root / "helper.py", "self.answer = ANSWER", 'raise RuntimeError("boom")')
    reloader.refresh()
    assert reloader.snapshot() is stack

    _edit(root / "script.py", "version one", "version two")
    reloader.refresh()
    assert reloader.snapshot() is stack
    assert "version two" in reloader.module.Bot.__doc__, "an unrelated edit applies"

    _edit(root / "helper.py", 'raise RuntimeError("boom")', "self.answer = ANSWER")
    reloader.refresh()
    assert reloader.snapshot() is not stack


def test_an_edit_to_a_skill_docstring_reaches_an_existing_agent(request):
    """The instance op an agent cached is rebuilt once its class op is redefined."""
    reloader, mock, root = request.getfixturevalue("reloaded")
    bot = reloader.module.Bot()
    with interpreter(reloader):
        bot.ask("one")
    assert not _users(mock)[-1].startswith("Q:")

    _edit(root / "script.py", '"""{question}"""', '"""Q: {question}"""')
    with interpreter(reloader):
        bot.ask("two")
    assert "Q: two" in _users(mock)[-1]


def _users(mock) -> list[str]:
    return [
        str(m["content"]) for m in mock.received_messages[-1] if m["role"] == "user"
    ]


def test_an_agent_whose_class_was_renamed_keeps_it(request):
    reloader, _, root = request.getfixturevalue("reloaded")
    bot = reloader.module.Bot()
    before = type(bot)

    _edit(root / "script.py", "class Bot:", "class Robot:")
    _edit(root / "script.py", "Bot.__doc__", "Robot.__doc__")
    _edit(root / "script.py", "MAIN = Bot()", "MAIN = Robot()")
    assert type(bot) is before
    assert "version one" in before.__doc__, "the old name still binds the old class"


def test_zero_argument_super_works_in_an_updated_class(tmp_path, monkeypatch):
    """A method copied onto the existing class finds that class through `super()`."""

    source = (
        "class Base:\n"
        "    def greet(self) -> str:\n"
        "        return 'base'\n\n\n"
        "class Child(Base):\n"
        "    def greet(self) -> str:\n"
        "        return 'child one, ' + super().greet()\n"
    )
    (tmp_path / "family.py").write_text(source)
    (tmp_path / "main.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    reloader = autoreload.Reloader(tmp_path / "main.py", dict)
    try:
        import family

        child = family.Child()
        (tmp_path / "family.py").write_text(source.replace("child one", "child two"))
        reloader.refresh()
        assert type(child) is family.Child
        assert child.greet() == "child two, base"
    finally:
        reloader.close()


def test_a_script_named_like_another_module_does_not_replace_it(tmp_path):
    """A script called ``json.py`` has no reloadable copy; the real `json` stays put."""
    import json
    import sys

    script = tmp_path / "json.py"
    script.write_text("VALUE = 1\n")
    reloader = autoreload.Reloader(script, dict)
    try:
        assert sys.modules["json"] is json
        assert reloader.module is None
    finally:
        reloader.close()


FAMILY = """\
import enum

from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Operation


@Operation.define
def greet() -> str:
    return "hello"


class Base(ObjectInterpretation):
    @implements(greet)
    def greet(self) -> str:
        return "base"


class Child(Base):
    @implements(greet)
    def greet(self) -> str:
        return "child one, " + super().greet()


class Mode(enum.StrEnum):
    ASK = "ask"
"""


def test_what_refers_to_a_re_run_class_moves_to_the_class_it_updated(
    tmp_path, monkeypatch
):
    """`super()` in a wrapped method, enum members, and a class the edit adds."""
    from effectful.ops.semantics import handler

    (tmp_path / "family.py").write_text(FAMILY)
    (tmp_path / "main.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    reloader = autoreload.Reloader(tmp_path / "main.py", dict)
    try:
        import family

        child, child_class, mode = family.Child(), family.Child, family.Mode
        (tmp_path / "family.py").write_text(
            FAMILY.replace("child one", "child two")
            + "\n\nclass Grandchild(Child):\n    pass\n"
        )
        reloader.refresh()
        family.Mode  # not imported by the script, so it re-runs when next read

        with handler(child):
            assert family.greet() == "child two, base"
        assert family.Mode is mode and isinstance(family.Mode.ASK, mode)
        assert family.Mode("ask") is family.Mode.ASK
        assert issubclass(family.Grandchild, child_class)
    finally:
        reloader.close()
