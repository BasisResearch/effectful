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
    reloader.module.Bot
    stack = reloader.snapshot()

    _edit(root / "helper.py", "self.answer = ANSWER", 'raise RuntimeError("boom")')
    assert reloader.apply([root / "helper.py"]) is True
    assert reloader.snapshot() is stack

    _edit(root / "script.py", "version one", "version two")
    assert reloader.apply([root / "script.py"]) is True, (
        "an unrelated edit still applies"
    )
    assert reloader.snapshot() is stack

    _edit(root / "helper.py", 'raise RuntimeError("boom")', "self.answer = ANSWER")
    assert reloader.apply([root / "helper.py"]) is True
    assert reloader.snapshot() is not stack


def test_the_watcher_survives_an_error_in_a_reload(request, monkeypatch):
    reloader, _, root = request.getfixturevalue("reloaded")

    monkeypatch.setattr(reloader, "apply", lambda *a, **k: 1 / 0)
    assert reloader._applied({str(root / "script.py")}, wait=True) is False


def test_an_edit_to_a_skill_docstring_reaches_an_existing_agent(request):
    """The instance op an agent cached is rebuilt once its class op is redefined."""
    reloader, mock, root = request.getfixturevalue("reloaded")
    bot = reloader.module.Bot()
    with interpreter(reloader):
        bot.ask("one")
    assert not _users(mock)[-1].startswith("Q:")

    _edit(root / "script.py", '"""{question}"""', '"""Q: {question}"""')
    assert reloader.apply([root / "script.py"]) is True
    autoreload.rebind(bot, reloader.current_class(type(bot)))
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
    assert reloader.apply([root / "script.py"]) is True
    autoreload.rebind(bot, reloader.current_class(type(bot)))
    assert type(bot) is before


def test_a_script_named_like_another_module_does_not_replace_it(tmp_path):
    """A script called ``json.py`` has no reloadable copy; the real `json` stays put."""
    import json
    import sys

    from effectful.handlers.llm.harness import autoreload

    script = tmp_path / "json.py"
    script.write_text("VALUE = 1\n")
    reloader = autoreload.Reloader(script, dict)
    try:
        assert sys.modules["json"] is json
        assert reloader.module is None
    finally:
        reloader.close()
