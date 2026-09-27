"""Tests for the launcher's ``--autoreload``, run in-process on a temporary directory.

A `Reloader` is installed over a two-file script, edits are written to disk by this
process, which makes them live on the next read, and the next read or call is checked
against them. The harness modules are already imported when a test runs, so only the
temporary files reload here; the stack itself is rebuilt from them by the same
mechanism.
"""

import importlib
import sys
import types

import pytest

pytest.importorskip("reactivity.hmr")

from effectful.handlers.llm.harness import autoreload, harness  # noqa: E402
from effectful.handlers.llm.harness.hooks import completion  # noqa: E402
from effectful.handlers.llm.harness.synthesis.function import (  # noqa: E402
    _recover_skill_def,
)
from effectful.internals.runtime import interpreter  # noqa: E402
from effectful.ops.semantics import coproduct, fwd, handler  # noqa: E402
from effectful.ops.syntax import ObjectInterpretation, implements  # noqa: E402
from tests.conftest import (  # noqa: E402
    MockCompletionHandler,
    make_text_response,
    make_tool_call_response,
)

HELPER = '''\
import pathlib

from effectful.handlers.llm import Tool
from effectful.handlers.llm.harness.hooks import completion
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Operation
from tests.conftest import make_text_response

GREETING = "one"


@Operation.define
def ping() -> str:
    return "pong"


class Box:
    @Operation.define
    @classmethod
    def label(cls) -> str:
        return "box"
ANSWER = "one"


class Answering(ObjectInterpretation):
    """A stack handler bound, at construction, to what ANSWER was."""

    def __init__(self):
        self.answer = ANSWER

    @implements(completion)
    def _completion(self, *args, **kwargs):
        fwd(*args, **kwargs)
        return make_text_response(self.answer)


@Tool.define
def shout(text: str) -> str:
    """Return `text` in capitals."""
    return text.upper()


@Tool.define
def write_here() -> str:
    """Add a line to this very file."""
    path = pathlib.Path(__file__)
    path.write_text(path.read_text() + "\\nWRITTEN = True\\n")
    return "written"


def extra() -> str:
    return "extra"
'''

SCRIPT = '''\
import dataclasses

from helper import GREETING, shout, write_here  # noqa: F401

from effectful.handlers.llm import Skill


@dataclasses.dataclass
class Bot:
    """A bot, version one."""

    __agent_id__: str = ""

    @Skill.define
    def ask(self, question: str) -> str:
        """{question}"""


Bot.__doc__ = f"{Bot.__doc__} It says {GREETING}."

if __name__ == "__main__":
    MAIN = Bot()
'''


def _mocked_harness(mock):
    return coproduct(
        harness(
            model="mock/model",
            eval_provider="none",
            type_checker="none",
            tool_calling="json",
        ),
        mock,
    )


@pytest.fixture
def reloaded(tmp_path, monkeypatch):
    (tmp_path / "helper.py").write_text(HELPER)
    script = tmp_path / "script.py"
    script.write_text(SCRIPT)
    monkeypatch.syspath_prepend(str(tmp_path))
    mock = MockCompletionHandler([make_text_response("ok")])
    reloader = autoreload.Reloader(script, lambda: _mocked_harness(mock))
    try:
        yield reloader, mock, tmp_path
    finally:
        reloader.close()


@pytest.fixture
def helper_stack(tmp_path, monkeypatch):
    """As `reloaded`, with a handler from the reloadable `helper` on top of the stack."""
    (tmp_path / "helper.py").write_text(HELPER)
    script = tmp_path / "script.py"
    script.write_text(SCRIPT)
    monkeypatch.syspath_prepend(str(tmp_path))
    mock = MockCompletionHandler([make_text_response("ok")])
    reloader = autoreload.Reloader(
        script,
        lambda: coproduct(
            _mocked_harness(mock), importlib.import_module("helper").Answering()
        ),
    )
    try:
        yield reloader, mock, tmp_path
    finally:
        reloader.close()


def _edit(path, old, new):
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))


def _systems(mock) -> list[str]:
    return [
        str(m["content"]) for m in mock.received_messages[-1] if m["role"] == "system"
    ]


def test_an_edit_to_an_imported_value_reaches_an_existing_agent(reloaded):
    reloader, mock, root = reloaded
    bot = reloader.module.Bot()
    before = type(bot)
    with interpreter(reloader):
        bot.ask("one")
    assert "It says one." in _systems(mock)[0]

    _edit(root / "helper.py", 'GREETING = "one"', 'GREETING = "two"')
    with interpreter(reloader):
        bot.ask("two")
    assert type(bot) is before is reloader.module.Bot, "the class kept its identity"
    assert "It says two." in before.__doc__, "the script re-ran too"
    assert _systems(mock) == [str(bot.__history__[0]["content"])]
    assert "It says two." in _systems(mock)[0]


def test_an_existing_agent_replaces_its_system_message_once(reloaded):
    reloader, mock, root = reloaded
    bot = reloader.module.Bot()
    with interpreter(reloader):
        bot.ask("one")

    _edit(root / "script.py", "A bot, version one.", "A bot, version two.")
    with interpreter(reloader):
        bot.ask("two")
    stored = bot.__history__[0]
    assert len(_systems(mock)) == 1 and "version two" in _systems(mock)[0]
    assert "version two" in str(stored["content"])

    with interpreter(reloader):
        bot.ask("three")
    assert bot.__history__[0] is stored, "a later call replaced it again"


def test_a_definition_an_edit_removes_stays_bound(reloaded):
    """As under `importlib.reload`: nothing can tell a stale definition from state."""
    reloader, _, root = reloaded
    reloader.module.Bot
    helper = sys.modules["helper"]

    reloads = reloader.reloads
    _edit(root / "helper.py", 'def extra() -> str:\n    return "extra"\n', "")
    reloader.refresh()
    assert helper.extra() == "extra"
    assert reloader.reloads > reloads, "the module re-ran"


def test_a_file_that_does_not_parse_keeps_its_running_version(reloaded):
    reloader, _, root = reloaded
    before = reloader.module.Bot
    script = root / "script.py"
    old = script.read_text()
    # Shifts every line, then fails to parse.
    script.write_text("import dataclasses\n" + old + "\n\ndef broken(:\n")

    assert reloader.module.Bot is before
    _, skill_def = _recover_skill_def(before.ask)
    assert skill_def.name == "ask"


def test_an_agents_edit_to_its_own_module_is_live_when_its_call_returns(reloaded):
    """The write happens mid-call, in this process; no watcher is involved."""
    reloader, mock, root = reloaded
    mock.responses[:] = [
        make_tool_call_response("write_here", "{}"),
        make_text_response("ok"),
    ]
    bot = reloader.module.Bot()
    helper = sys.modules["helper"]
    assert not hasattr(helper, "WRITTEN")

    with interpreter(reloader):
        bot.ask("go")
    assert helper.WRITTEN is True


def test_a_kept_module_is_not_re_run(reloaded):
    reloader, _, root = reloaded
    reloader.module.Bot
    helper = sys.modules["helper"]
    helper.__autoreload__ = False

    _edit(root / "helper.py", 'GREETING = "one"', 'GREETING = "two"')
    reloader.refresh()
    assert helper.GREETING == "one"


def test_the_running_scripts_classes_follow_an_edit_to_it(reloaded, monkeypatch):
    """``__main__`` is never re-run; the script's first edit updates its classes."""
    reloader, _, root = reloaded
    script = root / "script.py"
    main = types.ModuleType("__main__")
    exec(compile(script.read_text(), str(script), "exec"), vars(main))
    monkeypatch.setitem(sys.modules, "__main__", main)
    bot = main.MAIN
    before = type(bot)

    _edit(script, "version one", "version two")
    reloader.refresh()
    assert type(bot) is before is main.Bot is reloader.module.Bot
    assert "version two" in before.__doc__


def test_a_handler_installed_around_calls_follows_a_rebuilt_stack(helper_stack):
    """`handler(...)` around a script's loop composes onto the stack as it now is."""
    reloader, _, root = helper_stack
    bot = reloader.module.Bot()
    seen: list[str] = []

    class Recorder(ObjectInterpretation):
        @implements(completion)
        def _completion(self, *args, **kwargs):
            response = fwd(*args, **kwargs)
            seen.append(response.choices[0].message.content)
            return response

    with interpreter(reloader), handler(Recorder()):
        bot.ask("one")
        _edit(root / "helper.py", 'ANSWER = "one"', 'ANSWER = "two"')
        bot.ask("two")
    assert seen == ["one", "two"]


def test_a_redefined_operation_keeps_its_identity(reloaded):
    """An edit to a module that defines an operation updates it in place."""
    reloader, _, root = reloaded
    reloader.module.Bot
    helper = sys.modules["helper"]
    ping, label = helper.ping, helper.Box.label
    assert (ping(), label()) == ("pong", "box")

    _edit(root / "helper.py", 'return "pong"', 'return "PONG"')
    _edit(root / "helper.py", 'return "box"', 'return "BOX"')
    reloader.refresh()
    with handler({ping: lambda: "mine"}):
        assert helper.ping is ping and helper.Box.label is label
        assert ping() == "mine", "a handler keyed before the edit still applies"
    assert (ping(), label()) == ("PONG", "BOX")
