"""Tests for the launcher's ``--autoreload``, run in-process on a temporary directory.

A `Reloader` is installed over a two-file script, which runs as a stand-in
``__main__``; edits are written to disk by this process and applied by the next agent
call or by `Reloader.refresh`, and the result is checked. Only the temporary files
reload here; the harness modules were imported before the reloader started.
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


def _running(tmp_path, monkeypatch, build):
    """Start a `Reloader` over the script, and run the script as ``__main__``."""
    (tmp_path / "helper.py").write_text(HELPER)
    script = tmp_path / "script.py"
    script.write_text(SCRIPT)
    monkeypatch.syspath_prepend(str(tmp_path))
    reloader = autoreload.Reloader(script, build)
    main = types.ModuleType("__main__")
    main.__file__ = str(script)
    exec(compile(SCRIPT, str(script), "exec"), vars(main))
    monkeypatch.setitem(sys.modules, "__main__", main)
    return reloader, main


@pytest.fixture
def reloaded(tmp_path, monkeypatch):
    mock = MockCompletionHandler([make_text_response("ok")])
    reloader, main = _running(tmp_path, monkeypatch, lambda: _mocked_harness(mock))
    try:
        yield reloader, mock, tmp_path, main
    finally:
        reloader.close()


@pytest.fixture
def helper_stack(tmp_path, monkeypatch):
    """As `reloaded`, with a handler from the reloadable `helper` on top of the stack."""
    mock = MockCompletionHandler([make_text_response("ok")])
    reloader, main = _running(
        tmp_path,
        monkeypatch,
        lambda: coproduct(
            _mocked_harness(mock), importlib.import_module("helper").Answering()
        ),
    )
    try:
        yield reloader, mock, tmp_path, main
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
    reloader, mock, root, main = reloaded
    bot = main.Bot()
    before = type(bot)
    with interpreter(reloader):
        bot.ask("one")
    assert "It says one." in _systems(mock)[0]

    _edit(root / "helper.py", 'GREETING = "one"', 'GREETING = "two"')
    with interpreter(reloader):
        bot.ask("two")
    assert type(bot) is before is main.Bot, "the class kept its identity"
    assert "It says two." in before.__doc__, "the script re-ran too"
    assert _systems(mock) == [str(bot.__history__[0]["content"])]
    assert "It says two." in _systems(mock)[0]


def test_an_existing_agent_replaces_its_system_message_once(reloaded):
    reloader, mock, root, main = reloaded
    bot = main.Bot()
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
    reloader, _, root, main = reloaded
    main.Bot
    helper = sys.modules["helper"]

    _edit(
        root / "helper.py", 'def extra() -> str:\n    return "extra"\n', "EDITED = 1\n"
    )
    reloader.refresh()
    assert helper.EDITED == 1, "the module re-ran"
    assert helper.extra() == "extra"


def test_a_file_that_does_not_parse_keeps_its_running_version(reloaded):
    reloader, _, root, main = reloaded
    before = main.Bot
    script = root / "script.py"
    old = script.read_text()
    # Shifts every line, then fails to parse.
    script.write_text("import dataclasses\n" + old + "\n\ndef broken(:\n")
    reloader.refresh()

    assert main.Bot is before
    _, skill_def = _recover_skill_def(before.ask)
    assert skill_def.name == "ask"


def test_an_agents_edit_to_its_own_module_is_live_when_its_call_returns(reloaded):
    """The write happens mid-call, in this process; no watcher is involved."""
    reloader, mock, root, main = reloaded
    mock.responses[:] = [
        make_tool_call_response("write_here", "{}"),
        make_text_response("ok"),
    ]
    bot = main.Bot()
    helper = sys.modules["helper"]
    assert not hasattr(helper, "WRITTEN")

    with interpreter(reloader):
        bot.ask("go")
    assert helper.WRITTEN is True


def test_a_kept_module_is_not_re_run(reloaded):
    reloader, _, root, main = reloaded
    main.Bot
    helper = sys.modules["helper"]
    helper.__autoreload__ = False

    _edit(root / "helper.py", 'GREETING = "one"', 'GREETING = "two"')
    reloader.refresh()
    assert helper.GREETING == "one"


def test_the_running_script_is_re_run_in_place_without_its_main_block(reloaded):
    """``__main__`` itself re-runs on its first edit, and its main block stays shut."""
    reloader, _, root, main = reloaded
    bot, before, started = main.MAIN, main.Bot, main.MAIN
    ask = vars(before)["ask"]

    _edit(root / "script.py", "version one", "version two")
    reloader.refresh()
    assert type(bot) is before is main.Bot
    assert vars(before)["ask"] is ask, "a handler keyed by it still applies"
    assert "version two" in before.__doc__
    assert main.MAIN is started, "the main block did not run again"
    assert main.__name__ == "__main__"


def test_a_handler_installed_around_calls_follows_a_rebuilt_stack(helper_stack):
    """`handler(...)` around a script's loop composes onto the stack as it now is."""
    reloader, _, root, main = helper_stack
    bot = main.Bot()
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
    reloader, _, root, main = reloaded
    main.Bot
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


def test_a_build_that_fails_keeps_the_stack_and_is_retried(helper_stack):
    """A handler whose constructor raises leaves the running stack, and the next edit rebuilds."""
    reloader, _, root, main = helper_stack
    stack = reloader.snapshot()

    _edit(root / "helper.py", "self.answer = ANSWER", 'raise RuntimeError("boom")')
    reloader.refresh()
    assert reloader.snapshot() is stack

    _edit(root / "script.py", "version one", "version two")
    reloader.refresh()
    assert reloader.snapshot() is stack
    assert "version two" in main.Bot.__doc__, "an unrelated edit applies"

    _edit(root / "helper.py", 'raise RuntimeError("boom")', "self.answer = ANSWER")
    reloader.refresh()
    assert reloader.snapshot() is not stack


def test_an_edit_to_a_skill_docstring_reaches_an_existing_agent(reloaded):
    """The instance op an agent cached is rebuilt once its class op is redefined."""
    reloader, mock, root, main = reloaded
    bot = main.Bot()
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


def test_an_agent_whose_class_was_renamed_keeps_it(reloaded):
    reloader, _, root, main = reloaded
    bot = main.Bot()
    before = type(bot)

    _edit(root / "script.py", "class Bot:", "class Robot:")
    _edit(root / "script.py", "Bot.__doc__", "Robot.__doc__")
    _edit(root / "script.py", "MAIN = Bot()", "MAIN = Robot()")
    reloader.refresh()
    assert hasattr(main, "Robot"), "the edit applied"
    assert type(bot) is before
    assert "version one" in before.__doc__, "the old name still binds the old class"


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
