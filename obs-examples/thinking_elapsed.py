import dataclasses

from collections import defaultdict
from functools import reduce
from collections.abc import Callable
from typing import Any, Hashable, override

from effectful.handlers.llm import Tool, Template
from effectful.ops.types import NotHandled
from effectful.handlers.llm.completions import (
    LiteLLMProvider, LoggingHandler, LoggingListener,
    CallStackListener, CallInfo,
)
from effectful.ops.semantics import coproduct, handler
from time import time


@dataclasses.dataclass
class ThinkingRecord:
    """A single thinking/reasoning extraction paired with its source template."""
    template: Template[..., Any]
    reasoning_content: str | None
    thinking_blocks: list[Any] | None


class ThinkingListener(LoggingListener):
    """Extracts thinking and reasoning content from litellm completion responses."""

    def __init__(
        self,
        get_template_info: Callable[[], CallInfo[Template[..., Any]]],
    ) -> None:
        self.thinking_records: list[ThinkingRecord] = []
        self._get_template_info = get_template_info

    @override
    def exit_completion(self, resp: Any) -> None:
        if resp is not None:
            message = resp.choices[0].message
            reasoning_content = message.get("reasoning_content")
            thinking_blocks = message.get("thinking_blocks")
            if reasoning_content or thinking_blocks:
                self.thinking_records.append(
                    ThinkingRecord(
                        template=self._get_template_info().func,
                        reasoning_content=reasoning_content,
                        thinking_blocks=thinking_blocks,
                    )
                )


class ElapsedListener(LoggingListener):
    """Tracks the elapsed time of each :class:`Template` call."""

    def __init__(
        self,
        get_func_info: Callable[[], CallInfo[Tool[..., Any]]],
    ) -> None:
        self.elapsed: defaultdict[Hashable, float] = defaultdict(float)
        self._get_func_info = get_func_info

    @override
    def enter_completion(self) -> None:
        self._get_func_info().info['time'] = time()

    @override
    def exit_completion(self, resp: Any) -> None:
        func_info = self._get_func_info()
        time_elapsed = time() - func_info.info['time']
        self.elapsed[func_info.func] += time_elapsed


@Template.define
def find_treasure() -> str:
    """Ask Bob to find where the treasure is."""
    raise NotHandled

@Template.define
def bob() -> str:
    """Ask Alice to find where the treasure is."""
    raise NotHandled

@Tool.define
def alice() -> str:
    """Returns where the treasure is."""
    return "school"

@Template.define
def pick_fruit() -> str:
    """Return the name of a fruit."""
    raise NotHandled


def test_handler():
    provider = LiteLLMProvider(
        model='anthropic/claude-sonnet-4-20250514',
        thinking={"type": "enabled", "budget_tokens": 1024}
    )

    callstack = CallStackListener()
    thinking = ThinkingListener(callstack.current_template_info)
    elapsed = ElapsedListener(callstack.current_func_info)

    combined = reduce(coproduct, [
        provider,
        LoggingHandler(thinking),
        LoggingHandler(elapsed),
        LoggingHandler(callstack),
    ])

    with handler(combined):
        print(pick_fruit())
        print(find_treasure())

        print('----------------------------------------')
        for record in thinking.thinking_records:
            print(record)

        print('----------------------------------------')
        for func, elapsed_time in elapsed.elapsed.items():
            print(f"{func}:{elapsed_time:.2f}s")


if __name__ == '__main__':
    test_handler()
