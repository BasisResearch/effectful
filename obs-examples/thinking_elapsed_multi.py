import dataclasses

from collections import defaultdict
from typing import Any, Hashable, override

from effectful.handlers.llm import Tool, Template
from effectful.ops.types import NotHandled
from effectful.handlers.llm.completions import completion
from effectful.ops.semantics import fwd, coproduct, handler
from effectful.handlers.llm.completions import (
    LiteLLMProvider, LoggingHandler, CallStackListener
)
from time import time


@dataclasses.dataclass
class ThinkingRecord:
    """A single thinking/reasoning extraction paired with its source template."""
    template: Template[...,Any]
    reasoning_content: str | None
    thinking_blocks: list[Any] | None


class ThinkingListener(CallStackListener):
    """Extracts thinking and reasoning content from litellm completion responses."""

    def __init__(self) -> None:
        super().__init__()
        self.thinking_records: list[ThinkingRecord] = []

    @override
    def exit_completion(self, resp: Any) -> None:
        if resp is not None:
            message = resp.choices[0].message
            reasoning_content = message.get("reasoning_content")
            thinking_blocks = message.get("thinking_blocks")
            if reasoning_content or thinking_blocks:
                self.thinking_records.append(
                    ThinkingRecord(
                        template=self.current_template_info().func,
                        reasoning_content=reasoning_content,
                        thinking_blocks=thinking_blocks,
                    )
                )
        super().exit_completion(resp)

class ElapsedListener(CallStackListener):
    """Tracks the elapsed time of each :class:`Tool` or :class:`Template` call."""

    def __init__(self) -> None:
        super().__init__()
        self.elapsed:defaultdict[Hashable,float] = defaultdict(float)

    @override
    def enter_completion(self):
        super().enter_completion()
        self.current_func_info().info['time'] = time()

    @override
    def exit_completion(self, resp: Any) -> None:
        time_elapsed = time() - self.current_func_info().info['time']
        self.elapsed[self.current_func_info().func] += time_elapsed
        super().exit_completion(resp)


class ThinkingElapsedListener(ThinkingListener, ElapsedListener):
    """Combines thinking extraction and elapsed time tracking."""
    def __init__(self):
        super().__init__()


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

    listener = ThinkingElapsedListener()
    obsprovider = LoggingHandler(listener)


    with handler(provider), handler(obsprovider):
        print(pick_fruit())
        print(find_treasure())

        print('----------------------------------------')
        for thinking in listener.thinking_records:
            print(thinking)

        print('----------------------------------------')
        for func, time in listener.elapsed.items():
            print(f"{func}:{time:.2f}s")

if __name__ == '__main__':
    test_handler()
