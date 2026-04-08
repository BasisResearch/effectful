"""Observability handler.

Composes with effectful's handler system to enable access to call
traces and raw completion outputs. The exact logging behavior is
specified by implementing a listener interface.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Hashable, cast, override

from effectful.handlers.llm import Tool, Template
from effectful.ops.types import NotHandled
from effectful.handlers.llm.completions import completion
from effectful.ops.semantics import fwd, coproduct, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.handlers.llm.completions import (
    LiteLLMProvider,
)
from functools import reduce

from time import time


class ObservabilityListener:
    def enter_tool_call[**P,Q](self, tool: Tool[P,Q]) -> None:
        pass

    def exit_tool_call[**P,Q](self, tool: Tool[P,Q], result: Q | None) -> None:
        pass

    def enter_template_call[**P,Q](self, template: Template[P,Q]) -> None:
        pass

    def exit_template_call[**P,Q](self, template: Template[P,Q], result: Q | None) -> None:
        pass

    def enter_completion(self) -> None:
        pass

    def exit_completion(self, resp: Any) -> None:
        pass


class EmptyCallStackException(Exception):
    pass

class CallStackListener(ObservabilityListener):
    def __init__(self):
        self.callstack: list[Any] = []

    @override
    def enter_tool_call[**P,Q](self, tool: Tool[P,Q]) -> None:
        super().enter_tool_call(tool)
        self.callstack.append(tool)

    @override
    def exit_tool_call[**P,Q](self, tool: Tool[P, Q], result: Q | None) -> None:
        assert len(self.callstack) > 0 and tool is self.callstack[-1]
        self.callstack.pop()
        super().exit_tool_call(tool, result)

    @override
    def enter_template_call[**P,Q](self, template: Template[P,Q]) -> None:
        super().enter_template_call(template)
        self.callstack.append(template)

    @override
    def exit_template_call[**P,Q](self, template: Template[P,Q], result: Q | None) -> None:
        assert len(self.callstack) > 0 and template is self.callstack[-1]
        self.callstack.pop()
        super().exit_template_call(template, result)

    def current_function(self) -> Any:
        try:
            return self.callstack[-1]
        except IndexError:
            raise EmptyCallStackException()

    def current_template(self) -> Any:
        try:
            return next(func for func in reversed(self.callstack) if
                        isinstance(func, Template))
        except StopIteration:
            raise EmptyCallStackException()

@dataclass
class ThinkingRecord:
    template: Any
    reasoning_content: str | None
    thinking_blocks: list[Any] | None


class ThinkingListener(CallStackListener):
    def __init__(self):
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
                        template=self.current_template(),
                        reasoning_content=reasoning_content,
                        thinking_blocks=thinking_blocks,
                    )
                )
        super().exit_completion(resp)

class ElapsedListener(CallStackListener):
    """Tracks the elapsed time of each :class:`Tool` or
    :class:`Template` call."""

    def __init__(self):
        super().__init__()
        self.timestack = []
        self.elapsed:defaultdict[Hashable,float] = defaultdict(float)

    @override
    def enter_completion(self):
        super().enter_completion()
        self.timestack.append(time())

    @override
    def exit_completion(self, resp: Any) -> None:
        time_elapsed = time() - self.timestack[-1]
        self.elapsed[self.current_template()] += time_elapsed
        self.timestack.pop()
        super().exit_completion(resp)



class ObservabilityHandler(ObjectInterpretation):
    """Tracks the call stack of :class:`Tool` and :class:`Template`
    and invokes a callback functions contained in an
    :class:`ObservabilityListener`
    """

    def __init__(self, listener: ObservabilityListener):
        self.listener = listener

    @implements(completion)
    def _observe_completion(self, *args, **kwargs) -> Any:
        self.listener.enter_completion()
        response: Any = None
        try:
            response = fwd(*args, **kwargs)
            return response
        finally:
            self.listener.exit_completion(response)

    @implements(Tool.__apply__)
    def _call_tool[**P,T](
            self, tool: Tool[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        result_opt: T | None = None
        try:
            self.listener.enter_tool_call(tool)
            result = cast(T, fwd(tool,*args,**kwargs))
            result_opt = result
            return result
        finally:
            self.listener.exit_tool_call(tool, result_opt)


    @implements(Template.__apply__)
    def _call_template[**P,T](
            self, template: Template[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        result_opt: T | None = None
        try:
            self.listener.enter_template_call(template)
            result = cast(T, fwd(template,*args,**kwargs))
            result_opt = result
            return result
        finally:
            self.listener.exit_template_call(template, result_opt)



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


class ThinkingElapsedListener(ThinkingListener, ElapsedListener):
    def __init__(self):
        super().__init__()


def test_handler():
# provider = LiteLLMProvider(
#     model='anthropic/claude-sonnet-4-20250514',
#     thinking={"type": "enabled", "budget_tokens": 1024}
# )
    provider = LiteLLMProvider(
        model='openai/gemma',
        api_key='',
        api_base='http://127.0.0.1:8080/',
        temperature=1.0,
        top_p=0.95,
        top_k=64
    )

    listener = ThinkingElapsedListener()
    obsprovider = ObservabilityHandler(listener)

    with handler(reduce(coproduct, [provider, obsprovider])):
        print(pick_fruit())
        print(find_treasure())

        print('----------------------------------------')
        for thinking in listener.thinking_records:
            print(thinking)

        print('----------------------------------------')
        for func, time in listener.elapsed.items():
            print(f"{func}:{time:.2f}s")
