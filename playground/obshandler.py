"""Observability handler.

Composes with effectful's handler system to enable access to call
traces and raw completion outputs. The exact logging behavior is
specified by implementing a listener interface.
"""

from abc import ABCMeta, abstractmethod
import random
from dataclasses import dataclass
from typing import Callable, Any, cast, override

from effectful.handlers.llm import Tool, Template
from effectful.ops.types import NotHandled
from effectful.handlers.llm.completions import completion
from effectful.ops.semantics import fwd, coproduct, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.handlers.llm.completions import (
    LiteLLMProvider,
)

from time import time


# How does mro work? Should I call super?
class ObservabilityListener(metaclass=ABCMeta):
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
        self.callstack.append(tool)

    @override
    def exit_tool_call[**P,Q](self, tool: Tool[P, Q], result: Q | None) -> None:
        self.callstack.pop()

    @override
    def enter_template_call[**P,Q](self, template: Template[P,Q]) -> None:
        self.callstack.append(template)

    @override
    def exit_template_call[**P,Q](self, template: Template[P,Q], result: Q | None) -> None:
        assert(len(self.callstack)>0)
        self.callstack.pop()

    def current_function(self) -> Any:
        try:
            return self.callstack[-1]
        except IndexError:
            raise EmptyCallStackException()

    def current_template(self) -> Any:
        try:
            return next(func for func in reversed(self.callstack) if
                        isinstance(func, Template))
        except IndexError:
            raise EmptyCallStackException()

class CompletionLogger(CallStackListener):
    def __init__(self):
        super().__init__()
        self.timestack = []

    @override
    def enter_completion(self):
        super().enter_completion()
        print(f"{self.current_template()} is calling completion")
        self.timestack.append(time())

    @override
    def exit_completion(self, resp: Any) -> None:
        time_elapsed = time() - self.timestack[-1]
        self.timestack.pop()
        print(f"Completion for {self.current_template()} finished in {time_elapsed:.2f} seconds")
        super().exit_completion(resp)



class ObservabilityHandler(ObjectInterpretation):
    """Tracks the call stack of :class:`Tool` and invokes a callback
    function on the callstack paired with raw completion responses"""

    def __init__[**P,T](self, listener: ObservabilityListener):
        self.listener = listener

    @implements(completion)
    def _observe_completion(self, *args, **kwargs) -> Any:
        model = kwargs.get("model", args[0] if args else "unknown")
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



# provider = LiteLLMProvider(
#     model='gpt-5.4'
# )
provider = LiteLLMProvider(
    model='openai/gemma',
    api_key='',
    api_base='http://127.0.0.1:8080/',
    temperature=1.0,
    top_p=0.95,
    top_k=64
)


obsprovider = ObservabilityHandler(listener=CompletionLogger())

themes = ['zombies', 'the universe', 'exorcism']


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

combined_provider = coproduct(provider, obsprovider)

def test_handler(provider):
    with handler(provider):
        print(find_treasure())
