"""Custom effect handlers for the code agent.

These compose with effectful's handler system to add logging, token budgets,
tool confirmation, and other cross-cutting concerns without modifying agent code.

Memory is modeled as algebraic effects: operations define the interface,
FileMemoryHandler provides file-backed implementation. Swap handlers for
testing, DB-backed storage, or remote memory.
"""

import random
from typing import Callable, Any
from effectful.handlers.llm import Tool, Template
from effectful.ops.types import NotHandled
from effectful.handlers.llm.completions import completion
from effectful.ops.semantics import fwd, coproduct, handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.handlers.llm.completions import (
    LiteLLMProvider,
)


class ObservabilityHandler(ObjectInterpretation):
    """Tracks the call stack of :class:`Tool` and invokes a callback
    function on the callstack paired with raw completion responses"""

    def __init__[**P,T](self, resp_fn : Callable[[Any, Tool[P,T]],None]):
        self.callstack = []
        self.resp_fn = resp_fn

    @implements(completion)
    def _observe_completion(self, *args, **kwargs) -> Any:
        print("calling completion")
        model = kwargs.get("model", args[0] if args else "unknown")
        old_stack = self.callstack.copy()
        response = fwd(*args, **kwargs)
        self.resp_fn(old_stack,response)
        return response

    @implements(Tool.__apply__)
    def _call_tool[**P,T](
        self, tool: Tool[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        print("calling _call_tool")
        try:
            # print(f"adding {tool} to {self.callstack}")
            self.callstack.append(tool)
            response = fwd(tool,*args,**kwargs)
        finally:
            # print(f"popping tool from {self.callstack}")
            self.callstack.pop()
            # print(f"and now we have {self.callstack}")


        return response

    @implements(Template.__apply__)
    def _call_template[**P,T](
        self, tool: Tool[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        print("calling _call_template")
        try:
            print(f"adding {tool} to {self.callstack}")
            self.callstack.append(tool)
            response = fwd(tool,*args,**kwargs)
        finally:
            # print(f"popping tool from {self.callstack}")
            self.callstack.pop()
            # print(f"and now we have {self.callstack}")


        return response


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


obsprovider = ObservabilityHandler(
    resp_fn = lambda st,resp: print(f"{st.}>{resp}")
)

themes = ['zombies', 'the universe', 'exorcism']

# @Tool.define
# def random_theme() -> str:
#     """Don't take any argument. Return a randomly picked theme."""
#     return random.choice(list(get_themes()))

# @Tool.define
# def get_themes() -> list[str]:
#     """Don't take any argument. Return a list of 3 strings, each of which represents a movie genre."""
#     raise NotHandled


# @Template.define
# def twosentencehorror() -> str:
#     """Write a two-sentence horror story on a theme picked by the tool `random_theme`."""
#     raise NotHandled

# @Template.define
# def pick_fruit() -> str:
#     """Return the name of a fruit."""
#     raise NotHandled

@Template.define
def find_treasure() -> str:
    """Call the tool `bob` to find where the treasure is."""
    raise NotHandled

@Template.define
def bob() -> str:
    """Call the tool `alice` to find where the treasure is."""
    raise NotHandled


@Tool.define
def alice() -> str:
    """Returns where the treasure is."""
    return "hell"

combined_provider = coproduct(provider, obsprovider)

def test_handler(provider):
    with handler(provider):
        print(find_treasure())
