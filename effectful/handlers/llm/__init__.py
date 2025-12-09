import dataclasses
import functools
import inspect
import types
from collections.abc import Callable, Iterable
from typing import Any

from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled, Operation


def _collect_lexical_context(frame: types.FrameType) -> dict[str, Any]:
    """Collect all symbols from the caller's lexical context.

    Returns a dict mapping names to objects.
    Captures everything except:
    - Private/dunder names (starting with _)
    - Modules
    """
    lexical_context = {**frame.f_globals, **frame.f_locals}

    collected: dict[str, Any] = {}
    for name, obj in lexical_context.items():
        if name.startswith("_"):
            continue
        if isinstance(obj, types.ModuleType):
            continue
        collected[name] = obj

    return collected


@dataclasses.dataclass(frozen=True)
class Template[**P, T]:
    __signature__: inspect.Signature
    __prompt_template__: str
    tools: tuple[Operation, ...]
    lexical_context: dict[str, Any] = dataclasses.field(default_factory=dict)

    @defop
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        raise NotHandled

    def __get__(self, instance, _owner):
        if instance is not None:
            return functools.partial(self, instance)
        else:
            return self

    @classmethod
    def define(cls, _func=None, *, tools: Iterable[Operation] = ()):
        # Capture caller's frame to collect lexical context
        caller_frame = inspect.currentframe()
        assert caller_frame is not None
        caller_frame = caller_frame.f_back
        assert caller_frame is not None

        lexical_ctx = _collect_lexical_context(caller_frame)

        def decorator(body: Callable[P, T]):
            if not body.__doc__:
                raise ValueError("Expected a docstring on body")

            return cls(
                __signature__=inspect.signature(body),
                __prompt_template__=body.__doc__,
                tools=tuple(tools),
                lexical_context=lexical_ctx,
            )

        if _func is None:
            return decorator
        return decorator(_func)
