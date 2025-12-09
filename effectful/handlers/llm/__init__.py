import dataclasses
import functools
import inspect
import textwrap
import types
from collections.abc import Callable, Iterable
from typing import Any

from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled, Operation


def _collect_lexical_context(frame) -> dict[str, tuple[str, Any]]:
    """Collect all symbols from the caller's lexical context.

    Returns a dict mapping names to (source_code/repr, object) tuples.
    Captures everything except:
    - Private/dunder names (starting with _)
    - Modules
    """
    lexical_context = {**frame.f_globals, **frame.f_locals}

    collected: dict[str, tuple[str, Any]] = {}
    for name, obj in lexical_context.items():
        source = _get_source_for_object(obj, name)
        if source is not None:
            collected[name] = (source, obj)

    return collected


def _get_source_for_object(obj: Any, name: str) -> str | None:
    """Get source code or representation for an object.

    Returns a string representation suitable for including in a prompt.
    """
    # For functions, try to get source
    if isinstance(obj, types.FunctionType):
        try:
            return textwrap.dedent(inspect.getsource(obj)).strip()
        except (OSError, TypeError):
            # Fallback for functions without source (e.g., defined in REPL)
            doc = obj.__doc__ or "No docstring"
            return f"# <function {obj.__name__}>\n# {doc}"

    # For classes/types
    if isinstance(obj, type):
        try:
            return textwrap.dedent(inspect.getsource(obj)).strip()
        except (OSError, TypeError):
            doc = obj.__doc__ or "No docstring"
            return f"# <class {obj.__name__}>\n# {doc}"

    # For generic aliases (list[int], Callable[[str], int], etc.)
    if hasattr(obj, "__origin__"):
        return f"{name} = {obj}"

    # For callable instances (objects with __call__)
    if callable(obj):
        obj_type = type(obj)
        try:
            return textwrap.dedent(inspect.getsource(obj_type)).strip()
        except (OSError, TypeError):
            doc = getattr(obj, "__doc__", None) or "No docstring"
            return f"# <callable {name}: {obj_type.__name__}>\n# {doc}"

    # For dataclass instances, show the instance
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return f"{name} = {obj!r}"

    try:
        repr_str = repr(obj)
        if len(repr_str) > 500:
            return f"{name} = <{type(obj).__name__}>"
        return f"{name} = {repr_str}"
    except Exception:
        return f"{name} = <{type(obj).__name__}>"


@dataclasses.dataclass(frozen=True)
class Template[**P, T]:
    __signature__: inspect.Signature
    __prompt_template__: str
    tools: tuple[Operation, ...]
    lexical_context: dict[str, tuple[str, Any]] = dataclasses.field(
        default_factory=dict
    )

    @defop
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        raise NotHandled

    def __get__(self, instance, _owner):
        if instance is not None:
            return functools.partial(self, instance)
        else:
            return self

    def get_lexical_context_source(self) -> str:
        """Return the source code of all captured lexical symbols."""
        return "\n\n".join(source for source, _ in self.lexical_context.values())

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
