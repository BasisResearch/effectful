import dataclasses
import functools
import inspect
import textwrap
import types
from collections.abc import Callable, Iterable
from typing import Any

from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled, Operation


<<<<<<< HEAD
<<<<<<< HEAD
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
=======
def _collect_lexical_functions(frame) -> dict[str, tuple[str, types.FunctionType]]:
    """Collect functions from the caller's lexical context.
=======
def _collect_lexical_context(frame) -> dict[str, tuple[str, Any]]:
    """Collect functions and types from the caller's lexical context.
>>>>>>> 297e9e7 (Function signature reformatting)

    Returns a dict mapping names to (source_code, object) tuples.
    Captures both functions and classes defined in the current module.
    """
    lexical_context = {**frame.f_globals, **frame.f_locals}
    current_module_name = frame.f_globals.get("__name__", "__main__")

    collected: dict[str, tuple[str, Any]] = {}
    for name, obj in lexical_context.items():
        # Skip private/dunder names
        if name.startswith("_"):
            continue

        # Check if it's a function or class from the current module
        is_function = isinstance(obj, types.FunctionType)
        is_class = isinstance(obj, type)

        if not (is_function or is_class):
            continue

        if getattr(obj, "__module__", None) != current_module_name:
            continue

        try:
            source = textwrap.dedent(inspect.getsource(obj)).strip()
        except OSError:
            # Fallback for objects without source (e.g., defined in REPL)
            if is_function:
                source = f"# <function {obj.__name__} from {obj.__module__}>\n# {obj.__doc__ or 'No docstring'}"
<<<<<<< HEAD
>>>>>>> 5c2d51c (Collecting lexical context)
            collected[name] = (source, obj)
=======
            else:
                source = f"# <class {obj.__name__} from {obj.__module__}>\n# {obj.__doc__ or 'No docstring'}"

        collected[name] = (source, obj)
>>>>>>> 297e9e7 (Function signature reformatting)

    return collected


<<<<<<< HEAD
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


=======
>>>>>>> 5c2d51c (Collecting lexical context)
@dataclasses.dataclass(frozen=True)
class Template[**P, T]:
    __signature__: inspect.Signature
    __prompt_template__: str
    tools: tuple[Operation, ...]
<<<<<<< HEAD
<<<<<<< HEAD
    lexical_context: dict[str, tuple[str, Any]] = dataclasses.field(
=======
    lexical_functions: dict[str, tuple[str, types.FunctionType]] = dataclasses.field(
>>>>>>> 5c2d51c (Collecting lexical context)
=======
    lexical_context: dict[str, tuple[str, Any]] = dataclasses.field(
>>>>>>> 297e9e7 (Function signature reformatting)
        default_factory=dict
    )

    # Backwards compatibility alias
    @property
    def lexical_functions(self) -> dict[str, tuple[str, types.FunctionType]]:
        """Get only the functions from lexical context (backwards compatibility)."""
        return {
            name: (source, obj)
            for name, (source, obj) in self.lexical_context.items()
            if isinstance(obj, types.FunctionType)
        }

    @property
    def lexical_types(self) -> dict[str, tuple[str, type]]:
        """Get only the types/classes from lexical context."""
        return {
            name: (source, obj)
            for name, (source, obj) in self.lexical_context.items()
            if isinstance(obj, type)
        }

    @defop
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        raise NotHandled

    def __get__(self, instance, _owner):
        if instance is not None:
            return functools.partial(self, instance)
        else:
            return self

    def get_lexical_context_source(self) -> str:
<<<<<<< HEAD
<<<<<<< HEAD
        """Return the source code of all captured lexical symbols."""
        return "\n\n".join(source for source, _ in self.lexical_context.values())
=======
        """Return the source code of all captured lexical functions."""
        return "\n\n".join(source for source, _ in self.lexical_functions.values())
=======
        """Return the source code of all captured lexical symbols."""
        return "\n\n".join(source for source, _ in self.lexical_context.values())
>>>>>>> 297e9e7 (Function signature reformatting)

    def get_lexical_function(self, name: str) -> types.FunctionType | None:
        """Get a captured lexical function by name."""
        if name in self.lexical_context:
            obj = self.lexical_context[name][1]
            if isinstance(obj, types.FunctionType):
                return obj
        return None

    def get_lexical_type(self, name: str) -> type | None:
        """Get a captured lexical type by name."""
        if name in self.lexical_context:
            obj = self.lexical_context[name][1]
            if isinstance(obj, type):
                return obj
        return None
>>>>>>> 5c2d51c (Collecting lexical context)

    @classmethod
    def define(cls, _func=None, *, tools: Iterable[Operation] = ()):
        # Capture caller's frame to collect lexical context
        caller_frame = inspect.currentframe()
        assert caller_frame is not None
        caller_frame = caller_frame.f_back
        assert caller_frame is not None

<<<<<<< HEAD
<<<<<<< HEAD
        lexical_ctx = _collect_lexical_context(caller_frame)
=======
        lexical_funcs = _collect_lexical_functions(caller_frame)
>>>>>>> 5c2d51c (Collecting lexical context)
=======
        lexical_ctx = _collect_lexical_context(caller_frame)
>>>>>>> 297e9e7 (Function signature reformatting)

        def decorator(body: Callable[P, T]):
            if not body.__doc__:
                raise ValueError("Expected a docstring on body")

            return cls(
                __signature__=inspect.signature(body),
                __prompt_template__=body.__doc__,
                tools=tuple(tools),
<<<<<<< HEAD
<<<<<<< HEAD
                lexical_context=lexical_ctx,
=======
                lexical_functions=lexical_funcs,
>>>>>>> 5c2d51c (Collecting lexical context)
=======
                lexical_context=lexical_ctx,
>>>>>>> 297e9e7 (Function signature reformatting)
            )

        if _func is None:
            return decorator
        return decorator(_func)
