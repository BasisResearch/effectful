import abc
import functools
import inspect
import re
import string
import types
import typing
from collections import ChainMap, OrderedDict
from collections.abc import Callable, Mapping, MutableMapping
from typing import Annotated, Any

from effectful.ops.types import Annotation, Operation


class _IsRecursiveAnnotation(Annotation):
    """
    A special type annotation for return types in the signature of a
    :class:`Template` that indicates it may make recursive calls.

    .. warning::

        :class:`IsRecursive` annotations are only defined to ascribe
        return annotations, and if used in a parameter will raise a
        :class:`TypeError` at tool construction time.



    **Example usage**:

    We illustrate the use of :class:`IsRecursive` below:

    >>> from typing import Annotated
    >>> from effectful.handlers.llm import Template
    >>> from effectful.handlers.llm.template import IsRecursive

    >>>
    @Template.define
    def factorial(n: int) -> Annotated[int, IsRecursive]:
       \"""Compute the n factorial for n={n}. Can call itself (`factorial`) recursively, but must be on smaller arguments.\"""
       raise NotHandled
    """

    @classmethod
    def infer_annotations(cls, sig: inspect.Signature) -> inspect.Signature:
        for name, ty in sig.parameters.items():
            if not ty or not typing.get_origin(ty) is Annotated:
                continue
            if any(isinstance(arg, cls) for arg in typing.get_args(ty)):
                raise TypeError(
                    f"Illegal annotation {ty} for parameter {name}, IsRecursive must only be used to annotate return types."
                )
        return sig


IsRecursive = _IsRecursiveAnnotation()


def _is_recursive_signature(sig: inspect.Signature):
    if typing.get_origin(sig.return_annotation) is not Annotated:
        return False
    annotations = typing.get_args(sig.return_annotation)
    return any(annotation is IsRecursive for annotation in annotations)


class Tool[**P, T](Operation[P, T]):
    """A :class:`Tool` is a function that may be called by a :class:`Template`.

    **Example usage:**

    Templates may call any tool that is in their lexical scope.
    In the following example, the LLM suggests a vacation destination using the :code:`cities` and :code:`weather` tools.::

        @Tool.define
        def cities() -> list[str]:
            \"\"\"Return a list of cities that can be passed to `weather`.\"\"\"
            return ["Chicago", "New York", "Barcelona"]

        @Tool.define
        def weather(city: str) -> str:
            \"\"\"Given a city name, return a description of the weather in that city.\"\"\"
            status = {"Chicago": "cold", "New York": "wet", "Barcelona": "sunny"}
            return status.get(city, "unknown")

        @Template.define  # cities and weather auto-captured from lexical scope
        def vacation() -> str:
            \"\"\"Use the `cities` and `weather` tools to suggest a city that has good weather.\"\"\"
            raise NotHandled

    Class methods may be used as templates, in which case any other methods decorated with :func:`Tool.define` will be provided as tools.

    """

    def __init__(
        self, signature: inspect.Signature, name: str, default: Callable[P, T]
    ):
        if not default.__doc__:
            raise ValueError("Tools must have docstrings.")
        signature = IsRecursive.infer_annotations(signature)
        super().__init__(signature, name, default)

    @classmethod
    def define(cls, *args, **kwargs) -> "Tool[P, T]":
        """Define a tool.

        See :func:`effectful.ops.types.Operation.define` for more information on the use of :func:`Tool.define`.

        """
        return typing.cast("Tool[P, T]", super().define(*args, **kwargs))


def _module_docstring_system_prompt(fn_or_cls: Any) -> str:
    """Build a system prompt from the defining module docstring."""
    mod = inspect.getmodule(fn_or_cls)
    if mod is None or not mod.__doc__:
        return ""
    return inspect.cleandoc(mod.__doc__)


class Template[**P, T](Tool[P, T]):
    """A :class:`Template` is a function that is implemented by a large language model.

    **Constructing Templates:**

    Templates are constructed by calling :func:`Template.define`.
    `Template.define` should be used as a decorator on a function or method.
    The function must be fully type-annotated and have a docstring.
    The body of the function must contain only :code:`raise NotHandled`.
    See :func:`effectful.ops.types.Operation.define` for more information on the use of :func:`Template.define`.

    The template docstring is a `format string <https://docs.python.org/3/library/string.html#format-string-syntax>`__, which may refer to the template arguments.
    When the template is called, the arguments and docstring are formatted into a prompt for the LLM and the LLM's response is returned.

    The following template writes limericks on a given theme:

    >>> @Template.define
    ... def limerick(theme: str) -> str:
    ...     \"\"\"Write a limerick on the theme of {theme}. Do not use any tools.\"\"\"
    ...     raise NotHandled

    **Structured output:**

    Templates may return types that are not strings.
    The output from the LLM is then decoded before being returned to the user.

    For example, this template returns integers:

    >>> @Template.define
    ... def primes(first_digit: int) -> int:
    ...     \"\"\"Give a prime number with {first_digit} as the first digit. Do not use any tools.\"\"\"
    ...     raise NotHandled

    Structured generation is used to constrain the LLM to return values that can be decoded without error.

    Templates can return complex data structures, such as dataclasses:

    >>> import dataclasses
    >>> @dataclasses.dataclass
    ... class KnockKnockJoke:
    ...     whos_there: str
    ...     punchline: str

    >>> @Template.define
    ... def write_joke(theme: str) -> KnockKnockJoke:
    ...     \"\"\"Write a knock-knock joke on the theme of {theme}. Do not use any tools.\"\"\"
    ...     raise NotHandled

    Many common Python data types are decodable without additional effort.
    To register a decoder for a custom type, see :func:`effectful.handlers.llm.encoding.type_to_encodable_type`.

    **Using tools:**

    Instances of :class:`Tool` that are in the lexical scope of a :class:`Template` may be called by the LLM during template completion.
    Templates are themselves tools which enables the construction of complex agent workflows.
    When a method is defined as a template, other methods on the class that are decorated with :func:`Tool.define` or :func:`Template.define` are provided to the template as tools.

    """

    __context__: ChainMap[str, Any]
    __system_prompt__: str

    @classmethod
    def _validate_prompt(
        cls,
        template: "Template",
        context: ChainMap[str, Any],
    ) -> None:
        """Validate that all format string variables in the docstring
        refer to names resolvable at call time.

        Each variable must be either a parameter in the signature
        or a name captured in the lexical context.

        :raises TypeError: If any format string variable cannot be resolved.
        """
        doc = template.__prompt_template__
        formatter = string.Formatter()
        param_names = set(template.__signature__.parameters.keys())
        context_keys = set(context.keys())
        allowed_names = param_names | context_keys

        unresolved: list[str] = []
        for _, field_name, _, _ in formatter.parse(doc):
            if field_name is None:
                continue
            # Extract root identifier from compound names like
            match = re.match(r"^(\w+)", field_name)
            root = match.group(1) if match else field_name
            if root not in allowed_names:
                unresolved.append(field_name)

        if unresolved:
            raise TypeError(
                f"Template '{template.__name__}' docstring references undefined "
                f"variables {list(sorted(unresolved))} that are not in the signature "
                f"{{{template.__signature__}}} or lexical scope."
            )

    @property
    def __prompt_template__(self) -> str:
        assert self.__default__.__doc__ is not None
        return self.__default__.__doc__

    @property
    def tools(self) -> Mapping[str, Tool]:
        """Operations and Templates available as tools. Auto-capture from lexical context."""
        result = {}
        is_recursive = _is_recursive_signature(self.__signature__)

        for name, obj in self.__context__.items():
            # Collect tools directly in context
            if isinstance(obj, Tool):
                result[name] = obj

            # Collect tools as methods on Agent instances in context
            elif isinstance(obj, Agent):
                for cls in type(obj).__mro__:
                    for attr_name in vars(cls):
                        if isinstance(getattr(obj, attr_name), Tool):
                            result[attr_name] = getattr(obj, attr_name)

        # Deduplicate by tool identity and remove self-references.
        #
        # The same Tool can appear under multiple names when it is both
        # visible in the enclosing scope *and* discovered via an Agent
        # instance's MRO.  Since Tools are hashable Operations and
        # instance-method Tools are cached per instance, we keep only
        # the last name for each unique tool object.  We also remove
        # the template itself from the tool map unless it is explicitly
        # marked as recursive (see test_template_method, test_template_method_nested_class).
        tool2name = {tool: name for name, tool in sorted(result.items())}
        for name, tool in tuple(result.items()):
            if tool2name[tool] != name or (tool is self and not is_recursive):
                del result[name]

        return result

    def __get__[S](self, instance: S | None, owner: type[S] | None = None):
        if hasattr(self, "_name_on_instance") and hasattr(
            instance, self._name_on_instance
        ):
            return getattr(instance, self._name_on_instance)

        result = super().__get__(instance, owner)
        self_param_name = list(self.__signature__.parameters.keys())[0]
        result.__context__ = self.__context__.new_child({self_param_name: instance})
        if isinstance(instance, Agent):
            assert isinstance(result, Template) and not hasattr(result, "__history__")
            result.__history__ = instance.__history__  # type: ignore[attr-defined]
            result.__system_prompt__ = "\n\n".join(
                part
                for part in (getattr(result, "__system_prompt__", ""), instance.__system_prompt__)
                if part
            )
        return result

    @classmethod
    def define[**Q, V](
        cls, default: Callable[Q, V], *args, **kwargs
    ) -> "Template[Q, V]":
        """Define a prompt template.

        :func:`define` takes a function and can be used as a decorator.
        The function's docstring should be a prompt, which may be templated in the function arguments.
        The prompt will be provided with any instances of :class:`Tool` that exist in the lexical context as callable tools.

        See :func:`effectful.ops.types.Operation.define` for more information on the use of :func:`Template.define`.

        """
        frame = inspect.currentframe()
        assert frame is not None
        frame = frame.f_back
        assert frame is not None

        # Skip class body frames: in Python, class bodies are not lexical
        # scopes for methods, so their locals should not be captured.
        qualname = frame.f_locals.get("__qualname__")
        if qualname is not None:
            for name in reversed(qualname.split(".")):
                if name == "<locals>":
                    break
                assert frame is not None
                frame = frame.f_back

        # Use the qualname of the decorated function to identify which
        # frames are *lexical* enclosers (as opposed to dynamic callers).
        # A segment preceding "<locals>" in the qualname is an enclosing
        # function; everything else (class names, the function itself) is not.
        assert frame is not None
        _fn = default
        if isinstance(_fn, staticmethod | classmethod):
            _fn = _fn.__func__
        parts = _fn.__qualname__.split(".")
        enclosing_fns = [
            parts[i] for i in range(len(parts) - 1) if parts[i + 1] == "<locals>"
        ]
        enclosing_fns.reverse()  # innermost first for frame walking

        globals_proxy: types.MappingProxyType[str, Any] = types.MappingProxyType(
            frame.f_globals
        )
        contexts: list[types.MappingProxyType[str, Any]] = []
        for fn_name in enclosing_fns:
            while frame is not None and frame.f_locals is not frame.f_globals:
                if frame.f_code.co_name == fn_name:
                    contexts.append(types.MappingProxyType(frame.f_locals))
                    frame = frame.f_back
                    break
                frame = frame.f_back
        contexts.append(globals_proxy)
        context: ChainMap[str, Any] = ChainMap(
            *typing.cast(list[MutableMapping[str, Any]], contexts)
        )
        op = super().define(default, *args, **kwargs)
        op.__context__ = context  # type: ignore[attr-defined]
        op.__system_prompt__ = _module_docstring_system_prompt(_fn)  # type: ignore[attr-defined]
        # Keep validation on original define-time callables, but skip the bound wrapper path.
        # to avoid dropping `self` from the signature and falsely rejecting valid prompt fields like `{self.name}`.
        is_bound_wrapper = (
            isinstance(default, types.MethodType) and default.__self__ is not None
        )
        if not isinstance(op, staticmethod | classmethod) and not is_bound_wrapper:
            cls._validate_prompt(typing.cast(Template, op), context)

        return typing.cast(Template[Q, V], op)


class Agent(abc.ABC):
    """Mixin that gives each instance a persistent LLM message history.

    Subclass and decorate methods with :func:`Template.define`.
    Each instance accumulates messages across calls so the LLM sees
    prior conversation context.

    Agents compose freely with :func:`dataclasses.dataclass` and other
    base classes.  Instance attributes are available in template
    docstrings via ``{self.attr}``.

    Example::

        import dataclasses
        from effectful.handlers.llm import Agent, Template
        from effectful.handlers.llm.completions import LiteLLMProvider
        from effectful.ops.semantics import handler
        from effectful.ops.types import NotHandled

        @dataclasses.dataclass
        class ChatBot(Agent):
            bot_name: str = dataclasses.field(default="ChatBot")

            @Template.define
            def send(self, user_input: str) -> str:
                \"""Friendly bot named {self.bot_name}. User writes: {user_input}\"""
                raise NotHandled

        provider = LiteLLMProvider()
        chatbot = ChatBot()

        with handler(provider):
            chatbot.send("Hi! How are you? I am in France.")
            chatbot.send("Remind me again, where am I?")  # sees prior context

    """

    __history__: OrderedDict[str, Mapping[str, Any]]
    __system_prompt__: str

    @classmethod
    def _build_system_prompt(cls) -> str:
        """Build an Agent-specific system prompt from class docstring."""
        class_doc = cls.__dict__.get("__doc__")
        if class_doc:
            cleaned = inspect.cleandoc(class_doc)
            if cleaned:
                return cleaned
        return ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        class_doc = cls.__dict__.get("__doc__")
        if class_doc is None or not inspect.cleandoc(class_doc):
            raise ValueError(
                "Agent subclasses must define a non-empty class docstring."
            )
        if not hasattr(cls, "__history__"):
            prop = functools.cached_property(lambda _: OrderedDict())
            prop.__set_name__(cls, "__history__")
            cls.__history__ = prop
        if not hasattr(cls, "__system_prompt__"):
            sp = functools.cached_property(
                lambda self: type(self)._build_system_prompt()
            )
            sp.__set_name__(cls, "__system_prompt__")
            cls.__system_prompt__ = sp
