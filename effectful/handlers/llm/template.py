import abc
import functools
import inspect
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

        # Deduplicate by tool identity — Tools are hashable Operations
        # and instance method Tools are cached per instance.
        tool2name = {tool: name for name, tool in result.items()}
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
        return result

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        if getattr(self, "__history__", None) is not None:
            from effectful.handlers.llm.completions import get_message_sequence
            from effectful.ops.semantics import handler

            with handler({get_message_sequence: lambda: self.__history__}):  # type: ignore
                return super().__call__(*args, **kwargs)
        else:
            return super().__call__(*args, **kwargs)

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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        prop = functools.cached_property(lambda _: OrderedDict())
        prop.__set_name__(cls, "__history__")
        cls.__history__ = prop

        for name, attr in list(cls.__dict__.items()):
            if not isinstance(attr, Template) or isinstance(
                attr.__default__, staticmethod | classmethod
            ):
                continue

            def _template_prop_fn[T: Template](self, *, template: T) -> T:
                inst_template = template.__get__(self, type(self))
                setattr(inst_template, "__history__", self.__history__)
                return inst_template

            _template_property = functools.cached_property(
                functools.partial(_template_prop_fn, template=attr)
            )
            _template_property.__set_name__(cls, name)
            setattr(cls, name, _template_property)
