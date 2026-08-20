"""LLM-implemented functions via algebraic effects.

`effectful.handlers.llm` lets you write Python functions whose bodies are
implemented by a large language model, and call them like ordinary code.

## Core concepts

- **`Skill`** — a fully type-annotated Python function whose body is `raise
  NotHandled` and whose docstring is a [format
  string](https://docs.python.org/3/library/string.html#format-string-syntax)
  prompt. Calling a skill (under a provider) formats its arguments into the
  prompt, invokes the model, and decodes the response to the skill's declared
  return type. Define one with the `Skill.define` decorator.

- **`Tool`** — a normal Python callable exposed to the model. Its signature and
  docstring become the schema the model sees; the model calls it by name with
  JSON arguments and receives the encoded result. Tools in a skill's lexical
  scope are offered to the model automatically; because scope is ordinary Python
  scope, an `Agent` (or an enclosing function) naturally partitions tools and
  skills into disjoint sets. Define one with `Tool.define`.

- **`Agent`** — a class mixin giving each instance a persistent message history,
  so its `Skill` methods accumulate conversation context across calls.
  Instance attributes are available in prompts via `{self.attr}`.

- **`Encodable`** — the type-driven JSON bridge used internally to encode Python
  values into the model's context and decode the model's output (structured
  return values and tool-call arguments) back into typed Python objects.

## Tool calling and structured output

During a skill call the model may take multiple turns: on each turn it can
call any `Tool` in scope (results are fed back and the loop continues) or
produce a final answer. The final answer is decoded to the skill's return
type via constrained/structured generation, so non-`str` return types (ints,
dataclasses, etc.) come back as real Python values. A handler may also mark a
tool call as *finalizing*, letting the model "answer" by calling a tool: its
return value becomes the result and the loop terminates.
"""

import abc
import collections
import collections.abc
import doctest
import functools
import inspect
import json
import pickle
import re
import string
import types
import typing
import uuid

import effectful.ops.types

__all__ = ["Agent", "Skill", "Template", "Tool", "Encodable"]


class Tool[**P, T](effectful.ops.types.Operation[P, T]):
    """A `Tool` is a function that may be called by a `Skill`.

    A `Tool` wraps a normal Python callable; its signature (parameter types
    and return type) and docstring define the schema the model sees, and the
    model invokes it by name with JSON arguments.

    ## Example usage

    Skills may call any tool that is in their lexical scope. In the
    following example, the LLM suggests a vacation destination using the
    `cities` and `weather` tools:

    ```python
    @Tool.define
    def cities() -> list[str]:
        \"\"\"Return a list of cities that can be passed to `weather`.\"\"\"
        return ["Chicago", "New York", "Barcelona"]

    @Tool.define
    def weather(city: str) -> str:
        \"\"\"Given a city name, return a description of the weather in that city.\"\"\"
        status = {"Chicago": "cold", "New York": "wet", "Barcelona": "sunny"}
        return status.get(city, "unknown")

    @Skill.define  # cities and weather auto-captured from lexical scope
    def vacation() -> str:
        \"\"\"Use the `cities` and `weather` tools to suggest a city that has good weather.\"\"\"
    ```

    Class methods may be used as skills, in which case any other methods
    decorated with `Tool.define` will be provided as tools.

    """

    def __init__(
        self, default: collections.abc.Callable[P, T], name: str | None = None
    ):
        if not default.__doc__:
            raise ValueError("Tools must have docstrings.")
        super().__init__(default, name=name)

    @classmethod
    def define(cls, *args, **kwargs) -> "Tool[P, T]":
        """Define a tool.

        See `effectful.ops.types.Operation.define` for more information on the
        use of `Tool.define`.

        """
        return typing.cast("Tool[P, T]", super().define(*args, **kwargs))


class Skill[**P, T](Tool[P, T]):
    """A `Skill` is a function that is implemented by a large language model.

    ## Constructing Skills

    Apply `Skill.define` as a decorator to a fully type-annotated function or
    method whose body is either empty or `raise NotHandled`. The docstring is a
    [format string](https://docs.python.org/3/library/string.html#format-string-syntax)
    prompt: its `{...}` fields are filled at call time (see *Prompt assembly*
    below) and the LLM's response is decoded to the return type.

    `Skill.define` validates the definition and raises if:

    - the function has no docstring (every `Tool` needs one);
    - a `{...}` field names something that is neither a parameter nor a name in
      lexical scope — every field must resolve at call time;
    - a doctest example (`>>>`) in the docstring contains an active `{...}` field:
      doctests must be constant, since the whole docstring is formatted into the
      prompt at call time; escape any literal braces as `{{` and `}}`.

    See `effectful.ops.types.Operation.define` for more on `Skill.define`.

    The following skill writes limericks on a given theme:

    ```python
    @Skill.define
    def limerick(theme: str) -> str:
        \"\"\"Write a limerick on the theme of {theme}. Do not use any tools.\"\"\"
    ```

    ## Structured output

    Skills may return types that are not strings.
    The output from the LLM is then decoded before being returned to the user.

    For example, this skill returns integers:

    ```python
    @Skill.define
    def primes(first_digit: int) -> int:
        \"\"\"Give a prime number with {first_digit} as the first digit. Do not use any tools.\"\"\"
    ```

    Structured generation is used to constrain the LLM to return values that can be decoded without error.

    Skills can return complex data structures, such as dataclasses:

    ```python
    @dataclass
    class KnockKnockJoke:
        whos_there: str
        punchline: str

    @Skill.define
    def write_joke(theme: str) -> KnockKnockJoke:
        \"\"\"Write a knock-knock joke on the theme of {theme}. Do not use any tools.\"\"\"
    ```

    Many common Python data types are decodable without additional effort.
    To register a decoder for a custom type, see `effectful.handlers.llm.encoding.type_to_encodable_type`.

    ## Using tools

    Instances of `Tool` in a `Skill`'s lexical scope may be called by the LLM
    during completion, and are offered automatically. Scope follows ordinary
    Python rules: enclosing-function locals, module globals, and — for a method
    skill — sibling `Tool`/`Skill` methods on the same class. A skill
    cannot call a tool it cannot lexically see, so it should use only tools that
    are in scope and relevant to the task. Skills are themselves tools,
    enabling composition into agent workflows.

    ## Prompt assembly

    A call produces two messages. The **system message** is assembled once per
    conversation, in two `#` halves — the harness the call runs under, then the
    task itself — each ordered most-constant-first so the document caches well:

    | # | Section heading | Content | Constant over |
    | - | --------------- | ------- | ------------- |
    | 1 | `# Harness` | What the installed handlers provide, a `##` subsection each: the framework concepts (the package overview plus a `###` per concept — `Skill`, `Tool`, `Agent`, `Encodable`) followed by one per capability handler (a Python REPL, code synthesis, readers for lexically scoped values), every one of them sourced from a real docstring | the handler stack |
    | 2 | `# <name><signature>` | The call, introspected from this skill, as the `##` subsections below | the call |
    | 2.1 | `## Module <name>` | Source of the skill's module (docstring if source is unavailable) | the module |
    | 2.2 | `## Agent <cls>` (or `## Skill`) | Agent docstring, then a `### <name><signature>` spec — prompt with `{...}` holes intact and argument JSON schemas — for every skill sharing the instance's history (an `Agent`'s methods, or just this skill) | the instance |
    | 2.3 | `## Imported modules` | Table of in-scope imports (name → module) | the scope |
    | 2.4 | `## Lexical scope` | Table of other in-scope bindings (name → type) | the scope |

    Section 1 is contributed entirely by handlers, so a stack that installs none
    of them omits it; any section that ends up empty is left out of the document
    entirely.

    The **user message** is the per-call part — only its changing values are
    re-sent each turn; everything constant lives in the system message above. It
    has two parts:

    | # | Part | Content |
    | - | ---- | ------- |
    | 1 | Header | `<name><signature>` — identifies which skill this turn calls |
    | 2 | Body | The docstring with each `{...}` hole replaced by the encoded value of that argument or in-scope name (non-text values, such as images, as separate content blocks) |

    """

    __context__: collections.ChainMap[str, typing.Any]

    @classmethod
    def _validate_doctests_constant(cls, skill: "Skill", doc: str) -> None:
        """Validate that no format string variables are spliced into doctests.

        The whole docstring is ``str.format``-ed into the prompt at call time,
        so an active replacement field inside a ``>>>`` example would be
        substituted, breaking the example. Doctests must therefore be constant:
        the example source, expected output and exception message may contain
        only escaped braces (``{{``/``}}``), never active fields.

        :raises TypeError: If any doctest example contains an active field.
        """
        try:
            parts = doctest.DocTestParser().parse(doc, skill.__name__)
        except ValueError:
            # Malformed doctest -- not a prompt-field concern; it surfaces when
            # the doctests are actually run, so skip the constancy check here.
            return

        formatter = string.Formatter()
        spliced: list[str] = []
        for part in parts:
            if not isinstance(part, doctest.Example):
                continue
            for text in (part.source, part.want, part.exc_msg or ""):
                try:
                    spliced.extend(
                        field_name
                        for _, field_name, _, _ in formatter.parse(text)
                        if field_name is not None
                    )
                except ValueError:
                    # An unbalanced brace (e.g. a bare ``{`` or ``}``) is also
                    # non-constant: ``str.format`` would reject it at call time.
                    spliced.append("<unbalanced brace>")

        if spliced:
            # Render the auto-numbered empty field ``{}`` readably.
            shown = sorted({f or "{}" for f in spliced})
            raise TypeError(
                f"Skill '{skill.__name__}' splices {shown} "
                f"into a doctest example. Doctests must be constant -- they are "
                f"formatted into the prompt at call time, so they may not contain "
                f"format fields. Escape literal braces as '{{{{' and '}}}}'."
            )

    @classmethod
    def _validate_prompt(
        cls,
        skill: "Skill",
        context: collections.ChainMap[str, typing.Any],
    ) -> None:
        """Validate that all format string variables in the docstring
        refer to names resolvable at call time.

        Each variable must be either a parameter in the signature
        or a name captured in the lexical context. Additionally, doctest
        examples in the docstring must be constant (see
        :meth:`_validate_doctests_constant`).

        :raises TypeError: If any format string variable cannot be resolved, or
            a format field is spliced into a doctest example.
        """
        assert skill.__doc__ is not None
        doc = skill.__doc__
        cls._validate_doctests_constant(skill, doc)
        formatter = string.Formatter()
        param_names = set(skill.__signature__.parameters.keys())
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
                f"Skill '{skill.__name__}' docstring references undefined "
                f"variables {list(sorted(unresolved))} that are not in the signature "
                f"{{{skill.__signature__}}} or lexical scope."
            )

    def __get__[S](self, instance: S | None, owner: type[S] | None = None):
        if hasattr(self, "_name_on_instance") and hasattr(
            instance, self._name_on_instance
        ):
            return getattr(instance, self._name_on_instance)

        result = super().__get__(instance, owner)
        self_param_name = list(self.__signature__.parameters.keys())[0]
        result.__context__ = self.__context__.new_child({self_param_name: instance})
        if isinstance(instance, Agent):
            assert isinstance(result, Skill) and not hasattr(result, "__history__")
            result.__history__ = instance.__history__  # type: ignore[attr-defined]
            result.__self__ = instance  # type: ignore[attr-defined]
        return result

    @classmethod
    def define[**Q, V](
        cls, default: collections.abc.Callable[Q, V], *args, **kwargs
    ) -> "Skill[Q, V]":
        """Define a skill.

        `define` takes a function and can be used as a decorator.
        The function's docstring should be a prompt, which may be templated in the function arguments.
        The prompt will be provided with any instances of `Tool` that exist in the lexical context as callable tools.

        See `effectful.ops.types.Operation.define` for more information on the use of `Skill.define`.

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

        globals_proxy: types.MappingProxyType[str, typing.Any] = types.MappingProxyType(
            frame.f_globals
        )
        contexts: list[types.MappingProxyType[str, typing.Any]] = []
        for fn_name in enclosing_fns:
            while frame is not None and frame.f_locals is not frame.f_globals:
                if frame.f_code.co_name == fn_name:
                    contexts.append(types.MappingProxyType(frame.f_locals))
                    frame = frame.f_back
                    break
                frame = frame.f_back
        contexts.append(globals_proxy)
        context: collections.ChainMap[str, typing.Any] = collections.ChainMap(
            *typing.cast(
                list[collections.abc.MutableMapping[str, typing.Any]], contexts
            )
        )
        op = super().define(default, *args, **kwargs)
        op.__context__ = context  # type: ignore[attr-defined]
        # Keep validation on original define-time callables, but skip the bound wrapper path.
        # to avoid dropping `self` from the signature and falsely rejecting valid prompt fields like `{self.name}`.
        is_bound_wrapper = (
            isinstance(default, types.MethodType) and default.__self__ is not None
        )
        if not isinstance(op, staticmethod | classmethod) and not is_bound_wrapper:
            cls._validate_prompt(typing.cast(Skill, op), context)

        return typing.cast(Skill[Q, V], op)


# alias for backwards compatibility
Template = Skill


class Agent(abc.ABC):
    """Mixin that gives each instance a persistent LLM message history.

    Subclass and decorate methods with `Skill.define`.
    Each instance accumulates messages across calls so the LLM sees
    prior conversation context.

    Agents compose freely with `dataclasses.dataclass` and other
    base classes.  Instance attributes are available in skill
    docstrings via `{self.attr}`.

    Set `self.__agent_id__` (a plain attribute, read lazily -- see below) to make
    this instance's history and declared dataclass fields persist across
    process restarts when a persistence handler (see
    `effectful.handlers.llm.harness.durability.persistence.SQLitePersister`) is installed.
    Leave it unset (the default) for a normal, transient instance -- it still
    gets a private history, just not backed by any database, and it is never
    checkpointed even if a persistence handler happens to be active.

    `Agent` itself is deliberately *not* a dataclass (making it one would
    make every subclass, even ones with a hand-written `__init__`, look like
    a dataclass too -- `dataclasses.is_dataclass()` is inherited -- which
    breaks any such subclass under `effectful`'s generic dataclass-replace
    evaluation machinery). Nothing here depends on constructor timing, so
    there's no chaining requirement of any kind: `__agent_id__` and
    `__persistent__` are derived lazily, on first access, from whatever
    `self.__agent_id__` happens to be at that point -- a subclass just needs
    `self.__agent_id__` to end up set to a stable string, however it prefers to
    do that (a `@dataclass` field, a custom `__init__`, or nothing at all,
    for a transient instance).

    Don't force access to `__history__` from within your own `__init__` --
    it's meant to load lazily, on first real use, not at construction time.

    Example:

    ```python
    @dataclass
    class ChatBot(Agent):
        bot_name: str

        @Skill.define
        def send(self, user_input: str) -> str:
            \"""Friendly bot named {self.bot_name}. User writes: {user_input}\"""

    def main():
        chatbot = ChatBot()
        chatbot.send("Hi! How are you? I am in France.")
        chatbot.send("Remind me again, where am I?")  # sees prior context
    ```

    ## Encapsulation via lexical scope

    Since scope is ordinary Python scope, defining agents inside a function
    partitions their `Skill`s and `Tool`s into disjoint sets:

    ```python
    class Chatbot(Agent):
        @Skill.define
        def respond(self, user_query: str) -> str: ...

    class TravelAdvisor(Agent):
        @Skill.define
        def recommend(self, user_query: str) -> str: ...
        @Tool.define
        def search_weather(self, city: str) -> str: ...

    def main():
        chatbot, advisor = Chatbot(), TravelAdvisor()

        @Skill.define
        def simulate(chatbot, advisor) -> str:
            \"""Use {chatbot} and {advisor} to simulate a conversation.\"""
            ...
    ```

    `chatbot.respond` sees only its own methods (plus module-level definitions),
    not `advisor`'s; `simulate` sees `chatbot` and `advisor`, but they cannot see
    `simulate`. Inlining these definitions into module scope instead would let
    every skill see every other. Agents that need overlapping toolsets should
    share tools through a common base class or mixin rather than redefining them.

    """

    def __init__(self, __agent_id__: str | None = None):
        if __agent_id__ is not None:
            self.__agent_id__ = __agent_id__

    __agent_id__: str

    @property
    @typing.final
    def __is_persistent__(self) -> bool:
        if not hasattr(self, "__agent_id__"):
            self.__agent_id__ = f"EPHEMERAL-{uuid.uuid4()}"
        return len(self.__agent_id__) > 0 and not self.__agent_id__.startswith(
            "EPHEMERAL-"
        )

    @functools.cached_property
    def __history__(
        self,
    ) -> collections.abc.MutableSequence[collections.abc.Mapping[str, typing.Any]]:
        history: collections.abc.MutableSequence[
            collections.abc.Mapping[str, typing.Any]
        ] = []
        if self.__is_persistent__:
            # Deferred import: completions.py imports Agent/Skill from this
            # module, so this can only be resolved at call time, not at module
            # load time. The query below and `SQLitePersister.__init__`'s
            # `CREATE TABLE checkpoints` must be kept in sync.
            from effectful.handlers.llm.harness.durability.persistence import (
                SQLitePersister,
            )

            conn = SQLitePersister._checkpoint_connection()
            if conn is not None:
                with conn:
                    row = conn.execute(
                        "SELECT state, history FROM checkpoints WHERE agent_id = ?",
                        (self.__agent_id__,),
                    ).fetchone()
                if row is not None:
                    state_blob, history_json = row
                    for key, value in pickle.loads(state_blob).items():
                        setattr(self, key, value)
                    history = list(json.loads(history_json))
        return history


if typing.TYPE_CHECKING:
    type Encodable[T] = typing.Annotated[T, "encoded"]
else:

    class Encodable:
        """The type-driven JSON bridge between Python values and the LLM.

        `Encodable[T]` maps a Python type `T` to a Pydantic-compatible type
        whose JSON schema and (de)serialization the harness uses to move
        values across the model boundary in both directions:

        - **Encoding (Python -> model):** argument and tool-result *values*
            spliced into prompts are serialized to JSON via `Encodable[type]`,
            so the model sees a faithful, schema-shaped rendering of each value
            (including non-text values such as images, emitted as content
            blocks).
        - **Decoding (model -> Python):** a `Skill`'s structured return
            value and the arguments of every tool call are validated and
            decoded from the model's JSON back into real Python objects through
            the same `Encodable[type]` schema, so the value handed to your code
            already has the declared type.

        Custom types register their JSON representation with
        `TypeToPydanticType`. Because the
        encoding is derived from the *type*, it is the single source of truth
        for both the schema shown to the model and the validation applied to
        its output.
        """

        def __class_getitem__(cls, item):
            from effectful.handlers.llm.harness.serialization import TypeToPydanticType

            return TypeToPydanticType().evaluate(item)
