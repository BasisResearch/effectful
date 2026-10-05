"""The LLM types: :class:`Skill`, :class:`Tool`, :class:`Encodable`, and
the :class:`Agent` runtime type for Skill-owning classes.

Start with :mod:`effectful.handlers.llm` for a minimal Skill and the programming
model. Read the class docstrings here for each type's contract. The harness in
:mod:`effectful.handlers.llm.harness` implements Skill calls. ``Template`` is a
deprecated alias for ``Skill``.
"""

import abc
import collections
import collections.abc
import doctest
import functools
import inspect
import json
import linecache
import pickle
import re
import string
import types
import typing
import uuid

import typing_extensions

import effectful.ops.types

__all__ = ["Agent", "Skill", "Template", "Tool", "Encodable"]


class Tool[**P, T](effectful.ops.types.Operation[P, T]):
    """A Tool is a typed Python callable the model may invoke during a Skill
    turn. Its signature tells the model what arguments to pass and what it
    returns; its docstring explains when to call it. The body runs as ordinary
    Python. Define an explicit Tool on an annotated function or method::

        from effectful.handlers.llm import Tool

        @Tool.define
        def weather(city: str) -> str:
            \"\"\"Return a short weather report for ``city``.\"\"\"
            return {"Chicago": "cold"}.get(city, "unknown")

    With the default ``tool_collection="explicit"``, in-scope Tools are offered
    to the model; ``"none"`` disables lexical Tool collection. A Skill is also
    a Tool, so a Tool call can start a nested Skill turn. Tool advertisement is
    a convenient call interface, not a limit on what model-authored Python can
    reach. Enforce a Tool's own preconditions in its implementation.
    """

    def __init__(
        self, default: collections.abc.Callable[P, T], name: str | None = None
    ):
        if not default.__doc__:
            raise ValueError("Tools must have docstrings.")
        super().__init__(default, name=name)

    if typing.TYPE_CHECKING:
        # Operation.__get__'s overloads, returning Tool; the runtime is inherited.
        @typing.overload
        def __get__[S, **Q](
            self: "Tool[typing.Concatenate[S, Q], T]",
            instance: S,
            owner: "type[S] | None" = None,
        ) -> "Tool[Q, T]": ...

        @typing.overload
        def __get__[S, **Q](
            self: "Tool[typing.Concatenate[S, Q], T]",
            instance: None,
            owner: "type[S]",
        ) -> "Tool[typing.Concatenate[S, Q], T]": ...

        @typing.overload
        def __get__[S, **Q](
            self: "Tool[typing.Concatenate[type[S], Q], T]",
            instance: "S | None",
            owner: "type[S]",
        ) -> "Tool[Q, T]": ...

        @typing.overload
        def __get__[S](
            self, instance: "S | None", owner: "type[S] | None" = None
        ) -> "typing.Self": ...

        def __get__(self, instance: typing.Any, owner: typing.Any = None) -> typing.Any:
            return super().__get__(instance, owner)

    @typing.overload
    @classmethod
    def define[V](cls, default: type[V], *args, **kwargs) -> "Tool[[], V]": ...

    @typing.overload
    @classmethod
    def define[**Q, V](
        cls, default: "staticmethod[Q, V]", *args, **kwargs
    ) -> "effectful.ops.types._StaticMethodOperationDescriptor[Tool[Q, V]]": ...

    @typing.overload
    @classmethod
    def define[V](
        cls, default: "functools.singledispatchmethod[V]", *args, **kwargs
    ) -> "Tool[typing.Concatenate[typing.Any, ...], V]": ...

    @typing.overload
    @classmethod
    def define[**Q, V](
        cls, default: collections.abc.Callable[Q, V], *args, **kwargs
    ) -> "Tool[Q, V]": ...

    @typing.overload
    @classmethod
    def define[S, **Q, V](
        cls, default: "classmethod[S, Q, V]", *args, **kwargs
    ) -> "effectful.ops.types._ClassMethodOpDescriptor[S, Q, V, Tool[Q, V]]": ...

    @classmethod
    def define(cls, default: typing.Any, *args, **kwargs) -> typing.Any:
        """Define a tool.

        Binds the result's type parameters from ``default`` (as `Skill.define`
        does), so a static checker sees ``Tool[<params>, <return>]`` rather
        than an unbound ``Tool[Never, Never]`` -- which is what lets a
        model-written call *expression* to a tool be type-checked against the
        tool's real (possibly generic) signature.

        See `effectful.ops.types.Operation.define` for more information on the
        use of `Tool.define`.

        """
        return super().define(default, *args, **kwargs)


class Skill[**P, T](Tool[P, T]):
    """A Skill is a typed Python function or method answered by a language model.
    Its parameters are the inputs, its return annotation is the answer type, and
    its docstring is the request. Define one on a fully annotated function with
    an empty body::

        from effectful.handlers.llm import Skill

        @Skill.define
        def summarize(text: str) -> str:
            \"\"\"Summarize {text} in one sentence.\"\"\"

    Calling ``summarize(...)`` opens one conversation turn. Each model response
    is a round. A Tool is a typed Python callable offered to the model; Tool
    results and validation feedback can lead to further rounds in that turn.
    A ``str`` reply is used verbatim; other return types are decoded as
    structured values and checked against their annotations.

    A Skill method uses its receiver (``self``) to share successful top-level
    turns with later calls on that object. Ordinary fields can appear in its
    prompt as ``{self.field}``. In-scope Tools, including other Skills, can be
    called during the turn. Define the method on the class to bind it: assigning
    an already defined free Skill to an object later does not give it the object's
    history. For example, an ordinary class can own a Skill and a Tool::

        from dataclasses import dataclass
        from effectful.handlers.llm import Skill, Tool

        @dataclass
        class Librarian:
            notes: dict[str, str]

            @Tool.define
            def lookup(self, topic: str) -> str:
                \"\"\"Return the note for ``topic``.\"\"\"
                return self.notes.get(topic, "no note")

            @Skill.define
            def answer(self, question: str) -> str:
                \"\"\"Answer {question} using ``lookup`` when useful.\"\"\"

    A nested Skill call creates its own turn. On the same receiver it reads
    prior committed history but neither sees the outer turn's current messages
    nor commits its own; on another receiver it can commit independently.

    A Skill docstring is a format string. Use fields such as ``{text}`` only
    for parameters or names visible where the Skill was defined. Escape
    literal braces as ``{{`` and ``}}``. Doctest examples must stay constant:
    do not put an active format field in a ``>>>`` example. The definition is
    rejected if these requirements or the docstring requirement are not met.
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

    def __set_name__(self, owner: type, name: str) -> None:
        """Auto-agentify the class this skill is defined in.

        Defining a `Skill` method is sufficient for its class to behave as an
        `Agent`: this hook (called by Python at class creation, before any
        decorator such as `dataclass` wraps the class) grafts `Agent`'s
        behavior onto ``owner`` without touching its MRO. Two steps, because
        virtual subclassing affects only ``isinstance``/``issubclass``, never
        attribute lookup:

        - copy `Agent`'s class-level descriptors (``__history__``,
          ``__is_persistent__``) onto ``owner``, unless something in its MRO
          already provides them;
        - ``Agent.register(owner)``, so ``isinstance(obj, Agent)`` checks
          (here and in the harness) recognize its instances.
        """
        super().__set_name__(owner, name)
        if issubclass(owner, Agent):
            return
        if issubclass(owner, effectful.ops.types.Term) or owner.__dictoffset__ == 0:
            return
        for attr in ("__history__", "__is_persistent__"):
            if not any(attr in vars(k) for k in owner.__mro__):
                setattr(owner, attr, Agent.__dict__[attr])
        Agent.register(owner)

    @typing.overload
    def __get__[S, **Q](
        self: "Skill[typing.Concatenate[S, Q], T]",
        instance: S,
        owner: "type[S] | None" = None,
    ) -> "Skill[Q, T]": ...

    @typing.overload
    def __get__[S, **Q](
        self: "Skill[typing.Concatenate[S, Q], T]",
        instance: None,
        owner: "type[S]",
    ) -> "Skill[typing.Concatenate[S, Q], T]": ...

    @typing.overload
    def __get__[S, **Q](
        self: "Skill[typing.Concatenate[type[S], Q], T]",
        instance: "S | None",
        owner: "type[S]",
    ) -> "Skill[Q, T]": ...

    @typing.overload
    def __get__[S](
        self, instance: "S | None", owner: "type[S] | None" = None
    ) -> "typing.Self": ...

    def __get__[S](
        self, instance: "S | None", owner: "type[S] | None" = None
    ) -> "Skill[..., T] | typing.Self":
        if (cached := self._cached_instance_op(instance)) is not None:
            return typing.cast("Skill[..., T]", cached)

        result: Skill[..., T] = super().__get__(instance, owner)  # type: ignore[assignment]
        self_param_name = list(self.__signature__.parameters.keys())[0]
        result.__context__ = self.__context__.new_child({self_param_name: instance})
        if isinstance(instance, Agent):
            assert isinstance(result, Skill) and not hasattr(result, "__history__")
            result.__history__ = instance.__history__  # type: ignore[attr-defined]
            result.__self__ = instance  # type: ignore[attr-defined]
        return result

    # Skills reject type and singledispatchmethod defaults, unlike Tool.define.
    @typing.overload
    @classmethod
    def define(cls, default: type, *args, **kwargs) -> typing.NoReturn: ...

    @typing.overload
    @classmethod
    def define[**Q, V](
        cls, default: "staticmethod[Q, V]", *args, **kwargs
    ) -> "effectful.ops.types._StaticMethodOperationDescriptor[Skill[Q, V]]": ...

    @typing.overload
    @classmethod
    def define(
        cls, default: functools.singledispatchmethod, *args, **kwargs
    ) -> typing.NoReturn: ...

    @typing.overload
    @classmethod
    def define[**Q, V](
        cls, default: collections.abc.Callable[Q, V], *args, **kwargs
    ) -> "Skill[Q, V]": ...

    @typing.overload
    @classmethod
    def define[S, **Q, V](
        cls, default: "classmethod[S, Q, V]", *args, **kwargs
    ) -> "effectful.ops.types._ClassMethodOpDescriptor[S, Q, V, Skill[Q, V]]": ...

    @classmethod
    def define(cls, default: typing.Any, *args, **kwargs) -> typing.Any:
        """Define a skill.

        Captures the defining module's globals and true enclosing-function locals
        as ``__context__`` (enclosers are found by matching ``__qualname__``
        segments before ``<locals>`` against the frame stack; class bodies are
        skipped), records the module source for ``_recover_skill_def``, then
        validates the prompt (`_validate_prompt`, `_validate_doctests_constant`).

        See `effectful.ops.types.Operation.define` for the decorator forms.
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
        op.__context__ = context
        if isinstance(_fn, types.FunctionType):
            # The module's source as the function was compiled from it, for
            # `_recover_skill_def`: the file may be edited while this runs.
            linecache.checkcache(_fn.__code__.co_filename)
            _fn.__module_source__ = linecache.getlines(_fn.__code__.co_filename)  # type: ignore[attr-defined]
        # Keep validation on original define-time callables, but skip the bound wrapper path.
        # to avoid dropping `self` from the signature and falsely rejecting valid prompt fields like `{self.name}`.
        is_bound_wrapper = (
            isinstance(default, types.MethodType) and default.__self__ is not None
        )
        if not isinstance(op, staticmethod | classmethod) and not is_bound_wrapper:
            cls._validate_prompt(typing.cast(Skill, op), context)

        return op


# alias for backwards compatibility
Template = Skill


class Agent(abc.ABC):
    """An Agent is the runtime type for an object whose Skill methods share
    conversation history. Defining a Skill method gives an ordinary class this
    behavior and registers it as a virtual ``Agent``; applications normally
    do not inherit it. Explicit inheritance can help a static checker when
    an API requires this type. Reuse the same receiver (``self``) for related
    top-level calls, or a different receiver
    for independent history.

    The receiver's ordinary fields can appear in a Skill prompt as
    ``{self.field}`` and can be used by model-authored Python. Mutating those
    fields is a Python side effect: a failed Skill turn does not undo it.

    For process persistence, give the receiver a stable ``__agent_id__`` and
    install a persistence handler before its first Skill call. The handler
    checkpoints successful history and declared dataclass fields; dynamic
    attributes and live functions are not restored::

        import dataclasses
        from effectful.handlers.llm import Skill

        @dataclasses.dataclass
        class Reviewer:
            __agent_id__: str
            notes: list[str] = dataclasses.field(default_factory=list)

            @Skill.define
            def review(self, draft: str) -> str:
                \"\"\"Review {draft} using {self.notes} when relevant.\"\"\"
    """

    def __init__(self, __agent_id__: str | None = None):
        if __agent_id__ is not None:
            self.__agent_id__ = __agent_id__

    __agent_id__: str

    @property
    @typing.final
    def __is_persistent__(self) -> bool:
        """Whether a persistence handler should checkpoint this instance: false for an unset or ``EPHEMERAL-`` id."""
        if not hasattr(self, "__agent_id__"):
            self.__agent_id__ = f"EPHEMERAL-{uuid.uuid4()}"
        return len(self.__agent_id__) > 0 and not self.__agent_id__.startswith(
            "EPHEMERAL-"
        )

    @functools.cached_property
    def __history__(
        self,
    ) -> collections.abc.MutableSequence[collections.abc.Mapping[str, typing.Any]]:
        """This instance's messages; cached on first access, which is when an active persistence handler restores it."""
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
        """`Encodable[T]` is the type-driven bridge between Python values of type `T` and
        the model.

        Values going to the model -- interpolated arguments, tool results -- are
        rendered as the JSON encoding of their type, with images and other media as
        content blocks. Values coming back -- a Skill's structured answer, a tool
        call's arguments -- are validated and decoded through the same schema, so
        Python code receives the declared type.

        Pydantic-compatible types need nothing. Register a representation for any other
        type with `Encodable.register`; it fixes both the schema the model sees and the
        validation applied to what the model returns::

            from typing import Annotated

            import pydantic

            from effectful.handlers.llm import Encodable

            class Money:
                def __init__(self, cents: int):
                    self.cents = cents

            @Encodable.register(Money)
            def encode_money(typ):
                return Annotated[
                    typ,
                    pydantic.InstanceOf,
                    pydantic.BeforeValidator(lambda v: v if isinstance(v, Money) else Money(v)),
                    pydantic.PlainSerializer(lambda m: m.cents),
                    pydantic.WithJsonSchema({"type": "integer"}),
                ]
        """

        def __class_getitem__(cls, item):
            from effectful.handlers.llm.harness.serialization import TypeToPydanticType

            return TypeToPydanticType().evaluate(item)

        @classmethod
        def register[F: collections.abc.Callable[..., typing.Any]](
            cls, ty: typing_extensions.TypeForm
        ) -> collections.abc.Callable[[F], F]:
            """Give a type an encoding, or replace the one it has.

            The decorated function receives a type expression whose arguments
            are already encoded, and returns a Pydantic-compatible annotation
            of that same type -- adding validators, a serializer and a JSON
            schema, never changing what the annotation denotes.

            >>> import typing, pydantic
            >>> from effectful.handlers.llm import Encodable
            >>> class Money:
            ...     def __init__(self, cents: int):
            ...         self.cents = cents

            Pydantic cannot build a schema for `Money`, so nothing the model
            writes decodes to one:

            >>> pydantic.TypeAdapter(Money)  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
              ...
            pydantic.errors.PydanticSchemaGenerationError: Unable to generate pydantic-core schema for Money

            >>> @Encodable.register(Money)  # type: ignore[attr-defined]
            ... def _encode_money(ty):
            ...     return typing.Annotated[
            ...         ty,
            ...         pydantic.InstanceOf,
            ...         pydantic.BeforeValidator(
            ...             lambda v: v if isinstance(v, Money) else Money(v)
            ...         ),
            ...         pydantic.PlainSerializer(lambda money: money.cents),
            ...         pydantic.WithJsonSchema({"type": "integer"}),
            ...     ]
            >>> adapter = pydantic.TypeAdapter(Encodable[Money])
            >>> adapter.json_schema()
            {'type': 'integer'}
            >>> adapter.dump_python(Money(250), mode="json")
            250
            >>> adapter.validate_python(250).cents
            250
            """
            from effectful.handlers.llm.harness.serialization import TypeToPydanticType

            return TypeToPydanticType.register(ty)
