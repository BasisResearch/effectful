"""Answering a `Skill` by synthesizing its body.

This is the declarative "CodeAdapt" workflow: the LLM writes code implementing
the body of the Skill rather than reasoning out the answer itself.
`FinalBodySynthesizer` offers the synthesis tool *alongside* the Skill's normal
completion paths rather than replacing them -- across turns the model may freely
call any other tool in scope (their results are fed back as usual), and it may
still answer the return type directly via structured output. The loop terminates
when it either answers directly or calls ``write_and_run_body``. To force the
synthesis path, pass ``tool_choice="required"``; handler config is forwarded to
the model request.

The function is synthesized by reusing the existing ``Callable`` synthesis
machinery: the tool's argument is typed as ``Callable[[params], ret]``, so
`call_assistant`'s tool-call decoding parses, type-checks, compiles and executes
the model's code into a real function before it is applied. An eval provider
(`~effectful.handlers.llm.harness.execution.builtin.BuiltinExecutor` or
`~effectful.handlers.llm.harness.execution.restricted.RestrictedPythonExecutor`)
must therefore be installed.

Failures compose with
`~effectful.handlers.llm.harness.durability.retrying.TenacityRetryer`: a function
that fails to synthesize surfaces as a `ToolCallDecodingError`, and one that
raises when applied to the inputs as a `ToolCallExecutionError`; both are fed
back to the model as a tool message and the loop continues so it can revise::

    with (
        handler(AgentLoop()),
        handler(LiteLLMConfigurer(model="gpt-5-mini")),
        handler(HistoryBuilder()),
        handler(FinalBodySynthesizer()),
        handler(TenacityRetryer()),
    ):
        ...
"""

import ast
import collections.abc
import functools
import inspect
import textwrap
import types
import typing
from collections.abc import Callable

import pydantic

import effectful.handlers.llm.harness.execution.hooks
import effectful.handlers.llm.harness.validation.hooks
from effectful.handlers.llm.harness.durability.transaction import (
    ClearScope,
    HistoryBuilder,
    compact_,
)
from effectful.handlers.llm.harness.hooks import (
    PromptInjectingInterpretation,
    ToolCallExecutionError,
    call_agent,
    call_assistant,
    call_tool,
)
from effectful.handlers.llm.harness.serialization import (
    _IS_FINAL_KEY,
    _TYPE_CHECK_ANCHOR_KEY,
    DecodedToolCall,
    EncodedFunction,
    TypeToPydanticType,
    _inline_refs,
    _serialize_callable,
)
from effectful.handlers.llm.harness.synthesis.function import (
    SplicedRegion,
    SynthesizedFunction,
    _def_nodes,
    _recover_skill_def,
)
from effectful.handlers.llm.types import Encodable, Skill, Tool
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import implements


def _splice_body(
    generated: ast.Module,
    module_ast: ast.Module,
    skill_def: ast.FunctionDef | ast.AsyncFunctionDef,
) -> SplicedRegion:
    """Splice a synthesized function in as the anchor Skill's *own body*.

    Unlike `splice_into_source` (which appends ``return <fn>`` and checks that the
    Skill returns the synthesized *function*), this treats the synthesized
    function as the Skill's implementation: the Skill keeps its own
    authoritative signature and its body becomes ``[<helpers/imports the model
    wrote>, *<the synthesized function's body>]``.  mypy then checks that body
    against the Skill's declared parameter and return types -- so a body that
    fails to return the declared type is rejected.  The synthesized function's own
    parameter list (including any ``self``) is intentionally discarded: the
    Skill's real signature is the contract.

    ``generated`` is the model's whole ``code`` parsed to a module; its
    *last* statement is the implementation, any earlier statements are helper
    definitions/imports.  For example, given the Skill ::

        @Skill.define
        def parity(numbers: Sequence[int]) -> bool:
            '''True iff the sum of {numbers} is odd.
            >>> parity([1, 2])  # doctest: +SKIP
            True
            '''

    a model that submits this ``generated`` (note the header on its final ``def``
    -- ``numbers: list`` -- is discarded) ::

        import math
        def _odd(n: int) -> bool:
            return n % 2 == 1
        def parity(numbers: list) -> bool:
            return _odd(sum(numbers))

    is spliced into the Skill's real source as ::

        @Skill.define
        def parity(numbers: Sequence[int]) -> bool:   # authoritative header kept
            import math
            def _odd(n: int) -> bool:
                return n % 2 == 1
            return _odd(sum(numbers))                  # from the final def's body

    so mypy checks the grafted body against ``numbers: Sequence[int]`` and
    ``-> bool``.  The helper ``_odd`` and ``import math`` (everything before the
    final ``def``) become locals at the top of the body; only the final ``def``'s
    *body* is taken, under the Skill's own header.

    Returns the modified module source and the ``[lo, hi]`` line span from the
    ``def`` line through the last body line, or ``None`` when the anchor's source
    can't be recovered (REPL/notebook skill -- the caller skips rather than
    guesses). Raises ``RuntimeError`` on source drift, via `_recover_skill_def`.
    """
    last = generated.body[-1]
    assert isinstance(last, ast.FunctionDef | ast.AsyncFunctionDef)

    # Keep the Skill's real header (authoritative annotations, `self` for
    # methods); replace only its body with the model's helpers/imports followed by
    # the synthesized function's body statements, so the declared return type is
    # enforced. Any docstring/doctests in the recovered source are dropped.
    skill_def.body = [*generated.body[:-1], *last.body]

    # Report the def line through the end of the body. Unlike `splice_into_source`,
    # the region starts at the `def` line (not the first body statement): mypy
    # anchors "Missing return statement"/"empty-body" there, and a body that doesn't
    # return the Skill's declared type is a real defect we want to catch. The
    # header is the Skill's own (recovered, resolvable) signature -- sourceless
    # skills return `None` above and skip -- so including it adds no spurious
    # signature diagnostics. Decorator lines sit above `spliced.lineno` and stay out.
    # `skill_def` is still a node in `module_ast` (only its body changed), so its
    # walk-order index is stable across the unparse round-trip.
    def_index = _def_nodes(module_ast).index(skill_def)
    checked_source = ast.unparse(ast.fix_missing_locations(module_ast))
    spliced = _def_nodes(ast.parse(checked_source))[def_index]
    lo = spliced.lineno
    hi = spliced.body[-1].end_lineno or lo
    return checked_source, lo, hi


class SkillBody:
    """The synthesized *body* of a `Skill`, as opposed to a general `Callable`.

    Used only as the type of `write_and_run_body`'s ``implementation`` parameter (see
    `effectful.handlers.llm.harness.synthesis.body.FinalBodySynthesizer`).  A `SkillBody[[P],
    R]` carries the Skill's parameter and return types exactly like a
    `Callable`, but gets its own `TypeToPydanticType` case (`_pydantic_skill_body`)
    so the synthesized function is type-checked against the enclosing Skill's
    source and its doctests run with self/recursive calls routed to the synthesized
    implementation.  The enclosing `Skill` is recovered from the decode context
    (the ``anchor``), so no state rides on the type itself.
    """

    def __class_getitem__(cls, item):
        return types.GenericAlias(cls, item)


class SynthesizedSkillBody(SynthesizedFunction):
    """Structured output for synthesizing a `Skill`'s body (`write_and_run_body`).

    Decoded through `_pydantic_skill_body`: the function is type-checked against
    the enclosing Skill's source and its doctests are run with self/recursive
    calls routed to the synthesized implementation.

    Unlike `SynthesizedFunction`, the parameter and return *annotations* are not
    required: a Skill body is type-checked against the Skill's own signature
    (see `splice_skill_body`), so the model may omit or vary them -- in
    particular it need not annotate the ``self`` receiver of an instance-method
    Skill.
    """

    code: str = pydantic.Field(
        ...,
        description=textwrap.dedent("""
        The complete Python source implementing the Skill shown in its spec.
        The code MUST satisfy the following constraints, or it will fail validation:

        <constraints>
        1. The code MUST be one complete syntactically valid Python module.
        2. The code MUST NOT use star imports or ``__future__`` imports.
        3. The function definition MUST be the LAST statement - do not add any code after it.
        4. Write the function with the Skill's signature; parameter and return
        annotations are optional.
        5. Do not include a docstring or doctests; the Skill's are supplied automatically.
        </constraints>
        """),
    )

    # A Skill body is checked against the Skill's own (already-annotated)
    # signature, so the synthesized body's annotations are optional.
    _require_annotations: typing.ClassVar[bool] = False


@TypeToPydanticType.register(SkillBody)
def _pydantic_skill_body(ty: typing.Any) -> typing.Any:
    """`TypeToPydanticType` case for a free-function `Skill` body.

    Like `_pydantic_callable`, but the synthesized function is checked against the
    enclosing Skill's source (the ``anchor`` in the decode context) and its
    doctests are run with the Skill's own name/op routed back to the synthesized
    implementation, so a doctest that calls the Skill (including for recursion)
    exercises the freshly synthesized code rather than re-invoking the model.
    """
    typed_enc = SynthesizedSkillBody._create_model_from_callable_type(
        ty if typing.get_args(ty) else Callable[..., typing.Any],  # type: ignore[arg-type]
    )

    def _validate(
        value: SynthesizedSkillBody | dict | str | Callable,
        info: pydantic.ValidationInfo,
    ) -> Callable:
        if isinstance(value, str):
            value = typed_enc.model_validate({"code": value})
        elif isinstance(value, dict):
            value = typed_enc.model_validate(value)
        elif callable(value):
            return typing.cast(Callable, value)
        ctx = info.context or {}
        anchor = ctx.get(_TYPE_CHECK_ANCHOR_KEY)
        if anchor is not None:
            # skill bodies should not have access to call-local variables
            assert isinstance(anchor, Skill)
            ctx = anchor.__context__

        filename = f"<synthesis:{id(value.code)}>"
        module: ast.Module = effectful.handlers.llm.harness.execution.hooks.parse(
            value.code, filename
        )

        # `None` means the Skill's source can't be recovered (REPL/exec/notebook
        # skill): skip the type check rather than guess, but still route the
        # doctests below -- that only needs the anchor op, not its source.
        anchor_asts = _recover_skill_def(anchor) if anchor is not None else None
        if anchor_asts is not None:
            spliced = _splice_body(module, *anchor_asts)
            effectful.handlers.llm.harness.validation.hooks.type_check(*spliced)

        bytecode: types.CodeType = (
            effectful.handlers.llm.harness.execution.hooks.compile(module, filename)
        )
        g: dict[str, typing.Any] = {k: v for k, v in ctx.items() if k.isidentifier()}
        effectful.handlers.llm.harness.execution.hooks.exec(bytecode, g)
        result = g[module.body[-1].name]  # type: ignore

        if anchor is None:
            effectful.handlers.llm.harness.validation.hooks.run_doctests(result, g)
            return result
        # Shadow the global name the doctests call and route the Skill op back
        # into the synthesized function.
        result = functools.wraps(anchor)(result)
        g.update({anchor.__name__: result})
        with handler({anchor: result}):
            effectful.handlers.llm.harness.validation.hooks.run_doctests(result, g)
        return result

    if typing.get_args(ty):
        ty_ = collections.abc.Callable[typing.get_args(ty)]  # type: ignore
    else:
        ty_ = collections.abc.Callable

    return typing.Annotated[
        pydantic.InstanceOf[ty_],  # type: ignore
        pydantic.BeforeValidator(_validate),
        pydantic.PlainSerializer(lambda value: _serialize_callable(value)),
        pydantic.WithJsonSchema(
            _inline_refs(pydantic.TypeAdapter(typed_enc).json_schema()),
            mode="validation",
        ),
        pydantic.WithJsonSchema(
            EncodedFunction.model_json_schema(), mode="serialization"
        ),
    ]


class MethodSkillBody(SkillBody):
    """A `SkillBody` for an *instance-method* Skill.

    Carries the method/free distinction on the type's origin (context-free schema
    generation reads it) so `write_and_run_body`'s description names the leading
    receiver ``self`` and the receiver is exempt from the annotation requirement --
    the model no longer has to reverse-engineer that the first parameter is ``self``.
    The Skill's real signature (which includes the receiver) remains the
    type-check contract; see `splice_skill_body`.
    """


class SynthesizedMethodSkillBody(SynthesizedSkillBody):
    """Structured output for synthesizing an *instance-method* `Skill`'s body.

    Decoded through `_pydantic_skill_body`: the function is type-checked against
    the enclosing Skill's source and its doctests are run with self/recursive
    calls routed to the synthesized implementation.

    Unlike `SynthesizedFunction`, the parameter and return *annotations* are not
    required: a Skill body is type-checked against the Skill's own signature
    (see `splice_skill_body`), so the model may omit or vary them -- in
    particular it need not annotate the ``self`` receiver of an instance-method
    Skill.
    """

    code: str = pydantic.Field(
        ...,
        description=textwrap.dedent("""
        The complete Python source implementing the instance-method Skill shown in
        its spec. The code MUST satisfy the following constraints, or it will fail
        validation:

        <constraints>
        1. The code MUST be one complete syntactically valid Python module.
        2. The code MUST NOT use star imports or ``__future__`` imports.
        3. The function definition MUST be the LAST statement - do not add any code after it.
        4. Write the function with the Skill's signature: its FIRST parameter is the
        instance receiver ``self`` (which you may leave unannotated); all other parameter
        and return annotations are optional too.
        5. Do not include a docstring or doctests; the Skill's are supplied automatically.
        </constraints>
        """),
    )

    @classmethod
    def _param_names(cls, param_types: typing.Iterable[typing.Any]) -> list[str]:
        # The method's callable type already carries the receiver as its first
        # parameter (with an uninformative Agent-class type); relabel it ``self`` so
        # the model reproduces it rather than inventing one -- do NOT prepend a receiver.
        names = super()._param_names(param_types)
        if names:
            names[0] = "self"
        return names

    @classmethod
    def _extra_instructions(cls) -> str:
        return (
            "\n\nThis implements an instance method: the first parameter is the "
            "instance receiver `self`. Include it as the first parameter; you may "
            "leave it unannotated."
        )


def _class_skill_of(op: typing.Any) -> typing.Any | None:
    """The class-level `Skill` underlying an Agent-method Skill ``op``.

    Returns ``None`` for a free-function skill (whose ``__default__`` is a plain
    function rather than a bound method).
    """
    default = getattr(op, "__default__", None)
    if isinstance(default, types.MethodType):
        return default.__func__.__wrapped__  # type: ignore[attr-defined]
    return None


def _method_instance(op: typing.Any, class_skill: typing.Any) -> typing.Any | None:
    """The instance ``op`` is bound to, if ``op`` is ``class_skill`` on *some*
    instance; otherwise ``None``.
    """
    if class_skill is not None and _class_skill_of(op) is class_skill:
        return op.__default__.__self__
    return None


@TypeToPydanticType.register(MethodSkillBody)
def _pydantic_method_skill_body(ty: typing.Any) -> typing.Any:
    """`TypeToPydanticType` case for an instance-method `Skill` body.

    Registered separately from `SkillBody` (rather than reached via subclass
    MRO) so the method/free distinction is an explicit dispatch: it surfaces the
    leading ``self`` receiver in the signature hint, and its doctests -- which build
    their own instances -- route ``agent.method(...)`` on *any* instance to the
    synthesized implementation.
    """
    typed_enc = SynthesizedMethodSkillBody._create_model_from_callable_type(
        ty if typing.get_args(ty) else Callable[..., typing.Any],  # type: ignore[arg-type]
    )

    def _validate(
        value: SynthesizedMethodSkillBody | dict | str | Callable,
        info: pydantic.ValidationInfo,
    ) -> Callable:
        if isinstance(value, str):
            value = typed_enc.model_validate({"code": value})
        elif isinstance(value, dict):
            value = typed_enc.model_validate(value)
        elif callable(value):
            return typing.cast(Callable, value)
        ctx = info.context or {}
        anchor = ctx.get(_TYPE_CHECK_ANCHOR_KEY)
        if anchor is not None:
            # skill bodies should not have access to call-local variables
            assert isinstance(anchor, Skill)
            ctx = anchor.__context__

        filename = f"<synthesis:{id(value.code)}>"
        module: ast.Module = effectful.handlers.llm.harness.execution.hooks.parse(
            value.code, filename
        )
        anchor_asts = _recover_skill_def(anchor) if anchor is not None else None
        if anchor_asts is not None:
            spliced = _splice_body(module, *anchor_asts)
            effectful.handlers.llm.harness.validation.hooks.type_check(*spliced)

        bytecode: types.CodeType = (
            effectful.handlers.llm.harness.execution.hooks.compile(module, filename)
        )
        g: dict[str, typing.Any] = {k: v for k, v in ctx.items() if k.isidentifier()}
        effectful.handlers.llm.harness.execution.hooks.exec(bytecode, g)
        result = g[module.body[-1].name]  # type: ignore

        class_skill = _class_skill_of(anchor) if anchor is not None else None
        if class_skill is None:
            effectful.handlers.llm.harness.validation.hooks.run_doctests(result, g)
            return result
        # A fresh instance's `agent.method(...)` dispatches through
        # `call_agent`, which we intercept and redirect to the synthesized
        # implementation.
        result = functools.wraps(class_skill)(result)

        def _doctest_apply(op, *args, **kwargs):
            instance = _method_instance(op, class_skill)
            if instance is None:
                return fwd()
            return class_skill(instance, *args, **kwargs)

        with handler({call_agent: _doctest_apply, class_skill: result}):
            effectful.handlers.llm.harness.validation.hooks.run_doctests(result, g)
        return result

    if typing.get_args(ty):
        ty_ = collections.abc.Callable[typing.get_args(ty)]  # type: ignore
    else:
        ty_ = collections.abc.Callable

    return typing.Annotated[
        pydantic.InstanceOf[ty_],  # type: ignore
        pydantic.BeforeValidator(_validate),
        pydantic.PlainSerializer(_serialize_callable),
        pydantic.WithJsonSchema(
            _inline_refs(pydantic.TypeAdapter(typed_enc).json_schema()),
            mode="validation",
        ),
        pydantic.WithJsonSchema(
            EncodedFunction.model_json_schema(), mode="serialization"
        ),
    ]


def _callable_type_from_signature(
    signature: inspect.Signature,
) -> type[types.FunctionType]:
    """Construct a `Callable` type from a signature.

    Raises if the signature is recursive (e.g. a Skill that returns itself)
    or contains variadic parameters (which cannot be expressed in a `Callable`
    type).
    """
    param_types = []
    for pname, param in signature.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f"Cannot synthesize a function for parameter "
                f"'{pname}' of kind {param.kind.description}: variadic parameters "
                "cannot be expressed as a Callable type signature."
            )
        param_types.append(
            param.annotation
            if param.annotation is not inspect.Parameter.empty
            else typing.Any
        )
    return_type = signature.return_annotation
    return collections.abc.Callable[param_types, return_type]  # type: ignore


class FinalBodySynthesizer(PromptInjectingInterpretation):
    """You can state a Skill's answer directly, or you can *compute* it by
    writing an implementation and submitting it with the `write_and_run_body`
    tool. This section is about the tool. Reach for it when
    working the answer out by hand would be error-prone — a search, an
    enumeration, a constraint to check against — or when the Skill's doctests are
    the standard your answer has to meet.

    A direct answer is also accepted, and is the right choice when you already
    hold the value: do not wrap a value you have in hand inside a function that
    ignores its arguments and returns a constant.

    The tool's own description says what to submit and what the code must
    satisfy; follow it rather than any recollection of how such a tool usually
    works. Two things it does not tell you. Your function may reference names
    from the lexical scope (see the *Lexical scope* table). And what your
    submission is judged on is the Skill's doctests: the harness attaches the
    Skill's docstring to your function and runs *its* examples, with recursive
    calls to the Skill routed back to your implementation. A solution whose
    doctests fail — or that raises when applied — is rejected and returned to you
    to revise, so the answer only stands once those examples pass.

    A Skill whose declared *return type* is itself a function is a different
    thing, easily confused with this one: you answer it by writing the function
    it returns, as an ordinary direct answer, and this tool is not involved.
    Three rules invert there, and nothing else states them. The signature to
    write is the *returned* function's, taken from the return type — not the
    Skill's own, and with no `self` receiver even when the Skill is a method.
    Every parameter and the return type must be annotated there, where for this
    tool they are optional. And your docstring is kept rather than replaced, so
    if the Skill asks for doctests certifying what you wrote, write them: they
    are run, and they are what your answer is accepted on.

    A successful `write_and_run_body` call ends the call immediately: no further
    turn is taken, and the value of applying your function to the original
    arguments is the Skill's answer. Because it ends the call, it must be the
    *only* tool call in its turn — call any other tools you need on earlier
    turns, and call `write_and_run_body` by itself once you are ready to answer.

    This answers the *current* call only. A submission is not a standing answer:
    if an earlier user message in this conversation was a previous call that you
    answered this way, that answer has already been returned to the program and
    has nothing to do with the question you are being asked now. To answer this
    call by synthesis you must call `write_and_run_body` again.
    """

    # The docstring above is model-facing: it is the `Harness` section this
    # handler adds to the system prompt (see `PromptInjectingInterpretation`),
    # which is why it is written as instructions rather than as description.

    class _SubmitSolutionTool[T](Tool[[collections.abc.Callable[..., T]], T]):
        """The `Tool` a synthesized Skill body is submitted through.

        A distinct type so `call_agent` can tell whether a request already carries
        this handler's tool, and so `call_tool` can recognize a call to it as
        the Skill's answer; the capability itself is described to the model by
        the handler's docstring.
        """

        __toolname__: typing.ClassVar[typing.Literal["write_and_run_body"]] = (
            "write_and_run_body"
        )

        @classmethod
        def define(  # type: ignore[override]
            cls,
            skill: Skill[..., T],
            bound_args: inspect.BoundArguments,
        ) -> Tool[[collections.abc.Callable[..., T]], T]:
            if isinstance(skill.__default__, types.MethodType):
                signature = inspect.signature(skill.__default__.__func__)
                args, kwargs = (
                    (skill.__default__.__self__,) + bound_args.args,
                    bound_args.kwargs,
                )
                body_type = MethodSkillBody[  # type: ignore
                    typing.get_args(_callable_type_from_signature(signature))
                ]
                return_type = signature.return_annotation
            else:
                signature = inspect.signature(skill)
                args, kwargs = bound_args.args, bound_args.kwargs
                body_type = SkillBody[  # type: ignore
                    typing.get_args(_callable_type_from_signature(signature))
                ]
                return_type = signature.return_annotation

            # Put the result through the same validation machinery as the direct path,
            # under the same environment as in call_assistant
            return_encoding: pydantic.TypeAdapter[typing.Any] = pydantic.TypeAdapter(
                Encodable[return_type]  # type: ignore
            )
            env = skill.__context__.new_child(
                bound_args.arguments
                | {_TYPE_CHECK_ANCHOR_KEY: skill, _IS_FINAL_KEY: True}
            )

            def write_and_run_body(
                implementation: body_type,  # type: ignore
                clear: ClearScope = ClearScope.NONE,
            ) -> return_type:  # type: ignore
                """
                Answer this Skill by submitting a Python function that implements
                it (see the `FinalBodySynthesizer` section of the system
                prompt); its return value on the original arguments becomes the
                answer.

                `clear` compacts the conversation as the answer lands; its own
                schema below says what each scope drops. Whichever you pick,
                this submission survives whole -- your message, the source you
                submit and its result -- so anything you want your later self to
                know, write as comments in the body you submit.
                """
                result = implementation(*args, **kwargs)  # type: ignore
                return return_encoding.validate_python(result, context=env)

            return super().define(write_and_run_body, name=cls.__toolname__)

    @implements(call_tool)
    def call_tool(self, tool_call: DecodedToolCall) -> typing.Any:
        """Mark a *successful* ``write_and_run_body`` call as the Skill's answer,
        and honour the ``clear`` it was submitted with.

        This is the rule that terminates the completion loop on the synthesis
        path: the model is free to answer directly instead, and every other tool
        call forwards untouched, so setting ``is_final`` here is the only thing
        that distinguishes a submission from an ordinary tool result.

        Only a successful one: when `TenacityRetryer` captures a submission that
        raised -- the synthesized function errored on the real arguments -- it
        hands back the `ToolCallExecutionError` in place of a result, and that is
        no answer to finalize on. Leaving `is_final` alone there is what gives the
        model the next turn to revise, and compacting there would throw away the
        error it needs to do so.

        The compaction runs here rather than anywhere later because there *is* no
        later: this call ends the loop. `compact` keeps the round that asked for
        it, which is what makes that safe -- the surviving history ends on this
        submission and its result rather than on a request nobody answered, and
        the source the model submitted stays in the assistant message's arguments
        where it can read it back on the next call.
        """
        message, result, is_final = fwd(tool_call)
        # Two decisions, not one: whether this submission *is* the answer, and
        # how much transcript it asked to drop. Only the first may gate
        # ``is_final`` -- folding the ``clear`` test into the same condition
        # leaves a successful submission with the default ``clear="none"``
        # unfinalized, and the completion loop then runs forever. `compact_` is
        # already a no-op for that scope, so it needs no guard here.
        if isinstance(tool_call.tool, self._SubmitSolutionTool) and not isinstance(
            result, ToolCallExecutionError
        ):
            compact_(
                HistoryBuilder.get_history(),
                tool_call.id,
                tool_call.bound_args.arguments.get("clear", ClearScope.NONE),
            )
            return message, result, True
        else:
            return message, result, is_final

    @implements(call_agent)
    def call_agent[**P, T](
        self, skill: Skill[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """Offer ``write_and_run_body`` for the duration of this call.

        The tool is built per call, closing over *these* bound arguments: it is
        what applies the submitted function to them, so it cannot be shared
        across calls. It is added by an inner `call_assistant` rule rather than
        up front, because the request's tool set is assembled further down the
        stack and only exists once the assistant is actually being called.

        The guard makes the injection idempotent: a nested call reaching this
        rule again with the tool already in the set forwards unchanged, so a
        recursive Skill is not offered several generations of its own
        submission tool.
        """
        bound_args = skill.__signature__.bind(*args, **kwargs)
        bound_args.apply_defaults()
        tool = self._SubmitSolutionTool.define(skill, bound_args)

        def _add_synthesis_tool(messages, response_type, env, tools=frozenset()):
            if any(isinstance(t, self._SubmitSolutionTool) for t in tools):
                return fwd()
            return fwd(messages, response_type, env, tools | {tool})

        with handler({call_assistant: _add_synthesis_tool}):
            return fwd()
