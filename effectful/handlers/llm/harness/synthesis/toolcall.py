import ast
import collections.abc
import dataclasses
import inspect
import typing
import uuid

import pydantic

import effectful.handlers.llm.harness.execution.hooks
import effectful.handlers.llm.harness.validation.hooks
from effectful.handlers.llm.harness.hooks import (
    AssistantResult,
    Message,
    PromptInjectingInterpretation,
    call_assistant,
)
from effectful.handlers.llm.harness.legibility.lexical import _tool_paths
from effectful.handlers.llm.harness.serialization import (
    _TYPE_CHECK_ANCHOR_KEY,
    DecodedToolCall,
    TypeToPydanticType,
    _NameAndTool,
    _tool_description,
)
from effectful.handlers.llm.harness.synthesis.function import _recover_skill_def
from effectful.handlers.llm.harness.synthesis.snippet import (
    StatefulReplSynthesizer,
    _splice_snippet,
)
from effectful.handlers.llm.types import Encodable, Tool
from effectful.internals.unification import freetypevars
from effectful.ops.semantics import fwd
from effectful.ops.syntax import implements

_EXPR_FILENAME_PREFIX = "<call_expr-"


@dataclasses.dataclass(frozen=True)
class CallExpression(DecodedToolCall[typing.Any]):
    """A decoded tool-call expression: a `DecodedToolCall` for the *underlying*
    tool, remembering the source it was decoded from.

    The decoded form of the ``call`` argument of an expression tool call (see
    `ExpressionToolCaller`).  Decoding does everything except the call itself:
    the source is parsed, type-checked in the anchor Skill's scope, its callee
    resolved (and checked against the tool the expression was submitted for),
    and its argument expressions evaluated -- so what remains is exactly a
    decoded call to the underlying raw tool, and that is what this *is*:
    `ExpressionToolCaller` substitutes it (stamped with the outer call's
    ``id``/``name``) for the wrapper's `DecodedToolCall`, so every `call_tool`
    handler -- the default rule, retryers, tracers -- sees the real tool bound
    to real argument values, and every way the model's expression can be wrong
    surfaces as a decode error rather than an execution error.

    ``source`` (optional on the base, required here) is the round-trip form:
    the advertised schema for an expression call is ``{"call": <source>}``, and
    the evaluated ``bound_args`` may hold values with no JSON encoding at all.
    """

    # Required here: an expression call always knows the source it came from.
    # A bare re-annotation would silently inherit the base's `None` default
    # (dataclasses look defaults up with `getattr`); an explicit default-less
    # `field()` restores MISSING.
    source: str = dataclasses.field()

    # The specific tool a subtype requires the expression to call: `None` on
    # the base class ("any in-scope tool"), and set on the per-wrapper
    # subclasses `_ExpressionToolCallTool.define` creates, which is how the
    # target tool reaches the decoder through the parameter *annotation*.
    # Assigned via `setattr` after class creation rather than in a `type()`
    # dict: a `Tool` is an `Operation`, which is a descriptor, and class
    # creation would invoke its `__set_name__` (class-level *access* is safe --
    # `Operation.__get__` returns the operation itself for a `None` instance).
    __expected_tool__: typing.ClassVar[typing.Any] = None


def _scan_escaping_walrus(root: ast.expr) -> None:
    """Reject a walrus assignment that would bind a name in the enclosing scope.

    An expression must be pure with respect to the session it evaluates in: its
    bindings are discarded (the `eval` operation propagates none), it is not
    recorded in `StatefulReplSynthesizer.repl_history`, and later snippets are
    type-checked without it -- so a walrus that *escapes* would produce a name
    the model can see succeed and then silently lose.  Rejecting it here turns
    that trap into immediate feedback.

    Only the escaping kind is rejected.  A walrus inside a ``lambda`` body binds
    in the lambda's own scope and cannot touch the session; a walrus in a
    comprehension deliberately binds in the *enclosing* scope (PEP 572), so it
    is as escaping as a top-level one.  A lambda's parameter defaults evaluate
    in the enclosing scope, so they stay in the scan even though the body is
    skipped.
    """
    stack: list[ast.AST] = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.NamedExpr):
            raise ValueError(
                "the expression uses a walrus assignment (`:=`) that would bind "
                "a name in the surrounding scope; expressions may not bind names "
                "-- bind names with the `exec_code` tool instead, then reference "
                "them here"
            )
        if isinstance(node, ast.Lambda):
            stack.extend(node.args.defaults)
            stack.extend(d for d in node.args.kw_defaults if d is not None)
            continue
        stack.extend(ast.iter_child_nodes(node))


def _eval_node(node: ast.expr, filename: str, env: dict[str, typing.Any]) -> typing.Any:
    """Evaluate one subexpression of the call through the `compile`/`eval` ops.

    The node keeps its original locations under the call's own `linecache`-
    registered ``filename``, so a raising argument's traceback points into the
    expression the model actually wrote.  A raising subexpression is re-raised
    as ``ValueError`` (chained, so the traceback survives): this happens at
    *decode* time, and only a ``ValueError`` is turned into a validation error
    by pydantic -- anything else would escape the decoder raw.
    """
    try:
        code = effectful.handlers.llm.harness.execution.hooks.compile(
            ast.Expression(node), filename, "eval"
        )
        return effectful.handlers.llm.harness.execution.hooks.eval(code, env)
    except Exception as e:
        raise ValueError(
            f"evaluating `{ast.unparse(node)}` raised {type(e).__name__}: {e}"
        ) from e


@TypeToPydanticType.register(CallExpression)
def _pydantic_type_call_expression(ty):
    """Encode a `CallExpression` as a JSON string of Python source.

    The expression counterpart of the `types.CodeType` encoding in
    `~effectful.handlers.llm.harness.synthesis.snippet`.  Decoding parses the
    source through the `parse` operation under a unique per-expression filename
    (so tracebacks resolve and re-encoding recovers the source), validates its
    shape -- exactly one expression, whose outermost node is a call, with no
    scope-escaping walrus -- type-checks it spliced into the anchor Skill's
    body after the accumulated REPL session, and then *evaluates everything but
    the call*: the callee (checked against the subtype's expected tool) and
    each argument expression, in the Skill's lexical environment layered under
    the live REPL bindings.  The type check is where a polymorphic tool's type
    variables are inferred and enforced: mypy/ty see the call against the
    tool's generic signature in real module source, which JSON-schema argument
    decoding could never express (issue #489).
    """
    # `ty` is the base class or a per-wrapper subclass; class-level access of a
    # `Tool`-valued attribute returns the tool itself (see `CallExpression`).
    expected: Tool | None = ty.__expected_tool__

    def validate(
        value: CallExpression | str, info: pydantic.ValidationInfo
    ) -> CallExpression:
        if isinstance(value, CallExpression):
            return value
        if not isinstance(value, str):
            raise ValueError(
                f"expected a Python expression as a string, got {type(value).__name__}"
            )

        ctx = info.context or {}
        anchor = ctx.get(_TYPE_CHECK_ANCHOR_KEY)

        filename = f"{_EXPR_FILENAME_PREFIX}{uuid.uuid4()}>"
        module = effectful.handlers.llm.harness.execution.hooks.parse(value, filename)

        if len(module.body) != 1 or not isinstance(module.body[0], ast.Expr):
            raise ValueError(
                "expected exactly one Python expression -- no statements, "
                "assignments or imports; bind names with the `exec_code` tool "
                "instead"
            )
        call_node = module.body[0].value
        if not isinstance(call_node, ast.Call):
            raise ValueError(
                "the expression must be a single call to the tool, "
                "e.g. `tool_name(arg1, arg2)`"
            )
        _scan_escaping_walrus(call_node)

        # Type-check the expression in its execution context, exactly as a REPL
        # snippet is (see the `Encodable[CodeType]` decoder): splice the
        # accumulated session snippets plus this expression into the anchor
        # Skill's body and check the expression's span -- before anything is
        # evaluated, so an ill-typed call never runs any of its arguments.
        if anchor is not None and _recover_skill_def(anchor) is not None:
            anchor_asts = _recover_skill_def(anchor)
            assert anchor_asts is not None
            module_ast, skill_def = anchor_asts
            prior = StatefulReplSynthesizer.repl_history()
            prior_src = "".join(s if s.endswith("\n") else s + "\n" for s in prior)
            session = ast.parse(prior_src + value)
            first_new_stmt = len(ast.parse(prior_src).body)
            effectful.handlers.llm.harness.validation.hooks.type_check(
                *_splice_snippet(session, module_ast, skill_def, first_new_stmt),
                lenient=True,
            )

        # The evaluation environment: the Skill's lexical env (the decode
        # context) under the live REPL bindings, so a session rebinding shadows
        # the seeded value exactly as it does inside `exec_code`.
        env = {k: v for k, v in ctx.items() if k.isidentifier()}
        env.update(StatefulReplSynthesizer.repl_env())

        head = _eval_node(call_node.func, filename, env)
        if expected is not None:
            if head is not expected:
                raise ValueError(
                    f"the expression must be a call to the tool "
                    f"{expected.__name__!r} it was submitted for, but its callee "
                    f"evaluated to {head!r}; to call a different tool, invoke "
                    f"that tool instead"
                )
        elif not isinstance(head, Tool):
            raise ValueError(
                f"the expression's callee must be a Tool, but it evaluated to {head!r}"
            )

        # Evaluate the argument expressions in source order, as the call itself
        # would, and bind the values against the tool's real signature.
        args: list[typing.Any] = []
        for node in call_node.args:
            if isinstance(node, ast.Starred):
                args.extend(_eval_node(node.value, filename, env))
            else:
                args.append(_eval_node(node, filename, env))
        kwargs: dict[str, typing.Any] = {}
        for kw in call_node.keywords:
            if kw.arg is None:
                kwargs.update(_eval_node(kw.value, filename, env))
            else:
                kwargs[kw.arg] = _eval_node(kw.value, filename, env)
        try:
            bound_args = inspect.signature(head).bind(*args, **kwargs)
        except TypeError as e:
            raise ValueError(
                f"the expression's arguments do not fit the signature of "
                f"{head.__name__!r} ({inspect.signature(head)}): {e}"
            ) from e

        # `id`/`name` belong to the enclosing raw tool call, which this
        # argument decoder cannot see; `ExpressionToolCaller._call_assistant`
        # stamps them on when it substitutes this call for the wrapper's.
        return ty(
            tool=head, bound_args=bound_args, id="", name=head.__name__, source=value
        )

    return typing.Annotated[
        ty,
        pydantic.InstanceOf,
        pydantic.BeforeValidator(validate),
        pydantic.PlainSerializer(lambda value: value.source),
        pydantic.WithJsonSchema({"type": "string"}),
    ]


class ExpressionToolCaller(PromptInjectingInterpretation):
    """The tools of this Skill's lexical scope are called by writing Python,
    not by filling in JSON arguments. Each such tool takes a single parameter
    `call`: a string containing ONE Python expression that invokes the tool,
    for example `extend_sequence(examples, make_example())`.

    The expression is type-checked and then evaluated in this Skill's lexical
    scope, so arguments may be arbitrary Python expressions over the names in
    the *Lexical scope* table and any bindings made in the REPL session (if one
    is available). Write the call exactly as you would in the surrounding code;
    each tool's own description states the exact reference to invoke it by --
    a bare name for a module-level tool, an attribute access such as
    `self.retrieve(...)` for a tool held by this agent. The expression must be
    a single call to the advertised tool -- no statements, no assignments, and
    no `:=` bindings (bind names with `exec_code` first if you need to). The
    call's return value becomes the tool result.
    """

    # The docstring above is model-facing: it is the `Harness` section this
    # handler adds to the system prompt (see `PromptInjectingInterpretation`), so
    # implementation notes belong in comments like this one.
    #
    # This is the code-generation replacement for
    # `~effectful.handlers.llm.harness.legibility.lexical.LexicalToolExtractor`
    # (issues #489/#505): instead of advertising each lexical tool with a JSON
    # `parameters` schema -- degenerate for a polymorphic tool, whose parameter
    # types carry TypeVars -- every tool reachable from the Skill's scope is
    # wrapped so the model must write a call *expression*. Decoding the
    # expression does all the work up to the call itself (type check, callee
    # check, argument evaluation; see `_pydantic_type_call_expression`), so a
    # bad expression is a `ToolCallDecodingError` the retry loop feeds back.
    # The decoded `CallExpression` IS a `DecodedToolCall` for the underlying
    # tool; `_call_assistant` substitutes it for the wrapper's call, so
    # `call_tool` and every handler of it see the real tool bound to real
    # argument values.
    #
    # Like `LexicalToolExtractor`, install it below anything else that
    # contributes tools: the anchor Skill must not be wrapped (the default
    # `call_assistant` rule subtracts it by identity, which cannot see through
    # a wrapper), so it is excluded here rather than wrapped.
    #
    # Requires an eval provider (`BuiltinExecutor` or
    # `RestrictedPythonExecutor`) for the `parse`/`compile`/`eval` operations.

    @typing.final
    class _ExpressionToolCallTool[T](Tool[[CallExpression], T]):
        """A synthetic wrapper calling one lexically scoped tool through a
        model-written Python expression.

        A distinct type only so `ExpressionToolCaller` can recognize its own
        wrappers among the tools in a request; the capability is described to
        the model by the handler's docstring.
        """

        @classmethod
        def define(  # type: ignore[override]
            cls, tool: Tool, path: str
        ) -> "Tool[[CallExpression], typing.Any]":
            """Construct the expression-calling wrapper for `tool`.

            ``path`` is the expression that names the tool from the Skill's
            scope (see `_tool_paths`) -- ``self.retrieve`` for a method tool of
            the Skill's own agent, bare ``story_funny`` for a direct binding.
            Advertising it is what spares the model a wasted round trip
            discovering, from a type-check rejection, which form this
            particular tool needs.

            The wrapper's ``call`` parameter is annotated with a per-tool
            `CallExpression` subtype, which is how the decoder knows which tool
            the expression must invoke; by the time the wrapper runs, the
            decoded value already holds the evaluated arguments, and applying
            the tool is all that is left.
            """
            assert isinstance(tool, Tool)
            name = tool.__name__
            call_type: type[CallExpression] = type(
                f"_CallExpressionTo_{name}", (CallExpression,), {}
            )
            call_type.__expected_tool__ = tool  # see `CallExpression`

            def tool_fn(call: CallExpression) -> typing.Any:
                # Normally never reached: `ExpressionToolCaller._call_assistant`
                # substitutes the decoded `CallExpression` -- itself a
                # `DecodedToolCall` for the underlying tool -- for the wrapper's
                # call, so `call_tool` applies the tool directly. Kept
                # functional so the wrapper is a complete tool on its own
                # (e.g. offered by a stack without the substitution).
                return tool(*call.bound_args.args, **call.bound_args.kwargs)

            tool_fn.__name__ = name
            tool_fn.__qualname__ = getattr(tool, "__qualname__", name)
            tool_fn.__module__ = tool.__module__
            # Everything the JSON advertisement would have said about the tool
            # -- signature, docstring, parameter and return type schemas (with a
            # textual fallback for types, e.g. TypeVar-parameterized ones, that
            # have no schema) -- plus the instruction for the `call` parameter.
            tool_fn.__doc__ = (
                f"Call the tool `{name}` by writing Python code.\n\n"
                f"Provide `call`: a string containing a SINGLE Python "
                f"expression that invokes `{path}(...)` -- use exactly that "
                f"reference, which is how the tool is named in this Skill's "
                f"lexical scope. The expression is type-checked and evaluated "
                f"in this Skill's lexical scope; its value becomes the tool "
                f"result. See the `{ExpressionToolCaller.__name__}` section "
                f"of the system prompt.\n\n"
                f"The tool being called:\n\n"
                f"{_tool_description(tool, param_schemas=True)}"
            )
            tool_fn.__annotations__ = {"call": call_type, "return": typing.Any}
            return super().define(tool_fn)

    @classmethod
    def _should_wrap(cls, tool: Tool) -> bool:
        """Whether `tool` is offered through the expression pathway.

        Always, here: this handler is the uniform code-calling mode. The
        `MixedToolCaller` subclass overrides this with `_json_advertisable`.
        """
        return True

    @implements(call_assistant)
    def _call_assistant[T](
        self,
        messages: collections.abc.Sequence[Message],
        response_type: type[T],
        env: collections.abc.Mapping[str, typing.Any],
        tools: collections.abc.Set[Tool] = frozenset(),
    ) -> AssistantResult[T]:
        # Offer the lexical tools only, wrapped per `_should_wrap`; tools
        # arriving via `tools` are other handlers' own (REPL access, synthesis,
        # readers) and pass through unwrapped, as does the anchor Skill (see
        # the class comment).
        anchor = env.get(_TYPE_CHECK_ANCHOR_KEY)
        offered = {
            self._ExpressionToolCallTool.define(t, path) if self._should_wrap(t) else t
            for t, path in _tool_paths(env).items()
            if t is not anchor
        }
        message, tool_calls, result = fwd(messages, response_type, env, tools | offered)
        # Substitute each wrapper call with the `CallExpression` it decoded --
        # itself a `DecodedToolCall`, for the *underlying* tool bound to real
        # argument values -- stamped with the outer call's id/name, so every
        # `call_tool` handler (the default rule, retryers, tracers) sees the
        # actual call rather than the wrapper's. This handler is innermost, so
        # the substitution happens before any of them.
        tool_calls = [
            dataclasses.replace(tc.bound_args.arguments["call"], id=tc.id, name=tc.name)
            if isinstance(tc.tool, self._ExpressionToolCallTool)
            else tc
            for tc in tool_calls
        ]
        return message, tool_calls, result


class MixedToolCaller(ExpressionToolCaller):
    """Most tools of this Skill's lexical scope are ordinary tools: call them
    by name with JSON arguments matching their schema. Tools whose signatures
    a JSON schema cannot capture -- generic (type-variable) parameters,
    variadic `*args`/`**kwargs`, or parameter types with no JSON encoding --
    are instead called by writing Python: such a tool takes a single `call`
    parameter, a string containing ONE Python expression that invokes it, and
    its description states the exact reference to use (for example
    `self.retrieve(...)`).

    A `call` expression is type-checked and then evaluated in this Skill's
    lexical scope, so its arguments may be arbitrary Python expressions over
    the names in the *Lexical scope* table and any bindings made in the REPL
    session (if one is available). It must be a single call to the advertised
    tool -- no statements, no assignments, and no `:=` bindings (bind names
    with `exec_code` first if you need to). Either way, the call's return
    value becomes the tool result.
    """

    # The docstring above is model-facing (see `ExpressionToolCaller`).
    #
    # The default lexical tool caller: schema-constrained JSON arguments where
    # a schema can describe the tool faithfully (`_should_wrap` is False), the
    # expression pathway where it cannot. Everything but the partition
    # predicate -- wrapping, decode, `CallExpression` substitution -- is
    # inherited. A model that prefers writing code can still call any JSON
    # tool from the REPL (`exec_code`) when that handler is installed.

    @classmethod
    def _should_wrap(cls, tool: Tool) -> bool:
        """Wrap `tool` unless the JSON pathway can offer it unambiguously.

        Three conditions gate the JSON pathway, each a way its advertisement
        would misrepresent the tool rather than merely fail:

        * **No type variables** -- in the signature's annotations
          (`freetypevars`) or as PEP 695 type parameters on the underlying
          callable. A generic tool's parameters degrade to untyped ``{}``
          schemas and its JSON arguments cannot be decoded to concrete values
          (issue #489).
        * **No variadic parameters** -- ``*args``/``**kwargs`` have no
          faithful JSON parameter schema, and JSON arguments cannot be bound
          to them.
        * **The advertisement encodes** -- the same probe
          `~effectful.handlers.llm.harness.legibility.lexical.LexicalToolExtractor`
          applies: a parameter type with no `Encodable` schema at all.

        A tool that fails any of these is wrapped, and remains fully callable
        through the expression pathway.
        """
        signature = tool.__signature__
        if any(
            p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for p in signature.parameters.values()
        ):
            return True
        if getattr(inspect.unwrap(tool), "__type_params__", ()):
            return True
        annotations = [p.annotation for p in signature.parameters.values()]
        annotations.append(signature.return_annotation)
        if any(freetypevars(ann) for ann in annotations):
            return True
        try:
            probe: pydantic.TypeAdapter[_NameAndTool] = pydantic.TypeAdapter(
                Encodable[_NameAndTool]
            )
            probe.dump_python(_NameAndTool(tool.__name__, tool), mode="json")
        except Exception:
            return True
        return False
