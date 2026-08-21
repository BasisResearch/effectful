"""Tests for expression-based tool calling (`synthesis/toolcall.py`, issues #489/#505).

This module is separate to avoid lexical context pollution from other skills;
the anchor module -- with polymorphic, variadic, higher-order and Agent-method
tools in scope -- is imported from a real file per test so the splice-based
type check can recover its source.
"""

import ast
import contextlib
import importlib.util
import json
import sys
from collections.abc import Callable
from typing import Any

import pydantic
import pytest

from effectful.handlers.llm import Tool
from effectful.handlers.llm.harness.durability.transaction import HistoryBuilder
from effectful.handlers.llm.harness.execution.builtin import BuiltinExecutor
from effectful.handlers.llm.harness.execution.hooks import compile as compile_op
from effectful.handlers.llm.harness.execution.hooks import eval as eval_op
from effectful.handlers.llm.harness.execution.hooks import parse as parse_op
from effectful.handlers.llm.harness.execution.restricted import RestrictedPythonExecutor
from effectful.handlers.llm.harness.hooks import (
    AgentLoop,
    ToolCallDecodingError,
    ToolCallExecutionError,
    call_assistant,
    call_tool,
)
from effectful.handlers.llm.harness.legibility.lexical import LexicalToolExtractor
from effectful.handlers.llm.harness.provision.litellm import LiteLLMConfigurer
from effectful.handlers.llm.harness.serialization import (
    DecodedToolCall,
    _NameAndTool,
    _serialize_name_and_tool,
)
from effectful.handlers.llm.harness.synthesis.snippet import StatefulReplSynthesizer
from effectful.handlers.llm.harness.synthesis.toolcall import (
    CallExpression,
    ExpressionToolCaller,
    MixedToolCaller,
)
from effectful.handlers.llm.harness.validation.mypy import MypyTypeChecker
from effectful.handlers.llm.harness.validation.ty import TyTypeChecker
from effectful.handlers.llm.types import Encodable
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements

from .conftest import (
    EFFECTFUL_LLM_MODEL,
    MockCompletionHandler,
    make_text_response,
    make_tool_call_response,
    requires_llm,
)

# Every contract asserted below holds under either type checker (both resolve
# a tool's real signature now that `Tool.define` binds its type parameters
# from the wrapped function), so the module runs once per handler.
TYPE_CHECKER: Callable[[], Any] = MypyTypeChecker


@pytest.fixture(
    params=[MypyTypeChecker, TyTypeChecker], ids=["mypy", "ty"], autouse=True
)
def type_checker(request, monkeypatch):
    monkeypatch.setitem(globals(), "TYPE_CHECKER", request.param)
    return request.param


# A tool in this module's scope, for decoding `CallExpression`s directly
# (outside any Skill call / wrapper).
@Tool.define
def dbl(x: int) -> int:
    """Double x."""
    return x * 2


# ============================================================================
# Fixture module: polymorphic, variadic, higher-order and Agent-method tools,
# and a Skill with all of them in lexical scope. Imported from a real file so
# `_recover_skill_def` can splice into its source.
# ============================================================================

_POLY_SRC = '''
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from effectful.handlers.llm import Agent, Skill, Tool
from effectful.ops.types import NotHandled, Operation

calls = []


@Tool.define
def extend_sequence[T](examples: Sequence[T], new_example: T) -> Sequence[T]:
    """Extends the input sequence with a new example."""
    calls.append(("extend_sequence", list(examples), new_example))
    return [*examples, new_example]


@Tool.define
def other_tool() -> int:
    """Another tool."""
    return 7


@Tool.define
def handled_tool(x: int) -> int:
    """A tool implemented by an installed handler."""
    raise NotHandled


@Tool.define
def op_tool(op: Operation) -> str:
    """A tool whose parameter type has no JSON advertisement at all."""
    return op.__name__


@Tool.define
def vsum(*xs: int) -> int:
    """Sum any number of integers."""
    calls.append(("vsum", xs))
    return sum(xs)


@Tool.define
def fmt(**parts: str) -> str:
    """Join keyword parts as k=v, sorted by key."""
    calls.append(("fmt", dict(parts)))
    return ",".join(f"{k}={v}" for k, v in sorted(parts.items()))


@Tool.define
def apply_twice(f: Callable[[int], int], x: int) -> int:
    """Apply f to x twice."""
    calls.append(("apply_twice", x))
    return f(f(x))


@Tool.define
def make_adder(n: int) -> Callable[[int], int]:
    """Return a function that adds n."""
    def adder(x: int) -> int:
        return x + n
    return adder


@dataclass
class Counter(Agent):
    """A counter with tool methods."""

    n: int

    @Tool.define
    def bump(self, k: int) -> int:
        """Increase the count by k and return it."""
        self.n += k
        return self.n

    @Tool.define
    def reset(self) -> typing.Self:
        """Reset the count to zero and return this counter."""
        self.n = 0
        return self


counter = Counter(n=10)


@Skill.define
def grow(examples: Sequence[int]) -> str:
    """Use the available tools to grow {examples}, then summarize."""
'''


def _import_fixture(tmp_path, source: str, modname: str):
    p = tmp_path / f"{modname}.py"
    p.write_text(source)
    spec = importlib.util.spec_from_file_location(modname, str(p))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def poly_mod(tmp_path, request):
    modname = f"_toolcall_fixture_{request.node.name}".replace("[", "_").replace(
        "]", ""
    )
    mod = _import_fixture(tmp_path, _POLY_SRC, modname)
    yield mod
    sys.modules.pop(modname, None)


def _run_scripted(
    mod, responses, *extra_handlers, repl: bool = False, caller=ExpressionToolCaller
):
    """Run ``mod.grow([1, 2, 3])`` against a scripted model.

    ``extra_handlers`` are installed *first* -- innermost, so a capture handler
    for `call_assistant` sees the tool set as the default rule receives it,
    after every later-installed handler (which intercepts earlier) has
    contributed its tools.
    """
    mock = MockCompletionHandler(responses)
    stack = [
        *(handler(h) for h in extra_handlers),
        handler(AgentLoop()),
        handler(caller()),
        handler(LiteLLMConfigurer(model="test-model")),
        handler(HistoryBuilder()),
        handler(TYPE_CHECKER()),
        handler(BuiltinExecutor()),
    ]
    if repl:
        stack.append(handler(StatefulReplSynthesizer()))
    stack.append(handler(mock))

    with contextlib.ExitStack() as es:
        for h in stack:
            es.enter_context(h)
        return mod.grow([1, 2, 3])


def _call(expr: str, tool_name: str = "extend_sequence"):
    return make_tool_call_response(tool_name, json.dumps({"call": expr}))


# ============================================================================
# The `eval` operation
# ============================================================================


def _eval_expr(provider, source: str, env: dict[str, Any]) -> Any:
    with handler(provider):
        module = parse_op(source, "<test-eval>")
        code = compile_op(ast.Expression(module.body[0].value), "<test-eval>", "eval")
        return eval_op(code, env)


@pytest.mark.parametrize(
    "provider",
    [BuiltinExecutor, RestrictedPythonExecutor],
    ids=["builtin", "restricted"],
)
def test_eval_returns_value(provider):
    assert _eval_expr(provider(), "sum(xs) + y", {"xs": [1, 2, 3], "y": 10}) == 16


def test_eval_restricted_blocks_escape():
    with pytest.raises(Exception):
        _eval_expr(RestrictedPythonExecutor(), '__import__("os").system("true")', {})
    with pytest.raises(Exception):
        _eval_expr(RestrictedPythonExecutor(), "().__class__.__mro__", {})


@pytest.mark.parametrize(
    "provider",
    [BuiltinExecutor, RestrictedPythonExecutor],
    ids=["builtin", "restricted"],
)
def test_eval_binding_effects_discarded(provider):
    # The op-level contract: even a walrus that evaluates fine leaves no
    # binding in the caller's env. (The decoder rejects walrus before this
    # point; the contract holds for direct op callers regardless.)
    env: dict[str, Any] = {"x": 1}
    assert _eval_expr(provider(), "(y := x + 1)", env) == 2
    assert "y" not in env


# ============================================================================
# `CallExpression` decoding: shape validation and decode-time evaluation,
# outside any Skill call (no type-check anchor)
# ============================================================================


def _decode(source: str, **scope: Any) -> CallExpression:
    adapter: pydantic.TypeAdapter[CallExpression] = pydantic.TypeAdapter(
        Encodable[CallExpression]
    )
    with handler(BuiltinExecutor()):
        return adapter.validate_python(source, context={"dbl": dbl, **scope})


def test_decode_valid_call_roundtrip():
    expr = _decode("dbl(4)")
    assert isinstance(expr, CallExpression)
    # A decoded expression IS a decoded tool call for the underlying tool.
    assert isinstance(expr, DecodedToolCall)
    # Decoding resolved the callee and evaluated the arguments; nothing ran.
    assert expr.tool is dbl
    assert expr.bound_args.args == (4,)
    assert expr.source == "dbl(4)"
    # The serializer emits the original source.
    adapter: pydantic.TypeAdapter[CallExpression] = pydantic.TypeAdapter(
        Encodable[CallExpression]
    )
    assert adapter.dump_python(expr, mode="json") == "dbl(4)"


def test_decode_evaluates_argument_expressions():
    expr = _decode("dbl(sum(xs) + 1)", xs=[1, 2, 3])
    assert expr.bound_args.args == (7,)


@pytest.mark.parametrize(
    "source",
    [
        "x = 1",  # a statement
        "dbl(1)\ndbl(2)",  # more than one expression
        "xs",  # not a call
        "dbl(1) +",  # syntax error
        "dbl((y := 1))",  # top-level walrus
        "dbl([(y := v) for v in xs][0])",  # comprehension walrus escapes (PEP 572)
        "dbl(x=(y := 1))",  # walrus in a keyword argument
        "max(1, 2)",  # callee is not a Tool
        "missing(1)",  # callee not in scope
        "dbl(1 // 0)",  # argument raises during decode-time evaluation
        "dbl(1, 2)",  # arity mismatch caught by signature binding
    ],
)
def test_decode_rejects(source):
    with pytest.raises((pydantic.ValidationError, SyntaxError)):
        _decode(source, xs=[1, 2, 3])


def test_decode_accepts_lambda_walrus():
    # A lambda-local walrus binds in the lambda's own scope and is harmless.
    expr = _decode("dbl((lambda: (z := 5))())")
    assert expr.bound_args.args == (5,)


def test_decode_rejects_lambda_default_walrus():
    # ...but a walrus in a lambda's parameter *default* evaluates in the
    # enclosing scope and stays rejected.
    with pytest.raises(pydantic.ValidationError):
        _decode("dbl((lambda a=(z := 5): a)())")


# ============================================================================
# Splice-based type checking against the anchor Skill (the core of #489)
# ============================================================================


def test_generic_tool_call_well_typed(poly_mod):
    result = _run_scripted(
        poly_mod,
        [_call("extend_sequence(examples, 42)"), make_text_response("done")],
    )
    assert result == "done"
    assert poly_mod.calls == [("extend_sequence", [1, 2, 3], 42)]


def test_generic_tool_call_ill_typed_rejected(poly_mod):
    # `5` is not a `Sequence`: the spliced type check resolves the tool's real
    # generic signature (both checkers do, since `Tool.define` binds the
    # result's type parameters from the wrapped function) and rejects the call
    # at decode time -- the retryable kind `TenacityRetryer` feeds back to the
    # model. The ill-typed call never evaluates.
    with pytest.raises(ToolCallDecodingError):
        _run_scripted(
            poly_mod,
            [_call("extend_sequence(5, 6)"), make_text_response("done")],
        )
    assert poly_mod.calls == []


# ============================================================================
# Tool shapes: variadic, splatted, higher-order, Agent-bound methods
# ============================================================================


def test_variadic_positional_tool(poly_mod):
    result = _run_scripted(
        poly_mod, [_call("vsum(1, 2, 3)", "vsum"), make_text_response("done")]
    )
    assert result == "done"
    assert poly_mod.calls == [("vsum", (1, 2, 3))]


def test_variadic_positional_splat(poly_mod):
    _run_scripted(
        poly_mod, [_call("vsum(*examples)", "vsum"), make_text_response("done")]
    )
    assert poly_mod.calls == [("vsum", (1, 2, 3))]


def test_variadic_keyword_tool(poly_mod):
    _run_scripted(
        poly_mod, [_call('fmt(a="1", b="2")', "fmt"), make_text_response("done")]
    )
    assert poly_mod.calls == [("fmt", {"a": "1", "b": "2"})]


def test_variadic_keyword_splat(poly_mod):
    _run_scripted(
        poly_mod,
        [_call('fmt(**{"a": "x"}, b="y")', "fmt"), make_text_response("done")],
    )
    assert poly_mod.calls == [("fmt", {"a": "x", "b": "y"})]


def test_higher_order_argument(poly_mod):
    # A lambda argument is evaluated at decode time into a real function.
    _run_scripted(
        poly_mod,
        [
            _call("apply_twice(lambda v: v + 1, 3)", "apply_twice"),
            make_text_response("done"),
        ],
    )
    assert poly_mod.calls == [("apply_twice", 3)]


def test_higher_order_result(poly_mod):
    # A tool *returning* a callable: the result is encoded (as source) into the
    # tool message and the loop completes.
    result = _run_scripted(
        poly_mod, [_call("make_adder(10)", "make_adder"), make_text_response("done")]
    )
    assert result == "done"


def test_bound_method_tool(poly_mod):
    # A Tool method reached through an in-scope Agent instance: the expression
    # names it as an attribute access, and the head check resolves to the same
    # cached instance-bound operation `_tools_in_scope` advertised.
    result = _run_scripted(
        poly_mod, [_call("counter.bump(5)", "bump"), make_text_response("done")]
    )
    assert result == "done"
    assert poly_mod.counter.n == 15


def test_bound_method_self_returning(poly_mod):
    # A method tool annotated `-> typing.Self` returns its Agent, which is
    # encoded into the tool message.
    result = _run_scripted(
        poly_mod, [_call("counter.reset()", "reset"), make_text_response("done")]
    )
    assert result == "done"
    assert poly_mod.counter.n == 0


# ============================================================================
# The `ExpressionToolCaller` handler
# ============================================================================


def test_head_check_rejects_other_tool(poly_mod):
    # The expression is decoded under `extend_sequence`'s wrapper but calls a
    # different tool; the decoder refuses it before anything is applied.
    with pytest.raises(ToolCallDecodingError, match="must be a call"):
        _run_scripted(
            poly_mod,
            [_call("other_tool()", "extend_sequence"), make_text_response("done")],
        )


def test_expression_call_dispatches_through_handler(poly_mod):
    # Calling the tool in an expression goes through the Operation, so an
    # installed interpretation for it intercepts as usual.
    seen = []

    class _Impl(ObjectInterpretation):
        @implements(poly_mod.handled_tool)
        def _handled(self, x: int) -> int:
            seen.append(x)
            return x * 10

    result = _run_scripted(
        poly_mod,
        [_call("handled_tool(4)", "handled_tool"), make_text_response("done")],
        _Impl(),
    )
    assert result == "done"
    assert seen == [4]


def test_call_tool_receives_underlying_call(poly_mod):
    # The handler substitutes the decoded `CallExpression` -- a
    # `DecodedToolCall` for the *underlying* tool with evaluated arguments and
    # the outer call's id/name -- for the wrapper's call, so every `call_tool`
    # handler sees the real call.
    seen: list = []

    class _Spy(ObjectInterpretation):
        @implements(call_tool)
        def _ct(self, tc):
            seen.append(tc)
            return fwd(tc)

    _run_scripted(
        poly_mod,
        [_call("extend_sequence(examples, 42)"), make_text_response("done")],
        _Spy(),
    )
    (tc,) = seen
    assert isinstance(tc, CallExpression) and isinstance(tc, DecodedToolCall)
    assert tc.tool is poly_mod.extend_sequence
    assert dict(tc.bound_args.arguments) == {"examples": [1, 2, 3], "new_example": 42}
    assert tc.id == "call_1"  # the outer raw call's id, stamped on
    assert tc.name == "extend_sequence"
    assert tc.source == "extend_sequence(examples, 42)"


def test_call_expression_wire_roundtrip(poly_mod):
    # Serializing the substituted call through `Encodable[DecodedToolCall]`
    # emits what the model actually sent -- `{"call": <source>}` under the
    # advertised name -- not a re-encoding of the evaluated argument values
    # (which need not be encodable at all).
    seen: list = []

    class _Spy(ObjectInterpretation):
        @implements(call_tool)
        def _ct(self, tc):
            seen.append(tc)
            return fwd(tc)

    _run_scripted(
        poly_mod,
        [_call("extend_sequence(examples, 42)"), make_text_response("done")],
        _Spy(),
    )
    adapter: pydantic.TypeAdapter[DecodedToolCall] = pydantic.TypeAdapter(
        Encodable[DecodedToolCall]
    )
    wire = adapter.dump_python(seen[0], mode="json")
    assert wire["function"]["name"] == "extend_sequence"
    assert json.loads(wire["function"]["arguments"]) == {
        "call": "extend_sequence(examples, 42)"
    }
    assert wire["id"] == "call_1"


def test_tool_body_error_is_execution_error(poly_mod):
    # Decoding evaluates everything but the call itself, so an error raised by
    # the *tool body* still surfaces at `call_tool` as an execution error.
    class _Impl(ObjectInterpretation):
        @implements(poly_mod.handled_tool)
        def _handled(self, x: int) -> int:
            raise RuntimeError("boom")

    with pytest.raises(ToolCallExecutionError, match="boom"):
        _run_scripted(
            poly_mod,
            [_call("handled_tool(4)", "handled_tool"), make_text_response("done")],
            _Impl(),
        )


def test_anchor_skill_not_offered(poly_mod):
    # The anchor Skill must not be offered (wrapped or otherwise) to itself.
    captured: set = set()

    class _Capture(ObjectInterpretation):
        @implements(call_assistant)
        def _ca(self, messages, response_type, env, tools=frozenset()):
            captured.update(tools)
            return fwd(messages, response_type, env, tools)

    _run_scripted(poly_mod, [make_text_response("done")], _Capture())
    names = {t.__name__ for t in captured}
    assert {"extend_sequence", "other_tool", "vsum", "fmt", "bump", "reset"} <= names
    # A tool that has no JSON advertisement at all is still callable by code.
    assert "op_tool" in names
    assert "grow" not in names
    assert all(
        isinstance(t, ExpressionToolCaller._ExpressionToolCallTool) for t in captured
    )


def test_505_generic_tool_in_scope_does_not_break_unrelated_skill(poly_mod):
    # The scenario of issue #505: a generic tool merely in scope must not
    # break an unrelated skill call (it used to crash the request outright;
    # under the expression pathway it is advertised as a code-called wrapper).
    result = _run_scripted(poly_mod, [make_text_response("just text")])
    assert result == "just text"


def test_json_mode_skips_unadvertisable_tool(poly_mod):
    # Under the JSON pathway a tool whose advertisement cannot be encoded is
    # skipped (with a warning) instead of breaking every request it is merely
    # in scope for. (A *generic* tool still advertises there, but degraded to
    # untyped `{}` parameter schemas -- the #489 decode ambiguity the
    # expression pathway exists to fix.)
    captured: set = set()

    class _Capture(ObjectInterpretation):
        @implements(call_assistant)
        def _ca(self, messages, response_type, env, tools=frozenset()):
            captured.update(tools)
            return fwd(messages, response_type, env, tools)

    mock = MockCompletionHandler([make_text_response("just text")])
    with (
        # `_Capture` first: innermost, so it sees the tool set after the
        # later-installed extractor has contributed (and skipped) its tools.
        handler(_Capture()),
        handler(AgentLoop()),
        handler(LexicalToolExtractor()),
        handler(LiteLLMConfigurer(model="test-model")),
        handler(HistoryBuilder()),
        handler(mock),
    ):
        assert poly_mod.grow([1, 2, 3]) == "just text"
    names = {t.__name__ for t in captured}
    assert "other_tool" in names and "extend_sequence" in names
    assert "op_tool" not in names


# ============================================================================
# Advertisement parity: the wrapper's description carries what the JSON
# `parameters` schema would have
# ============================================================================


def test_wrapper_description_parity(poly_mod):
    wrapper = ExpressionToolCaller._ExpressionToolCallTool.define(
        poly_mod.extend_sequence, "extend_sequence"
    )
    spec = _serialize_name_and_tool(_NameAndTool("extend_sequence", wrapper))
    desc = spec["function"]["description"]
    # The wrapped tool's docstring and signature.
    assert "Extends the input sequence with a new example." in desc
    assert "extend_sequence" in desc
    # TypeVar-carrying parameter types fall back to their textual annotation.
    assert "Annotated JSON schema of each parameter type" in desc
    assert "Sequence[" in desc

    # A monomorphic tool's parameter schema is a real JSON schema.
    wrapper2 = ExpressionToolCaller._ExpressionToolCallTool.define(
        poly_mod.handled_tool, "handled_tool"
    )
    spec2 = _serialize_name_and_tool(_NameAndTool("handled_tool", wrapper2))
    desc2 = spec2["function"]["description"]
    assert '"integer"' in desc2  # both the parameter and return schemas


def test_tool_paths(poly_mod):
    # `_tool_paths` names each tool by the expression that reaches it: a bare
    # name for a direct binding, an attribute access for an Agent-held tool --
    # and a bound receiver takes precedence over an outer alias.
    from effectful.handlers.llm.harness.legibility.lexical import _tool_paths

    env = {"extend_sequence": poly_mod.extend_sequence, "counter": poly_mod.counter}
    paths = _tool_paths(env)
    assert paths[poly_mod.extend_sequence] == "extend_sequence"
    assert paths[poly_mod.counter.bump] == "counter.bump"
    assert paths[poly_mod.counter.reset] == "counter.reset"

    # The same agent bound earlier under `self` (as a method Skill's env lists
    # its bound arguments first) wins the path.
    paths = dict(_tool_paths({"self": poly_mod.counter, **env}))
    assert paths[poly_mod.counter.bump] == "self.bump"


def test_wrapper_advertises_reference_path(poly_mod):
    # The wrapper's description tells the model the exact reference to call the
    # tool by -- `counter.bump(...)` for an Agent-held tool -- so it need not
    # discover the bare-vs-attribute distinction from a type-check rejection.
    wrapper = ExpressionToolCaller._ExpressionToolCallTool.define(
        poly_mod.counter.bump, "counter.bump"
    )
    spec = _serialize_name_and_tool(_NameAndTool("bump", wrapper))
    assert "`counter.bump(...)`" in spec["function"]["description"]


# ============================================================================
# `MixedToolCaller`: JSON where a schema can describe the tool, code elsewhere
# ============================================================================


def test_mixed_should_wrap(poly_mod):
    # Schema-describable tools -- including higher-order ones (the JSON pathway
    # has a synthesis encoding for `Callable`) and a no-argument `-> Self`
    # method (runtime-encoded return) -- are JSON-eligible; generic, variadic
    # and unadvertisable ones are wrapped.
    should_wrap = MixedToolCaller._should_wrap
    assert not should_wrap(poly_mod.other_tool)
    assert not should_wrap(poly_mod.handled_tool)
    assert not should_wrap(poly_mod.apply_twice)
    assert not should_wrap(poly_mod.make_adder)
    assert not should_wrap(poly_mod.counter.bump)
    assert not should_wrap(poly_mod.counter.reset)
    assert should_wrap(poly_mod.extend_sequence)  # TypeVars
    assert should_wrap(poly_mod.vsum)  # *args
    assert should_wrap(poly_mod.fmt)  # **kwargs
    assert should_wrap(poly_mod.op_tool)  # no schema at all


def test_mixed_partition(poly_mod):
    # Under `MixedToolCaller`, schema-describable tools are offered raw (JSON
    # arguments) and the rest as expression wrappers.
    captured: set = set()

    class _Capture(ObjectInterpretation):
        @implements(call_assistant)
        def _ca(self, messages, response_type, env, tools=frozenset()):
            captured.update(tools)
            return fwd(messages, response_type, env, tools)

    _run_scripted(
        poly_mod, [make_text_response("done")], _Capture(), caller=MixedToolCaller
    )
    wrapped = {
        t.__name__
        for t in captured
        if isinstance(t, ExpressionToolCaller._ExpressionToolCallTool)
    }
    raw = {t.__name__ for t in captured} - wrapped
    assert {"extend_sequence", "vsum", "fmt", "op_tool"} <= wrapped
    assert {"other_tool", "handled_tool", "apply_twice", "make_adder", "bump"} <= raw
    assert "grow" not in wrapped | raw  # the anchor is still excluded


def test_mixed_both_pathways_in_one_run(poly_mod):
    # A code-expression call and a JSON-argument call served by the same
    # handler in one conversation.
    seen = []

    class _Impl(ObjectInterpretation):
        @implements(poly_mod.handled_tool)
        def _handled(self, x: int) -> int:
            seen.append(x)
            return x * 10

    result = _run_scripted(
        poly_mod,
        [
            make_tool_call_response("vsum", json.dumps({"call": "vsum(*examples)"})),
            make_tool_call_response("handled_tool", json.dumps({"x": 4})),
            make_text_response("done"),
        ],
        _Impl(),
        caller=MixedToolCaller,
    )
    assert result == "done"
    assert poly_mod.calls == [("vsum", (1, 2, 3))]
    assert seen == [4]


# ============================================================================
# REPL interplay
# ============================================================================


def test_expression_uses_repl_binding(poly_mod):
    # A snippet binds `n` in the session; the expression references it. The
    # splice-based type check sees the binding (via `repl_history`) and the
    # evaluation sees the value (via `repl_env`).
    result = _run_scripted(
        poly_mod,
        [
            make_tool_call_response("exec_code", json.dumps({"code": "n = 4"})),
            _call("extend_sequence(examples, n)"),
            make_text_response("done"),
        ],
        repl=True,
    )
    assert result == "done"
    assert poly_mod.calls == [("extend_sequence", [1, 2, 3], 4)]


def test_expression_works_without_repl(poly_mod):
    # No `StatefulReplSynthesizer` installed: `repl_history`/`repl_env` fall
    # back to their defaults and the expression evaluates against the Skill
    # env alone.
    result = _run_scripted(
        poly_mod,
        [
            _call("extend_sequence(list(examples), len(examples))"),
            make_text_response("ok"),
        ],
    )
    assert result == "ok"
    assert poly_mod.calls == [("extend_sequence", [1, 2, 3], 3)]


# ============================================================================
# Generic skills (issue #489, skill side): the response schema is built from
# the return annotation, instantiated through `type[T]`-annotated arguments
# ============================================================================

_GENERIC_SKILL_SRC = '''
import dataclasses
from collections.abc import Callable, Sequence

from effectful.handlers.llm import Skill


@dataclasses.dataclass(eq=False)
class Box[T]:
    value: T


@dataclasses.dataclass(frozen=True)
class Item:
    text: str


@Skill.define
def refine[T](box: Box[T]) -> T:
    """Return an improved version of the value in {box}."""


@Skill.define
def refine_as[T](box: Box[T], typ: type[T]) -> T:
    """Return an improved version of the value in {box}."""


@Skill.define
def pick[T](items: Sequence[T]) -> T:
    """Pick the best of {items}."""


@Skill.define
def vpick[T](*options: T) -> T:
    """Pick the best of {options}."""


@Skill.define
def kpick[T](**named: T) -> T:
    """Pick the best of {named}."""


@Skill.define
def make_fn[T](typ: type[T]) -> Callable[[T], T]:
    """Write a function transforming a value of {typ.__name__}."""
'''


@pytest.fixture
def generic_mod(tmp_path, request):
    modname = f"_generic_fixture_{request.node.name}".replace("[", "_").replace("]", "")
    mod = _import_fixture(tmp_path, _GENERIC_SKILL_SRC, modname)
    yield mod
    sys.modules.pop(modname, None)


def _run_generic(mod, skill_call, responses, *extra_handlers):
    # Unlike `_run_scripted`'s capture handlers, extras here (e.g.
    # `FinalBodySynthesizer`) implement `call_agent`, and `AgentLoop._call` is
    # terminal for it -- so they must install *after* `AgentLoop` to intercept
    # before it, as in the real `harness()` stack.
    mock = MockCompletionHandler(responses)
    stack = [
        handler(AgentLoop()),
        handler(LiteLLMConfigurer(model="test-model")),
        handler(HistoryBuilder()),
        *(handler(h) for h in extra_handlers),
        handler(TYPE_CHECKER()),
        handler(BuiltinExecutor()),
        handler(mock),
    ]
    with contextlib.ExitStack() as es:
        for h in stack:
            es.enter_context(h)
        return skill_call()


def test_generic_callable_skill_parametric_impl(generic_mod):
    # A skill with a `type[T]` parameter and a polymorphic Callable return
    # runs end-to-end (the `type[X]` prompt/spec encoding no longer crashes),
    # and a *parametric* implementation satisfies the splice type check.
    fn = _run_generic(
        generic_mod,
        lambda: generic_mod.make_fn(int),
        [
            make_text_response(
                json.dumps(
                    {
                        "value": {
                            "module_code": "def ident[U](x: U) -> U:\n    return x\n"
                        }
                    }
                )
            )
        ],
    )
    assert fn(21) == 21


def test_generic_callable_skill_concrete_impl_rejected(generic_mod):
    # A *concrete* implementation is rejected by both checkers: the anchor's
    # source signature promises a universally quantified `Callable[[T], T]`,
    # and a synthesized function is held to exactly what is written -- it can
    # (and must) be polymorphic itself, as in the test above. A skill whose
    # per-call answers are genuinely type-specific is overpromising in its
    # signature, not hitting a checker limitation.
    from effectful.handlers.llm.harness.hooks import ResultDecodingError

    with pytest.raises(ResultDecodingError):
        _run_generic(
            generic_mod,
            lambda: generic_mod.make_fn(int),
            [
                make_text_response(
                    json.dumps(
                        {
                            "value": {
                                "module_code": "def dbl(x: int) -> int:\n    return x + x\n"
                            }
                        }
                    )
                )
            ],
        )


def test_generic_skill_body_synthesis_parametricity(generic_mod):
    # Via `FinalBodySynthesizer`, a parametric body passes both checkers and
    # flows the runtime value out; a literal-returning body -- concrete where
    # `T` is expected -- is rejected. This is the checker being right about
    # parametricity, and it is why a value-level generic skill (textgrad's
    # `_accumulate`) cannot be answered by code today.
    from effectful.handlers.llm.harness.synthesis.body import FinalBodySynthesizer

    box = generic_mod.Box(value=[1, 2, 3])
    result = _run_generic(
        generic_mod,
        lambda: generic_mod.refine(box),
        [
            make_tool_call_response(
                "submit_solution",
                json.dumps(
                    {"implementation": "def refine(box):\n    return box.value\n"}
                ),
            )
        ],
        FinalBodySynthesizer(),
    )
    assert result == [1, 2, 3]

    with pytest.raises(ToolCallDecodingError):
        _run_generic(
            generic_mod,
            lambda: generic_mod.refine(box),
            [
                make_tool_call_response(
                    "submit_solution",
                    json.dumps(
                        {"implementation": "def refine(box):\n    return [9, 9]\n"}
                    ),
                )
            ],
            FinalBodySynthesizer(),
        )


def test_generic_skill_direct_output_decodes_instantiated(generic_mod):
    # A `-> T` skill answered by structured output decodes as T's
    # *instantiation*, bound through the `type[T]`-annotated argument -- the
    # caller literally passes the type, so no structural inference is
    # involved -- and the reply comes back as real typed values, here `Item`
    # instances rather than raw dicts. This is what lets textgrad's
    # `_accumulate` be written in decoded form.
    box = generic_mod.Box(value=[generic_mod.Item(text="be kind")])
    result = _run_generic(
        generic_mod,
        lambda: generic_mod.refine_as(box, typ=list[generic_mod.Item]),
        [
            make_text_response(
                json.dumps({"value": [{"text": "be kinder"}, {"text": "cite sources"}]})
            )
        ],
    )
    assert result == [
        generic_mod.Item(text="be kinder"),
        generic_mod.Item(text="cite sources"),
    ]
    assert all(isinstance(item, generic_mod.Item) for item in result)


def test_generic_skill_direct_output_validates_instantiated(generic_mod):
    # The instantiation is enforced: a reply that does not fit T's binding
    # fails decoding (the retryable kind).
    from effectful.handlers.llm.harness.hooks import ResultDecodingError

    box = generic_mod.Box(value=[generic_mod.Item(text="be kind")])
    with pytest.raises(ResultDecodingError):
        _run_generic(
            generic_mod,
            lambda: generic_mod.refine_as(box, typ=list[generic_mod.Item]),
            [make_text_response(json.dumps({"value": [{"wrong_field": 1}]}))],
        )


def test_generic_skill_collection_arg_infers_instantiation(generic_mod):
    # The best-effort second pass: type reconstruction from values is sound
    # (if incomplete) for collections, so `Sequence[T]` called with a list of
    # `Item`s binds `T = Item` with no explicit `type[T]` argument, and the
    # reply decodes typed.
    result = _run_generic(
        generic_mod,
        lambda: generic_mod.pick(
            [generic_mod.Item(text="a"), generic_mod.Item(text="b")]
        ),
        [make_text_response(json.dumps({"value": {"text": "a"}}))],
    )
    assert result == generic_mod.Item(text="a")


def test_unbound_type_param_response_schema_is_strict_legal(generic_mod):
    # A leftover free type parameter must not reach the provider as a
    # type-less schema node: OpenAI-style strict structured outputs reject
    # those with a `BadRequestError` no retry loop can fix. The fallback asks
    # for the value as a JSON-encoded string instead.
    import litellm

    from effectful.handlers.llm.harness.hooks import (
        _BoxedResponse,
        _instantiate_return_type,
    )

    box = generic_mod.Box(value=[generic_mod.Item(text="x")])
    bound = generic_mod.refine.__signature__.bind(box)
    bound.apply_defaults()
    response_type = _instantiate_return_type(generic_mod.refine, bound)
    rf = pydantic.create_model(
        "BoxedResponse", value=Encodable[response_type], __base__=_BoxedResponse
    )
    schema = litellm.utils.type_to_response_format_param(rf)["json_schema"]["schema"]

    def no_typeless(node):
        # The failure shapes strict mode rejects: `{}` (e.g. JsonValue's $def)
        # and `{"title": ...}`-only nodes (a bare TypeVar's rendering).
        if isinstance(node, dict):
            assert node and set(node) - {"title", "description"}, (
                f"type-less schema node: {node}"
            )
            for child in node.values():
                no_typeless(child)
        elif isinstance(node, list):
            for child in node:
                no_typeless(child)

    assert schema["properties"]["value"]["type"] == "string"
    no_typeless(schema)


def test_generic_skill_variadic_args_infer_instantiation(generic_mod):
    # A variadic parameter's annotation describes each *element*: `*options: T`
    # called with `Item`s binds `T = Item` (not the packed tuple type), and the
    # reply decodes typed. Same for `**named: T` over the dict's values.
    result = _run_generic(
        generic_mod,
        lambda: generic_mod.vpick(
            generic_mod.Item(text="a"), generic_mod.Item(text="b")
        ),
        [make_text_response(json.dumps({"value": {"text": "a"}}))],
    )
    assert result == generic_mod.Item(text="a")

    result = _run_generic(
        generic_mod,
        lambda: generic_mod.kpick(
            first=generic_mod.Item(text="a"), second=generic_mod.Item(text="b")
        ),
        [make_text_response(json.dumps({"value": {"text": "b"}}))],
    )
    assert result == generic_mod.Item(text="b")


def test_generic_skill_variadic_conflict_refuses_direct_reply(generic_mod):
    # Conflicting element types leave the parameter unbound (all-or-nothing:
    # never bound to whichever element unified first), which routes to the
    # refusal/redirect path rather than a wrong schema. A final-answer tool is
    # installed so the request is answerable at all (without one, the fast
    # failure below preempts the API call entirely).
    from effectful.handlers.llm.harness.synthesis.body import FinalBodySynthesizer

    with pytest.raises(TypeError):
        _run_generic(
            generic_mod,
            lambda: generic_mod.vpick(1, "a"),
            [make_text_response(json.dumps({"value": 1}))],
            FinalBodySynthesizer(),
        )


def test_generic_skill_without_binding_refuses_direct_reply(generic_mod):
    # The incompleteness boundary: `T` inside an arbitrary generic dataclass
    # (`Box[T]`) is beyond what value-driven inference can reconstruct, and a
    # direct reply for an uninstantiated return has no sound decoding (raw
    # JSON would hand the caller untyped data where the signature promised
    # `T`). So decoding refuses every direct reply, with feedback naming the
    # sound alternative. A final-answer tool is installed so the request is
    # answerable at all (without one, the fast failure below preempts the API
    # call entirely).
    from effectful.handlers.llm.harness.hooks import ResultDecodingError
    from effectful.handlers.llm.harness.synthesis.body import FinalBodySynthesizer

    box = generic_mod.Box(value=[generic_mod.Item(text="be kind")])
    with pytest.raises(ResultDecodingError, match="final answer"):
        _run_generic(
            generic_mod,
            lambda: generic_mod.refine(box),
            [make_text_response(json.dumps({"value": [{"text": "be kinder"}]}))],
            FinalBodySynthesizer(),
        )


def test_undecodable_return_with_no_tools_fails_before_api_call(generic_mod):
    # With no tools in the request, an undecodable response type makes every
    # round trip a guaranteed refusal: fail fast, before any API call, with a
    # developer-facing error the retry loop does not retry.
    from effectful.handlers.llm.harness.serialization import _UndecodableReturn

    mock = MockCompletionHandler([make_text_response("never sent")])
    with handler(mock), pytest.raises(TypeError, match="final-answer"):
        call_assistant([], _UndecodableReturn, {}, frozenset())
    assert mock.call_count == 0


def test_generic_skill_without_binding_redirects_to_code_mode(generic_mod):
    # The full redirect under the real retry loop: the model's direct reply is
    # refused, the feedback steers it to `submit_solution`, and the synthesized
    # (parametric) implementation's applied result is the answer.
    from effectful.handlers.llm.harness.durability.retrying import TenacityRetryer
    from effectful.handlers.llm.harness.synthesis.body import FinalBodySynthesizer

    box = generic_mod.Box(value=[generic_mod.Item(text="be kind")])
    result = _run_generic(
        generic_mod,
        lambda: generic_mod.refine(box),
        [
            make_text_response(json.dumps({"value": [{"text": "be kinder"}]})),
            make_tool_call_response(
                "submit_solution",
                json.dumps(
                    {"implementation": "def refine(box):\n    return box.value\n"}
                ),
            ),
        ],
        FinalBodySynthesizer(),
        TenacityRetryer(),
    )
    assert result == [generic_mod.Item(text="be kind")]


# ============================================================================
# Live model
# ============================================================================


@requires_llm
@pytest.mark.parametrize(
    "caller", [ExpressionToolCaller, MixedToolCaller], ids=["code", "mixed"]
)
def test_live_polymorphic_tool_call(poly_mod, caller):
    # A real model calls a polymorphic tool through the expression pathway
    # (issue #489's motivating case) -- under the uniform code mode and under
    # the mixed default, where that tool is one of the wrapped ones.
    from effectful.handlers.llm.harness.durability.retrying import TenacityRetryer

    stack = [
        handler(AgentLoop()),
        handler(caller()),
        handler(LiteLLMConfigurer(model=EFFECTFUL_LLM_MODEL)),
        handler(HistoryBuilder()),
        handler(TYPE_CHECKER()),
        handler(BuiltinExecutor()),
        handler(TenacityRetryer()),
    ]
    with contextlib.ExitStack() as es:
        for h in stack:
            es.enter_context(h)
        result = poly_mod.grow([1, 2, 3])
    assert isinstance(result, str)
    # The prompt directs the model to use the tools; at least one call went
    # through the expression pathway with correctly typed arguments.
    assert poly_mod.calls
