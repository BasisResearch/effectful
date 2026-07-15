"""Tests for effectful.handlers.llm.evaluation."""

import ast
import builtins
import contextlib
import importlib.util
import io
import sys
import types
from collections import ChainMap
from collections.abc import Callable
from typing import Any

import pydantic
import pytest
from RestrictedPython import RestrictingNodeTransformer

from effectful.handlers.llm.encoding import (
    TYPE_CHECK_ANCHOR_KEY,
    Encodable,
    SynthesizedFunction,
)
from effectful.handlers.llm.evaluation import (
    ReplSession,
    RestrictedEvalProvider,
    UnsafeEvalProvider,
    _splice_repl,
    scan_non_nestable,
    splice_into_source,
    type_check,
)
from effectful.handlers.llm.evaluation import compile as compile_op
from effectful.handlers.llm.evaluation import exec as exec_op
from effectful.handlers.llm.evaluation import parse as parse_op
from effectful.ops.semantics import handler

# ============================================================================
# Splice-based type checking (splice_into_source + type_check)
#
# The type checker splices LLM-generated code into the real source of the
# Template's module -- at the template function's own (possibly nested)
# position -- and runs mypy on the whole module, raising only on diagnostics
# inside the spliced region. These tests exercise that contract against real
# anchor functions.
# ============================================================================


# Module-level "Template" anchors. Their bodies are placeholders; the type
# checker replaces them with the generated code. ``count_a`` deliberately shares
# a name with the function the model is asked to synthesize (issue #542).
count_a: Callable[[str], int] = lambda s: 0  # noqa: E731


class _Ctx:
    x: int = 0


def _count_char(char: str) -> Callable[[str], int]:
    raise NotImplementedError


def _takes_ctx() -> Callable[[_Ctx], int]:
    raise NotImplementedError


def _loose() -> Callable[..., object]:
    raise NotImplementedError


def _make_counter(helper_const: int) -> Callable[[int], int]:
    def templ(a: int) -> Callable[[int], int]:
        raise NotImplementedError

    return templ  # type: ignore[return-value]


def _raises(generated_src: str, anchor: Any) -> bool:
    # Splice then type-check as separate steps (the decode path's shape), under a
    # provider so the `type_check` op resolves.
    try:
        with handler(UnsafeEvalProvider()):
            spliced = splice_into_source(ast.parse(generated_src), anchor)
            if spliced is not None:
                type_check(*spliced)
        return False
    except TypeError:
        return True


def _import_fixture(tmp_path, source: str, modname: str):
    p = tmp_path / f"{modname}.py"
    p.write_text(source)
    spec = importlib.util.spec_from_file_location(modname, str(p))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def _anchor_from_source(tmp_path, source: str, attr: str, modname: str) -> Any:
    return getattr(_import_fixture(tmp_path, source, modname), attr)


def test_good_function_typechecks():
    assert not _raises(
        "def count_a(s: str) -> int:\n    return s.count('a')\n", _count_char
    )


def test_wrong_return_type_raises():
    assert _raises("def count_a(s: str) -> str:\n    return s\n", _count_char)


def test_synthesized_name_collides_with_context_var_issue_542():
    # The synthesized ``count_a`` collides with the module-level ``count_a``
    # binding. Nested as a local it *shadows* it (no ``[no-redef]``), where the
    # old flat-stub mechanism raised and had to rename around the collision.
    assert not _raises(
        "def count_a(s: str) -> int:\n    return s.count('a')\n", _count_char
    )


def test_helpers_alongside_target_are_checked():
    src = (
        "def helper(y: int) -> int:\n    return y * 2\n"
        "def count_a(s: str) -> int:\n    return helper(len(s))\n"
    )
    assert not _raises(src, _count_char)


def test_closure_template_sees_enclosing_local():
    anchor = _make_counter(5)  # nested ``templ``, whose factory binds helper_const
    assert not _raises("def f(z: int) -> int:\n    return z + helper_const\n", anchor)


def test_unannotated_container_is_not_a_false_positive():
    # Heterogeneous list via ``append`` runs fine; with ``--check-untyped-defs``
    # off mypy skips the unannotated body, so it must not be rejected.
    src = "def cf(z):\n    acc = []\n    acc.append(1)\n    acc.append('x')\n    return acc\n"
    assert not _raises(src, _loose)


def test_star_import_raises():
    with pytest.raises(TypeError):
        scan_non_nestable(
            ast.parse("from os import *\ndef g(s: str) -> int:\n    return 0\n")
        )


def test_future_import_raises():
    with pytest.raises(TypeError):
        scan_non_nestable(
            ast.parse(
                "from __future__ import annotations\n"
                "def g(s: str) -> int:\n    return 0\n"
            )
        )


def test_annotation_collision_with_context_type_raises():
    # A helper named like the context type ``_Ctx`` and used as a body annotation
    # is shadowed -> rejected (fail-closed; matches runtime, where the helper
    # wins). The old rename pass rejected this identically; not a regression.
    src = "def _Ctx(q):\n    return q\ndef f(z: _Ctx) -> int:\n    return z.x\n"
    assert _raises(src, _takes_ctx)


def test_unrecoverable_source_skips():
    # An ``exec``-defined function has no recoverable source -> skip, never raise.
    ns: dict[str, Any] = {}
    exec("def t(c: str):\n    raise NotImplementedError", ns)
    assert not _raises(
        "def count_a(s: str) -> int:\n    return s.count('a')\n", ns["t"]
    )


def test_empty_module_raises():
    assert _raises("", _count_char)


def test_last_statement_not_function_raises():
    assert _raises("x = 1\n", _count_char)


def test_gate_unrelated_module_error_does_not_block(tmp_path):
    # The anchor's module contains an UNRELATED type error; a correct synthesis
    # must still pass, because region-scoping ignores out-of-region diagnostics.
    source = (
        "from collections.abc import Callable\n\n\n"
        "def unrelated() -> int:\n    return 'boom'\n\n\n"
        "def templ(char: str) -> Callable[[str], int]:\n    raise NotImplementedError\n"
    )
    anchor = _anchor_from_source(tmp_path, source, "templ", "splice_gate_fixture")
    assert not _raises("def count_a(s: str) -> int:\n    return s.count('a')\n", anchor)


def test_method_templates_all_flavors(tmp_path):
    # staticmethod/classmethod/instance templates all type-check: the splice leaves
    # decorators untouched, so the method kind (self/cls semantics) is preserved and
    # mypy never spuriously demands a missing ``self``. ``template.__default__`` is
    # the static/classmethod *wrapper* (or the plain function for an instance
    # method), exactly as ``Template.define`` stores it.
    source = (
        "from collections.abc import Callable\n\n\n"
        "class C:\n"
        "    @staticmethod\n"
        "    def sm(char: str) -> Callable[[str], int]:\n"
        "        raise NotImplementedError\n\n"
        "    @classmethod\n"
        "    def cm(cls, char: str) -> Callable[[str], int]:\n"
        "        raise NotImplementedError\n\n"
        "    def im(self, char: str) -> Callable[[str], int]:\n"
        "        raise NotImplementedError\n"
    )
    cls = _import_fixture(tmp_path, source, "splice_method_fixture").C
    good = "def count_a(s: str) -> int:\n    return s.count('a')\n"
    bad = "def count_a(s: str) -> str:\n    return s\n"
    for name in ("sm", "cm", "im"):
        anchor = cls.__dict__[name]  # the wrapper for static/classmethod
        assert not _raises(good, anchor), f"{name}: correct body should pass"
        assert _raises(bad, anchor), f"{name}: wrong return type should raise"


def test_context_enum_and_dataclass_are_checked(tmp_path):
    # The PR's headline value (issue #576): an Enum / dataclass in the template's
    # module is reachable through its real import, where the old quotation
    # mechanism mishandled such types. A correct generated function passes; a
    # misuse is still caught -- all without synthesizing any type stub.
    source = (
        "import dataclasses\n"
        "import enum\n"
        "from collections.abc import Callable\n\n\n"
        "class Color(enum.Enum):\n"
        "    RED = 1\n"
        "    GREEN = 2\n\n\n"
        "@dataclasses.dataclass\n"
        "class Point:\n"
        "    x: int\n"
        "    y: int\n\n\n"
        "def templ() -> Callable[[Color, Point], int]:\n"
        "    raise NotImplementedError\n"
    )
    anchor = _anchor_from_source(tmp_path, source, "templ", "splice_enum_fixture")
    good = (
        "def f(c: Color, p: Point) -> int:\n    return p.x if c is Color.RED else p.y\n"
    )
    bad = "def f(c: Color, p: Point) -> int:\n    return p.nonexistent\n"
    assert not _raises(good, anchor)
    assert _raises(bad, anchor)


def test_generated_reference_to_unsourced_name_raises():
    # A name present only at runtime (injected into the env, absent from the
    # template's source) is unrecoverable -> mypy [name-defined] in-region ->
    # raise. The fail-closed case-3 / injected-name narrowing.
    assert _raises(
        "def count_a(s: str) -> int:\n    return _definitely_not_in_scope(s)\n",
        _count_char,
    )


def test_linecache_backed_template_is_checked():
    # REPL/``exec``-defined templates have no file; their source lives in
    # ``linecache`` (the #690 path). Recovery must read it, so the check runs.
    import linecache

    fname = "<synthesis:splice-linecache-test>"
    src = (
        "from collections.abc import Callable\n\n\n"
        "def templ(char: str) -> Callable[[str], int]:\n"
        "    raise NotImplementedError\n"
    )
    linecache.cache[fname] = (len(src), None, src.splitlines(keepends=True), fname)
    ns: dict[str, Any] = {}
    exec(compile(src, fname, "exec"), ns)
    anchor = ns["templ"]  # __code__.co_filename == fname; source only in linecache
    assert not _raises("def count_a(s: str) -> int:\n    return s.count('a')\n", anchor)
    assert _raises("def count_a(s: str) -> str:\n    return s\n", anchor)


# --- End-to-end: decode through ``Encodable[Callable]`` with the anchor in
# the (result) decode context. The argument path has no anchor and is skipped.


def test_decode_with_anchor_typechecks_and_runs():
    with handler(UnsafeEvalProvider()):
        ta = pydantic.TypeAdapter(Encodable[Callable[[str], int]])
        fn = ta.validate_python(
            SynthesizedFunction(
                module_code="def count_a(s: str) -> int:\n    return s.count('a')"
            ),
            context={TYPE_CHECK_ANCHOR_KEY: _count_char},
        )
        assert fn("banana") == 3


def test_decode_with_anchor_rejects_bad_code():
    with handler(UnsafeEvalProvider()):
        ta = pydantic.TypeAdapter(Encodable[Callable[[str], int]])
        with pytest.raises(Exception):
            ta.validate_python(
                SynthesizedFunction(
                    module_code="def count_a(s: str) -> str:\n    return s"
                ),
                context={TYPE_CHECK_ANCHOR_KEY: _count_char},
            )


def test_decode_without_anchor_skips_typecheck():
    # Argument-path decoding carries no anchor -> the type check is skipped, so
    # even ill-typed code decodes (it is out of scope for this stage).
    with handler(UnsafeEvalProvider()):
        ta = pydantic.TypeAdapter(Encodable[Callable[[str], int]])
        fn = ta.validate_python(
            SynthesizedFunction(
                module_code="def count_a(s: str) -> str:\n    return s"
            ),
            context={},
        )
        assert callable(fn)


def test_decode_with_anchor_rejects_non_nestable():
    # The non-nestable scan is a decode-time precondition of the splice: with an
    # anchor, a star import (illegal once nested in the Template body) is rejected
    # before type checking.
    with handler(UnsafeEvalProvider()):
        ta = pydantic.TypeAdapter(Encodable[Callable[[str], int]])
        with pytest.raises(Exception):
            ta.validate_python(
                SynthesizedFunction(
                    module_code="from os import *\ndef count_a(s: str) -> int:\n    return 0"
                ),
                context={TYPE_CHECK_ANCHOR_KEY: _count_char},
            )


def test_decode_without_anchor_allows_non_nestable():
    # No anchor -> no splice -> the scan is skipped, and the star import is legal at
    # the module level where the code is exec'd. So it decodes and runs fine.
    with handler(UnsafeEvalProvider()):
        ta = pydantic.TypeAdapter(Encodable[Callable[[str], int]])
        fn = ta.validate_python(
            SynthesizedFunction(
                module_code="from os import *\ndef count_a(s: str) -> int:\n    return 0"
            ),
            context={},
        )
        assert callable(fn)


# ============================================================================
# RestrictedEvalProvider security tests
# ============================================================================


def test_restricted_blocks_private_attribute_access():
    """RestrictedPython blocks access to underscore-prefixed attributes by default."""
    source = SynthesizedFunction(
        module_code="""def get_private(s: str) -> int:
    return s.__class__.__name__"""
    )
    # Should raise due to restricted attribute access
    with pytest.raises(Exception):  # Could be NameError or AttributeError
        with handler(RestrictedEvalProvider()):
            fn = pydantic.TypeAdapter(Encodable[Callable[[str], int]]).validate_python(
                source, context={}
            )
            fn("test")


def test_restricted_with_custom_policy():
    """Can pass custom policy via kwargs."""

    # Create a custom policy that's the same as default (just to test the plumbing)
    class CustomPolicy(RestrictingNodeTransformer):
        pass

    source = SynthesizedFunction(
        module_code="""def add(a: int, b: int) -> int:
    return a + b"""
    )
    with handler(RestrictedEvalProvider(policy=CustomPolicy)):
        fn = pydantic.TypeAdapter(Encodable[Callable[[int, int], int]]).validate_python(
            source, context={}
        )
    assert fn(2, 3) == 5


def test_builtins_in_env_does_not_bypass_security():
    """Including __builtins__ in env should not bypass RestrictedEvalProvider security.

    RestrictedEvalProvider explicitly filters out __builtins__ from the env
    to prevent callers from replacing the restricted builtins with full Python builtins.
    This test verifies that even if __builtins__ is passed in the context,
    dangerous operations remain blocked.
    """

    # Attempt to pass full builtins in the context, which should be filtered out
    dangerous_ctx = {"__builtins__": builtins.__dict__}

    # Test 1: open() should not be usable even with __builtins__ in context
    source_open = SynthesizedFunction(
        module_code="""def read_file(path: str) -> str:
    return open(path).read()"""
    )
    with pytest.raises(Exception):  # Could be NameError, ValueError, or other
        with handler(RestrictedEvalProvider()):
            fn = pydantic.TypeAdapter(Encodable[Callable[[str], str]]).validate_python(
                source_open, context=dangerous_ctx
            )
            fn("/etc/passwd")

    # Test 2: __import__ should not be usable
    source_import = SynthesizedFunction(
        module_code="""def get_os_name() -> str:
    os = __import__('os')
    return os.name"""
    )
    with pytest.raises(Exception):
        with handler(RestrictedEvalProvider()):
            fn = pydantic.TypeAdapter(Encodable[Callable[[], str]]).validate_python(
                source_import, context=dangerous_ctx
            )
            fn()

    # Test 3: Verify safe code still works with dangerous context
    source_safe = SynthesizedFunction(
        module_code="""def add(a: int, b: int) -> int:
    return a + b"""
    )
    with handler(RestrictedEvalProvider()):
        fn = pydantic.TypeAdapter(Encodable[Callable[[int, int], int]]).validate_python(
            source_safe, context=dangerous_ctx
        )
        assert fn(2, 3) == 5, "Safe code should still work"

    # Test 4: Private attribute access should still be blocked
    source_private = SynthesizedFunction(
        module_code="""def get_class(s: str) -> str:
    return s.__class__.__name__"""
    )
    with pytest.raises(Exception):
        with handler(RestrictedEvalProvider()):
            fn = pydantic.TypeAdapter(Encodable[Callable[[str], str]]).validate_python(
                source_private, context=dangerous_ctx
            )
            fn("test")


# ============================================================================
# ReplSession (#678)
# ============================================================================


def _code(source: str) -> types.CodeType:
    """Compile `source` to a code object the way the `exec_code` tool boundary
    does -- through `Encodable[CodeType]`, which routes the active eval
    provider's `parse`/`compile` ops (so a handler must be installed)."""
    return pydantic.TypeAdapter(Encodable[types.CodeType]).validate_python(source)


def test_repl_seeds_from_lexical_context():
    """Names in the seed context are usable in executed code."""
    with handler(UnsafeEvalProvider()):
        session = ReplSession({"readings": [10, 20, 30]})
        assert session.exec_code(_code("print(sum(readings))")) == "60\n"


def test_repl_persists_bindings_across_calls():
    """A binding created in one call is visible in the next."""
    with handler(UnsafeEvalProvider()):
        session = ReplSession({})
        session.exec_code(_code("total = 41"))
        assert session.exec_code(_code("print(total)")) == "41\n"


def test_repl_rebinds_across_calls():
    """A seeded/prior name can be rebound using its prior value."""
    with handler(UnsafeEvalProvider()):
        session = ReplSession({"x": 10})
        session.exec_code(_code("x = x + 1"))
        assert session.exec_code(_code("print(x)")) == "11\n"


def test_repl_runs_complete_multistatement_block():
    """A complete `def` + call in one snippet runs in one shot (the case
    `single`-mode compilation would reject)."""
    with handler(UnsafeEvalProvider()):
        session = ReplSession({})
        out = session.exec_code(
            _code("def double(n):\n    return n * 2\nprint(double(21))")
        )
        assert out == "42\n"


def test_repl_captures_print():
    with handler(UnsafeEvalProvider()):
        assert ReplSession({}).exec_code(_code("print('hi')")) == "hi\n"


def test_repl_rejects_invalid_source_at_construction():
    """Invalid source is rejected when it is decoded to a code object -- before it
    ever reaches the session -- and valid code in the same provider still runs."""
    with handler(UnsafeEvalProvider()):
        with pytest.raises(pydantic.ValidationError):
            _code("def f(:")
        assert ReplSession({}).exec_code(_code("print('ok')")) == "ok\n"


def test_repl_exception_is_isolated():
    """A runtime exception is reported in the call's output; the session survives
    and retains prior state."""
    with handler(UnsafeEvalProvider()):
        session = ReplSession({})
        session.exec_code(_code("kept = 7"))
        out = session.exec_code(_code("print(1 / 0)"))
        assert "ZeroDivisionError" in out
        assert session.exec_code(_code("print(kept)")) == "7\n"


def test_repl_system_exit_propagates():
    """`SystemExit` propagates, mirroring `InteractiveInterpreter.runcode`; every
    other exception (including `KeyboardInterrupt`) is caught and surfaced as
    output rather than escaping the call."""
    with handler(UnsafeEvalProvider()):
        session = ReplSession({})
        with pytest.raises(SystemExit):
            session.exec_code(_code("raise SystemExit"))
        out = session.exec_code(_code("raise KeyboardInterrupt"))
        assert "KeyboardInterrupt" in out


def test_repl_cross_snippet_traceback_shows_correct_source():
    """A function defined in an earlier call that raises in a later call formats
    with its *own* source line, not the later call's source -- the per-snippet
    filename keeps each cell's source in linecache."""
    with handler(UnsafeEvalProvider()):
        session = ReplSession({})
        session.exec_code(_code("def boom():\n    return 1 / 0"))
        out = session.exec_code(_code("boom()"))
        assert "return 1 / 0" in out  # boom's real source


def test_repl_new_binding_does_not_leak_into_seed_context():
    """A binding created in executed code stays in the session; the seed mapping
    the session was created from is not mutated."""
    with handler(UnsafeEvalProvider()):
        seed = {"base": 10}
        session = ReplSession(seed)
        session.exec_code(_code("derived = base + 5"))
        assert session.exec_code(_code("print(derived)")) == "15\n"
        assert "derived" not in seed  # the lexical seed is untouched


def test_repl_binding_is_visible_to_the_rest_of_the_call():
    """Seeded from the per-call `ChainMap`, a binding the REPL makes lands in a
    shared shadow layer of that chain -- so the rest of the Template call sees it
    -- while the read-only lexical layer underneath does not, keeping it scoped to
    the call (mirroring `exec`)."""
    with handler(UnsafeEvalProvider()):
        lexical = {"base": 10}
        env: ChainMap[str, Any] = ChainMap({}, lexical)
        ReplSession(env).exec_code(_code("shared = base + 5"))
        assert env["shared"] == 15  # visible to the rest of the call via the chain
        assert "shared" not in lexical  # but not leaked into the lexical context


def test_repl_keeps_stdout_and_stderr_separate():
    """stdout (print output) and stderr (tracebacks) accumulate in separate,
    introspectable buffers; `exec_code` returns this call's stdout then stderr."""
    with handler(UnsafeEvalProvider()):
        session = ReplSession({})
        out = session.exec_code(_code("print('hi')\n1 / 0"))
        assert session.stdout.getvalue() == "hi\n"
        assert "ZeroDivisionError" in session.stderr.getvalue()
        assert "hi" not in session.stderr.getvalue()  # the streams don't mix
        assert out == "hi\n" + session.stderr.getvalue()  # returned: stdout then stderr


def test_repl_runsource_routes_through_ops():
    """`runsource` compiles through the `parse`/`compile` ops (so it needs an
    installed provider), keeping it self-consistent with `runcode`/`exec_code`
    rather than falling back to the native single-mode compiler."""
    with pytest.raises(NotImplementedError):
        ReplSession({}).runsource("x = 1")  # no provider -> the parse op raises
    with handler(UnsafeEvalProvider()):
        session = ReplSession({})
        assert session.runsource("kept = 7") is False  # complete: compiled + ran
        assert session.locals["kept"] == 7


# ----------------------------------------------------------------------------
# ReplSession under RestrictedEvalProvider (state + print fixed in #686)
# ----------------------------------------------------------------------------


def test_repl_rebinds_across_calls_restricted():
    with handler(RestrictedEvalProvider()):
        session = ReplSession({"x": 10})
        session.exec_code(_code("x = x + 1"))
        assert session.exec_code(_code("print(x)")) == "11\n"


def test_repl_captures_print_restricted():
    with handler(RestrictedEvalProvider()):
        assert ReplSession({}).exec_code(_code("print('hi')")) == "hi\n"


# ============================================================================
# RestrictedEvalProvider state-retention and print (#685)
# ============================================================================


def _restricted_run(source: str, ns: dict, capture: bool = False) -> str | None:
    """Run one snippet through the parse/compile/exec ops under
    RestrictedEvalProvider, optionally capturing stdout."""
    with handler(RestrictedEvalProvider()):
        code = compile_op(parse_op(source, "<f>"), "<f>")
        if capture:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec_op(code, ns)
            return buf.getvalue()
        exec_op(code, ns)
        return None


def test_restricted_exec_copies_back_rebound_seed():
    """#685: rebinding a name already present in the namespace writes the new
    value back, not just never-before-seen names."""
    ns = {"x": 1}
    _restricted_run("x = 99", ns)
    assert ns["x"] == 99


def test_restricted_exec_copies_back_rebound_new_key():
    """#685: a key that becomes a 'seed' after its first definition is still
    rebindable on subsequent calls."""
    ns: dict = {}
    _restricted_run("y = 1", ns)
    _restricted_run("y = 2", ns)
    assert ns["y"] == 2


def test_restricted_exec_persists_and_rebinds_across_calls():
    """#685: the namespace is a real REPL session — a binding from one call is
    usable in the next, and rebinding it using its prior value works."""
    ns: dict = {}
    _restricted_run("x = 10", ns)
    _restricted_run("x = x + 1", ns)
    assert ns["x"] == 11


def test_restricted_exec_print_captured_to_stdout():
    """#685: RestrictedPython's `print` is routed to the real stdout so
    output-capturing callers see it (rather than NameError on `_print_`)."""
    out = _restricted_run("print('hi')", {}, capture=True)
    assert out == "hi\n"


# ============================================================================
# REPL code type-checking (issue #690)
#
# `exec_code` type-checks the cumulative session code -- prior snippets plus the
# current one -- spliced into the enclosing Template's body (`_splice_repl` + the
# `type_check` op, `lenient=True`), so names resolve in their real execution
# context. These tests drive that pipeline against a real anchor and assert the
# contract by exception type / runtime effect -- never by matching a mypy message
# or a filename.
# ============================================================================


def _repl_anchor(readings: list[int]) -> int:
    """A module-level stand-in for a Template function: the REPL splice target. Its
    real source is recoverable (it lives in this test module) and its parameter
    `readings` stands in for a name from the session's seed env."""
    raise NotImplementedError


def _repl_raises(prior: list[str], snippet: str) -> bool:
    """type_check the cumulative REPL code spliced into `_repl_anchor`'s body; True
    if it reports an in-region error."""
    checked = _splice_repl(prior, snippet, _repl_anchor)
    assert checked is not None
    with handler(UnsafeEvalProvider()):
        try:
            type_check(*checked, lenient=True)
            return False
        except TypeError:
            return True


# --- Type-check semantics: checked in the Template body, leniently ---


def test_repl_seed_env_read_is_checked():
    """A snippet reading a seed name (a Template parameter) resolves in the spliced
    body and is checked -- the case the old module-scope gate could only decline.
    A correct use is clean; misusing the seed's type is a real error."""
    assert not _repl_raises([], "total = sum(readings)\nprint(total)")
    assert _repl_raises([], "bad: str = sum(readings)\nprint(bad)")


def test_repl_cross_snippet_names_resolve():
    """A helper defined in an earlier cell resolves when a later cell uses it (the
    cumulative splice); calling it with the wrong type is caught."""
    prior = ["def helper(n: int) -> int:\n    return n + 1"]
    assert not _repl_raises(prior, "print(helper(3))")
    assert _repl_raises(prior, "print(helper('bad'))")


def test_repl_rebind_across_cells_is_lenient():
    """A cell may rebind a name to a new type and use it -- normal REPL editing,
    allowed by `--allow-redefinition`, not a type error."""
    assert not _repl_raises(["x = 1"], "x = 'now a string'\nprint(x.upper())")


def test_repl_body_need_not_return_template_type():
    """REPL code isn't a function returning the Template's declared type; the return
    contract is waived, so a body with no matching return is clean."""
    assert not _repl_raises([], "y = sum(readings)\nprint(y)")


def test_repl_illtyped_but_runnable_snippet_is_caught():
    """An ill-typed snippet that would execute with no runtime error is reported --
    plain execution never catches it."""
    assert _repl_raises([], "n: int = 'oops'\nprint(n)")


def test_repl_check_region_is_the_current_snippet():
    """Only the current snippet's lines are reported: an error confined to a prior
    snippet is out of region (not raised); the same error in the current snippet is
    in region (raised)."""
    assert not _repl_raises(["bad: int = 'x'"], "ok = 1\nprint(ok)")
    assert _repl_raises([], "bad: int = 'x'\nprint(bad)")


# --- exec_code behavior with an anchor: report-not-gate ---


def test_repl_exec_code_reports_illtyped_but_runs_and_survives():
    """With an anchor, an ill-typed-but-runnable snippet writes a diagnostic to
    stderr yet still runs, and the session survives -- a later valid snippet
    reading the seed still produces its value."""
    with handler(UnsafeEvalProvider()):
        session = ReplSession({"readings": [1, 2, 3]}, anchor=_repl_anchor)
        before = len(session.stderr.getvalue())
        session.exec_code(_code("bad: int = 'x'"))
        assert len(session.stderr.getvalue()) > before  # a type error was reported
        assert session.locals["bad"] == "x"  # ... and the snippet still ran
        assert session.exec_code(_code("print(sum(readings))")) == "6\n"


def test_repl_exec_code_valid_snippet_is_not_reported():
    """A well-typed snippet runs and produces no diagnostic."""
    with handler(UnsafeEvalProvider()):
        session = ReplSession({"readings": [1, 2, 3]}, anchor=_repl_anchor)
        out = session.exec_code(_code("good: int = sum(readings)\nprint(good)"))
        assert out == "6\n"
        assert session.stderr.getvalue() == ""


def test_repl_no_anchor_runs_without_type_checking():
    """Without an anchor (outside a managed Template call) the session runs code and
    never type-checks it, so an ill-typed-but-runnable snippet produces no
    diagnostic."""
    with handler(UnsafeEvalProvider()):
        session = ReplSession({})  # no anchor
        out = session.exec_code(_code("bad: int = 'x'\nprint(bad)"))
        assert out == "x\n"
        assert session.stderr.getvalue() == ""


# --- decode boundary: non-nestable constructs rejected at Encodable[CodeType] ---


def test_encodable_code_rejects_future_import():
    """`from __future__ import ...` is illegal once nested in a function body, so it
    is rejected when the code object is decoded, before it can run."""
    with handler(UnsafeEvalProvider()):
        with pytest.raises(pydantic.ValidationError):
            _code("from __future__ import annotations\nx = 1")


def test_encodable_code_rejects_star_import():
    """A star import is likewise rejected at decode."""
    with handler(UnsafeEvalProvider()):
        with pytest.raises(pydantic.ValidationError):
            _code("from os import *\nx = 1")
