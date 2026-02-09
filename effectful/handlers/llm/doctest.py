"""Doctest semantic constraints for Templates.

Provides a :class:`DoctestHandler` that uses ``>>>`` examples in template
docstrings as semantic constraints rather than literal prompts.

**Case 1 (tool-calling)**: When the template returns a non-Callable type, a
calibration loop runs the doctest inputs through the LLM once per template
definition and caches the entire conversation (including any incorrect
attempts) as a few-shot prefix for future calls, emulating a learning
process.

**Case 2 (code generation)**: When the template returns a ``Callable`` type,
the generated code is required to pass the doctests as post-hoc validation.

In both cases, ``>>>`` examples are stripped from the prompt sent to the LLM
so it cannot memorise the expected outputs.
"""

import ast
import collections
import collections.abc
import doctest
import inspect
import textwrap
import typing
from collections.abc import Mapping
from typing import Any

from effectful.handlers.llm.completions import (
    Message,
    _make_message,
    append_message,
    call_user,
    get_message_sequence,
)
from effectful.handlers.llm.evaluation import test
from effectful.handlers.llm.template import Template
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def extract_doctests(docstring: str) -> tuple[str, list[doctest.Example]]:
    """Separate a docstring into text-without-examples and a list of examples.

    Uses :class:`doctest.DocTestParser` to identify ``>>>`` blocks, then
    reconstructs the docstring with those blocks removed.

    Returns ``(stripped_text, examples)`` where *stripped_text* is the
    docstring with all interactive examples removed.
    """
    parser = doctest.DocTestParser()
    parts = parser.parse(docstring)
    text_parts = [p for p in parts if isinstance(p, str)]
    examples = [p for p in parts if isinstance(p, doctest.Example)]
    return "".join(text_parts), examples


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _is_callable_return(template: Template) -> bool:
    """Return ``True`` if *template* synthesises a ``Callable``."""
    ret = template.__signature__.return_annotation
    origin = typing.get_origin(ret)
    if origin is not None:
        # e.g. Callable[[str], int] -> origin is collections.abc.Callable
        return origin is collections.abc.Callable
    if isinstance(ret, type):
        return issubclass(ret, collections.abc.Callable)  # type: ignore[arg-type]
    return False


class DoctestHandler(ObjectInterpretation):
    """Use ``>>>`` examples in template docstrings as semantic constraints.

    Install with ``handler(DoctestHandler())`` alongside a provider and an
    eval provider.  See the module docstring for the two cases handled.
    """

    # Per-template extraction cache (stripped template + examples).
    _extraction_cache: dict[Template, tuple[str, list[doctest.Example]]]

    # Case 1: calibration conversation prefix, cached per template.
    _prefix_cache: dict[Template, list[Message]]

    # Case 2: per-call formatted doctest source for test() validation.
    _doctest_stack: list[str]

    # Case 1: prefix messages to inject before the next call_user.
    _pending_prefix: list[Message] | None

    # Re-entrancy guard for calibration.
    _calibrating: bool

    def __init__(self) -> None:
        self._extraction_cache = {}
        self._doctest_stack = []
        self._prefix_cache = {}
        self._pending_prefix = None
        self._calibrating = False

    # -- helpers ------------------------------------------------------------

    def _get_doctests(self, template: Template) -> tuple[str, list[doctest.Example]]:
        """Return cached ``(stripped_template, examples)`` for *template*."""
        try:
            return self._extraction_cache[template]
        except KeyError:
            result = extract_doctests(template.__prompt_template__)
            self._extraction_cache[template] = result
            return result

    @implements(Template.__apply__)
    def _handle_template[**P, T](
        self,
        template: Template[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        _, examples = self._get_doctests(template)

        if not examples:
            return fwd()

        if _is_callable_return(template):
            # Case 2 – code generation: push formatted doctests for test().
            bound_args = inspect.signature(template).bind(*args, **kwargs)
            bound_args.apply_defaults()
            env = template.__context__.new_child(bound_args.arguments)
            formatted = textwrap.dedent(template.__prompt_template__).format_map(env)
            self._doctest_stack.append(formatted)
            return fwd()

        # Case 1 – tool-calling: calibration + prefix.
        if not self._calibrating and template not in self._prefix_cache:
            self._calibrate(template, examples)

        if template in self._prefix_cache and self._prefix_cache[template]:
            # Schedule prefix injection for _strip_prompt, which runs
            # after call_system (so the system message is already first).
            self._pending_prefix = self._prefix_cache[template]
            try:
                return fwd()
            finally:
                self._pending_prefix = None

        return fwd()

    # -- call_user (stateless stripping) ------------------------------------

    @implements(call_user)
    def _strip_prompt(
        self,
        template: str,
        env: Mapping[str, Any],
    ) -> Message:
        """Strip ``>>>`` examples and inject any pending calibration prefix.

        This runs after ``call_system`` has already appended the system
        message, so injecting prefix messages here keeps the correct order:
        system → prefix user/assistant turns → actual user message.
        """
        # Inject cached calibration prefix (Case 1) into the message
        # sequence before the actual user message.
        if self._pending_prefix is not None:
            for msg in self._pending_prefix:
                append_message(msg)
            self._pending_prefix = None

        stripped, _ = extract_doctests(template)
        return fwd(stripped, env)

    # -- test (Case 2 validation) -------------------------------------------

    @implements(test)
    def _run_from_stack(self, obj: object, ctx: typing.Mapping[str, Any]) -> None:
        if not self._doctest_stack:
            return
        doctest_source = self._doctest_stack.pop()
        if not doctest_source.strip():
            return

        globs = dict(ctx)
        parser = doctest.DocTestParser()
        test_case = parser.get_doctest(
            doctest_source,
            globs,
            name=(
                f"{getattr(obj, '__name__', obj.__class__.__name__)}"
                ".__template_doctest__"
            ),
            filename=None,
            lineno=0,
        )
        if not test_case.examples:
            return

        output: list[str] = []
        runner = doctest.DocTestRunner(verbose=False)
        runner.run(test_case, out=output.append)
        results = runner.summarize(verbose=False)
        if results.failed:
            report = "".join(output).strip()
            if not report:
                report = (
                    f"{results.failed} doctest(s) failed "
                    f"out of {results.attempted} attempted."
                )
            raise TypeError(f"doctest failed:\n{report}")

    # -- Case 1 calibration -------------------------------------------------
    def _calibrate(
        self,
        template: Template,
        examples: list[doctest.Example],
    ) -> None:
        """Run a calibration loop for tool-calling templates.

        For each doctest example that calls *template*, the template is
        invoked with the example's arguments (prompt stripped of doctests).
        All conversation turns — including any incorrect attempts — are
        accumulated into a prefix that is cached for future calls, so the
        LLM can learn from the full experience.
        """
        prefix_messages: list[Message] = []
        self._calibrating = True

        try:
            for example in examples:
                call_args, call_kwargs = _parse_template_call(
                    example, template.__name__
                )
                if call_args is None:
                    continue  # not a call to this template

                # Run in an isolated message sequence.
                cal_msgs: collections.OrderedDict[str, Message] = (
                    collections.OrderedDict()
                )
                with handler({get_message_sequence: lambda: cal_msgs}):
                    result = template(*call_args, **call_kwargs)

                # Check output; append corrective feedback if wrong.
                checker = doctest.OutputChecker()
                actual = repr(result) + "\n"
                # example.options is dict[int, bool]; reduce to int flags.
                optionflags = 0
                for flag, val in example.options.items():
                    if val:
                        optionflags |= flag
                if not checker.check_output(example.want, actual, optionflags):
                    append_message(
                        _make_message(
                            {
                                "role": "user",
                                "content": (
                                    f"That was incorrect. "
                                    f"Expected {example.want.strip()!r} "
                                    f"but got {repr(result)!r}."
                                ),
                            }
                        )
                    )

                # Keep user/assistant turns (skip system messages since
                # call_system will re-add it during the actual call).
                prefix_messages.extend(
                    m for m in cal_msgs.values() if m["role"] != "system"
                )
        finally:
            self._calibrating = False

        self._prefix_cache[template] = prefix_messages


def _parse_template_call(
    example: doctest.Example, template_name: str
) -> tuple[list[Any] | None, dict[str, Any] | None]:
    """Extract positional and keyword args from a doctest example.

    Returns ``(args, kwargs)`` if the example is a call to *template_name*,
    or ``(None, None)`` otherwise.
    """
    source = example.source.strip()
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError:
        return None, None

    expr = tree.body
    if not isinstance(expr, ast.Call):
        return None, None
    if not isinstance(expr.func, ast.Name):
        return None, None
    if expr.func.id != template_name:
        return None, None

    try:
        pos_args = [ast.literal_eval(a) for a in expr.args]
        kw_args = {
            kw.arg: ast.literal_eval(kw.value)
            for kw in expr.keywords
            if kw.arg is not None
        }
    except (ValueError, TypeError):
        return None, None

    return pos_args, kw_args
