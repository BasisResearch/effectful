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
import typing
from collections.abc import Mapping
from typing import Any

from effectful.handlers.llm.completions import (
    Message,
    _make_message,
    append_message,
    call_user,
)
from effectful.handlers.llm.evaluation import test
from effectful.handlers.llm.template import Template
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements

_SENTINEL = object()

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

    # Case 2: per-call cached doctest examples for test() validation.
    _doctest_stack: list[list[doctest.Example]]

    # Case 1: prefix messages to inject before the next call_user.
    _pending_prefix: list[Message] | None

    # Re-entrancy guard: set of templates currently being calibrated.
    _calibrating: set[Template]

    def __init__(self) -> None:
        self._extraction_cache = {}
        self._doctest_stack = []
        self._prefix_cache = {}
        self._pending_prefix = None
        self._calibrating = set()

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
            # Case 2 – code generation: push cached examples for test().
            self._doctest_stack.append(examples)
            return fwd()

        # Case 1 – tool-calling: calibration + prefix.
        if template not in self._calibrating and template not in self._prefix_cache:
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
        examples = self._doctest_stack.pop()
        if not examples:
            return

        name = (
            f"{getattr(obj, '__name__', obj.__class__.__name__)}"
            ".__template_doctest__"
        )
        test_case = doctest.DocTest(
            examples=examples,
            globs=dict(ctx),
            name=name,
            filename=None,
            lineno=0,
            docstring=None,
        )

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
        """Run calibration as a mini ReAct agent with Agent-style history.

        Reuses the same persistent-history mechanism as :class:`Agent`: a
        shared :class:`~collections.OrderedDict` bound to
        ``template.__history__`` that accumulates messages across calls.
        Each doctest example is evaluated in order; the LLM sees all prior
        conversation turns (including any corrective feedback for incorrect
        answers) when processing subsequent examples, enabling it to learn
        from the full experience.
        """
        self._calibrating.add(template)

        # Agent-style history: a single OrderedDict shared across all
        # calibration examples, exactly like Agent.__history__.
        shared_history: collections.OrderedDict[str, Message] = (
            collections.OrderedDict()
        )

        # Temporarily bind the shared history to the template, using the
        # same mechanism Agent.__get__ uses.
        old_history = getattr(template, "__history__", _SENTINEL)
        template.__history__ = shared_history  # type: ignore[attr-defined]

        try:
            for example in examples:
                call_args, call_kwargs = _parse_template_call(
                    example, template.__name__
                )
                if call_args is None or call_kwargs is None:
                    continue  # not a call to this template

                # Call the template; the provider reads template.__history__
                # and writes back after completion, so messages naturally
                # accumulate in shared_history.
                result = template(*call_args, **call_kwargs)

                # Check output; append corrective feedback if wrong so
                # subsequent examples benefit from the correction.
                checker = doctest.OutputChecker()
                actual = repr(result) + "\n"
                optionflags = 0
                for flag, val in example.options.items():
                    if val:
                        optionflags |= flag
                if not checker.check_output(example.want, actual, optionflags):
                    feedback = _make_message(
                        {
                            "role": "user",
                            "content": (
                                f"That was incorrect. "
                                f"Expected {example.want.strip()!r} "
                                f"but got {repr(result)!r}."
                            ),
                        }
                    )
                    shared_history[feedback["id"]] = feedback
        finally:
            self._calibrating.discard(template)
            # Restore original state.
            if old_history is _SENTINEL:
                try:
                    del template.__history__  # type: ignore[attr-defined]
                except AttributeError:
                    pass
            else:
                template.__history__ = old_history  # type: ignore[attr-defined]

        # Cache non-system messages as the prefix for future calls.
        self._prefix_cache[template] = [
            m for m in shared_history.values() if m["role"] != "system"
        ]


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
