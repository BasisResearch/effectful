"""Doctest semantic constraints for Templates.

Provides a :class:`DoctestHandler` that uses ``>>>`` examples in template
docstrings as semantic constraints rather than literal prompts.

**Case 1 (tool-calling)**: When the template returns a non-Callable type, a
calibration loop runs the doctest inputs through the LLM once per template
definition and caches the entire conversation (including any incorrect
attempts) as a few-shot prefix for future calls, emulating a mini ReAct
agent that learns from its mistakes.

**Case 2 (code generation)**: When the template returns a ``Callable`` type,
the generated code is required to pass the doctests as post-hoc validation.

In both cases, ``>>>`` examples are stripped from the prompt sent to the LLM
so it cannot memorise the expected outputs.
"""

import ast
import collections
import contextlib
import doctest
import typing
from collections.abc import Mapping
from typing import Any

from effectful.handlers.llm.completions import (
    Message,
    _make_message,
    call_user,
)
from effectful.handlers.llm.encoding import CallableEncodable, Encodable
from effectful.handlers.llm.evaluation import test
from effectful.handlers.llm.template import Template
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements

_SENTINEL = object()


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

    # Re-entrancy guard: set of templates currently being calibrated.
    _calibrating: set[Template]

    def __init__(self) -> None:
        self._extraction_cache = {}
        self._doctest_stack = []
        self._prefix_cache = {}
        self._calibrating = set()

    # -- helpers ------------------------------------------------------------

    @classmethod
    def extract_doctests(cls, docstring: str) -> tuple[str, list[doctest.Example]]:
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

    @staticmethod
    def _parse_template_call(
        example: doctest.Example, template_name: str
    ) -> tuple[list[Any] | None, dict[str, Any] | None]:
        """Extract positional and keyword args from a doctest example.

        Returns ``(args, kwargs)`` if the example is a call to
        *template_name*, or ``(None, None)`` otherwise.
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

    @contextlib.contextmanager
    def _bind_history(
        self,
        template: Template,
        history: collections.OrderedDict[str, Message],
    ):
        """Temporarily bind *history* to ``template.__history__``.

        Uses the same attribute that :class:`Agent` binds via ``__get__``.
        The provider reads and writes back to it, so messages accumulate.
        """
        old = getattr(template, "__history__", _SENTINEL)
        template.__history__ = history  # type: ignore[attr-defined]
        try:
            yield
        finally:
            if old is _SENTINEL:
                try:
                    del template.__history__  # type: ignore[attr-defined]
                except AttributeError:
                    pass
            else:
                template.__history__ = old  # type: ignore[attr-defined]

    # -- Template.__apply__ -------------------------------------------------

    @implements(Template.__apply__)
    def _handle_template[**P, T](
        self,
        template: Template[P, T],
        *_args: P.args,
        **_kwargs: P.kwargs,
    ) -> T:
        if template not in self._extraction_cache:
            self._extraction_cache[template] = self.extract_doctests(
                template.__prompt_template__
            )
        _, examples = self._extraction_cache[template]

        if not examples:
            return fwd()

        if isinstance(
            Encodable.define(template.__signature__.return_annotation),
            CallableEncodable,
        ):
            # Case 2 – code generation: push cached examples for test().
            self._doctest_stack.append(examples)
            return fwd()

        # Case 1 – tool-calling: calibration + prefix.
        if template not in self._calibrating and template not in self._prefix_cache:
            self._calibrate(template, examples)

        prefix = self._prefix_cache.get(template, [])
        if prefix:
            # Pre-populate history with the cached calibration prefix
            # (Agent-style); the provider will copy it and prepend the
            # system message, so the LLM sees:
            #   system → prefix user/assistant turns → actual user message.
            prefix_history: collections.OrderedDict[str, Message] = (
                collections.OrderedDict((m["id"], m) for m in prefix)
            )
            with self._bind_history(template, prefix_history):
                return fwd()

        return fwd()

    # -- call_user ----------------------------------------------------------

    @implements(call_user)
    def _strip_prompt(
        self,
        template: str,
        env: Mapping[str, Any],
    ) -> Message:
        """Strip ``>>>`` examples from the prompt before the LLM sees it."""
        stripped, _ = self.extract_doctests(template)
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

        shared_history: collections.OrderedDict[str, Message] = (
            collections.OrderedDict()
        )

        with self._bind_history(template, shared_history):
            try:
                for example in examples:
                    self._run_calibration_example(
                        template, example, shared_history
                    )
            finally:
                self._calibrating.discard(template)

        self._prefix_cache[template] = [
            m for m in shared_history.values() if m["role"] != "system"
        ]

    def _run_calibration_example(
        self,
        template: Template,
        example: doctest.Example,
        history: collections.OrderedDict[str, Message],
    ) -> None:
        """Evaluate one doctest example and append corrective feedback."""
        call_args, call_kwargs = self._parse_template_call(
            example, template.__name__
        )
        if call_args is None or call_kwargs is None:
            return

        result = template(*call_args, **call_kwargs)

        checker = doctest.OutputChecker()
        actual = repr(result) + "\n"
        optionflags = sum(f for f, v in example.options.items() if v)

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
            history[feedback["id"]] = feedback
