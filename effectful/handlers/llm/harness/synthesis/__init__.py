"""Handlers that let the model answer with code rather than data.

A stateful REPL, a synthesized function, a synthesized body for the calling
:class:`~effectful.handlers.llm.types.Skill`, and expression-based tool calls
(the pathway that supports polymorphic tools). All of them run model-authored
Python through the operations of :mod:`~effectful.handlers.llm.harness.execution`, so each needs an eval
provider; ``harness(eval_provider="none")`` leaves them all out.

.. rubric:: Synthesis and certification

A Skill's return annotation is the reply contract for its turn. When that reply
is model-authored Python, there are several routes for producing it, each with a
different contract. Do not infer a check from the enclosing Skill's docstring
until the synthesis route is known:

.. list-table::
   :header-rows: 1

   * - Route
     - Source/type gate
     - Authoritative behavioral checks
     - Important boundary
   * - Directly returned ``Callable``
       (:mod:`~effectful.handlers.llm.harness.synthesis.function`)
     - parse and compile; source splice and static checking when source
       recovery succeeds and a checker is installed
     - doctests written inside the **model-authored returned function**
     - the model may omit or weaken those doctests
   * - ``write_and_run_body``
       (:mod:`~effectful.handlers.llm.harness.synthesis.body`)
     - parse and compile; source-level checking under the enclosing Skill's
       signature when available
     - fixed, **caller-authored Skill-docstring doctests**, followed by
       execution on the current arguments
     - the implementation is not installed permanently
   * - ``exec_code``
       (:mod:`~effectful.handlers.llm.harness.synthesis.snippet`)
     - lenient snippet checking before execution
     - whatever assertions or external calls the snippet actually runs
     - exploratory success is not final-artifact certification
   * - Structured return
     - decoding, schema, and installed validators
     - predicates encoded in the declared contract
     - shape validity alone says nothing about semantics

If source recovery fails, callable/body static checking is skipped rather than
replaced by an approximate check; ``type_checker="none"`` likewise makes the
checking hook a no-op. Syntax parsing, compilation, and applicable doctests
remain separate gates. When a checker does run, only diagnostics attributed to
the generated source span block synthesis; surrounding-source diagnostics are
context, not newly caused failures. Prior REPL snippets are prepended when
checking a later REPL snippet, but not when independently checking a directly
returned Callable or a synthesized Skill body.

.. rubric:: Generic Skills are universal contracts

Arguments to the Skill turn can instantiate the response schema. For example,
``type[T]`` gives the decoder a reliable way to learn which concrete schema the
caller expects. Static checking remains against the original generic Skill,
however::

    from collections.abc import Callable

    from effectful.handlers.llm import Skill


    @Skill.define
    def make_fn[T](typ: type[T]) -> Callable[[T], T]:
        \"\"\"Return a type-preserving function for values of {typ}.\"\"\"

The contract says ``make_fn`` works for every ``T``. A returned
``def f(x: int) -> int`` does not satisfy it merely because this invocation
passed ``int``; the returned implementation must itself be parametric. If the
intended behavior is to synthesize a different concrete implementation per
runtime class, use fixed concrete Skills, a non-generic base interface with
explicit applicability metadata, or another API that does not promise universal
parametricity.
"""
