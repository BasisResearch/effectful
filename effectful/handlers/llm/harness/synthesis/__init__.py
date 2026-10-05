"""Let a Skill answer by writing Python.

.. list-table:: Synthesis routes and checks
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
   * - Expression Tool call
       (:mod:`~effectful.handlers.llm.harness.synthesis.toolcall`)
     - parse and check one Python call expression; ``"auto"`` uses it for
       signatures JSON cannot encode, ``"code"`` for every collected Tool
     - the Tool's own result and any installed Skill argument contracts
     - expression calls to plain Tools do not enforce parameter metadata
   * - Structured return
     - decoding, schema, and installed validators
     - predicates encoded in the declared contract
     - shape validity alone says nothing about semantics

The code routes need an eval provider; structured returns do not. Static
checking also needs recoverable source and an installed type checker; when
source recovery fails, that check is
skipped. Parsing, execution, and applicable doctests remain separate checks.

A generic ``Callable`` return promises an implementation that works for every
type parameter, even when one call passes a concrete type::

    from collections.abc import Callable
    from effectful.handlers.llm import Skill

    @Skill.define
    def make_fn[T](typ: type[T]) -> Callable[[T], T]:
        \"\"\"Return a type-preserving function for values of {typ}.\"\"\"

A returned ``def f(x: int) -> int`` does not satisfy this generic contract.
"""
