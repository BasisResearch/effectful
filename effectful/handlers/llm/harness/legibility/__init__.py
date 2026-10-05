"""Handlers that assemble what the model sees.

The framework documentation in the system prompt, and the tools and definitions
drawn from a :class:`~effectful.handlers.llm.types.Skill`'s lexical scope.

.. rubric:: Scope, visibility, and capability

Lexical scope supplies the environment for a Skill turn, and "in scope" means
five different things at successive layers of that turn:

1. **Python lexical context.** ``Skill.define`` captures module globals and true
   enclosing-function locals; binding a Skill method adds ``self``
   (:mod:`effectful.handlers.llm.types`).
2. **Model-visible context.** The system message carries the defining module's
   source, the receiver's class and Skill documentation, and tables of the
   imports and other bindings in scope; the request interpolates only the
   values its format fields name. The section builders are in
   :mod:`~effectful.handlers.llm.harness.legibility.lexical`; the framework sections in
   :mod:`~effectful.handlers.llm.harness.legibility.framework`.
3. **REPL runtime context.** With an executor installed, each turn opens a
   session over the captured context; what survives it is in
   :mod:`~effectful.handlers.llm.harness.synthesis.snippet`.
4. **Static synthesis context.** Generated code is checked spliced into the
   defining module's recovered source: :mod:`~effectful.handlers.llm.harness.synthesis`.
5. **Execution authority.** The executor decides what generated code can read or
   invoke: :mod:`~effectful.handlers.llm.harness.execution`.

    **Tool advertisement is an affordance, not an authority boundary.**

Lexical separation makes intended information flow legible, and the builtin
executor lets generated code reach anything in scope regardless of what was
advertised. A claim that depends on adversarial confidentiality needs a process
boundary, a service API, or externally held data (:mod:`~effectful.handlers.llm.harness.execution`).

Use a true enclosing scope to give one Skill a deliberately small vocabulary::

    from effectful.handlers.llm import Skill, Tool


    def make_writer(style_guide: str):
        @Tool.define
        def approved_terms() -> list[str]:
            \"\"\"Return vocabulary approved for this document.\"\"\"
            return ["handler", "operation", "interpretation"]

        @Skill.define
        def write(topic: str) -> str:
            \"\"\"Write about {topic} under this guide: {style_guide}.

            Call `approved_terms` when choosing terminology.
            \"\"\"

        return write

The Tool and captured guide are reachable only through this returned Skill's
lexical context, subject to the authority caveat above. See
:mod:`effectful.handlers.llm.examples.basics.lexical_scope` and
:mod:`effectful.handlers.llm.examples.reasoning.hanoi`.

.. list-table::
   :header-rows: 1

   * - Binding
     - Described to the model
     - Offered as a Tool with ``explicit`` collection
     - Offered with ``auto`` collection
     - Readable in the REPL
     - Statically nameable by synthesized code
   * - Module-level ``Tool`` or ``Skill``
     - yes
     - yes
     - yes
     - yes
     - yes
   * - True enclosing-local ``Tool`` or ``Skill``
     - yes
     - yes
     - yes
     - yes
     - normally, when its defining source is recoverable
   * - Declared sibling Tool/Skill method on ``self``
     - receiver section and tool advertisement
     - yes
     - yes
     - yes, through ``self``
     - yes
   * - Tool/Skill dynamically assigned to an in-scope Skill owner
     - tool advertisement on a subsequent discovery pass
     - yes
     - yes
     - yes, through that object
     - runtime-callable, but a checker cannot infer an undeclared attribute
   * - Qualifying documented, fully annotated plain function
     - lexical table
     - no
     - yes
     - yes
     - yes if it exists in recovered source
   * - Qualifying plain function dynamically assigned to a Skill owner
     - not in the original table
     - no
     - yes
     - yes
     - generally no without a declared attribute or stub
   * - Current-turn REPL definition
     - visible after it is defined
     - only if it is itself a ``Tool``/``Skill``
     - yes if it qualifies
     - yes for the rest of this turn
     - later snippets are checked with prior snippets, but independently
       synthesized return values do not inherit the REPL prelude
   * - A Tool passed as a Skill argument
     - signature always; value only if interpolated, plus the Tool advertisement
     - yes
     - yes
     - yes
     - available as the enclosing Skill parameter
   * - A Skill-owning object passed as a Skill argument
     - signature always; value only if interpolated; its reachable capabilities
       are advertised
     - the object itself is not a Tool, but its Tools/Skills are
     - the object itself is not a Tool, but its Tools/Skills and qualifying
       methods are
     - yes
     - available as the enclosing Skill parameter

Discovery and invocation are separate stages. ``tool_collection`` chooses what
the extractors in :mod:`~effectful.handlers.llm.harness.legibility.lexical` discover, and runs each model
round, so offered Tools follow mutations of reachable Skill owners even when the
retained system message is stale. ``tool_calling`` chooses how a discovered tool
is invoked, by JSON arguments or a checked Python expression:
:mod:`~effectful.handlers.llm.harness.synthesis.toolcall`.

.. rubric:: Skill ownership and receiver identity

History sharing is determined by receiver identity, not by a base class: a class
that declares a Skill method is registered as an ``Agent`` at class creation,
and a free Skill assigned onto an object later stays free
(:mod:`effectful.handlers.llm.types`). Declare the Skill method on the class
when those semantics matter.

.. rubric:: Growing the harness through ``self``

A model can create a reusable capability in a REPL and attach it to its
receiver. Inside a Skill on an object with a declared ``lessons: dict[str, str]``
field, the following is **model-executed REPL code**, not an application-side
API call::

    from effectful.handlers.llm import Tool


    @Tool.define
    def remember_lesson(topic: str, lesson: str) -> str:
        \"\"\"Record a reusable lesson under a topic and return it.\"\"\"
        self.lessons[topic] = lesson
        return lesson

    self.remember_lesson = remember_lesson

Because discovery recursively inspects in-scope Skill owners, a later round or
turn is offered ``self.remember_lesson``; rebinding it revises the capability
and deleting it retires it. Keep two claims separate: **receiver capability
growth**, which works through runtime reachability, and **typed artifact
vocabulary growth**, which lets separately synthesized Python name and
type-check against the new component and requires a declaration, stub, or
module-visible source. See :mod:`effectful.handlers.llm.examples.reasoning.continual`,
and treat its generated code as untrusted application code.
"""
