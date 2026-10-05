"""Build the context and Tool list shown to a Skill's model.

A Skill's *lexical scope* is its defining module's globals, any enclosing
function's locals, and, for a method, its receiver (``self``). The harness uses
that scope to describe available names and discover Tools. A ``{name}`` field
in the Skill docstring inserts that value into the current request. The
:mod:`~effectful.handlers.llm.harness.legibility.lexical` module implements
scope and Tool discovery; :mod:`~effectful.handlers.llm.harness.legibility.framework`
adds library API documentation to the system prompt.

``tool_collection="explicit"`` advertises in-scope ``Tool`` and ``Skill``
values, even when their Python names begin with an underscore. ``"auto"`` also
advertises qualifying public, documented, annotated functions; it does not
implicitly publish private plain functions.
The Tool list is refreshed each model round, so a later round can discover a
Tool added to a reachable receiver. :mod:`~effectful.handlers.llm.harness.legibility.mcp`
adds Tools from configured MCP servers.

An advertised Tool is a convenient call interface, not a limit on what
model-authored Python can access. With the builtin executor, code can use
reachable Python objects even if they were never advertised as Tools. See
:mod:`~effectful.handlers.llm.harness.execution` for execution policy.

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

``tool_collection`` selects what is discovered; ``tool_calling`` selects how a
discovered Tool is invoked (JSON arguments or a checked Python expression).
Neither setting grants execution authority. A Tool added to a receiver at
runtime may be advertised and callable through that object, but static checking
cannot name an undeclared attribute; declare it when generated code must type
check against the receiver's source.

For a small, explicit scope, define a Skill and its Tools inside a factory
function::

    from effectful.handlers.llm import Skill, Tool

    def make_writer(style_guide: str):
        @Tool.define
        def approved_terms() -> list[str]:
            \"\"\"Return the approved vocabulary.\"\"\"
            return ["handler", "operation"]

        @Skill.define
        def write(topic: str) -> str:
            \"\"\"Write about {topic} using {style_guide}.\"\"\"

        return write

The returned Skill can reach the captured guide and Tool. See
:mod:`effectful.handlers.llm.examples.basics.lexical_scope`.

For automatic collection, a plain function can be offered without
``@Tool.define`` when it is public, documented, and fully annotated::

    from effectful.handlers.llm import Skill

    def lookup(topic: str) -> list[str]:
        \"\"\"Return trusted notes matching a topic.\"\"\"
        return ["A local note about " + topic]

    @Skill.define
    def answer(question: str) -> str:
        \"\"\"Answer {question}; call `lookup` when notes may help.\"\"\"

Run with ``harness(tool_collection="auto")``. With ``"explicit"`` (the
default), decorate ``lookup`` with ``@Tool.define`` to advertise it directly.

A model can add a Tool to a reachable receiver during a REPL turn. For a
receiver with a declared ``lessons: dict[str, str]`` field, this is
**model-executed REPL code**::

    from effectful.handlers.llm import Tool

    @Tool.define
    def remember_lesson(topic: str, lesson: str) -> str:
        \"\"\"Store a lesson under a topic.\"\"\"
        self.lessons[topic] = lesson
        return lesson

    self.remember_lesson = remember_lesson

The next discovery pass can advertise ``self.remember_lesson``. A Tool added
this way is callable at runtime, but generated code cannot type-check an
undeclared attribute. A Skill assigned to ``self`` this way also remains a
free Skill: declare it as a class method when it must share receiver history.
See :mod:`effectful.handlers.llm.examples.reasoning.continual`.
"""
