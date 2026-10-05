"""Writing idiomatic Effectful LLM programs.

Use this as implementation instructions. The classes in
:mod:`effectful.handlers.llm.types` define the API; this guide says which
abstraction to use, how to compose it, and which state and validation guarantees
the resulting program actually has.

.. rubric:: Mental model

An Effectful LLM program is ordinary Python with one required new kind of
callable:

    A ``Skill`` is a fully annotated Python function or method whose result is
    produced by an LLM instead of by its Python body.

Use ``@Skill.define`` to mark that boundary. The body is intentionally empty: the
docstring states the task, the parameters carry its inputs, and the return
annotation declares the Python type the model must produce::

    from effectful.handlers.llm import Skill


    @Skill.define
    def summarize(text: str) -> str:
        \"\"\"Summarize {text} in one sentence.\"\"\"


    # With an LLM harness installed, this remains an ordinary typed Python call.
    summary: str = summarize("Effect handlers interpret operations.")

The harness is the runtime handler that calls the model and decodes its reply.
From the caller's perspective, ``summarize(...)`` is still a normal Python
expression. That gives the whole programming model its single rule:

    **Each call to a Skill is one typed turn in an LLM conversation.**

During that turn, the model may invoke exposed typed Python callables, called
Tools. A plain Tool executes inside the current turn; a Tool that is itself a
Skill creates a nested Skill turn. Each model response is a **round**. Tool
results and validation errors can lead to another round without creating another
turn::

    Python Skill call  ->  [request -> model round -> Tool/result -> model round
                             -> checked reply]  ->  Python value
                            one conversation turn

The Skill call supplies the request: its docstring, signature, arguments, and
lexical context (globals from its defining module and locals captured from any
enclosing function). A Skill method also has its receiver (the ordinary object
bound as ``self``) and that receiver's prior conversation. The eventual Python
return value is the reply, decoded to the annotated return type. The model may make Tool
calls, execute code, receive validation errors, and retry within the turn. Only
crossing another Skill boundary creates another turn.

Ordinary Python then determines the conversation structure:

- Calling a free Skill, a decorated function rather than a bound method, starts a
  fresh one-turn conversation.
- Calling a bound Skill, a decorated method invoked on an object, selects the
  conversation associated with that receiver. Successful sequential top-level
  turns on that object join that conversation. Another object ordinarily starts
  another conversation; a configured persistence handler can instead restore one
  by stable identity.
- Calling a Skill from inside another Skill creates a complete nested turn. A
  nested turn on the same receiver completes and returns but does not add its
  messages to that receiver's conversation. One on a different receiver can add
  its messages to that receiver's conversation independently. When the model
  invokes it as a Tool, its typed reply still becomes a Tool result in the parent
  turn, which then continues with another model round.

The other harness concepts are consequences of the same rule:

.. list-table::
   :header-rows: 1

   * - Python property of the Skill turn
     - LLM consequence
   * - ``@Skill.define`` on a typed, empty-bodied function
     - the call is implemented as an LLM turn
   * - docstring and formatted arguments
     - the request for this turn
   * - signature and return annotation
     - the typed reply contract
   * - lexical scope
     - the values and capabilities available during the turn
   * - receiver identity
     - which committed conversation a bound turn reads and may extend
   * - receiver fields and other Python objects
     - state that can survive independently of conversation text
   * - call topology
     - whether a bound turn's messages join its receiver's conversation
   * - ordinary control flow around Skill calls
     - sequencing, whole-turn feedback loops, approval, fan-out, and isolation
   * - installed handlers
     - how the turn is documented, executed, checked, retried, rendered, and
       persisted

If a design question seems framework-specific, first restate it in those terms:
*what conversation turn is Python requesting, which receiver owns it, what is in
lexical scope, and what typed value must come back?* The rest of this guide
expands those four questions.

.. rubric:: Build a program from Skill turns

``Skill`` is the only interface an Effectful LLM program must adopt. Start with
an ordinary Python program and decorate only the operations whose implementation
requires a conversation turn with model judgment. Everything else follows from
Python structure:

- a free Skill retains no model conversation between top-level turns;
- a Skill method makes its ordinary receiver object the context-sharing owner;
- object attributes are program-owned state;
- lexical scope determines which values and callables the Skill can reach;
- signatures and annotations determine the values crossing the model boundary;
- ordinary Python controls sequencing, branching, concurrency, and side effects.

``Tool`` and ``Encodable`` are optional refinements for publication and
serialization. Defining a Skill method gives an ordinary class ``Agent``
behavior at runtime; subclass ``Agent`` explicitly only where a statically
typed API requires it. These types do not form a required application object
model.

This is a complete minimal program::

    import dataclasses

    from effectful.handlers.llm import Skill


    @dataclasses.dataclass(frozen=True)
    class Note:
        subject: str
        text: str


    @dataclasses.dataclass(frozen=True)
    class Answer:
        text: str
        sources: list[str]


    @dataclasses.dataclass
    class Assistant:
        \"\"\"Answer questions from trusted local notes.\"\"\"

        tone: str = "concise"
        notes: tuple[Note, ...] = (
            Note("python", "Python uses indentation to delimit blocks."),
            Note("effectful", "Effectful interprets typed operations with handlers."),
        )

        @Skill.define
        def answer(self, question: str) -> Answer:
            \"\"\"Answer in a {self.tone} style.

            Use these notes when relevant: {self.notes}
            Return each used note's subject in `sources`.

            Question: {question}
            \"\"\"


    def main() -> None:
        assistant = Assistant()
        print(assistant.answer("What does Effectful do?"))


    if __name__ == "__main__":
        main()

Run a script under the standard handler stack::

    python -m effectful.handlers.llm.harness assistant.py --render

Pass ``--model`` or set ``EFFECTFUL_LLM_MODEL``; the other flags, including
``-m`` for an installed module, are documented in :mod:`effectful.handlers.llm.harness.__main__`.
Applications can install the same stack directly::

    from effectful.handlers.llm.harness import harness
    from effectful.ops.semantics import handler

    with handler(harness(model="openai/gpt-5-mini")):
        main()

``Assistant`` is an ordinary dataclass. Defining ``answer`` as a Skill method is
enough for the harness to infer per-instance history; reusing the object reuses
its successful conversation. Study
:mod:`effectful.handlers.llm.examples.basics.conversation` for the minimal
pattern, :mod:`effectful.handlers.llm.examples.basics.rag` for optional explicit
Tool publication, and :mod:`effectful.handlers.llm.examples.basics.flight_booking`
for typed handoffs and validation.

.. rubric:: Choose the smallest correct boundary

Because each Skill boundary creates a conversation turn, use it only where the
program needs model judgment. Keep the surrounding mechanics in ordinary Python:

.. list-table::
   :header-rows: 1

   * - Need
     - Use
     - Rule of thumb
   * - deterministic branching, iteration, fan-out, approval, or retrying a
       whole Skill turn
     - ordinary Python
     - keep scheduling and side effects visible to the caller
   * - model judgment with a typed result
     - ``@Skill.define``
     - declare the narrowest useful input and return types; leave the body empty
   * - deterministic capability
     - ordinary function or method
     - place it in lexical scope; use ``Tool.define`` or automatic collection
       only when it should be advertised directly
   * - context shared across successful Skill turns
     - several Skill methods on one reused object
     - identity of the receiver, not a base class, selects the history
   * - independent model context
     - a free Skill or a receiver with a distinct transient/persistent identity
     - pass typed results explicitly; do not rely on nested transcript sharing
   * - persistent program state
     - ordinary declared fields
     - interpolate it for always-visible context or expose a lookup capability
   * - schema or semantic invariant
     - a return type or ``Annotated`` validator
     - reject invalid output before ordinary Python receives it
   * - domain quality signal
     - an evaluator Tool or an explicit Python loop
     - feed the result into a later Skill turn if it should affect revision

Do not turn a fixed workflow into a model decision without a reason. Conversely,
do not encode a semantic judgment as brittle Python merely to avoid a Skill.

.. rubric:: Write prompts as interfaces

Once a function marks a turn, its ordinary Python interface defines the request
and reply:

- A class that owns Skill methods can use its ordinary class docstring for
  standing role and instructions shared by those Skills. It need not inherit a
  library base.
- A Skill docstring contains the per-turn request. Its signature is the
  input/output contract; its body should be empty or raise ``NotHandled``.
- An advertised function's docstring tells the model when and how to call it.
  The implementation remains ordinary Python and must enforce its own
  operational preconditions whether it is explicitly marked ``Tool`` or
  collected automatically.

Skill docstrings are Python format strings. Only values named by active fields
such as ``{question}`` or ``{self.notes}`` are rendered into the turn's user
message. All arguments are nevertheless bound in the turn's REPL when an
executor-enabled harness is installed. Strings render as plain text; other
values use their ``Encodable`` JSON representation, with images as separate
content blocks. A ``str`` return is taken as
the model's prose; any other type is decoded from a JSON schema. Escape literal
braces as ``{{`` and ``}}``, and keep doctest examples constant: a ``>>>``
example cannot contain an active format field. The system message includes the
defining module's source when recoverable, or its docstring as a fallback. A
bound receiver's section also describes its sibling Skills. Runtime values are
rendered into the request only when named by format fields.

.. rubric:: Publish ordinary functions only when needed

Plain Tools give the model capabilities inside a Skill turn; they do not create
their own conversations. A documented, fully annotated ordinary function already
has the shape of such a capability. Keep it in the Skill's lexical scope and
enable automatic collection when it should be offered as a direct call::

    from effectful.handlers.llm import Skill


    def lookup(topic: str) -> list[str]:
        \"\"\"Return trusted notes matching `topic`.\"\"\"
        notes = {"python": "uses indentation", "effectful": "interprets operations"}
        return [text for subject, text in notes.items() if topic.lower() in subject]


    @Skill.define
    def answer(question: str) -> str:
        \"\"\"Answer {question}. Call `lookup` when local notes may help.\"\"\"

::

    python -m effectful.handlers.llm.harness assistant.py --tool-collection auto

With the default ``tool_collection="explicit"``, decorate the same function with
``@Tool.define`` to opt it into direct advertisement. This marker changes
discovery; it does not turn the surrounding program into a separate agent
framework. With an executor-enabled harness the ordinary function also remains
callable as Python in the Skill's lexical REPL context, whether or not it is
advertised.

Automatic collection accepts only functions that look deliberately published;
the heuristic is in :mod:`effectful.handlers.llm.harness.legibility.lexical`.

.. rubric:: Compose workflows in ordinary Python

Skill turns are ordinary typed Python calls, so Python composition defines the
LLM workflow. Keep evaluator loops, approval gates, and scheduling explicit. An
evaluator run after one turn can influence only a later turn; pass its feedback
back deliberately::

    writer = Writer()
    draft = writer.write(request)

    for _ in range(3):
        judgment = judge(request, draft)
        if judgment.accepted:
            break
        draft = writer.revise(request, draft, judgment.feedback)

The repeated ``writer`` instance carries its earlier successful turns, while a
free ``judge`` Skill gets a fresh conversation each time. See
:mod:`effectful.handlers.llm.examples.basics.research_agent` for the full pattern and
:mod:`effectful.handlers.llm.examples.optimization.avo` for an evaluator that the persistent variation
receiver can query before committing a candidate.

Put human approval between proposal and side effect, and keep the execution
capability out of the proposing Skill's reachable Tools::

    proposal = planner.propose_next(task, feedback)
    if approve(proposal):
        result = execute_approved_action(proposal.action, proposal.details)
    else:
        feedback = "Rejected by the user; propose a different action."

See :mod:`effectful.handlers.llm.examples.basics.hitl` for a complete approval loop whose executor is local
to ordinary Python rather than a sibling Tool. Lexical separation prevents
accidental Tool use; for an adversarial authorization boundary, have an external
service validate an approval capability that model-authored Python cannot read
or forge.

Consecutive ``await`` expressions remain sequential. Schedule independent Skill
calls together when true fan-out is intended; each call remains its own turn::

    evaluations = await asyncio.gather(
        *(asyncio.to_thread(evaluate_resume, resume, job) for resume in resumes)
    )
    summary = summarize_evaluations(job, evaluations)

See :mod:`effectful.handlers.llm.examples.basics.map_reduce` for the runnable map/reduce workflow.
``gather`` establishes concurrent scheduling, not state isolation; use distinct
transient receivers or persistent ids when branches must not share committed
context. ``to_thread`` works because the handler stack is a context variable
that threads inherit. Doctest execution rebinds ``doctest`` module globals, so
do not certify synthesized callables concurrently under different handler
stacks.

.. rubric:: Where to read next

1. :mod:`effectful.handlers.llm.types`: ``Skill``, ``Tool``, ``Agent`` and
   ``Encodable``, as the model also sees them.
2. :mod:`effectful.handlers.llm.harness`: how a call runs, the standard stack,
   configuring and debugging it; :mod:`effectful.handlers.llm.harness.hooks` to
   extend it.
3. :mod:`effectful.handlers.llm.harness.legibility`: what the model can see,
   call and reach, and why advertisement is not an authority boundary;
   :mod:`effectful.handlers.llm.harness.legibility.mcp` for MCP tools.
4. :mod:`effectful.handlers.llm.harness.synthesis`: answering with code, and
   what each route certifies.
5. :mod:`effectful.handlers.llm.harness.validation`: validation as a feedback
   architecture.
6. :mod:`effectful.handlers.llm.harness.durability`: state and lifetime,
   transactions, compaction and persistence.
7. :mod:`effectful.handlers.llm.harness.autoreload`: editing a running program.
8. :mod:`effectful.handlers.llm.examples`: composition patterns and complete
   programs by feature. Examples adapted from papers have a separate review
   guide in the documentation.
"""

from .types import *  # noqa: F403, F401
