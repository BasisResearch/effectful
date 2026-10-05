"""Handlers that make a call survive failure.

Message-history accumulation, transactional rollback, tool-output truncation,
retrying on malformed model output, and checkpointing a persisted
:class:`~effectful.handlers.llm.types.Agent` to SQLite.

.. rubric:: State and lifetime

Receiver identity links Skill turns into a conversation, but conversation text
is only one kind of continuity. Several kinds of state have different lifetimes:

.. list-table::
   :header-rows: 1

   * - State
     - Later rounds in this turn
     - Later Skill turn on the same receiver
     - After conversation compaction
     - If the enclosing turn fails
     - New process with ``persist_db``
   * - Current turn's in-flight messages
     - yes
     - visible only after a successful committable bound turn
     - selected messages are removed
     - discarded
     - only the last successful checkpoint
   * - Bound-Skill message history
     - yes
     - yes
     - only messages retained by the selected compaction scope
     - this receiver's transaction is unchanged; an independently successful
       different-receiver turn can commit
     - yes, with a stable persistent id
   * - Plain free-function Skill history
     - yes
     - no; the next turn starts empty
     - only within its current turn
     - discarded
     - no
   * - REPL bindings, imports, and definitions
     - yes
     - not seeded automatically; a stored closure/Tool/Skill can retain the old
       namespace
     - compaction does not clear the current session
     - the session ends, but objects stored elsewhere can keep its namespace
       alive
     - not unless reconstructed through persistent state
   * - Mutations to ordinary in-process objects, including ``self``
     - yes
     - yes while the objects live
     - unaffected
     - **not rolled back**
     - only if explicitly checkpointed or reconstructed
   * - Declared dataclass fields on a persistent Skill owner
     - yes
     - yes
     - unaffected
     - remain mutated; this failed turn is not checkpointed, but an earlier
       successful nested turn may be
     - yes, if picklable and not opted out
   * - Dynamically added receiver attributes
     - yes
     - yes in process
     - unaffected
     - not rolled back in process
     - no; SQLite walks declared dataclass fields, not ``__dict__``
   * - A synthesized callable returned to the caller
     - callable by Python
     - only if the program stores it somewhere durable
     - unaffected
     - unavailable if the Skill never returns it
     - only if the program stores a restart-safe representation

Three consequences are easy to miss:

1. A receiver's shared Skill history is already cross-turn in-context learning.
   An explicit list of records on ``self`` is additional searchable memory, not
   the only learning channel.
2. Compaction rewrites message history and neither resets nor rolls back
   Python state (:mod:`~effectful.handlers.llm.harness.durability.compaction`).
3. Process persistence is narrower than in-process persistence: a checkpoint
   holds history and declared dataclass fields, not dynamic attributes or live
   functions (:mod:`~effectful.handlers.llm.harness.durability.persistence`).

.. rubric:: Choose history by choosing the receiver

Use the same ordinary receiver object for cross-turn in-context learning and a
new transient object, or a distinct stable id under persistence, for an
independent conversation. A free Skill starts a fresh conversation on every
top-level turn::

    from effectful.handlers.llm import Skill


    class Reviewer:
        \"\"\"Review a sequence of related drafts and retain lessons between turns.\"\"\"

        @Skill.define
        def review(self, draft: str) -> str:
            \"\"\"Review {draft}, applying lessons from your earlier reviews.\"\"\"


    @Skill.define
    def isolated_review(draft: str) -> str:
        \"\"\"Review {draft} without a prior conversation.\"\"\"


    reviewer = Reviewer()
    first = reviewer.review("draft one")
    second = reviewer.review("draft two")       # sees the successful first turn
    independent = Reviewer().review("draft two")  # new transient object and conversation
    fresh = isolated_review("draft two")          # fresh on every top-level turn

Put searchable records or stable instructions in declared fields when Python,
not only the model transcript, must inspect them. Merely assigning
``self.notes`` preserves the value but does not render it into every later
prompt. Interpolate ``{self.notes}`` when it should always be visible, or expose
a catalog/search Tool for on-demand access. Study
:mod:`effectful.handlers.llm.examples.basics.conversation` for sequential history,
:mod:`effectful.handlers.llm.examples.basics.research_agent` for feedback across sibling Skills, and
:mod:`effectful.handlers.llm.examples.reasoning.world_model_agent` for an ordinary object with explicit
records and executable learned state.

The mechanisms: :mod:`~effectful.handlers.llm.harness.durability.transaction` (which turns see and commit
which history), :mod:`~effectful.handlers.llm.harness.durability.retrying` (what a rejected attempt leaves
behind), :mod:`~effectful.handlers.llm.harness.durability.truncation` (bounding tool output),
:mod:`~effectful.handlers.llm.harness.durability.compaction` (rewriting the transcript) and
:mod:`~effectful.handlers.llm.harness.durability.persistence` (checkpoints).
"""
