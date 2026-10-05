"""Control what survives a Skill turn.

Successful top-level Skill method calls share conversation messages with later
calls on the same receiver. A free Skill starts with fresh history each time.
:mod:`~effectful.handlers.llm.harness.durability.transaction` controls which
turns read and commit history; a failed turn does not commit its messages.

With a harness installed, use the same ordinary receiver object for cross-turn
in-context learning and a
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

.. list-table:: Which history a Skill call uses
   :header-rows: 1

   * - Call
     - History the callee sees
     - New messages
   * - Top-level method on a reused receiver
     - Its prior committed turns
     - Committed on success
   * - Nested method on the same receiver
     - Prior committed turns, not the parent's current messages
     - Discarded; its return goes to the parent as a Tool result
   * - Nested method on a different receiver
     - That receiver's prior committed turns
     - Committed independently on success
   * - Free Skill function
     - Fresh history
     - Discarded after the turn

Conversation history is different from Python object state. Mutating ``self``
or another reachable object can outlive a failed turn. A REPL keeps definitions
within its current turn, but later turns do not automatically inherit them.
:mod:`~effectful.handlers.llm.harness.durability.compaction` shortens message
history without resetting Python objects.

.. list-table:: State lifetime
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

For an independent trial, create a new receiver and mutable dependencies. If
persistence is enabled, use a fresh identity or database too; clearing one field
on a reused receiver leaves its conversation and other state intact.

Use :mod:`~effectful.handlers.llm.harness.durability.persistence` to checkpoint
history and declared dataclass fields to SQLite. Give a receiver a stable
persistent identity to restore it in another process; dynamic attributes and
live functions are not checkpointed. If a stored field should appear in every
request, name it in the Skill docstring, such as ``{self.notes}``, or offer a
Tool to retrieve it.

:mod:`~effectful.handlers.llm.harness.durability.retrying` feeds failed answers
back for another round, and :mod:`~effectful.handlers.llm.harness.durability.truncation`
limits Tool output kept in history. See
:mod:`effectful.handlers.llm.examples.basics.conversation` for shared history.
"""
