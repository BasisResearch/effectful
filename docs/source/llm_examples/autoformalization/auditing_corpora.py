"""The ClaimCheck benchmark: five Lean corpora, the claims made about them, and
the labelled ground truth the audit in `auditing.py` is scored against.

Nothing in this module may reach an auditing agent. The harness builds a skill's
system prompt partly from the source of the module the skill is *defined* in, so
the unit of exposure is the module: the agents live in `auditing_agents` and
`auditing_naive`, and neither imports this file. The dependency runs the other
way -- this module imports `Verdict` from `auditing_agents` to spell the answer
key -- which is safe in exactly the direction that matters. See the module
docstring of `auditing_agents` for what happened when the corpora did share a
file with them.

Split out of `auditing.py`, which drives the audit, scores it, and holds the
findings; here is only the data it runs on.
"""

import dataclasses
import functools
import pathlib
import re
import sys
import textwrap

from auditing_agents import Verdict

# ---------------------------------------------------------------------------
# The corpora. Five domains, ported item for item from upstream's benchmark
# (`test/integration/claims/*.dfy` for the lemmas,
# `test/integration/mappings/*.json` for the labels): the same 36
# requirement/theorem pairs, the same 27-faithful / 9-planted split, upstream's
# requirement sentences verbatim, and its lemma names transliterated to Lean's
# snake_case. Every one of them compiles (see `--verify`).
#
# Three properties of the Dafny original are load-bearing and are reproduced
# deliberately, because a first attempt at this example invented its own corpus
# and lost all three:
#
#  1. `Inv m` is an opaque atom. Its body is in the corpus and in no prompt, so a
#     faithful theorem of the shape `(h : Inv m) : <one conjunct of Inv>` cannot
#     be checked -- only trusted. That is the judgment anchoring corrupts.
#  2. Several conclusions are *themselves* named predicates the auditor has never
#     seen unfolded (`AllEdgesValid`, `NoDupSeq (AllIds m)`, `ValidColor`,
#     `HuesMatchHarmony`). Upstream's are imported from domain modules that are
#     not even present in its own repository.
#  3. Requirements are vague and un-operationalized -- "Hues follow the selected
#     harmony pattern", not "every hue equals the base plus a fixed offset mod
#     360". Three of the 36 contain a numeral.
#
# The planted flaws are upstream's, and note what they are *not*: not mangled
# conclusions. Seven of the nine are an added `requires`, a dropped `ensures`
# conjunct, or a conclusion compared to itself. Two of them --
# `no_card_duplicates` and `card_partition_no_dups` -- are the *same statement*
# under two different requirements, faithful for one and unfaithful for the
# other, which is the sharpest item in the benchmark and impossible to get right
# by reading the theorem alone.
# ---------------------------------------------------------------------------
COUNTER_CORPUS = r"""import Mathlib

set_option linter.unusedVariables false

namespace Counter

/-- The counter's state. -/
abbrev Model := ℤ

inductive Action where
  | inc
  | dec
  | reset

def Init : Model := 0

def Apply (m : Model) (a : Action) : Model :=
  match a with
  | .inc => m + 1
  | .dec => m - 1
  | .reset => 0

def Normalize (m : Model) : Model := max m 0

def Inv (m : Model) : Prop := 0 ≤ m

theorem counter_non_negative (m : Model) (h : Inv m) : 0 ≤ m := h

theorem init_satisfies_invariant : Inv Init := by
  simp [Inv, Init]

theorem step_preserves_invariant (m : Model) (a : Action) (h : Inv m) :
    Inv (Normalize (Apply m a)) := by
  simp [Inv, Normalize]

theorem dec_at_zero_keeps_zero (m : Model) (h : Inv m) (hz : m = 0) :
    Normalize (Apply m .dec) = 0 := by
  subst hz
  simp [Normalize, Apply]

theorem counter_non_neg_alt (m : Model) (h : Inv m) : m = m := rfl

theorem counter_non_neg_large (m : Model) (h : Inv m) (hb : 100 < m) : 0 ≤ m := h

theorem counter_lower_bound (m : Model) (h : Inv m) : -1 ≤ m :=
  le_trans (by norm_num) (show (0 : ℤ) ≤ m from h)

end Counter
"""
CANON_CORPUS = r"""import Mathlib

set_option linter.unusedVariables false

namespace Canon

abbrev NodeId := ℕ

structure Node where
  id : NodeId
  x : ℤ
  y : ℤ
deriving DecidableEq

structure Edge where
  src : NodeId
  dst : NodeId
deriving DecidableEq

structure Constraint where
  target : NodeId
  kind : ℕ
deriving DecidableEq

structure Model where
  nodes : List Node
  edges : List Edge
  constraints : List Constraint
deriving DecidableEq

def NodeIds (ns : List Node) : List NodeId := ns.map (·.id)

def AllConstraintsValid (cs : List Constraint) (ns : List Node) : Prop :=
  ∀ c ∈ cs, c.target ∈ NodeIds ns

def AllEdgesValid (es : List Edge) (ns : List Node) : Prop :=
  ∀ e ∈ es, e.src ∈ NodeIds ns ∧ e.dst ∈ NodeIds ns

def NoneMatch (cs : List Constraint) (id : NodeId) : Prop :=
  ∀ c ∈ cs, c.target ≠ id

def NoEdgesMention (es : List Edge) (id : NodeId) : Prop :=
  ∀ e ∈ es, e.src ≠ id ∧ e.dst ≠ id

inductive Action where
  | addNode (id : NodeId) (x y : ℤ)
  | removeNode (id : NodeId)

def Apply (m : Model) (a : Action) : Model :=
  match a with
  | .addNode id x y =>
      if id ∈ NodeIds m.nodes then m
      else { m with nodes := ⟨id, x, y⟩ :: m.nodes }
  | .removeNode id =>
      { m with nodes := m.nodes.filter (fun n => n.id != id) }

/-- Drop every constraint and edge that mentions a node the board no longer has. -/
def Normalize (m : Model) : Model :=
  { nodes := m.nodes
    edges := m.edges.filter (fun e =>
      decide (e.src ∈ NodeIds m.nodes) && decide (e.dst ∈ NodeIds m.nodes))
    constraints := m.constraints.filter (fun c => decide (c.target ∈ NodeIds m.nodes)) }

def Inv (m : Model) : Prop :=
  AllConstraintsValid m.constraints m.nodes ∧
  AllEdgesValid m.edges m.nodes ∧
  (NodeIds m.nodes).Nodup

theorem constraint_targets_exist (m : Model) (h : Inv m) :
    AllConstraintsValid m.constraints m.nodes := h.1

theorem edge_endpoints_exist (m : Model) (h : Inv m) :
    AllEdgesValid m.edges m.nodes := h.2.1

theorem add_existing_node_is_noop (m : Model) (id : NodeId) (x y : ℤ) (h : Inv m)
    (hid : id ∈ NodeIds m.nodes) : Apply m (.addNode id x y) = m := by
  simp [Apply, hid]

theorem remove_node_cleans_up (m : Model) (id : NodeId) (h : Inv m)
    (hid : id ∈ NodeIds m.nodes) :
    id ∉ NodeIds (Normalize (Apply m (.removeNode id))).nodes ∧
    NoneMatch (Normalize (Apply m (.removeNode id))).constraints id ∧
    NoEdgesMention (Normalize (Apply m (.removeNode id))).edges id := by
  have hgone : id ∉ NodeIds (Apply m (.removeNode id)).nodes := by
    simp [Apply, NodeIds]
  refine ⟨by simpa [Normalize] using hgone, ?_, ?_⟩
  · intro c hc hct
    simp only [Normalize, List.mem_filter, decide_eq_true_eq] at hc
    exact hgone (hct ▸ hc.2)
  · intro e he
    simp only [Normalize, List.mem_filter, Bool.and_eq_true,
      decide_eq_true_eq] at he
    exact ⟨fun hx => hgone (hx ▸ he.2.1), fun hx => hgone (hx ▸ he.2.2)⟩

theorem remove_node_drops_id (m : Model) (id : NodeId) (h : Inv m)
    (hid : id ∈ NodeIds m.nodes) :
    id ∉ NodeIds (Normalize (Apply m (.removeNode id))).nodes := by
  simp [Normalize, Apply, NodeIds]

theorem constraint_targets_exist_empty (m : Model) (h : Inv m)
    (hc : m.constraints.length = 0) :
    AllConstraintsValid m.constraints m.nodes := h.1

end Canon
"""
COLORWHEEL_CORPUS = r"""import Mathlib

set_option linter.unusedVariables false

namespace ColorWheel

inductive Harmony where
  | analogous
  | complementary
  | triadic
deriving DecidableEq

inductive Mood where
  | custom
  | calm
  | vibrant
deriving DecidableEq

structure Color where
  hue : ℕ
  sat : ℕ
  light : ℕ
deriving DecidableEq

structure Model where
  colors : List Color
  baseHue : ℕ
  harmony : Harmony
  mood : Mood
  contrastPair : ℕ × ℕ

def ValidBaseHue (h : ℕ) : Prop := h < 360

def ValidColor (c : Color) : Prop := c.sat ≤ 100 ∧ c.light ≤ 100

def ColorSatisfiesMood (c : Color) (md : Mood) : Prop :=
  match md with
  | .custom => True
  | .calm => c.sat ≤ 50
  | .vibrant => 50 ≤ c.sat

def HueOffsets : Harmony → List ℕ
  | .analogous => [0, 30, 60, 90, 120]
  | .complementary => [0, 180, 0, 180, 0]
  | .triadic => [0, 120, 240, 120, 240]

def HuesMatchHarmony (cs : List Color) (base : ℕ) (h : Harmony) : Prop :=
  ∀ i, ∀ hi : i < cs.length, (cs.get ⟨i, hi⟩).hue = (base + (HueOffsets h).getD i 0) % 360

def Inv (m : Model) : Prop :=
  m.colors.length = 5 ∧
  ValidBaseHue m.baseHue ∧
  (∀ c ∈ m.colors, ValidColor c) ∧
  (m.contrastPair.1 < 5 ∧ m.contrastPair.2 < 5) ∧
  (m.mood ≠ Mood.custom → ∀ c ∈ m.colors, ColorSatisfiesMood c m.mood) ∧
  HuesMatchHarmony m.colors m.baseHue m.harmony

theorem base_hue_in_range (m : Model) (h : Inv m) : ValidBaseHue m.baseHue := h.2.1

theorem always_five_colors (m : Model) (h : Inv m) : m.colors.length = 5 := h.1

theorem all_colors_valid (m : Model) (h : Inv m) : ∀ c ∈ m.colors, ValidColor c :=
  h.2.2.1

theorem contrast_pair_indices_valid (m : Model) (h : Inv m) :
    (0 ≤ m.contrastPair.1 ∧ m.contrastPair.1 < 5) ∧
    (0 ≤ m.contrastPair.2 ∧ m.contrastPair.2 < 5) :=
  ⟨⟨Nat.zero_le _, h.2.2.2.1.1⟩, ⟨Nat.zero_le _, h.2.2.2.1.2⟩⟩

theorem mood_constraints_satisfied (m : Model) (h : Inv m) (hm : m.mood ≠ Mood.custom) :
    ∀ c ∈ m.colors, ColorSatisfiesMood c m.mood := h.2.2.2.2.1 hm

theorem hues_follow_harmony (m : Model) (h : Inv m) :
    HuesMatchHarmony m.colors m.baseHue m.harmony := h.2.2.2.2.2

theorem palette_non_empty (m : Model) (h : Inv m) : 1 ≤ m.colors.length := by
  have := h.1
  omega

end ColorWheel
"""
DELEGATION_CORPUS = r"""import Mathlib

set_option linter.unusedVariables false

namespace DelegationAuth

abbrev Subject := ℕ
abbrev Capability := ℕ
abbrev EdgeId := ℕ

/-- One delegation edge: `frm` lets `dst` use `cap`. -/
structure Edge where
  id : EdgeId
  frm : Subject
  dst : Subject
  cap : Capability

structure Model where
  subjects : List Subject
  grants : List (Subject × Capability)
  delegations : List Edge
  nextEdge : EdgeId

def Init : Model := ⟨[], [], [], 0⟩

inductive Action where
  | grant (s : Subject) (c : Capability)
  | delegate (frm dst : Subject) (c : Capability)
  | revoke (e : EdgeId)

def Apply (m : Model) (a : Action) : Model :=
  match a with
  | .grant s c =>
      if s ∈ m.subjects then { m with grants := (s, c) :: m.grants } else m
  | .delegate f t c =>
      if f ∈ m.subjects ∧ t ∈ m.subjects then
        { m with
          delegations := ⟨m.nextEdge, f, t, c⟩ :: m.delegations
          nextEdge := m.nextEdge + 1 }
      else m
  | .revoke e =>
      if e ∈ m.delegations.map (·.id) then
        { m with delegations := m.delegations.filter (fun ed => ed.id != e) }
      else m

def Inv (m : Model) : Prop :=
  (∀ sc ∈ m.grants, sc.1 ∈ m.subjects) ∧
  (∀ ed ∈ m.delegations, ed.frm ∈ m.subjects ∧ ed.dst ∈ m.subjects) ∧
  (∀ ed ∈ m.delegations, ed.id < m.nextEdge)

theorem grant_subjects_exist (m : Model) (h : Inv m) :
    ∀ sc ∈ m.grants, sc.1 ∈ m.subjects := h.1

theorem delegation_endpoints_exist (m : Model) (h : Inv m) :
    ∀ ed ∈ m.delegations, ed.frm ∈ m.subjects ∧ ed.dst ∈ m.subjects := h.2.1

theorem edge_ids_fresh (m : Model) (h : Inv m) :
    ∀ ed ∈ m.delegations, ed.id < m.nextEdge := h.2.2

theorem grant_non_existent_is_noop (m : Model) (s : Subject) (c : Capability)
    (h : Inv m) (hs : s ∉ m.subjects) : Apply m (.grant s c) = m := by
  simp [Apply, hs]

theorem delegate_non_existent_is_noop (m : Model) (f t : Subject) (c : Capability)
    (h : Inv m) (hs : ¬(f ∈ m.subjects ∧ t ∈ m.subjects)) :
    Apply m (.delegate f t c) = m := by
  simp [Apply, hs]

theorem revoke_non_existent_is_noop (m : Model) (e : EdgeId) (h : Inv m)
    (he : e ∉ m.delegations.map (·.id)) : Apply m (.revoke e) = m := by
  simp [Apply, he]

theorem grant_non_existent_is_noop_init (m : Model) (s : Subject) (c : Capability)
    (h : Inv m) (hinit : m = Init) (hs : s ∉ m.subjects) :
    Apply m (.grant s c) = m := by
  simp [Apply, hs]

end DelegationAuth
"""
KANBAN_CORPUS = r"""import Mathlib

set_option linter.unusedVariables false

namespace Kanban

abbrev CardId := ℕ
abbrev ColId := ℕ

structure Model where
  cols : List ColId
  cards : List CardId
  lanes : List (ColId × List CardId)
  wip : List (ColId × ℕ)
  nextId : CardId

def Keys {α : Type} (l : List (ColId × α)) : List ColId := l.map (·.1)

def AllIds (m : Model) : List CardId := (m.lanes.map (·.2)).flatten

def NoDupSeq (l : List CardId) : Prop := l.Nodup

def OccursInLanes (m : Model) (id : CardId) : Prop := ∃ e ∈ m.lanes, id ∈ e.2

def LaneLen (m : Model) (k : ColId) : ℕ :=
  (((m.lanes.find? (fun e => e.1 == k)).map (·.2)).getD []).length

def WipOf (m : Model) (k : ColId) : ℕ :=
  ((m.wip.find? (fun e => e.1 == k)).map (·.2)).getD 0

inductive Action where
  | addCard (col : ColId)
  | moveCard (id : CardId) (toCol : ColId)

def pushInto (lanes : List (ColId × List CardId)) (k : ColId) (id : CardId) :
    List (ColId × List CardId) :=
  lanes.map (fun e => if e.1 == k then (e.1, id :: e.2) else e)

def dropFrom (lanes : List (ColId × List CardId)) (id : CardId) :
    List (ColId × List CardId) :=
  lanes.map (fun e => (e.1, e.2.filter (fun x => x != id)))

def Apply (m : Model) (a : Action) : Model :=
  match a with
  | .addCard col =>
      if col ∈ m.cols ∧ LaneLen m col < WipOf m col then
        { m with
          cards := m.nextId :: m.cards
          lanes := pushInto m.lanes col m.nextId
          nextId := m.nextId + 1 }
      else m
  | .moveCard id toCol =>
      if toCol ∈ m.cols ∧ LaneLen m toCol < WipOf m toCol then
        { m with lanes := pushInto (dropFrom m.lanes id) toCol id }
      else m

def Normalize (m : Model) : Model :=
  { m with lanes := m.lanes.filter (fun e => decide (e.1 ∈ m.cols)) }

def Inv (m : Model) : Prop :=
  m.cols.Nodup ∧
  NoDupSeq (AllIds m) ∧
  (∀ id, id ∈ m.cards ↔ OccursInLanes m id) ∧
  (Keys m.lanes = m.cols ∧ Keys m.wip = m.cols) ∧
  (∀ k ∈ m.cols, LaneLen m k ≤ WipOf m k) ∧
  (∀ id ∈ m.cards, id < m.nextId)

theorem columns_are_unique (m : Model) (h : Inv m) : NoDupSeq m.cols := h.1

theorem card_in_exactly_one_column (m : Model) (h : Inv m) :
    NoDupSeq (AllIds m) ∧ ∀ id, id ∈ m.cards ↔ OccursInLanes m id :=
  ⟨h.2.1, h.2.2.1⟩

theorem no_card_duplicates (m : Model) (h : Inv m) : NoDupSeq (AllIds m) := h.2.1

theorem wip_limits_respected (m : Model) (h : Inv m) :
    ∀ k ∈ m.cols, LaneLen m k ≤ WipOf m k := h.2.2.2.2.1

theorem add_card_to_full_column_is_noop (m : Model) (col : ColId) (h : Inv m)
    (hc : col ∈ m.cols) (hfull : WipOf m col ≤ LaneLen m col) :
    Apply m (.addCard col) = m := by
  have hneg : ¬(col ∈ m.cols ∧ LaneLen m col < WipOf m col) := by
    rintro ⟨-, hlt⟩
    omega
  simp [Apply, hneg]

theorem allocator_always_fresh (m : Model) (h : Inv m) :
    ∀ id ∈ m.cards, id < m.nextId := h.2.2.2.2.2

theorem lanes_and_wip_match_columns (m : Model) (h : Inv m) :
    Keys m.lanes = m.cols ∧ Keys m.wip = m.cols := h.2.2.2.1

theorem move_card_preserves_total (m : Model) (id : CardId) (toCol : ColId)
    (h : Inv m) :
    (AllIds (Normalize (Apply m (.moveCard id toCol)))).length =
      (AllIds (Normalize (Apply m (.moveCard id toCol)))).length := rfl

theorem card_partition_no_dups (m : Model) (h : Inv m) : NoDupSeq (AllIds m) := h.2.1

end Kanban
"""


# ---------------------------------------------------------------------------
# The mapping: which theorem was written to formalize which requirement,
# plus the ground truth. Upstream's `test/integration/mappings/*.json`
# carry exactly these `expected`/`reason` labels, which is what makes the
# audit scoreable rather than merely demonstrable. Requirement strings are
# copied from those files unchanged, deliberately: their vagueness is the
# variable under test.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Claim:
    """A requirement, the theorem said to formalize it, and the labelled truth.

    ``expected`` and ``why`` are the answer key. They are Python-side bookkeeping
    and must stay that way: they are never passed to a skill, and this module
    is never imported by the one the agents are defined in.
    """

    requirement: str
    theorem: str
    expected: Verdict
    why: str = ""


COUNTER_CLAIMS: tuple[Claim, ...] = (
    # Upstream's `counter`. The state is an `Int`, not a `Nat`, which is what
    # makes `0 <= m` a real claim and `-1 <= m` a real weakening -- the trap
    # would be unavailable over the naturals. `counter_non_negative` is the
    # item that separates the strategies most often upstream: its conclusion
    # *is* the invariant it assumes, so an auditor who has been told what the
    # requirement wants can talk itself into calling it circular.
    Claim(
        "The counter value is always non-negative",
        "counter_non_negative",
        Verdict.CONFIRMED,
    ),
    Claim(
        "The initial state satisfies the invariant",
        "init_satisfies_invariant",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Every action preserves the invariant after normalization",
        "step_preserves_invariant",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Decrementing at zero keeps the counter at zero",
        "dec_at_zero_keeps_zero",
        Verdict.CONFIRMED,
    ),
    Claim(
        "The counter value is always non-negative",
        "counter_non_neg_alt",
        Verdict.DISPUTED,
        "tautology: `m = m` proves nothing about non-negativity",
    ),
    Claim(
        "The counter value is always non-negative",
        "counter_non_neg_large",
        Verdict.DISPUTED,
        "vacuous precondition: `100 < m` narrows scope, doesn't cover m = 0",
    ),
    Claim(
        "The counter value is always non-negative",
        "counter_lower_bound",
        Verdict.DISPUTED,
        "weakened postcondition: concludes `-1 <= m` instead of `0 <= m`",
    ),
)

CANON_CLAIMS: tuple[Claim, ...] = (
    # Upstream's `canon`. Two conclusions are opaque predicates
    # (`AllConstraintsValid`, `AllEdgesValid`); `remove_node_cleans_up` is
    # the three-conjunct conclusion whose first conjunct alone is
    # `remove_node_drops_id`.
    Claim(
        "All constraint targets reference existing nodes",
        "constraint_targets_exist",
        Verdict.CONFIRMED,
    ),
    Claim(
        "All edge endpoints reference existing nodes",
        "edge_endpoints_exist",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Adding a node with an existing ID is a no-op",
        "add_existing_node_is_noop",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Removing a node cleans up related constraints and edges",
        "remove_node_cleans_up",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Removing a node cleans up related constraints and edges",
        "remove_node_drops_id",
        Verdict.DISPUTED,
        "missing conjunct: only checks the node is removed, doesn't verify "
        "constraint/edge cleanup",
    ),
    Claim(
        "All constraint targets reference existing nodes",
        "constraint_targets_exist_empty",
        Verdict.DISPUTED,
        "vacuous precondition: requiring no constraints makes the conclusion trivially "
        "true",
    ),
)

COLORWHEEL_CLAIMS: tuple[Claim, ...] = (
    # Upstream's `colorwheel`. The domain where the weakest arm falls apart
    # upstream, and the split is legible: it confirms the two theorems whose
    # conclusions are visible arithmetic (`always_five_colors`,
    # `contrast_pair_indices_valid`) and disputes the four whose conclusions
    # are named predicates it has never seen unfolded.
    Claim(
        "The base hue is always in valid range",
        "base_hue_in_range",
        Verdict.CONFIRMED,
    ),
    Claim(
        "There are always exactly 5 colors in the palette",
        "always_five_colors",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Every color has valid saturation and lightness values",
        "all_colors_valid",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Contrast pair indices are valid (between 0 and 4)",
        "contrast_pair_indices_valid",
        Verdict.CONFIRMED,
    ),
    Claim(
        "When a mood is set (not Custom), all colors satisfy the mood constraints",
        "mood_constraints_satisfied",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Hues follow the selected harmony pattern",
        "hues_follow_harmony",
        Verdict.CONFIRMED,
    ),
    Claim(
        "There are always exactly 5 colors in the palette",
        "palette_non_empty",
        Verdict.DISPUTED,
        "weakened postcondition: concludes the palette is non-empty instead of exactly "
        "5",
    ),
)

DELEGATION_CLAIMS: tuple[Claim, ...] = (
    # Upstream's `delegation-auth`. `delegate_non_existent_is_noop` is one of
    # only two items every single-call arm gets wrong on every model upstream
    # tried: the requirement reads as 'both subjects missing' and the
    # hypothesis says 'at least one missing', so the theorem is *stronger*
    # than what was asked -- which still counts as expressing it.
    Claim(
        "All granted capabilities reference existing subjects",
        "grant_subjects_exist",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Delegation endpoints (from, to) must be existing subjects",
        "delegation_endpoints_exist",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Edge IDs are always less than the next allocator (freshness)",
        "edge_ids_fresh",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Granting a capability to a non-existent subject is a no-op",
        "grant_non_existent_is_noop",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Delegating between non-existent subjects is a no-op",
        "delegate_non_existent_is_noop",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Revoking a non-existent delegation is a no-op",
        "revoke_non_existent_is_noop",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Granting a capability to a non-existent subject is a no-op",
        "grant_non_existent_is_noop_init",
        Verdict.DISPUTED,
        "vacuous precondition: `m = Init` restricts the claim to the empty policy only",
    ),
)

KANBAN_CLAIMS: tuple[Claim, ...] = (
    # Upstream's `kanban`, and the sharpest pair in the benchmark:
    # `no_card_duplicates` and `card_partition_no_dups` have identical
    # statements. Which one is faithful depends entirely on the requirement
    # it is set against, so no amount of reading the Lean decides it.
    Claim(
        "Column names are unique (no duplicate columns)",
        "columns_are_unique",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Every card appears in exactly one column (exact partition)",
        "card_in_exactly_one_column",
        Verdict.CONFIRMED,
    ),
    Claim(
        "No card ID appears twice across all lanes (no duplicates)",
        "no_card_duplicates",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Each column respects its WIP limit (number of cards does not exceed the "
        "limit)",
        "wip_limits_respected",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Adding a card to a full column is a no-op",
        "add_card_to_full_column_is_noop",
        Verdict.CONFIRMED,
    ),
    Claim(
        "The card allocator is always fresh (no allocated ID reused)",
        "allocator_always_fresh",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Lanes and WIP maps are defined exactly for existing columns",
        "lanes_and_wip_match_columns",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Moving a card preserves the total number of cards",
        "move_card_preserves_total",
        Verdict.DISPUTED,
        "tautology: compares one expression to itself",
    ),
    Claim(
        "Every card appears in exactly one column (exact partition)",
        "card_partition_no_dups",
        Verdict.DISPUTED,
        "missing conjunct: proves only that IDs are distinct, not the bidirectional "
        "membership that makes it a partition",
    ),
)


@dataclasses.dataclass(frozen=True)
class Domain:
    """One body of Lean and the claims made about it."""

    name: str
    corpus: str
    claims: tuple[Claim, ...]

    @functools.cached_property
    def theorems(self) -> list[str]:
        """The names of the theorems this corpus proves, in source order."""
        return re.findall(r"^theorem (\w+)", self.corpus, re.MULTILINE)

    def statement_of(self, qualified: str) -> str:
        """Extract the *statement* of ``<namespace>.<theorem>`` from this corpus:
        the text from ``theorem <name>`` up to the ``:=`` that begins its proof.

        Only this crosses the model boundary. The proof is dropped because
        ClaimCheck assumes it correct and audits the claim, and the enclosing
        namespace is dropped because it says which version of the file a theorem
        came from -- which the auditor is precisely not entitled to know.
        """
        namespace, _, name = qualified.rpartition(".")
        section = self.corpus
        if namespace:
            start = section.index(f"namespace {namespace}")
            end = section.index(f"end {namespace}", start)
            section = section[start:end]
        match = re.search(rf"^theorem {re.escape(name)}\b", section, re.MULTILINE)
        if match is None:
            raise KeyError(f"no theorem {qualified!r} in the corpus")
        # The proof begins at the first `:=` at or after the statement; no
        # statement in this corpus contains one, so the first occurrence is the
        # right one.
        body = section[match.start() :]
        # The proof begins at the first `:=`; no statement in this corpus contains
        # one. Guard it anyway -- a future statement with a `let` or a structure
        # literal would otherwise be truncated mid-way and sent as a fragment,
        # which is a wrong answer rather than an error.
        statement = textwrap.dedent(body[: body.index(":=")]).strip()
        if statement.count("(") != statement.count(")"):
            raise ValueError(
                f"extracting {qualified!r} cut an unbalanced statement at the first "
                f"`:=`; it probably contains one inside the statement:\n{statement}"
            )
        return statement

    @functools.cached_property
    def verify_corpus(self) -> bool:
        """Compile this corpus with Lean, and say whether it came out proved.

        The premise, checked: ClaimCheck is only interesting if the formal
        artifacts really are proved -- otherwise a disputed theorem might just be
        a broken one. `formalization.py` (LEAP) already drives a real Lean 4 +
        Mathlib toolchain, so this reuses its kernel rather than restating it,
        imported inside the property as `world_model_agent.py` imports
        `gridworlds`, so the example carries no Lean dependency unless the check
        is asked for.

        Cached because importing Mathlib is by far the slowest thing this example
        does, and a domain may be asked to verify more than once in a process.
        ``False`` means the toolchain is not built; a corpus that *fails* raises
        instead, since an unproved theorem makes every verdict about it
        meaningless.
        """
        # The examples are importable as ``docs.source.llm_examples...`` from the
        # repository root, which is on ``sys.path`` under the harness but not when
        # `auditing.py` is run directly; add it so both invocations work.
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
        from docs.source.llm_examples.autoformalization.formalization import (
            _SORRY,
            LeanKernel,
        )

        kernel = LeanKernel()
        if not kernel.available():
            print(
                f"Lean project not built at {kernel.project!r}; skipping "
                "verification.\nBuild it once (see formalization.py "
                "--check-toolchain):\n"
                "  elan default stable\n"
                f"  cd {kernel.project} && lake exe cache get && lake build"
            )
            return False

        print(
            f"Compiling {self.name} ({len(self.theorems)} theorems) with "
            "Lean 4 + Mathlib ..."
        )
        result = kernel.compile(self.corpus)
        if not result.ok:
            raise SystemExit(
                f"The {self.name} corpus does not compile:\n{result.messages}"
            )
        if _SORRY.search(self.corpus):
            raise SystemExit(
                f"The {self.name} corpus contains `sorry`; it is not proved."
            )
        return True


DOMAINS: dict[str, Domain] = {
    "counter": Domain("counter", COUNTER_CORPUS, COUNTER_CLAIMS),
    "canon": Domain("canon", CANON_CORPUS, CANON_CLAIMS),
    "colorwheel": Domain("colorwheel", COLORWHEEL_CORPUS, COLORWHEEL_CLAIMS),
    "delegation": Domain("delegation", DELEGATION_CORPUS, DELEGATION_CLAIMS),
    "kanban": Domain("kanban", KANBAN_CORPUS, KANBAN_CLAIMS),
}
