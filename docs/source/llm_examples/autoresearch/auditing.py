"""ClaimCheck: auditing whether a proved theorem is the theorem you meant.

Implements the core of ClaimCheck ("Narrowing the Gap Between Proof and Intent",
https://midspiral.com/blog/claimcheck-narrowing-the-gap-between-proof-and-intent/,
reference implementation at https://github.com/metareflection/claimcheck, MIT).
Its diagnosis is that a verifier proves code matches its *specification* and says
nothing about whether the specification matches your *intent*. The motivating case
is a Dafny election tally whose ``TallyMonotonic`` lemma was supposed to say
"adding a ballot can't decrease a tally" and in fact said ``Count(...) >= 0`` --
trivially true, since counts are naturals. Dafny reported 14 verified, 0 errors.
The proof was correct and the guarantee was not the one anybody wanted.

Its fix is *round-trip informalization*. Pass 1 translates the formal statement,
**and nothing else**, back into English; pass 2 compares that back-translation
against the requirement it was meant to formalize and classifies any divergence.
The load-bearing part is what pass 1 does not get: having never seen the
requirement, it cannot parrot it back, so agreement in pass 2 is evidence and not
an echo. The blog measures this -- across 5 domains and 36 requirement/lemma pairs,
the two-pass split scored 96.3% against a single combined prompt's 86.1%.

That is a claim about *scope*, which is what makes it an effectful example. The
reference implementation enforces it by hand-assembling prompt strings, under a
comment reading ``CRITICAL: This prompt must NOT include the original
requirements``. Here the comment is a signature:

  * Blindness by construction -- but the construction is the *module*, not the
    signature. ``Informalizer.informalize`` takes one argument, a Lean statement,
    and its ``Agent`` history holds no turn in which a requirement appeared. That
    is necessary and it is not sufficient: the harness puts the source of the
    template's defining module into the system prompt, so the agents live in
    ``auditing_agents.py`` and the corpora, the mapping and the expected verdicts
    live here, in a module they never import. Getting this wrong the first time
    is what ``The leak`` below is about. The instruction "do not look at the
    requirement" is still absent from the prompt -- it just takes a file boundary
    rather than a parameter list to make that safe.

  * ...and the ablation is visible in the signature too. ``--strategy
    single-pass`` asks one template to informalize-then-compare, so
    ``requirement`` is a parameter of the call doing the analysis; ``--strategy
    naive`` just asks "does this match?". What separates the three is precisely
    what is in scope for the model at the moment it commits to a reading of the
    formal statement -- a difference in the type signature, not in the wording.

    Measured 2026-07 over the full 40-claim corpus, the lower three rows at
    temperature 0 to match upstream's benchmark (gpt-5.5 is a reasoning model and
    rejects temperature 0, so it runs at the provider default):

    ==============  ========  ===========  =====
    model           two-pass  single-pass  naive
    ==============  ========  ===========  =====
    gpt-5.5            95.0%        95.0%  95.0%
    gpt-4.1            85.0%        90.0%  85.0%
    gpt-4o             85.0%        70.0%  82.5%
    gpt-4.1-mini       72.5%        70.0%  70.0%
    ==============  ========  ===========  =====

    Two-pass is ahead of naive in two of four models and never behind it; ahead
    of single-pass in two, tied in one, and 5pp behind in one. The one large
    effect -- gpt-4o, two-pass over single-pass, +15pp, six items -- is in the
    predicted direction and bigger than upstream's 10.2pp headline. But it is one
    model in one run, and the model directly above it in capability goes the
    other way. Four models is not enough to call this, and n=40 at one run per
    cell has roughly a 13% chance of detecting an 8pp effect even if real.

    **An earlier version of this table was wrong, and how it was wrong is worth
    more than the table.** See ``The leak`` below.

  * Coherent verdicts by construction. A ``Comparison`` certifies at decode time
    that ``match`` and ``weakening`` agree and that a mismatch names its
    discrepancy; an incoherent verdict raises and ``RetryLLMHandler`` feeds it
    back. Upstream's JSON schema admits ``match: true`` alongside
    ``weakeningType: "tautology"``, and nothing catches it.

  * The deterministic half stays in Python. The blog's between-pass checks --
    flag a back-translation rated ``trivial``, flag two requirements whose
    theorems have the same conclusion -- are a loop over typed
    ``Informalization`` values, not a model call.

  * The premise is checked by a real prover. Under ``--verify`` both corpora are
    compiled by the Lean 4 + Mathlib toolchain ``formalization.py`` already
    shells out to. It reports 39 theorems, 0 errors, no ``sorry`` -- and then the
    audit finds eighteen claims that do not mean what they were written to mean.
    The verifier's clean bill of health is the setup, not the punchline.

Demonstrates:
- Structural separation as *lexical scope*: an agent that cannot see a value
  because its template's signature has no parameter for it, contrasted against
  two ablations that can
- Decode-time certification of a structured verdict's internal coherence, turning
  a self-contradictory answer into a `RetryLLMHandler` retry
- Reuse of a sibling example's real external verifier (`formalization.py`'s
  `LeanKernel`) to establish a premise, rather than asserting it
- Two labelled corpora and an accuracy report that separates the two error
  directions -- which is what shows that every strategy here errs only ever by
  over-flagging, the same asymmetry upstream's own eval outputs record
- Fan-out over independent audits with ``asyncio.gather`` + ``asyncio.to_thread``
- Per-field guidance carried on the types as ``field(metadata={"description": ...})``
"""

# Simplifications vs. the source:
# - Lean 4 + Mathlib, not Dafny. The election corpus is a re-authored port of
#   upstream's demo (`demo/election0.dfy`, `demo/election.dfy`) plus planted
#   variants adapted from its `counter` test domain; the delegation corpus is
#   modelled on the shape of its `delegation-auth` domain. Lean was chosen
#   because a real, already-built toolchain is reachable from this repo, so
#   "every one of these theorems is proved" is a compile and not a claim. The
#   mapping is close: Dafny's `requires`/`ensures` split becomes hypotheses and
#   conclusion, and `nat` becoming `ℕ` preserves the original tautology traps
#   (`0 ≤ count ...` is as vacuous in Lean as `Count(...) >= 0` was in Dafny).
# - Statements are sent, proofs are not. Upstream sends the Dafny lemma body
#   along with its contract; here only the statement crosses the boundary, as
#   upstream's own `lemmascript` preset does ("only the signature + requires +
#   ensures is sent, never a body"). In Lean the meaning is entirely in the
#   statement, and ClaimCheck explicitly does not audit the proof.
# - Two domains, one model. The blog runs 5 domains and splits the passes across
#   two models (Haiku informalizes, Sonnet compares), reporting that the weaker
#   informalizer is sufficient. Both passes here run on whatever model the
#   harness was given, so that model-asymmetry result is *not* reproduced.
# - One call per claim, not one batched call per pass. Upstream batches every
#   lemma into a single request for throughput; auditing each claim separately
#   keeps the blindness argument obvious and lets the fan-out be the demo. This
#   also removes a confound -- see below.

# The leak
# --------
# The first version of this example defined the agents in *this* file, beside the
# corpora and the labelled mapping. The harness assembles a template's system
# prompt partly from the source of the module the template is defined in (the
# prompt-assembly table in `effectful.handlers.llm.types.Template` documents this
# plainly), so every agent -- including the informalizer whose whole job is not to
# know -- received all 40 `Claim(...)` entries with their expected verdicts and
# the written reason for each, both Lean corpora with the `Draft`/`Audited`
# namespaces that `statement_of` exists to strip, and the body of `Wf`, the
# invariant that is supposed to arrive as an opaque atom. A dump measured 78,909
# bytes of system prompt, 81% of it this file.
#
# The models used it. In those runs they returned discrepancy text matching the
# answer key's `why` strings word for word, including cases where the key for a
# *sibling* theorem was pasted onto a theorem that had no hypothesis at all. The
# whole first table -- and the finding drawn from it, that the error shape
# reproduced upstream's -- was an artifact. Both are withdrawn.
#
# Two things make this worth keeping in the file rather than quietly fixing.
# First, the design claim in the header ("nothing in its lexical scope can fetch
# one") was false in a way no amount of reading the prompts would reveal: the
# blindness lived in the signature, and the framework serialises the module. In
# this framework **the module is the confidentiality boundary**, which is why the
# agents now live in `auditing_agents.py` and this file is never imported by
# them. Any scored example that carries its own answer key needs the same split.
# Second, the corrected numbers are materially different in both directions:
# accuracies fell across the board (gpt-5.5 from 100% to 95%), and the ablation
# that showed nothing now shows two-pass ahead in most cells.
#
# What did *not* survive the fix is the error-direction finding. With the key
# visible, every strategy over-flagged; with it gone, the six runs on the two
# weaker models produce 49 unfaithful theorems waved through against 5 faithful
# ones disputed. That is the reverse of upstream, which records essentially no
# false confirms at all. The honest reading is that this Lean corpus and
# upstream's Dafny one fail in opposite directions: its planted flaws were all
# caught by every variant and the contest was over false alarms, whereas these
# planted flaws are genuinely hard for a mid-capability model to catch. That is a
# difference in corpus, not a difference in architecture, and it means the
# mechanism upstream's benchmark actually measures is not the one measured here.
#
# Notes on upstream's own numbers, which still stand:
#
# - The traps were never the discriminator, upstream's included. Reading the
#   reference implementation's own per-item eval outputs (`eval/results/*.json`):
#   across 446 recorded judgments there are three false confirms in total, all
#   from one misconfiguration. Every other error in every mode is an *over-flag
#   of a faithful lemma*. Upstream states it plainly in
#   `reports/STRUCTURAL-SEPARATION.md`: "All three variants catch all 8
#   deliberately bogus lemmas (100%). The accuracy difference comes entirely from
#   false disputes of valid lemmas -- structural separation reduces false
#   positives." So a corpus whose unfaithful theorems are all caught by all three
#   strategies is reproducing upstream's result, not failing to.
#
# - What actually discriminates there is the *invariant projection*: a faithful
#   lemma of the form `requires Inv(m); ensures <one conjunct of Inv>`, whose
#   conclusion is textually already inside its own hypothesis. 18 of upstream's
#   27 faithful pairs are this shape. A single-call model, having been shown the
#   requirement, infers what `Inv` must contain and rules the lemma vacuous; the
#   blind informalizer sees `Inv(m)` as an opaque atom and cannot form that
#   hypothesis at all. The DELEGATION_CLAIMS below include this shape (`Wf p`,
#   whose definition is in the corpus and in no prompt) precisely because it is
#   the mechanism -- and gpt-5.5 confirms every one of them under all three
#   strategies anyway.
#
# - The effect upstream measures is small and capability-bound. Only two of its
#   36 items are wrong for every single-call variant across both models it tried
#   -- 5.6pp, against a 27pp headline. Its own strongest-model run
#   (`naive-inv-opus.json`) scores 94.4% naive against two-pass's 96.3%, so the
#   gap had already nearly closed a model generation ago. The headline also
#   compares three runs of two-pass against one run each of the others, on 36
#   items where one item is 2.8pp.
#
# - And the 69.4% "Claude Code" arm is not a prompt ablation: `bench-cc.js` runs
#   the byte-identical single-prompt text over a different transport (agent
#   system prompt, no temperature control, JSON-schema instead of a forced tool
#   call). Upstream's `cc-twopass.json` scores 100% through that same transport.
#
# - On Lean specifically, upstream's own result reverses: its VERINA run (N=189,
#   Lean 4) has two-pass at 54.0% against a 57.1% baseline, concluding "Two-pass
#   is not strictly better... The blind informalization step adds an
#   interpretation layer that can overcomplicate comparisons."
#
# So: upstream's effect is small, capability-bound, and rests on four discordant
# items; this replication, once its own leak was fixed, puts two-pass ahead in
# most cells but on far too little data to call either way. What this file can
# honestly claim is the pipeline and the failure mode, not a verdict on the
# remedy.
#
# Known limits of the numbers above, in the order they would need fixing:
#
# - Power. 40 items, one run per cell. Detecting an 8-10pp paired effect needs
#   roughly 200 items; extra runs buy almost nothing at temperature 0, where
#   repeats are near-deterministic.
# - Corpus shape. 4 of the 22 faithful claims here are opaque-invariant
#   projections, against 26 of 27 upstream. The mechanism their benchmark
#   actually tests is barely present in this one.
# - Schema asymmetry. The naive arm returns the same `Comparison` type as the
#   others, so it gets the five-category weakening taxonomy for free; upstream's
#   `NAIVE_TOOL` is a bare yes/no with no taxonomy. This example's naive arm is
#   better equipped than upstream's, which would understate any gap.
# - Batching. Upstream's two-pass compare is batched per domain, so its
#   comparator sees the faithful and unfaithful theorem for one requirement side
#   by side -- a contrastive signal its per-item arms never get. Everything here
#   is per-item, which is the cleaner test but not the same test.
#
# Other simplifications:
# - No coverage check. Both here and upstream, a requirement that no theorem
#   addresses at all goes undetected: the audit only ever judges pairs it is
#   handed. Upstream lists this as a known limitation.

import argparse
import asyncio
import dataclasses
import enum
import pathlib
import re
import sys
import textwrap
import typing

# The agents live next door, and that is load-bearing rather than tidiness: the
# harness builds a template's system prompt partly from the source of the module
# the template is defined in, so anything sharing a file with an Agent is shown
# to it. Everything below -- the corpora, the labelled mapping, the expected
# verdicts and their rationales -- is exactly what the auditing agents must not
# see. See the module docstring of `auditing_agents` for what happened when they
# did share a file.
from auditing_agents import (
    Comparator,
    Comparison,
    Informalization,
    Informalizer,
    NaiveAuditor,
    SinglePassAuditor,
    Strength,
    Verdict,
    Weakening,
)

# ---------------------------------------------------------------------------
# The corpus. A verified election tally: `count` tallies the ballots cast for a
# candidate, and each theorem below was written to formalize one plain-English
# requirement. Every one of them compiles (see `--verify`).
#
# The two namespaces are two versions of the same specification file, the way
# upstream's demo has `election0.dfy` and the `election.dfy` that replaced it.
# `Draft` is what was shipped; `Audited` is what survived review. They use the
# same theorem names deliberately -- a developer names a theorem after the
# property they *meant* to prove, which is exactly why the name is no evidence
# that they proved it. Only the statement is ever shown to the model, so neither
# the namespace nor this commentary reaches it.
# ---------------------------------------------------------------------------

ELECTION_CORPUS = r"""import Mathlib

/-- Tally: how many of the ballots in `bs` were cast for candidate `c`. -/
def count : List ℕ → ℕ → ℕ
  | [], _ => 0
  | b :: bs, c => (if b = c then 1 else 0) + count bs c

namespace Audited

theorem empty_election (c : ℕ) : count [] c = 0 := by
  simp [count]

theorem single_ballot_for (c : ℕ) : count [c] c = 1 := by
  simp [count]

theorem single_ballot_against (c d : ℕ) (h : c ≠ d) : count [c] d = 0 := by
  simp [count, h]

theorem count_bounded (bs : List ℕ) (c : ℕ) : count bs c ≤ bs.length := by
  induction bs with
  | nil => simp [count]
  | cons b bs ih =>
    simp only [count, List.length_cons]
    split <;> omega

theorem combine_tallies (a b : List ℕ) (c : ℕ) :
    count (a ++ b) c = count a c + count b c := by
  induction a with
  | nil => simp [count]
  | cons x xs ih =>
    simp only [List.cons_append, count, ih]
    omega

theorem vote_increment (bs : List ℕ) (c : ℕ) :
    count (bs ++ [c]) c = count bs c + 1 := by
  rw [combine_tallies, single_ballot_for]

theorem vote_no_effect (bs : List ℕ) (c d : ℕ) (h : c ≠ d) :
    count (bs ++ [d]) c = count bs c := by
  rw [combine_tallies, single_ballot_against d c (Ne.symm h)]
  omega

theorem order_irrelevant {a b : List ℕ} (h : a.Perm b) (c : ℕ) :
    count a c = count b c := by
  induction h with
  | nil => rfl
  | cons x _ ih => simp only [count, ih]
  | swap x y l => simp only [count]; omega
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂

theorem unanimous_tally (bs : List ℕ) (c : ℕ) (h : ∀ x ∈ bs, x = c) :
    count bs c = bs.length := by
  induction bs with
  | nil => simp [count]
  | cons b bs ih =>
    have hb : b = c := h b (by simp)
    have hrest : ∀ x ∈ bs, x = c := fun x hx => h x (by simp [hx])
    simp [count, hb, ih hrest]
    omega

theorem unanimous_exclusion (bs : List ℕ) (c d : ℕ) (h : ∀ x ∈ bs, x = c)
    (hne : d ≠ c) : count bs d = 0 := by
  induction bs with
  | nil => simp [count]
  | cons b bs ih =>
    have hb : b = c := h b (by simp)
    have hrest : ∀ x ∈ bs, x = c := fun x hx => h x (by simp [hx])
    simp [count, hb, Ne.symm hne, ih hrest]

theorem tally_monotonic (bs : List ℕ) (v c : ℕ) :
    count bs c ≤ count (bs ++ [v]) c := by
  rw [combine_tallies]
  omega

end Audited

namespace Draft

theorem count_bounded (bs : List ℕ) (c : ℕ) : count bs c ≤ bs.length + 1 := by
  have := Audited.count_bounded bs c
  omega

theorem vote_increment (bs : List ℕ) (c : ℕ) :
    count (bs ++ [c]) c = count (bs ++ [c]) c := rfl

theorem order_irrelevant (bs : List ℕ) (v c : ℕ) :
    count ([v] ++ bs) c = count (bs ++ [v]) c := by
  rw [Audited.combine_tallies, Audited.combine_tallies]
  omega

set_option linter.unusedVariables false in
theorem unanimous_tally (bs : List ℕ) (c : ℕ) (hbig : 100 < bs.length)
    (h : ∀ x ∈ bs, x = c) : count bs c = bs.length :=
  Audited.unanimous_tally bs c h

theorem tally_monotonic (bs : List ℕ) (v c : ℕ) : 0 ≤ count (bs ++ [v]) c :=
  Nat.zero_le _

end Draft

namespace Revision

set_option linter.unusedVariables false in
theorem count_bounded (bs : List ℕ) (c : ℕ) (h : c ∈ bs) :
    count bs c ≤ bs.length :=
  Audited.count_bounded bs c

theorem combine_tallies (a b : List ℕ) (c : ℕ) :
    count (a ++ b) c ≥ count a c + count b c :=
  le_of_eq (Audited.combine_tallies a b c).symm

theorem order_irrelevant (a b : List ℕ) (h : a = b) (c : ℕ) :
    count a c = count b c := by
  rw [h]

theorem unanimous_exclusion (bs : List ℕ) (c d : ℕ) (h : ∀ x ∈ bs, x = c)
    (hne : d ≠ c) : count bs d ≤ count bs c := by
  rw [Audited.unanimous_exclusion bs c d h hne]
  exact Nat.zero_le _

set_option linter.unusedVariables false in
theorem tally_monotonic (bs : List ℕ) (v c : ℕ) (h : v = c) :
    count bs c ≤ count (bs ++ [v]) c :=
  Audited.tally_monotonic bs v c

end Revision
"""

# ---------------------------------------------------------------------------
# A second, harder domain: authority over resources, with an invariant. Closer
# in shape to the blog's own `delegation-auth` domain, and harder for three
# reasons -- the statements are longer, several carry an invariant hypothesis
# (`Wf p`), which upstream is explicit that an auditor should *expect* and not
# flag, and the divergences are in the hypotheses rather than the conclusions.
# `Draft.other_subject_unaffected` is the sharpest of them: its conclusion is
# character-for-character the faithful theorem's, and only the hypothesis names
# the wrong thing.
# ---------------------------------------------------------------------------

DELEGATION_CORPUS = r"""import Mathlib

/-- A grant: `subject` may act on `resource` up to `level`
    (0 = none, 1 = read, 2 = write). -/
structure Grant where
  subject : ℕ
  resource : ℕ
  level : ℕ

/-- A policy is a list of grants. -/
abbrev Policy := List Grant

/-- The authority a subject holds on a resource: the highest level any grant
    in the policy gives them there. -/
def authority : Policy → ℕ → ℕ → ℕ
  | [], _, _ => 0
  | g :: rest, s, r =>
      max (if g.subject = s ∧ g.resource = r then g.level else 0) (authority rest s r)

/-- The policy invariant, as a conjunction of three clauses. -/
def Wf (p : Policy) : Prop :=
  (∀ g ∈ p, g.level ≤ 2) ∧ (∀ g ∈ p, 0 < g.subject) ∧ (∀ g ∈ p, 0 < g.resource)

/-- `d` delegates to `t` on resource `r` at level `l`. -/
def delegate (p : Policy) (t r l : ℕ) : Policy := ⟨t, r, l⟩ :: p

namespace Audited

theorem no_grants_no_authority (s r : ℕ) : authority [] s r = 0 := by
  simp [authority]

theorem other_subject_unaffected (p : Policy) (g : Grant) (s r : ℕ)
    (h : g.subject ≠ s) : authority (g :: p) s r = authority p s r := by
  simp [authority, h]

theorem other_resource_unaffected (p : Policy) (g : Grant) (s r : ℕ)
    (h : g.resource ≠ r) : authority (g :: p) s r = authority p s r := by
  simp [authority, h]

theorem unrelated_grant_unaffected (p : Policy) (g : Grant) (s r : ℕ)
    (h : g.subject ≠ s ∨ g.resource ≠ r) :
    authority (g :: p) s r = authority p s r := by
  cases h with
  | inl h => simp [authority, h]
  | inr h => simp [authority, h]

theorem grant_never_reduces (p : Policy) (g : Grant) (s r : ℕ) :
    authority p s r ≤ authority (g :: p) s r := by
  simp [authority]

theorem levels_bounded (p : Policy) (h : Wf p) : ∀ g ∈ p, g.level ≤ 2 := h.1

theorem subjects_named (p : Policy) (h : Wf p) : ∀ g ∈ p, 0 < g.subject := h.2.1

theorem resources_named (p : Policy) (h : Wf p) : ∀ g ∈ p, 0 < g.resource := h.2.2

theorem wf_bounds_authority (p : Policy) (s r : ℕ) (h : Wf p) :
    authority p s r ≤ 2 := by
  induction p with
  | nil => simp [authority]
  | cons g rest ih =>
    have hg : g.level ≤ 2 := h.1 g (by simp)
    have hrest : Wf rest :=
      ⟨fun x hx => h.1 x (by simp [hx]), fun x hx => h.2.1 x (by simp [hx]),
        fun x hx => h.2.2 x (by simp [hx])⟩
    simp only [authority, max_le_iff]
    exact ⟨by split <;> omega, ih hrest⟩

theorem delegation_no_escalation (p : Policy) (d t r l : ℕ)
    (h : l ≤ authority p d r) :
    authority (delegate p t r l) t r ≤ max (authority p d r) (authority p t r) := by
  simp only [delegate, authority, max_le_iff]
  constructor
  · split <;> omega
  · omega

theorem grant_bounds_authority (p : Policy) (g : Grant) (s r : ℕ) :
    authority (g :: p) s r ≤ max (authority p s r) g.level := by
  simp only [authority, max_le_iff]
  constructor
  · split <;> omega
  · omega

end Audited

namespace Draft

theorem wf_bounds_authority (p : Policy) (s r : ℕ) (h : Wf p) :
    authority p s r ≤ 3 := by
  have := Audited.wf_bounds_authority p s r h
  omega

theorem other_subject_unaffected (p : Policy) (g : Grant) (s r : ℕ)
    (h : g.resource ≠ r) : authority (g :: p) s r = authority p s r :=
  Audited.other_resource_unaffected p g s r h

set_option linter.unusedVariables false in
theorem grant_never_reduces (p : Policy) (g : Grant) (s r : ℕ)
    (h : g.subject = s) : authority p s r ≤ authority (g :: p) s r :=
  Audited.grant_never_reduces p g s r

theorem delegation_no_escalation (p : Policy) (t r l : ℕ) :
    authority (delegate p t r l) t r ≤ max l (authority p t r) := by
  simp only [delegate, authority, max_le_iff]
  constructor
  · split <;> omega
  · omega

theorem grant_bounds_authority (p : Policy) (g : Grant) (s r : ℕ) :
    authority (g :: p) s r ≤ authority p s r + g.level := by
  simp only [authority, max_le_iff]
  constructor
  · split <;> omega
  · omega

set_option linter.unusedVariables false in
theorem levels_bounded (p : Policy) (h : Wf p) : ∀ g ∈ p, g.level ≤ g.level :=
  fun _ _ => le_refl _

set_option linter.unusedVariables false in
theorem subjects_named (p : Policy) (h : Wf p) : ∀ g ∈ p, 0 ≤ g.subject :=
  fun _ _ => Nat.zero_le _

end Draft
"""


def statement_of(corpus: str, qualified: str) -> str:
    """Extract the *statement* of ``<namespace>.<theorem>`` from Lean source: the
    text from ``theorem <name>`` up to the ``:=`` that begins its proof.

    Only this crosses the model boundary. The proof is dropped because ClaimCheck
    assumes it correct and audits the claim, and the enclosing namespace is
    dropped because it says which version of the file a theorem came from --
    which the auditor is precisely not entitled to know.
    """
    namespace, _, name = qualified.rpartition(".")
    section = corpus
    if namespace:
        start = section.index(f"namespace {namespace}")
        end = section.index(f"end {namespace}", start)
        section = section[start:end]
    match = re.search(rf"^theorem {re.escape(name)}\b", section, re.MULTILINE)
    if match is None:
        raise KeyError(f"no theorem {qualified!r} in the corpus")
    # The proof begins at the first `:=` at or after the statement; no statement
    # in this corpus contains one, so the first occurrence is the right one.
    body = section[match.start() :]
    return textwrap.dedent(body[: body.index(":=")]).strip()


# ---------------------------------------------------------------------------
# The mapping: which theorem was written to formalize which requirement, plus
# the ground truth. Upstream's `test/integration/mappings/*.json` carry exactly
# these `expected`/`reason` labels, which is what makes the audit scoreable
# rather than merely demonstrable.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Claim:
    """A requirement, the theorem said to formalize it, and the labelled truth.

    ``expected`` and ``why`` are the answer key. They are Python-side bookkeeping
    and must stay that way: they are never passed to a template, and this module
    is never imported by the one the agents are defined in.
    """

    requirement: str
    theorem: str
    expected: Verdict
    why: str = ""


ELECTION_CLAIMS: tuple[Claim, ...] = (
    # The eleven audited theorems: each says what it was written to say.
    Claim(
        "In an empty election with no ballots, every candidate has zero votes",
        "Audited.empty_election",
        Verdict.CONFIRMED,
    ),
    Claim(
        "A single ballot cast for a candidate gives that candidate exactly one vote",
        "Audited.single_ballot_for",
        Verdict.CONFIRMED,
    ),
    Claim(
        "A single ballot cast for one candidate gives every other candidate zero votes",
        "Audited.single_ballot_against",
        Verdict.CONFIRMED,
    ),
    Claim(
        "No candidate can receive more votes than the total number of ballots cast",
        "Audited.count_bounded",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Merging two ballot boxes produces a tally equal to the sum of the "
        "individual tallies",
        "Audited.combine_tallies",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Casting a ballot for a candidate increases that candidate's tally by "
        "exactly one",
        "Audited.vote_increment",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Casting a ballot for one candidate does not change any other candidate's "
        "tally",
        "Audited.vote_no_effect",
        Verdict.CONFIRMED,
    ),
    Claim(
        "The order in which ballots are counted does not affect the final tally",
        "Audited.order_irrelevant",
        Verdict.CONFIRMED,
    ),
    Claim(
        "In a unanimous election, the winning candidate's tally equals the total "
        "number of ballots",
        "Audited.unanimous_tally",
        Verdict.CONFIRMED,
    ),
    Claim(
        "In a unanimous election, every other candidate receives zero votes",
        "Audited.unanimous_exclusion",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Adding a ballot to the election cannot cause any candidate's tally to "
        "decrease",
        "Audited.tally_monotonic",
        Verdict.CONFIRMED,
    ),
    # A faithful theorem against a requirement that asks for strictly more: the
    # theorem is true and proved, and covers half of what was asked for.
    Claim(
        "Casting a ballot for a candidate increases that candidate's tally by "
        "exactly one and leaves every other candidate's tally unchanged",
        "Audited.vote_increment",
        Verdict.DISPUTED,
        "missing-case: covers the chosen candidate only, and says nothing about "
        "the others",
    ),
    # The five drafted theorems: all proved, none of them the claim intended.
    Claim(
        "No candidate can receive more votes than the total number of ballots cast",
        "Draft.count_bounded",
        Verdict.DISPUTED,
        "weakened-conclusion: bounds the tally by one more than the number of "
        "ballots, permitting a candidate to exceed it",
    ),
    Claim(
        "Casting a ballot for a candidate increases that candidate's tally by "
        "exactly one",
        "Draft.vote_increment",
        Verdict.DISPUTED,
        "tautology: both sides of the equation are the same term, so nothing is "
        "claimed about the increment",
    ),
    Claim(
        "The order in which ballots are counted does not affect the final tally",
        "Draft.order_irrelevant",
        Verdict.DISPUTED,
        "narrowed-scope: only one reordering -- moving a single ballot from the "
        "end to the front -- rather than any permutation",
    ),
    Claim(
        "In a unanimous election, the winning candidate's tally equals the total "
        "number of ballots",
        "Draft.unanimous_tally",
        Verdict.DISPUTED,
        "narrowed-scope: an extra hypothesis restricts the guarantee to elections "
        "of more than 100 ballots",
    ),
    Claim(
        "Adding a ballot to the election cannot cause any candidate's tally to "
        "decrease",
        "Draft.tally_monotonic",
        Verdict.DISPUTED,
        "wrong-property: says the new tally is non-negative, which is trivially "
        "true of a natural number, instead of comparing it to the old one",
    ),
    # The five revised theorems: a second attempt, and the reason this domain is
    # not quite as easy as the drafts make it look. Each of these reads correctly
    # at a glance -- the conclusion is the right shape, and the divergence is a
    # hypothesis that narrows it or a relation symbol that loosens it.
    Claim(
        "No candidate can receive more votes than the total number of ballots cast",
        "Revision.count_bounded",
        Verdict.DISPUTED,
        "narrowed-scope: the hypothesis restricts the bound to candidates who "
        "received at least one ballot, so it says nothing about a candidate with "
        "no votes",
    ),
    Claim(
        "Merging two ballot boxes produces a tally equal to the sum of the "
        "individual tallies",
        "Revision.combine_tallies",
        Verdict.DISPUTED,
        "weakened-conclusion: bounds the merged tally below by the sum instead of "
        "equating them, so it permits the merge to invent votes",
    ),
    Claim(
        "The order in which ballots are counted does not affect the final tally",
        "Revision.order_irrelevant",
        Verdict.DISPUTED,
        "tautology: the hypothesis is that the two ballot lists are equal, so the "
        "conclusion is congruence and no reordering is involved",
    ),
    Claim(
        "In a unanimous election, every other candidate receives zero votes",
        "Revision.unanimous_exclusion",
        Verdict.DISPUTED,
        "weakened-conclusion: bounds the other candidate's tally by the winner's "
        "rather than pinning it to zero",
    ),
    Claim(
        "Adding a ballot to the election cannot cause any candidate's tally to "
        "decrease",
        "Revision.tally_monotonic",
        Verdict.DISPUTED,
        "narrowed-scope: the hypothesis restricts the guarantee to the case where "
        "the added ballot is itself for the candidate in question",
    ),
)


DELEGATION_CLAIMS: tuple[Claim, ...] = (
    Claim(
        "A subject with no grants at all has no authority on any resource",
        "Audited.no_grants_no_authority",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Granting authority to one subject never changes any other subject's authority",
        "Audited.other_subject_unaffected",
        Verdict.CONFIRMED,
    ),
    Claim(
        "A grant concerning one resource never changes authority on a different "
        "resource",
        "Audited.other_resource_unaffected",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Adding a grant to a policy can never reduce anyone's authority",
        "Audited.grant_never_reduces",
        Verdict.CONFIRMED,
    ),
    Claim(
        "In a well-formed policy, no subject holds authority above write level (2)",
        "Audited.wf_bounds_authority",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Delegating a level the delegator actually holds cannot leave the delegate "
        "with more authority than the delegator has, beyond what the delegate "
        "already held",
        "Audited.delegation_no_escalation",
        Verdict.CONFIRMED,
    ),
    Claim(
        "Adding a grant cannot raise a subject's authority above the higher of "
        "what they already held and the new grant's level",
        "Audited.grant_bounds_authority",
        Verdict.CONFIRMED,
    ),
    Claim(
        "In a well-formed policy, no subject holds authority above write level (2)",
        "Draft.wf_bounds_authority",
        Verdict.DISPUTED,
        "weakened-conclusion: bounds authority by 3, one level above the write "
        "level the requirement names",
    ),
    # The sharpest trap in either domain: the conclusion is character-for-character
    # the faithful theorem's, and only the hypothesis names the wrong thing.
    Claim(
        "Granting authority to one subject never changes any other subject's authority",
        "Draft.other_subject_unaffected",
        Verdict.DISPUTED,
        "wrong-property: the hypothesis separates the grant's *resource* from the "
        "one queried, not its subject, so this is the other-resource property "
        "wearing the other-subject name",
    ),
    Claim(
        "Adding a grant to a policy can never reduce anyone's authority",
        "Draft.grant_never_reduces",
        Verdict.DISPUTED,
        "narrowed-scope: the hypothesis restricts the guarantee to the subject the "
        "new grant is for, which is the one case nobody doubted",
    ),
    Claim(
        "Delegating a level the delegator actually holds cannot leave the delegate "
        "with more authority than the delegator has, beyond what the delegate "
        "already held",
        "Draft.delegation_no_escalation",
        Verdict.DISPUTED,
        "wrong-property: the delegator never appears; the bound is the delegated "
        "level itself, so it holds however much authority was delegated and "
        "establishes no non-escalation at all",
    ),
    Claim(
        "Adding a grant cannot raise a subject's authority above the higher of "
        "what they already held and the new grant's level",
        "Draft.grant_bounds_authority",
        Verdict.DISPUTED,
        "weakened-conclusion: bounds by the sum of the two rather than their "
        "maximum, permitting authority the requirement forbids",
    ),
    # --- Invariant projections. -------------------------------------------
    # These are the claims that discriminate, and they are all *faithful*.
    # `Wf` is a conjunction of three clauses; each theorem below assumes `Wf p`
    # and concludes one conjunct of it -- so, read as text, the conclusion is
    # already sitting inside the hypothesis, and the theorem looks vacuous. It
    # is not: projecting a named invariant onto one of its consequences is the
    # ordinary way to make an invariant usable, and upstream's prompts say so
    # explicitly ("a lemma that extracts a concrete consequence from an
    # invariant is NOT vacuous").
    #
    # The definition of `Wf` is in the corpus but never in a prompt, because
    # only the *statement* is sent. So `Wf p` reaches the model as an opaque
    # atom, and whether it is judged vacuous turns on whether the model guesses
    # at the invariant's contents -- which is exactly what having been shown the
    # requirement first encourages it to do.
    Claim(
        "In a well-formed policy, no grant exceeds write level (2)",
        "Audited.levels_bounded",
        Verdict.CONFIRMED,
    ),
    Claim(
        "In a well-formed policy, every grant names a real subject",
        "Audited.subjects_named",
        Verdict.CONFIRMED,
    ),
    Claim(
        "In a well-formed policy, every grant names a real resource",
        "Audited.resources_named",
        Verdict.CONFIRMED,
    ),
    # Near-twins of the three above: same shape, same opaque hypothesis, but
    # these really are empty. A corpus with only the projections would reward a
    # checker that confirms everything; these punish that.
    Claim(
        "In a well-formed policy, no grant exceeds write level (2)",
        "Draft.levels_bounded",
        Verdict.DISPUTED,
        "tautology: concludes that each grant's level is at most itself, which "
        "holds of any number and says nothing about the write level",
    ),
    Claim(
        "In a well-formed policy, every grant names a real subject",
        "Draft.subjects_named",
        Verdict.DISPUTED,
        "tautology: concludes that each subject identifier is non-negative, "
        "which is automatic for a natural number and does not say it names "
        "anyone",
    ),
    # Faithful, but *stronger* than the requirement asks -- upstream reports
    # this "redundant strengthening" as its single largest false-positive class,
    # and the two items that defeat every one of its single-call variants are
    # both this shape. The requirement names one case; the theorem covers a
    # strictly broader one, which the taxonomy says is still a match.
    Claim(
        "A grant for a different subject does not change a subject's authority",
        "Audited.unrelated_grant_unaffected",
        Verdict.CONFIRMED,
    ),
)


@dataclasses.dataclass(frozen=True)
class Domain:
    """One body of Lean and the claims made about it."""

    name: str
    corpus: str
    claims: tuple[Claim, ...]


DOMAINS: dict[str, Domain] = {
    "election": Domain("election", ELECTION_CORPUS, ELECTION_CLAIMS),
    "delegation": Domain("delegation", DELEGATION_CORPUS, DELEGATION_CLAIMS),
}


# ---------------------------------------------------------------------------
# Driving one claim through a mode. Each claim gets its own agent instances, so
# nothing an audit learns can leak into the next one through a shared history.
# ---------------------------------------------------------------------------


class Mode(enum.StrEnum):
    TWO_PASS = "two-pass"
    SINGLE_PASS = "single-pass"
    NAIVE = "naive"


@dataclasses.dataclass(frozen=True)
class Audit:
    """One claim's result: what the pipeline decided, and what it read on the way.

    ``comparison`` is None when the pipeline never produced a well-formed verdict
    -- the model kept emitting an incoherent one and `RetryLLMHandler` ran out of
    attempts. That is upstream's third status, ``error``: not a confirmation and
    not a dispute, and it counts against the run rather than being dropped.
    """

    domain: str
    claim: Claim
    statement: str
    comparison: Comparison | None
    back_translation: Informalization | None  # None outside two-pass mode
    error: str | None = None

    @property
    def correct(self) -> bool:
        return (
            self.comparison is not None
            and self.comparison.verdict is self.claim.expected
        )

    @property
    def label(self) -> str:
        return f"{self.domain}/{self.claim.theorem}"


def audit_claim(domain: Domain, claim: Claim, mode: Mode) -> Audit:
    """Audit one requirement/theorem pair under the given mode.

    A claim the model cannot produce a decodable verdict for becomes an ``error``
    result rather than an exception: one intractable item should cost one item,
    not the other thirty-nine.
    """
    statement = statement_of(domain.corpus, claim.theorem)
    back: Informalization | None = None
    try:
        if mode is Mode.TWO_PASS:
            # Pass 1 receives `statement`. There is nowhere in this call for
            # `claim.requirement` to go.
            back = Informalizer().informalize(statement)
            comparison = Comparator().compare(claim.requirement, statement, back)
        else:
            auditor = (
                SinglePassAuditor() if mode is Mode.SINGLE_PASS else NaiveAuditor()
            )
            comparison = auditor.audit(claim.requirement, statement)
    except Exception as exc:
        return Audit(
            domain.name, claim, statement, None, back, f"{type(exc).__name__}: {exc}"
        )
    return Audit(domain.name, claim, statement, comparison, back)


async def audit_all(domains: typing.Sequence[Domain], mode: Mode) -> list[Audit]:
    """Audit every claim in every domain concurrently -- independent by
    construction, since each gets its own agent instances."""
    return list(
        await asyncio.gather(
            *(
                asyncio.to_thread(audit_claim, domain, claim, mode)
                for domain in domains
                for claim in domain.claims
            )
        )
    )


# ---------------------------------------------------------------------------
# The deterministic half. The blog's between-pass checks are diagnostics over
# typed values, so they are a loop in Python rather than another model call:
# code does what code can decide, and the model is asked only what needs
# judgment.
# ---------------------------------------------------------------------------


def pre_checks(audits: typing.Sequence[Audit]) -> list[str]:
    """Flag back-translations rated trivial, and distinct requirements whose
    theorems were read as guaranteeing the same thing."""
    notes: list[str] = []
    seen: dict[str, Claim] = {}
    for audit in audits:
        if (back := audit.back_translation) is None:
            continue
        if back.strength is Strength.TRIVIAL:
            notes.append(
                f"{audit.label} was read as a trivial claim ({back.conclusion})"
            )
        key = f"{audit.domain}: {' '.join(back.conclusion.lower().split())}"
        if (earlier := seen.get(key)) is not None:
            if earlier.requirement != audit.claim.requirement:
                notes.append(
                    f"{audit.label} and {earlier.theorem} were read as "
                    "guaranteeing the same thing, but formalize different "
                    "requirements"
                )
        else:
            seen[key] = audit.claim
    return notes


# ---------------------------------------------------------------------------
# Reporting. The two error directions are reported apart: a missed dispute is a
# weak theorem waved through, which is the failure ClaimCheck exists to prevent,
# while a false dispute costs a developer an argument with the tool.
# ---------------------------------------------------------------------------


def report(audits: typing.Sequence[Audit], mode: Mode) -> None:
    print(f"\n{'=' * 78}\nClaimCheck audit -- strategy: {mode.value}\n{'=' * 78}\n")

    for audit in audits:
        mark = "ok " if audit.correct else "MISS"
        if audit.comparison is None:
            print(f"[{mark}] {audit.label}: error")
        else:
            category = (
                ""
                if audit.comparison.weakening is Weakening.NONE
                else f" [{audit.comparison.weakening.value}]"
            )
            print(f"[{mark}] {audit.label}: {audit.comparison.verdict.value}{category}")
        print(f"       requirement: {audit.claim.requirement}")
        print(f"       statement:   {' '.join(audit.statement.split())}")
        if audit.back_translation is not None:
            back = audit.back_translation
            print(
                f"       read as:     {back.natural_language} "
                f"(strength: {back.strength.value})"
            )
        if audit.comparison is not None and audit.comparison.discrepancy:
            print(f"       discrepancy: {audit.comparison.discrepancy}")
        if audit.error:
            print(f"       no verdict:  {audit.error}")
        if not audit.correct:
            print(f"       EXPECTED {audit.claim.expected.value}: {audit.claim.why}")
        print()

    if notes := pre_checks(audits):
        print("Pre-check diagnostics (deterministic, no model involved):")
        for note in notes:
            print(f"  - {note}")
        print()

    # A "missed dispute" is an unfaithful theorem the audit confirmed. Claims the
    # pipeline never returned a verdict for are counted apart from both error
    # directions -- they are a failure of the harness, not of judgment.
    errored = [a for a in audits if a.comparison is None]
    missed = [
        a
        for a in audits
        if a.comparison is not None
        and a.claim.expected is Verdict.DISPUTED
        and not a.correct
    ]
    false_alarms = [
        a
        for a in audits
        if a.comparison is not None
        and a.claim.expected is Verdict.CONFIRMED
        and not a.correct
    ]
    correct = sum(a.correct for a in audits)

    # Per-domain accuracy as well as overall: the domains differ in difficulty,
    # and an aggregate hides which one a strategy actually struggles with.
    by_domain: dict[str, list[Audit]] = {}
    for audit in audits:
        by_domain.setdefault(audit.domain, []).append(audit)
    if len(by_domain) > 1:
        for name, group in by_domain.items():
            hits = sum(a.correct for a in group)
            print(f"  {name:12} {hits}/{len(group)} ({hits / len(group):.1%})")

    print(
        f"Accuracy: {correct}/{len(audits)} "
        f"({correct / len(audits):.1%})\n"
        f"  unfaithful theorems waved through: {len(missed)}"
        + (f" ({', '.join(a.label for a in missed)})" if missed else "")
        + f"\n  faithful theorems disputed:        {len(false_alarms)}"
        + (f" ({', '.join(a.label for a in false_alarms)})" if false_alarms else "")
        + f"\n  no verdict (retries exhausted):    {len(errored)}"
        + (f" ({', '.join(a.label for a in errored)})" if errored else "")
    )


# ---------------------------------------------------------------------------
# The premise, checked. ClaimCheck is only interesting if the formal artifacts
# really are proved -- otherwise a disputed theorem might just be a broken one.
# `formalization.py` (LEAP) already drives a real Lean 4 + Mathlib toolchain, so
# the check reuses its kernel rather than restating it. Imported inside the
# function, as `world_model_agent.py` imports `gridworlds`, so the example has no
# Lean dependency unless the check is asked for.
# ---------------------------------------------------------------------------


def verify_corpus(domains: typing.Sequence[Domain]) -> bool:
    """Compile each domain's corpus with Lean, and report what it proves."""
    # The examples are importable as ``docs.source.llm_examples...`` from the
    # repository root, which is on ``sys.path`` under the harness but not when
    # this file is run directly; add it so both invocations work.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
    from docs.source.llm_examples.autoresearch.formalization import (
        _SORRY,
        LeanKernel,
    )

    kernel = LeanKernel()
    if not kernel.available():
        print(
            f"Lean project not built at {kernel.project!r}; skipping verification.\n"
            "Build it once (see autoresearch/formalization.py --check-toolchain):\n"
            "  elan default stable\n"
            f"  cd {kernel.project} && lake exe cache get && lake build"
        )
        return False

    total = 0
    for domain in domains:
        theorems = re.findall(r"^theorem (\w+)", domain.corpus, re.MULTILINE)
        print(
            f"Compiling {domain.name} ({len(theorems)} theorems) with "
            "Lean 4 + Mathlib ..."
        )
        result = kernel.compile(domain.corpus)
        if not result.ok:
            raise SystemExit(
                f"The {domain.name} corpus does not compile:\n{result.messages}"
            )
        if _SORRY.search(domain.corpus):
            raise SystemExit(
                f"The {domain.name} corpus contains `sorry`; it is not proved."
            )
        total += len(theorems)
    print(
        f"VERIFIED: {total} theorems, 0 errors, no `sorry`. Every claim below is "
        "proved.\nThe audit that follows is not about whether they are true.\n"
    )
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Named `--strategy`, not `--mode`: the harness parses its own flags with
    # argparse's prefix matching on, so a script flag named `--mode` is swallowed
    # as an abbreviation of the harness's `--model`.
    parser.add_argument(
        "--strategy",
        type=Mode,
        choices=list(Mode),
        default=Mode.TWO_PASS,
        help="Audit strategy: the two-pass split (informalizer never sees the "
        "requirement), one combined call, or a bare 'does this match?'",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compile the corpus with a real Lean toolchain first, establishing "
        "that every theorem audited below is actually proved",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Compile the corpus and exit, without calling any model",
    )
    parser.add_argument(
        "--domain",
        choices=[*DOMAINS, "all"],
        default="all",
        help="Which corpus to audit: the election tally, the harder "
        "authority/delegation policy, or both",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Audit only the first N claims of each domain (a cheap smoke test)",
    )
    args = parser.parse_args()

    domains = list(DOMAINS.values()) if args.domain == "all" else [DOMAINS[args.domain]]
    if args.limit:
        domains = [
            dataclasses.replace(d, claims=d.claims[: args.limit]) for d in domains
        ]

    if args.verify_only:
        verify_corpus(domains)
        return
    if args.verify:
        verify_corpus(domains)

    report(asyncio.run(audit_all(domains, args.strategy)), args.strategy)


if __name__ == "__main__":
    main()
