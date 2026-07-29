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

    Measured 2026-07 over the full 36-claim corpus, the first three rows at
    temperature 0 to match upstream's benchmark (gpt-5.5 is a reasoning model and
    rejects temperature 0, so it runs at the provider default). ``b-c`` is the
    discordant split for two-pass against naive -- items two-pass alone got right,
    then items naive alone got right -- with an exact McNemar p:

    ==============  ========  ===========  ======  ===========
    model           two-pass  single-pass  naive   b-c (p)
    ==============  ========  ===========  ======  ===========
    gpt-4o             72.2%        88.9%  94.4%   1-9 (0.021)
    gpt-4.1-mini       88.9%        91.7%  86.1%   3-2 (1.000)
    gpt-4.1           100.0%        97.2%  91.7%   3-0 (0.250)
    gpt-5.5            36.1%       100.0% 100.0%   0-23 (<0.001)
    ==============  ========  ===========  ======  ===========

    The published ordering does not reproduce; two of the four cells invert
    significantly. Every two-pass failure is one-sided -- 8 of gpt-4o's 10 errors
    and all 23 of gpt-5.5's are faithful theorems *disputed*, against naive's 0
    in both.

    The gpt-5.5 cell is the informative one, because it isolates a mechanism.
    Upstream's ``ROUNDTRIP_COMPARE_PROMPT`` carries no invariant caveat (its
    ``NAIVE_PROMPT`` and ``CLAIMCHECK_PROMPT`` both do), and this file reproduces
    that asymmetry exactly. The run before it gave the comparator the caveat
    anyway, and scored:

    ==============  ========  ===========  ======
    model           two-pass  single-pass  naive
    ==============  ========  ===========  ======
    gpt-4o             72.2%        94.4%  97.2%
    gpt-4.1-mini       91.7%        86.1%  83.3%
    gpt-4.1            97.2%        91.7%  94.4%
    gpt-5.5            97.2%       100.0%  97.2%
    ==============  ========  ===========  ======

    36.1% to 97.2%. Two corpus items also changed between the runs
    (``contrast_pair_indices_valid`` regained Dafny's lower bound and
    ``columns_are_unique`` its opaque predicate), so the two tables are not a
    clean single-variable ablation -- but those two items are wrong in 1 of 12
    runs each, and cannot account for a 61-point move. The caveat is what moved
    it. So the blind pass is not self-sufficient:
    it produces a reading like "the counter is non-negative -- strength: trivial"
    for ``(h : Inv m) : 0 ≤ m``, correctly, because a reader who cannot see
    inside ``Inv`` genuinely cannot tell whether the conclusion restates the
    hypothesis. The comparator, told that a trivial rating is almost always a
    mismatch, then disputes a faithful theorem. Something has to rescue it.

    Upstream's rescue is almost certainly the batching. Its comparator sees all
    of a domain's pairs in one call, so ``ensures m >= 0`` arrives next to
    ``ensures m == m`` and ``ensures m >= -1`` under the same requirement, and
    the contrast settles which is trivial. Per-item, that signal is gone. This is
    a hypothesis this file does not test -- implementing the batch would test it,
    and is the single most valuable follow-up -- but it is consistent with
    upstream scoring 96.3% on a prompt that, run per-item, scores 36%.

    What *does* reproduce is which items are hard. Five of gpt-4o's eight false
    disputes -- ``base_hue_in_range``, ``always_five_colors``,
    ``all_colors_valid``, ``wip_limits_respected``, ``allocator_always_fresh`` --
    are on the eleven-item false-dispute list upstream's own weakest arm produces
    (`eval/results/cc.json`). And upstream reports the same two-pass reversal on
    the one non-Dafny corpus it tried (VERINA, N=189 Lean: two-pass 54.0% against
    a 57.1% baseline).

    One trap transliterates badly and it is the clearest cost of the port.
    ``grant_non_existent_is_noop_init`` -- upstream's vacuous
    ``requires m == Init()`` -- is waved through in 8 of 12 runs here, where
    every upstream arm catches it. In Dafny the extra precondition is its own
    line under its own keyword; in Lean ``(hinit : m = Init)`` is one binder
    among five and reads as ordinary. That is a language difference, not a
    strategy difference.

  * Coherent verdicts by construction. A ``Comparison`` certifies at decode time
    that ``match`` and ``weakening`` agree and that a mismatch names its
    discrepancy; an incoherent verdict raises and ``RetryLLMHandler`` feeds it
    back. Upstream's JSON schema admits ``match: true`` alongside
    ``weakeningType: "tautology"``, and nothing catches it.

  * The deterministic half stays in Python. The blog's between-pass checks --
    flag a back-translation rated ``trivial``, flag two requirements whose
    theorems have the same conclusion -- are a loop over typed
    ``Informalization`` values, not a model call.

  * The premise is checked by a real prover. Under ``--verify`` all five corpora
    are compiled by the Lean 4 + Mathlib toolchain ``formalization.py`` already
    shells out to. It reports 36 theorems, 0 errors, no ``sorry`` -- and then the
    audit finds nine claims that do not mean what they were written to mean. The
    verifier's clean bill of health is the setup, not the punchline.

Demonstrates:
- Structural separation as *lexical scope*: an agent that cannot see a value
  because its template's signature has no parameter for it, contrasted against
  two ablations that can
- Decode-time certification of a structured verdict's internal coherence, turning
  a self-contradictory answer into a `RetryLLMHandler` retry
- Reuse of a sibling example's real external verifier (`formalization.py`'s
  `LeanKernel`) to establish a premise, rather than asserting it
- Five labelled corpora and an accuracy report that separates the two error
  directions, which is what makes the comparison against upstream's per-item
  results possible at all
- Fan-out over independent audits with ``asyncio.gather`` + ``asyncio.to_thread``
- Per-field guidance carried on the types as ``field(metadata={"description": ...})``
"""

# Differences from the source, and what each costs:
# - Lean 4 + Mathlib, not Dafny. All five domains are transliterated item for
#   item from `test/integration/claims/*.dfy`: same 36 pairs, same 27/9 split,
#   same requirement sentences, same lemma names in snake_case. Lean was chosen
#   because a real, already-built toolchain is reachable from this repo, so
#   "every one of these theorems is proved" is a compile and not a claim.
#   Dafny's `requires`/`ensures` split becomes hypotheses and conclusion. Two
#   traps needed the signed integers Dafny's `int` gives and Lean's `ℕ` does not,
#   so `counter`'s `Model` is `ℤ`; that keeps `-1 ≤ m` a real weakening.
#   Three concessions to the language: Dafny's `to` and `from` are reserved
#   tokens in Lean (the delegation edge's fields are `dst` and `frm`), Dafny's
#   maps become association lists, and each domain sits in a namespace because
#   `Inv`, `Action` and `Init` collide with Mathlib. None of the three is visible
#   in an extracted statement.
# - Statements are sent, proofs are not. Upstream sends the Dafny lemma body
#   along with its contract; here only the statement crosses the boundary, as
#   upstream's own `lemmascript` preset does ("only the signature + requires +
#   ensures is sent, never a body"). In Lean the meaning is entirely in the
#   statement, and ClaimCheck explicitly does not audit the proof. This does
#   remove one signal: upstream's bodies are almost all literally `{ }`, which
#   is itself a hint that the lemma may be a restatement of its own hypothesis.
# - One model, not two. The blog splits the passes across two models (Haiku
#   informalizes, Sonnet compares), reporting that the weaker informalizer is
#   sufficient. Both passes here run on whatever model the harness was given, so
#   that model-asymmetry result is *not* reproduced.
# - One call per claim, not one batched call per pass. This is the one place the
#   port is deliberately *unlike* upstream, and it matters -- see `Batching`
#   below.
# - Opacity is redistributed, not preserved item for item. Dafny's
#   `NoDupSeq(m.cols)` and `ValidColor(m.colors[i])` are opaque names; so are the
#   Lean versions. But `LaneLen`/`WipOf`/`Keys` are inventions of this port,
#   because Dafny's `m.lanes[m.cols[i]]` map indexing has no direct Lean form
#   over association lists -- so `wip_limits_respected` and
#   `lanes_and_wip_match_columns` are *more* opaque here than in Dafny, while
#   `all_colors_valid` lost Dafny's odd hard-coded `forall i | 0 <= i < 5` and is
#   less. Two of the six items cited above as reproducing upstream's difficulty
#   are on the harder side of that trade, which is a caveat on the claim.
# - The prompts keep upstream's asymmetry. `NAIVE_PROMPT` and `CLAIMCHECK_PROMPT`
#   both tell the model that an invariant hypothesis is normal;
#   `ROUNDTRIP_COMPARE_PROMPT` says nothing of the kind. The three agents here
#   reproduce that exactly -- `Comparator` gets no caveat. An earlier revision
#   gave all three one, plus a conclusion-side carve-out upstream has nowhere;
#   that helped precisely the arm and the items under test, and the numbers above
#   are from after it was removed.

# The leak
# --------
# The first version of this example defined the agents in *this* file, beside the
# corpora and the labelled mapping. The harness assembles a template's system
# prompt partly from the source of the module the template is defined in (the
# prompt-assembly table in `effectful.handlers.llm.types.Template` documents this
# plainly), so every agent -- including the informalizer whose whole job is not to
# know -- received every `Claim(...)` entry with its expected verdict and the
# written reason for each, the Lean corpora complete with the namespaces
# `statement_of` exists to strip, and the body of the invariant that is supposed
# to arrive as an opaque atom. A dump measured 78,909 bytes of system prompt, 81%
# of it this file. (That was an earlier, hand-written corpus, since replaced by
# the port below; the leak is a property of the file layout, not of the corpus.)
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
# Second, the corrected numbers were materially different: accuracies fell across
# the board once the key was gone (gpt-5.5 from 100% to 95%), and the ablation
# moved. Every number in this file is from after the fix.
#
# Two further defects, both found only because the first fix forced a re-run:
#
# - Splitting the agents out by hand truncated the `Weakening` enum to three of
#   its six members, so `narrowed-scope`, `missing-case` and `wrong-property`
#   were unavailable while `Comparison.__post_init__` still demanded a category
#   for every mismatch. The models were being asked to classify into a taxonomy
#   with the relevant boxes missing; that produced retry-exhausted claims and
#   depressed every arm. Restoring it took gpt-4o two-pass from 85.0% to 92.5%
#   and its undecodable claims from 3 to 0.
#
# - The naive arm was not actually impoverished. Giving it a bare yes/no return
#   type was not enough, because it still *shared a module* with `Comparison`,
#   `Weakening` and the richer agents -- so the five-category taxonomy and the
#   other arms' prompts reached it through the module source anyway. A dump
#   confirmed every category name in the naive arm's system prompt. It now lives
#   in `auditing_naive.py`, alone, and the dump shows zero. That single change
#   cost the naive arm 10 points on gpt-4.1 and 10 on gpt-4o: most of what had
#   looked like "naive does fine" was the taxonomy leaking into it.
#
# Same lesson each time, which is why the file keeps saying it: in this framework
# the unit of exposure is the module, not the function, the parameter list or the
# return type.
#
# Why this corpus is a port and not an invention
# ----------------------------------------------
# The first two corpora here were hand-written -- an election tally and an
# authority policy, in ClaimCheck's spirit but not its letter. They did not
# reproduce its result, and reading upstream's own per-item eval outputs
# (`eval/results/*.json`) shows why. Recomputing the error direction, arm by arm
# (file names are upstream's; `n` is items x runs):
#
#   ======================  ===  ======  ==============  ==============
#   file                      n     acc  false confirms  false disputes
#   ======================  ===  ======  ==============  ==============
#   two-pass.json           108   96.3%               0               4
#   single-prompt.json       36   86.1%               0               5
#   cc.json                  36   69.4%               0              11
#   naive-inv-sonnet.json   108   88.9%               0              12
#   naive-inv-opus.json     108   94.4%               0               6
#   haiku-haiku.json        108   96.3%               3               1
#   ======================  ===  ======  ==============  ==============
#
# Nearly every error in nearly every arm is an over-flag of a *correct* lemma.
# The exception is the last row -- a two-pass run with a Haiku comparator, which
# waves `CardPartitionNoDups` through in all three runs -- and those are the only
# false confirms anywhere in upstream's recorded judgments. Upstream states the
# pattern in `reports/STRUCTURAL-SEPARATION.md`: "All three variants catch all 8
# deliberately bogus lemmas (100%). The accuracy difference comes entirely from
# false disputes of valid lemmas." (Its "8" is a miscount; the mapping files hold
# nine disputed items. And the claim is true of the three variants it tabulates,
# not of every run it recorded.)
#
# The hand-written corpora measured the opposite thing. Their traps were subtle
# enough that mid-capability models missed them -- twelve runs produced 69 traps
# waved through against 6 faithful theorems disputed. Two-pass cannot help there:
# its mechanism is to make the model *more* credulous (the blind informalizer
# reads a statement at face value and the comparator then matches it), which is a
# precision gain with nothing to gain on a recall benchmark.
#
# So the corpus is now a transliteration rather than a homage, and the properties
# it was rebuilt to carry are upstream's, not ones chosen here:
#
# - The items that actually discriminate are faithful lemmas whose formal
#   statement is an odd-looking rendering of the requirement -- a projection
#   (`ensures forall sc :: sc in m.grants ==> sc.0 in m.subjects` against "All
#   granted capabilities reference existing subjects"), a hypothesis *stronger*
#   than the requirement (`DelegateNonExistentIsNoop`), or a decomposition into
#   conjuncts (`CardInExactlyOneColumn`). Read with the requirement in hand the
#   honest answer is "I cannot confirm that covers all of it" and the model
#   disputes; read blind it is face value and the model confirms. Those two
#   delegation items are the only ones every single-call arm gets wrong on every
#   model upstream tried.
#
# - Nine of upstream's 36 conclusions are a bare named predicate whose definition
#   is not in the prompt -- and in upstream's case not even in its repository,
#   since the domain modules its claims files `include` are absent from the
#   clone. `AllEdgesValid`, `NoDupSeq (AllIds m)`, `ValidColor`,
#   `HuesMatchHarmony` are reproduced here for that reason.
#
# - 27 of 36 items are faithful. A confirm-biased strategy gets a free lift from
#   that majority class, and two-pass is confirm-biased by construction.
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
# So: upstream's effect is small, capability-bound, and rests on a handful of
# discordant items. What this file can honestly claim is the pipeline and the
# failure mode, not a verdict on the remedy.
#
# Known limits of the numbers above, in the order they would need fixing:
#
# - Batching, and it is now the leading explanation, not just a caveat. See the
#   36.1%-vs-97.2% contrast in the docstring above. `src/roundtrip.js` makes
#   *two* API calls for
#   a whole domain -- one informalize-all, one compare-all -- while
#   `singlePromptCheck` and `naiveCheck` each put their call inside a
#   `for (const l of lemmas)` loop. So upstream's winning arm sees all of a
#   domain's lemmas side by side, including the four `counter` lemmas that share
#   one requirement string, one faithful and three weakened. Its losing arms judge
#   each in isolation. Upstream advertises this as "Batching is a free lunch"
#   without treating it as a confound. Everything here is per-item for all three
#   strategies, which is the cleaner test and not the same test.
# - Power. 36 items, one run per cell. Against the correct comparator the target
#   effect is 2-7pp (see the accuracy table above), which needs roughly 200 paired
#   items; extra runs buy almost nothing at temperature 0, where repeats are
#   near-deterministic. n=36 is enough for a *clean* effect -- a 7-0 split reaches
#   p=0.016 -- but not for the 2-0 and 3-1 splits actually observed.
# - No per-item results are committed here, only the aggregate table above, so the
#   claim about which items discriminate cannot be checked from this repository
#   the way upstream's can from `eval/results/*.json`. Upstream's practice is the
#   better one.
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
    SinglePassAuditor,
    Strength,
    Verdict,
    Weakening,
)
from auditing_naive import NaiveAuditor

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
    # The proof begins at the first `:=`; no statement in this corpus contains
    # one. Guard it anyway -- a future statement with a `let` or a structure
    # literal would otherwise be truncated mid-way and sent as a fragment, which
    # is a wrong answer rather than an error.
    statement = textwrap.dedent(body[: body.index(":=")]).strip()
    if statement.count("(") != statement.count(")"):
        raise ValueError(
            f"extracting {qualified!r} cut an unbalanced statement at the first "
            f"`:=`; it probably contains one inside the statement:\n{statement}"
        )
    return statement


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
    and must stay that way: they are never passed to a template, and this module
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


DOMAINS: dict[str, Domain] = {
    "counter": Domain("counter", COUNTER_CORPUS, COUNTER_CLAIMS),
    "canon": Domain("canon", CANON_CORPUS, CANON_CLAIMS),
    "colorwheel": Domain("colorwheel", COLORWHEEL_CORPUS, COLORWHEEL_CLAIMS),
    "delegation": Domain("delegation", DELEGATION_CORPUS, DELEGATION_CLAIMS),
    "kanban": Domain("kanban", KANBAN_CORPUS, KANBAN_CLAIMS),
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
    verdict: Verdict | None
    explanation: str
    comparison: Comparison | None  # None in naive mode, which has no taxonomy
    back_translation: Informalization | None  # None outside two-pass mode
    error: str | None = None

    @property
    def correct(self) -> bool:
        return self.verdict is self.claim.expected

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
    comparison: Comparison | None = None
    try:
        if mode is Mode.NAIVE:
            # The impoverished arm: a yes/no and a sentence, no taxonomy.
            judgement = NaiveAuditor().audit(claim.requirement, statement)
            match, explanation = judgement.match, judgement.explanation
        else:
            if mode is Mode.TWO_PASS:
                # Pass 1 receives `statement`. There is nowhere in this call for
                # `claim.requirement` to go.
                back = Informalizer().informalize(statement)
                comparison = Comparator().compare(claim.requirement, statement, back)
            else:
                comparison = SinglePassAuditor().audit(claim.requirement, statement)
            match, explanation = comparison.match, comparison.explanation
    except Exception as exc:
        return Audit(
            domain.name,
            claim,
            statement,
            None,
            "",
            comparison,
            back,
            f"{type(exc).__name__}: {exc}",
        )
    verdict = Verdict.CONFIRMED if match else Verdict.DISPUTED
    return Audit(domain.name, claim, statement, verdict, explanation, comparison, back)


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
        category = (
            f" [{audit.comparison.weakening.value}]"
            if audit.comparison is not None
            and audit.comparison.weakening is not Weakening.NONE
            else ""
        )
        decided = "error" if audit.verdict is None else audit.verdict.value
        print(f"[{mark}] {audit.label}: {decided}{category}")
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
        elif audit.explanation:
            print(f"       reasoning:   {audit.explanation}")
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
    errored = [a for a in audits if a.verdict is None]
    missed = [
        a
        for a in audits
        if a.verdict is not None
        and a.claim.expected is Verdict.DISPUTED
        and not a.correct
    ]
    false_alarms = [
        a
        for a in audits
        if a.verdict is not None
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
        help="Which of upstream's five benchmark domains to audit, or all of them",
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
