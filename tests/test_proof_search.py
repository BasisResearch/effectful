from collections import defaultdict, deque

import pantograph as pg
from openai import OpenAI
from pantograph.expr import GoalState
from pantograph.message import ServerError
from pantograph.server import Server, Site, TacticFailure
from pydantic.dataclasses import dataclass

from effectful.handlers.llm import Template
from effectful.handlers.llm.providers import OpenAIAPIProvider
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled


@dataclass(frozen=True)
class Induction:
    target: str

    def to_string(self) -> str:
        return f"induction {self.target}"


@dataclass(frozen=True)
class Simp:
    lemmas: tuple[str, ...] | None = None

    def to_string(self) -> str:
        return f"simp [{', '.join(self.lemmas)}]" if self.lemmas else "simp"


@dataclass(frozen=True)
class Rw:
    lemmas: tuple[str, ...]

    def to_string(self) -> str:
        return f"rw [{', '.join(self.lemmas)}]"


@dataclass(frozen=True)
class Grind:
    def to_string(self) -> str:
        return "grind"


Tactic = Induction | Simp | Rw | Grind


type StateId = int


@dataclass(frozen=True)
class SearchNode:
    goal_state: GoalState
    parent: "SearchNode | None" = None
    tactic: Tactic | None = None
    site: Site | None = None


class ProofSearcher:
    def __init__(
        self, server: Server, state: GoalState, predict_tactic_fn, context: str
    ):
        self.server = server
        self.predict_tactic_fn = predict_tactic_fn
        self.context = context

        self.work = deque([state.state_id])
        self.active = defaultdict(list)
        self.active[state.state_id].append(SearchNode(state))

        self.done: list[SearchNode] = []
        self.failed: list[tuple[SearchNode, str]] = []

    # --- small utilities -----------------------------------------------------

    def _rename_tactic(self, gs: GoalState, idx: int) -> str:
        vs = gs.goals[idx].variables
        bad = [v for v in vs if "✝" in (v.name or "")]
        if not bad:
            return ""
        names = [v.name.replace("✝", "") for v in bad if v.name]
        return f"rename_i {' '.join(names)}"

    def _normalize(self, gs: GoalState, idx: int) -> GoalState:
        t = self._rename_tactic(gs, idx)
        if not t:
            return gs
        try:
            return self.server.goal_tactic(gs, tactic=t, site=Site(goal_id=idx))
        except (ServerError, TacticFailure):
            return gs

    # --- core ---------------------------------------------------------------

    def step(self) -> bool:
        if not self.work:
            return False

        sid = self.work.popleft()
        nodes = self.active.pop(sid, [])
        progressed = False

        for node in nodes:
            tactic = self._predict(node)
            nxt = self._apply(node.goal_state, tactic, node.site)

            if isinstance(nxt, str):
                self.failed.append((node, nxt))
                self.active[sid].append(node)
                continue

            child = SearchNode(nxt, parent=node.parent, tactic=tactic, site=node.site)

            if not nxt.goals:
                self.done.append(child)
                continue

            progressed = True
            for i in range(len(nxt.goals)):
                gs = self._normalize(nxt, i)
                n = SearchNode(gs, parent=child, site=Site(i))
                self.active[gs.state_id].append(n)
                self.work.append(gs.state_id)

        if not progressed:
            self.work.append(sid)

        return bool(self.done)

    # --- proof extraction ----------------------------------------------------

    def extract(self, node: SearchNode | None) -> list[SearchNode]:
        out = []
        while node:
            out.append(node)
            node = node.parent
        return out[::-1]

    def render(self, node: SearchNode) -> str:
        steps = self.extract(node)
        out = []
        indent = ""
        cur = None

        for s in steps:
            if s.site and s.site.goal_id < len(s.goal_state.goals):
                r = self._rename_tactic(s.goal_state, s.site.goal_id)
                if r:
                    out.append(indent + r)

                name = s.goal_state.goals[s.site.goal_id].name
                if name and name != cur:
                    out.append(f"{indent}case {name} =>")
                    indent = "  "
                    cur = name

            if s.tactic:
                out.append(indent + s.tactic.to_string())

        return self.context + "\n".join(out)

    # --- prediction ----------------------------------------------------------

    def _predict(self, node: SearchNode) -> Tactic:
        if (
            node.site
            and node.site.goal_id
            and node.site.goal_id < len(node.goal_state.goals)
        ):
            g = node.goal_state.goals[node.site.goal_id]
        else:
            g = node.goal_state.goals[0]

        vars = [TypeBinding(v.name, v.t) for v in g.variables if v.name]
        script = self.render(node)
        t = self.predict_tactic_fn(str(g), vars, script)

        print("at proof script:\n" + script)
        print("predicted:", t.to_string())
        return t

    # --- server wrapper ------------------------------------------------------

    def _apply(
        self, state: GoalState, tactic: Tactic, site: Site | None
    ) -> GoalState | str:
        try:
            return self.server.goal_tactic(
                state, tactic=tactic.to_string(), site=site or Site()
            )
        except (ServerError, TacticFailure) as e:
            return str(e)


@dataclass
class TypeBinding:
    name: str
    t: str

    def __str__(self):
        return f"{self.name}: {self.t}"


@Template.define
def predict_tactic(
    goal_state: str, variables: list[TypeBinding], proof_script: str
) -> Tactic:
    """
    You are an experienced proof engineer, working for the Lean FRO. You are proficient with the internals of the Lean theorem prover.
    You are currently working on a proof. This proof certifies mission critical software, and completing it will save engineers $200 worth of time.

    You have written the following proof script:

    {proof_script}

    The current goal state is:

    {goal_state}

    You have access to the following variables:

    {variables}

    You must predict a tactic between:

    - induction
    - simp (optionally specify which lemmas to provide)
    - rw [<lemmas-to-rewrite-with>]
    - grind

    Take a deep breath, think carefully, and predict the next tactic to perform. You can do it.
    """
    raise NotHandled


defs = """
def sum_upto (i: Nat) (j: Nat) (f: Nat → Nat) : Nat := match j with
| 0     => f i
| j + 1 => sum_upto i j f + f (j + 1)

notation "∑_{" i " ← " a "}^{" b "} " f => sum_upto a b (fun i => f)
"""
thm_stmt = """theorem sum_upto_mul_two' (n : Nat) : (∑_{i ← 0}^{n} i) = n * (n + 1) / 2 := by\n"""

server = pg.Server(imports=["Init"])
server.load_definitions(defs)

[
    thm,
] = server.load_sorry(thm_stmt + "  sorry")

searcher = ProofSearcher(server, thm.goal_state, predict_tactic, defs + thm_stmt)

with handler(OpenAIAPIProvider(OpenAI())):

    def step():
        searcher.step()

    progress = True

    while not step():
        pass

searcher.render(searcher.done[0])
