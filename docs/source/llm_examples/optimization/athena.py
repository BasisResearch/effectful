"""Methods as programs: GRAFT-ATHENA's closed loop over a typed numerical toolkit.

Implements "GRAFT-ATHENA: Self-Improving Agentic Teams for Autonomous Discovery and
Evolutionary Numerical Algorithms" (Toscano, Chai & Karniadakis, arXiv 2605.11117). Its
thesis is that an autonomous lab improves with every problem only if method knowledge
lives in a shared, enumerable, measurable action space with memory attached. Its
machinery -- a DAG-to-tree reduction with cross-rules and decision levels, unit-cube
fingerprints with a Jaccard metric, a sigmoid-gated neighbour prior, seven agent teams,
an LLM-graded reward -- compensates for two representational choices: the vocabulary is
untyped JSON, and an "action" is a label whose realisation as code is left to a model
with nothing to check it against.

Here both choices go the other way, and most of the machinery goes with them:

  * **A method is a program.** ``Method = Callable[..., Solution]``, written by the
    strategist, whose one parameter it annotates with the problem class the method
    accepts. The type checker, the method's own doctests and two return guards certify
    it before it runs; the annotation is its applicability metadata.
  * **The vocabulary is the lexical scope.** A typed toolkit of one-dimensional
    finite-difference and spectral components whose *signatures* carry the paper's
    cross-rules -- ``SineBasis(problem: Odd, ...)`` is admissible only for a problem
    certified odd, ``etdrk4(linear: DiagonalOperator, ...)`` only for a Galerkin
    reduction -- and whose validators carry its value rules (``DealiasedGrid`` insists
    on ``M > 3N``). Implementer skills a method may call as subroutines grow it.
  * **One strategist agent** with ``vary``, ``formalize`` and ``debrief`` replaces every
    team. It is ``avo.py``'s ``VariationAgent`` with the problem as an argument. There is
    no advisor: failure modes are a typed field of the evaluation, promotion is
    deterministic on commit, and the notes are the strategist's own.
  * **Learning across problems is primarily in context** -- the transcript, the notes,
    whatever the strategist binds onto ``self`` -- which is the "experience stored as
    text" the paper argues against. Beside it sits a structural store: every executed
    method with absolute measurements, retrieved by type applicability and ranked by
    measured convergence. Type applicability decides what may be *run* as a seed; it is
    not the learner.

The paper's closed loop (its Algorithm 1) is `evolve_lineage` from ``library.py``,
unchanged.

Demonstrates:
- A model-written program as the artifact, with its parameter annotation read back at
  runtime as data: ``applicable`` is an ``isinstance`` check against it
- Return guards on a synthesized callable (``annotated_types.Predicate``, as in
  ``basics/guardrails.py``): a method that does not accept the problem it was written
  for, or that blows up on a coarse grid, is rejected at decode and answered again
- A typed evaluation: failure modes and a *measured* convergence class as fields, with
  diagnostics that say what happened and prescribe nothing
- A nested model-implemented subroutine (``write_stencil``) certified by a post-condition
  that measures its order of accuracy, memoised, and promoted to the strategist's
  components only if a committed method used it
- A cold control (``--no-memory``: a fresh strategist per problem) and warm restarts
  through the harness's own ``--persist-db``

The test of this example is the paper's spectral-Burgers case (its Section 2.8 and
Appendices B.3-B.4): whether a strategist that has solved the earlier problems in the
sequence reaches, from a finite-difference seed, a method for periodic viscous Burgers
with exponential convergence in the mode count -- a sine-only Galerkin truncation,
closed-form diagonal viscous damping, a dealiased pseudospectral product and a
stiffness-aware integrator -- and what it reports when it does not. It reconstructs and
independently validates the Fourier-Galerkin *core* of that case: there the direction
was selected by a user override of the ranker and the modal trajectories were a neural
network trained on a per-mode residual; here the strategist must reach the formulation
itself and the modal system is integrated classically. Run it::

    python -m effectful.handlers.llm.harness docs/source/llm_examples/optimization/athena.py \\
        --model openrouter/openai/gpt-5.5 --reasoning-effort medium --budget 4
    ... --no-memory                    # the cold control
    ... --persist-db /tmp/athena.db    # run twice; the second starts warm

Read off the report, against the paper: the winning method's structure beside the
locked formulation of B.3.6, split into what the strategist chose and what a validator
would have forced; ``convergence == "exponential"`` with a fitted rate against the plain
solver's ``"algebraic"`` (the paper's Fig. 7G; its numbers are RL2 of 1.1e-3 at N = 48
and 6.6e-6 at N = 96); the pitfalls its reviewers caught (a viscous sign error shows as
``unstable_step``, an ``M = 3N`` grid as ``aliasing`` or as a validator rejection);
which record seeded the target and how many versions the warm run needed against the
cold one; and what the strategist bound onto ``self`` and whether a later method used
it. One run per arm, so the comparison is reported, not asserted.

Measured on 2026-09-04 with gpt-5.5 (medium reasoning), four steps per problem, about
fifteen minutes per arm. Both arms reached the locked formulation -- ``SineBasis``,
``viscous_operator``, ``DealiasedGrid``, ``etdrk4`` -- in their *first* variation step
on the target, and every one of its properties the toolkit guarantees was also one
the strategist chose:

    cold, seeded from the plain solver:  v1 RL2 8.6e-7 at N = 96, exponential (0.061)
                                         v4 RL2 2.4e-8 with 2N modes, time-step-limited
    warm, seeded from its advection      v1 RL2 4.0e-12 with 2N modes, exponential (0.11)
      method after five problems:        v3 RL2 7.3e-12, faster; 22 notes, 31 records

The warm arm's first candidate already carried the lessons its notes record from the
smooth Burgers and KdV problems (oversample the modes, conservative flux); the cold
arm needed four steps to get near it. On the neighbours both arms wrote exact modal
semigroups for heat and advection, dealiased ETDRK4 Fourier methods for smooth Burgers
and KdV, and a hand-rolled Chebyshev collocation for Poisson. Nothing either bound
onto ``self`` was a callable a later method named, and ``write_stencil`` was never
called: the toolkit sufficed. The integrity gate below never fired in these runs; in
the runs before it existed, with the references in the system prompt, every problem
was "solved" by returning its exact solution.

Where each of the paper's concepts lands:

  problem graph, tree, a problem as a path     the `Problem` class hierarchy; an instance
  action graph, chains, alphabets              the toolkit in lexical scope
  an action with its knobs                     a call to a component, in the method's source
  a method ``(a_1, ..., a_k)``                  a program, `Method`, with an annotated parameter
  cross-rules, Force/Zero-out, cascades        component signatures, checked at synthesis;
                                               value rules as validators (toolkit-guaranteed)
  decision levels, Tarjan, the I-map           the program's own construction order
  per-node hints, grading prose                component docstrings; the strategist's notes
  fingerprints, the Jaccard metric             the AST name set of a method, in the report only
  memory ``D``                                 `Strategist.memory`, every executed candidate,
                                               plus the transcript, notes and self-bound tools
  neighbour retrieval, the gated prior         `applicable`: records whose annotation accepts
                                               the problem, ranked by convergence class and rate
  neighbour-calibrated expectations            score relative to version 0 on the same problem
  ``History_n`` folding into ``D``             the conversation; `debrief` writes notes and compacts
  expansion, construction, encoding teams      the toolkit is hand-written; `write_stencil` and
                                               self-bound capabilities grow it
  formalization team; the user override        `formalize`; well-posedness as ``__post_init__``
                                               checks; the basis is the strategist's own call
  strategy team: proposer and critic           `vary`; the critic is the type checker, doctests
                                               and the return guards, fed back by the retryer
  implementation team, smoke loop              the method is the code; the `smoke` tool
  advisor team: failures, reward, edits        `MethodEvaluation`; the next `vary`; promotion
  closed loop, Algorithm 1                     `evolve_lineage`
  tree growth, inherited priors                `components`, promoted on commit; reuse measured
  reward rubric (accuracy, integrity, ...)     correctness and integrity gates x RL2 ratio
                                               x time ratio, at the roundoff floor
  per-team models                              ``--strategist-model``, ``--implementer-model``
  spectral PINN: MLP in time, hard IC, vRBA    the same Galerkin reduction, integrated classically
"""

# Simplifications vs. the source:
# - The neural pipeline is elided. What is reconstructed is the method whose convergence
#   the paper says is "inherited from the Galerkin basis"; a torch MLP in time with the
#   hard-IC ansatz, per-mode weighting and a quasi-Newton stage would be a later leaf.
# - GRAFT's reduction, rule operators, decision levels and I-map are subsumed by the type
#   checker and the program's own structure. Fingerprints, Jaccard and the gated prior are
#   replaced by type applicability and a convergence ranking; transfer moves whole methods.
# - Every team is one agent; no user-in-the-loop formalization dialogue; no stagnation
#   steering (avo.py leaves its supervisor out for lack of evidence, and so does this).
# - The neighbours use exact references (Cole-Hopf, a KdV soliton, manufactured
#   solutions) rather than the paper's benchmark data; Helmholtz and Allen-Cahn are
#   omitted. Dedalus or SciPy could serve as an independent oracle; out of scope.
# - The validators encode lessons the paper's reviewers discovered during formalization
#   (M > 3N, the odd-only sine basis, the sign of the viscous operator). That is typed
#   prior knowledge of the kind the paper's expert hints are, and the report separates
#   what the strategist chose from what the toolkit guaranteed.
# - `formalize` returns one dataclass naming the class in Literal fields rather than a
#   union of problem classes: two members with identical fields decode as the first.
# - Both arms see this module's source in the system prompt. The references live in
#   ``athena_references.py``, imported inside the evaluator, so they are not shown; a
#   strategist that knows the closed form anyway -- and it does -- meets the paper's
#   integrity axis as a deterministic gate: a candidate that reconstructs the
#   reference or discretizes nothing scores zero, and the evaluation says why. The
#   first runs, before that gate, returned exact solutions for every problem.
# - One run per arm and no variance estimate.

import argparse
import ast
import collections.abc
import contextlib
import contextvars
import copy
import dataclasses
import hashlib
import inspect
import linecache
import math
import signal
import threading
import time
import types
import typing
from typing import Annotated, Literal

import annotated_types
import numpy as np
import pydantic
import pydantic.dataclasses

from docs.source.llm_examples.optimization.library import (
    Diagnostic,
    Evaluation,
    Lineage,
    Metric,
    evolve_lineage,
    source_of,
    worker,
)
from docs.source.llm_examples.reasoning.continual import Note, harness_diff
from effectful.handlers.llm import Agent, Skill, Tool

type Field = np.ndarray
"""A one-dimensional float array of grid values."""

type Modes = np.ndarray
"""A one-dimensional array of modal coefficients: real for a sine basis, complex for a
Fourier basis."""

SMOKE_RESOLUTION = 32
SMOKE_WALL = 20.0
"""Coarse resolution and wall-clock ceiling for the smoke tool and the return guard."""

FLOOR = 1e-11
"""Errors below this are roundoff and are left out of a convergence fit."""


# ---------------------------------------------------------------------------
# Problems: what components depend on is a class, everything else is a field.
# ---------------------------------------------------------------------------

type Initial = Literal["neg_sin_pi", "cos_pi", "gaussian", "soliton"]
type Equation = Literal["heat", "advection", "burgers", "kdv"]
type Manufactured = Literal["sin_pi", "bump"]


class Periodic:
    """The solution is 2-periodic on [-1, 1]."""


class Dirichlet:
    """The solution vanishes at x = -1 and x = 1."""


class Odd:
    """The solution is odd under x -> -x, so a sine-only expansion is exact."""


@dataclasses.dataclass(frozen=True)
class Problem:
    """A problem on [-1, 1]; ``resolution`` is the discretization size a method should
    use (grid points or retained modes), which the evaluator varies."""

    name: str
    resolution: int
    statement: str = ""

    def __post_init__(self) -> None:
        if self.resolution < 4:
            raise ValueError("resolution must be at least 4")

    @property
    def spec(self) -> str:
        text = repr(self)
        return f"{text}\n{self.statement}" if self.statement else text


@dataclasses.dataclass(frozen=True)
class Evolution(Problem):
    """An initial-value problem u_t = F(u) run to ``t_final``; ``nu`` is the
    equation's one coefficient (viscosity for heat and Burgers, the wave speed for
    advection, the dispersion coefficient for KdV)."""

    equation: Equation = "heat"
    initial: Initial = "neg_sin_pi"
    nu: float = 0.1
    t_final: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.nu <= 0 or self.t_final <= 0:
            raise ValueError("nu and t_final must be positive")
        if (self.equation == "kdv") != (self.initial == "soliton"):
            raise ValueError("the soliton initial data is for kdv, and kdv needs it")
        if self.equation == "burgers" and self.initial not in ("neg_sin_pi", "cos_pi"):
            raise ValueError("burgers needs initial data with a periodic potential")
        if self.equation == "advection" and not isinstance(self, Periodic):
            raise ValueError("advection is posed periodically here")
        if isinstance(self, Dirichlet):
            ends = initial_data(self, np.array([-1.0, 1.0]))
            if np.abs(ends).max() > 1e-12:
                raise ValueError("Dirichlet initial data must vanish at x = -1 and 1")
        if isinstance(self, Odd):
            x = np.linspace(-1.0, 1.0, 33)
            u0 = initial_data(self, x)
            if not np.allclose(u0[::-1], -u0, atol=1e-12):
                raise ValueError(f"initial data {self.initial!r} is not odd")


@dataclasses.dataclass(frozen=True)
class Elliptic(Problem):
    """The boundary-value problem -u_xx = f with f manufactured from ``solution``."""

    solution: Manufactured = "sin_pi"

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self, Periodic) and self.solution != "sin_pi":
            raise ValueError("the periodic Poisson problem needs a periodic solution")


@dataclasses.dataclass(frozen=True)
class PeriodicEvolution(Evolution, Periodic):
    pass


@dataclasses.dataclass(frozen=True)
class OddPeriodicEvolution(PeriodicEvolution, Odd):
    pass


@dataclasses.dataclass(frozen=True)
class DirichletEvolution(Evolution, Dirichlet):
    pass


@dataclasses.dataclass(frozen=True)
class PeriodicElliptic(Elliptic, Periodic):
    pass


@dataclasses.dataclass(frozen=True)
class DirichletElliptic(Elliptic, Dirichlet):
    pass


type ConcreteProblem = (
    PeriodicEvolution
    | OddPeriodicEvolution
    | DirichletEvolution
    | PeriodicElliptic
    | DirichletElliptic
)


def wrap(x: Field) -> Field:
    """Fold positions into the period [-1, 1).

    >>> wrap(np.array([-1.5, 0.25, 1.0]))
    array([ 0.5 ,  0.25, -1.  ])
    """
    return (x + 1.0) % 2.0 - 1.0


def initial_data(problem: Evolution, x: Field) -> Field:
    """The initial condition u(0, x) sampled at ``x``.

    >>> p = PeriodicEvolution("h", 8, equation="heat", initial="cos_pi")
    >>> initial_data(p, np.array([0.0, 0.5, 1.0])).round(12)
    array([ 1.,  0., -1.])
    """
    if problem.initial == "neg_sin_pi":
        return -np.sin(np.pi * x)
    if problem.initial == "cos_pi":
        return np.cos(np.pi * x)
    if problem.initial == "gaussian":
        return np.exp(-(wrap(x) ** 2) / (2 * 0.1**2))
    k = math.sqrt(1.0 / (4.0 * problem.nu))
    return 3.0 / np.cosh(k * wrap(x + 0.25)) ** 2


def forcing(problem: Elliptic, x: Field) -> Field:
    """The right-hand side f = -u_xx of the manufactured Poisson problem.

    >>> forcing(DirichletElliptic("p", 8, solution="sin_pi"), np.array([0.5])).round(6)
    array([9.869604])
    """
    if problem.solution == "sin_pi":
        return np.pi**2 * np.sin(np.pi * x)
    return (1.0 + 4.0 * x + x**2) * np.exp(x)


@pydantic.dataclasses.dataclass(frozen=True)
class Formalization:
    """A problem statement made precise: the class is named by ``kind``,
    ``boundary`` and ``odd``, and `problem` builds it."""

    name: str
    kind: Literal["evolution", "elliptic"]
    boundary: Literal["periodic", "dirichlet"]
    odd: bool
    resolution: int
    equation: Equation = "heat"
    initial: Initial = "neg_sin_pi"
    nu: float = 0.1
    t_final: float = 1.0
    solution: Manufactured = "sin_pi"
    statement: str = ""

    def problem(self) -> ConcreteProblem:
        if self.kind == "elliptic":
            cls = PeriodicElliptic if self.boundary == "periodic" else DirichletElliptic
            return cls(self.name, self.resolution, self.statement, self.solution)
        if self.boundary == "dirichlet":
            if self.odd:
                raise ValueError("an odd solution is a periodic one here")
            evolution: type[PeriodicEvolution | DirichletEvolution] = DirichletEvolution
        else:
            evolution = OddPeriodicEvolution if self.odd else PeriodicEvolution
        return evolution(
            self.name,
            self.resolution,
            self.statement,
            self.equation,
            self.initial,
            self.nu,
            self.t_final,
        )


# ---------------------------------------------------------------------------
# The vocabulary: a typed toolkit whose signatures are the rules.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Solution:
    """A method's answer: values ``u`` at the points ``x`` it computed them on."""

    x: Field
    u: Field

    def __post_init__(self) -> None:
        x, u = np.asarray(self.x, dtype=float), np.asarray(self.u, dtype=float)
        if x.ndim != 1 or x.shape != u.shape:
            raise ValueError("x and u must be one-dimensional arrays of equal length")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "u", u)


type Method = collections.abc.Callable[..., Solution]
"""A method: one parameter, annotated with the problem class it accepts, to a
`Solution`."""


class UniformGrid:
    """``n`` equispaced points on [-1, 1]: a periodic grid omits x = 1, a Dirichlet
    grid includes both ends.

    >>> UniformGrid(PeriodicEvolution("p", 8), 4).x
    array([-1. , -0.5,  0. ,  0.5])
    >>> UniformGrid(DirichletElliptic("d", 8), 5).h
    0.5
    """

    def __init__(self, problem: Problem, n: int):
        self.n = n
        self.periodic = isinstance(problem, Periodic)
        if self.periodic:
            self.x = -1.0 + 2.0 * np.arange(n) / n
            self.h = 2.0 / n
        else:
            self.x = np.linspace(-1.0, 1.0, n)
            self.h = 2.0 / (n - 1)

    def __repr__(self) -> str:
        return f"UniformGrid(n={self.n}, periodic={self.periodic})"


class FourierBasis:
    """``e^{i pi m x}`` for |m| <= n_modes, with a collocation grid of 2 n_modes + 2
    points; modes are the complex coefficients for m = 0 .. n_modes.

    >>> b = FourierBasis(PeriodicEvolution("p", 8), 4)
    >>> bool(np.allclose(b.modes(np.cos(np.pi * b.x)), [0, 0.5, 0, 0, 0]))
    True
    >>> bool(np.allclose(b.values(b.modes(np.cos(np.pi * b.x))), np.cos(np.pi * b.x)))
    True
    """

    def __init__(self, problem: Periodic, n_modes: int):
        self.n_modes = n_modes
        self.n_points = 2 * n_modes + 2
        self.x = -1.0 + 2.0 * np.arange(self.n_points) / self.n_points
        self.m = np.arange(n_modes + 1)
        self.k = np.pi * self.m
        self.phase = (-1.0) ** np.arange(n_modes + 2)

    def modes(self, u: Field) -> Modes:
        c = np.fft.rfft(u)[: self.n_modes + 1] / self.n_points
        return c * self.phase[: self.n_modes + 1]

    def values(self, modes: Modes) -> Field:
        c = np.zeros(self.n_modes + 2, dtype=complex)
        c[: self.n_modes + 1] = modes * self.phase[: self.n_modes + 1]
        return np.fft.irfft(c * self.n_points, n=self.n_points)

    def __repr__(self) -> str:
        return f"FourierBasis(n_modes={self.n_modes})"


class SineBasis:
    """``sin(pi m x)`` for m = 1 .. n_modes -- the parity fold, admissible only for a
    problem certified `Odd` -- with a collocation grid of 2 n_modes + 2 points; modes
    are the real coefficients b_m.

    >>> b = SineBasis(OddPeriodicEvolution("o", 8, equation="burgers"), 4)
    >>> bool(np.allclose(b.modes(-np.sin(np.pi * b.x)), [-1, 0, 0, 0]))
    True
    >>> bool(np.allclose(b.values(b.modes(-np.sin(np.pi * b.x))), -np.sin(np.pi * b.x)))
    True
    """

    def __init__(self, problem: Odd, n_modes: int):
        self.n_modes = n_modes
        self.n_points = 2 * n_modes + 2
        self.x = -1.0 + 2.0 * np.arange(self.n_points) / self.n_points
        self.m = np.arange(1, n_modes + 1)
        self.k = np.pi * self.m
        self.phase = (-1.0) ** self.m

    def modes(self, u: Field) -> Modes:
        c = np.fft.rfft(u) / self.n_points
        return -2.0 * (c[1 : self.n_modes + 1] * self.phase).imag

    def values(self, modes: Modes) -> Field:
        c = np.zeros(self.n_modes + 2, dtype=complex)
        c[1 : self.n_modes + 1] = -0.5j * modes * self.phase
        return np.fft.irfft(c * self.n_points, n=self.n_points)

    def __repr__(self) -> str:
        return f"SineBasis(n_modes={self.n_modes})"


type Basis = FourierBasis | SineBasis


class DealiasedGrid:
    """A finer periodic grid of ``m_points`` on which products of two ``basis``
    functions are evaluated without aliasing, which needs ``m_points > 3 * n_modes``.

    >>> b = FourierBasis(PeriodicEvolution("p", 8), 4)
    >>> DealiasedGrid(b, 12)
    Traceback (most recent call last):
    ...
    ValueError: m_points must exceed 3 * n_modes = 12: a discrete grid of length M = 3N can alias the product mode 2N into the retained mode -N
    >>> DealiasedGrid(b, 14).m_points
    14
    """

    def __init__(self, basis: Basis, m_points: int):
        if m_points <= 3 * basis.n_modes:
            raise ValueError(
                f"m_points must exceed 3 * n_modes = {3 * basis.n_modes}: a discrete "
                f"grid of length M = 3N can alias the product mode 2N into the "
                f"retained mode -N"
            )
        if m_points % 2:
            raise ValueError("m_points must be even")
        self.basis = basis
        self.m_points = m_points
        self.x = -1.0 + 2.0 * np.arange(m_points) / m_points

    def refine(self, u: Field) -> Field:
        """Interpolate collocation values onto the finer grid."""
        n = self.basis.n_modes
        c = np.zeros(self.m_points // 2 + 1, dtype=complex)
        c[: n + 1] = np.fft.rfft(u)[: n + 1] / self.basis.n_points
        return np.fft.irfft(c * self.m_points, n=self.m_points)

    def project(self, w: Field) -> Field:
        """Project values on the finer grid back onto the basis, on its grid."""
        n, p = self.basis.n_modes, self.basis.n_points
        c = np.zeros(n + 2, dtype=complex)
        c[: n + 1] = np.fft.rfft(w)[: n + 1] / self.m_points
        return np.fft.irfft(c * p, n=p)

    def __repr__(self) -> str:
        return f"DealiasedGrid({self.basis!r}, m_points={self.m_points})"


@dataclasses.dataclass(frozen=True)
class DiagonalOperator:
    """A linear operator diagonal in a basis: one eigenvalue per mode."""

    eigenvalues: Modes

    def __call__(self, modes: Modes) -> Modes:
        return self.eigenvalues * modes


@dataclasses.dataclass(frozen=True)
class ExplicitStep:
    """A time step that an explicit stability analysis produced."""

    dt: float

    @staticmethod
    def tightest(*steps: "ExplicitStep") -> "ExplicitStep":
        return ExplicitStep(min(s.dt for s in steps))


def stencil_weights(order: int, offsets: collections.abc.Sequence[int]) -> Field:
    """Finite-difference weights for the ``order``-th derivative on the integer
    ``offsets`` (Fornberg's algorithm, as a Vandermonde solve).

    >>> stencil_weights(1, [-1, 0, 1])
    array([-0.5,  0. ,  0.5])
    >>> stencil_weights(2, [-1, 0, 1])
    array([ 1., -2.,  1.])
    """
    s = np.asarray(offsets, dtype=float)
    rows = np.vstack([s**j for j in range(len(s))])
    rhs = np.zeros(len(s))
    rhs[order] = math.factorial(order)
    return np.linalg.solve(rows, rhs)


def fd_matrix(
    grid: UniformGrid, order: Literal[1, 2, 3], accuracy: Literal[2, 4]
) -> np.ndarray:
    """The dense differentiation matrix of a central stencil, one-sided at the ends
    of a Dirichlet grid.

    >>> g = UniformGrid(DirichletElliptic("d", 8), 6)
    >>> np.allclose(fd_matrix(g, 1, 2) @ g.x**2, 2 * g.x)
    True
    """
    n, h = grid.n, grid.h
    half = (order + accuracy - 1) // 2
    width = 2 * half + 2
    d = np.zeros((n, n))
    central = stencil_weights(order, range(-half, half + 1)) / h**order
    for i in range(n):
        if grid.periodic:
            d[i, (i + np.arange(-half, half + 1)) % n] += central
        elif i < half:
            d[i, :width] = stencil_weights(order, range(-i, width - i)) / h**order
        elif i >= n - half:
            offsets = range(n - width - i, n - i)
            d[i, n - width :] = stencil_weights(order, offsets) / h**order
        else:
            d[i, i - half : i + half + 1] = central
    return d


def fd_derivative(
    u: Field, grid: UniformGrid, order: Literal[1, 2, 3], accuracy: Literal[2, 4]
) -> Field:
    """The ``order``-th derivative of ``u`` on ``grid`` by finite differences of the
    given ``accuracy``.

    >>> g = UniformGrid(PeriodicEvolution("p", 8), 64)
    >>> err = fd_derivative(np.sin(np.pi * g.x), g, 1, 4) - np.pi * np.cos(np.pi * g.x)
    >>> bool(np.abs(err).max() < 1e-4)
    True
    """
    return fd_matrix(grid, order, accuracy) @ u


def upwind(u: Field, grid: UniformGrid, speed: Field | float) -> Field:
    """The first derivative biased against the local ``speed`` (first order).

    >>> g = UniformGrid(PeriodicEvolution("p", 8), 4)
    >>> upwind(np.array([0.0, 1.0, 2.0, 3.0]), g, 1.0)
    array([-6.,  2.,  2.,  2.])
    """
    backward = (u - np.roll(u, 1)) / grid.h
    forward = (np.roll(u, -1) - u) / grid.h
    if not grid.periodic:
        backward[0] = forward[0]
        forward[-1] = backward[-1]
    return np.where(np.asarray(speed) >= 0, backward, forward)


def minmod(a: Field, b: Field) -> Field:
    """The minmod slope limiter.

    >>> minmod(np.array([1.0, -2.0, 3.0]), np.array([2.0, 1.0, 1.0]))
    array([1., 0., 1.])
    """
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def spectral_derivative(u: Field, basis: Basis, order: int) -> Field:
    """The ``order``-th derivative of collocation values, exactly for a function the
    basis resolves.

    >>> b = FourierBasis(PeriodicEvolution("p", 8), 8)
    >>> d = spectral_derivative(np.sin(np.pi * b.x), b, 2)
    >>> np.allclose(d, -np.pi**2 * np.sin(np.pi * b.x))
    True
    """
    c = np.fft.rfft(u) / basis.n_points
    k = np.pi * np.arange(len(c))
    c = c * (1j * k) ** order
    c[-1] = 0.0
    return np.fft.irfft(c * basis.n_points, n=basis.n_points)


def pseudospectral_product(a: Field, b: Field, grid: DealiasedGrid) -> Field:
    """The product ``a * b`` of two collocation fields, formed on the dealiased grid
    and projected back onto the basis.

    >>> basis = FourierBasis(PeriodicEvolution("p", 8), 8)
    >>> grid = DealiasedGrid(basis, 32)
    >>> u = np.cos(np.pi * basis.x)
    >>> np.allclose(pseudospectral_product(u, u, grid), 0.5 + 0.5 * np.cos(2 * np.pi * basis.x))
    True
    """
    return grid.project(grid.refine(a) * grid.refine(b))


def viscous_operator(basis: Basis, nu: float) -> DiagonalOperator:
    """``nu * d^2/dx^2`` in the basis: damping ``-nu (pi m)^2`` per mode.

    >>> op = viscous_operator(FourierBasis(PeriodicEvolution("p", 8), 2), 0.1)
    >>> bool(np.allclose(op.eigenvalues, [0, -0.1 * math.pi**2, -0.4 * math.pi**2]))
    True
    """
    return DiagonalOperator(-nu * basis.k**2)


def dispersive_operator(basis: FourierBasis, delta2: float) -> DiagonalOperator:
    """``-delta2 * d^3/dx^3`` in the Fourier basis: ``i delta2 (pi m)^3`` per mode.

    >>> op = dispersive_operator(FourierBasis(PeriodicEvolution("p", 8), 1), 1.0)
    >>> bool(np.allclose(op.eigenvalues, [0, 1j * math.pi**3]))
    True
    """
    return DiagonalOperator(1j * delta2 * basis.k**3)


def cfl_dt(
    grid: UniformGrid | Basis,
    speed: float,
    cfl: Annotated[float, pydantic.Field(gt=0, le=1)] = 0.5,
) -> ExplicitStep:
    """The advective step ``cfl * h / speed``, with ``h = pi / k_max`` for a basis.

    >>> cfl_dt(UniformGrid(PeriodicEvolution("p", 8), 4), 2.0, 0.5)
    ExplicitStep(dt=0.125)
    """
    h = grid.h if isinstance(grid, UniformGrid) else np.pi / grid.k.max()
    return ExplicitStep(cfl * float(h) / max(abs(speed), 1e-300))


def diffusive_dt(grid: UniformGrid, nu: float, safety: float = 0.5) -> ExplicitStep:
    """The explicit step ``safety * h^2 / (2 nu)`` for a second-order diffusion stencil.

    >>> diffusive_dt(UniformGrid(PeriodicEvolution("p", 8), 4), 0.5, 1.0)
    ExplicitStep(dt=0.25)
    """
    return ExplicitStep(safety * grid.h**2 / (2.0 * nu))


def stiff_dt(operator: DiagonalOperator, safety: float = 0.5) -> ExplicitStep:
    """The forward-Euler stability limit ``safety * 2 / max|lambda|`` of a diagonal
    operator; the number that says whether treating it explicitly is affordable.

    >>> stiff_dt(DiagonalOperator(np.array([-1.0, -4.0])), 1.0)
    ExplicitStep(dt=0.5)
    """
    return ExplicitStep(safety * 2.0 / float(np.abs(operator.eigenvalues).max()))


def _steps(t_final: float, dt: float) -> tuple[int, float]:
    n = max(1, math.ceil(t_final / dt - 1e-12))
    return n, t_final / n


type Rhs = collections.abc.Callable[[np.ndarray], np.ndarray]


def forward_euler(
    f: Rhs, u0: np.ndarray, t_final: float, dt: ExplicitStep
) -> np.ndarray:
    """Integrate ``u' = f(u)`` to ``t_final`` by forward Euler.

    >>> float(forward_euler(lambda u: -u, np.array([1.0]), 1.0, ExplicitStep(0.5))[0])
    0.25
    """
    n, h = _steps(t_final, dt.dt)
    u = np.array(u0)
    for _ in range(n):
        u = u + h * f(u)
    return u


def rk4(f: Rhs, u0: np.ndarray, t_final: float, dt: ExplicitStep) -> np.ndarray:
    """Integrate ``u' = f(u)`` to ``t_final`` by the classical Runge-Kutta method.

    >>> u = rk4(lambda u: -u, np.array([1.0]), 1.0, ExplicitStep(0.1))
    >>> bool(abs(u[0] - math.exp(-1)) < 1e-6)
    True
    """
    n, h = _steps(t_final, dt.dt)
    u = np.array(u0)
    for _ in range(n):
        k1 = f(u)
        k2 = f(u + 0.5 * h * k1)
        k3 = f(u + 0.5 * h * k2)
        k4 = f(u + h * k3)
        u = u + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return u


def etdrk4(
    linear: DiagonalOperator,
    nonlinear: Rhs,
    u0: Modes,
    t_final: float,
    dt: ExplicitStep,
) -> Modes:
    """Integrate ``u' = L u + N(u)`` with ``L`` diagonal by ETDRK4 (Kassam-Trefethen):
    the linear part exactly, so only ``N`` limits the step.

    >>> L = DiagonalOperator(np.array([-100.0]))
    >>> u = etdrk4(L, lambda u: np.zeros_like(u), np.array([1.0]), 1.0, ExplicitStep(0.25))
    >>> bool(abs(u[0] - math.exp(-100)) < 1e-12)
    True
    """
    n, h = _steps(t_final, dt.dt)
    lam = np.asarray(linear.eigenvalues, dtype=complex)
    e, e2 = np.exp(h * lam), np.exp(h * lam / 2)
    r = np.exp(2j * np.pi * (np.arange(1, 33) - 0.5) / 32)
    lr = h * lam[:, None] + r[None, :]
    q = h * np.mean((np.exp(lr / 2) - 1) / lr, axis=1)
    f1 = h * np.mean((-4 - lr + np.exp(lr) * (4 - 3 * lr + lr**2)) / lr**3, axis=1)
    f2 = h * np.mean((2 + lr + np.exp(lr) * (-2 + lr)) / lr**3, axis=1)
    f3 = h * np.mean((-4 - 3 * lr - lr**2 + np.exp(lr) * (4 - lr)) / lr**3, axis=1)
    real = not np.iscomplexobj(u0) and np.allclose(lam.imag, 0.0)
    if real:
        e, e2, q, f1, f2, f3 = (c.real for c in (e, e2, q, f1, f2, f3))
    u = np.array(u0)
    for _ in range(n):
        nu_ = nonlinear(u)
        a = e2 * u + q * nu_
        na = nonlinear(a)
        b = e2 * u + q * na
        nb = nonlinear(b)
        c = e2 * a + q * (2 * nb - nu_)
        nc = nonlinear(c)
        u = e * u + nu_ * f1 + 2 * (na + nb) * f2 + nc * f3
    return u


def imex_cnab2(
    linear: DiagonalOperator,
    nonlinear: Rhs,
    u0: Modes,
    t_final: float,
    dt: ExplicitStep,
) -> Modes:
    """Integrate ``u' = L u + N(u)`` with ``L`` diagonal: Crank-Nicolson on ``L``,
    second-order Adams-Bashforth on ``N``.

    >>> L = DiagonalOperator(np.array([-100.0]))
    >>> u = imex_cnab2(L, lambda u: np.zeros_like(u), np.array([1.0]), 1.0, ExplicitStep(0.01))
    >>> bool(abs(u[0]) < 1e-3)
    True
    """
    n, h = _steps(t_final, dt.dt)
    lam = linear.eigenvalues
    u = np.array(u0)
    previous = nonlinear(u)
    for _ in range(n):
        current = nonlinear(u)
        u = ((1 + h * lam / 2) * u + h * (1.5 * current - 0.5 * previous)) / (
            1 - h * lam / 2
        )
        previous = current
    return u


def backward_euler(matrix: np.ndarray, u0: Field, t_final: float, dt: float) -> Field:
    """Integrate the linear system ``u' = A u`` by backward Euler.

    >>> float(backward_euler(np.array([[-1.0]]), np.array([1.0]), 1.0, 0.5)[0])
    0.4444444444444444
    """
    n, h = _steps(t_final, dt)
    lhs = np.eye(len(u0)) - h * matrix
    u = np.array(u0)
    for _ in range(n):
        u = np.linalg.solve(lhs, u)
    return u


def crank_nicolson(matrix: np.ndarray, u0: Field, t_final: float, dt: float) -> Field:
    """Integrate the linear system ``u' = A u`` by the trapezoidal rule.

    >>> float(crank_nicolson(np.array([[-1.0]]), np.array([1.0]), 1.0, 1.0)[0])
    0.3333333333333333
    """
    n, h = _steps(t_final, dt)
    eye = np.eye(len(u0))
    lhs, rhs = eye - 0.5 * h * matrix, eye + 0.5 * h * matrix
    u = np.array(u0)
    for _ in range(n):
        u = np.linalg.solve(lhs, rhs @ u)
    return u


def plain_solver(problem: Problem) -> Solution:
    """Second-order central differences on a uniform grid, stepped by `rk4` at the
    explicit limits, or a direct solve for a Poisson problem: the straightforward
    composition, converging as h^2.

    >>> s = plain_solver(PeriodicEvolution("h", 32, equation="heat", initial="cos_pi", nu=0.1, t_final=0.1))
    >>> bool(np.abs(s.u - np.cos(np.pi * s.x) * math.exp(-0.1 * math.pi**2 * 0.1)).max() < 1e-3)
    True
    """
    grid = UniformGrid(problem, problem.resolution)
    if isinstance(problem, Elliptic):
        a = -fd_matrix(grid, 2, 2)
        f = forcing(problem, grid.x)
        if isinstance(problem, Dirichlet):
            a[0, :], a[-1, :], f[0], f[-1] = 0.0, 0.0, 0.0, 0.0
            a[0, 0] = a[-1, -1] = 1.0
        else:
            a[0, :] = 1.0 / grid.n
            f[0] = 0.0
        return Solution(grid.x, np.linalg.solve(a, f))
    assert isinstance(problem, Evolution)
    d1, d2, d3 = (fd_matrix(grid, k, 2) for k in (1, 2, 3))
    u0 = initial_data(problem, grid.x)
    speed = float(np.abs(u0).max()) if problem.equation != "advection" else problem.nu
    dt = cfl_dt(grid, max(speed, 1e-3))
    if problem.equation in ("heat", "burgers"):
        dt = ExplicitStep.tightest(dt, diffusive_dt(grid, problem.nu))
    if problem.equation == "kdv":
        spectrum = DiagonalOperator(np.linalg.eigvals(problem.nu * d3))
        dt = ExplicitStep.tightest(dt, stiff_dt(spectrum))
    interior = np.ones(grid.n)
    if isinstance(problem, Dirichlet):
        interior[0] = interior[-1] = 0.0

    def rhs(u: Field) -> Field:
        if problem.equation == "heat":
            f = problem.nu * (d2 @ u)
        elif problem.equation == "advection":
            f = -problem.nu * (d1 @ u)
        elif problem.equation == "burgers":
            f = -u * (d1 @ u) + problem.nu * (d2 @ u)
        else:
            f = -u * (d1 @ u) - problem.nu * (d3 @ u)
        return interior * f

    return Solution(grid.x, rk4(rhs, u0, problem.t_final, dt))


# ---------------------------------------------------------------------------
# A model-implemented subroutine, certified by measuring its order of accuracy.
# ---------------------------------------------------------------------------


@pydantic.dataclasses.dataclass(frozen=True)
class StencilSpec:
    """A derivative of ``order`` to ``accuracy`` on a `UniformGrid`."""

    order: Annotated[int, pydantic.Field(ge=1, le=4)]
    accuracy: Literal[2, 4, 6]

    @property
    def key(self) -> str:
        return f"stencil_d{self.order}_p{self.accuracy}"


type Stencil = collections.abc.Callable[[Field, UniformGrid], Field]

CURRENT_SPEC: contextvars.ContextVar[StencilSpec] = contextvars.ContextVar("SPEC")


def has_declared_order(stencil: Stencil) -> bool:
    """The stencil is exact on polynomials of degree ``order + accuracy - 1`` on a
    Dirichlet grid and its error halves ``accuracy`` times per refinement on a
    periodic one."""
    spec = CURRENT_SPEC.get()
    try:
        grid = UniformGrid(DirichletElliptic("check", 8), 24)
        degree = spec.order + spec.accuracy - 1
        exact = np.polyval(np.polyder(np.ones(degree + 1), spec.order), grid.x)
        got = stencil(np.polyval(np.ones(degree + 1), grid.x), grid)
        if not np.allclose(got, exact, rtol=1e-6, atol=1e-6):
            return False
        errors = []
        for n in (32, 64):
            grid = UniformGrid(PeriodicEvolution("check", 8), n)
            exact = np.pi**spec.order * np.sin(np.pi * grid.x + spec.order * np.pi / 2)
            errors.append(np.abs(stencil(np.sin(np.pi * grid.x), grid) - exact).max())
        return errors[1] < 1e-12 or errors[0] / errors[1] > 2 ** (spec.accuracy - 0.5)
    except Exception:
        return False


class Implementer:
    """You write finite-difference stencils as small, exact numpy functions. Use
    `stencil_weights` for the coefficients rather than typing them; on a periodic
    grid apply the stencil with `np.roll`, and on a Dirichlet grid use one-sided
    stencils of the same accuracy at the ends. Your function is checked on a
    polynomial and on a refinement pair before it is accepted.
    """

    @Skill.define
    def write_stencil(
        self, spec: StencilSpec
    ) -> Annotated[Stencil, annotated_types.Predicate(has_declared_order)]:
        """Write ``stencil(u: Field, grid: UniformGrid) -> Field`` computing the
        derivative of order {spec.order} to accuracy {spec.accuracy} of ``u`` on
        ``grid`` (see `UniformGrid`: ``grid.h``, ``grid.periodic``, ``grid.n``)."""


def write_stencil(spec: StencilSpec) -> Stencil:
    """A model-written derivative stencil for ``spec``, certified before it is
    returned and reused from source afterwards.

    Call this from inside a method when the toolkit's `fd_derivative` is short of
    the order or accuracy you need.
    """
    run = RUN.get()
    run.used.add(spec.key)
    source = run.strategist.components.get(spec.key) or run.products.get(spec.key)
    if source is not None:
        return typing.cast(Stencil, resynthesize(source))
    token = CURRENT_SPEC.set(spec)
    try:
        with (
            worker(run.implementer_model)
            if run.implementer_model
            else contextlib.nullcontext()
        ):
            stencil = run.implementer.write_stencil(spec)
    finally:
        CURRENT_SPEC.reset(token)
    run.products[spec.key] = module_source_of(stencil)
    return stencil


def module_source_of(fn: object) -> str:
    """The whole synthesized module a callable came from, so that its imports and
    helpers travel with it; the callable's own block when it came from a real file."""
    try:
        path = fn.__code__.co_filename  # type: ignore[attr-defined]
        module = "".join(linecache.getlines(path)) if path.startswith("<") else ""
    except Exception:
        module = ""
    return module or source_of(fn) or ""


def resynthesize(source: str, entry: str | None = None) -> collections.abc.Callable:
    """Rebuild a callable from stored source: the named function, else the last one.
    The source is registered in ``linecache`` so it can be recovered again."""
    namespace: dict[str, typing.Any] = dict(globals())
    filename = f"<athena:record:{hashlib.sha1(source.encode()).hexdigest()[:12]}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), namespace)
    module = ast.parse(source)
    defs = [n.name for n in module.body if isinstance(n, ast.FunctionDef)]
    if entry not in namespace and not defs:
        raise ValueError("the stored source defines no function")
    return namespace[entry if entry in namespace else defs[-1]]


# ---------------------------------------------------------------------------
# Evaluation: the reference, the metrics, the failure modes, and the convergence class.
# ---------------------------------------------------------------------------

type FailureMode = Literal[
    "nonfinite",
    "unstable_step",
    "oscillation",
    "aliasing",
    "unresolved",
    "boundary_mismatch",
    "parity_broken",
]


@pydantic.dataclasses.dataclass(frozen=True)
class MethodEvaluation(Evaluation):
    """An `Evaluation` with the failure modes detected analytically and the
    convergence class measured over a resolution sweep."""

    failure_modes: list[FailureMode] = dataclasses.field(default_factory=list)
    convergence: Literal["algebraic", "exponential", "none"] = "none"
    rate: float = 0.0

    def __str__(self) -> str:
        lines = [super().__str__()]
        lines.append(f"failure modes: {', '.join(self.failure_modes) or 'none'}")
        lines.append(f"convergence: {self.convergence} (rate {self.rate:.3g})")
        return "\n".join(lines)


class _Timeout(Exception):
    pass


@contextlib.contextmanager
def wall_clock(seconds: float) -> collections.abc.Iterator[None]:
    """Raise `_Timeout` after ``seconds`` -- on the main thread, where SIGALRM works."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _alarm(signum: int, frame: object) -> None:
        raise _Timeout(f"exceeded the {seconds:.0f}s wall-clock ceiling")

    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def run_method(
    method: Method, problem: Problem, wall: float
) -> tuple[Solution | None, float, str | None]:
    """Run a method under a wall-clock ceiling: the solution, the seconds it took, and
    the error if it raised."""
    start = time.perf_counter()
    try:
        with wall_clock(wall):
            solution = method(problem)
        if not isinstance(solution, Solution):
            solution = Solution(*solution)
    except Exception as exc:
        return None, time.perf_counter() - start, f"{type(exc).__name__}: {exc}"
    return solution, time.perf_counter() - start, None


def coarse(problem: Problem) -> Problem:
    """The same problem at the smoke resolution and, for an evolution, a short horizon."""
    if isinstance(problem, Evolution):
        return dataclasses.replace(
            problem, resolution=SMOKE_RESOLUTION, t_final=min(problem.t_final, 0.1)
        )
    return dataclasses.replace(problem, resolution=SMOKE_RESOLUTION)


def fit_convergence(
    resolutions: collections.abc.Sequence[int], errors: collections.abc.Sequence[float]
) -> tuple[Literal["algebraic", "exponential", "none"], float]:
    """Classify a sweep of errors: algebraic decay (``h^p``, rate ``p``) keeps a
    constant ratio per doubling; exponential decay (``10^(-c N)``, rate ``c`` in
    decades per unit resolution) squares it, and the last doubling must gain more
    than fifth order would. Errors at roundoff count as reached, and a sweep at
    roundoff throughout is exponential with no measurable rate.

    >>> fit_convergence([8, 16, 32, 64], [1e-1, 2.5e-2, 6.25e-3, 1.5625e-3])
    ('algebraic', 2.0)
    >>> fit_convergence([8, 16, 32, 64], [1e-1, 1e-2, 1e-4, 1e-8])
    ('exponential', 0.125)
    >>> fit_convergence([8, 16, 32, 64], [1.6, 1.0, 0.28, 0.036])
    ('algebraic', 1.82)
    >>> fit_convergence([8, 16, 32, 64], [1.0, 1.1, 0.9, 1.0])
    ('none', 0.0)
    >>> fit_convergence([8, 16, 32], [1e-16, 1e-16, 1e-16])
    ('exponential', 0.0)
    """
    usable = [
        (r, e) for r, e in zip(resolutions, errors) if math.isfinite(e) and e > FLOOR
    ]
    saturated = [
        (r, FLOOR)
        for r, e in zip(resolutions, errors)
        if math.isfinite(e) and e <= FLOOR
    ]
    if not usable and saturated:
        return "exponential", 0.0
    points = usable + saturated[:1]
    if len(points) < 3:
        if len(points) == 2 and saturated and points[0][1] / points[1][1] > 1e5:
            r0, r1 = points[0][0], points[1][0]
            return "exponential", round(math.log10(points[0][1] / FLOOR) / (r1 - r0), 4)
        return "none", 0.0
    if points[-1][1] > points[0][1] / 2:
        return "none", 0.0
    ratios = [math.log(points[i][1] / points[i + 1][1]) for i in range(len(points) - 1)]
    if any(r <= 0 for r in ratios):
        return "none", 0.0
    steady = ratios[: len(usable) - 1]
    acceleration = [steady[i + 1] / steady[i] for i in range(len(steady) - 1)]
    accelerating = not acceleration or min(acceleration) >= 1.5
    if accelerating and ratios[-1] >= 5 * math.log(2):
        ns = np.array([r for r, _ in points], dtype=float)
        logs = np.log10([e for _, e in points])
        return "exponential", round(float(-np.polyfit(ns, logs, 1)[0]), 4)
    return "algebraic", round(float(np.mean(ratios) / math.log(2)), 2)


def _sweep(resolution: int) -> list[int]:
    return sorted({max(4, resolution // d) for d in (8, 4, 2, 1)})


def evaluate(
    problem: Problem, *, wall: float, baseline: Method | None = None
) -> collections.abc.Callable[[Method], MethodEvaluation]:
    """An evaluator for ``problem``: metrics against the exact reference, failure
    modes, a measured convergence class, and a score relative to the first method it
    scores finitely (``baseline`` if given, else version 0)."""
    # Imported here, not at module scope, so the exact solutions are in no Skill's
    # lexical scope and not printed into the system prompt.
    from docs.source.llm_examples.optimization.athena_references import reference

    def failures(p: Problem, s: Solution, ref: Field) -> list[tuple[FailureMode, str]]:
        found: list[tuple[FailureMode, str]] = []
        if not np.all(np.isfinite(s.u)):
            return [("nonfinite", "the solution contains inf or nan")]
        bound = np.abs(ref).max()
        if np.abs(s.u).max() > 1.5 * bound + 1e-3:
            found.append(
                (
                    "unstable_step",
                    f"max|u| = {np.abs(s.u).max():.3g} exceeds the reference's "
                    f"{bound:.3g}: the solution grew where the equation damps",
                )
            )
        tv, tv_ref = np.abs(np.diff(s.u)).sum(), np.abs(np.diff(ref)).sum()
        if tv > 1.25 * tv_ref + 0.05 * bound:
            found.append(
                (
                    "oscillation",
                    f"total variation {tv:.3g} against the reference's {tv_ref:.3g}",
                )
            )
        uniform = isinstance(p, Periodic) and np.allclose(np.diff(s.x), s.x[1] - s.x[0])
        if uniform and len(s.u) >= 8:
            spectrum, spectrum_ref = (
                np.abs(np.fft.rfft(s.u)) ** 2,
                np.abs(np.fft.rfft(ref)) ** 2,
            )
            top = slice(len(spectrum) - max(1, len(spectrum) // 10), None)
            tail = spectrum[top].sum() / max(spectrum.sum(), 1e-300)
            tail_ref = spectrum_ref[top].sum() / max(spectrum_ref.sum(), 1e-300)
            if tail > 1e-6 and tail > 100 * tail_ref:
                found.append(
                    (
                        "aliasing",
                        f"{tail:.2g} of the energy sits in the top tenth of the "
                        f"spectrum, against {tail_ref:.2g} for the reference",
                    )
                )
            cutoff = spectrum_ref[-1] / max(spectrum_ref.sum(), 1e-300)
            if cutoff > 1e-8:
                found.append(
                    (
                        "unresolved",
                        f"the reference's relative energy at this grid's highest "
                        f"mode is {cutoff:.2g}: the grid does not resolve the solution",
                    )
                )
        if isinstance(p, Dirichlet) and max(abs(s.u[0]), abs(s.u[-1])) > 1e-6 * (
            1 + bound
        ):
            found.append(
                ("boundary_mismatch", f"u(-1) = {s.u[0]:.3g}, u(1) = {s.u[-1]:.3g}")
            )
        if isinstance(p, Odd):
            mirrored = np.interp(wrap(-s.x), s.x, s.u, period=2.0)
            even = np.linalg.norm(s.u + mirrored) / max(np.linalg.norm(s.u), 1e-300)
            if even > 1e-6:
                found.append(("parity_broken", f"even part {even:.2g} of the norm"))
        return found

    state: dict[str, float] = {}

    def measure(
        method: Method, p: Problem
    ) -> tuple[float, float, Solution | None, str | None]:
        solution, seconds, error = run_method(method, p, wall)
        if solution is None or error is not None:
            return math.inf, seconds, solution, error
        ref = reference(p, solution.x, initial_data)
        err = np.linalg.norm(solution.u - ref) / max(np.linalg.norm(ref), 1e-300)
        return float(err) if math.isfinite(err) else math.inf, seconds, solution, None

    def _evaluate(method: Method) -> MethodEvaluation:
        diagnostics = [Diagnostic("problem", problem.spec)]
        err, seconds, solution, error = measure(method, problem)
        if solution is None:
            diagnostics.append(Diagnostic("raised", error or "no solution"))
            diagnostics.append(
                Diagnostic("code under test", module_source_of(method).strip())
            )
            return MethodEvaluation(
                score=0.0, diagnostics=diagnostics, failure_modes=["nonfinite"]
            )
        ref = reference(problem, solution.x, initial_data)
        modes = failures(problem, solution, ref)
        if len(solution.x) < problem.resolution:
            modes.append(
                (
                    "unresolved",
                    f"{len(solution.x)} points returned for a resolution of "
                    f"{problem.resolution}",
                )
            )
        sweep: list[tuple[int, float]] = []
        for n in _sweep(problem.resolution):
            e, _, _, _ = (
                measure(method, dataclasses.replace(problem, resolution=n))
                if n != problem.resolution
                else (err, seconds, solution, None)
            )
            sweep.append((n, e))
        convergence, rate = fit_convergence(
            [n for n, _ in sweep], [e for _, e in sweep]
        )
        names = [m for m, _ in modes]
        why_not_a_method = shortcut(module_source_of(method))
        correct = (
            math.isfinite(err)
            and err < 1.0
            and not {"nonfinite", "unstable_step"} & set(names)
            and why_not_a_method is None
        )
        seconds = max(seconds, 1e-2)
        if correct and not state:
            state.update(rl2=err, seconds=seconds)
        score = (
            math.sqrt(
                (max(state["rl2"], FLOOR) / max(err, FLOOR))
                * (state["seconds"] / seconds)
            )
            if correct and state
            else 0.0
        )
        linf = float(np.abs(solution.u - ref).max())
        l1 = float(np.abs(solution.u - ref).mean())
        diagnostics += [Diagnostic(m, detail) for m, detail in modes]
        if why_not_a_method is not None:
            diagnostics.append(
                Diagnostic(
                    "integrity",
                    f"{why_not_a_method}: a method discretizes the equation and "
                    f"solves it, so this scores zero however small its error",
                )
            )
        diagnostics.append(
            Diagnostic(
                "resolution sweep (RL2)",
                ", ".join(f"{n}: {e:.3g}" for n, e in sweep)
                + f" -> {convergence} convergence, rate {rate:.3g}"
                + (
                    " (at roundoff throughout)"
                    if convergence == "exponential" and rate == 0.0
                    else ""
                ),
            )
        )
        diagnostics.append(
            Diagnostic(
                "verdict",
                (
                    f"RL2 {err:.3g} in {seconds:.3g}s; the score is the geometric mean "
                    f"of the RL2 and time ratios against version 0 on this problem "
                    f"(RL2 {state['rl2']:.3g}, {state['seconds']:.3g}s)"
                    if correct
                    else f"incorrect (RL2 {err:.3g}); an incorrect method scores zero"
                ),
            )
        )
        diagnostics.append(
            Diagnostic("code under test", module_source_of(method).strip())
        )
        return MethodEvaluation(
            score=score,
            metrics=[
                Metric("rl2", err),
                Metric("linf", linf),
                Metric("l1", l1),
                Metric("seconds", seconds),
                Metric("points", float(len(solution.x))),
            ],
            diagnostics=diagnostics,
            failure_modes=sorted(set(names)),
            convergence=convergence,
            rate=rate,
        )

    if baseline is not None:
        _evaluate(baseline)
    return _evaluate


# ---------------------------------------------------------------------------
# Memory: every executed method, retrieved by type applicability.
# ---------------------------------------------------------------------------


def accepted_class(method: Method) -> typing.Any:
    """The class (or union) a method's parameter annotation names, or ``None``."""
    try:
        hints = typing.get_type_hints(method)
        params = list(inspect.signature(method).parameters)
        annotation = hints[params[0]]
    except Exception:
        return None
    members = (
        typing.get_args(annotation)
        if typing.get_origin(annotation) in (typing.Union, types_union())
        else (annotation,)
    )
    return annotation if all(isinstance(m, type) for m in members) else None


def types_union() -> typing.Any:
    return type(int | str)


@dataclasses.dataclass(frozen=True)
class Record:
    """One executed method: the problem it ran on, what it accepts, its source, and
    its absolute measurements (diagnostics withheld to keep the record small)."""

    problem: str
    accepts: str
    entry: str
    source: str
    evaluation: MethodEvaluation
    origin: str

    def method(self) -> Method:
        return typing.cast(Method, resynthesize(self.source, self.entry))

    def accepts_class(self) -> typing.Any:
        try:
            return eval(self.accepts, dict(globals()))
        except Exception:
            return None

    def __str__(self) -> str:
        ev = self.evaluation
        rl2 = next((m.value for m in ev.metrics if m.name == "rl2"), math.inf)
        return (
            f"{self.origin}: {self.entry}({self.accepts}) on {self.problem} -- RL2 "
            f"{rl2:.3g}, {ev.convergence} (rate {ev.rate:.3g}), failures "
            f"{', '.join(ev.failure_modes) or 'none'}"
        )


def annotation_name(annotation: typing.Any) -> str:
    """A class or union of classes by the names this module knows them by.

    >>> annotation_name(PeriodicEvolution | PeriodicElliptic)
    'PeriodicEvolution | PeriodicElliptic'
    """
    if annotation is None:
        return "?"
    members = typing.get_args(annotation) or (annotation,)
    return " | ".join(m.__name__ for m in members)


def record(
    problem: Problem, method: Method, ev: MethodEvaluation, origin: str
) -> Record:
    return Record(
        problem=problem.name,
        accepts=annotation_name(accepted_class(method)),
        entry=getattr(method, "__name__", "method"),
        source=module_source_of(method),
        evaluation=dataclasses.replace(ev, diagnostics=[]),
        origin=origin,
    )


def applicable_records(
    memory: collections.abc.Sequence[Record], problem: Problem
) -> list[Record]:
    """The records whose annotation accepts ``problem``, best first: correct before
    incorrect, then by convergence class, rate, specificity and recency."""
    rank = {"exponential": 2, "algebraic": 1, "none": 0}

    def specificity(accepts: typing.Any) -> int:
        members = typing.get_args(accepts) or (accepts,)
        return max(len(m.__mro__) for m in members)

    scored = []
    for index, r in enumerate(memory):
        accepts = r.accepts_class()
        if accepts is None or not isinstance(problem, accepts):
            continue
        ev = r.evaluation
        scored.append(
            (
                (
                    ev.score > 0,
                    rank[ev.convergence],
                    ev.rate,
                    specificity(accepts),
                    index,
                ),
                r,
            )
        )
    return [r for _, r in sorted(scored, key=lambda pair: pair[0], reverse=True)]


# ---------------------------------------------------------------------------
# The strategist: one agent, avo.py's variation operator with the problem as an argument.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Run:
    """What is in scope for the current problem: read by the guards, the tools and
    `write_stencil` through the `RUN` context variable."""

    strategist: "Strategist"
    implementer: Implementer
    implementer_model: str | None
    wall: float
    step_budget: int
    problem: Problem | None = None
    products: dict[str, str] = dataclasses.field(default_factory=dict)
    used: set[str] = dataclasses.field(default_factory=set)
    spent: int = 0
    smoke_calls: int = 0


RUN: contextvars.ContextVar[Run] = contextvars.ContextVar("RUN")


def accepts_the_current_problem(method: Method) -> bool:
    """The method's parameter annotation is a class the current problem is an
    instance of."""
    accepts = accepted_class(method)
    problem = RUN.get().problem
    return accepts is not None and problem is not None and isinstance(problem, accepts)


def runs_finite_on_a_coarse_grid(method: Method) -> bool:
    """The method returns a finite `Solution` on the coarse problem within the
    smoke ceiling."""
    problem = RUN.get().problem
    if problem is None:
        return False
    solution, _, error = run_method(method, coarse(problem), SMOKE_WALL)
    return (
        error is None and solution is not None and bool(np.all(np.isfinite(solution.u)))
    )


type CertifiedMethod = Annotated[
    Method,
    annotated_types.Predicate(accepts_the_current_problem),
    annotated_types.Predicate(runs_finite_on_a_coarse_grid),
]


@dataclasses.dataclass
class Strategist(Agent):
    """You are a numerical analyst improving the method for one problem at a time,
    over a long run in which the problems change and you do not: what you learn on
    one is yours on the next.

    A method is a Python function from a problem to a `Solution`, written over the
    toolkit in this module -- grids and bases, finite-difference and spectral
    operators, integrators -- and the annotation on its one parameter is the class of
    problem it accepts. Structure lives in the class hierarchy: a `SineBasis` takes
    an `Odd` problem and nothing else, `etdrk4` takes a `DiagonalOperator` and nothing
    else, a `DealiasedGrid` refuses a grid that would alias. The type checker reads
    your annotation; write the most specific class your method relies on.

    You have real instruments. `smoke` runs a candidate on a coarse grid over a short
    horizon and reports its error, failure modes and a rough convergence class; USE IT
    before you answer, and read what it says. `applicable` lists the methods in your
    memory that accept the current problem, best first. The evaluation you are shown
    measures convergence over a resolution sweep: an algebraic rate is what a stencil
    gives you, an exponential one is what a resolved spectral method gives you, and
    the failure modes name what went wrong without saying what to do about it.

    A method discretizes the equation and solves it. Writing down a solution you
    already know -- a closed form, a translated initial condition, a quadrature of
    the exact answer -- is not a method, scores zero whatever its error, and is
    reported as such. The evaluation says so under "integrity" when it happens.

    Keep what you learn. Append `Note`s to `self.notes` for what transfers between
    problems; bind helpers, `Tool`s or `Skill`s onto `self` and they are offered to
    you on later calls; and when the transcript has served its purpose, write what
    matters onto `self` and compact with `exec_code(compact="conversation")`.
    """

    memory: list[Record] = dataclasses.field(default_factory=list)
    components: dict[str, str] = dataclasses.field(default_factory=dict)
    notes: list[Note] = dataclasses.field(default_factory=list)
    __agent_id__: str = ""

    @property
    def memory_catalog(self) -> str:
        return (
            "\n".join(f"- [{n.id}] {n.title}: {n.body}" for n in self.notes)
            or "(no notes yet)"
        )

    @Tool.define
    def applicable(self) -> list[Record]:
        """The stored methods whose parameter annotation accepts the current problem,
        best first; each is rebuilt from its source by ``record.method()``."""
        problem = RUN.get().problem
        assert problem is not None
        return applicable_records(self.memory, problem)

    @Tool.define
    def smoke(self, method: Method) -> MethodEvaluation:
        """Run a candidate method on the current problem at a coarse resolution over
        a short horizon: error, time, failure modes and a rough convergence class,
        scored against the plain solver on the same coarse problem.

        Pass the function itself: define it in the REPL and call this on its name.
        """
        run = RUN.get()
        assert run.problem is not None
        run.smoke_calls += 1
        if run.spent >= run.step_budget:
            return MethodEvaluation(
                score=0.0,
                diagnostics=[
                    Diagnostic(
                        "BUDGET EXHAUSTED",
                        f"not a score: you have used all {run.step_budget} smoke runs "
                        f"for this step. Return the best method you have measured.",
                    )
                ],
            )
        run.spent += 1
        smoke_problem = coarse(run.problem)
        try:
            evaluation = evaluate(
                smoke_problem, wall=SMOKE_WALL, baseline=plain_solver
            )(method)
        except Exception as exc:
            return MethodEvaluation(
                score=0.0,
                diagnostics=[Diagnostic("raised", f"{type(exc).__name__}: {exc}")],
                failure_modes=["nonfinite"],
            )
        return dataclasses.replace(
            evaluation,
            diagnostics=[
                *evaluation.diagnostics,
                Diagnostic(
                    "smoke runs remaining",
                    f"{run.step_budget - run.spent} of {run.step_budget} in this step",
                ),
            ],
        )

    @Skill.define
    def formalize(self, request: str) -> Formalization:
        """Make this problem statement precise:

        <request>
        {request}
        </request>

        Name the class it belongs to -- an evolution or an elliptic problem, periodic
        or Dirichlet, and whether the solution is odd under x -> -x (only claim odd
        when the initial data is; the claim is checked) -- and fix the equation,
        initial data, coefficient, horizon and resolution the statement gives. Keep
        the statement's own words in ``statement``.
        """

    @Skill.define
    def vary(
        self, problem: Problem, current: Method, evaluation: MethodEvaluation
    ) -> CertifiedMethod:
        """Improve the method for this problem.

        <problem>
        {problem.spec}
        </problem>

        The problem is bound in your REPL session as ``problem``, the current method
        as ``current`` (its source is below; call it to reproduce the numbers), and
        its evaluation as ``evaluation``.

        <current_method>
        {current}
        </current_method>

        <evaluation>
        {evaluation}
        </evaluation>

        Your notes:

        <notes>
        {self.memory_catalog}
        </notes>

        Work like a numerical analyst, not like an oracle:

        - Read the evaluation first: the failure modes, the resolution sweep and its
          convergence class say what the current method is and is not doing.
        - `applicable` lists stored methods that accept this problem; a good one is a
          better starting point than a blank page, and ``self.memory`` is readable in
          full in the REPL.
        - Write a candidate in the REPL, hand it to `smoke`, read what came back, and
          fix what it tells you. You have a budget of smoke runs per step; the
          diagnostics say how many are left. An idea you have not smoked is not an
          improvement.
        - `write_stencil(StencilSpec(order, accuracy))` gives you a certified
          finite-difference stencil the toolkit lacks, callable from inside a method.
        - Solve the equation. A candidate that returns a known solution instead of
          discretizing and integrating scores zero, however exact it is.

        Return a function ``def method(problem: <Class>) -> Solution`` where ``<Class>``
        is the most specific problem class its body relies on: it is rejected at once
        if the class does not accept this problem or if it fails on a coarse grid.
        Make it self-contained -- import what it needs inside, and do not refer to
        ``current``, ``self`` or other session names -- because it is stored as source
        and may be rebuilt for a later problem. It is re-scored after you return it,
        and rejected if it does not improve on the current method, so return something
        you have MEASURED.

        Your function's docstring MUST contain doctests certifying deterministic parts
        of its contract, prefixing each input line with the doctest prompt (three ``>``
        characters and a space; spelled out so that this instruction is not itself
        collected as a test); a one-line doctest on a tiny resolution is enough.
        """

    @Skill.define
    def debrief(self, summary: str) -> list[Note]:
        """This problem is finished; here is what happened:

        <summary>
        {summary}
        </summary>

        Return the notes worth carrying to the next problem -- what worked, what
        failed and why, which stored method to start from for which class of problem
        -- as `Note`s with short ids and titles. Then bind onto ``self`` anything
        reusable you built in the REPL, and compact the conversation with
        ``exec_code(compact="conversation")`` so the next problem starts from your
        notes rather than from this transcript.
        """


# ---------------------------------------------------------------------------
# The loop and the report
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Outcome:
    """What one problem produced, for the report."""

    problem: Problem
    applicable: list[Record]
    seed: Record
    lineage: Lineage[Method]
    smoke_calls: int
    promoted: list[str]
    seconds: float


def solve(
    problem: Problem, run: Run, budget: int, strategist_model: str | None
) -> Outcome:
    """One problem: seed from memory, evolve, record every executed candidate,
    promote what a committed version used, debrief."""
    strategist = run.strategist
    run.problem, run.products, run.used, run.smoke_calls = problem, {}, set(), 0
    scope = worker(strategist_model) if strategist_model else contextlib.nullcontext()

    candidates = applicable_records(strategist.memory, problem)
    seed = candidates[0]
    print(f"\n{'=' * 72}\n{problem.spec}\n")
    print(f"applicable: {len(candidates)} record(s); seeding from {seed.origin}")

    evaluator = evaluate(problem, wall=run.wall)
    executed: list[tuple[Method, MethodEvaluation, str]] = []

    def evaluate_and_record(method: Method) -> MethodEvaluation:
        ev = evaluator(method)
        executed.append((method, ev, f"{problem.name}/candidate"))
        return ev

    def vary(current: Method, ev: Evaluation) -> Method:
        run.spent = 0
        assert isinstance(ev, MethodEvaluation)
        with scope:
            return strategist.vary(problem, current, ev)

    started = time.monotonic()
    lineage = evolve_lineage(
        vary=vary, evaluator=evaluate_and_record, seed=seed.method(), budget=budget
    )
    seconds = time.monotonic() - started

    committed = {id(v.artifact): v for v in lineage.versions}
    for method, ev, origin in executed:
        version = committed.get(id(method))
        origin = f"{problem.name}/v{version.index}" if version else origin
        strategist.memory.append(record(problem, method, ev, origin))

    promoted = []
    for version in lineage.versions:
        run.used = set()
        run_method(version.artifact, coarse(problem), SMOKE_WALL)
        for key in run.used & run.products.keys():
            strategist.components[key] = run.products[key]
            promoted.append(key)
    run.products = {}

    summary = "\n".join(
        [
            f"seeded from {seed.origin}",
            *(f"v{v.index} ({v.note}): {v.evaluation}" for v in lineage.versions),
            f"{lineage.rejected} rejected of {lineage.attempts} attempts",
        ]
    )
    with scope:
        strategist.notes.extend(strategist.debrief(summary))
    return Outcome(
        problem, candidates, seed, lineage, run.smoke_calls, promoted, seconds
    )


TOOLKIT = (
    "UniformGrid", "FourierBasis", "SineBasis", "DealiasedGrid", "DiagonalOperator",
    "ExplicitStep", "stencil_weights", "fd_matrix", "fd_derivative", "upwind", "minmod",
    "spectral_derivative", "pseudospectral_product", "viscous_operator",
    "dispersive_operator", "cfl_dt", "diffusive_dt", "stiff_dt", "forward_euler", "rk4",
    "etdrk4", "imex_cnab2", "backward_euler", "crank_nicolson", "plain_solver",
    "write_stencil", "initial_data", "forcing",
)  # fmt: skip

FORMULATION = {
    "sine-only Galerkin basis": ("SineBasis",),
    "closed-form diagonal viscous damping": ("viscous_operator",),
    "dealiased pseudospectral product": ("pseudospectral_product", "DealiasedGrid"),
    "stiffness-aware integrator": ("etdrk4", "imex_cnab2", "crank_nicolson"),
}
"""The locked formulation of the paper's B.3.6, as the components that realise it."""

GUARANTEED = {
    "M > 3N dealiasing": "DealiasedGrid",
    "sine basis only for an Odd problem": "SineBasis",
    "the sign of the viscous operator": "viscous_operator",
}
"""Properties a validator or a signature forces, which the strategist did not choose."""


def names_in(source: str) -> set[str]:
    """Every identifier a source mentions: names, attributes and imported modules.

    >>> sorted(names_in("import os.path\\nx = np.fft.rfft(u)"))
    ['fft', 'np', 'os', 'path', 'rfft', 'u', 'x']
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, ast.Import):
            names.update(part for a in n.names for part in a.name.split("."))
        elif isinstance(n, ast.ImportFrom) and n.module:
            names.update(n.module.split("."))
    return names


SOLVING = frozenset(
    {
        "fd_matrix", "fd_derivative", "stencil_weights", "upwind", "write_stencil",
        "spectral_derivative", "pseudospectral_product", "viscous_operator",
        "dispersive_operator", "forward_euler", "rk4", "etdrk4", "imex_cnab2",
        "backward_euler", "crank_nicolson", "plain_solver",
        "solve", "lstsq", "fft", "rfft", "irfft", "ifft",
    }
)  # fmt: skip
"""What discretizing and solving an equation looks like in source: an operator, a
stencil, an integrator, or a linear solve or transform of numpy's own."""

ORACLE = frozenset({"hermgauss", "athena_references"})
"""What reconstructing the reference looks like in source."""


def shortcut(source: str) -> str | None:
    """Why a method is not one: it reconstructs the reference, or it discretizes
    nothing; ``None`` for a method.

    >>> shortcut("def m(p: Problem) -> Solution:\\n    return plain_solver(p)")
    >>> shortcut("def m(p: Problem) -> Solution:\\n    return Solution(x, initial_data(p, x - p.t_final))")
    'uses no operator, integrator, solve or transform'
    """
    names = names_in(source)
    if names & ORACLE:
        return "reconstructs the reference's own quadrature"
    if not names & SOLVING:
        return "uses no operator, integrator, solve or transform"
    return None


def report(
    outcomes: list[Outcome], strategist: Strategist, baseline: dict[str, typing.Any]
) -> None:
    print("\n" + "=" * 72 + "\nREPORT\n")
    for o in outcomes:
        best = o.lineage.best
        source = module_source_of(best.artifact)
        used = names_in(source) & set(TOOLKIT)
        print(f"{o.problem.name} ({type(o.problem).__name__}), {o.seconds:.0f}s:")
        print(f"  applicable: {', '.join(r.origin for r in o.applicable)}")
        print(
            f"  seeded from {o.seed.origin}; {len(o.lineage.versions) - 1} committed, "
            f"{o.lineage.rejected} rejected, {o.smoke_calls} smoke runs"
        )
        for v in o.lineage.versions:
            accepts = accepted_class(v.artifact)
            ev = v.evaluation
            assert isinstance(ev, MethodEvaluation)
            rl2 = next((m.value for m in ev.metrics if m.name == "rl2"), math.inf)
            points = next((m.value for m in ev.metrics if m.name == "points"), 0.0)
            print(
                f"  v{v.index}: accepts {annotation_name(accepts)}, "
                f"score {v.score:.3g}, RL2 {rl2:.3g} on {points:.0f} points, "
                f"{ev.convergence} (rate {ev.rate:.3g}), "
                f"failures {', '.join(ev.failure_modes) or 'none'}"
            )
        print(
            f"  winner uses: {', '.join(sorted(used)) or '(nothing from the toolkit)'}"
        )
        if isinstance(o.problem, Odd):
            print(
                "  against B.3.6, chosen by the strategist (toolkit components "
                "named in the source; an inline reimplementation does not count):"
            )
            for label, components in FORMULATION.items():
                hit = sorted(used & set(components))
                print(
                    f"    [{'x' if hit else ' '}] {label}"
                    + (f" via {', '.join(hit)}" if hit else "")
                )
            print("  guaranteed by the toolkit, where used:")
            for label, component in GUARANTEED.items():
                print(
                    f"    [{'x' if component in used else ' '}] {label} ({component})"
                )
        if (why := shortcut(source)) is not None:
            print(f"  WARNING: the winner is not a method: it {why}")
        if o.promoted:
            print(f"  promoted components: {', '.join(o.promoted)}")

    def revisable(attributes: dict[str, typing.Any]) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            **{
                k: v
                for k, v in attributes.items()
                if not k.startswith("_") and k not in ("memory", "components", "notes")
            }
        )

    print("\nharness changes over the run (memory, components and notes aside):")
    print(harness_diff(revisable(vars(strategist)), vars(revisable(baseline))))  # type: ignore[arg-type]
    print("\nself-created capabilities and their reuse:")
    bound = sorted(
        set(vars(revisable(vars(strategist)))) - set(vars(revisable(baseline)))
    )
    for name in bound:
        users = [
            f"{o.problem.name}/v{v.index}"
            for o in outcomes
            for v in o.lineage.versions
            if name in names_in(module_source_of(v.artifact))
        ]
        print(f"  {name}: referenced by {', '.join(users) or 'no committed method'}")
    print(f"  components: {', '.join(strategist.components) or 'none'}")
    print(f"  notes: {len(strategist.notes)}")
    print(f"  memory: {len(strategist.memory)} record(s)")


# ---------------------------------------------------------------------------
# The problem sequence and the target
# ---------------------------------------------------------------------------

TARGET_STATEMENT = """\
Solve the viscous Burgers equation u_t + u u_x = nu u_xx on x in [-1, 1] for t in
(0, 1] with nu = 0.01, initial condition u(0, x) = -sin(pi x), and periodic boundary
conditions (the solution stays odd, so u(t, -1) = u(t, 1) = 0). Report u at t = 1;
the reference is the Cole-Hopf solution. The solution steepens into a thin layer at
x = 0 of width about nu, which is where a method earns its keep.
"""

SEQUENCE: dict[str, Problem] = {
    p.name: p
    for p in (
        PeriodicEvolution("heat", 64, "", "heat", "cos_pi", 0.05, 0.5),
        PeriodicEvolution("advection", 64, "", "advection", "gaussian", 1.0, 1.0),
        PeriodicEvolution("burgers_smooth", 64, "", "burgers", "cos_pi", 0.1, 0.5),
        PeriodicEvolution("kdv", 128, "", "kdv", "soliton", 1e-3, 0.5),
        DirichletElliptic("poisson", 64, "", "bump"),
    )
}
"""The neighbours memory is built on, in order, before the target."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problems",
        nargs="+",
        default=[*SEQUENCE, "burgers"],
        metavar="NAME",
        help=f"Problems in order; one of {', '.join(SEQUENCE)} or 'burgers' (the target)",
    )
    parser.add_argument(
        "--budget", type=int, default=4, help="Variation steps per problem"
    )
    parser.add_argument(
        "--step-budget",
        type=int,
        default=6,
        help="Smoke runs the strategist may make per step",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override every problem's resolution",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="The cold control: a fresh strategist per problem",
    )
    parser.add_argument(
        "--strategist-model", default=None, help="Model for the strategist"
    )
    parser.add_argument(
        "--implementer-model", default=None, help="Model for write_stencil"
    )
    parser.add_argument(
        "--wall", type=float, default=60.0, help="Seconds a method may run"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed (unused by the numerics)"
    )
    args = parser.parse_args()

    implementer = Implementer()

    def fresh(tag: str) -> Strategist:
        strategist = Strategist(__agent_id__=f"strategist-{tag}")
        strategist.__history__  # noqa: B018  # a checkpointed strategist restores here
        if not strategist.memory:
            ev = MethodEvaluation(score=1.0, convergence="algebraic", rate=2.0)
            strategist.memory.append(
                record(Problem("any", 8), plain_solver, ev, "expert")
            )
        return strategist

    strategist = fresh("shared")
    baseline = copy.deepcopy(vars(strategist))
    run = Run(
        strategist, implementer, args.implementer_model, args.wall, args.step_budget
    )
    token = RUN.set(run)
    outcomes: list[Outcome] = []
    try:
        for name in args.problems:
            if args.no_memory:
                run.strategist = strategist = fresh(name)
                baseline = copy.deepcopy(vars(strategist))
            if name == "burgers":
                scope = (
                    worker(args.strategist_model)
                    if args.strategist_model
                    else contextlib.nullcontext()
                )
                with scope:
                    problem: Problem = strategist.formalize(TARGET_STATEMENT).problem()
                print(f"formalized as {type(problem).__name__}")
            else:
                problem = SEQUENCE[name]
            if args.resolution:
                problem = dataclasses.replace(problem, resolution=args.resolution)
            outcomes.append(solve(problem, run, args.budget, args.strategist_model))
    finally:
        RUN.reset(token)

    report(outcomes, strategist, baseline)
    target = next((o for o in outcomes if isinstance(o.problem, Odd)), None)
    if target is not None:
        ev = target.lineage.best.evaluation
        assert isinstance(ev, MethodEvaluation)
        print(
            f"\ntarget: {ev.convergence} convergence (rate {ev.rate:.3g}) against the "
            f"paper's exponential; RL2 "
            f"{next((m.value for m in ev.metrics if m.name == 'rl2'), math.inf):.3g} at "
            f"resolution {target.problem.resolution} against 6.6e-6 at N = 96"
        )


if __name__ == "__main__":
    main()
