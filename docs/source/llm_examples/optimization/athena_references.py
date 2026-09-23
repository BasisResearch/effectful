"""Exact reference solutions for ``athena.py``'s problems.

Kept out of that module on purpose: the harness prints a Skill's module source into
the system prompt, and a strategist that can read the reference will return it
instead of solving the equation. ``athena.evaluate`` imports this lazily, so nothing
here is in any Skill's lexical scope. Nothing is imported from ``athena`` either:
the launcher runs it as ``__main__``, and a second import would make a second set of
problem classes.
"""

import collections.abc
import math
import typing

import numpy as np

HERMITE = np.polynomial.hermite.hermgauss(200)
"""Gauss-Hermite nodes and weights for the Cole-Hopf integral (Basdevant et al. 1986)."""

type Sampler = collections.abc.Callable[[typing.Any, np.ndarray], np.ndarray]


def wrap(x: np.ndarray) -> np.ndarray:
    return (x + 1.0) % 2.0 - 1.0


def cole_hopf(problem: typing.Any, x: np.ndarray, initial: Sampler) -> np.ndarray:
    """Viscous Burgers by the Cole-Hopf transform, for initial data with a periodic
    potential."""
    z, w = HERMITE
    eta = math.sqrt(4.0 * problem.nu * problem.t_final) * z
    y = x[:, None] - eta[None, :]
    u0 = initial(problem, y)
    potential = (np.cos(np.pi * y) - 1.0) / np.pi
    if problem.initial == "cos_pi":
        potential = np.sin(np.pi * y) / np.pi
    exponent = -potential / (2.0 * problem.nu)
    phi = np.exp(exponent - exponent.max())
    return (u0 * phi) @ w / (phi @ w)


def reference(problem: typing.Any, x: np.ndarray, initial: Sampler) -> np.ndarray:
    """The exact solution of ``problem`` at the points ``x``; ``initial`` samples an
    evolution problem's initial data."""
    if hasattr(problem, "solution"):
        if problem.solution == "sin_pi":
            return np.sin(np.pi * x)
        return (1.0 - x**2) * np.exp(x)
    nu, t = problem.nu, problem.t_final
    if problem.equation == "advection":
        return initial(problem, wrap(x - nu * t))
    if problem.equation == "burgers":
        return cole_hopf(problem, x, initial)
    if problem.equation == "kdv":
        return initial(problem, x - t)
    if problem.initial == "gaussian":
        width = math.sqrt(0.1**2 + 2.0 * nu * t)
        images = [x - 2.0 * m for m in range(-4, 5)]
        return np.sum(
            [0.1 / width * np.exp(-(xi**2) / (2.0 * width**2)) for xi in images],
            axis=0,
        )
    return initial(problem, x) * math.exp(-nu * math.pi**2 * t)
