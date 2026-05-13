"""Minimal parametric inversion example."""

import math
import os
import sys
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

if __package__ is None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root not in sys.path:
        sys.path.insert(0, root)

from effectful.ops.semantics import evaluate, fwd, handler
from effectful.ops.syntax import ObjectInterpretation, defdata, defop, implements
from effectful.ops.types import NotHandled, Term

add = defdata.dispatch(float).__add__
mul = defdata.dispatch(float).__mul__
sub = defdata.dispatch(float).__sub__
div = defdata.dispatch(float).__truediv__

ThetaValue = float | int | Term


@defop
def floor(x: float) -> int:
    if not isinstance(x, Term):
        return math.floor(x)
    raise NotHandled


@defop
def floor_inv(y: float, u: ThetaValue):
    return y + u


@defop
def add_inv(y: float, other: ThetaValue):
    return [other, y - other]


@dataclass(frozen=True)
class Step:
    op: object
    inputs: list[float | Term]
    output: float | Term


class Trace(ObjectInterpretation):
    def __init__(self):
        super().__init__()
        self.steps: list[Step] = []
        self.theta_exprs: list[ThetaValue] = []
        self.z_unit_ops: list[object] = []

    def _new_unit(self) -> Term:
        op = defop(float, name=f"z_unit_{len(self.z_unit_ops) + 1}")
        self.z_unit_ops.append(op)
        return op()

    def _record(self, op: object, inputs: list[float | Term], theta: list[ThetaValue], output):
        self.steps.append(Step(op=op, inputs=inputs, output=output))
        self.theta_exprs.extend(theta)

    @implements(add)
    def on_add(self, x1: float | Term, x2: float | Term):
        out = fwd(x1, x2)
        if isinstance(out, Term):
            theta = [self._new_unit()] if isinstance(x1, Term) and isinstance(x2, Term) else []
            self._record(add, [x1, x2], theta, out)
        return out

    @implements(mul)
    def on_mul(self, x1: float | Term, x2: float | Term):
        out = fwd(x1, x2)
        if isinstance(out, Term):
            self._record(mul, [x1, x2], [], out)
        return out

    @implements(sub)
    def on_sub(self, x1: float | Term, x2: float | Term):
        out = fwd(x1, x2)
        if isinstance(out, Term):
            self._record(sub, [x1, x2], [], out)
        return out

    @implements(floor)
    def on_floor(self, x: float | Term):
        out = fwd(x)
        if isinstance(out, Term):
            self._record(floor, [x], [self._new_unit()], out)
        return out


def trace_theta(expr: Term) -> tuple[list[Step], list[ThetaValue], list[object]]:
    with handler(Trace()) as tr:
        evaluate(expr)
    return list(tr.steps), list(reversed(tr.theta_exprs)), list(tr.z_unit_ops)


def compile_theta_from_omega(
    theta_exprs: list[ThetaValue],
    *,
    x_ops: list[object],
    x_from_omega: dict[object, Callable[[float], float]],
):
    omega_ops = [defop(float, name=f"omega_{i + 1}") for i in range(len(x_ops))]

    def theta_from_omega(omega_values) -> list[ThetaValue]:
        env = {
            x_op: (lambda *_, v=x_from_omega[x_op](u), **__: v)
            for x_op, u in zip(x_ops, omega_values)
        }
        with handler(env):
            return [evaluate(t) if isinstance(t, Term) else t for t in theta_exprs]

    return omega_ops, theta_from_omega


def build_inverse(output_var: Term, steps: list[Step], thetas: list[ThetaValue]) -> dict[str, list[ThetaValue]]:
    theta_q = deque(thetas)
    expr_by_id: dict[int, ThetaValue] = {id(steps[-1].output): output_var}
    recovered: dict[str, list[ThetaValue]] = {}

    for step in reversed(steps):
        y = expr_by_id.pop(id(step.output))

        if step.op is floor:
            (u,) = [theta_q.popleft()]
            (x,) = step.inputs
            expr_by_id[id(x)] = floor_inv(y, u)
        elif step.op is add:
            x1, x2 = step.inputs
            if isinstance(x1, Term) and isinstance(x2, Term):
                (other,) = [theta_q.popleft()]
                inv1, inv2 = add_inv(y, other)
                expr_by_id[id(x1)] = inv1
                expr_by_id[id(x2)] = inv2
            elif isinstance(x1, Term):
                expr_by_id[id(x1)] = sub(y, x2)
            elif isinstance(x2, Term):
                expr_by_id[id(x2)] = sub(y, x1)
            else:
                raise ValueError("expected a symbolic add input")
        elif step.op is mul:
            x1, x2 = step.inputs
            if isinstance(x1, Term) and not isinstance(x2, Term):
                expr_by_id[id(x1)] = div(y, x2)
            elif not isinstance(x1, Term) and isinstance(x2, Term):
                expr_by_id[id(x2)] = div(y, x1)
            else:
                raise ValueError("minimal example only supports one symbolic mul input")
        elif step.op is sub:
            x1, x2 = step.inputs
            if isinstance(x1, Term) and not isinstance(x2, Term):
                expr_by_id[id(x1)] = add(y, x2)
            elif not isinstance(x1, Term) and isinstance(x2, Term):
                expr_by_id[id(x2)] = sub(x1, y)
            else:
                raise ValueError("minimal example only supports one symbolic sub input")
        else:
            raise ValueError(f"no inverse rule for op {step.op}")

    for step in steps:
        for x in step.inputs:
            if isinstance(x, Term) and id(x) in expr_by_id:
                recovered.setdefault(x.op.__name__, []).append(expr_by_id[id(x)])
    return recovered


def unit_interval(u: float) -> float:
    return min(1.0 - 1e-12, max(0.0, float(u)))


def run_example(name: str, expr: Term, y_obs: float, omega: list[float]):
    y = defop(float, name=f"y_{name}")
    steps, theta_exprs, z_ops = trace_theta(expr)
    theta_ops = [defop(float, name=f"{name}_theta_{i + 1}") for i in range(len(theta_exprs))]
    inv = build_inverse(y(), steps, [t() for t in theta_ops])
    omega_ops, theta_from_omega = compile_theta_from_omega(
        theta_exprs,
        x_ops=z_ops,
        x_from_omega={op: unit_interval for op in z_ops},
    )
    theta_vals = theta_from_omega(omega)

    env = {y: lambda: y_obs}
    env.update({op: (lambda *_, v=value, **__: v) for op, value in zip(theta_ops, theta_vals)})
    with handler(env):
        recovered = evaluate(inv)

    print(name)
    print("  omega =", {op.__name__: value for op, value in zip(omega_ops, omega)})
    print("  theta =", theta_vals)
    print("  y =", y_obs)
    print("  recovered =", {k: [float(v) for v in vals] for k, vals in recovered.items()})


if __name__ == "__main__":
    X = defop(float, name="X")

    run_example("floor", floor(mul(10.0, X())), y_obs=3.0, omega=[0.25])
    run_example("offset", floor(mul(10.0, add(X(), 1.0))), y_obs=3.0, omega=[0.25])
    run_example("repeat", add(X(), X()), y_obs=1.0, omega=[0.25])
