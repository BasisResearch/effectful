"""This example demonstrates a beam search over a program that uses a `choose`
effect for nondeterminism and `score` effect to weigh its choices.

"""

import functools
import heapq
import random
import typing
from collections.abc import Callable
from dataclasses import dataclass
from pprint import pprint

from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, defop, implements


@defop
def choose[T](choices: list[T]) -> T:
    result = random.choice(choices)
    print(f"choose({choices}) = {result}")
    return result


@defop
def score(value: float) -> None:
    pass


class Suspend(Exception): ...


class ReplayIntp(ObjectInterpretation):
    def __init__(self, trace):
        self.trace = trace
        self.step = 0

    @implements(choose)
    def _(self, *args, **kwargs):
        if self.step < len(self.trace):
            result = self.trace[self.step][1]
            self.step += 1
            return result
        return fwd()


class TraceIntp(ObjectInterpretation):
    def __init__(self):
        self.trace = []

    @implements(choose)
    def _(self, *args, **kwargs):
        result = fwd()
        self.trace.append(((args, kwargs), result))
        return result


class ScoreIntp(ObjectInterpretation):
    def __init__(self):
        self.score = 0.0

    @implements(score)
    def _(self, value):
        self.score += value


class ChooseOnceIntp(ObjectInterpretation):
    def __init__(self):
        self.is_first_call = True

    @implements(choose)
    def _(self, *args, **kwargs):
        if not self.is_first_call:
            raise Suspend

        self.is_first_call = False
        return fwd()


@dataclass
class BeamCandidate[S, T]:
    """Represents a candidate execution path in beam search."""

    trace: list[S]
    score: float
    in_progress: bool
    result: T | None

    def __lt__(self, other: "BeamCandidate[S, T]") -> bool:
        return self.score < other.score

    def expand[**P](self, model_fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs):
        in_progress = False
        result = None
        score_intp = ScoreIntp()
        trace_intp = TraceIntp()
        with (
            handler(score_intp),
            handler(ChooseOnceIntp()),
            handler(ReplayIntp(self.trace)),
            handler(trace_intp),
        ):
            try:
                result = model_fn(*args, **kwargs)
            except Suspend:
                in_progress = True

        return BeamCandidate(trace_intp.trace, score_intp.score, in_progress, result)


def beam_search[**P, S, T](
    model_fn: Callable[P, T], beam_width=3
) -> Callable[P, BeamCandidate[S, T]]:
    @functools.wraps(model_fn)
    def wrapper(*args, **kwargs):
        beam = [BeamCandidate([], 0.0, True, None)]

        while True:
            expandable = [c for c in beam if c.in_progress] * beam_width
            if not expandable:
                return beam

            new_candidates = [c.expand(model_fn, *args, **kwargs) for c in expandable]

            for c in new_candidates:
                heapq.heappushpop(beam, c) if len(
                    beam
                ) >= beam_width else heapq.heappush(beam, c)

    return wrapper


if __name__ == "__main__":

    def model():
        s1 = choose(range(100))
        score(s1)
        s2 = choose(range(-100, 100))
        score(s2)
        s3 = choose(range(-100, 100))
        score(s3)
        return s3

    result: BeamCandidate = beam_search(model)()
    pprint(result)
