"""
This is an implementation of Koppel et. al's 2018 paper "Capturing the Future by Replaying the Past".
"""

from dataclasses import dataclass

from effectful.internals.sugar import ObjectInterpretation, implements
from effectful.ops.core import Operation
from effectful.ops.handler import handler


@Operation
def flip():
    raise NotImplementedError


@Operation
def fail():
    raise NotImplementedError


@dataclass
class Thermometer(ObjectInterpretation):
    future: list[bool]
    past: list[bool]

    class Flip(BaseException):
        pass

    class Fail(BaseException):
        pass

    @implements(flip)
    def do_flip(self):
        if self.future:
            pass

    @implements(fail)
    def do_fail(self):
        raise Thermometer.Fail


def choose(*objs):
    for o in objs:
        if flip():
            return o

    return fail()


def thermometer(thunk):
    results = []

    for lhs in [True, False]:
        try:
            with handler(Thermometer(lhs)):
                results.append(thunk())
        except Thermometer.Fail:
            pass

    return results


def program_1():
    return 3 * (5 if flip() else 7)
