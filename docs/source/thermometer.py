"""
This is an implementation of Koppel et. al's 2018 paper "Capturing the Future by Replaying the Past".
"""

from dataclasses import dataclass
from collections import deque

from effectful.internals.sugar import ObjectInterpretation, implements
from effectful.ops.core import Operation
from effectful.ops.handler import handler


@Operation
def flip():
    raise NotImplementedError


@Operation
def fail():
    raise NotImplementedError


def choose(*options):
    for option in options:
        if flip():
            return option

    return fail()


@dataclass
class Thermometer(ObjectInterpretation):
    route: list[bool]

    class Flip(BaseException):
        pass

    class Fail(BaseException):
        pass

    @implements(flip)
    def do_flip(self):
        if not self.route:
            raise Thermometer.Flip
        else:
            return self.route.popleft()

    @implements(fail)
    def do_fail(self):
        raise Thermometer.Fail

def thermometer(thunk):
    routes = [deque()]
    results = []

    while routes:
        route = routes.pop()

        try:
            with handler(Thermometer(route.copy())):
                results.append(thunk())
        except Thermometer.Flip:
            r1 = route.copy()
            r1.append(False)

            r2 = route
            r2.append(True)

            routes.append(r1)
            routes.append(r2)
        except Thermometer.Fail:
            pass

    return list(results)


def program_1():
    return 3 * choose(7, 5)

def program_2():
    return choose(1, 2) if flip() else choose(3, 4)

def program_3():
    return choose(1, choose(2, 3), 4)

def program_4():
    return thermometer(choose(program_1, program_2))
