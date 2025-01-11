import functools
import operator

import effectful.handlers.operator
from effectful.handlers.operator import OPERATORS
from effectful.ops.semantics import coproduct, evaluate, fwd, handler
from effectful.ops.syntax import defop
from effectful.ops.types import Term

add = OPERATORS[operator.add]


def beta_add(x: int, y: int) -> int:
    match x, y:
        case int(), int():
            return x + y
        case _:
            return fwd(None)


def commute_add(x, y):
    match x, y:
        case Term(), int():
            return y + x
        case _:
            return fwd(None)


def assoc_add(x, y):
    match x, y:
        case _, Term(op, (a, b)) if op == add:
            return (x + a) + b
        case _:
            return fwd(None)


beta_rules = {add: beta_add}
commute_rules = {add: commute_add}
assoc_rules = {add: assoc_add}

eager_mixed = functools.reduce(coproduct, (beta_rules, commute_rules, assoc_rules))

x = defop(int, name="x")
y = defop(int, name="y")

e = 1 + 1 + (x() + 1) + (5 + y())
print(e)

with handler(eager_mixed):
    print(evaluate(e))
