import contextlib
import itertools
import logging
from typing import TypeVar

import pytest
from typing_extensions import ParamSpec

from effectful.internals.prompts import bind_result, value_or_result
from effectful.internals.sugar import ObjectInterpretation, implements
from effectful.ops.core import Interpretation, Operation, define
from effectful.ops.handler import coproduct, fwd, handler
from effectful.ops.interpreter import interpreter

logger = logging.getLogger(__name__)

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")


@define(Operation)
def plus_1(x: int) -> int:
    return x + 1


@define(Operation)
def plus_2(x: int) -> int:
    return x + 2


@define(Operation)
def times_plus_1(x: int, y: int) -> int:
    return x * y + 1


def defaults(*ops: Operation[..., int]) -> Interpretation[int, int]:
    return {op: bind_result(value_or_result(op.default)) for op in ops}


def times_n_handler(n: int, *ops: Operation[..., int]) -> Interpretation[int, int]:
    return {op: bind_result(lambda v, *args, **kwargs: fwd(v) * n) for op in ops}


OPERATION_CASES = (
    [[plus_1, (i,)] for i in range(5)]
    + [[plus_2, (i,)] for i in range(5)]
    + [[times_plus_1, (i, j)] for i, j in itertools.product(range(5), range(5))]
)
N_CASES = [1, 2, 3]


@pytest.mark.parametrize("op,args", OPERATION_CASES)
def test_affine_continuation_compose(op, args):
    def f():
        return op(*args)

    h_twice = {op: bind_result(lambda v, *a, **k: fwd(fwd(v)))}

    assert (
        interpreter(defaults(op))(f)()
        == interpreter(coproduct(defaults(op), h_twice))(f)()
    )


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n1", N_CASES)
@pytest.mark.parametrize("n2", N_CASES)
def test_compose_associative(op, args, n1, n2):
    def f():
        return op(*args)

    h0 = defaults(op)
    h1 = times_n_handler(n1, op)
    h2 = times_n_handler(n2, op)

    intp1 = coproduct(h0, coproduct(h1, h2))
    intp2 = coproduct(coproduct(h0, h1), h2)

    assert interpreter(intp1)(f)() == interpreter(intp2)(f)()


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n1", N_CASES)
@pytest.mark.parametrize("n2", N_CASES)
def test_compose_commute_orthogonal(op, args, n1, n2):
    def f():
        return op(*args) + new_op(*args)

    new_op = define(Operation)(lambda *args: op(*args) + 3)

    h0 = defaults(op, new_op)
    h1 = times_n_handler(n1, op)
    h2 = times_n_handler(n2, new_op)

    intp1 = coproduct(h0, h1, h2)
    intp2 = coproduct(h0, h2, h1)

    assert interpreter(intp1)(f)() == interpreter(intp2)(f)()


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n1", N_CASES)
@pytest.mark.parametrize("n2", N_CASES)
def test_handler_associative(op, args, n1, n2):
    def f():
        return op(*args)

    h0 = defaults(op)
    h1 = times_n_handler(n1, op)
    h2 = times_n_handler(n2, op)

    expected = interpreter(coproduct(h0, h1, h2))(f)()

    with handler(h0), handler(h1), handler(h2):
        assert f() == expected

    with handler(coproduct(h0, h1)), handler(h2):
        assert f() == expected

    with handler(h0), handler(coproduct(h1, h2)):
        assert f() == expected


@pytest.mark.parametrize("op,args", OPERATION_CASES)
@pytest.mark.parametrize("n", N_CASES)
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_stop_without_fwd(op, args, n, depth):
    def f():
        return op(*args)

    expected = f()

    with contextlib.ExitStack() as stack:
        for _ in range(depth):
            stack.enter_context(handler(times_n_handler(n, op)))

        stack.enter_context(handler(defaults(op)))

        assert f() == expected


def test_sugar_subclassing():
    class ScaleBy(ObjectInterpretation):
        def __init__(self, scale):
            self._scale = scale

        @implements(plus_1)
        def plus_1(self, v):
            return v + self._scale

        @implements(plus_2)
        def plus_2(self, v):
            return v + 2 * self._scale

    class ScaleAndShiftBy(ScaleBy):
        def __init__(self, scale, shift):
            super().__init__(scale)
            self._shift = shift

        @implements(plus_1)
        def plus_1(self, v):
            return super().plus_1(v) + self._shift

        # plus_2 inhereted from ScaleBy

    with handler(ScaleBy(4)):
        assert plus_1(4) == 8
        assert plus_2(4) == 12

    with handler(ScaleAndShiftBy(4, 1)):
        assert plus_1(4) == 9
        assert plus_2(4) == 12


def test_lazy_1():
    from effectful.ops.core import Constant, Variable, Term

    @Operation
    def Add(x: int, y: int) -> int:
        raise NotImplementedError

    @bind_result
    def eager_add(_, x, y):
        if isinstance(x, type(one)) and isinstance(y, type(one)):
            return Constant(x.value + y.value)
        else:
            return fwd(None)

    eager = {Add: eager_add}

    @bind_result
    def lazy_add(_, x, y):
        return Term(Add, (x, y), {})

    lazy = {Add: lazy_add}

    @bind_result
    def simplify_add(_, x, y):
        if not isinstance(x, type(one)) and isinstance(y, type(one)):
            # x + c -> c + x
            return Add(y, x)
        elif isinstance(y, type(tm)):
            # a + (b + c) -> (a + b) + c
            return Add(Add(x, y.args[0]), y.args[1])
        else:
            return fwd(None)

    simplify_assoc_commut = {Add: simplify_add}

    @bind_result
    def unit_add(_, x, y):
        if x == zero:
            return y
        elif y == zero:
            return x
        else:
            return fwd(None)

    simplify_unit = {Add: unit_add}

    mixed = coproduct(lazy, eager)
    simplified = coproduct(simplify_assoc_commut, simplify_unit)
    mixed_simplified = coproduct(mixed, simplified)

    x, y, z = Variable("x", int), Variable("y", int), Variable("z", int)
    zero, one, two, three = Constant(0), Constant(1), Constant(2), Constant(3)
    tm = Term(Add, (one, two), {})

    with interpreter(eager):
        assert Add(one, two) == three

    with interpreter(lazy):
        assert Add(one, two) == Term(Add, (one, two), {})
        assert Add(one, Add(two, three)) == Term(Add, (one, Term(Add, (two, three), {})), {})
        assert Add(x, y) == Term(Add, (x, y), {})
        assert Add(x, one) != Add(y, one)

    with interpreter(mixed):
        assert Add(one, two) == three
        assert Add(Add(one, two), x) == Term(Add, (three, x), {})

    with interpreter(mixed_simplified):
        assert Add(one, two) == three
        assert Add(three, x) == Add(x, three)
        assert Add(Add(one, two), x) == Add(x, three)
        assert Add(one, Add(x, two)) == Add(x, three)
        assert Add(Add(one, Add(y, one)), Add(one, Add(x, one))) == Add(Add(y, x), Constant(4))

        assert Add(one, Add(Add(x, y), two)) == Add(Add(x, y), three)
        assert Add(one, Add(Add(x, Add(y, one)), one)) == Add(Add(x, y), three)

        assert Add(Add(Add(x, x), Add(x, x)), Add(Add(x, x), Add(x, x))) == \
            Add(Add(Add(Add(Add(Add(Add(x, x), x), x), x), x), x), x) == \
            Add(x, Add(x, Add(x, Add(x, Add(x, Add(x, Add(x, x)))))))

        assert Add(x, zero) == x


def test_lazy_2():
    from effectful.ops.core import Constant, Variable, Term, evaluate

    @Operation
    def Add(x: int, y: int) -> int:
        raise NotImplementedError

    @Operation
    def Lam(var: Variable, body: Term) -> Term:
        raise NotImplementedError
    
    @Operation
    def App(f: Term, arg: Term) -> Term:
        raise NotImplementedError

    def substitute(subs: dict[Variable, Term | Variable | Constant]):

        def _traverse(term: Term):  # TODO fix evaluate and use that instead
            if isinstance(term, type(one)):
                return term
            elif isinstance(term, type(x)):
                return subs.get(term, term)
            elif term.op == Lam:
                var, body = term.args
                return Lam(var, _traverse(body))
            else:
                return term.op(
                    *[_traverse(a) for a in term.args],
                    **{k: _traverse(v) for k, v in term.kwargs.items()}
                )

        return _traverse
    
    def fvs(term: Term):
        # TODO fix evaluate and use that instead
        if isinstance(term, type(one)):
            return set()
        elif isinstance(term, type(x)):
            return {term}
        else:
            return set.union(*[fvs(a) for a in term.args], *[fvs(v) for v in term.kwargs.values()])

    def lazy_op(op):
        @bind_result
        def _lazy_op(_, *args):
            return Term(op, args, {})
        return _lazy_op

    lazy = {Add: lazy_op(Add), Lam: lazy_op(Lam), App: lazy_op(App)}

    @bind_result
    def alpha_lam(_, var: Variable, body: Term):
        """alpha reduction"""
        if not var.symbol.startswith("mangled_"):
            # TODO mangle more aggressively to avoid collisions
            mangled_var = Variable("mangled_" + var.symbol, var.type)
            mangled_body = substitute({var: mangled_var})(body)
            return Lam(mangled_var, mangled_body)
        else:
            return fwd(None)

    alpha_conversion = {Lam: alpha_lam}

    @bind_result
    def eager_add(_, x, y):
        if isinstance(x, type(Constant(0))) and isinstance(y, type(Constant(0))):
            return Constant(x.value + y.value)
        else:
            return fwd(None)

    @bind_result
    def eager_app(_, f: Term | Variable, arg: Term):
        """beta reduction"""
        if isinstance(f, type(tm)) and f.op == Lam:
            var, body = f.args
            return substitute({var: arg})(body)
        else:
            return fwd(None)

    @bind_result
    def eta_lam(_, var: Variable, body: Term):
        """eta reduction"""
        if var not in fvs(body):
            return body
        else:
            return fwd(None)

    eager = coproduct(coproduct(lazy, alpha_conversion), {Add: eager_add, App: eager_app, Lam: eta_lam})

    x, y, z = Variable("x", int), Variable("y", int), Variable("z", int)
    zero, one, two, three = Constant(0), Constant(1), Constant(2), Constant(3)
    tm = Term(Add, (one, two), {})

    with interpreter(eager):
        f1 = Lam(x, Add(x, one))
        assert substitute({x: one})(f1) == substitute({y: one})(f1) == f1
        assert App(f1, one) == two
        assert Lam(y, f1) == f1

        f2 = Lam(x, Lam(y, Add(x, y)))
        assert App(App(f2, one), two) == three
        assert Lam(y, f2) == f2

        app2 = Lam(z, Lam(x, Lam(y, App(App(z, x), y))))
        assert App(App(App(app2, f2), one), two) == three

        compose = Lam(x, Lam(y, Lam(z, App(x, App(y, z)))))
        f1_twice = App(App(compose, f1), f1)
        assert App(f1_twice, one) == three


def test_bind_with_handler():
    import functools
    from effectful.ops.core import Constant, Term, evaluate
    
    @Operation
    def Add(x: int, y: int) -> int:
        raise NotImplementedError

    @Operation
    def Lam(var: Term, body: Term) -> Term:
        raise NotImplementedError
    
    @Operation
    def App(f: Term, arg: Term) -> Term:
        raise NotImplementedError

    def substitution(subs: dict[str, Term | Constant]):

        def _traverse(term: Term):  # TODO fix evaluate and use that instead
            if isinstance(term, (str, type(one))):
                return term
            else:
                return term.op(
                    *[_traverse(a) for a in term.args],
                    **{k: _traverse(v) for k, v in term.kwargs.items()}
                )

        intp = {_genvar(v): functools.partial(lambda x: x, sub) for v, sub in subs.items()}
        return interpreter(intp)(_traverse)
    
    def lazy_op(op):
        @bind_result
        def _lazy_op(_, *args):
            return Term(op, args, {})
        return _lazy_op

    @Operation
    def getvar(v: str):
        return _genvar(v)()

    @functools.cache
    def _genvar(v: str) -> Operation:
        return Operation(lambda: Term(getvar, (v,), {}))

    lazy = {Add: lazy_op(Add), Lam: lazy_op(Lam), App: lazy_op(App)}

    @bind_result
    def alpha_lam(_, var: Term, body: Term):
        """alpha reduction"""
        if not var.args[0].startswith("mangled_"):
            # TODO mangle more aggressively to avoid collisions
            mangled_var = getvar("mangled_" + var.args[0])
            mangled_body = substitution({var.args[0]: mangled_var})(body)
            return Lam(mangled_var, mangled_body)
        else:
            return fwd(None)

    alpha_conversion = {Lam: alpha_lam}

    @bind_result
    def eager_add(_, x, y):
        if isinstance(x, type(Constant(0))) and isinstance(y, type(Constant(0))):
            return Constant(x.value + y.value)
        else:
            return fwd(None)

    @bind_result
    def eager_app(_, f: Term, arg: Term):
        """beta reduction"""
        if f.op == Lam:
            var, body = f.args
            return substitution({var.args[0]: arg})(body)
        else:
            return fwd(None)

    def fvs(term: Term) -> set:
        env = set()

        @bind_result
        def getvar_scope(_, v: str):
            env.add(v)
            return fwd(None)

        @bind_result
        def lam_scope(_, var: Term, body: Term):
            env.remove(var.args[0])
            return fwd(None)

        intp_scope = {getvar: getvar_scope, Lam: lam_scope}
        intp_lazy = coproduct(coproduct(lazy, {getvar: lazy_op(getvar)}), intp_scope)
        interpreter(intp_lazy)(substitution({}))(term)
        return env

    @bind_result
    def eta_lam(_, var: Term, body: Term):
        """eta reduction"""
        if var.args[0] not in fvs(body):
            return body
        else:
            return fwd(None)

    eager = coproduct(coproduct(lazy, alpha_conversion), {Add: eager_add, App: eager_app, Lam: eta_lam})

    x, y, z = getvar("x"), getvar("y"), getvar("z")
    zero, one, two, three = Constant(0), Constant(1), Constant(2), Constant(3)

    with interpreter(eager):
        f1 = Lam(x, Add(x, one))
        assert substitution({x.args[0]: one})(f1) == substitution({y.args[0]: one})(f1) == f1
        assert App(f1, one) == two
        assert Lam(y, f1) == f1

        f2 = Lam(x, Lam(y, Add(x, y)))
        assert App(App(f2, one), two) == three
        assert Lam(y, f2) == f2

        app2 = Lam(z, Lam(x, Lam(y, App(App(z, x), y))))
        assert App(App(App(app2, f2), one), two) == three

        compose = Lam(x, Lam(y, Lam(z, App(x, App(y, z)))))
        f1_twice = App(App(compose, f1), f1)
        assert App(f1_twice, one) == three
