import collections.abc
import dataclasses
import functools
import itertools
import numbers
from collections.abc import Callable, Generator, Mapping
from typing import Any

import effectful.handlers.jax.numpy as jnp
import effectful.handlers.numbers  # noqa: F401
import jax
import tree
from effectful.handlers.jax._handlers import is_eager_array
from effectful.ops.semantics import handler
from effectful.ops.syntax import defop
from effectful.ops.types import Interpretation, Term


@dataclasses.dataclass
class Monoid[T]:
    add: Callable[[T, T], T]
    zero: T
    name: str | None = None

    def __init__(self, add, zero, name=None):
        self.add = add
        self.zero = zero
        self.name = name
        self.typ = type(zero)

    def __str__(self):
        if self.name is None:
            return repr(self)
        return self.name

    def __call__(self, x, y):
        if isinstance(x, self.typ) and x == self.zero:
            return y
        if isinstance(y, self.typ) and y == self.zero:
            return x
        return self.add(x, y)

    def is_idempotent(self) -> bool:
        """
        Check whether a monoid is idempotent.

        A monoid (A, ⊕, e) is idempotent if
            ∀x ∈ A: x ⊕ x = x
        """
        return self.add in (min, max, arg_min, arg_max)

    def scalar_mul(self):
        """
        Returns the scalar multiplication w.r.t. a monoid.

        The scalar multiplication of a monoid (A, ⊕, e) is
        a function (⋅): A × ℕ → A, inductively defined as
            a⋅0 = e
            a⋅n = a⋅(n-1) + a

        Warning: scalar multiplication is not commutative,
            the scalar is always the second argument.
        """
        if self.is_idempotent():
            return lambda x, _: x
        if self.add is add:
            return mul
        if self.add is logaddexp:
            return add
        if self.add is mul:
            return pow
        raise ValueError("Unknown")

    def distributes_with(self, op) -> bool:
        """
        Returns whether a monoid is distributive w.r.t. a jax op.
        """
        if self.add is add:
            return op is jnp.multiply
        elif self.add is min:
            return op is jnp.add
        elif self.add is max:
            return op is jnp.multiply or op is jnp.min or op is jnp.add
        elif self.add is logaddexp:
            return op is jnp.add
        else:
            return False


# Semiring laws:
# (R, +) is a commutative monoid with identity 0
# (R, *) is a monoid with identity 1
# a * 0 = 0 = 0 * a
# a * (b + c) = (a * b) + (a * c)
# (b + c) * a = (b * a) + (c * a)


@defop
def add[T](a: T, b: T) -> T:
    if any(isinstance(x, Term) and not is_eager_array(x) for x in (a, b)):
        raise NotImplementedError
    return a + b  # type: ignore


@defop
def mul[T](a: T, b: T) -> T:
    if (isinstance(a, numbers.Number) and a == 0) or (
        isinstance(b, numbers.Number) and b == 0
    ):
        return 0  # type: ignore
    if any(isinstance(x, Term) and not is_eager_array(x) for x in (a, b)):
        raise NotImplementedError
    return a * b  # type: ignore


@defop
def min[T](a: T, b: T) -> T:
    if any(isinstance(x, Term) for x in (a, b)):
        raise NotImplementedError
    return a if a < b else b  # type: ignore


@defop
def max[T](a: T, b: T) -> T:
    if any(isinstance(x, Term) for x in (a, b)):
        raise NotImplementedError
    return a if a > b else b  # type: ignore


@defop
def arg_min(a, b):
    if isinstance(a, tuple) and isinstance(a[0], numbers.Number) and a[0] == float("inf"):
        return b
    if isinstance(b, tuple) and isinstance(b[0], numbers.Number) and b[0] == float("inf"):
        return a
    if any(isinstance(x, Term) for x in tree.flatten((a, b))):
        raise NotImplementedError
    return a if a[0] < b[0] else b


@defop
def arg_max(a, b):
    if (
        isinstance(a, tuple)
        and isinstance(a[0], numbers.Number)
        and a[0] == float("-inf")
    ):
        return b
    if (
        isinstance(b, tuple)
        and isinstance(b[0], numbers.Number)
        and b[0] == float("-inf")
    ):
        return a
    if any(isinstance(x, Term) for x in tree.flatten((a, b))):
        raise NotImplementedError
    return a if a[0] > b[0] else b


@defop
def logaddexp(a, b):
    return jnp.logaddexp(a, b)


@defop
def tensor_cartesian_prod(xs, ys):
    x, y = jnp.meshgrid(xs, ys, indexing="ij")
    return jnp.stack([x.ravel(), y.ravel()]).T


def _promote_add(add, a, b):
    if isinstance(a, Generator):
        assert isinstance(b, Generator)
        return (v for v in (*a, *b))
    elif isinstance(a, Mapping):
        assert isinstance(b, Mapping)
        result = {
            k: a[k]
            if k not in b
            else b[k]
            if k not in a
            else _promote_add(add, a[k], b[k])
            for k in set(a) | set(b)
        }
        return result
    elif callable(a):
        assert callable(b)
        return lambda *args, **kwargs: _promote_add(
            add, a(*args, **kwargs), b(*args, **kwargs)
        )
    elif isinstance(a, Interpretation):
        assert isinstance(b, Interpretation)
        assert a.keys() == b.keys()
        result = {k: _promote_add(add, handler(a)(a[k]), handler(b)(b[k])) for k in a}
        return result
    else:
        return add(a, b)


def promote(self):
    add = functools.partial(_promote_add, self.add)
    return Monoid(add, self.zero, "Promoted" + self.name)


SumMonoid: Monoid[float] = Monoid(add, 0.0, "Sum")

ProdMonoid: Monoid[float] = Monoid(mul, 1.0, "Prod")

LogSumMonoid: Monoid[float] = Monoid(logaddexp, float("-inf"), "LogSum")

MinMonoid: Monoid[float] = Monoid(min, float("inf"), "Min")

MaxMonoid: Monoid[float] = Monoid(max, float("-inf"), "Max")

ArgMinMonoid: Monoid[tuple[float, Any]] = Monoid(arg_min, (float("inf"), None), "ArgMin")

ArgMaxMonoid: Monoid[tuple[float, Any]] = Monoid(arg_max, (float("-inf"), None), "ArgMax")

TensorCartesianProdMonoid: Monoid[jax.Array] = Monoid(
    tensor_cartesian_prod, jnp.array([]), "TensorCartesionProd"
)

StreamChainMonoid: Monoid[collections.abc.Generator] = Monoid(
    lambda a, b: (v for v in itertools.chain(a, b)), (), name="StreamChain"
)

# note: empty tuple is not a valid identity
StreamProdMonoid: Monoid[collections.abc.Generator] = Monoid(
    lambda a, b: ((v1, v2) for (v1, v2) in itertools.product(a, b)), (), name="StreamProd"
)
