import collections.abc
import functools
import itertools
from collections.abc import Mapping
from typing import ParamSpec, TypeAlias, TypeVar

import effectful.handlers.numbers  # noqa: F401
from effectful.ops.semantics import evaluate, handler
from effectful.ops.syntax import ObjectInterpretation, deffn, defop, implements
from effectful.ops.types import Interpretation, Term

from .semiring import Semiring

P = ParamSpec("P")
S = TypeVar("S")
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
A = TypeVar("A")
B = TypeVar("B")

Runner: TypeAlias = Interpretation[T, collections.abc.Iterable[T]]


@defop
def D(*args) -> dict:
    if not all(isinstance(kv, tuple) and len(kv) == 2 for kv in args):
        raise ValueError("Expected a sequence of key-value pairs")
    raise NotImplementedError


@defop
def fold(semiring: Semiring[T], streams: Runner, body: Mapping[K, T]) -> Mapping[K, T]:
    raise NotImplementedError


class BaselineFold(ObjectInterpretation):
    @implements(fold)
    def fold(semiring, streams, body):
        if any(isinstance(v, Term) for v in streams.values()):
            raise NotImplementedError

        def promote_add(add, a, b):
            if isinstance(b, collections.abc.Generator) or isinstance(
                a, collections.abc.Generator
            ):
                a = a if isinstance(a, collections.abc.Generator) else (a,)
                b = b if isinstance(b, collections.abc.Generator) else (b,)
                return (v for v in (*a, *b))
            elif isinstance(b, collections.abc.Mapping):
                result = {
                    k: a[k]
                    if k not in b
                    else b[k]
                    if k not in a
                    else promote_add(add, a[k], b[k])
                    for k in set(a) | set(b)
                }
                return result
            elif isinstance(b, collections.abc.Callable):
                return lambda *args, **kwargs: promote_add(
                    add, a(*args, **kwargs), b(*args, **kwargs)
                )
            else:
                return add(a, b)

        def to_dict(*args):
            if any((isinstance(k, Term) or isinstance(v, Term)) for (k, v) in args):
                raise NotImplementedError
            return dict(*args)

        def generator() -> collections.abc.Iterable[Mapping[K, T]]:
            all_vals = itertools.product(*list(streams.values()))
            for vals in all_vals:
                keys = list(streams.keys())
                with handler({k: deffn(v) for (k, v) in zip(keys, vals, strict=True)}):
                    b = evaluate(body)
                    if isinstance(b, Term) and b.op is D:
                        yield dict(b.args)  # type: ignore
                    else:
                        yield b  # type: ignore

        return functools.reduce(functools.partial(promote_add, semiring.add), generator())
