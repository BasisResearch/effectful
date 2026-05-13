import collections.abc
import functools
import itertools
import operator
import typing
from collections import Counter, defaultdict
from collections.abc import Callable, Generator, Iterable, Mapping
from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Annotated, Any

from effectful.internals.disjoint_set import DisjointSet
from effectful.ops.semantics import coproduct, evaluate, fvsof, fwd, handler, typeof
from effectful.ops.syntax import (
    ObjectInterpretation,
    Scoped,
    deffn,
    implements,
    iter_,
    syntactic_eq,
    syntactic_hash,
)
from effectful.ops.types import (
    Expr,
    Interpretation,
    NotHandled,
    Operation,
    Term,
)

# Note: The streams value type should be something like Iterable[T], but some of
# our target stream types (e.g. jax.Array) are not subtypes of Iterable
type Streams[T] = Mapping[Operation[[], T], Any]

type Body[T] = (
    Iterable[T]
    | Callable[..., Body[T]]
    | T
    | Mapping[Any, Body[T]]
    | Interpretation[T, Body[T]]
)


def outer_stream(
    streams: Streams,
) -> Iterable[tuple[Operation, Expr, dict[Operation, Expr]]]:
    """Returns the streams that can be ordered outermost in the loop nest as
    well as the remaining streams in the nest.

    """
    stream_vars = set(streams.keys())
    pred = {k: fvsof(v) & stream_vars for k, v in streams.items()}
    topo = TopologicalSorter(pred)
    topo.prepare()
    return (
        (op, streams[op], {k: v for (k, v) in streams.items() if k != op})
        for op in topo.get_ready()
    )


class Monoid[T]:
    """A monoid with per-instance dispatch tables for ``plus`` and ``reduce``.

    Each instance owns its own :func:`functools.singledispatch` registries.
    Backends and call sites extend behavior via ``instance.plus.register(type)``
    and ``instance.reduce.register(type)``. Rewrites that key on the class-level
    ``Monoid.plus`` / ``Monoid.reduce`` operations still fire, because
    per-instance operations delegate to them.
    """

    _name: str
    identity: T

    def __init__(self, identity: T, name: str):
        self._name = name
        self.identity = identity

        # per-instance dispatch tables
        self._plus_dispatch = functools.singledispatch(self._plus_default)
        self._reduce_dispatch = functools.singledispatch(self._reduce_default)

        # expose register/dispatch on the per-instance cached operations
        plus_op = type(self).plus.__get__(self, type(self))
        plus_op.register = self._plus_dispatch.register
        plus_op.dispatch = self._plus_dispatch.dispatch

        reduce_op = type(self).reduce.__get__(self, type(self))
        reduce_op.register = self._reduce_dispatch.register
        reduce_op.dispatch = self._reduce_dispatch.dispatch

    def __repr__(self):
        return f"Monoid({self._name!r})"

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))

    def _plus_default(self, *args):
        raise TypeError(f"Unexpected arguments to {self._name}.plus")

    def _plus_iterable(self, *args):
        """Broadcast plus elementwise over an iterable, preserving input type
        for tuples and lists; generators stay generators.
        """
        zipped = zip(*args, strict=True)
        result = (self.plus(*vs) for vs in zipped)
        if isinstance(args[0], tuple | list):
            return type(args[0])(result)
        return result

    def _plus_mapping(self, *args):
        if isinstance(args[0], Interpretation):
            keys = args[0].keys()
            for b in args[1:]:
                if not isinstance(b, Interpretation):
                    raise TypeError(f"Expected interpretation but got {b}")
                if not keys == b.keys():
                    raise ValueError(
                        f"Expected interpretation of {keys} but got {b.keys()}"
                    )
            return {k: self.plus(*(handler(b)(b[k]) for b in args)) for k in keys}

        for b in args[1:]:
            if not isinstance(b, Mapping):
                raise TypeError(f"Expected mapping but got {b}")
        all_values = collections.defaultdict(list)
        for d in args:
            for k, v in d.items():
                all_values[k].append(v)
        return {k: self.plus(*vs) for (k, vs) in all_values.items()}

    def _reduce_default(self, body, streams):
        if not streams:
            return self.identity

        # find and reduce a ground stream
        for stream_key, stream_body, streams_tail in outer_stream(streams):
            if isinstance(stream_body, Term):
                continue

            stream_values_iter = iter(stream_body)

            # if we iterate and get a term instead of a real iterator, skip
            if isinstance(stream_values_iter, Term) and stream_values_iter.op is iter_:
                continue

            new_reduces = []
            for stream_val in stream_values_iter:
                with handler({stream_key: deffn(stream_val)}):
                    eval_args = evaluate((body, streams_tail))
                    assert isinstance(eval_args, tuple)
                    new_reduces.append(self.reduce(*eval_args))

            return self.plus(*new_reduces)

        raise NotHandled

    def _reduce_callable(self, body, streams):
        return lambda *a, **k: self.reduce(body(*a, **k), streams)

    def _reduce_mapping(self, body, streams):
        return {k: self.reduce(v, streams) for (k, v) in body.items()}

    def _reduce_iterable(self, body, streams):
        """Broadcast reduce elementwise over an iterable, preserving input type
        for tuples and lists; generators stay generators.
        """
        result = (self.reduce(x, streams) for x in body)
        if isinstance(body, tuple | list):
            return type(body)(result)
        return result

    # --- public operations -------------------------------------------------

    @Operation.define
    def plus[S](self, *args: S) -> S:
        """Monoid addition with broadcasting over common collection types,
        callables, and interpretations.

        Any :class:`Term` arg routes to symbolic evaluation. Registered
        handlers can therefore assume their args are concrete values.
        Composite handlers (tuple/list, Mapping) recurse through
        :meth:`plus` so interior Terms are caught at the next call.
        """
        if not args:
            return typing.cast(S, self.identity)
        if any(isinstance(x, Term) for x in args):
            raise NotHandled
        return self._plus_dispatch.dispatch(type(args[0]))(*args)

    @Operation.define
    def reduce[A, B, U: Body](
        self,
        body: Annotated[U, Scoped[A | B]],
        streams: Annotated[Streams, Scoped[A]],
    ) -> Annotated[U, Scoped[B]]:
        """Reduce ``body`` over ``streams``."""
        return self._reduce_dispatch.dispatch(typeof(body))(body, streams)


class MonoidWithZero[T](Monoid[T]):
    zero: T

    def __init__(self, name: str, identity: T, zero: T):
        super().__init__(name=name, identity=identity)
        self.zero = zero


def _register_sequence_broadcasting(monoid: "Monoid") -> None:
    """Register elementwise broadcasting over tuples, lists, and generators.

    Appropriate when the monoid's values are scalars and these collections are
    containers to broadcast over. Skipped by monoids whose values *are*
    sequences (e.g. :data:`CartesianProduct`) or whose tuples carry meaning
    other than "container" (e.g. :data:`ArgMin` / :data:`ArgMax`, where a
    tuple is a (score, value) pair).
    """
    monoid.plus.register(tuple | list | Generator)(monoid._plus_iterable)
    monoid.reduce.register(tuple | list | Generator)(monoid._reduce_iterable)


def _register_mapping_broadcasting(monoid: "Monoid") -> None:
    """Register broadcasting over dict-like containers and interpretations.

    Safe for any monoid: mappings carry one value per key, and broadcasting
    merges per-key values via the monoid.
    """
    monoid.plus.register(Mapping)(monoid._plus_mapping)
    monoid.reduce.register(Mapping)(monoid._reduce_mapping)


def _register_callable_broadcasting(monoid: "Monoid") -> None:
    """Register lifting of :meth:`reduce` under callables.

    ``monoid.reduce(f, streams)`` becomes ``lambda *a: monoid.reduce(f(*a),
    streams)``. Safe for any monoid.
    """
    monoid.reduce.register(Callable)(monoid._reduce_callable)


_cartesian_product_id_op = Operation.define(object, name="CartesianProductId")

Min = Monoid(name="Min", identity=float("inf"))
Max = Monoid(name="Max", identity=-float("inf"))
ArgMin = Monoid(name="ArgMin", identity=(Min.identity, None))
ArgMax = Monoid(name="ArgMax", identity=(Max.identity, None))
Sum = Monoid(name="Sum", identity=0)
Product = MonoidWithZero(name="Product", identity=1, zero=0)
CartesianProduct = Monoid(name="CartesianProduct", identity=_cartesian_product_id_op())

# Scalar-valued monoids: tuples/lists/generators are containers to broadcast over.
for _m in (Min, Max, Sum, Product):
    _register_sequence_broadcasting(_m)

# Mapping and callable broadcasting are safe for every monoid.
for _m in (Min, Max, ArgMin, ArgMax, Sum, Product, CartesianProduct):
    _register_mapping_broadcasting(_m)
    _register_callable_broadcasting(_m)


@Min.plus.register(int | float)
def _(*args):
    return min(args)


@Max.plus.register(int | float)
def _(*args):
    return max(args)


@Sum.plus.register(int | float)
def _(*args):
    return sum(args)


@Product.plus.register(int | float)
def _(*args):
    return functools.reduce(operator.mul, args)


@ArgMin.plus.register(tuple)
def _(*args):
    if not all(isinstance(a[0], int | float) for a in args):
        raise NotHandled
    return min(args, key=lambda a: a[0])


@ArgMax.plus.register(tuple)
def _(*args):
    if not all(isinstance(a[0], int | float) for a in args):
        raise NotHandled
    return max(args, key=lambda a: a[0])


@CartesianProduct.plus.register(Iterable)
def _(*args):
    def to_tuple(x):
        return x if isinstance(x, tuple) else (x,)

    return [sum((to_tuple(v) for v in vals), ()) for vals in itertools.product(*args)]


@dataclass
class _ExtensiblePredicate[T]:
    elems: set[T]

    def register(self, t: T) -> None:
        self.elems.add(t)

    def __call__(self, t: T) -> bool:
        return t in self.elems


is_commutative = _ExtensiblePredicate({Max, Min, Sum, Product})
is_idempotent = _ExtensiblePredicate({Max, Min})


@dataclass
class _ExtensibleBinaryRelation[S, T]:
    tuples: set[tuple[S, T]]

    def register(self, s: S, t: T) -> None:
        self.tuples.add((s, t))

    def __call__(self, s: S, t: T) -> bool:
        return (s, t) in self.tuples


distributes_over = _ExtensibleBinaryRelation(
    {(Max, Min), (Min, Max), (Sum, Min), (Sum, Max), (Product, Sum)}
)


def _is_monoid_plus(op: Operation) -> bool:
    """True if ``op`` is the ``plus`` operation of some :class:`Monoid`."""
    owner = getattr(op, "__self__", None)
    return isinstance(owner, Monoid) and op is owner.plus


def _is_monoid_reduce(op: Operation) -> bool:
    """True if ``op`` is the ``reduce`` operation of some :class:`Monoid`."""
    owner = getattr(op, "__self__", None)
    return isinstance(owner, Monoid) and op is owner.reduce


class PlusEmpty(ObjectInterpretation):
    """plus() = 0"""

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if not args:
            return monoid.identity
        return fwd()


class PlusSingle(ObjectInterpretation):
    """plus(x) = x"""

    @implements(Monoid.plus)
    def plus(self, _, *args):
        if len(args) == 1:
            return args[0]
        return fwd()


class PlusIdentity(ObjectInterpretation):
    """x₁ + ... + 0 + ... + xₙ = x₁ + ... + xₙ"""

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if any(x is monoid.identity for x in args):
            return monoid.plus(*(x for x in args if x is not monoid.identity))
        return fwd()


class PlusAssoc(ObjectInterpretation):
    """x + (y + z) = (x + y) + z = x + y + z"""

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        def is_nested_plus(x):
            return isinstance(x, Term) and x.op is monoid.plus

        if any(is_nested_plus(x) for x in args):
            flat_args = itertools.chain.from_iterable(
                t.args if is_nested_plus(t) else (t,) for t in args
            )
            assert len(args) > 0
            return monoid.plus(*flat_args)
        return fwd()


class PlusDistr(ObjectInterpretation):
    """x + (y * z) = x * y + x * z"""

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if any(
            isinstance(x, Term)
            and _is_monoid_plus(x.op)
            and distributes_over(monoid, x.op.__self__)
            for x in args
        ):
            non_terms = []

            # group terms by their monoid
            by_monoid: dict[Monoid, list[Term]] = defaultdict(list)
            for t in args:
                if isinstance(t, Term) and _is_monoid_plus(t.op):
                    by_monoid[t.op.__self__].append(t)
                else:
                    non_terms.append(t)

            # distribute over each group
            progress = False
            final_sum = []
            for m, terms in by_monoid.items():
                if (
                    len(terms) > 1
                    and distributes_over(monoid, m)
                    and not distributes_over(m, monoid)
                ):
                    progress = True
                    term_args = (t.args for t in terms)
                    dist_terms = (
                        monoid.plus(*args) for args in itertools.product(*term_args)
                    )
                    final_sum.append(m.plus(*dist_terms))
                else:
                    final_sum += terms
            if progress:
                return monoid.plus(*non_terms, *final_sum)
        return fwd()


class PlusZero(ObjectInterpretation):
    """x₁ * ... * 0 * ... * xₙ = 0"""

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if not (isinstance(monoid, MonoidWithZero)):
            return fwd()
        if any(x is monoid.zero for x in args):
            return monoid.zero
        return fwd()


class PlusConsecutiveDups(ObjectInterpretation):
    """x ⊕ x ⊕ y = x ⊕ y"""

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if not is_idempotent(monoid):
            return fwd()

        dedup_args = (
            args[i]
            for i in range(len(args))
            if i == 0 or not syntactic_eq(args[i - 1], args[i])
        )
        return fwd(monoid, *dedup_args)


class PlusDups(ObjectInterpretation):
    """x ⊕ y ⊕ x = x ⊕ y"""

    @dataclass
    class _HashableTerm:
        term: Term

        def __eq__(self, other):
            return syntactic_eq(self, other)

        def __hash__(self):
            return syntactic_hash(self)

    @implements(Monoid.plus)
    def plus(self, monoid, *args):
        if not (is_idempotent(monoid) and is_commutative(monoid)):
            return fwd()

        # elim dups
        args_count = Counter(self._HashableTerm(t) for t in args)
        if len(args_count) < len(args):
            dedup_args = []
            for t in args:
                ht = self._HashableTerm(t)
                if ht in args_count:
                    dedup_args.append(t)
                    del args_count[ht]
            return fwd(monoid, *dedup_args)
        return fwd()


NormalizePlusIntp = functools.reduce(
    coproduct,
    typing.cast(
        list[Interpretation],
        [
            PlusEmpty(),
            PlusSingle(),
            PlusIdentity(),
            PlusAssoc(),
            PlusDistr(),
            PlusZero(),
            PlusConsecutiveDups(),
            PlusDups(),
        ],
    ),
)


class ReduceNoStreams(ObjectInterpretation):
    """Implements the identity
    reduce(R, ∅, body) = 0
    """

    @implements(Monoid.reduce)
    def reduce(self, monoid, _, streams):
        if len(streams) == 0:
            return monoid.identity
        return fwd()


class ReduceFusion(ObjectInterpretation):
    """Implements the identity
    reduce(R, S1, reduce(R, S2, body)) = reduce(R, S1 ∪ S2, body)
    """

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        if isinstance(body, Term) and body.op is monoid.reduce:
            return monoid.reduce(body.args[0], streams | body.args[1])
        return fwd()


class ReduceSplit(ObjectInterpretation):
    """Implements the identity
    reduce(R, S, b1 + ... + bn) = reduce(R, S, b1) + ... + reduce(R, S, bn)
    """

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        if not is_commutative(monoid):
            return fwd()
        if isinstance(body, Term) and body.op is monoid.plus:
            return monoid.plus(*(monoid.reduce(x, streams) for x in body.args))
        return fwd()


class ReduceFactorization(ObjectInterpretation):
    """
    Implements factorization of independent terms.
    For example, when having two independent distributions,
    we can rewrite their marginalization as:
        ∫p(x)⋅q(y)dxdy => ∫p(x)dx ⋅ ∫q(y)dy

    More specifically, in terms of reduces we are performing:
        reduce(R, (S₁ × ... × Sₖ) , A₁ * ... * Aₖ)
        => reduce(R, S₁, A₁) * ... * reduce(R, Sₖ, Aₖ)
        where free(Aᵢ) ∩ free(Aⱼ) ∩ S = ∅
          and free(Aᵢ) ∩ S ⊆ Sᵢ
    """

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        if not is_commutative(monoid):
            return fwd()
        if (
            isinstance(body, Term)
            and _is_monoid_plus(body.op)
            and distributes_over(body.op.__self__, monoid)
        ):
            inner_monoid: Monoid = body.op.__self__
            stream_vars = set(streams.keys())
            factors = [(arg, fvsof(arg)) for arg in body.args]
            stream_ids = {v: i for (i, v) in enumerate(stream_vars)}
            ds = DisjointSet(len(streams))

            # streams are in the same partition as their dependencies
            for stream_var, stream_id in stream_ids.items():
                stream_body = streams[stream_var]
                deps = sorted([stream_ids[v] for v in fvsof(stream_body) & stream_vars])
                ds.union(stream_id, *deps)

            # factors are in the same partition as their dependencies
            for _, factor_fvs in factors:
                factor_streams = sorted(
                    [stream_ids[v] for v in (factor_fvs & stream_vars)]
                )
                ds.union(*factor_streams)

            placed_streams = set()
            new_reduces = []
            for stream_key in streams:
                if stream_key in placed_streams:
                    continue

                partition = ds.find(stream_ids[stream_key])
                partition_streams = {
                    k: v
                    for (k, v) in streams.items()
                    if ds.find(stream_ids[k]) == partition
                }
                partition_stream_keys = set(partition_streams.keys())

                partition_factors = [
                    t for t in factors if (t[1] & partition_stream_keys)
                ]

                assert all(
                    (t[1] & stream_vars) <= partition_stream_keys
                    for t in partition_factors
                ), "partition contains all streams required by factor"

                partition_term = inner_monoid.plus(*(t[0] for t in partition_factors))
                new_reduces.append((partition_term, partition_streams))
                placed_streams |= partition_stream_keys

            constant_factors = [t for (t, fvs) in factors if not (fvs & stream_vars)]

            if len(new_reduces) > 1:
                result = inner_monoid.plus(
                    *constant_factors, *(monoid.reduce(*args) for args in new_reduces)
                )
                return result

        return fwd()


def inner_stream(
    streams: dict[Operation, Expr],
) -> Iterable[tuple[dict[Operation, Expr], Operation, Expr]]:
    """Returns the streams that can be ordered innermost in the loop nest as
    well as the remaining streams in the nest.

    """
    stream_vars = set(streams.keys())

    no_dependents = set()
    succ = defaultdict(set)
    for k, v in streams.items():
        preds = fvsof(v) & stream_vars
        if preds:
            for pred in preds:
                succ[pred].add(k)
        else:
            no_dependents.add(k)

    topo = TopologicalSorter(succ)
    topo.prepare()
    return (
        ({k: v for (k, v) in streams.items() if k != op}, op, streams[op])
        for op in set(topo.get_ready()) | no_dependents
    )


class ReduceDistributeCartesianProduct(ObjectInterpretation):
    """Eliminates a reduce over a cartesian product.
        ∑_x₁ ∑_x₂ ... ∑_xₙ ∏_i f(xᵢ) = ∏_i ∑_xᵢ f(xᵢ)
    This transform is also called inversion in the lifting
    literature (e.g. [1]).

    More specifically, this transform implements the identity
    reduce(⨁, reduce(⨂, body2, {vv: v()}), {v: reduce(×, body1, S1)} ∪ S2)
        = reduce(⨁, reduce(⨂, reduce(⨁, body2, {vv: body1}), S1), S2)
    where × is the cartesian product and ⨂ distributes over ⨁.

    Note: This could be generalized to grouped inversion [2].

    [1] Braz, Rd, Eyal Amir, and Dan Roth. "Lifted first-order
    probabilistic inference." IJCAI. 2005.
    [2] Taghipour, Nima, et al. "Completeness results for lifted
    variable elimination." AISTATS. 2013.
    """

    @implements(Monoid.reduce)
    def reduce(self, sum_monoid: Monoid, sum_body, sum_streams):
        if not (is_commutative(sum_monoid) and isinstance(sum_body, Term)):
            return fwd()

        # body is a product or multiplication of products
        if _is_monoid_plus(sum_body.op) and distributes_over(
            sum_body.op.__self__, sum_monoid
        ):
            prod_reduces = sum_body.args
        else:
            prod_reduces = [sum_body]

        products: list[tuple[Monoid, Callable, Operation, Term]] = []
        for prod_reduce in prod_reduces:
            if not (
                isinstance(prod_reduce, Term) and _is_monoid_reduce(prod_reduce.op)
            ):
                return fwd()
            prod_monoid: Monoid = prod_reduce.op.__self__
            prod_body = prod_reduce.args[0]
            prod_streams = typing.cast(Mapping, prod_reduce.args[1])
            if not (
                distributes_over(prod_monoid, sum_monoid)
                and (len(products) == 0 or products[-1][0] == prod_monoid)
            ):
                return fwd()

            if len(prod_streams) > 1 or len(prod_streams) == 0:
                return fwd()
            (prod_op, prod_stream) = next(iter(prod_streams.items()))
            products.append(
                (prod_monoid, deffn(prod_body, prod_op), prod_op, prod_stream)
            )

        assert len(products) > 0

        for outer_sum_streams, cprod_op, cprod_term in inner_stream(sum_streams):
            if not (
                isinstance(cprod_term, Term)
                and cprod_term.op is CartesianProduct.reduce
            ):
                continue
            (cprod_body, cprod_streams) = cprod_term.args

            if not all(
                prod_stream.op == cprod_op for (_, _, _, prod_stream) in products
            ):
                continue

            prod_op = Operation.define(products[0][2])
            prod_monoid = products[0][0]
            inner_sum = sum_monoid.reduce(
                prod_monoid.plus(
                    *(prod_body(prod_op()) for (_, prod_body, _, _) in products)
                ),
                {prod_op: cprod_body},
            )
            prod = prod_monoid.reduce(inner_sum, cprod_streams)
            outer_sum = (
                sum_monoid.reduce(prod, outer_sum_streams)
                if outer_sum_streams
                else prod
            )
            return outer_sum

        return fwd()


NormalizeReduceIntp = functools.reduce(
    coproduct,
    typing.cast(
        list[Interpretation],
        [
            ReduceNoStreams(),
            ReduceFusion(),
            ReduceSplit(),
            ReduceFactorization(),
            ReduceDistributeCartesianProduct(),
        ],
    ),
)

NormalizeIntp = coproduct(NormalizePlusIntp, NormalizeReduceIntp)
