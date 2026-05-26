"""NumPyro distribution support for weighted streams.

``weighted(dist)`` is the smart constructor for treating a numpyro
distribution as a weighted stream. By default it stays symbolic — i.e.
``weighted(d)`` returns a ``Term`` whose ``args[0]`` is ``d`` — so that
specialized reduction rules (closed-form expectations, quadrature, etc.)
can pattern-match on the distribution.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints

import effectful.handlers.jax.numpy as ejnp
from effectful.handlers.jax import jax_getitem
from effectful.handlers.jax.monoid import LogSumExp
from effectful.handlers.numpyro import (
    CategoricalLogitsTerm,
    CategoricalProbsTerm,
    NormalTerm,
)
from effectful.ops.monoid import (
    Monoid,
    NormalizeIntp,
    Product,
    Sum,
    WeightedStream,
    to_stream,
    weighted,
)
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, deffn, implements
from effectful.ops.types import NotHandled, Operation, Term


@weighted.register(dist.Distribution)
def _weighted_dist(_d):
    raise NotHandled


@to_stream.register(dist.Distribution)
def _stream_dist(_d):
    raise NotHandled


@to_stream.register(constraints.Constraint)
def _stream_constraint(_c):
    raise NotHandled


def _weighted_dist_arg(v) -> dist.Distribution | None:
    """If ``v`` is ``Term(weighted, [d])`` with ``d`` a numpyro Distribution,
    return ``d``; otherwise ``None``.
    """
    if not (isinstance(v, Term) and v.op is weighted):
        return None
    (d,) = v.args
    return d if isinstance(d, dist.Distribution) else None


@dataclass
class NumpyroSampling(ObjectInterpretation):
    """Replace ``weighted(d)`` with a sample-backed :class:`WeightedStream`.

    Draws ``n_samples`` i.i.d. samples from ``d`` and attaches a uniform weight
    ``1/n_samples`` (linear space) or ``-log(n_samples)`` (log space, when the
    outer monoid is :data:`LogSumExp`). The resulting :class:`WeightedStream` is
    then handled by the standard :class:`ReduceWeightedStream` rewrite.

    """

    rng_key: jax.Array
    n_samples: int = 1000

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        new_streams = dict(streams)
        progress = False
        for k, v in streams.items():
            d = _weighted_dist_arg(v)
            if d is None:
                continue
            samples = d.sample(self.rng_key, sample_shape=(self.n_samples,))
            if monoid is LogSumExp:
                w_val = -jnp.log(self.n_samples)
                w_monoid: Monoid = Sum
            else:
                w_val = 1.0 / self.n_samples
                w_monoid = Product
            new_streams[k] = WeightedStream(
                stream=samples,
                weight=deffn(w_val, Operation.define(k)),
                monoid=w_monoid,
            )
            progress = True
        if progress:
            return monoid.reduce(body, new_streams)
        return fwd()


@dataclass
class NumpyroGaussHermite(ObjectInterpretation):
    """Gauss–Hermite quadrature for ``weighted(Normal(μ, σ))``.

    For ``X ∼ Normal(μ, σ²)``, the change of variable ``u = (x-μ)/(σ√2)`` gives
    ::

        E[f(X)] = (1/√π) ∫ f(μ + σ√2 · u) e^{-u²} du
               ≈ Σᵢ (wᵢ/√π) · f(μ + σ√2 · uᵢ)

    where ``{uᵢ, wᵢ}`` are the physicists' Hermite nodes/weights from
    :func:`numpy.polynomial.hermite.hermgauss`. The rule replaces
    ``weighted(d)`` with a :class:`WeightedStream` of length ``n_nodes`` and
    lets the standard :class:`ReduceWeightedStream` machinery finish.

    Weight monoid is :data:`Product` for linear-space bodies (e.g.
    ``Sum.reduce``) and :data:`Sum` for log-space bodies (e.g.
    ``LogSumExp.reduce``); both pairs distribute correctly.

    """

    n_nodes: int = 20

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        new_streams = dict(streams)
        progress = False
        for k, v in streams.items():
            d = _weighted_dist_arg(v)
            if not isinstance(d, dist.Normal | NormalTerm):
                continue
            new_streams[k] = self._gauss_hermite(d, monoid)
            progress = True
        if progress:
            return monoid.reduce(body, new_streams)
        return fwd()

    def _gauss_hermite(self, d, monoid: Monoid) -> WeightedStream:
        u, w = np.polynomial.hermite.hermgauss(self.n_nodes)
        u_jax = jnp.asarray(u, dtype=jnp.float32)
        w_jax = jnp.asarray(w, dtype=jnp.float32)

        nodes = d.loc + jnp.sqrt(2.0) * d.scale * u_jax
        if monoid is LogSumExp:
            weights = jnp.log(w_jax) - 0.5 * jnp.log(jnp.pi)
            w_monoid: Monoid = Sum
        else:
            weights = w_jax / jnp.sqrt(jnp.pi)
            w_monoid = Product

        # Position-match the node value back to its weight via argmin. The
        # weight function is invoked symbolically by ``ReduceWeightedStream``
        # (with a Term arg), so we use the effectful-wrapped jnp so the
        # lookup becomes a Term that evaluates to the right scalar once the
        # default reduce binds the stream variable to a concrete node.
        def weight_fn(x, _nodes=nodes, _w=weights):
            idx = ejnp.argmin(ejnp.abs(_nodes - x))
            return jax_getitem(_w, (idx,))

        return WeightedStream(stream=nodes, weight=weight_fn, monoid=w_monoid)


@dataclass
class NumpyroCategorical(ObjectInterpretation):
    """Exact enumeration ('quadrature') for ``weighted(Categorical(...))``.

    A categorical with ``K`` outcomes has finite integer support
    ``{0, …, K-1}``; integration reduces to an exact finite sum. The rule
    replaces ``weighted(d)`` with a :class:`WeightedStream` whose stream is
    ``jnp.arange(K)`` and whose weight indexes into the per-outcome
    probability vector.

    Weight monoid is :data:`Product` for linear-space bodies and :data:`Sum`
    for log-space bodies (under :data:`LogSumExp`), matching the
    distributes-over pairs used by :class:`ReduceWeightedStream`.

    """

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        new_streams = dict(streams)
        progress = False
        for k, v in streams.items():
            d = _weighted_dist_arg(v)
            ws = self._categorical(d, monoid)
            if ws is None:
                continue
            new_streams[k] = ws
            progress = True
        if progress:
            return monoid.reduce(body, new_streams)
        return fwd()

    def _categorical(self, d, monoid: Monoid) -> WeightedStream | None:
        # Pick the natural representation for the target weight monoid so we
        # don't go probs→log or logits→probs→log unnecessarily.
        if monoid is LogSumExp:
            w_monoid: Monoid = Sum
            if isinstance(d, dist.CategoricalLogits | CategoricalLogitsTerm):
                weights = jax.nn.log_softmax(jnp.asarray(d.logits), axis=-1)
            elif isinstance(d, dist.CategoricalProbs | CategoricalProbsTerm):
                weights = jnp.log(jnp.asarray(d.probs))
            else:
                return None
        else:
            w_monoid = Product
            if isinstance(d, dist.CategoricalProbs | CategoricalProbsTerm):
                weights = jnp.asarray(d.probs)
            elif isinstance(d, dist.CategoricalLogits | CategoricalLogitsTerm):
                weights = jax.nn.softmax(jnp.asarray(d.logits), axis=-1)
            else:
                return None

        indices = jnp.arange(weights.shape[-1])

        # The support value *is* the index, so the lookup is direct.
        def weight_fn(x, _w=weights):
            return jax_getitem(_w, (x,))

        return weighted(stream=indices, weight=weight_fn, monoid=w_monoid)


class NumpyroLogProb(ObjectInterpretation):
    """Lower ``weighted(d)`` to its symbolic log-prob form.

    Generic fallback: produces a :class:`WeightedStream` whose stream is the
    symbolic ``stream(d.support)``, weight is ``d.log_prob``, and weight monoid
    is :data:`Sum` (log-space multiplication). With ``distributes_over(Sum,
    LogSumExp)`` registered, a surrounding ``LogSumExp.reduce`` will then
    desugar via :class:`ReduceWeightedStream` into the standard expectation
    integrand.

    """

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        new_streams = dict(streams)
        progress = False
        for k, v in streams.items():
            d = _weighted_dist_arg(v)
            if d is None:
                continue
            new_streams[k] = weighted(
                stream=to_stream(d.support), weight=d.log_prob, monoid=Sum
            )
            progress = True
        if progress:
            return monoid.reduce(body, new_streams)
        return fwd()


NormalizeIntp.extend(NumpyroCategorical())
