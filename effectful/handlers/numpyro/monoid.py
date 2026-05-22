"""NumPyro distribution support for weighted streams.

``weighted(dist)`` is the smart constructor for treating a numpyro
distribution as a weighted stream. By default it stays symbolic — i.e.
``weighted(d)`` returns a ``Term`` whose ``args[0]`` is ``d`` — so that
specialized reduction rules (closed-form expectations, quadrature, etc.)
can pattern-match on the distribution.

Two general-purpose reduction rules are provided here:

* :class:`NumpyroSampling` — Monte Carlo approximation. Replaces
  ``weighted(d)`` with a sample-backed :class:`WeightedStream` of ``n_samples``
  i.i.d. draws and uniform weights, then delegates to the standard
  :class:`ReduceWeightedStream` machinery.

* :class:`NumpyroLogProb` — generic symbolic lowering. Replaces
  ``weighted(d)`` with ``WeightedStream(stream(d.support), d.log_prob, Sum)``.
  ``Sum`` acts as multiplication in log space and
  ``distributes_over(Sum, LogSumExp)`` is registered, so a subsequent
  ``LogSumExp.reduce`` is desugared by ``ReduceWeightedStream`` into the
  standard log-space expectation integrand:

      LogSumExp.reduce(Sum.plus(d.log_prob(x), body), {x: stream(d.support)})
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import numpyro.distributions.constraints as constraints

from effectful.handlers.jax.monoid import LogSumExp
from effectful.ops.monoid import (
    Monoid,
    Product,
    Sum,
    WeightedStream,
    stream,
    weighted,
)
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, deffn, implements
from effectful.ops.types import NotHandled, Operation, Term

# --- smart constructors stay symbolic for distributions / constraints -----


@weighted.register(dist.Distribution)
def _weighted_dist(_d):
    raise NotHandled


@stream.register(dist.Distribution)
def _stream_dist(_d):
    raise NotHandled


@stream.register(constraints.Constraint)
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


# --- rule: Monte Carlo sampling -------------------------------------------


@dataclass
class NumpyroSampling(ObjectInterpretation):
    """Replace ``weighted(d)`` with a sample-backed :class:`WeightedStream`.

    Draws ``n_samples`` i.i.d. samples from ``d`` and attaches a uniform
    weight ``1/n_samples`` (linear space) or ``-log(n_samples)`` (log
    space, when the outer monoid is :data:`LogSumExp`). The resulting
    :class:`WeightedStream` is then handled by the standard
    :class:`ReduceWeightedStream` rewrite.
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


# --- rule: symbolic log-prob lowering -------------------------------------


class NumpyroLogProb(ObjectInterpretation):
    """Lower ``weighted(d)`` to its symbolic log-prob form.

    Generic fallback: produces a :class:`WeightedStream` whose stream is
    the symbolic ``stream(d.support)``, weight is ``d.log_prob``, and
    weight monoid is :data:`Sum` (log-space multiplication). With
    ``distributes_over(Sum, LogSumExp)`` registered, a surrounding
    ``LogSumExp.reduce`` will then desugar via :class:`ReduceWeightedStream`
    into the standard expectation integrand.
    """

    @implements(Monoid.reduce)
    def reduce(self, monoid, body, streams):
        new_streams = dict(streams)
        progress = False
        for k, v in streams.items():
            d = _weighted_dist_arg(v)
            if d is None:
                continue
            new_streams[k] = WeightedStream(
                stream=stream(d.support), weight=d.log_prob, monoid=Sum
            )
            progress = True
        if progress:
            return monoid.reduce(body, new_streams)
        return fwd()
