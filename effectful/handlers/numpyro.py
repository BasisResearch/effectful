try:
    import numpyro.distributions as dist
except ImportError:
    raise ImportError("Numpyro is required to use effectful.handlers.numpyro")


import functools
from collections.abc import Collection, Mapping
from typing import Any

import jax

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import bind_dims, jax_getitem, sizesof, unbind_dims
from effectful.handlers.jax._handlers import _register_jax_op, is_eager_array
from effectful.ops.semantics import typeof
from effectful.ops.syntax import defdata, defop, defterm
from effectful.ops.types import NotHandled, Operation, Term


class Naming(dict[Operation[[], jax.Array], int]):
    """
    A mapping from dimensions (indexed from the right) to names.
    """

    def __init__(self, name_to_dim: Mapping[Operation[[], jax.Array], int]):
        assert all(v < 0 for v in name_to_dim.values())
        super().__init__(name_to_dim)

    @staticmethod
    def from_shape(
        names: Collection[Operation[[], jax.Array]], event_dims: int
    ) -> "Naming":
        """Create a naming from a set of indices and the number of event dimensions.

        The resulting naming converts tensors of shape
        ``| batch_shape | named | event_shape |``
        to tensors of shape ``| batch_shape | event_shape |, | named |``.

        """
        assert event_dims >= 0
        return Naming({n: -event_dims - len(names) + i for i, n in enumerate(names)})

    def apply(self, value: jax.Array) -> jax.Array:
        indexes: list[Any] = [slice(None)] * (len(value.shape))
        for n, d in self.items():
            indexes[len(value.shape) + d] = n()
        return jax_getitem(value, tuple(indexes))

    def __repr__(self):
        return f"Naming({super().__repr__()})"


@unbind_dims.register  # type: ignore
def _unbind_distribution(
    d: dist.Distribution, *names: Operation[[], jax.Array]
) -> dist.Distribution:
    batch_shape = None

    def _validate_batch_shape(t):
        nonlocal batch_shape
        if len(t.shape) < len(names):
            raise ValueError(
                "All tensors must have at least as many dimensions as names"
            )

        if batch_shape is None:
            batch_shape = t.shape[: len(names)]

        if (
            len(t.shape) < len(batch_shape)
            or t.shape[: len(batch_shape)] != batch_shape
        ):
            raise ValueError("All tensors must have the same batch shape.")

    def _to_named(a):
        nonlocal batch_shape
        # FIXME: Some distributions take scalar arguments that are never
        # batched. Ignore these. We should be able to raise an error in some
        # cases that we see a scalar tensor, and a smarter version of this code
        # would do so.
        if isinstance(a, jax.Array) and a.shape != ():
            _validate_batch_shape(a)
            return unbind_dims(a, *names)
        elif isinstance(a, dist.Distribution):
            return unbind_dims(a, *names)
        else:
            return a

    d = defterm(d)

    # FIXME: This assumes that the only operations that return distributions are
    # distribution constructors.
    if not (isinstance(d, Term) and issubclass(typeof(d), dist.Distribution)):
        raise NotImplementedError

    new_d = d.op(
        *[_to_named(a) for a in d.args],
        **{k: _to_named(v) for (k, v) in d.kwargs.items()},
    )
    return new_d


@bind_dims.register  # type: ignore
def _bind_dims_distribution(
    d: dist.Distribution, *names: Operation[[], jax.Array]
) -> dist.Distribution:
    def _to_positional(a, indices):
        typ = typeof(a)
        if issubclass(typ, jax.Array):
            # broadcast to full indexed shape
            existing_dims = set(sizesof(a).keys())
            missing_dims = set(indices) - existing_dims

            a_indexed = unbind_dims(
                jnp.broadcast_to(
                    a, tuple(indices[dim] for dim in missing_dims) + a.shape
                ),
                *missing_dims,
            )
            return bind_dims(a_indexed, *indices)
        elif issubclass(typ, dist.Distribution):
            # We assume that only one distriution appears in our arguments. This
            # is sufficient for cases like Independent and
            # TransformedDistribution
            return bind_dims(a, *indices)
        else:
            return a

    d = defterm(d)

    # FIXME: This assumes that the only operations that return distributions are
    # distribution constructors.
    if not (isinstance(d, Term) and issubclass(typeof(d), dist.Distribution)):
        raise NotImplementedError

    sizes = sizesof(d)
    indices = {k: sizes[k] for k in names}

    pos_args = [_to_positional(a, indices) for a in d.args]
    pos_kwargs = {k: _to_positional(v, indices) for (k, v) in d.kwargs.items()}
    new_d = d.op(*pos_args, **pos_kwargs)

    return new_d


def _broadcast_to_named(t, sizes):
    missing_dims = set(sizes) - set(sizesof(t))
    t_broadcast = jnp.broadcast_to(
        t, tuple(sizes[dim] for dim in missing_dims) + t.shape
    )
    return jax_getitem(t_broadcast, tuple(dim() for dim in missing_dims))


def expand_to_batch_shape(tensor, batch_ndims, expanded_batch_shape):
    """
    Expands a tensor of shape batch_shape + remaining_shape to
    expanded_batch_shape + remaining_shape.

    Args:
        tensor: JAX array with shape batch_shape + event_shape
        expanded_batch_shape: tuple of the desired expanded batch dimensions
        event_ndims: number of dimensions in the event_shape

    Returns:
        A JAX array with shape expanded_batch_shape + event_shape
    """
    # Split the shape into batch and event parts
    assert len(tensor.shape) >= batch_ndims

    batch_shape = tensor.shape[:batch_ndims] if batch_ndims > 0 else ()
    remaining_shape = tensor.shape[batch_ndims:]

    # Ensure the expanded batch shape is compatible with the current batch shape
    if len(expanded_batch_shape) < batch_ndims:
        raise ValueError(
            "Expanded batch shape must have at least as many dimensions as current batch shape"
        )
    new_batch_shape = jnp.broadcast_shapes(batch_shape, expanded_batch_shape)

    # Create the new shape
    new_shape = new_batch_shape + remaining_shape

    # Broadcast the tensor to the new shape
    expanded_tensor = jnp.broadcast_to(tensor, new_shape)

    return expanded_tensor


@Term.register
class _DistributionTerm(dist.Distribution):
    """A distribution wrapper that satisfies the Term interface.

    Represented as a term of the form D(*args, **kwargs) where D is the
    distribution constructor.

    Note: When we construct instances of this class, we put distribution
    parameters that can be expanded in the args list and those that cannot in
    the kwargs list.

    """

    _constr: type[dist.Distribution]
    _op: Operation[..., dist.Distribution]
    _args: tuple
    _kwargs: dict
    __pos_base_dist: dist.Distribution | None = None

    def __init__(self, constr, op, *args, **kwargs):
        assert issubclass(constr, dist.Distribution)

        self._constr = constr
        self._op = op
        self._args = args
        self._kwargs = kwargs

    @functools.cached_property
    def _indices(self) -> Mapping[Operation[[], jax.Array], int]:
        return sizesof(self)

    @functools.cached_property
    def _pos_base_dist(self) -> dist.Distribution:
        bound = bind_dims(self, *self._indices)
        return self._constr(*bound.args, **bound.kwargs)

    @functools.cached_property
    def _is_eager(self) -> bool:
        return all(
            (not isinstance(x, Term) or is_eager_array(x))
            for x in (*self.args, *self.kwargs.values())
        )

    @property
    def op(self):
        return self._op

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

    @defop  # type: ignore
    @property
    def batch_shape(self):
        if not (self._is_eager):
            raise NotHandled
        return self._pos_base_dist.batch_shape[len(self._indices) :]

    @defop  # type: ignore
    @property
    def has_rsample(self) -> bool:
        if not (self._is_eager):
            raise NotHandled
        return self._pos_base_dist.has_rsample

    @defop  # type: ignore
    @property
    def event_shape(self):
        if not (self._is_eager):
            raise NotHandled
        return self._pos_base_dist.event_shape

    def _reindex_sample(self, value, sample_shape):
        index = (slice(None),) * len(sample_shape) + tuple(i() for i in self._indices)
        ret = jax_getitem(value, index)
        return ret

    @defop
    def rsample(self, key, sample_shape=()):
        if not (self._is_eager and is_eager_array(key)):
            raise NotHandled

        return self._reindex_sample(
            self._pos_base_dist.rsample(key, sample_shape), sample_shape
        )

    @defop
    def sample(self, key, sample_shape=()):
        if not (self._is_eager and is_eager_array(key)):
            raise NotHandled

        return self._reindex_sample(
            self._pos_base_dist.sample(key, sample_shape), sample_shape
        )

    @defop
    def log_prob(self, value):
        if not (self._is_eager and is_eager_array(value)):
            raise NotHandled

        # value has shape named_batch_shape + sample_shape + batch_shape + event_shape
        n_batch_event = len(self.batch_shape) + len(self.event_shape)
        sample_shape = (
            value.shape if n_batch_event == 0 else value.shape[:-n_batch_event]
        )
        value = bind_dims(_broadcast_to_named(value, self._indices), *self._indices)
        dims = list(range(len(value.shape)))
        n_named_batch = len(self._indices)
        perm = (
            dims[n_named_batch : n_named_batch + len(sample_shape)]
            + dims[:n_named_batch]
            + dims[n_named_batch + len(sample_shape) :]
        )
        assert len(perm) == len(value.shape)

        # perm_value has shape sample_shape + named_batch_shape + batch_shape + event_shape
        perm_value = jnp.permute_dims(value, perm)
        pos_log_prob = _register_jax_op(self._pos_base_dist.log_prob)(perm_value)
        ind_log_prob = self._reindex_sample(pos_log_prob, sample_shape)
        return ind_log_prob

    @defop  # type: ignore
    @property
    def mean(self):
        if not self._is_eager:
            raise NotHandled
        try:
            return self._reindex_sample(self._pos_base_dist.mean, ())
        except NotImplementedError:
            raise RuntimeError(f"mean is not implemented for {type(self).__name__}")

    @defop  # type: ignore
    @property
    def variance(self):
        if not self._is_eager:
            raise NotHandled
        try:
            return self._reindex_sample(self._pos_base_dist.variance, ())
        except NotImplementedError:
            raise RuntimeError(f"variance is not implemented for {type(self).__name__}")

    @defop
    def enumerate_support(self, expand=True):
        if not self._is_eager:
            raise NotHandled
        return self._reindex_sample(self._pos_base_dist.enumerate_support(expand), ())

    @defop
    def entropy(self):
        if not self._is_eager:
            raise NotHandled
        return self._pos_base_dist.entropy()

    @defop
    def to_event(self, reinterpreted_batch_ndims=None):
        raise NotHandled

    @defop
    def expand(self, batch_shape):
        if not self._is_eager:
            raise NotHandled

        def expand_arg(a, batch_shape):
            if is_eager_array(a):
                return expand_to_batch_shape(a, len(self.batch_shape), batch_shape)
            return a

        if self.batch_shape == batch_shape:
            return self

        expanded_args = [expand_arg(a, batch_shape) for a in self.args]
        expanded_kwargs = {
            k: expand_arg(v, batch_shape) for (k, v) in self.kwargs.items()
        }
        return self.op(*expanded_args, **expanded_kwargs)

    def __repr__(self):
        return Term.__repr__(self)

    def __str__(self):
        return Term.__str__(self)


batch_shape = _DistributionTerm.batch_shape
event_shape = _DistributionTerm.event_shape
has_rsample = _DistributionTerm.has_rsample
rsample = _DistributionTerm.rsample
sample = _DistributionTerm.sample
log_prob = _DistributionTerm.log_prob
mean = _DistributionTerm.mean
variance = _DistributionTerm.variance
enumerate_support = _DistributionTerm.enumerate_support
entropy = _DistributionTerm.entropy
to_event = _DistributionTerm.to_event
expand = _DistributionTerm.expand


@defop
def Cauchy(*args, **kwargs) -> dist.Cauchy:
    raise NotHandled


@defdata.register(dist.Cauchy)
class CauchyTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.Cauchy, *args, **kwargs)


@defterm.register(dist.Cauchy)
def _embed_cauchy(d: dist.Cauchy) -> Term[dist.Cauchy]:
    return Cauchy(d.loc, d.scale)


@defop
def Gumbel(*args, **kwargs) -> dist.Gumbel:
    raise NotHandled


@defdata.register(dist.Gumbel)
class GumbelTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.Gumbel, *args, **kwargs)


@defterm.register(dist.Gumbel)
def _embed_gumbel(d: dist.Gumbel) -> Term[dist.Gumbel]:
    return Gumbel(d.loc, d.scale)


@defop
def Laplace(*args, **kwargs) -> dist.Laplace:
    raise NotHandled


@defdata.register(dist.Laplace)
class LaplaceTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.Laplace, *args, **kwargs)


@defterm.register(dist.Laplace)
def _embed_laplace(d: dist.Laplace) -> Term[dist.Laplace]:
    return Laplace(d.loc, d.scale)


@defop
def LogNormal(*args, **kwargs) -> dist.LogNormal:
    raise NotHandled


@defdata.register(dist.LogNormal)
class LogNormalTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.LogNormal, *args, **kwargs)


@defterm.register(dist.LogNormal)
def _embed_lognormal(d: dist.LogNormal) -> Term[dist.LogNormal]:
    return LogNormal(d.loc, d.scale)


@defop
def Logistic(*args, **kwargs) -> dist.Logistic:
    raise NotHandled


@defdata.register(dist.Logistic)
class LogisticTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.Logistic, *args, **kwargs)


@defterm.register(dist.Logistic)
def _embed_logistic(d: dist.Logistic) -> Term[dist.Logistic]:
    return Logistic(d.loc, d.scale)


@defop
def Normal(*args, **kwargs) -> dist.Normal:
    raise NotHandled


@defdata.register(dist.Normal)
class NormalTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.Normal, *args, **kwargs)


@defterm.register(dist.Normal)
def _embed_normal(d: dist.Normal) -> Term[dist.Normal]:
    return Normal(d.loc, d.scale)


@defop
def StudentT(*args, **kwargs) -> dist.StudentT:
    raise NotHandled


@defdata.register(dist.StudentT)
class StudentTTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.StudentT, *args, **kwargs)


@defterm.register(dist.StudentT)
def _embed_studentt(d: dist.StudentT) -> Term[dist.StudentT]:
    return StudentT(d.loc, d.scale)


@defop
def BernoulliProbs(*args, **kwargs) -> dist.BernoulliProbs:
    raise NotHandled


@defdata.register(dist.BernoulliProbs)
class BernoulliProbsTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.BernoulliProbs, *args, **kwargs)


@defterm.register(dist.BernoulliProbs)
def _embed_bernoulliprobs(d: dist.BernoulliProbs) -> Term[dist.BernoulliProbs]:
    return BernoulliProbs(d.probs)


@defop
def CategoricalProbs(*args, **kwargs) -> dist.CategoricalProbs:
    raise NotHandled


@defdata.register(dist.CategoricalProbs)
class CategoricalProbsTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.CategoricalProbs, *args, **kwargs)


@defterm.register(dist.CategoricalProbs)
def _embed_categoricalprobs(d: dist.CategoricalProbs) -> Term[dist.CategoricalProbs]:
    return CategoricalProbs(d.probs)


@defop
def GeometricProbs(*args, **kwargs) -> dist.GeometricProbs:
    raise NotHandled


@defdata.register(dist.GeometricProbs)
class GeometricProbsTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.GeometricProbs, *args, **kwargs)


@defterm.register(dist.GeometricProbs)
def _embed_geometricprobs(d: dist.GeometricProbs) -> Term[dist.GeometricProbs]:
    return GeometricProbs(d.probs)


@defop
def BernoulliLogits(*args, **kwargs) -> dist.BernoulliLogits:
    raise NotHandled


@defdata.register(dist.BernoulliLogits)
class BernoulliLogitsTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.BernoulliLogits, *args, **kwargs)


@defterm.register(dist.BernoulliLogits)
def _embed_bernoullilogits(d: dist.BernoulliLogits) -> Term[dist.BernoulliLogits]:
    return BernoulliLogits(d.logits)


@defop
def CategoricalLogits(*args, **kwargs) -> dist.CategoricalLogits:
    raise NotHandled


@defdata.register(dist.CategoricalLogits)
class CategoricalLogitsTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.CategoricalLogits, *args, **kwargs)


@defterm.register(dist.CategoricalLogits)
def _embed_categoricallogits(d: dist.CategoricalLogits) -> Term[dist.CategoricalLogits]:
    return CategoricalLogits(d.logits)


@defop
def GeometricLogits(*args, **kwargs) -> dist.GeometricLogits:
    raise NotHandled


@defdata.register(dist.GeometricLogits)
class GeometricLogitsTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.GeometricLogits, *args, **kwargs)


@defterm.register(dist.GeometricLogits)
def _embed_geometriclogits(d: dist.GeometricLogits) -> Term[dist.GeometricLogits]:
    return GeometricLogits(d.logits)


@defop
def Beta(*args, **kwargs) -> dist.Beta:
    raise NotHandled


@defdata.register(dist.Beta)
class BetaTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.Beta, *args, **kwargs)


@defterm.register(dist.Beta)
def _embed_beta(d: dist.Beta) -> Term[dist.Beta]:
    return Beta(d.concentration1, d.concentration0)


@defop
def Kumaraswamy(*args, **kwargs) -> dist.Kumaraswamy:
    raise NotHandled


@defdata.register(dist.Kumaraswamy)
class KumaraswamyTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.Kumaraswamy, *args, **kwargs)


@defterm.register(dist.Kumaraswamy)
def _embed_kumaraswamy(d: dist.Kumaraswamy) -> Term[dist.Kumaraswamy]:
    return Kumaraswamy(d.concentration1, d.concentration0)


@defop
def BinomialProbs(*args, **kwargs) -> dist.BinomialProbs:
    raise NotHandled


@defdata.register(dist.BinomialProbs)
class BinomialProbsTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.BinomialProbs, *args, **kwargs)


@defterm.register(dist.BinomialProbs)
def _embed_binomialprobs(d: dist.BinomialProbs) -> Term[dist.BinomialProbs]:
    return BinomialProbs(d.probs, d.total_count)


@defop
def NegativeBinomialProbs(*args, **kwargs) -> dist.NegativeBinomialProbs:
    raise NotHandled


@defdata.register(dist.NegativeBinomialProbs)
class NegativeBinomialProbsTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.NegativeBinomialProbs, *args, **kwargs)


@defterm.register(dist.NegativeBinomialProbs)
def _embed_negativebinomialprobs(
    d: dist.NegativeBinomialProbs,
) -> Term[dist.NegativeBinomialProbs]:
    return NegativeBinomialProbs(d.probs, d.total_count)


@defop
def MultinomialProbs(*args, **kwargs) -> dist.MultinomialProbs:
    raise NotHandled


@defdata.register(dist.MultinomialProbs)
class MultinomialProbsTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.MultinomialProbs, *args, **kwargs)


@defterm.register(dist.MultinomialProbs)
def _embed_multinomialprobs(d: dist.MultinomialProbs) -> Term[dist.MultinomialProbs]:
    return MultinomialProbs(d.probs, d.total_count)


@defop
def BinomialLogits(*args, **kwargs) -> dist.BinomialLogits:
    raise NotHandled


@defdata.register(dist.BinomialLogits)
class BinomialLogitsTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.BinomialLogits, *args, **kwargs)


@defterm.register(dist.BinomialLogits)
def _embed_binomiallogits(d: dist.BinomialLogits) -> Term[dist.BinomialLogits]:
    return BinomialLogits(d.logits, d.total_count)


@defop
def NegativeBinomialLogits(*args, **kwargs) -> dist.NegativeBinomialLogits:
    raise NotHandled


@defdata.register(dist.NegativeBinomialLogits)
class NegativeBinomialLogitsTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.NegativeBinomialLogits, *args, **kwargs)


@defterm.register(dist.NegativeBinomialLogits)
def _embed_negativebinomiallogits(
    d: dist.NegativeBinomialLogits,
) -> Term[dist.NegativeBinomialLogits]:
    return NegativeBinomialLogits(d.logits, d.total_count)


@defop
def MultinomialLogits(*args, **kwargs) -> dist.MultinomialLogits:
    raise NotHandled


@defdata.register(dist.MultinomialLogits)
class MultinomialLogitsTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.MultinomialLogits, *args, **kwargs)


@defterm.register(dist.MultinomialLogits)
def _embed_multinomiallogits(d: dist.MultinomialLogits) -> Term[dist.MultinomialLogits]:
    return MultinomialLogits(d.logits, d.total_count)


@defop
def Chi2(*args, **kwargs) -> dist.Chi2:
    raise NotHandled


@defdata.register(dist.Chi2)
class Chi2Term(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.Chi2, *args, **kwargs)


@defterm.register(dist.Chi2)
def _embed_chi2(d: dist.Chi2) -> Term[dist.Chi2]:
    return Chi2(d.df)


@defop
def Dirichlet(*args, **kwargs) -> dist.Dirichlet:
    raise NotHandled


@defdata.register(dist.Dirichlet)
class DirichletTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.Dirichlet, *args, **kwargs)


@defterm.register(dist.Dirichlet)
def _embed_dirichlet(d: dist.Dirichlet) -> Term[dist.Dirichlet]:
    return Dirichlet(d.concentration)


@defop
def DirichletMultinomial(*args, **kwargs) -> dist.DirichletMultinomial:
    raise NotHandled


@defdata.register(dist.DirichletMultinomial)
class DirichletMultinomialTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.DirichletMultinomial, *args, **kwargs)


@defterm.register(dist.DirichletMultinomial)
def _embed_dirichletmultinomial(
    d: dist.DirichletMultinomial,
) -> Term[dist.DirichletMultinomial]:
    return DirichletMultinomial(d.concentration, d.total_count)


@defop
def Exponential(*args, **kwargs) -> dist.Exponential:
    raise NotHandled


@defdata.register(dist.Exponential)
class ExponentialTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.Exponential, *args, **kwargs)


@defterm.register(dist.Exponential)
def _embed_exponential(d: dist.Exponential) -> Term[dist.Exponential]:
    return Exponential(d.rate)


@defop
def Poisson(*args, **kwargs) -> dist.Poisson:
    raise NotHandled


@defdata.register(dist.Poisson)
class PoissonTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.Poisson, *args, **kwargs)


@defterm.register(dist.Poisson)
def _embed_poisson(d: dist.Poisson) -> Term[dist.Poisson]:
    return Poisson(d.rate)


@defop
def Gamma(*args, **kwargs) -> dist.Gamma:
    raise NotHandled


@defdata.register(dist.Gamma)
class GammaTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.Gamma, *args, **kwargs)


@defterm.register(dist.Gamma)
def _embed_gamma(d: dist.Gamma) -> Term[dist.Gamma]:
    return Gamma(d.concentration, d.rate)


@defop
def HalfCauchy(*args, **kwargs) -> dist.HalfCauchy:
    raise NotHandled


@defdata.register(dist.HalfCauchy)
class HalfCauchyTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.HalfCauchy, *args, **kwargs)


@defterm.register(dist.HalfCauchy)
def _embed_halfcauchy(d: dist.HalfCauchy) -> Term[dist.HalfCauchy]:
    return HalfCauchy(d.scale)


@defop
def HalfNormal(*args, **kwargs) -> dist.HalfNormal:
    raise NotHandled


@defdata.register(dist.HalfNormal)
class HalfNormalTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.HalfNormal, *args, **kwargs)


@defterm.register(dist.HalfNormal)
def _embed_halfnormal(d: dist.HalfNormal) -> Term[dist.HalfNormal]:
    return HalfNormal(d.scale)


@defop
def LKJCholesky(*args, **kwargs) -> dist.LKJCholesky:
    raise NotHandled


@defdata.register(dist.LKJCholesky)
class LKJCholeskyTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.LKJCholesky, *args, **kwargs)


@defterm.register(dist.LKJCholesky)
def _embed_lkjcholesky(d: dist.LKJCholesky) -> Term[dist.LKJCholesky]:
    return LKJCholesky(d.dim, d.concentration)


@defop
def MultivariateNormal(*args, **kwargs) -> dist.MultivariateNormal:
    raise NotHandled


@defdata.register(dist.MultivariateNormal)
class MultivariateNormalTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.MultivariateNormal, *args, **kwargs)


@defterm.register(dist.MultivariateNormal)
def _embed_multivariatenormal(
    d: dist.MultivariateNormal,
) -> Term[dist.MultivariateNormal]:
    return MultivariateNormal(d.loc, d.scale_tril)


@defop
def Pareto(*args, **kwargs) -> dist.Pareto:
    raise NotHandled


@defdata.register(dist.Pareto)
class ParetoTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.Pareto, *args, **kwargs)


@defterm.register(dist.Pareto)
def _embed_pareto(d: dist.Pareto) -> Term[dist.Pareto]:
    return Pareto(d.scale, d.alpha)


@defop
def Uniform(*args, **kwargs) -> dist.Uniform:
    raise NotHandled


@defdata.register(dist.Uniform)
class UniformTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.Uniform, *args, **kwargs)


@defterm.register(dist.Uniform)
def _embed_uniform(d: dist.Uniform) -> Term[dist.Uniform]:
    return Uniform(d.low, d.high)


@defop
def VonMises(*args, **kwargs) -> dist.VonMises:
    raise NotHandled


@defdata.register(dist.VonMises)
class VonMisesTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.VonMises, *args, **kwargs)


@defterm.register(dist.VonMises)
def _embed_vonmises(d: dist.VonMises) -> Term[dist.VonMises]:
    return VonMises(d.loc, d.concentration)


@defop
def Weibull(*args, **kwargs) -> dist.Weibull:
    raise NotHandled


@defdata.register(dist.Weibull)
class WeibullTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.Weibull, *args, **kwargs)


@defterm.register(dist.Weibull)
def _embed_weibull(d: dist.Weibull) -> Term[dist.Weibull]:
    return Weibull(d.scale, d.concentration)


@defop
def Wishart(*args, **kwargs) -> dist.Wishart:
    raise NotHandled


@defdata.register(dist.Wishart)
class WishartTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.Wishart, *args, **kwargs)


@defterm.register(dist.Wishart)
def _embed_wishart(d: dist.Wishart) -> Term[dist.Wishart]:
    return Wishart(d.df, d.scale_tril)


@defop
def Delta(*args, **kwargs) -> dist.Delta:
    raise NotHandled


@defdata.register(dist.Delta)
class DeltaTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.Delta, *args, **kwargs)


@defterm.register(dist.Delta)
def _embed_delta(d: dist.Delta) -> Term[dist.Delta]:
    return Delta(d.v, d.log_density, d.event_dim)


@defop
def LowRankMultivariateNormal(*args, **kwargs) -> dist.LowRankMultivariateNormal:
    raise NotHandled


@defdata.register(dist.LowRankMultivariateNormal)
class LowRankMultivariateNormalTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.LowRankMultivariateNormal, *args, **kwargs)


@defterm.register(dist.LowRankMultivariateNormal)
def _embed_lowrankmultivariatenormal(
    d: dist.LowRankMultivariateNormal,
) -> Term[dist.LowRankMultivariateNormal]:
    return LowRankMultivariateNormal(d.loc, d.cov_factor, d.cov_diag)


@defop
def RelaxedBernoulliLogits(*args, **kwargs) -> dist.RelaxedBernoulliLogits:
    raise NotHandled


@defdata.register(dist.RelaxedBernoulliLogits)
class RelaxedBernoulliLogitsTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.RelaxedBernoulliLogits, *args, **kwargs)


@defterm.register(dist.RelaxedBernoulliLogits)
def _embed_relaxedbernoullilogits(
    d: dist.RelaxedBernoulliLogits,
) -> Term[dist.RelaxedBernoulliLogits]:
    return RelaxedBernoulliLogits(d.temperature, d.logits)


@defop
def Independent(*args, **kwargs) -> dist.Independent:
    raise NotHandled


@defdata.register(dist.Independent)
class IndependentTerm(_DistributionTerm):
    def __init__(self, *args, **kwargs):
        super().__init__(dist.Independent, *args, **kwargs)


@defterm.register(dist.Independent)
def _embed_independent(d: dist.Independent) -> Term[dist.Independent]:
    return Independent(d.base_dist, d.reinterpreted_batch_ndims)
