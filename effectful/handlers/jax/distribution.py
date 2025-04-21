import functools
from typing import Annotated, Any, Collection, Mapping, TypeVar, cast

import jax
import numpyro.distributions as dist
import tree

import effectful.handlers.jax.numpy as jnp
from effectful.ops.dims import _bind_dims, _unbind_dims, bind_dims, unbind_dims
from effectful.ops.semantics import apply, runner, typeof
from effectful.ops.syntax import Scoped, defdata, defop, defterm
from effectful.ops.types import Operation, Term

from .handlers import _register_jax_op, is_eager_array, jax_getitem, sizesof
from .terms import _EagerArrayTerm

A = TypeVar("A")
B = TypeVar("B")


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


@_unbind_dims.register
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
        if isinstance(a, jax.Array):
            _validate_batch_shape(a)
            return unbind_dims(a, *names)
        elif isinstance(a, dist.Distribution):
            return unbind_dims(a, *names)
        else:
            return a

    # Convert to a term in a context that does not evaluate distribution constructors.
    def _apply(intp, op, *args, **kwargs):
        typ = op.__type_rule__(*args, **kwargs)
        if issubclass(typ, dist.Distribution):
            return defdata(op, *args, **kwargs)
        return apply.__default_rule__({}, op, *args, **kwargs)

    with runner({apply: _apply}):
        d = defterm(d)

    if not (isinstance(d, Term) and typeof(d) is dist.Distribution):
        raise NotImplementedError

    new_d = d.op(
        *[_to_named(a) for a in d.args],
        **{k: _to_named(v) for (k, v) in d.kwargs.items()},
    )
    return new_d


@_bind_dims.register
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
            # We are really assuming that only one distriution appears in our arguments. This is sufficient for cases
            # like Independent and TransformedDistribution
            return bind_dims(a, *indices)
        else:
            return a

    # Convert to a term in a context that does not evaluate distribution constructors.
    def _apply(intp, op, *args, **kwargs):
        typ = op.__type_rule__(*args, **kwargs)
        if issubclass(typ, dist.Distribution):
            return defdata(op, *args, **kwargs)
        return apply.__default_rule__({}, op, *args, **kwargs)

    with runner({apply: _apply}):
        d = defterm(d)

    if not (isinstance(d, Term) and typeof(d) is dist.Distribution):
        raise NotImplementedError

    sizes = sizesof(d)
    indices = {k: sizes[k] for k in names}

    pos_args = [_to_positional(a, indices) for a in d.args]
    pos_kwargs = {k: _to_positional(v, indices) for (k, v) in d.kwargs.items()}
    new_d = d.op(*pos_args, **pos_kwargs)

    return new_d


@functools.cache
def _register_distribution_op(
    dist_constr: type[dist.Distribution],
) -> Operation[Any, dist.Distribution]:
    # introduce a wrapper so that we can control type annotations
    def wrapper(*args, **kwargs) -> dist.Distribution:
        if any(isinstance(a, Term) for a in tree.flatten((args, kwargs))):
            raise NotImplementedError
        return dist_constr(*args, **kwargs)

    return defop(wrapper, name=dist_constr.__name__)


@defdata.register(dist.Distribution)
def _(op, *args, **kwargs):
    if all(
        not isinstance(a, Term) or is_eager_array(a) or isinstance(a, dist.Distribution)
        for a in tree.flatten((args, kwargs))
    ):
        return _DistributionTerm(op, *args, **kwargs)
    else:
        return defdata.dispatch(object)(op, *args, **kwargs)


def _broadcast_to_named(t, sizes):
    missing_dims = set(sizes) - set(sizesof(t))
    t_broadcast = jnp.broadcast_to(
        t, tuple(sizes[dim] for dim in missing_dims) + t.shape
    )
    return jax_getitem(t_broadcast, tuple(dim() for dim in missing_dims))


class _DistributionTerm(dist.Distribution):
    """A distribution wrapper that satisfies the Term interface.

    Represented as a term of the form D(*args, **kwargs) where D is the
    distribution constructor.

    Note: When we construct instances of this class, we put distribution
    parameters that can be expanded in the args list and those that cannot in
    the kwargs list.

    """

    _op: Operation[Any, dist.Distribution]
    _args: tuple
    _kwargs: dict

    def __init__(self, op: Operation[Any, dist.Distribution], *args, **kwargs):
        self._op = op
        self._args = args
        self._kwargs = kwargs
        self.__indices = None
        self.__pos_base_dist = None

    @property
    def _indices(self):
        if self.__indices is None:
            self.__indices = sizesof(self)
        return self.__indices

    @property
    def _pos_base_dist(self):
        if self.__pos_base_dist is None:
            self.__pos_base_dist = bind_dims(self, *self._indices)
        return self.__pos_base_dist

    @property
    def op(self):
        return self._op

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def batch_shape(self):
        return self._pos_base_dist.batch_shape[len(self._indices) :]

    @property
    def has_rsample(self) -> bool:
        return self._pos_base_dist.has_rsample

    @property
    def event_shape(self):
        return self._pos_base_dist.event_shape

    def rsample(self, key, sample_shape=()):
        return self._reindex_sample(
            self._pos_base_dist.rsample(key, sample_shape), sample_shape
        )

    def sample(self, key, sample_shape=()):
        return self._reindex_sample(
            self._pos_base_dist.sample(key, sample_shape), sample_shape
        )

    def _reindex_sample(self, value, sample_shape):
        index = (slice(None),) * len(sample_shape) + tuple(i() for i in self._indices)
        ret = jax_getitem(value, index)
        assert isinstance(ret, _EagerArrayTerm)
        return ret

    def log_prob(self, value):
        value = bind_dims(_broadcast_to_named(value, self._indices), *self._indices)
        return self._reindex_sample(
            _register_jax_op(self._pos_base_dist.log_prob)(value), ()
        )

    @property
    def mean(self):
        return self._reindex_sample(self._pos_base_dist.mean, ())

    @property
    def variance(self):
        return self._reindex_sample(self._pos_base_dist.variance, ())

    def enumerate_support(self, expand=True):
        return self._reindex_sample(self._pos_base_dist.enumerate_support(expand), ())

    def entropy(self):
        return self._pos_base_dist.entropy()

    def to_event(self, reinterpreted_batch_ndims=None):
        raise NotImplementedError

    def expand(self, batch_shape):
        batch_shape = tuple(batch_shape)
        if batch_shape == self.batch_shape:
            return self

        # Helper function to expand a single argument
        def expand_arg(arg):
            if isinstance(arg, jax.Array) and hasattr(arg, "shape"):
                # Broadcast array arguments to the new batch shape
                arg_batch_shape = (
                    arg.shape[: -self._pos_base_dist.event_dim]
                    if self._pos_base_dist.event_dim > 0
                    else arg.shape
                )
                if len(arg_batch_shape) > 0:
                    # Ensure the batch dimensions are compatible
                    if len(batch_shape) >= len(arg_batch_shape):
                        new_shape = batch_shape + arg.shape[len(arg_batch_shape) :]
                        return jnp.broadcast_to(arg, new_shape)
                    else:
                        # If the new batch shape has fewer dimensions, we need to ensure they're compatible
                        if arg_batch_shape[-len(batch_shape) :] == batch_shape:
                            return arg
                        else:
                            raise ValueError(
                                f"Cannot expand batch shape {arg_batch_shape} to {batch_shape}"
                            )
                else:
                    # If there are no batch dimensions, just repeat the argument
                    new_shape = batch_shape + arg.shape
                    return jnp.broadcast_to(arg, new_shape)
            else:
                return arg

        # Create expanded versions of all arguments
        expanded_args = [expand_arg(arg) for arg in self._args]

        # Create expanded versions of all keyword arguments
        expanded_kwargs = {key: expand_arg(arg) for key, arg in self._kwargs.items()}

        # Create a new distribution with the expanded arguments
        return type(self)(self._op, *expanded_args, **expanded_kwargs)

    __repr__ = Term.__repr__
    __str__ = Term.__str__


Term.register(_DistributionTerm)


@defterm.register(dist.Distribution)
def _embed_distribution(dist: dist.Distribution) -> Term[dist.Distribution]:
    raise ValueError(
        f"No embedding provided for distribution of type {type(dist).__name__}."
    )


@defterm.register(dist.Cauchy)
@defterm.register(dist.Gumbel)
@defterm.register(dist.Laplace)
@defterm.register(dist.LogNormal)
@defterm.register(dist.Logistic)
@defterm.register(dist.Normal)
@defterm.register(dist.StudentT)
def _embed_loc_scale(d: dist.Distribution) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.loc, d.scale)


@defterm.register(dist.BernoulliProbs)
@defterm.register(dist.CategoricalProbs)
@defterm.register(dist.GeometricProbs)
def _embed_probs(d: dist.Distribution) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.probs)


@defterm.register(dist.BernoulliLogits)
@defterm.register(dist.CategoricalLogits)
@defterm.register(dist.GeometricLogits)
def _embed_probs(d: dist.Distribution) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.logits)


@defterm.register(dist.Beta)
@defterm.register(dist.Kumaraswamy)
def _embed_beta(d: dist.Distribution) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.concentration1, d.concentration0)


@defterm.register(dist.BinomialProbs)
@defterm.register(dist.NegativeBinomialProbs)
@defterm.register(dist.MultinomialProbs)
def _embed_binomial_probs(d: dist.Distribution) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.probs, d.total_count)


@defterm.register(dist.BinomialLogits)
@defterm.register(dist.NegativeBinomialLogits)
@defterm.register(dist.MultinomialLogits)
def _embed_binomial_logits(d: dist.Distribution) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.logits, d.total_count)


@defterm.register
def _embed_chi2(d: dist.Chi2) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.df)


@defterm.register
def _embed_dirichlet(d: dist.Dirichlet) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.concentration)


@defterm.register
def _embed_dirichlet_multinomial(
    d: dist.DirichletMultinomial,
) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(
        d.concentration, total_count=d.total_count
    )


@defterm.register(dist.Exponential)
@defterm.register(dist.Poisson)
def _embed_exponential(d: dist.Distribution) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.rate)


@defterm.register
def _embed_gamma(d: dist.Gamma) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.concentration, d.rate)


@defterm.register(dist.HalfCauchy)
@defterm.register(dist.HalfNormal)
def _embed_half_cauchy(d: dist.Distribution) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.scale)


@defterm.register
def _embed_lkj_cholesky(d: dist.LKJCholesky) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.dim, concentration=d.concentration)


@defterm.register
def _embed_multivariate_normal(d: dist.MultivariateNormal) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.loc, scale_tril=d.scale_tril)


@defterm.register
def _embed_pareto(d: dist.Pareto) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.scale, d.alpha)


@defterm.register
def _embed_uniform(d: dist.Uniform) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.low, d.high)


@defterm.register
def _embed_von_mises(d: dist.VonMises) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.loc, d.concentration)


@defterm.register
def _embed_weibull(d: dist.Weibull) -> Term[dist.Distribution]:
    return _register_distribution_op(dist.Weibull)(d.scale, d.concentration)


@defterm.register
def _embed_wishart(d: dist.Wishart) -> Term[dist.Distribution]:
    return _register_distribution_op(dist.Wishart)(d.df, d.scale_tril)


@defterm.register
def _embed_delta(d: dist.Delta) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(
        d.v, log_density=d.log_density, event_dim=d.event_dim
    )


@defterm.register
def _embed_low_rank_multivariate_normal(
    d: dist.LowRankMultivariateNormal,
) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.loc, d.cov_factor, d.cov_diag)


@defterm.register
def _embed_relaxed_bernoulli_logits(
    d: dist.RelaxedBernoulliLogits,
) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.temperature, d.logits)


@defterm.register
def _embed_independent(d: dist.Independent) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.base_dist, d.reinterpreted_batch_ndims)


BernoulliLogits = _register_distribution_op(dist.BernoulliLogits)
BernoulliProbs = _register_distribution_op(dist.BernoulliProbs)
Beta = _register_distribution_op(dist.Beta)
BinomialProbs = _register_distribution_op(dist.BinomialProbs)
BinomialLogits = _register_distribution_op(dist.BinomialLogits)
CategoricalLogits = _register_distribution_op(dist.CategoricalLogits)
CategoricalProbs = _register_distribution_op(dist.CategoricalProbs)
Cauchy = _register_distribution_op(dist.Cauchy)
Chi2 = _register_distribution_op(dist.Chi2)
Delta = _register_distribution_op(dist.Delta)
Dirichlet = _register_distribution_op(dist.Dirichlet)
DirichletMultinomial = _register_distribution_op(dist.DirichletMultinomial)
Distribution = _register_distribution_op(dist.Distribution)
Exponential = _register_distribution_op(dist.Exponential)
Gamma = _register_distribution_op(dist.Gamma)
GeometricLogits = _register_distribution_op(dist.GeometricLogits)
GeometricProbs = _register_distribution_op(dist.GeometricProbs)
Gumbel = _register_distribution_op(dist.Gumbel)
HalfCauchy = _register_distribution_op(dist.HalfCauchy)
HalfNormal = _register_distribution_op(dist.HalfNormal)
Independent = _register_distribution_op(dist.Independent)
Kumaraswamy = _register_distribution_op(dist.Kumaraswamy)
LKJCholesky = _register_distribution_op(dist.LKJCholesky)
Laplace = _register_distribution_op(dist.Laplace)
LogNormal = _register_distribution_op(dist.LogNormal)
Logistic = _register_distribution_op(dist.Logistic)
LowRankMultivariateNormal = _register_distribution_op(dist.LowRankMultivariateNormal)
MultinomialProbs = _register_distribution_op(dist.MultinomialProbs)
MultinomialLogits = _register_distribution_op(dist.MultinomialLogits)
MultivariateNormal = _register_distribution_op(dist.MultivariateNormal)
NegativeBinomialProbs = _register_distribution_op(dist.NegativeBinomialProbs)
NegativeBinomialLogits = _register_distribution_op(dist.NegativeBinomialLogits)
Normal = _register_distribution_op(dist.Normal)
Pareto = _register_distribution_op(dist.Pareto)
Poisson = _register_distribution_op(dist.Poisson)
RelaxedBernoulliLogits = _register_distribution_op(dist.RelaxedBernoulliLogits)
StudentT = _register_distribution_op(dist.StudentT)
Uniform = _register_distribution_op(dist.Uniform)
VonMises = _register_distribution_op(dist.VonMises)
Weibull = _register_distribution_op(dist.Weibull)
Wishart = _register_distribution_op(dist.Wishart)
