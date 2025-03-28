import functools
from typing import Annotated, Any, Collection, Mapping, TypeVar

import jax
import numpyro.distributions as dist
import tree

from effectful.ops.semantics import apply, runner, typeof
from effectful.ops.syntax import Scoped, defdata, defop, defterm
from effectful.ops.types import Operation, Term

from .handlers import jax_getitem, sizesof, to_array

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


@defop
def named_distribution(
    d: Annotated[dist.Distribution, Scoped[A | B]],
    *names: Annotated[Operation[[], jax.Array], Scoped[B]],
) -> Annotated[dist.Distribution, Scoped[A | B]]:
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
            return typing.cast(jax.Array, a)[tuple(n() for n in names)]
        elif isinstance(a, dist.Distribution):
            return named_distribution(a, *names)
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
    assert new_d.event_shape == d.event_shape
    return new_d


@defop
def positional_distribution(
    d: Annotated[dist.Distribution, Scoped[A]],
) -> tuple[dist.Distribution, Naming]:
    def _to_positional(a, indices):
        if isinstance(a, jax.Array):
            # broadcast to full indexed shape
            existing_dims = set(sizesof(a).keys())
            missing_dims = set(indices) - existing_dims

            a_indexed = torch.broadcast_to(
                a, torch.Size([indices[dim] for dim in missing_dims]) + a.shape
            )[tuple(n() for n in missing_dims)]
            return to_array(a_indexed, *indices)
        elif isinstance(a, dist.Distribution):
            return positional_distribution(a)[0]
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

    shape = d.shape()
    indices = sizesof(d)
    naming = Naming.from_shape(indices, len(shape))

    pos_args = [_to_positional(a, indices) for a in d.args]
    pos_kwargs = {k: _to_positional(v, indices) for (k, v) in d.kwargs.items()}
    new_d = d.op(*pos_args, **pos_kwargs)

    assert new_d.event_shape == d.event_shape
    return new_d, naming


@functools.cache
def _register_distribution_op(
    dist_constr: type[dist.Distribution],
) -> Operation[Any, dist.Distribution]:
    # introduce a wrapper so that we can control type annotations
    def wrapper(*args, **kwargs) -> dist.Distribution:
        if any(isinstance(a, Term) for a in tree.flatten((args, kwargs))):
            raise NotImplementedError
        return dist_constr(*args, **kwargs)

    return defop(wrapper)


@defdata.register(dist.Distribution)
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
        self._args = tuple(defterm(a) for a in args)
        self._kwargs = {k: defterm(v) for (k, v) in kwargs.items()}
        self._indices = tuple(sizesof(self).keys())
        pos_args = tuple(to_array(a, *self._indices) for a in self._args)
        pos_kwargs = {k: to_array(v, *self._indices) for (k, v) in self._kwargs.items()}
        self._pos_base_dist = self._op(*pos_args, **pos_kwargs)

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
    def event_shape(self):
        return self._pos_base_dist.event_shape

    def _sample_n(self, key, n: int):
        return jax_getitem(
            self._pos_base_dist.sample(seed=key, sample_shape=(n,)), self._indices
        )

    def log_prob(self, value) -> jax.Array:
        # todo
        indices = sizesof(value)
        pos_log_prob = self._pos_base_dist.log_prob(to_array(value, *indices))
        return jax_getitem(pos_log_prob, self._indices)


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
def _embed_binomial(d: dist.Distribution) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.total_count, d.probs)


@defterm.register
def _embed_chi2(d: dist.Chi2) -> Term[dist.Distribution]:
    return _register_distribution_op(dist.Chi2)(d.df)


@defterm.register
def _embed_dirichlet(d: dist.Dirichlet) -> Term[dist.Distribution]:
    return _register_distribution_op(dist.Dirichlet)(d.concentration)


@defterm.register(dist.Exponential)
@defterm.register(dist.Poisson)
def _embed_exponential(d: dist.Distribution) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.rate)


@defterm.register
def _embed_gamma(d: dist.Gamma) -> Term[dist.Distribution]:
    return _register_distribution_op(dist.Gamma)(d.concentration, d.rate)


@defterm.register(dist.HalfCauchy)
@defterm.register(dist.HalfNormal)
def _embed_half_cauchy(d: dist.Distribution) -> Term[dist.Distribution]:
    return _register_distribution_op(type(d))(d.scale)


@defterm.register
def _embed_lkj_cholesky(d: dist.LKJCholesky) -> Term[dist.Distribution]:
    return _register_distribution_op(dist.LKJCholesky)(
        d.dim, concentration=d.concentration
    )


@defterm.register
def _embed_multivariate_normal(
    d: dist.MultivariateNormal,
) -> Term[dist.Distribution]:
    return _register_distribution_op(dist.MultivariateNormal)(
        d.loc, scale_tril=d.scale_tril
    )


@defterm.register
def _embed_pareto(d: dist.Pareto) -> Term[dist.Distribution]:
    return _register_distribution_op(dist.Pareto)(d.scale, d.alpha)


@defterm.register
def _embed_uniform(d: dist.Uniform) -> Term[dist.Distribution]:
    return _register_distribution_op(dist.Uniform)(d.low, d.high)


@defterm.register
def _embed_von_mises(d: dist.VonMises) -> Term[dist.Distribution]:
    return _register_distribution_op(dist.VonMises)(d.loc, d.concentration)


@defterm.register
def _embed_weibull(d: dist.Weibull) -> Term[dist.Distribution]:
    return _register_distribution_op(dist.Weibull)(d.scale, d.concentration)


@defterm.register
def _embed_wishart(d: dist.Wishart) -> Term[dist.Distribution]:
    return _register_distribution_op(dist.Wishart)(d.df, d.scale_tril)


@defterm.register
def _embed_delta(d: dist.Delta) -> Term[dist.Distribution]:
    return _register_distribution_op(dist.Delta)(
        d.v, log_density=d.log_density, event_dim=d.event_dim
    )
