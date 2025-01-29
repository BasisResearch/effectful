import functools
import typing
import warnings
from typing import (
    Annotated,
    Any,
    Collection,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

try:
    import pyro
except ImportError:
    raise ImportError("Pyro is required to use effectful.handlers.pyro.")

import pyro.distributions as dist
from pyro.distributions.torch_distribution import (
    TorchDistribution,
    TorchDistributionMixin,
)

try:
    import torch
except ImportError:
    raise ImportError("PyTorch is required to use effectful.handlers.pyro.")

from typing_extensions import ParamSpec

from effectful.handlers.torch import Indexable, sizesof, to_tensor
from effectful.ops.semantics import call
from effectful.ops.syntax import Scoped, defop, defterm
from effectful.ops.types import Operation, Term

P = ParamSpec("P")
A = TypeVar("A")
B = TypeVar("B")


@defop
def pyro_sample(
    name: str,
    fn: TorchDistributionMixin,
    *args,
    obs: Optional[torch.Tensor] = None,
    obs_mask: Optional[torch.BoolTensor] = None,
    mask: Optional[torch.BoolTensor] = None,
    infer: Optional[pyro.poutine.runtime.InferDict] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Operation to sample from a Pyro distribution. See :func:`pyro.sample`.
    """
    with pyro.poutine.mask(mask=mask if mask is not None else True):
        return pyro.sample(
            name, fn, *args, obs=obs, obs_mask=obs_mask, infer=infer, **kwargs
        )


class PyroShim(pyro.poutine.messenger.Messenger):
    """Pyro handler that wraps all sample sites in a custom effectful type.

    .. note::

      This handler should be installed around any Pyro model that you want to
      use effectful handlers with.

    **Example usage**:

    >>> import pyro.distributions as dist
    >>> from effectful.ops.semantics import fwd, handler
    >>> torch.distributions.Distribution.set_default_validate_args(False)

    It can be used as a decorator:

    >>> @PyroShim()
    ... def model():
    ...     return pyro.sample("x", dist.Normal(0, 1))

    It can also be used as a context manager:

    >>> with PyroShim():
    ...     x = pyro.sample("x", dist.Normal(0, 1))

    When :class:`PyroShim` is installed, all sample sites perform the
    :func:`pyro_sample` effect, which can be handled by an effectful
    interpretation.

    >>> def log_sample(name, *args, **kwargs):
    ...     print(f"Sampled {name}")
    ...     return fwd()

    >>> with PyroShim(), handler({pyro_sample: log_sample}):
    ...     x = pyro.sample("x", dist.Normal(0, 1))
    ...     y = pyro.sample("y", dist.Normal(0, 1))
    Sampled x
    Sampled y
    """

    _current_site: Optional[str]

    def __enter__(self):
        if any(isinstance(m, PyroShim) for m in pyro.poutine.runtime._PYRO_STACK):
            warnings.warn("PyroShim should be installed at most once.")
        return super().__enter__()

    @staticmethod
    def _broadcast_to_named(
        t: torch.Tensor, shape: torch.Size, indices: Mapping[Operation[[], int], int]
    ) -> Tuple[torch.Tensor, "Naming"]:
        """Convert a tensor `t` to a fully positional tensor that is
        broadcastable with the positional representation of tensors of shape
        |shape|, |indices|.

        """
        t_indices = sizesof(t)

        if len(t.shape) < len(shape):
            t = t.expand(shape)

        # create a positional dimension for every named index in the target shape
        name_to_dim = {}
        for i, (k, v) in enumerate(reversed(list(indices.items()))):
            if k in t_indices:
                t = to_tensor(t, [k])
            else:
                t = t.expand((v,) + t.shape)
            name_to_dim[k] = -len(shape) - i - 1

        # create a positional dimension for every remaining named index in `t`
        n_batch_and_dist_named = len(t.shape)
        for i, k in enumerate(reversed(list(sizesof(t).keys()))):
            t = to_tensor(t, [k])
            name_to_dim[k] = -n_batch_and_dist_named - i - 1

        return t, Naming(name_to_dim)

    def _pyro_sample(self, msg: pyro.poutine.runtime.Message) -> None:
        if typing.TYPE_CHECKING:
            assert msg["type"] == "sample"
            assert msg["name"] is not None
            assert msg["infer"] is not None
            assert isinstance(msg["fn"], TorchDistributionMixin)

        if pyro.poutine.util.site_is_subsample(msg) or pyro.poutine.util.site_is_factor(
            msg
        ):
            return

        if getattr(self, "_current_site", None) == msg["name"]:
            if "_markov_scope" in msg["infer"] and self._current_site:
                msg["infer"]["_markov_scope"].pop(self._current_site, None)

            dist = msg["fn"]
            obs = msg["value"] if msg["is_observed"] else None

            # pdist shape: | named1 | batch_shape | event_shape |
            # obs shape: | batch_shape | event_shape |, | named2 | where named2 may overlap named1
            indices = sizesof(dist)
            pdist, naming = positional_distribution(dist)

            if msg["mask"] is None:
                mask = torch.tensor(True)
            elif isinstance(msg["mask"], bool):
                mask = torch.tensor(msg["mask"])
            else:
                mask = msg["mask"]

            pos_mask, _ = PyroShim._broadcast_to_named(mask, dist.batch_shape, indices)

            pos_obs: Optional[torch.Tensor] = None
            if obs is not None:
                pos_obs, naming = PyroShim._broadcast_to_named(
                    obs, dist.shape(), indices
                )

            for var, dim in naming.name_to_dim.items():
                frame = pyro.poutine.indep_messenger.CondIndepStackFrame(
                    name=str(var), dim=dim, size=-1, counter=0
                )
                msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]

            msg["fn"] = pdist
            msg["value"] = pos_obs
            msg["mask"] = pos_mask
            msg["infer"]["_index_naming"] = naming  # type: ignore

            assert sizesof(msg["value"]) == {}
            assert sizesof(msg["mask"]) == {}

            return

        try:
            self._current_site = msg["name"]
            msg["value"] = pyro_sample(
                msg["name"],
                msg["fn"],
                obs=msg["value"] if msg["is_observed"] else None,
                infer=msg["infer"].copy(),
            )
        finally:
            self._current_site = None

        # flags to guarantee commutativity of condition, intervene, trace
        msg["stop"] = True
        msg["done"] = True
        msg["mask"] = False
        msg["is_observed"] = True
        msg["infer"]["is_auxiliary"] = True
        msg["infer"]["_do_not_trace"] = True

    def _pyro_post_sample(self, msg: pyro.poutine.runtime.Message) -> None:
        infer = msg.get("infer")
        if infer is None or "_index_naming" not in infer:
            return

        # note: Pyro uses a TypedDict for infer, so it doesn't know we've stored this key
        naming = infer["_index_naming"]  # type: ignore

        value = msg["value"]

        if value is not None:
            # note: is it safe to assume that msg['fn'] is a distribution?
            assert isinstance(msg["fn"], TorchDistribution)
            dist_shape: tuple[int, ...] = msg["fn"].batch_shape + msg["fn"].event_shape
            if len(value.shape) < len(dist_shape):
                value = value.broadcast_to(
                    torch.broadcast_shapes(value.shape, dist_shape)
                )
            value = naming.apply(value)
            msg["value"] = value


class Naming:
    """
    A mapping from dimensions (indexed from the right) to names.
    """

    def __init__(self, name_to_dim: Mapping[Operation[[], int], int]):
        assert all(v < 0 for v in name_to_dim.values())
        self.name_to_dim = name_to_dim

    @staticmethod
    def from_shape(names: Collection[Operation[[], int]], event_dims: int) -> "Naming":
        """Create a naming from a set of indices and the number of event dimensions.

        The resulting naming converts tensors of shape
        ``| batch_shape | named | event_shape |``
        to tensors of shape ``| batch_shape | event_shape |, | named |``.

        """
        assert event_dims >= 0
        return Naming({n: -event_dims - len(names) + i for i, n in enumerate(names)})

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        indexes: List[Any] = [slice(None)] * (len(value.shape))
        for n, d in self.name_to_dim.items():
            indexes[len(value.shape) + d] = n()
        return Indexable(value)[tuple(indexes)]

    def __repr__(self):
        return f"Naming({self.name_to_dim})"


@defop
def named_distribution(
    dist: Annotated[TorchDistribution, Scoped[A]],
    *names: Annotated[Operation[[], int], Scoped[B]],
) -> Annotated[TorchDistribution, Scoped[A | B]]:
    match defterm(dist):
        case Term(op=_call, args=(dist_constr, *args)) if _call is call:
            named_args = []
            for a in args:
                assert isinstance(a, torch.Tensor)
                named_args.append(
                    Indexable(typing.cast(torch.Tensor, a))[tuple(n() for n in names)]
                )
            assert callable(dist_constr)
            return defterm(dist_constr(*named_args))
        case _:
            raise NotImplementedError


@defop
def positional_distribution(
    dist: Annotated[TorchDistribution, Scoped[A]]
) -> Tuple[TorchDistribution, Naming]:
    match defterm(dist):
        case Term(op=_call, args=(dist_constr, *args)) if _call is call:
            assert callable(dist_constr)
            base_dist = dist_constr(*args)
            indices = sizesof(base_dist).keys()
            n_base = len(base_dist.batch_shape) + len(base_dist.event_shape)
            naming = Naming.from_shape(indices, n_base)
            pos_args = [to_tensor(a, indices) for a in args]
            pos_dist = dist_constr(*pos_args)
            return defterm(pos_dist), naming
        case _:
            raise NotImplementedError


@Term.register
class _DistributionTerm(TorchDistribution):
    """A distribution wrapper that satisfies the Term interface.

    Represented as a term of the form call(D, *args, **kwargs) where D is the
    distribution constructor.

    """

    op: Operation = call
    args: tuple
    kwargs: Mapping[str, Any] = {}

    __match_args__ = ("op", "args", "kwargs")

    def __init__(self, dist_constr: Type[TorchDistribution], *args):
        self.args = (dist_constr,) + tuple(defterm(a) for a in args)

    @property
    def _base_dist(self):
        return self.args[0](*self.args[1:])

    @property
    def has_rsample(self):
        return self._base_dist.has_rsample

    @property
    def batch_shape(self):
        return self._base_dist.batch_shape

    @property
    def event_shape(self):
        return self._base_dist.event_shape

    @property
    def has_enumerate_support(self):
        return self._base_dist.has_enumerate_support

    @property
    def arg_constraints(self):
        return self._base_dist.arg_constraints

    @property
    def support(self):
        return self._base_dist.support

    def sample(self, sample_shape=torch.Size()):
        return self._base_dist.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self._base_dist.rsample(sample_shape)

    def log_prob(self, value):
        return self._base_dist.log_prob(value)

    def enumerate_support(self, expand=True):
        return self._base_dist.enumerate_support(expand)

    def __getitem__(self, key: Collection[Operation[[], int]]):
        return named_distribution(self, *key)


@defterm.register(TorchDistribution)
@defterm.register(TorchDistributionMixin)
@functools.singledispatch
def _embed_dist(dist: TorchDistribution) -> Term[TorchDistribution]:
    raise ValueError(
        "No embedding provided for distribution of type {type(dist).__name__}."
    )


@_embed_dist.register(dist.Bernoulli)
def _embed_bernoulli(d: dist.Bernoulli) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Bernoulli, d.probs)


@_embed_dist.register(dist.Beta)
def _embed_beta(d: dist.Beta) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Beta, d.concentration1, d.concentration0)


@_embed_dist.register(dist.Binomial)
def _embed_binomial(d: dist.Binomial) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Binomial, d.total_count, d.probs)


@_embed_dist.register(dist.Categorical)
def _embed_categorical(d: dist.Categorical) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Categorical, d.probs)


@_embed_dist.register(dist.Cauchy)
def _embed_cauchy(d: dist.Cauchy) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Cauchy, d.loc, d.scale)


@_embed_dist.register(dist.Chi2)
def _embed_chi2(d: dist.Chi2) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Chi2, d.df)


@_embed_dist.register(dist.ContinuousBernoulli)
def _embed_continuous_bernoulli(
    d: dist.ContinuousBernoulli,
) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.ContinuousBernoulli, d.probs)


@_embed_dist.register(dist.Dirichlet)
def _embed_dirichlet(d: dist.Dirichlet) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Dirichlet, d.concentration)


@_embed_dist.register(dist.Exponential)
def _embed_exponential(d: dist.Exponential) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Exponential, d.rate)


@_embed_dist.register(dist.FisherSnedecor)
def _embed_fisher_snedecor(d: dist.FisherSnedecor) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.FisherSnedecor, d.df1, d.df2)


@_embed_dist.register(dist.Gamma)
def _embed_gamma(d: dist.Gamma) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Gamma, d.concentration, d.rate)


@_embed_dist.register(dist.Geometric)
def _embed_geometric(d: dist.Geometric) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Geometric, d.probs)


@_embed_dist.register(dist.Gumbel)
def _embed_gumbel(d: dist.Gumbel) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Gumbel, d.loc, d.scale)


@_embed_dist.register(dist.HalfCauchy)
def _embed_half_cauchy(d: dist.HalfCauchy) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.HalfCauchy, d.scale)


@_embed_dist.register(dist.HalfNormal)
def _embed_half_normal(d: dist.HalfNormal) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.HalfNormal, d.scale)


@_embed_dist.register(dist.Independent)
def _embed_independent(d: dist.Independent) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Independent, d.base_dist, d.reinterpreted_batch_ndims)


@_embed_dist.register(dist.Kumaraswamy)
def _embed_kumaraswamy(d: dist.Kumaraswamy) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Kumaraswamy, d.concentration1, d.concentration0)


@_embed_dist.register(dist.LKJCholesky)
def _embed_lkj_cholesky(d: dist.LKJCholesky) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.LKJCholesky, d.dim, d.concentration)


@_embed_dist.register(dist.Laplace)
def _embed_laplace(d: dist.Laplace) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Laplace, d.loc, d.scale)


@_embed_dist.register(dist.LogNormal)
def _embed_log_normal(d: dist.LogNormal) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.LogNormal, d.loc, d.scale)


@_embed_dist.register(dist.LogisticNormal)
def _embed_logistic_normal(d: dist.LogisticNormal) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.LogisticNormal, d.loc, d.scale)


@_embed_dist.register(dist.Multinomial)
def _embed_multinomial(d: dist.Multinomial) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Multinomial, d.total_count, d.probs)


@_embed_dist.register(dist.MultivariateNormal)
def _embed_multivariate_normal(
    d: dist.MultivariateNormal,
) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.MultivariateNormal, d.loc, d.scale_tril)


@_embed_dist.register(dist.NegativeBinomial)
def _embed_negative_binomial(d: dist.NegativeBinomial) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.NegativeBinomial, d.total_count, d.probs)


@_embed_dist.register(dist.Normal)
def _embed_normal(d: dist.Normal) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Normal, d.loc, d.scale)


@_embed_dist.register(dist.OneHotCategorical)
def _embed_one_hot_categorical(d: dist.OneHotCategorical) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.OneHotCategorical, d.probs)


@_embed_dist.register(dist.OneHotCategoricalStraightThrough)
def _embed_one_hot_categorical_straight_through(
    d: dist.OneHotCategoricalStraightThrough,
) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.OneHotCategoricalStraightThrough, d.probs)


@_embed_dist.register(dist.Pareto)
def _embed_pareto(d: dist.Pareto) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Pareto, d.scale, d.alpha)


@_embed_dist.register(dist.Poisson)
def _embed_poisson(d: dist.Poisson) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Poisson, d.rate)


@_embed_dist.register(dist.RelaxedBernoulli)
def _embed_relaxed_bernoulli(d: dist.RelaxedBernoulli) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.RelaxedBernoulli, d.temperature, d.probs)


@_embed_dist.register(dist.RelaxedOneHotCategorical)
def _embed_relaxed_one_hot_categorical(
    d: dist.RelaxedOneHotCategorical,
) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.RelaxedOneHotCategorical, d.temperature, d.probs)


@_embed_dist.register(dist.StudentT)
def _embed_student_t(d: dist.StudentT) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.StudentT, d.df, d.loc, d.scale)


@_embed_dist.register(dist.Uniform)
def _embed_uniform(d: dist.Uniform) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Uniform, d.low, d.high)


@_embed_dist.register(dist.VonMises)
def _embed_von_mises(d: dist.VonMises) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.VonMises, d.loc, d.concentration)


@_embed_dist.register(dist.Weibull)
def _embed_weibull(d: dist.Weibull) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Weibull, d.scale, d.concentration)


@_embed_dist.register(dist.Wishart)
def _embed_wishart(d: dist.Wishart) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Wishart, d.df, d.scale_tril)


def pyro_module_shim(
    module: type[pyro.nn.module.PyroModule],
) -> type[pyro.nn.module.PyroModule]:
    """Wrap a :class:`PyroModule` in a :class:`PyroShim`.

    Returns a new subclass of :class:`PyroModule` that wraps calls to
    :func:`forward` in a :class:`PyroShim`.

    **Example usage**:

    .. code-block:: python

        class SimpleModel(PyroModule):
            def forward(self):
                return pyro.sample("y", dist.Normal(0, 1))

        SimpleModelShim = pyro_module_shim(SimpleModel)

    """

    class PyroModuleShim(module):  # type: ignore
        def forward(self, *args, **kwargs):
            with PyroShim():
                return super().forward(*args, **kwargs)

    return PyroModuleShim
