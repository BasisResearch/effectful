import functools
import typing
import warnings
from collections.abc import Collection, Mapping
from typing import (
    Annotated,
    Any,
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

from effectful.handlers.torch import sizesof, to_tensor
from effectful.ops.semantics import typeof
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
    obs: torch.Tensor | None = None,
    obs_mask: torch.BoolTensor | None = None,
    mask: torch.BoolTensor | None = None,
    infer: pyro.poutine.runtime.InferDict | None = None,
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

    _current_site: str | None

    def __enter__(self):
        if any(isinstance(m, PyroShim) for m in pyro.poutine.runtime._PYRO_STACK):
            warnings.warn("PyroShim should be installed at most once.")
        return super().__enter__()

    @staticmethod
    def _broadcast_to_named(
        t: torch.Tensor,
        shape: torch.Size,
        indices: Mapping[Operation[[], torch.Tensor], int],
    ) -> tuple[torch.Tensor, "Naming"]:
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

        # PyroShim turns each call to pyro.sample into two calls. The first
        # dispatches to pyro_sample and the effectful stack. The effectful stack
        # eventually calls pyro.sample again. We use state in PyroShim to
        # recognize that we've been called twice, and we dispatch to the pyro
        # stack.
        #
        # This branch handles the second call, so it massages the message to be
        # compatible with Pyro. In particular, it removes all named dimensions
        # and stores naming information in the message. Names are replaced by
        # _pyro_post_sample.
        if getattr(self, "_current_site", None) == msg["name"]:
            if "_index_naming" in msg:
                return

            # We need to identify this pyro shim during post-sample.
            msg["_pyro_shim_id"] = id(self)  # type: ignore[typeddict-unknown-key]

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

            assert set(sizesof(mask).keys()) <= (
                set(indices.keys()) | set(sizesof(obs).keys())
            )
            pos_mask, _ = PyroShim._broadcast_to_named(mask, dist.batch_shape, indices)

            pos_obs: torch.Tensor | None = None
            if obs is not None:
                pos_obs, naming = PyroShim._broadcast_to_named(
                    obs, dist.shape(), indices
                )

            # Each of the batch dimensions on the distribution gets a
            # cond_indep_stack frame.
            for var, dim in naming.name_to_dim.items():
                # There can be additional batch dimensions on the observation
                # that do not get frames, so only consider dimensions on the
                # distribution.
                if var in indices:
                    frame = pyro.poutine.indep_messenger.CondIndepStackFrame(
                        name=str(var),
                        # dims are indexed from the right of the batch shape
                        dim=dim + len(pdist.event_shape),
                        size=indices[var],
                        counter=0,
                    )
                    msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]

            msg["fn"] = pdist
            msg["value"] = pos_obs
            msg["mask"] = pos_mask
            msg["_index_naming"] = naming  # type: ignore

            assert sizesof(msg["value"]) == {}
            assert sizesof(msg["mask"]) == {}

        # This branch handles the first call to pyro.sample by calling pyro_sample.
        else:
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
        assert msg["value"] is not None

        # If this message has been handled already by a different pyro shim, ignore.
        if "_pyro_shim_id" in msg and msg["_pyro_shim_id"] != id(self):  # type: ignore[typeddict-item]
            return

        if getattr(self, "_current_site", None) == msg["name"]:
            assert "_index_naming" in msg

            # note: Pyro uses a TypedDict for infer, so it doesn't know we've stored this key
            naming = msg["_index_naming"]  # type: ignore

            value = msg["value"]

            # note: is it safe to assume that msg['fn'] is a distribution?
            dist_shape: tuple[int, ...] = msg["fn"].batch_shape + msg["fn"].event_shape  # type: ignore
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

    def __init__(self, name_to_dim: Mapping[Operation[[], torch.Tensor], int]):
        assert all(v < 0 for v in name_to_dim.values())
        self.name_to_dim = name_to_dim

    @staticmethod
    def from_shape(
        names: Collection[Operation[[], torch.Tensor]], event_dims: int
    ) -> "Naming":
        """Create a naming from a set of indices and the number of event dimensions.

        The resulting naming converts tensors of shape
        ``| batch_shape | named | event_shape |``
        to tensors of shape ``| batch_shape | event_shape |, | named |``.

        """
        assert event_dims >= 0
        return Naming({n: -event_dims - len(names) + i for i, n in enumerate(names)})

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        indexes: list[Any] = [slice(None)] * (len(value.shape))
        for n, d in self.name_to_dim.items():
            indexes[len(value.shape) + d] = n()
        return value[tuple(indexes)]

    def __repr__(self):
        return f"Naming({self.name_to_dim})"


@defop
def named_distribution(
    d: Annotated[TorchDistribution, Scoped[A | B]],
    *names: Annotated[Operation[[], torch.Tensor], Scoped[B]],
) -> Annotated[TorchDistribution, Scoped[A | B]]:
    def _to_named(a):
        if isinstance(a, torch.Tensor):
            return typing.cast(torch.Tensor, a)[tuple(n() for n in names)]
        elif isinstance(a, TorchDistribution):
            return named_distribution(a, *names)
        else:
            return a

    d = defterm(d)
    if not (isinstance(d, Term) and typeof(d) is TorchDistribution):
        raise NotImplementedError

    new_d = d.op(
        *[_to_named(a) for a in d.args],
        **{k: _to_named(v) for (k, v) in d.kwargs.items()},
    )
    assert new_d.event_shape == d.event_shape
    return new_d


@defop
def positional_distribution(
    d: Annotated[TorchDistribution, Scoped[A]],
) -> tuple[TorchDistribution, Naming]:
    def _to_positional(a, indices):
        if isinstance(a, torch.Tensor):
            return to_tensor(a, indices)
        elif isinstance(a, TorchDistribution):
            return positional_distribution(a)[0]
        else:
            return a

    d = defterm(d)
    if not (isinstance(d, Term) and typeof(d) is TorchDistribution):
        raise NotImplementedError

    shape = d.shape()
    indices = sizesof(d).keys()
    naming = Naming.from_shape(indices, len(shape))

    new_d = d.op(
        *[_to_positional(a, indices) for a in d.args],
        **{k: _to_positional(v, indices) for (k, v) in d.kwargs.items()},
    )

    assert new_d.event_shape == d.event_shape
    return new_d, naming


@functools.cache
def _register_distribution_op(
    dist_constr: type[TorchDistribution],
) -> Operation[Any, TorchDistribution]:
    # introduce a wrapper so that we can control type annotations
    def wrapper(*args, **kwargs) -> TorchDistribution:
        return dist_constr(*args, **kwargs)

    return defop(wrapper)


class _DistributionTerm(Term[TorchDistribution], TorchDistribution):
    """A distribution wrapper that satisfies the Term interface.

    Represented as a term of the form call(D, *args, **kwargs) where D is the
    distribution constructor.

    Note: When we construct instances of this class, we put distribution
    parameters that can be expanded in the args list and those that cannot in
    the kwargs list.

    """

    _op: Operation[Any, TorchDistribution]
    _args: tuple
    _kwargs: dict

    def __init__(self, dist_constr: type[TorchDistribution], *args, **kwargs):
        self._op = _register_distribution_op(dist_constr)
        self._args = tuple(defterm(a) for a in args)
        self._kwargs = {k: defterm(v) for (k, v) in kwargs.items()}

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
    def _base_dist(self):
        return self._op(*self.args, **self.kwargs)

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


@defterm.register(TorchDistribution)
@defterm.register(TorchDistributionMixin)
def _embed_distribution(dist: TorchDistribution) -> Term[TorchDistribution]:
    raise ValueError(
        f"No embedding provided for distribution of type {type(dist).__name__}."
    )


@defterm.register(dist.ExpandedDistribution)
def _embed_expanded(d: dist.ExpandedDistribution) -> Term[TorchDistribution]:
    if d._batch_shape == d.base_dist.batch_shape:
        return d.base_dist
    raise ValueError("Nontrivial ExpandedDistribution not implemented.")


@defterm.register(dist.Independent)
def _embed_independent(d) -> Term[TorchDistribution]:
    return _DistributionTerm(type(d), d.base_dist, d.reinterpreted_batch_ndims)


@defterm.register(dist.FoldedDistribution)
def _embed_folded(d) -> Term[TorchDistribution]:
    return _DistributionTerm(type(d), d.base_dist)


@defterm.register(dist.MaskedDistribution)
def _embed_masked(d) -> Term[TorchDistribution]:
    return _DistributionTerm(type(d), d.base_dist, d._mask)


@defterm.register(dist.Cauchy)
@defterm.register(dist.Gumbel)
@defterm.register(dist.Laplace)
@defterm.register(dist.LogNormal)
@defterm.register(dist.LogisticNormal)
@defterm.register(dist.Normal)
@defterm.register(dist.StudentT)
def _embed_loc_scale(d) -> Term[TorchDistribution]:
    return _DistributionTerm(type(d), d.loc, d.scale)


@defterm.register(dist.Bernoulli)
@defterm.register(dist.Categorical)
@defterm.register(dist.ContinuousBernoulli)
@defterm.register(dist.Geometric)
@defterm.register(dist.OneHotCategorical)
@defterm.register(dist.OneHotCategoricalStraightThrough)
def _embed_probs(d) -> Term[TorchDistribution]:
    return _DistributionTerm(type(d), d.probs)


@defterm.register(dist.Beta)
@defterm.register(dist.Kumaraswamy)
def _embed_beta(d) -> Term[TorchDistribution]:
    return _DistributionTerm(type(d), d.concentration1, d.concentration0)


@defterm.register(dist.Binomial)
def _embed_binomial(d: dist.Binomial) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Binomial, d.total_count, d.probs)


@defterm.register(dist.Chi2)
def _embed_chi2(d: dist.Chi2) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Chi2, d.df)


@defterm.register(dist.Dirichlet)
def _embed_dirichlet(d: dist.Dirichlet) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Dirichlet, d.concentration)


@defterm.register(dist.Exponential)
def _embed_exponential(d: dist.Exponential) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Exponential, d.rate)


@defterm.register(dist.FisherSnedecor)
def _embed_fisher_snedecor(d: dist.FisherSnedecor) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.FisherSnedecor, d.df1, d.df2)


@defterm.register(dist.Gamma)
def _embed_gamma(d: dist.Gamma) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Gamma, d.concentration, d.rate)


@defterm.register(dist.HalfCauchy)
@defterm.register(dist.HalfNormal)
def _embed_half_cauchy(d) -> Term[TorchDistribution]:
    return _DistributionTerm(type(d), d.scale)


@defterm.register(dist.LKJCholesky)
def _embed_lkj_cholesky(d: dist.LKJCholesky) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.LKJCholesky, d.dim, concentration=d.concentration)


@defterm.register(dist.Multinomial)
def _embed_multinomial(d: dist.Multinomial) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Multinomial, d.total_count, d.probs)


@defterm.register(dist.MultivariateNormal)
def _embed_multivariate_normal(
    d: dist.MultivariateNormal,
) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.MultivariateNormal, d.loc, scale_tril=d.scale_tril)


@defterm.register(dist.NegativeBinomial)
def _embed_negative_binomial(d: dist.NegativeBinomial) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.NegativeBinomial, d.total_count, d.probs)


@defterm.register(dist.Pareto)
def _embed_pareto(d: dist.Pareto) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Pareto, d.scale, d.alpha)


@defterm.register(dist.Poisson)
def _embed_poisson(d: dist.Poisson) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Poisson, d.rate)


@defterm.register(dist.RelaxedBernoulli)
@defterm.register(dist.RelaxedOneHotCategorical)
def _embed_relaxed(d) -> Term[TorchDistribution]:
    return _DistributionTerm(type(d), d.temperature, d.probs)


@defterm.register(dist.Uniform)
def _embed_uniform(d: dist.Uniform) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Uniform, d.low, d.high)


@defterm.register(dist.VonMises)
def _embed_von_mises(d: dist.VonMises) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.VonMises, d.loc, d.concentration)


@defterm.register(dist.Weibull)
def _embed_weibull(d: dist.Weibull) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Weibull, d.scale, d.concentration)


@defterm.register(dist.Wishart)
def _embed_wishart(d: dist.Wishart) -> Term[TorchDistribution]:
    return _DistributionTerm(dist.Wishart, d.df, d.scale_tril)


@defterm.register(dist.Delta)
def _embed_delta(d: dist.Delta) -> Term[TorchDistribution]:
    return _DistributionTerm(
        dist.Delta, d.v, log_density=d.log_density, event_dim=d.event_dim
    )


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
