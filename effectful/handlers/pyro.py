import functools
import typing
from collections.abc import Collection, Mapping
from typing import Any

import pyro.poutine.subsample_messenger

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

from effectful.handlers.torch import (
    bind_dims,
    sizesof,
    unbind_dims,
)
from effectful.internals.runtime import interpreter
from effectful.ops.semantics import apply, evaluate, handler, typeof
from effectful.ops.syntax import defdata, defop
from effectful.ops.types import NotHandled, Operation, Term


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

    # Tracks the named dimensions on any sample site that we have handled.
    # Ideally, this information would be carried on the sample message itself.
    # However, when using guides, sample sites are completely replaced by fresh
    # guide sample sites that do not carry the same infer dict.
    #
    # We can only restore the named dimensions on samples that we have handled
    # at least once in the shim.
    _index_naming: dict[str, Naming]

    def __init__(self):
        self._index_naming = {}

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

        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)

        if len(t.shape) < len(shape):
            t = t.expand(shape)

        # create a positional dimension for every named index in the target shape
        name_to_dim = {}
        for i, (k, v) in enumerate(reversed(list(indices.items()))):
            if k in t_indices:
                t = bind_dims(t, k)
            else:
                t = t.expand((v,) + t.shape)
            name_to_dim[k] = -len(shape) - i - 1

        # create a positional dimension for every remaining named index in `t`
        n_batch_and_dist_named = len(t.shape)
        for i, k in enumerate(reversed(list(sizesof(t).keys()))):
            t = bind_dims(t, k)
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

        if "pyro_shim_status" in msg["infer"]:
            handler_id, handler_stage = msg["infer"]["pyro_shim_status"]  # type: ignore
        else:
            handler_id = id(self)
            handler_stage = 0
            msg["infer"]["pyro_shim_status"] = (handler_id, handler_stage)  # type: ignore

        if handler_id != id(self):  # Never handle a message that is not ours.
            return

        assert handler_stage in (0, 1)

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
        if handler_stage == 1:
            if "_markov_scope" in msg["infer"]:
                msg["infer"]["_markov_scope"].pop(msg["name"], None)

            dist = msg["fn"]
            obs = msg["value"] if msg["is_observed"] else None

            # pdist shape: | named1 | batch_shape | event_shape |
            # obs shape: | batch_shape | event_shape |, | named2 | where named2 may overlap named1
            indices = sizesof(dist)
            naming = Naming.from_shape(indices, len(dist.shape()))
            pdist = bind_dims(dist, *indices.keys())

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
                        name=f"__index_plate_{var}",
                        # dims are indexed from the right of the batch shape
                        dim=dim + len(pdist.event_shape),
                        size=indices[var],
                        counter=0,
                    )
                    msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]

            msg["fn"] = pdist
            msg["value"] = pos_obs
            msg["mask"] = pos_mask

            # stash the index naming on the sample message so that future
            # consumers of the trace can get at it
            msg["_index_naming"] = naming  # type: ignore

            self._index_naming[msg["name"]] = naming

            assert sizesof(msg["value"]) == {}
            assert sizesof(msg["mask"]) == {}

        # This branch handles the first call to pyro.sample by calling pyro_sample.
        else:
            infer = msg["infer"].copy()
            infer["pyro_shim_status"] = (handler_id, 1)  # type: ignore

            msg["value"] = pyro_sample(
                msg["name"],
                msg["fn"],
                obs=msg["value"] if msg["is_observed"] else None,
                infer=infer,
            )

            # flags to guarantee commutativity of condition, intervene, trace
            msg["stop"] = True
            msg["done"] = True
            msg["mask"] = False
            msg["is_observed"] = True
            msg["infer"]["is_auxiliary"] = True
            msg["infer"]["_do_not_trace"] = True

    def _pyro_post_sample(self, msg: pyro.poutine.runtime.Message) -> None:
        if typing.TYPE_CHECKING:
            assert msg["name"] is not None
            assert msg["value"] is not None
            assert msg["infer"] is not None

        # If there is no shim status, assume that we are looking at a guide sample.
        # In this case, we should handle the sample and claim it as ours if we have naming
        # information for it.
        if "pyro_shim_status" not in msg["infer"]:
            # Except, of course, for subsample messages, which we should ignore.
            if (
                pyro.poutine.util.site_is_subsample(msg)
                or msg["name"] not in self._index_naming
            ):
                return
            msg["infer"]["pyro_shim_status"] = (id(self), 1)  # type: ignore

        # If this message has been handled already by a different pyro shim, ignore.
        handler_id, handler_stage = msg["infer"]["pyro_shim_status"]  # type: ignore
        if handler_id != id(self) or handler_stage < 1:
            return

        value = msg["value"]

        naming = self._index_naming.get(msg["name"], Naming({}))
        infer = msg["infer"] if msg["infer"] is not None else {}
        assert "enumerate" not in infer or len(naming.name_to_dim) == 0, (
            "Enumeration is not currently supported in PyroShim."
        )

        # note: is it safe to assume that msg['fn'] is a distribution?
        dist_shape: tuple[int, ...] = msg["fn"].batch_shape + msg["fn"].event_shape  # type: ignore
        if len(value.shape) < len(dist_shape):
            value = value.broadcast_to(torch.broadcast_shapes(value.shape, dist_shape))

        value = naming.apply(value)
        msg["value"] = value


PyroDistribution = (
    pyro.distributions.torch_distribution.TorchDistribution
    | pyro.distributions.torch_distribution.TorchDistributionMixin
)


@unbind_dims.register(pyro.distributions.torch_distribution.TorchDistribution)  # type: ignore
@unbind_dims.register(pyro.distributions.torch_distribution.TorchDistributionMixin)  # type: ignore
def _unbind_dims_distribution(
    value: pyro.distributions.torch_distribution.TorchDistribution,
    *names: Operation[[], torch.Tensor],
) -> pyro.distributions.torch_distribution.TorchDistribution:
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
        if isinstance(a, torch.Tensor):
            _validate_batch_shape(a)
            return typing.cast(torch.Tensor, a)[tuple(n() for n in names)]
        elif isinstance(a, TorchDistribution):
            return unbind_dims(a, *names)
        else:
            return a

    # Convert to a term in a context that does not evaluate distribution constructors.
    with handler({apply: defdata}):
        d = typing.cast(TorchDistribution, evaluate(value))

    if not (isinstance(d, Term) and typeof(d) is TorchDistribution):
        raise NotHandled

    new_d = d.op(
        *[_to_named(a) for a in d.args],
        **{k: _to_named(v) for (k, v) in d.kwargs.items()},
    )
    assert new_d.event_shape == d.event_shape
    return new_d


@bind_dims.register(pyro.distributions.torch_distribution.TorchDistribution)  # type: ignore
@bind_dims.register(pyro.distributions.torch_distribution.TorchDistributionMixin)  # type: ignore
def _bind_dims_distribution(
    value: pyro.distributions.torch_distribution.TorchDistribution,
    *names: Operation[[], torch.Tensor],
) -> pyro.distributions.torch_distribution.TorchDistribution:
    def _to_positional(a, indices):
        if isinstance(a, torch.Tensor):
            # broadcast to full indexed shape
            existing_dims = set(sizesof(a).keys())
            missing_dims = set(indices) - existing_dims

            a_indexed = torch.broadcast_to(
                a, torch.Size([indices[dim] for dim in missing_dims]) + a.shape
            )[tuple(n() for n in missing_dims)]
            return bind_dims(a_indexed, *names)
        elif isinstance(a, TorchDistribution):
            return bind_dims(a, *names)
        else:
            return a

    with handler({apply: defdata}):
        d = typing.cast(TorchDistribution, evaluate(value))

    if not (isinstance(d, Term) and typeof(d) is TorchDistribution):
        raise NotHandled

    sizes = sizesof(d)
    indices = {k: sizes[k] for k in names}

    pos_args = [_to_positional(a, indices) for a in d.args]
    pos_kwargs = {k: _to_positional(v, indices) for (k, v) in d.kwargs.items()}
    new_d = d.op(*pos_args, **pos_kwargs)

    assert new_d.event_shape == d.event_shape
    return new_d


@functools.cache
def _register_distribution_op(
    dist_constr: type[TorchDistribution],
) -> Operation[Any, TorchDistribution]:
    # introduce a wrapper so that we can control type annotations
    def wrapper(*args, **kwargs) -> TorchDistribution:
        return dist_constr(*args, **kwargs)

    return defop(wrapper, name=dist_constr.__name__)


@defdata.register(pyro.distributions.torch_distribution.TorchDistribution)
@defdata.register(pyro.distributions.torch_distribution.TorchDistributionMixin)
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

    def __init__(
        self, ty: type, op: Operation[Any, TorchDistribution], *args, **kwargs
    ):
        self._op = op
        self._args = args
        self._kwargs = kwargs

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


@evaluate.register(TorchDistribution)
@evaluate.register(TorchDistributionMixin)
def _embed_distribution(dist: TorchDistribution) -> Term[TorchDistribution]:
    raise ValueError(
        f"No embedding provided for distribution of type {type(dist).__name__}."
    )


################################################################################
# Note: Accessing attributes on a distribution actually mutates the
# distribution, so it is unsafe to access attributes in a context that overrides
# torch_getitem and the partial evaluation rules.
################################################################################


@evaluate.register
def _embed_expanded(d: dist.ExpandedDistribution) -> Term[TorchDistribution]:
    with interpreter({}):
        batch_shape_raw = d._batch_shape
        base_dist_raw = d.base_dist

    batch_shape = evaluate(batch_shape_raw)
    base_dist = evaluate(base_dist_raw)
    base_batch_shape = base_dist.batch_shape  # type: ignore
    if batch_shape == base_batch_shape:
        return base_dist

    raise ValueError("Nontrivial ExpandedDistribution not implemented.")


@evaluate.register
def _embed_independent(d: dist.Independent) -> Term[TorchDistribution]:
    with interpreter({}):
        base_dist_raw = d.base_dist
        reinterpreted_batch_ndims_raw = d.reinterpreted_batch_ndims

    base_dist = evaluate(base_dist_raw)
    reinterpreted_batch_ndims = evaluate(reinterpreted_batch_ndims_raw)

    return _register_distribution_op(type(d))(base_dist, reinterpreted_batch_ndims)


@evaluate.register
def _embed_folded(d: dist.FoldedDistribution) -> Term[TorchDistribution]:
    with interpreter({}):
        base_dist_raw = d.base_dist

    base_dist = evaluate(base_dist_raw)

    return _register_distribution_op(type(d))(base_dist)  # type: ignore


@evaluate.register
def _embed_masked(d: dist.MaskedDistribution) -> Term[TorchDistribution]:
    with interpreter({}):
        base_dist_raw = d.base_dist
        mask_raw = d._mask

    base_dist = evaluate(base_dist_raw)
    mask = evaluate(mask_raw)

    return _register_distribution_op(type(d))(base_dist, mask)


@evaluate.register(dist.Cauchy)
@evaluate.register(dist.Gumbel)
@evaluate.register(dist.Laplace)
@evaluate.register(dist.LogNormal)
@evaluate.register(dist.Logistic)
@evaluate.register(dist.LogisticNormal)
@evaluate.register(dist.Normal)
@evaluate.register(dist.StudentT)
def _embed_loc_scale(d: TorchDistribution) -> Term[TorchDistribution]:
    with interpreter({}):
        loc_raw = d.loc
        scale_raw = d.scale

    loc = evaluate(loc_raw)
    scale = evaluate(scale_raw)

    return _register_distribution_op(type(d))(loc, scale)


@evaluate.register(dist.Bernoulli)
@evaluate.register(dist.Categorical)
@evaluate.register(dist.ContinuousBernoulli)
@evaluate.register(dist.Geometric)
@evaluate.register(dist.OneHotCategorical)
@evaluate.register(dist.OneHotCategoricalStraightThrough)
def _embed_probs(d: TorchDistribution) -> Term[TorchDistribution]:
    with interpreter({}):
        probs_raw = d.probs

    probs = evaluate(probs_raw)

    return _register_distribution_op(type(d))(probs)


@evaluate.register(dist.Beta)
@evaluate.register(dist.Kumaraswamy)
def _embed_beta(d: TorchDistribution) -> Term[TorchDistribution]:
    with interpreter({}):
        concentration1_raw = d.concentration1
        concentration0_raw = d.concentration0

    concentration1 = evaluate(concentration1_raw)
    concentration0 = evaluate(concentration0_raw)

    return _register_distribution_op(type(d))(concentration1, concentration0)


@evaluate.register
def _embed_binomial(d: dist.Binomial) -> Term[TorchDistribution]:
    with interpreter({}):
        total_count_raw = d.total_count
        probs_raw = d.probs

    total_count = evaluate(total_count_raw)
    probs = evaluate(probs_raw)

    return _register_distribution_op(dist.Binomial)(total_count, probs)


@evaluate.register
def _embed_chi2(d: dist.Chi2) -> Term[TorchDistribution]:
    with interpreter({}):
        df_raw = d.df

    df = evaluate(df_raw)

    return _register_distribution_op(dist.Chi2)(df)


@evaluate.register
def _embed_dirichlet(d: dist.Dirichlet) -> Term[TorchDistribution]:
    with interpreter({}):
        concentration_raw = d.concentration

    concentration = evaluate(concentration_raw)

    return _register_distribution_op(dist.Dirichlet)(concentration)


@evaluate.register
def _embed_exponential(d: dist.Exponential) -> Term[TorchDistribution]:
    with interpreter({}):
        rate_raw = d.rate

    rate = evaluate(rate_raw)

    return _register_distribution_op(dist.Exponential)(rate)


@evaluate.register
def _embed_fisher_snedecor(d: dist.FisherSnedecor) -> Term[TorchDistribution]:
    with interpreter({}):
        df1_raw = d.df1
        df2_raw = d.df2

    df1 = evaluate(df1_raw)
    df2 = evaluate(df2_raw)

    return _register_distribution_op(dist.FisherSnedecor)(df1, df2)


@evaluate.register
def _embed_gamma(d: dist.Gamma) -> Term[TorchDistribution]:
    with interpreter({}):
        concentration_raw = d.concentration
        rate_raw = d.rate

    concentration = evaluate(concentration_raw)
    rate = evaluate(rate_raw)

    return _register_distribution_op(dist.Gamma)(concentration, rate)


@evaluate.register(dist.HalfCauchy)
@evaluate.register(dist.HalfNormal)
def _embed_half_cauchy(d: TorchDistribution) -> Term[TorchDistribution]:
    with interpreter({}):
        scale_raw = d.scale

    scale = evaluate(scale_raw)

    return _register_distribution_op(type(d))(scale)


@evaluate.register
def _embed_lkj_cholesky(d: dist.LKJCholesky) -> Term[TorchDistribution]:
    with interpreter({}):
        dim_raw = d.dim
        concentration_raw = d.concentration

    dim = evaluate(dim_raw)
    concentration = evaluate(concentration_raw)

    return _register_distribution_op(dist.LKJCholesky)(dim, concentration=concentration)


@evaluate.register
def _embed_multinomial(d: dist.Multinomial) -> Term[TorchDistribution]:
    with interpreter({}):
        total_count_raw = d.total_count
        probs_raw = d.probs

    total_count = evaluate(total_count_raw)
    probs = evaluate(probs_raw)

    return _register_distribution_op(dist.Multinomial)(total_count, probs)


@evaluate.register
def _embed_multivariate_normal(d: dist.MultivariateNormal) -> Term[TorchDistribution]:
    with interpreter({}):
        loc_raw = d.loc
        scale_tril_raw = d.scale_tril

    loc = evaluate(loc_raw)
    scale_tril = evaluate(scale_tril_raw)

    return _register_distribution_op(dist.MultivariateNormal)(
        loc, scale_tril=scale_tril
    )


@evaluate.register
def _embed_negative_binomial(d: dist.NegativeBinomial) -> Term[TorchDistribution]:
    with interpreter({}):
        total_count_raw = d.total_count
        probs_raw = d.probs

    total_count = evaluate(total_count_raw)
    probs = evaluate(probs_raw)

    return _register_distribution_op(dist.NegativeBinomial)(total_count, probs)


@evaluate.register
def _embed_pareto(d: dist.Pareto) -> Term[TorchDistribution]:
    with interpreter({}):
        scale_raw = d.scale
        alpha_raw = d.alpha

    scale = evaluate(scale_raw)
    alpha = evaluate(alpha_raw)

    return _register_distribution_op(dist.Pareto)(scale, alpha)


@evaluate.register
def _embed_poisson(d: dist.Poisson) -> Term[TorchDistribution]:
    with interpreter({}):
        rate_raw = d.rate

    rate = evaluate(rate_raw)

    return _register_distribution_op(dist.Poisson)(rate)


@evaluate.register(dist.RelaxedBernoulli)
@evaluate.register(dist.RelaxedOneHotCategorical)
def _embed_relaxed(d: TorchDistribution) -> Term[TorchDistribution]:
    with interpreter({}):
        temperature_raw = d.temperature
        probs_raw = d.probs

    temperature = evaluate(temperature_raw)
    probs = evaluate(probs_raw)

    return _register_distribution_op(type(d))(temperature, probs)


@evaluate.register
def _embed_uniform(d: dist.Uniform) -> Term[TorchDistribution]:
    with interpreter({}):
        low_raw = d.low
        high_raw = d.high

    low = evaluate(low_raw)
    high = evaluate(high_raw)

    return _register_distribution_op(dist.Uniform)(low, high)


@evaluate.register
def _embed_von_mises(d: dist.VonMises) -> Term[TorchDistribution]:
    with interpreter({}):
        loc_raw = d.loc
        concentration_raw = d.concentration

    loc = evaluate(loc_raw)
    concentration = evaluate(concentration_raw)

    return _register_distribution_op(dist.VonMises)(loc, concentration)


@evaluate.register
def _embed_weibull(d: dist.Weibull) -> Term[TorchDistribution]:
    with interpreter({}):
        scale_raw = d.scale
        concentration_raw = d.concentration

    scale = evaluate(scale_raw)
    concentration = evaluate(concentration_raw)

    return _register_distribution_op(dist.Weibull)(scale, concentration)


@evaluate.register
def _embed_wishart(d: dist.Wishart) -> Term[TorchDistribution]:
    with interpreter({}):
        df_raw = d.df
        scale_tril_raw = d.scale_tril

    df = evaluate(df_raw)
    scale_tril = evaluate(scale_tril_raw)

    return _register_distribution_op(dist.Wishart)(df, scale_tril)


@evaluate.register
def _embed_delta(d: dist.Delta) -> Term[TorchDistribution]:
    with interpreter({}):
        v_raw = d.v
        log_density_raw = d.log_density
        event_dim_raw = d.event_dim

    v = evaluate(v_raw)
    log_density = evaluate(log_density_raw)
    event_dim = evaluate(event_dim_raw)

    return _register_distribution_op(dist.Delta)(
        v, log_density=log_density, event_dim=event_dim
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
