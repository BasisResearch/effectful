import random
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, OrderedDict, Tuple, TypeVar, Union
from weakref import ref

import numpy as np
import pyroapi
import torch.distributions as distributions
import torch.optim
from pyro.distributions import validation_enabled
from torch import (
    Size,
    Tensor,
    get_rng_state,
    manual_seed,
    no_grad,
    set_rng_state,
    zeros_like,
)
from torch.distributions import Distribution
from torch.distributions.constraints import Constraint
from typing_extensions import Concatenate, ParamSpec

from effectful.internals.prompts import result
from effectful.ops.core import Operation
from effectful.ops.handler import coproduct, fwd, handler
from effectful.ops.runner import product, reflect


@dataclass
class SampleMsg:
    name: str
    val: Tensor
    dist: Distribution
    obs: Optional[Tensor]


@dataclass
class ParamMsg:
    name: str
    val: Tensor


Message = Union[ParamMsg, SampleMsg]
Trace = OrderedDict[str, Message]

P = ParamSpec("P")
T = TypeVar("T")

Seed = Tuple[Tensor, tuple, dict]


@Operation
def sample(name: str, dist: Distribution, obs: Optional[Tensor] = None) -> Tensor:
    raise RuntimeError("No default implementation of sample")


@Operation
def param(
    var_name: str,
    initial_value: Optional[Union[Tensor, Callable[[], Tensor]]] = None,
    constraint: Optional[Constraint] = None,
    event_dim: Optional[int] = None,
) -> Tensor:
    raise RuntimeError("No default implementation of param")


@Operation
def clear_param_store() -> None:
    raise RuntimeError("No default implementation of clear_param_store")


@Operation
def get_param_store() -> dict[str, Tensor]:
    raise RuntimeError("No default implementation of get_param_store")


@Operation
def get_rng_seed() -> Seed:
    raise RuntimeError("No default implementation of get_rng_seed")


@Operation
def set_rng_seed(seed: Union[int, Seed]):
    raise RuntimeError("No default implementation of get_rng_seed")


@contextmanager
def trace():
    from collections import OrderedDict

    TRACE = OrderedDict()

    def do_sample(var_name: str, dist: Distribution, **kwargs) -> Tensor:
        res: Tensor = fwd(result.get())
        TRACE[var_name] = SampleMsg(
            name=var_name, val=res, dist=dist, obs=kwargs.get("obs")
        )
        return res

    def do_param(
        var_name: str,
        initial_value: Optional[Union[Tensor, Callable[[], Tensor]]] = None,
        constraint: Optional[Constraint] = None,
        event_dim: Optional[int] = None,
    ) -> Tensor:
        res: Tensor = fwd(result.get())
        TRACE[var_name] = ParamMsg(name=var_name, val=res)

        return res

    with handler({sample: do_sample, param: do_param}):
        yield TRACE


def replay(trace: Trace):
    def do_sample(var_name: str, dist: Distribution, **kwargs) -> Tensor:
        if var_name in trace:
            return trace[var_name].val
        else:
            return fwd(result.get())

    return handler(
        {
            sample: do_sample,
        }
    )


def seed_impl():
    def do_get_rng_seed() -> Union[int, Seed]:
        return fwd((get_rng_state(), random.getstate(), np.random.get_state()))

    def do_set_rng_seed(seed: Union[int, Seed]):
        if isinstance(seed, int):
            manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        else:
            set_rng_state(seed[0])
            random.setstate(seed[1])
            np.random.set_state(seed[2])
        return fwd(None)

    def do_sample(name: str, dist: Distribution, obs=None, **kwargs):
        assert isinstance(name, str)
        if obs is not None:
            return obs
        elif dist.has_rsample:
            return dist.rsample()
        else:
            return dist.sample()

    return {
        get_rng_seed: do_get_rng_seed,
        set_rng_seed: do_set_rng_seed,
        sample: do_sample,
    }


@contextmanager
def seed(seed: int):
    old_seed = get_rng_seed()
    try:
        set_rng_seed(seed)
        yield
    finally:
        set_rng_seed(old_seed)


def param_impl():
    PARAM_STORE = {}

    def do_param(
        name: str,
        initial_value: Union[Tensor, None, Callable[[], Tensor]] = None,
        constraint: Constraint = distributions.constraints.real,
        event_dim: Optional[int] = None,
    ) -> Tensor:
        if event_dim is not None:
            raise NotImplementedError(
                "minipyro.plate does not support the event_dim arg"
            )

        def fn(init_value, constraint):
            if name in PARAM_STORE:
                unconstrained_value, constraint = PARAM_STORE[name]
            else:
                # Initialize with a constrained value.
                assert init_value is not None
                with no_grad():
                    constrained_value = init_value.detach()
                    unconstrained_value = distributions.transform_to(constraint).inv(
                        constrained_value
                    )
                unconstrained_value.requires_grad_()
                PARAM_STORE[name] = unconstrained_value, constraint

            # Transform from unconstrained space to constrained space.
            constrained_value = distributions.transform_to(constraint)(
                unconstrained_value
            )
            constrained_value.unconstrained = ref(unconstrained_value)
            return constrained_value

        return fwd(fn(initial_value, constraint))

    def do_get_param_store():
        return fwd(PARAM_STORE)

    def do_clear_param_store():
        PARAM_STORE.clear()

    return {
        param: do_param,
        get_param_store: do_get_param_store,
        clear_param_store: do_clear_param_store,
    }


def plate(name: str, size: int, dim: Optional[int] = None):
    if dim is None:
        raise NotImplementedError(
            "mini-pyro doesn't implement the `dim` argument to `plate`"
        )

    def do_sample(sampled_name: str, dist: Distribution, **kwargs) -> Tensor:
        batch_shape = list(dist.batch_shape)

        if len(batch_shape) < -dim or batch_shape[dim] != size:
            batch_shape = [1] * (-dim - len(batch_shape)) + list(batch_shape)
            batch_shape[dim] = size
            return sample(sampled_name, dist.expand(Size(batch_shape)))
        else:
            return fwd(result.get())

    return handler({sample: do_sample})


base_runner = coproduct(seed_impl(), param_impl())
default_runner = product(
    base_runner, {k: lambda *_, **__: reflect(result.get()) for k in base_runner}
)


def block(
    hide_fn: Callable[Concatenate[Operation, object, P], bool] = lambda *_, **__: True
):
    def blocking(fn: Operation, *args, **kwargs):
        res = result.get()
        if hide_fn(fn, res, *args, **kwargs):
            return reflect(res)
        else:
            return fwd(res)

    return handler(
        product(
            default_runner,
            {
                sample: partial(blocking, sample),
                param: partial(blocking, param),
            },
        )
    )


# This is a thin wrapper around the `torch.optim.Adam` class that
# dynamically generates optimizers for dynamically generated parameters.
# See http://docs.pyro.ai/en/stable/optimization.html
class Adam:
    def __init__(self, optim_args):
        self.optim_args = optim_args
        # Each parameter will get its own optimizer, which we keep track
        # of using this dictionary keyed on parameters.
        self.optim_objs = {}

    def __call__(self, params):
        for param in params:
            # If we've seen this parameter before, use the previously
            # constructed optimizer.
            if param in self.optim_objs:
                optimizer = self.optim_objs[param]
            # If we've never seen this parameter before, construct
            # an Adam optimizer and keep track of it.
            else:
                optimizer = torch.optim.Adam([param], **self.optim_args)
                self.optim_objs[param] = optimizer
            # Take a gradient step for the parameter param.
            optimizer.step()


# This is a unified interface for stochastic variational inference in Pyro.
# The actual construction of the loss is taken care of by `loss`.
# See http://docs.pyro.ai/en/stable/inference_algos.html
class SVI:
    def __init__(self, model, guide, optim, loss):
        self.model = model
        self.guide = guide
        self.optim = optim
        self.loss = loss

    # This method handles running the model and guide, constructing the loss
    # function, and taking a gradient step.
    def step(self, *args, **kwargs):
        # This wraps both the call to `model` and `guide` in a `trace` so that
        # we can record all the parameters that are encountered. Note that
        # further tracing occurs inside of `loss`.
        with trace() as param_capture:
            # We use block here to allow tracing to record parameters only.
            with block(hide_fn=lambda op, *_, **__: op == sample):
                loss = self.loss(self.model, self.guide, *args, **kwargs)
        # Differentiate the loss.
        loss.backward()
        # Grab all the parameters from the trace.
        params = [site.val.unconstrained() for site in param_capture.values()]
        # Take a step w.r.t. each parameter in params.
        self.optim(params)
        # Zero out the gradients so that they don't accumulate.
        for p in params:
            p.grad = zeros_like(p)
        return loss.item()


# This is a basic implementation of the Evidence Lower Bound, which is the
# fundamental objective in Variational Inference.
# See http://pyro.ai/examples/svi_part_i.html for details.
# This implementation has various limitations (for example it only supports
# random variables with reparameterized samplers), but all the ELBO
# implementations in Pyro share the same basic logic.
def elbo(model, guide, *args, **kwargs):
    # Run the guide with the arguments passed to SVI.step() and trace the execution,
    # i.e. record all the calls to Pyro primitives like sample() and param().
    with trace() as guide_trace:
        guide(*args, **kwargs)
    # Now run the model with the same arguments and trace the execution. Because
    # model is being run with replay, whenever we encounter a sample site in the
    # model, instead of sampling from the corresponding distribution in the model,
    # we instead reuse the corresponding sample from the guide. In probabilistic
    # terms, this means our loss is constructed as an expectation w.r.t. the joint
    # distribution defined by the guide.
    with trace() as model_trace:
        with replay(guide_trace):
            model(*args, **kwargs)
    # We will accumulate the various terms of the ELBO in `elbo`.
    elbo = 0.0
    # Loop over all the sample sites in the model and add the corresponding
    # log p(z) term to the ELBO. Note that this will also include any observed
    # data, i.e. sample sites with the keyword `obs=...`.
    for site in model_trace.values():
        if isinstance(site, SampleMsg):
            elbo = elbo + site.dist.log_prob(site.val).sum()
    # Loop over all the sample sites in the guide and add the corresponding
    # -log q(z) term to the ELBO.
    for site in guide_trace.values():
        if isinstance(site, SampleMsg):
            elbo = elbo - site.dist.log_prob(site.val).sum()
    # Return (-elbo) since by convention we do gradient descent on a loss and
    # the ELBO is a lower bound that needs to be maximized.
    return -elbo


# This is a wrapper for compatibility with full Pyro.
def Trace_ELBO(**kwargs):
    return elbo


# This is a Jit wrapper around elbo() that (1) delays tracing until the first
# invocation, and (2) registers pyro.param() statements with torch.jit.trace.
# This version does not support variable number of args or non-tensor kwargs.
class JitTrace_ELBO:
    def __init__(self, **kwargs):
        self.ignore_jit_warnings = kwargs.pop("ignore_jit_warnings", False)
        self._compiled = None
        self._param_trace = None

    def __call__(self, model, guide, *args):
        # On first call, initialize params and save their names.
        if self._param_trace is None:
            with block(), trace() as tr, block(
                hide_fn=lambda op, *_, **__: op != param
            ):
                elbo(model, guide, *args)
            self._param_trace = tr

        # Augment args with reads from the global param store.
        unconstrained_params = tuple(
            param(name).unconstrained() for name in self._param_trace
        )
        params_and_args = unconstrained_params + args

        # On first call, create a compiled elbo.
        if self._compiled is None:

            def compiled(*params_and_args):
                unconstrained_params = params_and_args[: len(self._param_trace)]
                args = params_and_args[len(self._param_trace) :]
                for name, unconstrained_param in zip(
                    self._param_trace, unconstrained_params
                ):
                    constrained_param = param(name)  # assume param has been initialized
                    assert constrained_param.unconstrained() is unconstrained_param
                    self._param_trace[name].value = constrained_param
                with replay(self._param_trace):
                    return elbo(model, guide, *args)

            with validation_enabled(False), warnings.catch_warnings():
                if self.ignore_jit_warnings:
                    warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                self._compiled = torch.jit.trace(
                    compiled, params_and_args, check_trace=False
                )

        return self._compiled(*params_and_args)


pyroapi.register_backend(
    "effectful-minipyro",
    {
        "infer": "effectful.handlers.minipyro",
        "optim": "effectful.handlers.minipyro",
        "pyro": "effectful.handlers.minipyro",
    },
)
