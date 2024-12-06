"""
effectful-minipyro
------------------

This file is a minimal implementation of the Pyro Programming Language,
similar in spirit to the minipyro implementation shipped with Pyro.
It adapts the API of minipyro (method signatures, etc.) to use the
newer Effectful system. Like the original minipyro, this file is
independent of the rest of Pyro, with the exception of the
:mod:`pyro.distributions` module.

This implementation conforms to the :mod:`pyroapi` module's interface, which
allows effectful-minipyro to be run against `pyroapi`'s test suite.
"""

import random
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple, Optional, OrderedDict, TypeVar, Union
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

from effectful.internals.sugar import ObjectInterpretation, implements
from effectful.ops.core import Operation
from effectful.ops.handler import (
    bind_result,
    bind_result_to_method,
    coproduct,
    fwd,
    handler,
)

P = ParamSpec("P")
T = TypeVar("T")

# Poutine has a notion of 'messages', which are dictionaries
# that are passed between handlers (or 'Messengers') in order
# to facilitate coordination and composition using "magic" slots.
# When an effect is triggered, it has a corresponding message which
# is sent up and down the effect stack.

# Effectful does not coordinate between handlers this way, but we
# can use them in order to provide the tracing functionality of minipyro


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


class Seed(NamedTuple):
    """
    All the seeds for the random-number systems generators
    used by minipyro
    """

    torch: Tensor
    python: tuple
    numpy: dict


# The following definitions are all `Operations`, which are functions
# whose meanings are dependent on the context.
# They have no inherent meaning, so their default implementations
# just throw `RuntimeError`s.


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


# These next two pairs of effects of the form `set_X`/`get_X` should
# likely be implemented using `effectful.internals.State`, a built-in
# state handler.
# To keep with the minipyro API, we write them explicitly.


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


# What follows is an `Interpretation`, which is a `Mapping` from
# `Operation`s to meanings for those `Operation`s.
# It is written in the form of an `ObjectInterpretation`, which
# maps each `Operation` to a method on an `object`, allowing the
# `Operation`s to share state.


class Tracer(ObjectInterpretation):
    """
    An `Interpretation` which handles the `sample` and `param` `Operation`s,
    which are immediately forwarded, but their arguments and results are recorded.

    This record is the 'trace', which is stored in the `TRACE` field.
    """

    TRACE: Trace

    def __init__(self):
        self.TRACE = OrderedDict()

    @implements(sample)
    @bind_result_to_method
    def sample(self, ires, var_name: str, dist: Distribution, **kwargs):
        # When we recieve a sample message, we don't know how to
        # handle it ourselves, so we forward it to the next handler:
        res: Tensor = fwd(ires)

        # Once we've seen the result, we record it, along with the argument
        # that caused it
        self.TRACE[var_name] = SampleMsg(
            name=var_name, val=res, dist=dist, obs=kwargs.get("obs")
        )

        return res

    @implements(param)
    @bind_result_to_method
    def param(
        self,
        ires,
        var_name: str,
        initial_value: Optional[Union[Tensor, Callable[[], Tensor]]] = None,
        constraint: Optional[Constraint] = None,
        event_dim: Optional[int] = None,
    ) -> Tensor:
        # Similar to `Tracer.sample`

        res: Tensor = fwd(ires)
        self.TRACE[var_name] = ParamMsg(name=var_name, val=res)

        return res


class Replay(ObjectInterpretation):
    """
    An `Interpretation` which takes a `Trace` as its argument.

    It ensures that any call to `sample` is handled the same way it
    was handled in the trace.
    """

    def __init__(self, trace: Trace):
        self.trace = trace

    @implements(sample)
    @bind_result_to_method
    def sample(self, res, var_name: str, *args, **kwargs):
        if var_name in self.trace:
            return self.trace[var_name].val
        else:
            return fwd(res)


# In minipyro, `Messenger`s can only be used has handlers,
# but we have to make that choice explicit in Effectful.
# These helpers do that.


def replay(trace: Trace):
    return handler(Replay(trace))


@contextmanager
def trace():
    with handler(Tracer()) as t:
        yield t.TRACE


class NativeSeed(ObjectInterpretation):
    """
    This is an interpretation which handles the
    `[get/set]_rng_seed` operations.

    It also provides a base case for `sample`.
    """

    @implements(get_rng_seed)
    def get_rng_seed(self):
        return fwd(
            Seed(
                torch=get_rng_state(),
                python=random.getstate(),
                numpy=np.random.get_state(),
            )
        )

    @implements(set_rng_seed)
    def set_rng_seed(self, seed: Union[int, Seed]):
        if isinstance(seed, int):
            manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        else:
            set_rng_state(seed.torch)
            random.setstate(seed.python)
            np.random.set_state(seed.numpy)
        return fwd(None)

    @implements(sample)
    def sample(self, name: str, dist: Distribution, obs=None, **kwargs):
        assert isinstance(name, str)
        if obs is not None:
            return obs
        elif dist.has_rsample:
            return dist.rsample()
        else:
            return dist.sample()


@contextmanager
def seed(seed: int):
    """
    `contextmanager` for installing/uninstalling a seed value.
    Helpful for fixing an RNG state when calling a model.
    """
    old_seed = get_rng_seed()
    try:
        set_rng_seed(seed)
        yield
    finally:
        set_rng_seed(old_seed)


class NativeParam(ObjectInterpretation):
    """
    Provides a base implementation for the `param` `Operation`.

    Stores the parameter store in the `PARAM_STORE` field, which
    are accessible through the `get_param_store` and `clear_param_store`
    `Operation`s.
    """

    def __init__(self, initial_store=None):
        self.PARAM_STORE = initial_store or {}

    @implements(param)
    def param(
        self,
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
            if name in self.PARAM_STORE:
                unconstrained_value, constraint = self.PARAM_STORE[name]
            else:
                # Initialize with a constrained value.
                assert init_value is not None
                with no_grad():
                    constrained_value = init_value.detach()
                    unconstrained_value = distributions.transform_to(constraint).inv(
                        constrained_value
                    )
                unconstrained_value.requires_grad_()
                self.PARAM_STORE[name] = unconstrained_value, constraint

            # Transform from unconstrained space to constrained space.
            constrained_value = distributions.transform_to(constraint)(
                unconstrained_value
            )
            constrained_value.unconstrained = ref(unconstrained_value)
            return constrained_value

        # Forward our value to any potential upstream transformations
        return fn(initial_value, constraint)

    @implements(get_param_store)
    def get_param_store(self):
        return self.PARAM_STORE

    @implements(clear_param_store)
    def clear_param_store(self):
        self.PARAM_STORE.clear()


class Plate(ObjectInterpretation):
    """
    An `Interpretation` which automatically broadcasts the `sample` `Operation`
    """

    def __init__(self, name: str, size: int, dim: Optional[int]):
        if dim is None:
            raise NotImplementedError(
                "mini-pyro requires the `dim` argument to `plate`"
            )

        self.name = name
        self.size = size
        self.dim = dim

    @implements(sample)
    @bind_result_to_method
    def do_sample(self, res, sampled_name: str, dist: Distribution, **kwargs) -> Tensor:
        batch_shape = list(dist.batch_shape)

        if len(batch_shape) < -self.dim or batch_shape[self.dim] != self.size:
            batch_shape = [1] * (-self.dim - len(batch_shape)) + list(batch_shape)
            batch_shape[self.dim] = self.size
            return sample(sampled_name, dist.expand(Size(batch_shape)))
        else:
            return fwd(res)


# Helper for using `Plate` as a `handler`
def plate(name: str, size: int, dim: Optional[int] = None):
    return handler(Plate(name, size, dim))


# This is the "default runner", which contains the base implementations
# of `sample` and `param`.
# These must be installed (using `handler`) before running a minipyro
# program.
default_runner = coproduct(NativeSeed(), NativeParam())


def block(
    hide_fn: Callable[Concatenate[Operation, object, P], bool] = lambda *_, **__: True,
):
    """
    Block is a helper for masking out a subset of calls to either
    `sample` or `param`.

    Whenever `sample` or `param` are called, `hide_fn` is called with the operation
    and its arguments. If `hide_fn` returns true, then the operation is "blocked",
    and interpreted directly by the `default_runner`. Otherwise, they are handled
    normally.
    """

    @bind_result
    def blocking(res, op: Operation, *args, **kwargs):
        if hide_fn(op, res, *args, **kwargs):
            with handler(default_runner):
                return res if res is not None else op(*args, **kwargs)
        else:
            return fwd(res)

    return handler({sample: partial(blocking, sample), param: partial(blocking, param)})  # type: ignore


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
