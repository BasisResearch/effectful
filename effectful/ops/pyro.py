import functools
import random
from contextlib import contextmanager
from functools import partial
from weakref import ref
from typing import Callable, Union, Optional, TypeVar, OrderedDict, TypedDict, Tuple

import numpy as np
import pyroapi
from torch import Tensor, Size, get_rng_state, manual_seed, set_rng_state, no_grad, zeros_like, tensor
import torch.distributions as distributions
from torch.distributions import Distribution
from torch.distributions.constraints import Constraint
import torch.optim
from typing_extensions import ParamSpec, Concatenate

from effectful.internals.prompts import bind_result
from effectful.ops import core
from effectful.ops.runner import runner, reflect
from effectful.ops.core import Operation, define
from effectful.ops.handler import handler, fwd, install


def empty_operation(name):
    def empty_op(*_, **__):
        raise RuntimeError(f"{name} has no default implementation")

    empty_op.__name__ = name

    return define(Operation)(empty_op)


class SampleMsg(TypedDict):
    name: str
    val: Tensor
    dist: Distribution
    obs: Optional[Tensor]


class ParamMsg(TypedDict):
    name: str
    val: Tensor


Message = Union[ParamMsg, SampleMsg]
Trace = OrderedDict[str, Message]

sample = empty_operation("sample")
param = empty_operation("param")
get_param_store = empty_operation("get_param_store")
get_rng_seed = empty_operation("get_rng_seed")
set_rng_seed = empty_operation("set_rng_seed")


@contextmanager
def trace():
    from collections import OrderedDict
    the_trace = OrderedDict()

    def do_sample(result: Optional[Tensor],
                  var_name: str,
                  dist: Distribution,
                  **kwargs) -> Tensor:
        result = fwd(result)
        the_trace[var_name] = SampleMsg(name=var_name, val=result, dist=dist, obs=kwargs.get("obs"))
        return result

    def do_param(result: Optional[Tensor],
                 var_name: str,
                 initial_value: Union[Tensor, Callable[[], Tensor]],
                 constraint: Optional[Constraint] = None,
                 event_dim: Optional[int] = None) -> Tensor:
        result = fwd(result)
        the_trace[var_name] = ParamMsg(name=var_name, val=result)

        return result

    with handler({sample: do_sample, param: do_param}):
        yield the_trace


def replay(trace: Trace):
    return handler({
        sample: lambda _, v, d: fwd(trace[v]["val"])
    })


P = ParamSpec("P")
T = TypeVar("T")


def block(hide_fn: Callable[Concatenate[Operation[P, T], Optional[T], P], bool]):
    def blocking(fn: Operation[P, T], result: Optional[T], *args: P.args, **kwargs: P.kwargs):
        if hide_fn(fn, result, *args, **kwargs):
            return reflect(result)
        else:
            return fwd(result)

    return handler({
        sample: partial(blocking, fn=sample),
        param: partial(block, fn=param)
    })


Seed = Tuple[Tensor, tuple, dict]


def seed_impl():
    def do_get_rng_seed(_: None) -> Union[int, Seed]:
        return fwd((get_rng_state(), random.getstate(), np.random.get_state()))

    def do_set_rng_seed(_: None, seed: Union[int, Seed]):
        if isinstance(seed, int):
            manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        else:
            set_rng_state(seed[0])
            random.seed(seed[1])
            np.random.set_state(seed[2])
        return fwd(None)

    def do_sample(name: str, dist: Distribution, **kwargs):
        assert isinstance(name, str)
        return fwd(dist.rsample() if dist.has_rsample else dist.sample())

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


install(seed_impl())


def param_impl():
    the_store = {}

    def do_param(res: None,
                 name: str,
                 initial_value: Union[Tensor, None, Callable[[], Tensor]] = None,
                 constraint: Constraint = distributions.constraints.real,
                 event_dim: Optional[int] = None) -> Tensor:
        if event_dim is not None:
            raise NotImplementedError("minipyro.plate does not support the event_dim arg")

        def fn(init_value, constraint):
            if name in the_store:
                unconstrained_value, constraint = the_store[name]
            else:
                # Initialize with a constrained value.
                assert init_value is not None
                with no_grad():
                    constrained_value = init_value.detach()
                    unconstrained_value = distributions.transform_to(constraint).inv(
                        constrained_value
                    )
                unconstrained_value.requires_grad_()
                the_store[name] = unconstrained_value, constraint

            # Transform from unconstrained space to constrained space.
            constrained_value = distributions.transform_to(constraint)(
                unconstrained_value
            )
            constrained_value.unconstrained = ref(unconstrained_value)
            return constrained_value

        return fwd(fn(initial_value, constraint))

    def do_get_param_store(res: None):
        return fwd(the_store)

    return {param: bind_result(do_param), get_param_store: bind_result(do_get_param_store)}


install(param_impl())


def plate(name: str, size: int, dim: Optional[int] = None):
    if dim is None:
        raise NotImplementedError("mini-pyro doesn't implement the `dim` argument to `plate`")

    def do_sample(result: Tensor, sampled_name: str, dist: Distribution, **kwargs) -> Tensor:
        batch_shape = dist.batch_shape

        if len(batch_shape) < -dim or batch_shape[dim] != size:
            batch_shape = [1] * (-dim - len(batch_shape)) + list(batch_shape)
            batch_shape[dim] = size
            return sample(sampled_name, dist.expand(Size(batch_shape)))
        else:
            return fwd(result)

    return handler({
        sample: bind_result(do_sample)
    })


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
            with block(hide_fn=lambda msg: isinstance(msg, SampleMsg)):
                loss = self.loss(self.model, self.guide, *args, **kwargs)
        # Differentiate the loss.
        loss.backward()
        # Grab all the parameters from the trace.
        params = [site["value"].unconstrained() for site in param_capture.values()]
        # Take a step w.r.t. each parameter in params.
        self.optim(params)
        # Zero out the gradients so that they don't accumulate.
        for p in params:
            p.grad = zeros_like(p)
        return loss.item()


pyroapi.register_backend("effectful-minipyro", {
    # "optim": "effectful.ops.pyro",
    "pyro": "effectful.ops.pyro"
})
