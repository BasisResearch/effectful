import argparse
import time

from jax import random
from jax.numpy.linalg import norm
from weighted.handlers.jax import DenseTensorReduce
from weighted.handlers.optimization import ReduceDistributeTerm
from weighted.handlers.optimization.cartesian_product import SplitCartesianProductReduce
from weighted.handlers.optimization.jax import StackIndex
from weighted.ops.monoid import mul
from weighted.ops.reduce import BaselineReduce
from weighted.ops.sugar import CartesianProd, Prod, Sum

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import jax
from effectful.ops.semantics import evaluate, handler
from effectful.ops.syntax import deffn, defop


def construct_hmm():
    """
    Returns an open term for a simple HMM model.
    where the observed variables (x) only depend on the latent
    variables (z), and the latents are conditionally independent
    given direct ancestor.

     z[t-1] --> z[t] --> z[t+1]
        |        |         |
        V        V         V
     x[t-1]     x[t]     x[t+1]
    """
    t = defop(jax.Array, name="t")()  # time index
    z = defop(jax.Array, name="z")()  # latent state index
    x = defop(jax.Array, name="x")()  # emission index

    t_stream = {t.op: jnp.arange(args.nb_time_steps)}
    tm1_stream = {t.op: jnp.arange(args.nb_time_steps - 1)}
    z_stream = {z.op: CartesianProd(t_stream, jnp.arange(args.latent_size))}
    x_stream = {x.op: CartesianProd(t_stream, jnp.arange(args.emission_size))}

    transition_factor = defop(jax.Array, name="transition")()
    emission_factor = defop(jax.Array, name="emission")()
    initial_state = defop(jax.Array, name="initial")()
    observation = defop(jax.Array, name="observation")()

    with handler(BaselineReduce()):  # immediately unroll factors
        hmm_factor = (
            initial_state[z[0]]
            * Prod(t_stream, emission_factor[z[t], x[t]] * observation[t, x[t]])
            * Prod(tm1_stream, transition_factor[z[t], z[t + 1]])
        )

    hmm = Sum(z_stream | x_stream, hmm_factor)

    param_ops = (transition_factor, emission_factor, initial_state, observation)
    param_ops = (x.op for x in param_ops)
    return hmm, param_ops


def create_hmm_params(transition_op, emission_op, initial_state_op, observation_op):
    key = random.PRNGKey(0)
    key1, key2, key3, key4 = random.split(key, 4)

    params = {
        transition_op: random.uniform(key1, (args.latent_size, args.latent_size)),
        emission_op: random.uniform(key2, (args.latent_size, args.emission_size)),
        initial_state_op: random.uniform(key3, (args.latent_size,)),
        observation_op: random.randint(
            key4, (args.nb_time_steps,), 0, args.emission_size
        ),
    }
    for op in (transition_op, emission_op, initial_state_op):
        params[op] /= norm(params[op], axis=-1, keepdims=True)
    params[observation_op] = jax.nn.one_hot(params[observation_op], args.emission_size)

    return {k: deffn(v) for k, v in params.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--emission_size", type=int, default=3)
    parser.add_argument("--nb_time_steps", type=int, default=4)
    args = parser.parse_args()

    # first, we construct an abstract HMM, unroll it...
    with handler(SplitCartesianProductReduce()), handler({mul: jnp.multiply}):
        model, param_ops = construct_hmm()
    # ... and optimize the reduce ordering
    with handler(ReduceDistributeTerm()), handler(StackIndex()):
        model = evaluate(model)

    # finally, we create arrays for the open HMM params
    param_intp = create_hmm_params(*param_ops)
    # .. and perform inference by computing the reduce
    t1 = time.perf_counter()
    with handler(param_intp), handler(DenseTensorReduce()):
        result = evaluate(model)
    print(result, f"(done in {time.perf_counter() - t1:.2f}s)")
