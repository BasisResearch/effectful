import argparse
from functools import reduce
from operator import mul

import effectful.handlers.jax.numpy as jnp
from effectful.handlers.jax import jax, jax_getitem
from effectful.ops.semantics import coproduct, evaluate, handler
from effectful.ops.syntax import defop
from jax import random

from weighted.handlers.jax import D, DenseTensorFold
from weighted.handlers.optimization import FoldEliminateDterm, FoldReorderReduction
from weighted.ops.sugar import Sum


class HMM:
    def __init__(self, emission_size: int, hidden_size: int):
        self.emission_size = emission_size
        self.hidden_size = hidden_size
        self.transitions = defop(jax.Array, name="transition_matrix")
        self.emissions = defop(jax.Array, name="emission_matrix")
        self.initial_state = defop(jax.Array, name="initial_state")

    def as_d(self, observed_emissions: jax.Array):
        time_steps = observed_emissions.size + 1
        hidden_vars = {t: defop(jax.Array, name=f"h{t}") for t in range(time_steps)}
        emission_vars = {t: defop(jax.Array, name=f"e{t}") for t in range(time_steps)}
        emission_streams = {
            op: jnp.arange(self.emission_size) for op in emission_vars.values()
        }
        hidden_streams = {op: jnp.arange(self.hidden_size) for op in hidden_vars.values()}
        streams = emission_streams | hidden_streams
        hidden_vars = {k: v() for k, v in hidden_vars.items()}  # type: ignore
        emission_vars = {k: v() for k, v in emission_vars.items()}  # type: ignore

        factors = [jax_getitem(self.initial_state(), (hidden_vars[0],))]  # type: ignore
        for t in range(time_steps):
            factor = jax_getitem(self.emissions(), (hidden_vars[t], emission_vars[t]))  # type: ignore
            factors.append(factor)
            if t > 0:
                factor = jax_getitem(
                    self.transitions(),  # type: ignore
                    (hidden_vars[t - 1], hidden_vars[t]),
                )
                factors.append(factor)

        for t, observation in enumerate(observed_emissions):
            v = jax.nn.one_hot(observation, self.emission_size)
            factors.append(jax_getitem(v, (emission_vars[t],)))

        body = reduce(mul, factors)
        d_indices = (emission_vars[time_steps - 1],)
        return Sum(streams, D((d_indices, body)))  # type: ignore

    def get_interpretation(self, t_matrix, e_matrix, i_matrix):
        # make sure everything is nice and normalized
        t_matrix /= t_matrix.sum(axis=-1, keepdims=True)
        e_matrix /= e_matrix.sum(axis=-1, keepdims=True)
        i_matrix /= i_matrix.sum(axis=-1, keepdims=True)

        return {
            self.transitions: lambda: t_matrix,
            self.emissions: lambda: e_matrix,
            self.initial_state: lambda: i_matrix,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=4)
    parser.add_argument("--emission_size", type=int, default=3)
    parser.add_argument("--nb_time_steps", type=int, default=2)
    args = parser.parse_args()

    # first, let's create some tensors to parameterize the HMM
    key = random.PRNGKey(42)
    key1, key2, key3, key4 = random.split(key, 4)
    matrices = [
        random.uniform(key1, (args.hidden_size, args.hidden_size)),
        random.uniform(key2, (args.hidden_size, args.emission_size)),
        random.uniform(key3, (args.hidden_size,)),
    ]

    # the dummy observation we want to do inference on
    evidence = random.randint(key4, (args.nb_time_steps,), 0, args.emission_size)

    # second, we construct an abstract HMM
    model = HMM(args.emission_size, args.hidden_size)
    hmm_term = model.as_d(evidence)

    # now we do inference by interpreting the hmm parameters and folding
    hmm_interpretation = model.get_interpretation(*matrices)

    fold_interpretation = [
        DenseTensorFold(),
        FoldReorderReduction(),
        FoldEliminateDterm(),
    ]
    fold_interpretation = reduce(coproduct, fold_interpretation)  # type: ignore

    # note: do not evaluate the parameters of the HMM before folding,
    # or jax will try to construct the full joint distribution like a mad man

    with handler(fold_interpretation), handler(hmm_interpretation):  # type: ignore
        result: jax.Array = evaluate(hmm_term)  # type: ignore

    print("Next token prediction conditional on", evidence, "is", jnp.argmax(result))
