import jax
import matplotlib.pyplot as plt
import pandas as pd

from effectful.handlers.jax import jax_getitem
from effectful.handlers.jax import numpy as jnp
from effectful.handlers.numpyro import Normal
from effectful.handlers.weighted.jax import GradientOptimizationReduce, reals
from effectful.handlers.weighted.jax import interpretation as jax_reduce
from effectful.handlers.weighted.optimization.distribution import (
    interpretation as simplify_normals,
)
from effectful.ops.semantics import evaluate, handler
from effectful.ops.syntax import deffn
from effectful.ops.weighted.reduce import defop
from effectful.ops.weighted.sugar import ArgMin


def get_dataset():
    """Loads the Waffle Divorce dataset."""
    dset_url = (
        "https://raw.githubusercontent.com/rmcelreath/"
        "rethinking/master/data/WaffleDivorce.csv"
    )
    dset = pd.read_csv(dset_url, sep=";")
    normalize = lambda x: (x - x.mean()) / x.std()
    columns = (dset.Divorce, dset.MedianAgeMarriage, dset.Marriage)
    columns = map(normalize, map(jnp.array, columns))
    array = jnp.stack(tuple(columns))
    return array


def main():
    key = jax.random.PRNGKey(0)
    entry = defop(jax.Array, name="entry")
    dset = get_dataset()

    # define the parameters of our model
    param_names = (
        "mu_a",
        "sigma_a",
        "mu_age",
        "sigma_age",
        "mu_marriage",
        "sigma_marriage",
        "sigma_divorce",
    )
    params = tuple(defop(jax.Array, name=x) for x in param_names)
    (mu_a, sigma_a, mu_age, sigma_age, mu_marriage, sigma_marriage, sigma_divorce) = (
        params
    )

    def get_distribution(marriage=None, age=None):
        # now, we can define our distribution
        d_a = Normal(mu_a(), sigma_a())
        mean = d_a.sample(key)
        if marriage is not None:
            d_marriage = Normal(mu_marriage(), sigma_marriage())
            mean += marriage * d_marriage.sample(key)
        if age is not None:
            d_age = Normal(mu_age(), sigma_age())
            mean += age * d_age.sample(key)
        return Normal(mean, sigma_divorce())

    # create a free term for the negative log-likelihood
    d = get_distribution(
        marriage=jax_getitem(entry(), (2,)),
        age=jax_getitem(entry(), (1,)),
    )
    divorce = jax_getitem(entry(), (0,))
    nll = -jnp.mean(d.log_prob(divorce))

    # let the program transforms do their magic
    with handler(simplify_normals), handler({entry: deffn(dset)}):
        nll = evaluate(nll)

    # Next, we look for the MAP using gradient descent
    param_streams = {v: reals() for v in params}
    params_initialization = {v: jnp.array(1.0) for v in params}
    param_terms = tuple(p() for p in params)
    grad_intp = GradientOptimizationReduce(
        learning_rate=0.1, steps=1000, init=params_initialization
    )
    with handler(grad_intp), handler(jax_reduce):
        opt_nll, posterior_values = ArgMin(param_streams, (nll, param_terms))

    # Finally, let's make a plot of our posterior
    posterior = dict(zip(params, posterior_values, strict=False))

    plt.scatter(dset[1], dset[0])
    xx = jnp.linspace(dset[1].min(), dset[1].max(), 100)
    yy = posterior[mu_a] + posterior[mu_age] * xx + posterior[mu_marriage]
    plt.plot(xx, yy, color="orange")

    std = jnp.sqrt(posterior[sigma_divorce] ** 2 + (xx * posterior[sigma_age]) ** 2)
    plt.fill_between(xx, yy - std, yy + std, alpha=0.2)
    plt.xlabel("Age of Marriage (normalized)")
    plt.ylabel("Divorce Rate (normalized)")
    plt.show()


if __name__ == "__main__":
    main()
