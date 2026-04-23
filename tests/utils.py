import functools

from effectful.handlers.weighted.jax import DenseTensorReduce
from effectful.handlers.weighted.optimization import (
    ReduceDistributeTerm,
    ReduceEliminateDterm,
    ReduceFactorization,
    ReduceFusion,
    ReducePropagateUnusedStreams,
    ReduceReorderReduction,
    ReduceSplit,
)
from effectful.ops.semantics import coproduct
from effectful.ops.types import Interpretation

JAX_INTP = DenseTensorReduce()
BASELINE_INTP: Interpretation = {}

REORDER_TRANS = ReduceReorderReduction()
SPLIT_TRANS = ReduceSplit()
PROPAGATE_TRANS = ReducePropagateUnusedStreams()
FACTORIZE_TRANS = ReduceFactorization()
FUSE_TRANS = ReduceFusion()
DISTRIBUTE_TERM_TRANS = ReduceDistributeTerm()
D_ELIMINATE_TRANS = ReduceEliminateDterm()

REDUCE_TRANSFORMS = (
    REORDER_TRANS,
    SPLIT_TRANS,
    PROPAGATE_TRANS,
    FACTORIZE_TRANS,
    FUSE_TRANS,
    DISTRIBUTE_TERM_TRANS,
    D_ELIMINATE_TRANS,
)

DEFAULT_TRANS = functools.reduce(
    coproduct,  # type: ignore
    [
        DISTRIBUTE_TERM_TRANS,
        REORDER_TRANS,
        FACTORIZE_TRANS,
        FUSE_TRANS,
        SPLIT_TRANS,
        PROPAGATE_TRANS,
        D_ELIMINATE_TRANS,
    ],
)


JAX_NO_D_INTP = coproduct(JAX_INTP, D_ELIMINATE_TRANS)
BASELINE_NO_D_INTP = coproduct(BASELINE_INTP, D_ELIMINATE_TRANS)

JAX_DEFAULT_INTP = coproduct(JAX_INTP, DEFAULT_TRANS)
BASELINE_DEFAULT_INTP = coproduct(BASELINE_INTP, DEFAULT_TRANS)

DEFAULT_TEST_REDUCE_INTP = {
    "jax": JAX_INTP,
    "jax_no_d": JAX_NO_D_INTP,
    "baseline_no_d": BASELINE_NO_D_INTP,
    "baseline_default": BASELINE_DEFAULT_INTP,
    "jax_default": JAX_DEFAULT_INTP,
}
