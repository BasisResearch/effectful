from functools import reduce

from effectful.ops.semantics import coproduct

from weighted.handlers.jax import DenseTensorFold
from weighted.handlers.optimization import (
    FoldDistributeTerm,
    FoldEliminateDterm,
    FoldFactorization,
    FoldFusion,
    FoldPropagateUnusedStreams,
    FoldReorderReduction,
    FoldSplit,
)
from weighted.ops.fold import BaselineFold

JAX_INTP = DenseTensorFold()
BASELINE_INTP = BaselineFold()

REORDER_TRANS = FoldReorderReduction()
SPLIT_TRANS = FoldSplit()
PROPAGATE_TRANS = FoldPropagateUnusedStreams()
FACTORIZE_TRANS = FoldFactorization()
FUSE_TRANS = FoldFusion()
DISTRIBUTE_TERM_TRANS = FoldDistributeTerm()
D_ELIMINATE_TRANS = FoldEliminateDterm()

FOLD_TRANSFORMS = (
    REORDER_TRANS,
    SPLIT_TRANS,
    PROPAGATE_TRANS,
    FACTORIZE_TRANS,
    FUSE_TRANS,
    DISTRIBUTE_TERM_TRANS,
    D_ELIMINATE_TRANS,
)

DEFAULT_TRANS = reduce(
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

DEFAULT_TEST_FOLD_INTP = {
    "jax": JAX_INTP,
    "jax_no_d": JAX_NO_D_INTP,
    "baseline_no_d": BASELINE_NO_D_INTP,
    "baseline_default": BASELINE_DEFAULT_INTP,
    "jax_default": JAX_DEFAULT_INTP,
}
