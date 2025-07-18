import pytest
from effectful.ops.semantics import coproduct
from effectful.ops.syntax import ObjectInterpretation

from weighted.handlers.jax import DenseTensorFold
from weighted.handlers.optimization import (
    FoldEliminateDterm,
    FoldFactorization,
    FoldFusion,
    FoldReorderReduction,
    FoldSplit,
)
from weighted.ops.fold import BaselineFold

FOLD_INTP: dict[str, ObjectInterpretation] = {
    "baseline_intp": BaselineFold(),
    "jax_intp": DenseTensorFold(),
    "reorder_fold": FoldReorderReduction(),
    "d_elim_fold": FoldEliminateDterm(),
    "factorize_fold": FoldFactorization(),
    "fuse_fold": FoldFusion(),
    "split_fold": FoldSplit(),
}

FOLD_INTP["baseline_d_intp"] = coproduct(  # type: ignore
    FOLD_INTP["baseline_intp"], FOLD_INTP["d_elim_fold"]
)
FOLD_INTP["jax_d_intp"] = coproduct(FOLD_INTP["jax_intp"], FOLD_INTP["d_elim_fold"])  # type: ignore

FOLD_INTP["baseline_reorder_intp"] = coproduct(  # type: ignore
    FOLD_INTP["baseline_d_intp"], FOLD_INTP["reorder_fold"]
)
FOLD_INTP["jax_reorder_intp"] = coproduct(  # type: ignore
    FOLD_INTP["jax_d_intp"], FOLD_INTP["reorder_fold"]
)

FOLD_INTP["jax_factorize_intp"] = coproduct(  # type: ignore
    FOLD_INTP["jax_reorder_intp"], FOLD_INTP["factorize_fold"]
)
FOLD_INTP["baseline_factorize_intp"] = coproduct(  # type: ignore
    FOLD_INTP["baseline_reorder_intp"], FOLD_INTP["factorize_fold"]
)

FOLD_INTP["jax_split_intp"] = coproduct(  # type: ignore
    FOLD_INTP["jax_factorize_intp"], FOLD_INTP["split_fold"]
)
FOLD_INTP["baseline_split_intp"] = coproduct(  # type: ignore
    FOLD_INTP["baseline_factorize_intp"], FOLD_INTP["split_fold"]
)


def get_fold_params(*names: str):
    return [pytest.param(FOLD_INTP[name], id=name) for name in names]
