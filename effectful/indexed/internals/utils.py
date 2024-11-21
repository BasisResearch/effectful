import functools
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pyro
import torch

from effectful.internals.sugar import gensym, torch_getitem
from effectful.ops.core import Operation, Term


@functools.lru_cache(maxsize=None)
def name_to_sym(name: str) -> Operation[[], int]:
    return gensym(int, name=name)


def lift_tensor(
    tensor: torch.Tensor,
    *,
    name_to_dim: Optional[Mapping[str, int]] = None,
    event_dim: int = 0,
) -> Tuple[torch.Tensor, List[Operation]]:
    """Lift a tensor to an indexed tensor using the mapping in name_to_dim.

    Parameters:
    - tensor (torch.Tensor): A tensor.
    - name_to_dim: A dictionary mapping names to dimensions. If not provided, the plates returned by get_index_plates()
    are used.

    """
    if name_to_dim is None:
        name_to_dim = _get_index_plates_to_name_to_dim()

    index_expr: List[Any] = [slice(None)] * len(tensor.shape)
    for name, dim_offset in name_to_dim.items():
        dim = dim_offset - event_dim
        # ensure that lifted tensors use the same free variables for the same name
        index_expr[dim] = name_to_sym(name)()

    vars_: List[Operation] = []
    for v in index_expr:
        if isinstance(v, Term):
            vars_.append(v.op)

    result = torch_getitem(tensor, tuple(index_expr))

    return result, vars_


@pyro.poutine.runtime.effectful(type="get_index_plates")
def get_index_plates() -> Dict[str, pyro.poutine.indep_messenger.CondIndepStackFrame]:
    return {}


def _get_index_plates_to_name_to_dim() -> Mapping[str, int]:
    name_to_dim = {}
    for name, f in get_index_plates().items():
        assert f.dim is not None
        name_to_dim[name] = f.dim
    return name_to_dim
