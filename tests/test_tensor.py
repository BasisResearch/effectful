from effectful.internals.sugar import gensym, TensorNeutral
from effectful.ops.core import embed, unembed, typeof
import torch

def test_tensor_neutral():
    i = gensym(torch.Tensor)()
    x = gensym(torch.Tensor)()

    y = torch.add(x[i], torch.index_select(torch.ones(1), 0, i))
    assert issubclass(typeof(y), torch.Tensor)

    z = torch.add(x[i], torch.ones(1))
    assert issubclass(typeof(z), torch.Tensor)