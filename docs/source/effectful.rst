Effectful
=========

Operations
----------

.. automodule:: effectful.ops
   :members:
   :undoc-members:

.. automodule:: effectful.ops.syntax
   :members:
   :undoc-members:

   .. autofunction:: effectful.ops.syntax.deffn(body: T, *args: Annotated[Operation, Bound()], **kwargs: Annotated[Operation, Bound()]) -> Callable[..., T])
   .. autofunction:: effectful.ops.syntax.defterm(value: T) -> Expr[T]
   .. autofunction:: effectful.ops.syntax.defdata(value: Term[T]) -> Expr[T]
   .. autofunction:: effectful.ops.semantics.fwd

.. automodule:: effectful.ops.semantics
   :members:
   :undoc-members:

   .. autofunction:: effectful.ops.semantics.apply
      
.. automodule:: effectful.ops.types
   :members:
   :undoc-members:

   
Handlers
--------

.. automodule:: effectful.handlers
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.pyro
   :members:
   :undoc-members:

   .. autofunction:: effectful.handlers.pyro.pyro_sample(name: str, fn: pyro.distributions.torch_distribution.TorchDistributionMixin, *args, obs: Optional[torch.Tensor] = None, obs_mask: Optional[torch.BoolTensor] = None, mask: Optional[torch.BoolTensor] = None, infer: Optional[pyro.poutine.runtime.InferDict] = None, **kwargs) -> torch.Tensor

.. automodule:: effectful.handlers.operator
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.torch
   :members:
   :undoc-members:

   .. autofunction:: effectful.handlers.torch.grad
   .. autofunction:: effectful.handlers.torch.jacfwd
   .. autofunction:: effectful.handlers.torch.jacrev
   .. autofunction:: effectful.handlers.torch.hessian
   .. autofunction:: effectful.handlers.torch.jvp
   .. autofunction:: effectful.handlers.torch.vjp
   .. autofunction:: effectful.handlers.torch.vmap
   .. autofunction:: torch_getitem(x: torch.Tensor, key: Tuple[IndexElement, ...]) -> torch.Tensor                  

.. automodule:: effectful.handlers.indexed
   :members:
   :undoc-members:
      

Internals
---------

.. automodule:: effectful.internals
   :members:
   :undoc-members:

.. automodule:: effectful.internals.base_impl
   :members:
   :undoc-members:

.. automodule:: effectful.internals.runtime
   :members:
   :undoc-members:
   
