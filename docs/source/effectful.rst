Effectful
=========

Operations
----------

.. automodule:: effectful.ops
   :members:
   :undoc-members:

Syntax
^^^^^^

.. automodule:: effectful.ops.syntax
   :members:
   :undoc-members:

   .. autofunction:: effectful.ops.syntax.defdata(value: Term[T]) -> Expr[T]

Semantics
^^^^^^^^^

.. automodule:: effectful.ops.semantics
   :members:
   :undoc-members:

Types
^^^^^

.. automodule:: effectful.ops.types
   :members:
   :undoc-members:


Handlers
--------

.. automodule:: effectful.handlers
   :members:
   :undoc-members:


LLM
^^^

.. automodule:: effectful.handlers.llm
   :members:
   :undoc-members:

Encoding
""""""""

.. automodule:: effectful.handlers.llm.encoding
   :members:
   :undoc-members:

Providers
"""""""""
      
.. automodule:: effectful.handlers.llm.providers
   :members:
   :undoc-members:
   
Jax
^^^

.. automodule:: effectful.handlers.jax
   :members:
   :undoc-members:

   .. autofunction:: effectful.handlers.jax.bind_dims
   .. autofunction:: effectful.handlers.jax.jax_getitem
   .. autofunction:: effectful.handlers.jax.jit
   .. autofunction:: effectful.handlers.jax.sizesof
   .. autofunction:: effectful.handlers.jax.unbind_dims

.. automodule:: effectful.handlers.jax.numpy
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.jax.scipy
   :members:
   :undoc-members:
   

Numpyro
^^^^^^^

.. automodule:: effectful.handlers.numpyro
   :members:
   :undoc-members:
      
Pyro
^^^^

.. automodule:: effectful.handlers.pyro
   :members:
   :undoc-members:

Torch
^^^^^

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

Indexed
^^^^^^^

.. automodule:: effectful.handlers.indexed
   :members:
   :undoc-members:


Internals
---------

.. automodule:: effectful.internals
   :members:
   :undoc-members:

Runtime
^^^^^^^

.. automodule:: effectful.internals.runtime
   :members:
   :undoc-members:

Unification
^^^^^^^^^^^

.. automodule:: effectful.internals.unification
   :members:
   :undoc-members:
