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

Types
"""""

.. automodule:: effectful.handlers.llm.types
   :members:
   :undoc-members:

Harness
"""""""

.. automodule:: effectful.handlers.llm.harness
   :members:
   :undoc-members:

Command-line launcher
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: effectful.handlers.llm.harness.__main__
   :members:
   :undoc-members:

Hooks
~~~~~

.. automodule:: effectful.handlers.llm.harness.hooks
   :members:
   :undoc-members:

Serialization
~~~~~~~~~~~~~

.. automodule:: effectful.handlers.llm.harness.serialization
   :members:
   :undoc-members:

Provision
~~~~~~~~~

.. automodule:: effectful.handlers.llm.harness.provision
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.provision.litellm
   :members:
   :undoc-members:

Legibility
~~~~~~~~~~

.. automodule:: effectful.handlers.llm.harness.legibility
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.legibility.framework
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.legibility.lexical
   :members:
   :undoc-members:

Execution
~~~~~~~~~

.. automodule:: effectful.handlers.llm.harness.execution
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.execution.hooks
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.execution.builtin
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.execution.restricted
   :members:
   :undoc-members:

Validation
~~~~~~~~~~

.. automodule:: effectful.handlers.llm.harness.validation
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.validation.hooks
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.validation.mypy
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.validation.ty
   :members:
   :undoc-members:

Synthesis
~~~~~~~~~

.. automodule:: effectful.handlers.llm.harness.synthesis
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.synthesis.snippet
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.synthesis.function
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.synthesis.body
   :members:
   :undoc-members:

Durability
~~~~~~~~~~

.. automodule:: effectful.handlers.llm.harness.durability
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.durability.transaction
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.durability.retrying
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.durability.persistence
   :members:
   :undoc-members:

Observability
~~~~~~~~~~~~~

.. automodule:: effectful.handlers.llm.harness.observability
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.observability.rich
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.observability.dump
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.observability.langfuse
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
