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

Autoreload
~~~~~~~~~~

.. automodule:: effectful.handlers.llm.harness.autoreload
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

.. automodule:: effectful.handlers.llm.harness.legibility.mcp
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
   :private-members:

Validation
~~~~~~~~~~

.. automodule:: effectful.handlers.llm.harness.validation
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.validation.hooks
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.validation.pydantic
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

.. automodule:: effectful.handlers.llm.harness.synthesis.toolcall
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

.. automodule:: effectful.handlers.llm.harness.durability.compaction
   :members:
   :undoc-members:

.. automodule:: effectful.handlers.llm.harness.durability.truncation
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

Examples
""""""""

.. automodule:: effectful.handlers.llm.examples

.. automodule:: effectful.handlers.llm.examples.acp

.. automodule:: effectful.handlers.llm.examples.acp.assistant

.. automodule:: effectful.handlers.llm.examples.acp.client

.. automodule:: effectful.handlers.llm.examples.acp.library

.. automodule:: effectful.handlers.llm.examples.acp.server

.. automodule:: effectful.handlers.llm.examples.autoformalization

.. automodule:: effectful.handlers.llm.examples.autoformalization.informalization

.. automodule:: effectful.handlers.llm.examples.autoformalization.library

.. automodule:: effectful.handlers.llm.examples.autoformalization.verification

.. automodule:: effectful.handlers.llm.examples.autoresearch

.. automodule:: effectful.handlers.llm.examples.autoresearch.illustration

.. automodule:: effectful.handlers.llm.examples.autoresearch.implementation

.. automodule:: effectful.handlers.llm.examples.autoresearch.investigation

.. automodule:: effectful.handlers.llm.examples.autoresearch.review

.. automodule:: effectful.handlers.llm.examples.autoresearch.writing

.. automodule:: effectful.handlers.llm.examples.basics

.. automodule:: effectful.handlers.llm.examples.basics.conversation

.. automodule:: effectful.handlers.llm.examples.basics.error_recovery

.. automodule:: effectful.handlers.llm.examples.basics.flight_booking

.. automodule:: effectful.handlers.llm.examples.basics.guardrails

.. automodule:: effectful.handlers.llm.examples.basics.hitl

.. automodule:: effectful.handlers.llm.examples.basics.image_input

.. automodule:: effectful.handlers.llm.examples.basics.image_tool

.. automodule:: effectful.handlers.llm.examples.basics.lexical_scope

.. automodule:: effectful.handlers.llm.examples.basics.map_reduce

.. automodule:: effectful.handlers.llm.examples.basics.rag

.. automodule:: effectful.handlers.llm.examples.basics.research_agent

.. automodule:: effectful.handlers.llm.examples.basics.text2sql

.. automodule:: effectful.handlers.llm.examples.choreographies

.. automodule:: effectful.handlers.llm.examples.choreographies.library

.. automodule:: effectful.handlers.llm.examples.choreographies.multi_agent_choreography

.. automodule:: effectful.handlers.llm.examples.optimization

.. automodule:: effectful.handlers.llm.examples.optimization.avo

.. automodule:: effectful.handlers.llm.examples.optimization.ds1000

.. automodule:: effectful.handlers.llm.examples.optimization.ds1000_data

.. automodule:: effectful.handlers.llm.examples.optimization.guidelines

.. automodule:: effectful.handlers.llm.examples.optimization.kernels

.. automodule:: effectful.handlers.llm.examples.optimization.library

.. automodule:: effectful.handlers.llm.examples.optimization.packing

.. automodule:: effectful.handlers.llm.examples.optimization.prompting

.. automodule:: effectful.handlers.llm.examples.optimization.textgrad

.. automodule:: effectful.handlers.llm.examples.reasoning

.. automodule:: effectful.handlers.llm.examples.reasoning.aime2024

.. automodule:: effectful.handlers.llm.examples.reasoning.constrained_paragraph

.. automodule:: effectful.handlers.llm.examples.reasoning.continual

.. automodule:: effectful.handlers.llm.examples.reasoning.countdown

.. automodule:: effectful.handlers.llm.examples.reasoning.fix_typos

.. automodule:: effectful.handlers.llm.examples.reasoning.gridworlds

.. automodule:: effectful.handlers.llm.examples.reasoning.hanoi

.. automodule:: effectful.handlers.llm.examples.reasoning.lineup

.. automodule:: effectful.handlers.llm.examples.reasoning.taboo

.. automodule:: effectful.handlers.llm.examples.reasoning.theory_of_mind

.. automodule:: effectful.handlers.llm.examples.reasoning.world_model_agent


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
