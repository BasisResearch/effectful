
.. index-inclusion-marker

Effectful 
=========

Effectful is an algebraic effect system for Python, intended for use in the
implementation of probabilistic programming languages. It is a core component of
the `ChiRho <https://basisresearch.github.io/chirho/getting_started.html>`_
causal modeling language.

Installation
------------

Install From Source
^^^^^^^^^^^^^^^^^^^^
.. code:: sh

   git clone git@github.com:BasisResearch/effectful.git
   cd effectful
   git checkout master
   pip install -e .[pyro]

Install With Optional PyTorch/Pyro Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``effectful`` has optional support for `PyTorch <https://pytorch.org/>`_ (tensors
with named dimensions) and `Pyro <https://pyro.ai/>`_ (wrappers for Pyro
effects).

To enable PyTorch support:

.. code:: sh

   pip install effectful[torch]

Pyro support (which includes PyTorch support):

.. code:: sh

   pip install effectful[pyro]

Getting Started
---------------

Here's an example demonstrating how ``effectful`` can be used to implement a simple DSL that performs arithmetic on terms with free variables.

.. code:: python

   import functools

   from effectful.ops.types import Term
   from effectful.ops.syntax import defop
   from effectful.ops.semantics import handler, evaluate, coproduct, fwd
   from effectful.handlers.numbers import add

   def beta_add(x: int, y: int) -> int:        
       match x, y:
           case int(), int():
               return x + y
           case _:
               return fwd()

   def commute_add(x: int, y: int) -> int:
       match x, y:
           case Term(), int():
               return y + x  
           case _:
               return fwd()

   def assoc_add(x: int, y: int) -> int:
       match x, y:
           case _, Term(op, (a, b)) if op == add:
               return (x + a) + b 
           case _:
               return fwd()

   beta_rules = {add: beta_add}
   commute_rules = {add: commute_add}
   assoc_rules = {add: assoc_add}

   eager_mixed = functools.reduce(coproduct, (beta_rules, commute_rules, assoc_rules))

We can represent free variables as operations with no arguments, generated using ``defop``:

.. code:: python

   >>> x = defop(int, name="x")
   >>> y = defop(int, name="y")

If we evaluate an expression containing free variables, we get a term:

.. code:: python

   >>> e = 1 + 1 + (x() + 1) + (5 + y())
   >>> print(e)
   add(2, add(add(x(), 1), add(5, y())))

We can make the evaluation strategy smarter by taking advantage of the commutativity and associativity of addition, as expressed by the ``commute_add`` and ``assoc_add`` handlers.

.. code:: python

   >>> with handler(eager_mixed):
   >>>     print(evaluate(e))
   add(8, add(x(), y()))
   
Learn More
----------

More examples and API documentation can be found in the `docs <https://basisresearch.github.io/effectful/index.html>`_.
