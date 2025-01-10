Lambda Calculus
===============

This file implements a simple call-by-value `lambda calculus <https://en.wikipedia.org/wiki/Lambda_calculus>`_ using ``effectful``.

It demonstrates the use of higher-order effects (i.e. effects that install handlers for other effects as part of their own operation).
Both :func:`Lam` and :func:`Let` are higher-order, as they handle their bound variables.

The :class:`Bound` and :class:`Scoped` annotations indicate the binding semantics---``effectful`` uses these annotations to compute the free variables of an expression.
An :class:`Operation` argument annotated with :class:`Bound` is considered bound in the scope of the operation, and will not be included in free variables of a term constructed with that operation.
An argument annotated with :class:`Scoped(n)` can see variables bound at levels greater than or equal to ``n``.
In the case of :func:`Let`, ``var`` is bound at level 0 and ``val`` is scoped at level 1, which indicates that ``var`` is not in scope in ``val`` so this is a non-recursive let-binding.

Reduction rules for the calculus are given as handlers for the syntax operations.

.. literalinclude:: ./lambda_.py
    :language: python
