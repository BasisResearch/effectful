Lambda Calculus
===============

This file implements a simple call-by-value `lambda calculus <https://en.wikipedia.org/wiki/Lambda_calculus>`_ using ``effectful``.

It demonstrates the use of higher-order effects (i.e. effects that install handlers for other effects as part of their own operation).
Both :func:`Lam` and :func:`Let` are higher-order, as they handle their bound variables.

The :class:`Scoped` annotation indicates the binding semantics---``effectful`` uses these annotations to compute the free variables of an expression.
An :class:`Operation` argument annotated with :class:`Scoped[A]` is considered bound within the scope identified by the type variable ``A``, and will not be included in the free variables of a term constructed with that operation.
Arguments sharing the same type variable in their :class:`Scoped` annotation share the same scope, while different type variables indicate independent scopes.
In the case of :func:`Let`, ``var`` is annotated with :class:`Scoped[A]` and ``body`` is also annotated with :class:`Scoped[A]`, indicating that ``var`` is bound within ``body``. The ``val`` argument has no :class:`Scoped` annotation, which means ``var`` is not in scope in ``val``, making this a non-recursive let-binding.

Reduction rules for the calculus are given as handlers for the syntax operations.

.. literalinclude:: ./lambda_.py
    :language: python
