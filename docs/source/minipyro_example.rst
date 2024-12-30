effectful-minipyro
==================

This file is a minimal implementation of the Pyro Programming Language,
similar in spirit to the minipyro implementation shipped with Pyro.
It adapts the API of minipyro (method signatures, etc.) to use the
newer Effectful system. Like the original minipyro, this file is
independent of the rest of Pyro, with the exception of the
:mod:`pyro.distributions` module.

This implementation conforms to the :mod:`pyroapi` module's interface, which
allows effectful-minipyro to be run against `pyroapi`'s test suite.

`View minipyro.py on github <https://github.com/BasisResearch/effectful/blob/master/docs/source/minipyro.py>`_

.. literalinclude:: ./minipyro.py
    :language: python
