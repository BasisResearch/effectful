Table of Contents
=================

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started
   introduction
   named_tensor_notation

.. toctree::
   :maxdepth: 1
   :caption: Examples

   minipyro_example
   lambda_example
   semi_ring_example
   beam_search_example

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   apidocs/index

.. Resolving ``effectful.handlers.jax`` through its ``__all__`` is what surfaces
   its re-exported operations, but it also empties that package's generated
   submodule toctree, so its one subpackage is linked here instead.

.. toctree::
   :hidden:

   apidocs/effectful/effectful.handlers.jax.numpy

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
