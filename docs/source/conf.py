# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

project = "effectful"
copyright = "2025, Basis"
author = "Basis"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "sphinx.ext.imgconverter",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinxcontrib.jquery",
    "autodoc2",
]

autodoc2_packages = [{"path": "../../effectful", "module": "effectful"}]
autodoc2_render_plugin = "rst"

# The ``automodule`` blocks this replaced all set ``:undoc-members:``, so leave
# "undoc" out: adding it drops roughly 40% of the published objects.
autodoc2_hidden_objects = ["inherited", "private", "dunder"]

# Document only an object's own docstring.  Falling back to a base class's
# repeats it once per subclass -- the 40 distribution wrappers in
# ``handlers.numpyro`` would all carry ``_DistributionTerm``'s text -- and
# reaches stdlib boilerplate for classes whose base is a ``dict`` or an ``ABC``.
autodoc2_docstrings = "direct"

# Render the ``__init__`` docstring alongside the class docstring.  The LLM
# harness handlers rely on this split: a handler's class docstring is injected
# into the model's system prompt, so constructor documentation -- which only
# the caller of ``Handler(...)`` can act on -- lives on ``__init__`` instead.
autodoc2_class_docstring = "merge"

# ``effectful.handlers.jax`` re-exports its operations from ``._handlers``, which
# static analysis cannot follow; resolving through ``__all__`` recovers them.
# Scoped to that one module: a regex matching a module with no ``__all__`` raises
# an uncaught NoAllError, and a matched module loses its submodule toctree.
autodoc2_module_all_regexes = [r"effectful\.handlers\.jax"]

# These docstrings are Markdown because they double as model-facing prompts (see
# ``PromptInjectingInterpretation``).  Named individually rather than by subtree:
# most of ``handlers.llm`` is reStructuredText, and the ``:parser:`` path this
# option takes bloats doctrees, so its blast radius is kept small.
autodoc2_docstring_parser_regexes = [
    (r"effectful\.handlers\.llm\.types", "myst"),
    (r"effectful\.handlers\.llm\.types\.(Tool|Skill|Agent)", "myst"),
    (r"effectful\.handlers\.llm\.harness\.hooks\.AgentLoop", "myst"),
    (
        r"effectful\.handlers\.llm\.harness\.synthesis\.snippet"
        r"\.StatefulReplSynthesizer",
        "myst",
    ),
    (
        r"effectful\.handlers\.llm\.harness\.durability\.retrying\.TenacityRetryer",
        "myst",
    ),
]

# One per ``singledispatch`` registration written as ``def _(...)``.
suppress_warnings = ["autodoc2.dup_item"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
# NOTE: `.rst` is the default suffix of sphinx, and nbsphinx will
# automatically add support for `.ipynb` suffix.

# do not execute cells
nbsphinx_execute = "never"

# Don't add .txt suffix to source files:
html_sourcelink_suffix = ""

# The master toctree document.
master_doc = "index"


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: list[str] = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_logo = "_static/img/chirho_logo_wide.png"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_style = "css/pyro.css"
