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
from typing import List

sys.path.insert(0, os.path.abspath("../../"))
import sphinx_rtd_theme  # noqa: E402

# -- Project information -----------------------------------------------------

project = "effectful"
copyright = "2024, Basis"
author = "Basis"


# -- Type hints configuration ------------------------------------------------

autodoc_type_aliases = {
    "R": "R",
    "State": "State",
    "Dynamics": "Dynamics",
    "AtomicIntervention": "AtomicIntervention",
    "CompoundIntervention": "CompoundIntervention",
    "Intervention": "Intervention",
    "AtomicObservation": "AtomicObservation",
    "CompoundObservation": "CompoundObservation",
    "Observation": "Observation",
    "Kernel": "Kernel",
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    # "sphinx.ext.doctest",
    "sphinx.ext.imgconverter",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_parser",
    # "sphinx_gallery.gen_gallery",
    # "sphinx_search.extension",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.jquery",
]

# Point sphinxcontrib.bibtex to the bibtex file.
bibtex_bibfiles = ["refs.bib"]

# Enable documentation inheritance

autodoc_inherit_docstrings = True

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
exclude_patterns: List[str] = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
# logo
html_logo = "_static/img/chirho_logo_wide.png"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_style = "css/pyro.css"
