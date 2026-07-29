#!/bin/bash

INCLUDED_NOTEBOOKS="docs/source/introduction.ipynb docs/source/named_tensor_notation.ipynb"

CI=1 pytest --nbval-lax --dist loadscope -n auto $INCLUDED_NOTEBOOKS
