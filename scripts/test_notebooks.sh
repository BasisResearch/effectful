#!/bin/bash

INCLUDED_NOTEBOOKS=""

CI=1 pytest --nbval-lax --dist loadscope -n auto $INCLUDED_NOTEBOOKS
