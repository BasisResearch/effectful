#!/bin/bash
set -euxo pipefail

SRC="tests/ effectful/"

mypy $SRC
isort --check --diff $SRC
black --check $SRC
flake8 $SRC
