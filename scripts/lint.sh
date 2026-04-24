#!/bin/bash
set -euxo pipefail

SRC="tests/ effectful/ docs/source/weighted_examples/"

mypy $SRC
for f in docs/source/*.py
do
    mypy $f
done

ruff check $SRC
ruff format --diff $SRC

nbqa 'mypy --no-incremental' docs
nbqa 'ruff check' docs
nbqa 'ruff format --diff' docs

