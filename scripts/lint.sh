#!/bin/bash
set -euxo pipefail

SRC="tests/ effectful/"

mypy $SRC
for f in docs/source/*.py
do
    mypy $f
done

ruff check $SRC
ruff format --diff $SRC

nbqa mypy docs
nbqa 'ruff check' docs
nbqa 'ruff format --diff' docs

