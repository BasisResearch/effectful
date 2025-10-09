#!/bin/bash
set -euxo pipefail

SRC="tests/ effectful/"

mypy $SRC
ruff check $SRC
ruff format --diff $SRC

nbqa 'mypy --no-incremental' docs
nbqa 'ruff check' docs
nbqa 'ruff format --diff' docs

