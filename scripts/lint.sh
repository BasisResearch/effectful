#!/bin/bash
set -euxo pipefail

SRC="tests/ effectful/"

mypy $SRC
ruff check $SRC
ruff format --diff $SRC
