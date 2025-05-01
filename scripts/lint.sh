#!/bin/bash
set -euxo pipefail

SRC="tests weighted examples"

mypy $SRC
ruff check $SRC
ruff format --diff $SRC
