#!/bin/bash
set -euxo pipefail

SRC="tests/ weighted/"

mypy $SRC
ruff check $SRC
ruff format --diff $SRC
