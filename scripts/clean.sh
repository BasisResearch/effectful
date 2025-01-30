#!/bin/bash
set -euxo pipefail

SRC="effectful tests"
ruff check --fix $SRC
ruff format $SRC
