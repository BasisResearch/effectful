#!/bin/bash
set -euxo pipefail

SRC="weighted tests"
ruff check --fix $SRC
ruff format $SRC
