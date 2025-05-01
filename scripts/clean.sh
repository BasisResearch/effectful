#!/bin/bash
set -euxo pipefail

SRC="weighted tests examples"
ruff check --fix $SRC
ruff format $SRC
