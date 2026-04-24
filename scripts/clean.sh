#!/bin/bash
set -euxo pipefail

SRC="effectful tests docs/source docs/source/weighted_examples"
ruff check --fix $SRC
ruff format $SRC

nbqa 'ruff check --fix' docs
nbqa 'ruff format' docs
