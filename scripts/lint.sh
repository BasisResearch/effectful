#!/bin/bash
set -euxo pipefail

mypy --ignore-missing-imports effectful/
isort --check --profile black --diff effectful/ tests/
black --check effectful/ tests/
flake8 effectful/ tests/