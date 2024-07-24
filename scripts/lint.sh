#!/bin/bash
set -euxo pipefail

mypy .
isort --check --diff .
black --check .
flake8 .
