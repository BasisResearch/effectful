#!/bin/bash
set -euxo pipefail

MKL_NUM_THREADS=1 pytest tests/ -s -n auto --cov=effectful/ --cov-report=term-missing ${@-} --cov-report html
