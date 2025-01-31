#!/bin/bash
set -euxo pipefail

pytest effectful/ tests/ -s -n auto --cov=effectful/ --cov-report=term-missing ${@-} --cov-report html
