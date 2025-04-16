#!/bin/bash
set -euxo pipefail

pytest effectful/ tests/ -n auto tests/test_handlers_numbers.py::test_defun_4
