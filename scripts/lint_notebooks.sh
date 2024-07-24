#!/bin/bash

nbqa mypy docs
nbqa isort --check --diff docs
nbqa black --check docs
nbqa flake8 docs
