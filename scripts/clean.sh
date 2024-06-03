#!/bin/bash
set -euxo pipefail

isort --profile black effectful/ tests/
black effectful/ tests/
