#!/bin/bash
set -euxo pipefail

isort .
black .
