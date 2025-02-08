#!/bin/bash
set -euxo pipefail

SRC="tests/ effectful/"
pyright $SRC
