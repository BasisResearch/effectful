#!/bin/bash
set -euxo pipefail

pytest effectful/ tests/
