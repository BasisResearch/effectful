# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Effectful is an algebraic effect system library for Python, providing algebraic effect handlers for metaprogramming and DSL implementation. It integrates with PyTorch, Pyro, JAX, and NumPyro for tensor operations and probabilistic programming. The core library has zero dependencies; all integrations are optional extras.

## Common Commands

### Development Setup
```bash
uv sync --all-extras --dev    # Install all dependencies
```

### Testing
```bash
make test                              # Lint + all tests
pytest effectful/ tests/ -n auto       # All tests (parallel)
pytest tests/test_ops_syntax.py -v     # Single test file
pytest -k test_name -v                 # Single test by name
```

CI runs tests in groups: `core`, `indexed`, `torch`, `pyro`, `jax`, `numpyro`, `llm`, `examples`. To run a specific group locally:
```bash
pytest tests/test_handlers_torch*.py -n auto
pytest effectful/ tests/test_ops_*.py tests/test_internals_*.py -n auto  # core group
```

Note: `--doctest-modules` is enabled by default via pyproject.toml.

### Linting & Formatting
```bash
make lint       # mypy + ruff check + ruff format --diff + nbqa on docs
make format     # Auto-fix: ruff --fix + ruff format
```

Ruff rules: `F`, `I`, `PERF`, `UP` (target Python 3.12).

### LLM Test Fixtures
```bash
make rebuild-fixtures    # Rebuilds LLM test fixtures (needs API keys)
```

## Architecture

### Core (`effectful/ops/`)

- **`types.py`** — Foundational types: `Operation[**Q, V]` (abstract effect), `Term[T]` (unevaluated operation application), `Interpretation` (handler dict mapping operations to functions), `NotHandled` exception.
- **`syntax.py`** — DSL for defining operations: `defop()`, `defdata()`, `deffn()`, `defstream()`. Includes `Scoped` annotation for variable binding semantics and `syntactic_eq()` for structural equality.
- **`semantics.py`** — Handler runtime: `handler()` context manager installs interpretations, `evaluate()` fully reduces expressions, `fwd()` delegates to parent handler, `coproduct()`/`product()` compose interpretations, `fvsof()` finds free variables.

### Handlers (`effectful/handlers/`)

Each handler implements interpretations for a specific library's operations:

- **`indexed.py`** — Named dimension indexing
- **`torch.py`** — PyTorch tensor operations with named dimensions
- **`pyro.py`** — Pyro probabilistic programming
- **`numpyro.py`** — NumPyro distributions (depends on JAX)
- **`jax/`** — JAX operations with numpy/scipy wrappers
- **`llm/`** — LLM integration (experimental): encoding Python→prompts, evaluating LLM outputs→Python, completions, templates

### Internals (`effectful/internals/`)

- **`unification.py`** — Pattern matching and unification algorithm (used by handler dispatch)
- **`runtime.py`** — Handler execution context (interpretation stack)
- **`product_n.py`** — N-ary product types

### Key Pattern

Operations are defined with `defop()`, producing callable objects. When no handler is installed, calling an operation creates a `Term` (lazy/symbolic). Handlers are installed via `handler()` context manager, providing concrete implementations. Multiple handlers compose via `coproduct()` (union) or `product()` (safe merge). The `fwd()` function inside a handler delegates to the next handler in the stack.
