"""Repro: test whether dataclass/BaseModel with Callable field works as output type."""

from collections.abc import Callable
from dataclasses import dataclass

import pydantic

from effectful.handlers.llm.completions import LiteLLMProvider, Template
from effectful.handlers.llm.encoding import Encodable
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled


# ── 1. Dataclass with a Callable field ──────────────────────────────────────

@dataclass
class DataclassWithCallable:
    name: str
    transform: Callable[[int], int]


# ── 2. Pydantic BaseModel with a Callable field ────────────────────────────

class BaseModelWithCallable(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    name: str
    transform: Callable[[int], int]


# ── Step A: Check Encodable[T] (no LLM call needed) ───────────────────────

print("=" * 60)
print("Step A: Encodable[T] smoke test")
print("=" * 60)

for label, ty in [
    ("dataclass w/ Callable field", DataclassWithCallable),
    ("BaseModel w/ Callable field", BaseModelWithCallable),
]:
    print(f"\n--- {label} ---")
    try:
        adapter = pydantic.TypeAdapter(Encodable[ty])
        print(f"  TypeAdapter(Encodable[{ty.__name__}]) OK")
        print(f"  JSON schema: {adapter.json_schema()}")
    except Exception as e:
        print(f"  Encodable FAILED: {type(e).__name__}: {e}")


# ── Step B: Template with callable-field return type (needs LLM) ───────────

@Template.define
def make_dataclass_with_callable() -> DataclassWithCallable:
    """Return a DataclassWithCallable with name "double" and a transform
    function that doubles its integer input."""
    raise NotHandled


@Template.define
def make_basemodel_with_callable() -> BaseModelWithCallable:
    """Return a BaseModelWithCallable with name "triple" and a transform
    function that triples its integer input."""
    raise NotHandled


print("\n" + "=" * 60)
print("Step B: End-to-end LLM call")
print("=" * 60)

model = "gpt-4o-mini"

for label, template_fn, test_val, expected_factor in [
    ("dataclass", make_dataclass_with_callable, 5, 2),
    ("BaseModel", make_basemodel_with_callable, 5, 3),
]:
    print(f"\n--- {label} w/ Callable field ---")
    try:
        with handler(LiteLLMProvider(model=model)):
            result = template_fn()
        print(f"  result = {result}")
        print(f"  result.name = {result.name}")
        print(f"  result.transform({test_val}) = {result.transform(test_val)}")
        assert callable(result.transform), "transform should be callable"
        print("  PASS")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
