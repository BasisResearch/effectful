import os
import re
from collections.abc import Mapping
from typing import Any, cast

import litellm
import pytest

from effectful.handlers.llm.encoding import Encodable
from effectful.handlers.llm.template import Tool
from tests.test_handlers_llm_encoding import ROUNDTRIP_CASES

HAS_OPENAI_KEY = "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]
requires_openai = pytest.mark.skipif(
    not HAS_OPENAI_KEY, reason="OPENAI_API_KEY environment variable not set"
)

CHEAP_MODEL = "gpt-4o-mini"


def _unique_type_cases() -> list[tuple[Any, Mapping[str, Any] | None, str]]:
    cases: list[tuple[Any, Mapping[str, Any] | None, str]] = []
    seen: set[str] = set()
    for c in ROUNDTRIP_CASES:
        values = cast(tuple[Any, Any, Any], c.values)
        ty, _, ctx_raw = values
        ctx = ctx_raw if isinstance(ctx_raw, Mapping) else None
        ctx_keys = tuple(sorted((ctx or {}).keys()))
        key = f"{ty!r}|{ctx_keys!r}"
        if key in seen:
            continue
        seen.add(key)
        case_id = c.id if isinstance(c.id, str) else getattr(ty, "__name__", repr(ty))
        cases.append((ty, ctx, case_id))
    return cases


_UNIQUE_TYPES = _unique_type_cases()
_tuple_schema_bug_xfail = pytest.mark.xfail(
    reason="Known tuple schema bug; expected to fail until fixed."
)
_provider_response_format_xfail = pytest.mark.xfail(
    reason="Known OpenAI/LiteLLM response_format limitation for this type."
)


def _case_marks(case_id: str) -> list[pytest.MarkDecorator]:
    marks: list[pytest.MarkDecorator] = []
    if case_id.startswith("tuple-") or case_id in {
        "dc-with-tuple",
        "nt-coord",
        "nt-person",
    }:
        marks.append(_tuple_schema_bug_xfail)
    if case_id.startswith(("list-", "img-", "tool-", "dtc-")):
        marks.append(_provider_response_format_xfail)
    return marks


TYPE_CASES = [
    pytest.param(
        ty,
        ctx,
        id=case_id,
        marks=_case_marks(case_id),
    )
    for ty, ctx, case_id in _UNIQUE_TYPES
]


def _type_label(ty: Any) -> str:
    return getattr(ty, "__name__", repr(ty))


def _completion_with_response_model(
    *,
    model: str,
    prompt: str,
    response_model: Any,
    tools: list[dict[str, Any]] | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
    }
    if tools is not None:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "none"

    # LiteLLM/OpenAI treats plain string responses as the default (no schema).
    if response_model is str:
        return litellm.completion(**kwargs)

    try:
        return litellm.completion(response_model=response_model, **kwargs)
    except Exception as exc:
        # Backward compatibility with LiteLLM versions that use response_format.
        if "response_model" not in str(exc) and "JSON serializable" not in str(exc):
            raise
        return litellm.completion(response_format=response_model, **kwargs)


def _completion_with_tools(
    *, model: str, prompt: str, tools: list[dict[str, Any]]
) -> Any:
    return litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
        tool_choice="none",
        max_tokens=200,
    )


def _build_tool_pair(ty: Any, suffix: str) -> tuple[Tool[..., Any], Tool[..., Any]]:
    safe_suffix = re.sub(r"[^0-9a-zA-Z_]+", "_", suffix)

    def _accept(value):
        raise RuntimeError("Integration schema tool should not be called.")

    _accept.__name__ = f"foo_{safe_suffix}"
    _accept.__doc__ = f"Accept a value of type {suffix}."
    _accept.__annotations__ = {"value": ty, "return": None}

    def _invert():
        raise RuntimeError("Integration schema tool should not be called.")

    _invert.__name__ = f"foo_inv_{safe_suffix}"
    _invert.__doc__ = f"Return a value of type {suffix}."
    _invert.__annotations__ = {"return": ty}

    return Tool.define(_accept), Tool.define(_invert)


@requires_openai
@pytest.mark.parametrize("ty,ctx", TYPE_CASES)
def test_litellm_completion_accepts_encodable_response_model_for_supported_types(
    ty: Any, ctx: Mapping[str, Any] | None
) -> None:
    enc = Encodable.define(ty, ctx)
    response = _completion_with_response_model(
        model=CHEAP_MODEL,
        prompt=f"Return an instance of {_type_label(ty)}.",
        response_model=enc.enc,
    )
    assert response is not None


@requires_openai
@pytest.mark.xfail(
    reason="Includes tuple-shaped tool schemas, which are a known provider schema bug."
)
def test_litellm_completion_accepts_all_supported_tool_schemas_in_single_call() -> None:
    tool_specs: list[dict[str, Any]] = []

    for ty, _ctx, case_id in _UNIQUE_TYPES:
        foo, foo_inv = _build_tool_pair(ty, case_id)
        for tool in (foo, foo_inv):
            tool_ty = cast(type[Any], type(tool))
            tool_enc = cast(Encodable[Any, Any], Encodable.define(tool_ty))
            tool_spec_obj = tool_enc.encode(tool)
            if isinstance(tool_spec_obj, Mapping):
                tool_specs.append(dict(tool_spec_obj))
            elif hasattr(tool_spec_obj, "model_dump"):
                tool_specs.append(cast(dict[str, Any], tool_spec_obj.model_dump()))
            else:
                raise TypeError(
                    f"Unexpected encoded tool spec type: {type(tool_spec_obj)}"
                )

    response = _completion_with_tools(
        model=CHEAP_MODEL,
        prompt="Return hello, do NOT call any tools.",
        tools=tool_specs,
    )
    assert response is not None
