import re
from collections.abc import Mapping
from typing import Any

import litellm
import pydantic
import pytest

from effectful.handlers.llm.encoding import Encodable
from effectful.handlers.llm.template import Tool
from tests.test_handlers_llm_encoding import ROUNDTRIP_CASES
from tests.test_handlers_llm_tool_calling_book import requires_openai

CHEAP_MODEL = "gpt-4o-mini"


def _unique_type_cases() -> list[tuple[Any, Mapping[str, Any] | None, str]]:
    cases: list[tuple[Any, Mapping[str, Any] | None, str]] = []
    seen: set[str] = set()
    for c in ROUNDTRIP_CASES:
        ty, _, ctx_raw = c.values
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


def _encode_tool_spec(tool: Tool[..., Any]) -> dict[str, Any]:
    tool_ty: type[Any] = type(tool)
    tool_enc: Encodable[Any, Any] = Encodable.define(tool_ty)
    tool_spec_obj = tool_enc.encode(tool)
    if isinstance(tool_spec_obj, Mapping):
        return dict(tool_spec_obj)
    elif hasattr(tool_spec_obj, "model_dump"):
        return dict(tool_spec_obj.model_dump())
    raise TypeError(f"Unexpected encoded tool spec type: {type(tool_spec_obj)}")


@requires_openai
@pytest.mark.parametrize("ty,ctx", TYPE_CASES)
def test_litellm_completion_accepts_encodable_response_model_for_supported_types(
    ty: Any, ctx: Mapping[str, Any] | None
) -> None:
    enc = Encodable.define(ty, ctx)
    kwargs: dict[str, Any] = {
        "model": CHEAP_MODEL,
        "messages": [
            {
                "role": "user",
                "content": f"Return an instance of {getattr(ty, '__name__', repr(ty))}.",
            }
        ],
        "max_tokens": 200,
    }
    if enc.enc is not str:
        kwargs["response_format"] = enc.enc
    response = litellm.completion(**kwargs)
    assert response is not None

    content = response.choices[0].message.content
    assert content is not None, (
        f"Expected content in response for {getattr(ty, '__name__', repr(ty))}"
    )

    deserialized = enc.deserialize(content)
    pydantic.TypeAdapter(enc.enc).validate_python(deserialized)

    decoded = enc.decode(deserialized)
    pydantic.TypeAdapter(enc.base).validate_python(decoded)


@requires_openai
@pytest.mark.parametrize("ty,ctx", TYPE_CASES)
def test_litellm_completion_accepts_tool_with_type_as_param(
    ty: Any, ctx: Mapping[str, Any] | None
) -> None:
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", getattr(ty, "__name__", repr(ty)))

    def _fn(value):
        raise RuntimeError("should not be called")

    _fn.__name__ = f"accept_{name}"
    _fn.__doc__ = f"Accept a value of type {name}."
    _fn.__annotations__ = {"value": ty, "return": None}

    tool: Tool[..., Any] = Tool.define(_fn)
    response = litellm.completion(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": "Return hello, do NOT call any tools."}],
        tools=[_encode_tool_spec(tool)],
        tool_choice="none",
        max_tokens=200,
    )
    assert response is not None


@requires_openai
@pytest.mark.parametrize("ty,ctx", TYPE_CASES)
def test_litellm_completion_accepts_tool_with_type_as_return(
    ty: Any, ctx: Mapping[str, Any] | None
) -> None:
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", getattr(ty, "__name__", repr(ty)))

    def _fn():
        raise RuntimeError("should not be called")

    _fn.__name__ = f"return_{name}"
    _fn.__doc__ = f"Return a value of type {name}."
    _fn.__annotations__ = {"return": ty}

    tool: Tool[..., Any] = Tool.define(_fn)
    response = litellm.completion(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": "Return hello, do NOT call any tools."}],
        tools=[_encode_tool_spec(tool)],
        tool_choice="none",
        max_tokens=200,
    )
    assert response is not None
