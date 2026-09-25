"""Reproduce a Skill failing to start when its lexical scope contains an array."""

import numpy as np
from litellm import ModelResponse

from effectful.handlers.llm import Skill
from effectful.handlers.llm.harness import harness
from effectful.handlers.llm.harness.hooks import completion
from effectful.ops.semantics import handler
from effectful.ops.types import NotHandled

values = np.array([1, 2])


@Skill.define
def describe() -> str:
    """Reply with OK."""
    raise NotHandled


with (
    handler(harness(num_retries=0)),
    handler(
        {
            completion: lambda **_: ModelResponse(
                choices=[{"message": {"role": "assistant", "content": "OK"}}]
            )
        }
    ),
):
    print(describe())
