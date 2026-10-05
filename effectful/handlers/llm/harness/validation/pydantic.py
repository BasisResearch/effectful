"""Validate annotated Skill arguments before a turn starts.

:class:`PydanticSkillArgValidator` applies Pydantic metadata on Skill
parameters, such as a range or predicate. A failing direct Python call raises
``pydantic.ValidationError`` before any model request. The standard harness
installs this handler when ``check_contracts=True``.

For example, the caller's argument is checked before the model receives it::

    from typing import Annotated

    from annotated_types import Predicate

    from effectful.handlers.llm import Skill

    def nonempty(value: str) -> bool:
        return bool(value.strip())

    @Skill.define
    def summarize(text: Annotated[str, Predicate(nonempty)]) -> str:
        \"\"\"Summarize {text} in one sentence.\"\"\"

JSON Tool-call arguments are also validated when decoded. Expression Tool
calls evaluate Python expressions instead, so this handler is needed there to
enforce Skill parameter constraints. It does not wrap plain Tools: parameter
metadata on a plain Tool is not enforced by the expression pathway. Return
constraints are checked by the answer decoder regardless of this handler.

See :mod:`effectful.handlers.llm.examples.basics.guardrails`.
"""

import collections.abc
import functools
import inspect
import typing

import pydantic

from effectful.handlers.llm.harness.hooks import (
    PromptInjectingInterpretation,
    call_agent,
)
from effectful.handlers.llm.harness.serialization import _TYPE_CHECK_ANCHOR_KEY
from effectful.handlers.llm.types import Encodable, Skill
from effectful.ops.semantics import fwd
from effectful.ops.syntax import implements


class PydanticSkillArgValidator(PromptInjectingInterpretation):
    """Annotation constraints on this Skill's arguments were checked before
    the turn started.
    Use the admitted values to answer the request. Constraints on your return
    value are checked before Python receives it; if validation fails and retries
    are installed, you will see the error and can repair the answer.
    """

    @implements(call_agent)
    def call_agent[**P, T](
        self, skill: Skill[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """Validate the annotated arguments, then forward the normalized call.

        Only a parameter carrying metadata of its own is touched, so a skill
        that declares no contracts is unaffected: nothing is validated, and no
        argument is round-tripped through the encoding (which would copy it).
        Variadic parameters are validated element-wise, since the metadata
        describes each item rather than the tuple or dict collecting them.

        The validated value *replaces* the bound one, so a validator that
        normalizes is applied rather than consulted and discarded, and it is the
        normalized arguments that get forwarded and rendered into the prompt.

        Validation runs under the call environment, so a pre-condition may be
        stated relative to the rest of the call -- ``info.context`` holds the
        other arguments and the skill's lexical scope, exactly as it does when
        the answer is decoded on the way back.
        """
        bound_args = skill.__signature__.bind(*args, **kwargs)
        bound_args.apply_defaults()
        annotated = {
            name: param
            for name, param in skill.__signature__.parameters.items()
            if name in bound_args.arguments
            and hasattr(param.annotation, "__metadata__")
        }
        env: collections.abc.Mapping[str, typing.Any] = skill.__context__.new_child(
            bound_args.arguments | {_TYPE_CHECK_ANCHOR_KEY: skill}
        )
        for name, param in annotated.items():
            encoding: pydantic.TypeAdapter[typing.Any] = pydantic.TypeAdapter(
                Encodable[param.annotation]  # type: ignore[name-defined]
            )
            check = functools.partial(encoding.validate_python, context=env)
            value = bound_args.arguments[name]
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                bound_args.arguments[name] = tuple(check(v) for v in value)
            elif param.kind is inspect.Parameter.VAR_KEYWORD:
                bound_args.arguments[name] = {k: check(v) for k, v in value.items()}
            else:
                bound_args.arguments[name] = check(value)

        return fwd(skill, *bound_args.args, **bound_args.kwargs)
