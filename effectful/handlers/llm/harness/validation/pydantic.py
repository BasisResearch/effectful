"""Enforcement of the pre-conditions a caller writes into a `Skill`'s parameters.

A parameter annotated with pydantic metadata -- a `pydantic.AfterValidator`, an
`annotated_types` constraint, a `pydantic.Field` -- states a contract its
argument must satisfy::

    @Skill.define
    def select_seat(user_input: Annotated[str, Predicate(is_seat_request)]) -> Seat:
        \"""Extract the seat from {user_input}.\"""

What `PydanticSkillArgValidator` changes is the *top-level* call -- a skill
invoked from Python, by a caller who wrote the annotation and can reasonably
expect it to mean something. Nothing in Python consults an annotation, so
without this handler the contract silently lapses there. Installed, the argument
is validated before the prompt is built, and a violation raises
`pydantic.ValidationError` -- a `ValueError`, so ordinary handling still applies
-- rather than reaching the model at all.

A skill the *model* calls as a tool is validated with or without this handler:
`call_assistant` decodes each argument through `Encodable` of the parameter's
own annotation, so the metadata is applied as the call is decoded, before the
skill is ever entered. Two consequences follow.

* For a model-supplied argument the validation happens twice -- once at decode,
  once here -- so a pre-condition that costs something (an LLM-backed predicate,
  say) pays it twice.
* The exception is the expression pathway
  (`~effectful.handlers.llm.harness.synthesis.toolcall.ExpressionToolCaller`),
  which evaluates a Python call expression instead of decoding JSON arguments
  and does not apply the parameter's metadata. There this handler is the only
  thing enforcing the contract, which is why it is installed for every
  tool-calling mode rather than only the JSON one.

Post-conditions need no handler
-------------------------------

The mirror image -- metadata on the *return* annotation -- is enforced by the
decoder itself, since `call_assistant` decodes an answer through `Encodable` of
the skill's declared return type and the caller's metadata rides along. A
rejection there is a decoding failure, which
`~effectful.handlers.llm.harness.durability.retrying.TenacityRetryer` feeds back
to the model as the instruction for its next attempt. So this handler governs
the way in; the way out is contracted whether or not it is installed.
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
    """The arguments you were given have already been checked. Where a
    parameter's annotation carries a constraint -- a value range, a pattern, a
    predicate that has to hold -- that constraint was evaluated before this call
    reached you, and an argument that failed it never got here.

    So do not re-verify them. Spending a turn confirming that an argument
    satisfies a condition it was admitted for is work the harness already did,
    and its result cannot differ. Take the arguments as given and answer the
    request.

    Your answer is checked the same way, against any constraint on the return
    annotation. If it fails, you get the validation error back and another
    attempt, so an answer that is close but out of the declared range is worth
    correcting before you send it.
    """

    @implements(call_agent)
    def call_agent[**P, T](
        self, skill: Skill[P, T], *args: P.args, **kwargs: P.kwargs
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
