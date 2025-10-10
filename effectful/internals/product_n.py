import dataclasses
import functools
from collections.abc import Callable, Mapping
from typing import Any

import tree

from effectful.ops.semantics import apply, coproduct, handler
from effectful.ops.syntax import defop
from effectful.ops.types import Interpretation, Operation


@dataclasses.dataclass
class CallByNeed[**P, T]:
    func: Callable[P, T]
    args: Any  # P.args
    kwargs: Any  # P.kwargs
    value: T | None = None
    initialized: bool = False

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        if not self.initialized:
            self.value = self.func(*self.args, **self.kwargs)
            self.initialized = True
        return self.value


@defop
def argsof(op: Operation) -> tuple[list, dict]:
    raise RuntimeError("Prompt argsof not bound.")


@dataclasses.dataclass
class Product[S, T]:
    values: object


def _pack(intp):
    from effectful.internals.runtime import interpreter

    return Product(interpreter(intp)(lambda x: x()))


def _unpack(x, prompt):
    if isinstance(x, Product):
        return x.values(prompt)
    return x


def productN(intps: Mapping[Operation, Interpretation]) -> Interpretation:
    # The resulting interpretation supports ops that exist in at least one input
    # interpretation
    result_ops = set(op for intp in intps.values() for op in intp)
    if result_ops is None:
        return {}

    renaming = {(prompt, op): defop(op) for prompt in intps for op in result_ops}

    # We enforce isolation between the named interpretations by giving every
    # operation a fresh name and giving each operation a translation from
    # the fresh names back to the names from their interpretation.
    #
    # E.g. { a: { f, g }, b: { f, h } } =>
    # { handler({f: f_a, g: g_a, h: h_default})(f_a), handler({f: f_a, g: g_a})(g_a),
    #   handler({f: f_b, h: h_b})(f_b), handler({f: f_b, h: h_b})(h_b) }
    translation_intps: dict[Operation, Interpretation] = {
        prompt: {op: renaming[(prompt, op)] for op in result_ops} for prompt in intps
    }

    # For every prompt, build an isolated interpretation that binds all operations.
    isolated_intps = {
        prompt: {
            renaming[(prompt, op)]: handler(translation_intps[prompt])(func)
            for op, func in intp.items()
        }
        for prompt, intp in intps.items()
    }

    def product_op(op, *args, **kwargs):
        """Compute the product of operation `op` in named interpretations
        `intps`. The product operation consumes product arguments and
        returns product results. These products are represented as
        interpretations.

        """
        assert isinstance(op, Operation)

        result_intp = {}

        def argsof_direct_call(prompt):
            return result_intp[prompt].args, result_intp[prompt].kwargs

        def argsof_apply(prompt):
            return result_intp[prompt].args[2:], result_intp[prompt].kwargs

        # Every prompt gets an argsof implementation. The implementation is
        # either for a direct call to a handler or for a call to an apply
        # handler.
        argsof_prompts = {}

        for prompt, intp in intps.items():
            # Args and kwargs are expected to be either interpretations with
            # bindings for each named analysis in intps or concrete values.
            # `get_for_intp` extracts the value that corresponds to this
            # analysis.
            #
            # TODO: `get_for_intp` has to guess whether a dict value is an
            # interpretation or not. This is probably a latent bug.
            intp_args, intp_kwargs = tree.map_structure(
                lambda x: _unpack(x, prompt), (args, kwargs)
            )

            # Making result a CallByNeed has two functions. It avoids some
            # work when the result is not requested and it delays evaluation
            # so that when the result is requested in `get_for_intp`, it
            # evaluates in a context that binds the results of the other
            # named interpretations.
            isolated_intp = isolated_intps[prompt]
            renamed_op = renaming[(prompt, op)]
            if op in intp:
                result = CallByNeed(
                    handler(isolated_intp)(renamed_op), *intp_args, **intp_kwargs
                )
                argsof_impl = argsof_direct_call
            elif apply in intp:
                result = CallByNeed(
                    handler(isolated_intp)(renaming[(prompt, apply)]),
                    renamed_op,
                    *intp_args,
                    **intp_kwargs,
                )
                argsof_impl = argsof_apply
            else:
                # TODO: If an intp does not handle an operation and has no apply
                # handler, use the default rule. In the future, we would like to
                # instead defer to the enclosing interpretation. This is
                # difficult right now, because the output interpretation handles
                # all operations with product handlers which would have to be
                # skipped over.
                result = CallByNeed(
                    handler(coproduct(isolated_intp, translation_intps[prompt]))(
                        op.__default_rule__
                    ),
                    *intp_args,
                    **intp_kwargs,
                )
                argsof_impl = argsof_direct_call

            result_intp[prompt] = result
            argsof_prompts[prompt] = argsof_impl

        result_intp[argsof] = lambda prompt: argsof_prompts[prompt](prompt)
        return _pack(result_intp)

    product_intp: Interpretation = {
        op: functools.partial(product_op, op) for op in result_ops
    }
    return product_intp
