from contextlib import contextmanager

import pyro
import pyro.primitives
from pyro.poutine.runtime import Message

from ..ops.core import Operation
from ..ops.handler import product, fwd


@Operation
def pyro_effect(msg: Message) -> Message:
    pyro.poutine.runtime.default_process_message(msg)
    return msg


def lift_messenger(m: pyro.poutine.messenger.Messenger):
    def lifted(msg: Message) -> Message:
        m._process_message(msg)

        if msg["stop"]:
            pyro.poutine.runtime.default_process_message(msg)
            return msg

        fwd(None)
        m._postprocess_message(msg)

        if msg["continuation"]:
            print(
                "Warning: messages with continuations are not handled by effectful.",
                file=sys.stderr,
            )
        return msg

    return lifted


@contextmanager
def lift_poutine_stack():
    def default_handler(msg: Message) -> Message:
        pyro.poutine.runtime.default_process_message(msg)
        return msg

    interp = {pyro_effect: default_handler}
    for messenger in pyro.poutine.runtime._PYRO_STACK:
        m_interp = {pyro_effect: lift_messenger(messenger)}
        interp = product(interp, m_interp)

    def my_apply_stack(msg: Message):
        pyro_effect(msg)

    old_apply_stack = pyro.poutine.runtime.apply_stack
    old_stack = pyro.poutine.runtime._PYRO_STACK
    try:
        pyro.poutine.runtime.apply_stack = my_apply_stack
        pyro.primitives.apply_stack = my_apply_stack
        pyro.poutine.runtime._PYRO_STACK = [None]
        yield interp
    finally:
        pyro.poutine.runtime.apply_stack = old_apply_stack
        pyro.primitives.apply_stack = old_apply_stack
        pyro.poutine.runtime._PYRO_STACK = old_stack
