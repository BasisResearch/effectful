"""
Generator expression bytecode reconstruction module.

This module provides functionality to reconstruct AST representations from compiled
generator expressions by analyzing their bytecode. The primary use case is to recover
the original structure of generator comprehensions from their compiled form.

The only public-facing interface is the `reconstruct` function, which takes a
generator object and returns an AST node representing the original comprehension.
All other functions and classes in this module are internal implementation details.

Example:
    >>> g = (x * 2 for x in range(10) if x % 2 == 0)
    >>> ast_node = reconstruct(g)
    >>> # ast_node is now an ast.Expression representing the original expression
"""

import ast
import copy
import dis
import enum
import functools
import inspect
import sys
import types
import typing
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field, replace

CompExp = ast.GeneratorExp | ast.ListComp | ast.SetComp | ast.DictComp


class Placeholder(ast.Name):
    """Placeholder for AST nodes that are not yet resolved."""

    def __init__(self, id=".PLACEHOLDER", ctx=ast.Load()):
        super().__init__(id=id, ctx=ctx)


class DummyIterName(ast.Name):
    """Dummy name for the iterator variable in generator expressions."""

    def __init__(self, id=".0", ctx=ast.Load()):
        super().__init__(id=id, ctx=ctx)


class Null(ast.Constant):
    """Placeholder for NULL values generated in bytecode."""

    def __init__(self, value=None):
        super().__init__(value=value)


class CompLambda(ast.Lambda):
    """Placeholder AST node representing a lambda function used in comprehensions."""

    def __init__(self, body: CompExp):
        assert isinstance(body, CompExp)
        assert sum(1 for x in ast.walk(body) if isinstance(x, DummyIterName)) == 1
        assert len(body.generators) > 0
        assert isinstance(body.generators[0].iter, DummyIterName)
        args = ast.arguments(
            posonlyargs=[ast.arg(DummyIterName().id)],
            args=[],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        )
        super().__init__(args=args, body=body)

    def __copy__(self):
        """Support copy.copy operation."""
        assert isinstance(self.body, CompExp)
        return CompLambda(self.body)

    def __deepcopy__(self, memo):
        """Support copy.deepcopy operation."""
        assert isinstance(self.body, CompExp)
        return CompLambda(copy.deepcopy(self.body, memo))

    def inline(self, iterator: ast.expr) -> CompExp:
        assert isinstance(self.body, CompExp)
        res: CompExp = copy.deepcopy(self.body)
        res.generators[0].iter = iterator
        return res


@dataclass(frozen=True)
class ReconstructionState:
    """State maintained during AST reconstruction from bytecode.

    This class tracks all the information needed while processing bytecode
    instructions to reconstruct the original comprehension's AST. It acts
    as the working memory during the reconstruction process, maintaining
    both the evaluation stack state and the high-level comprehension structure
    being built.

    The reconstruction process works by simulating the Python VM's execution
    of the bytecode, but instead of executing operations, it builds AST nodes
    that represent those operations.

    Attributes:
        result: The current comprehension expression being built. Initially
                a placeholder, it gets updated as the bytecode is processed.
                It can be a GeneratorExp, ListComp, SetComp, DictComp, or
                a Lambda for lambda expressions.

        stack: Simulates the Python VM's value stack. Contains AST nodes or
               values that would be on the stack during execution. Operations
               like LOAD_FAST push to this stack, while operations like
               BINARY_ADD pop operands and push results.
    """

    result: ast.expr = field(default_factory=Placeholder)
    stack: list[ast.expr] = field(default_factory=list)


# Python version enum for version-specific handling
class PythonVersion(enum.Enum):
    PY_310 = 10
    PY_313 = 13


# Global handler registry
OpHandler = Callable[[ReconstructionState, dis.Instruction], ReconstructionState]

OP_HANDLERS: dict[str, OpHandler] = {}


@typing.overload
def register_handler(
    opname: str, *, version: PythonVersion = PythonVersion(sys.version_info.minor)
) -> Callable[[OpHandler], OpHandler]: ...


@typing.overload
def register_handler(
    opname: str,
    handler: OpHandler,
    *,
    version: PythonVersion = PythonVersion(sys.version_info.minor),
) -> OpHandler: ...


def register_handler(
    opname: str,
    handler=None,
    *,
    version: PythonVersion = PythonVersion(sys.version_info.minor),
):
    """Register a handler for a specific operation name and optional version"""
    if handler is None:
        return functools.partial(register_handler, opname, version=version)

    # Skip registration if version doesn't match current version
    if version != PythonVersion(sys.version_info.minor):
        return handler

    # Only check opmap if the version matches (or no version specified)
    assert opname in dis.opmap, f"Invalid operation name: '{opname}'"

    if opname in OP_HANDLERS:
        raise ValueError(f"Handler for '{opname}' (version {version}) already exists.")

    @functools.wraps(handler)
    def _wrapper(
        state: ReconstructionState, instr: dis.Instruction
    ) -> ReconstructionState:
        assert instr.opname == opname, (
            f"Handler for '{opname}' called with wrong instruction"
        )

        new_state = handler(state, instr)

        # post-condition: check stack effect
        expected_stack_effect = dis.stack_effect(instr.opcode, instr.arg)
        actual_stack_effect = len(new_state.stack) - len(state.stack)
        if not (
            len(state.stack) == len(new_state.stack) == 0 or instr.opname == "END_FOR"
        ):
            assert len(state.stack) + expected_stack_effect >= 0, (
                f"Handler for '{opname}' would result in negative stack size"
            )
            assert actual_stack_effect == expected_stack_effect, (
                f"Handler for '{opname}' has incorrect stack effect: "
                f"expected {expected_stack_effect}, got {actual_stack_effect}"
            )

        return new_state

    OP_HANDLERS[opname] = _wrapper
    return handler  # return the original handler for multiple decorator usage


# ============================================================================
# GENERATOR COMPREHENSION HANDLERS
# ============================================================================


@register_handler("GEN_START", version=PythonVersion.PY_310)
def handle_gen_start(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # GEN_START is the first instruction in generator expressions in Python 3.10
    # It initializes the generator
    assert isinstance(state.result, Placeholder), (
        "GEN_START must be the first instruction"
    )
    return replace(state, result=ast.GeneratorExp(elt=Placeholder(), generators=[]))


@register_handler("RETURN_GENERATOR", version=PythonVersion.PY_313)
def handle_return_generator(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # RETURN_GENERATOR is the first instruction in generator expressions in Python 3.13+
    # It initializes the generator
    assert isinstance(state.result, Placeholder), (
        "RETURN_GENERATOR must be the first instruction"
    )
    new_result = ast.GeneratorExp(elt=Placeholder(), generators=[])
    new_stack = state.stack + [new_result]
    return replace(state, result=new_result, stack=new_stack)


@register_handler("YIELD_VALUE")
def handle_yield_value(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # YIELD_VALUE pops a value from the stack and yields it
    # This is the expression part of the generator
    assert isinstance(state.result, ast.GeneratorExp), (
        "YIELD_VALUE must be called after RETURN_GENERATOR"
    )
    assert isinstance(state.result.elt, Placeholder), (
        "YIELD_VALUE must be called before yielding"
    )
    assert len(state.result.generators) > 0, "YIELD_VALUE should have generators"

    ret = ast.GeneratorExp(
        elt=ensure_ast(state.stack[-1]),
        generators=state.result.generators,
    )
    return replace(state, result=ret)


# ============================================================================
# LIST COMPREHENSION HANDLERS
# ============================================================================


@register_handler("BUILD_LIST")
def handle_build_list(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    assert instr.arg is not None
    size: int = instr.arg

    if size == 0:
        # Check if this looks like the start of a list comprehension pattern
        # In nested comprehensions, BUILD_LIST(0) starts a new list comprehe
        new_ret = ast.ListComp(elt=Placeholder(), generators=[])
        new_stack = state.stack + [state.result]
        return replace(state, stack=new_stack, result=new_ret)
    else:
        # BUILD_LIST with elements - create a regular list
        elements = [ensure_ast(elem) for elem in state.stack[-size:]]
        new_stack = state.stack[:-size]
        elt_node = ast.List(elts=elements, ctx=ast.Load())
        new_stack = new_stack + [elt_node]
        return replace(state, stack=new_stack)


@register_handler("LIST_APPEND")
def handle_list_append(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    assert isinstance(state.result, ast.ListComp)
    assert isinstance(state.result.elt, Placeholder)

    # add the body to the comprehension
    comp: ast.ListComp = copy.deepcopy(state.result)
    comp.elt = state.stack[-1]

    # swap the return value
    prev_result: CompExp = state.stack[-instr.argval - 1]
    new_stack = state.stack[:-1]
    new_stack[-instr.argval] = comp

    return replace(state, stack=new_stack, result=prev_result)


# ============================================================================
# SET COMPREHENSION HANDLERS
# ============================================================================


@register_handler("BUILD_SET")
def handle_build_set(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    assert instr.arg is not None
    size: int = instr.arg

    if size == 0:
        new_result = ast.SetComp(elt=Placeholder(), generators=[])
        new_stack = state.stack + [state.result]
        return replace(state, stack=new_stack, result=new_result)
    else:
        elements = [ensure_ast(elem) for elem in state.stack[-size:]]
        new_stack = state.stack[:-size]
        elt_node = ast.Set(elts=elements)
        new_stack = new_stack + [elt_node]
        return replace(state, stack=new_stack)


@register_handler("SET_ADD")
def handle_set_add(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    assert isinstance(state.result, ast.SetComp)
    assert isinstance(state.result.elt, Placeholder)

    # add the body to the comprehension
    comp: ast.SetComp = copy.deepcopy(state.result)
    comp.elt = state.stack[-1]

    # swap the return value
    prev_result: CompExp = state.stack[-instr.argval - 1]
    new_stack = state.stack[:-1]
    new_stack[-instr.argval] = comp

    return replace(state, stack=new_stack, result=prev_result)


# ============================================================================
# DICT COMPREHENSION HANDLERS
# ============================================================================


@register_handler("BUILD_MAP")
def handle_build_map(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    assert instr.arg is not None
    size: int = instr.arg

    if size == 0:
        new_stack = state.stack + [state.result]
        new_result = ast.DictComp(key=Placeholder(), value=Placeholder(), generators=[])
        return replace(state, stack=new_stack, result=new_result)
    else:
        # Pop key-value pairs for the dict
        keys: list[ast.expr | None] = [
            ensure_ast(state.stack[-2 * i - 2]) for i in range(size)
        ]
        values = [ensure_ast(state.stack[-2 * i - 1]) for i in range(size)]
        new_stack = state.stack[: -2 * size] if size > 0 else state.stack

        # Create dict AST
        dict_node = ast.Dict(keys=keys, values=values)
        new_stack = new_stack + [dict_node]
        return replace(state, stack=new_stack)


@register_handler("MAP_ADD")
def handle_map_add(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    assert isinstance(state.result, ast.DictComp)
    assert isinstance(state.result.key, Placeholder)
    assert isinstance(state.result.value, Placeholder)

    # add the body to the comprehension
    comp: ast.DictComp = copy.deepcopy(state.result)
    comp.key = state.stack[-2]
    comp.value = state.stack[-1]

    # swap the return value
    prev_result: CompExp = state.stack[-instr.argval - 2]
    new_stack = state.stack[:-2]
    new_stack[-instr.argval] = comp

    return replace(state, stack=new_stack, result=prev_result)


# ============================================================================
# LOOP CONTROL HANDLERS
# ============================================================================


@register_handler("RETURN_VALUE")
def handle_return_value(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # RETURN_VALUE ends the generator
    # Usually preceded by LOAD_CONST None
    if isinstance(state.result, CompExp):
        return replace(state, stack=state.stack[:-1])
    elif isinstance(state.result, Placeholder):
        new_result = ensure_ast(state.stack[0])
        assert isinstance(new_result, ast.expr)
        return replace(state, stack=state.stack[1:], result=new_result)
    else:
        raise TypeError("Unexpected RETURN_VALUE in reconstruction")


@register_handler("FOR_ITER")
def handle_for_iter(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # FOR_ITER pops an iterator from the stack and pushes the next item
    # If the iterator is exhausted, it jumps to the target instruction
    assert len(state.stack) > 0, "FOR_ITER must have an iterator on the stack"
    assert isinstance(state.result, CompExp), (
        "FOR_ITER must be called within a comprehension context"
    )

    # The iterator should be on top of stack
    iterator: ast.expr = state.stack[-1]

    # Create a new loop variable - we'll get the actual name from STORE_FAST
    # For now, use a placeholder
    loop_info = ast.comprehension(
        target=Placeholder(),
        iter=ensure_ast(iterator),
        ifs=[],
        is_async=0,
    )

    # Create new loops list with the new loop info
    assert isinstance(state.result, CompExp)
    new_ret = copy.deepcopy(state.result)
    new_ret.generators = new_ret.generators + [loop_info]

    new_stack = state.stack + [loop_info.target]
    assert isinstance(new_ret, CompExp)
    return replace(state, stack=new_stack, result=new_ret)


@register_handler("GET_ITER")
def handle_get_iter(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # GET_ITER converts the top stack item to an iterator
    # For AST reconstruction, we typically don't need to change anything
    # since the iterator will be used directly in the comprehension
    return state


@register_handler("JUMP_FORWARD")
def handle_jump_forward(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # JUMP_FORWARD is used to jump forward in the code
    # In generator expressions, this is often used to skip code in conditional logic
    return state


@register_handler("JUMP_ABSOLUTE", version=PythonVersion.PY_310)
def handle_jump_absolute(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # JUMP_ABSOLUTE is used to jump back to the beginning of a loop
    # In generator expressions, this typically indicates the end of the loop body
    return state


@register_handler("JUMP_BACKWARD", version=PythonVersion.PY_313)
def handle_jump_backward(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # JUMP_BACKWARD is used to jump back to the beginning of a loop (replaces JUMP_ABSOLUTE in 3.13)
    # In generator expressions, this typically indicates the end of the loop body
    return state


@register_handler("RESUME", version=PythonVersion.PY_313)
def handle_resume(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # RESUME is used for resuming execution after yield/await - mostly no-op for AST reconstruction
    return state


@register_handler("END_FOR", version=PythonVersion.PY_313)
def handle_end_for(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # END_FOR marks the end of a for loop - no action needed for AST reconstruction
    new_stack = state.stack  # [:-1]
    return replace(state, stack=new_stack)


@register_handler("RETURN_CONST", version=PythonVersion.PY_313)
def handle_return_const(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # RETURN_CONST returns a constant value (replaces some LOAD_CONST + RETURN_VALUE patterns)
    # Similar to RETURN_VALUE but with a constant
    if isinstance(state.result, ast.GeneratorExp):
        # For generators, this typically ends the generator with None
        return state
    else:
        raise TypeError("Unexpected RETURN_CONST in reconstruction")


@register_handler("RERAISE", version=PythonVersion.PY_313)
def handle_reraise(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # RERAISE re-raises an exception - generally ignore for AST reconstruction
    assert not state.stack  # in generator expressions, we shouldn't have a stack here
    return state


# ============================================================================
# VARIABLE OPERATIONS HANDLERS
# ============================================================================


@register_handler("LOAD_FAST")
def handle_load_fast(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    var_name: str = instr.argval

    if var_name == ".0":
        # Special handling for .0 variable (the iterator)
        new_stack = state.stack + [DummyIterName()]
    else:
        # Regular variable load
        new_stack = state.stack + [ast.Name(id=var_name, ctx=ast.Load())]

    return replace(state, stack=new_stack)


@register_handler("LOAD_DEREF")
def handle_load_deref(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LOAD_DEREF loads a value from a closure variable
    var_name = instr.argval
    new_stack = state.stack + [ast.Name(id=var_name, ctx=ast.Load())]
    return replace(state, stack=new_stack)


@register_handler("LOAD_CLOSURE")
def handle_load_closure(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LOAD_CLOSURE loads a closure variable
    var_name = instr.argval
    new_stack = state.stack + [ast.Name(id=var_name, ctx=ast.Load())]
    return replace(state, stack=new_stack)


@register_handler("LOAD_CONST")
def handle_load_const(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    const_value = instr.argval
    new_stack = state.stack + [ensure_ast(const_value)]
    return replace(state, stack=new_stack)


@register_handler("LOAD_GLOBAL")
def handle_load_global(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    global_name = instr.argval

    if instr.argrepr.endswith(" + NULL"):
        new_stack = state.stack + [ast.Name(id=global_name, ctx=ast.Load()), Null()]
    else:
        new_stack = state.stack + [ast.Name(id=global_name, ctx=ast.Load())]
    return replace(state, stack=new_stack)


@register_handler("LOAD_NAME")
def handle_load_name(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LOAD_NAME is similar to LOAD_GLOBAL but for names in the global namespace
    name = instr.argval
    new_stack = state.stack + [ast.Name(id=name, ctx=ast.Load())]
    return replace(state, stack=new_stack)


@register_handler("STORE_FAST")
def handle_store_fast(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    assert isinstance(state.result, CompExp) and state.result.generators, (
        "STORE_FAST must be called within a comprehension context"
    )
    var_name = instr.argval

    if not state.stack or (
        isinstance(state.stack[-1], ast.Name) and state.stack[-1].id == var_name
    ):
        # If the variable is already on the stack, we can skip adding it again
        # This is common in nested comprehensions where the same variable is reused
        return replace(state, stack=state.stack[:-1])

    new_stack = state.stack[:-1]
    new_result: CompExp = copy.deepcopy(state.result)
    new_result.generators[-1].target = ast.Name(id=var_name, ctx=ast.Store())
    return replace(state, stack=new_stack, result=new_result)


@register_handler("STORE_DEREF")
def handle_store_deref(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # STORE_DEREF stores a value into a closure variable
    assert isinstance(state.result, CompExp) and state.result.generators, (
        "STORE_DEREF must be called within a comprehension context"
    )
    var_name = instr.argval

    if not state.stack or (
        isinstance(state.stack[-1], ast.Name) and state.stack[-1].id == var_name
    ):
        # If the variable is already on the stack, we can skip adding it again
        # This is common in nested comprehensions where the same variable is reused
        return replace(state, stack=state.stack[:-1])

    new_stack = state.stack[:-1]
    new_result: CompExp = copy.deepcopy(state.result)
    new_result.generators[-1].target = ast.Name(id=var_name, ctx=ast.Store())
    return replace(state, stack=new_stack, result=new_result)


@register_handler("STORE_FAST_LOAD_FAST", version=PythonVersion.PY_313)
def handle_store_fast_load_fast(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # STORE_FAST_LOAD_FAST stores and then loads the same variable (optimization)
    # The instruction has two names: store_name and load_name
    # In Python 3.13, this is often used for loop variables

    # First handle the store part
    assert isinstance(state.result, CompExp) and state.result.generators, (
        "STORE_FAST_LOAD_FAST must be called within a comprehension context"
    )

    # In Python 3.13, the instruction argument contains both names
    # argval should be a tuple (store_name, load_name)
    assert isinstance(instr.argval, tuple)
    store_name, load_name = instr.argval

    new_stack = state.stack[:-1] + [ast.Name(id=load_name, ctx=ast.Load())]
    new_result: CompExp = copy.deepcopy(state.result)
    new_result.generators[-1].target = ast.Name(id=store_name, ctx=ast.Store())
    return replace(state, stack=new_stack, result=new_result)


@register_handler("LOAD_FAST_AND_CLEAR", version=PythonVersion.PY_313)
def handle_load_fast_and_clear(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LOAD_FAST_AND_CLEAR pushes a local variable onto the stack and clears it
    # For AST reconstruction, we treat this the same as LOAD_FAST
    var_name: str = instr.argval

    if var_name == ".0":
        # Special handling for .0 variable (the iterator)
        new_stack = state.stack + [DummyIterName()]
    else:
        # Regular variable load
        new_stack = state.stack + [ast.Name(id=var_name, ctx=ast.Load())]

    return replace(state, stack=new_stack)


@register_handler("LOAD_FAST_LOAD_FAST", version=PythonVersion.PY_313)
def handle_load_fast_load_fast(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LOAD_FAST_LOAD_FAST loads two variables (optimization in Python 3.13)
    # The instruction argument contains both variable names
    if isinstance(instr.argval, tuple):
        var1, var2 = instr.argval
    else:
        # Fallback: assume both names are the same
        var1 = var2 = instr.argval

    new_stack = state.stack

    # Load first variable
    if var1 == ".0":
        new_stack = new_stack + [DummyIterName()]
    else:
        new_stack = new_stack + [ast.Name(id=var1, ctx=ast.Load())]

    # Load second variable
    if var2 == ".0":
        new_stack = new_stack + [DummyIterName()]
    else:
        new_stack = new_stack + [ast.Name(id=var2, ctx=ast.Load())]

    return replace(state, stack=new_stack)


@register_handler("MAKE_CELL", version=PythonVersion.PY_313)
def handle_make_cell(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # MAKE_CELL creates a new cell in slot i for closure variables
    # This is used when variables from outer scopes are captured by inner scopes
    # For AST reconstruction purposes, this is just a variable scoping mechanism
    # that we can ignore since the AST doesn't track low-level closure details
    return state


@register_handler("COPY_FREE_VARS", version=PythonVersion.PY_313)
def handle_copy_free_vars(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # COPY_FREE_VARS copies n free (closure) variables from the closure into the frame
    # This removes the need for special code on the caller's side when calling closures
    # For AST reconstruction purposes, this is just a variable scoping mechanism
    # that we can ignore since the AST doesn't track runtime variable management
    return state


# ============================================================================
# STACK MANAGEMENT HANDLERS
# ============================================================================


@register_handler("POP_TOP")
def handle_pop_top(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # POP_TOP removes the top item from the stack
    # In generators, often used after YIELD_VALUE
    # Also used to clean up the duplicated middle value in failed chained comparisons
    new_stack = state.stack[:-1]
    return replace(state, stack=new_stack)


@register_handler("DUP_TOP", version=PythonVersion.PY_310)
def handle_dup_top(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # DUP_TOP duplicates the top stack item
    top_item = state.stack[-1]
    new_stack = state.stack + [top_item]
    return replace(state, stack=new_stack)


@register_handler("ROT_TWO", version=PythonVersion.PY_310)
def handle_rot_two(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # ROT_TWO swaps the top two stack items
    new_stack = state.stack[:-2] + [state.stack[-1], state.stack[-2]]
    return replace(state, stack=new_stack)


@register_handler("ROT_THREE", version=PythonVersion.PY_310)
def handle_rot_three(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # ROT_THREE rotates the top three stack items
    # TOS -> TOS1, TOS1 -> TOS2, TOS2 -> TOS
    new_stack = state.stack[:-3] + [state.stack[-2], state.stack[-1], state.stack[-3]]

    # Check if the top two items are the same (from DUP_TOP)
    # This heuristic indicates we're setting up for a chained comparison
    if len(state.stack) >= 3 and state.stack[-1] == state.stack[-2]:
        raise NotImplementedError("Chained comparison not implemented yet")

    return replace(state, stack=new_stack)


@register_handler("ROT_FOUR", version=PythonVersion.PY_310)
def handle_rot_four(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # ROT_FOUR rotates the top four stack items
    # TOS -> TOS1, TOS1 -> TOS2, TOS2 -> TOS3, TOS3 -> TOS
    new_stack = state.stack[:-4] + [
        state.stack[-2],
        state.stack[-1],
        state.stack[-4],
        state.stack[-3],
    ]
    return replace(state, stack=new_stack)


# Python 3.13 replacement for stack manipulation
@register_handler("SWAP", version=PythonVersion.PY_313)
def handle_swap(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # SWAP exchanges the top two stack items (replaces ROT_TWO in many cases)
    assert instr.arg is not None
    depth = instr.arg
    stack_size = len(state.stack)

    if depth > stack_size:
        # Not enough items on stack - this might be a pattern where some items were optimized away
        # For AST reconstruction, we can often ignore certain stack manipulations
        return state

    if depth == 2 and stack_size >= 2:
        # Equivalent to ROT_TWO
        new_stack = state.stack[:-2] + [state.stack[-1], state.stack[-2]]
        return replace(state, stack=new_stack)
    elif depth <= stack_size:
        # For other depths, swap TOS with the item at specified depth
        idx = stack_size - depth
        new_stack = state.stack.copy()
        new_stack[-1], new_stack[idx] = new_stack[idx], new_stack[-1]
        return replace(state, stack=new_stack)
    else:
        # Edge case - not enough items, just return unchanged
        return state


@register_handler("COPY", version=PythonVersion.PY_313)
def handle_copy(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # COPY duplicates the item at the specified depth (replaces DUP_TOP in many cases)
    assert instr.arg is not None
    depth = instr.arg
    if depth == 1:
        # Equivalent to DUP_TOP
        top_item = state.stack[-1]
        new_stack = state.stack + [top_item]
        return replace(state, stack=new_stack)
    else:
        # Copy the item at specified depth to top of stack
        stack_size = len(state.stack)
        if depth > stack_size:
            raise ValueError(f"COPY depth {depth} exceeds stack size {stack_size}")
        idx = stack_size - depth
        copied_item = state.stack[idx]
        new_stack = state.stack + [copied_item]
        return replace(state, stack=new_stack)


@register_handler("PUSH_NULL", version=PythonVersion.PY_313)
def handle_push_null(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    return replace(state, stack=state.stack + [Null()])


# ============================================================================
# BINARY ARITHMETIC/LOGIC OPERATION HANDLERS
# ============================================================================


def handle_binop(
    op: ast.operator, state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    right = ensure_ast(state.stack[-1])
    left = ensure_ast(state.stack[-2])
    new_stack = state.stack[:-2] + [ast.BinOp(left=left, op=op, right=right)]
    return replace(state, stack=new_stack)


# Python 3.13 BINARY_OP handler
@register_handler("BINARY_OP", version=PythonVersion.PY_313)
def handle_binary_op(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # BINARY_OP in Python 3.13 consolidates all binary operations
    # The operation type is determined by the instruction argument
    assert instr.arg is not None

    # Map argument values to AST operators based on Python 3.13 implementation
    op_map = {
        0: ast.Add(),  # +
        1: ast.BitAnd(),  # &
        2: ast.FloorDiv(),  # //
        3: ast.LShift(),  # <<
        5: ast.Mult(),  # *
        6: ast.Mod(),  # %
        7: ast.BitOr(),  # | (guessing based on pattern)
        8: ast.Pow(),  # **
        9: ast.RShift(),  # >>
        10: ast.Sub(),  # -
        11: ast.Div(),  # /
        12: ast.BitXor(),  # ^
    }

    op = op_map.get(instr.arg)
    if op is None:
        raise NotImplementedError(f"Unknown binary operation: {instr.arg}")

    return handle_binop(op, state, instr)


# Legacy binary operation handlers (for Python 3.10 compatibility)
handler_binop_add = register_handler(
    "BINARY_ADD",
    functools.partial(handle_binop, ast.Add()),
    version=PythonVersion.PY_310,
)
handler_binop_subtract = register_handler(
    "BINARY_SUBTRACT",
    functools.partial(handle_binop, ast.Sub()),
    version=PythonVersion.PY_310,
)
handler_binop_multiply = register_handler(
    "BINARY_MULTIPLY",
    functools.partial(handle_binop, ast.Mult()),
    version=PythonVersion.PY_310,
)
handler_binop_true_divide = register_handler(
    "BINARY_TRUE_DIVIDE",
    functools.partial(handle_binop, ast.Div()),
    version=PythonVersion.PY_310,
)
handler_binop_floor_divide = register_handler(
    "BINARY_FLOOR_DIVIDE",
    functools.partial(handle_binop, ast.FloorDiv()),
    version=PythonVersion.PY_310,
)
handler_binop_modulo = register_handler(
    "BINARY_MODULO",
    functools.partial(handle_binop, ast.Mod()),
    version=PythonVersion.PY_310,
)
handler_binop_power = register_handler(
    "BINARY_POWER",
    functools.partial(handle_binop, ast.Pow()),
    version=PythonVersion.PY_310,
)
handler_binop_lshift = register_handler(
    "BINARY_LSHIFT",
    functools.partial(handle_binop, ast.LShift()),
    version=PythonVersion.PY_310,
)
handler_binop_rshift = register_handler(
    "BINARY_RSHIFT",
    functools.partial(handle_binop, ast.RShift()),
    version=PythonVersion.PY_310,
)
handler_binop_or = register_handler(
    "BINARY_OR",
    functools.partial(handle_binop, ast.BitOr()),
    version=PythonVersion.PY_310,
)
handler_binop_xor = register_handler(
    "BINARY_XOR",
    functools.partial(handle_binop, ast.BitXor()),
    version=PythonVersion.PY_310,
)
handler_binop_and = register_handler(
    "BINARY_AND",
    functools.partial(handle_binop, ast.BitAnd()),
    version=PythonVersion.PY_310,
)


# ============================================================================
# UNARY OPERATION HANDLERS
# ============================================================================


def handle_unary_op(
    op: ast.unaryop, state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    operand = ensure_ast(state.stack[-1])
    new_stack = state.stack[:-1] + [ast.UnaryOp(op=op, operand=operand)]
    return replace(state, stack=new_stack)


handle_unary_negative = register_handler(
    "UNARY_NEGATIVE", functools.partial(handle_unary_op, ast.USub())
)
handle_unary_positive = register_handler(
    "UNARY_POSITIVE",
    functools.partial(handle_unary_op, ast.UAdd()),
    version=PythonVersion.PY_310,
)
handle_unary_invert = register_handler(
    "UNARY_INVERT", functools.partial(handle_unary_op, ast.Invert())
)
handle_unary_not = register_handler(
    "UNARY_NOT", functools.partial(handle_unary_op, ast.Not())
)


@register_handler("LIST_TO_TUPLE", version=PythonVersion.PY_310)
def handle_list_to_tuple(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LIST_TO_TUPLE converts a list on the stack to a tuple
    list_obj = ensure_ast(state.stack[-1])
    assert isinstance(list_obj, ast.List), "Expected a list for LIST_TO_TUPLE"

    # Create tuple AST from the list's elements
    tuple_node = ast.Tuple(elts=list_obj.elts, ctx=ast.Load())
    new_stack = state.stack[:-1] + [tuple_node]
    return replace(state, stack=new_stack)


@register_handler("CALL_INTRINSIC_1", version=PythonVersion.PY_313)
def handle_call_intrinsic_1(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # CALL_INTRINSIC_1 calls an intrinsic function with one argument
    if instr.argrepr == "INTRINSIC_STOPITERATION_ERROR":
        return state
    elif instr.argrepr == "INTRINSIC_LIST_TO_TUPLE":
        assert isinstance(state.stack[-1], ast.List), (
            "Expected a list for LIST_TO_TUPLE"
        )
        tuple_node = ast.Tuple(elts=state.stack[-1].elts, ctx=ast.Load())
        return replace(state, stack=state.stack[:-1] + [tuple_node])
    elif instr.argrepr == "INTRINSIC_UNARY_POSITIVE":
        assert len(state.stack) > 0
        new_val = ast.UnaryOp(op=ast.UAdd(), operand=state.stack[-1])
        return replace(state, stack=state.stack[:-1] + [new_val])
    else:
        raise TypeError(f"Unsupported generator intrinsic operation: {instr.argrepr}")


@register_handler("TO_BOOL", version=PythonVersion.PY_313)
def handle_to_bool(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # TO_BOOL converts the top stack item to a boolean
    # For AST reconstruction, we typically don't need an explicit bool() call
    # since the boolean context is usually handled by the conditional jump that follows
    # However, for some cases we might need to preserve the explicit conversion

    # For now, leave the value as-is since the jump instruction will handle the boolean logic
    return state


# ============================================================================
# COMPARISON OPERATION HANDLERS
# ============================================================================

CMP_OPMAP: dict[str, ast.cmpop] = {
    "<": ast.Lt(),
    "<=": ast.LtE(),
    ">": ast.Gt(),
    ">=": ast.GtE(),
    "==": ast.Eq(),
    "!=": ast.NotEq(),
}


@register_handler("COMPARE_OP")
def handle_compare_op(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    assert instr.arg is not None and instr.argval in dis.cmp_op, (
        f"Unsupported comparison operation: {instr.argval}"
    )

    right = ensure_ast(state.stack[-1])
    left = ensure_ast(state.stack[-2])

    # Map comparison operation codes to AST operators
    op_name = instr.argval
    compare_node = ast.Compare(left=left, ops=[CMP_OPMAP[op_name]], comparators=[right])
    new_stack = state.stack[:-2] + [compare_node]
    return replace(state, stack=new_stack)


@register_handler("CONTAINS_OP")
def handle_contains_op(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    right = ensure_ast(state.stack[-1])  # Container
    left = ensure_ast(state.stack[-2])  # Item to check

    # instr.arg determines if it's 'in' (0) or 'not in' (1)
    op = ast.NotIn() if instr.arg else ast.In()

    compare_node = ast.Compare(left=left, ops=[op], comparators=[right])
    new_stack = state.stack[:-2] + [compare_node]
    return replace(state, stack=new_stack)


@register_handler("IS_OP")
def handle_is_op(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    right = ensure_ast(state.stack[-1])
    left = ensure_ast(state.stack[-2])

    # instr.arg determines if it's 'is' (0) or 'is not' (1)
    op = ast.IsNot() if instr.arg else ast.Is()

    compare_node = ast.Compare(left=left, ops=[op], comparators=[right])
    new_stack = state.stack[:-2] + [compare_node]
    return replace(state, stack=new_stack)


# ============================================================================
# FUNCTION CALL HANDLERS
# ============================================================================


@register_handler("CALL", version=PythonVersion.PY_313)
def handle_call(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # CALL pops function and arguments from stack (replaces CALL_FUNCTION in Python 3.13)
    assert instr.arg is not None
    arg_count: int = instr.arg

    func = ensure_ast(state.stack[-arg_count - 2])

    # Pop arguments and function
    args = (
        [ensure_ast(arg) for arg in state.stack[-arg_count:]] if arg_count > 0 else []
    )
    if not isinstance(state.stack[-arg_count - 1], Null):
        args = [ensure_ast(state.stack[-arg_count - 1])] + args

    new_stack = state.stack[: -arg_count - 2]
    if isinstance(func, CompLambda):
        assert len(args) == 1
        return replace(state, stack=new_stack + [func.inline(args[0])])
    else:
        # Create function call AST
        call_node = ast.Call(func=func, args=args, keywords=[])
        new_stack = new_stack + [call_node]
        return replace(state, stack=new_stack)


@register_handler("CALL_FUNCTION", version=PythonVersion.PY_310)
def handle_call_function(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # CALL_FUNCTION pops function and arguments from stack
    assert instr.arg is not None
    arg_count: int = instr.arg
    # Pop arguments and function
    args = (
        [ensure_ast(arg) for arg in state.stack[-arg_count:]] if arg_count > 0 else []
    )
    func = ensure_ast(state.stack[-arg_count - 1])
    new_stack = state.stack[: -arg_count - 1]

    if isinstance(func, CompLambda):
        assert len(args) == 1
        return replace(state, stack=new_stack + [func.inline(args[0])])
    else:
        # Create function call AST
        call_node = ast.Call(func=func, args=args, keywords=[])
        new_stack = new_stack + [call_node]
        return replace(state, stack=new_stack)


@register_handler("CALL_METHOD", version=PythonVersion.PY_310)
def handle_call_method(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # CALL_METHOD calls a method - similar to CALL_FUNCTION but for methods
    assert instr.arg is not None
    arg_count: int = instr.arg
    # Pop arguments and method
    args = (
        [ensure_ast(arg) for arg in state.stack[-arg_count:]] if arg_count > 0 else []
    )
    method = ensure_ast(state.stack[-arg_count - 2])
    new_stack = state.stack[: -arg_count - 2]

    # Create method call AST
    call_node = ast.Call(func=method, args=args, keywords=[])
    new_stack = new_stack + [call_node]
    return replace(state, stack=new_stack)


# Python 3.10 version
@register_handler("MAKE_FUNCTION", version=PythonVersion.PY_310)
def handle_make_function_310(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # MAKE_FUNCTION creates a function from code object and name on stack
    assert isinstance(state.stack[-2], ast.Lambda | CompLambda)
    assert isinstance(state.stack[-1], ast.Constant) and isinstance(
        state.stack[-1].value, str
    ), "Function name must be a constant string."
    if instr.argrepr == "closure":
        # This is a closure, remove the environment tuple from the stack for AST purposes
        new_stack = state.stack[:-3]
    elif instr.argrepr == "":
        new_stack = state.stack[:-2]
    else:
        raise NotImplementedError(
            "MAKE_FUNCTION with defaults or annotations not implemented."
        )

    # Pop the function object and name from the stack
    # Conversion from CodeType to ast.Lambda should have happened already
    func: ast.Lambda | CompLambda = state.stack[-2]
    name: str = state.stack[-1].value

    assert any(
        name.endswith(suffix)
        for suffix in ("<genexpr>", "<lambda>", "<dictcomp>", "<listcomp>", "<setcomp>")
    ), f"Expected a comprehension or lambda function, got '{name}'"
    return replace(state, stack=new_stack + [func])


# Python 3.13 version
@register_handler("MAKE_FUNCTION", version=PythonVersion.PY_313)
def handle_make_function(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # MAKE_FUNCTION in Python 3.13 is simplified: it only takes a code object from the stack
    # and creates a function from it. No flags, no extra attributes on the stack.
    # All extra attributes are handled by separate SET_FUNCTION_ATTRIBUTE instructions.

    # Pop the function object from the stack (it's the only thing expected)
    # Conversion from CodeType to ast.Lambda should have happened already
    assert isinstance(state.stack[-1], ast.Lambda | CompLambda), (
        "Expected a function object (Lambda or CompLambda) on the stack."
    )
    return state


@register_handler("SET_FUNCTION_ATTRIBUTE", version=PythonVersion.PY_313)
def handle_set_function_attribute(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # SET_FUNCTION_ATTRIBUTE sets an attribute on a function object
    # In Python 3.13, this is used instead of flags in MAKE_FUNCTION
    # For AST reconstruction, we typically don't need to track function attributes
    # Just pop the attribute value and leave the function on the stack

    # Pop the attribute value but keep the function
    new_stack = state.stack[:-2] + [state.stack[-1]]  # Keep the function on top
    return replace(state, stack=new_stack)


# ============================================================================
# OBJECT ACCESS HANDLERS
# ============================================================================


@register_handler("LOAD_METHOD", version=PythonVersion.PY_310)
def handle_load_method(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LOAD_METHOD loads a method from an object
    # It pushes the bound method and the object (for the method call)
    obj = ensure_ast(state.stack[-1])
    method_name = instr.argval
    new_stack = state.stack[:-1]

    # Create method access as an attribute
    method_attr = ast.Attribute(value=obj, attr=method_name, ctx=ast.Load())

    # For LOAD_METHOD, we push both the method and the object
    # But for AST purposes, we just need the method attribute
    new_stack = new_stack + [method_attr, obj]
    return replace(state, stack=new_stack)


@register_handler("LOAD_ATTR", version=PythonVersion.PY_310)
def handle_load_attr_310(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LOAD_ATTR loads an attribute from the object on top of stack
    obj = ensure_ast(state.stack[-1])
    attr_name = instr.argval

    # Create attribute access AST
    attr_node = ast.Attribute(value=obj, attr=attr_name, ctx=ast.Load())
    new_stack = state.stack[:-1] + [attr_node]
    return replace(state, stack=new_stack)


@register_handler("LOAD_ATTR", version=PythonVersion.PY_313)
def handle_load_attr(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LOAD_ATTR loads an attribute from the object on top of stack
    obj = ensure_ast(state.stack[-1])
    attr_name = instr.argval

    # Create attribute access AST
    attr_node = ast.Attribute(value=obj, attr=attr_name, ctx=ast.Load())
    if instr.argrepr.endswith(" + NULL|self"):
        new_stack = state.stack[:-1] + [attr_node, Null()]
    else:
        new_stack = state.stack[:-1] + [attr_node]
    return replace(state, stack=new_stack)


@register_handler("BINARY_SUBSCR")
def handle_binary_subscr(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # BINARY_SUBSCR implements obj[index] - pops index and obj from stack
    index = ensure_ast(state.stack[-1])  # Index is on top
    obj = ensure_ast(state.stack[-2])  # Object is below index
    new_stack = state.stack[:-2]

    # Create subscript access AST
    subscr_node = ast.Subscript(value=obj, slice=index, ctx=ast.Load())
    new_stack = new_stack + [subscr_node]
    return replace(state, stack=new_stack)


# ============================================================================
# OTHER CONTAINER BUILDING HANDLERS
# ============================================================================


@register_handler("UNPACK_SEQUENCE")
def handle_unpack_sequence(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # UNPACK_SEQUENCE unpacks a sequence into multiple values
    # arg is the number of values to unpack
    assert instr.arg is not None
    unpack_count: int = instr.arg
    sequence = ensure_ast(state.stack[-1])  # noqa: F841
    new_stack = state.stack[:-1]

    # For tuple unpacking in comprehensions, we typically see patterns like:
    # ((k, v) for k, v in items) where items is unpacked into k and v
    # Create placeholder variables for the unpacked values
    for i in range(unpack_count):
        var_name = f"_unpack_{i}"
        new_stack = new_stack + [ast.Name(id=var_name, ctx=ast.Load())]

    return replace(state, stack=new_stack)


@register_handler("BUILD_TUPLE")
def handle_build_tuple(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    assert instr.arg is not None
    tuple_size: int = instr.arg
    # Pop elements for the tuple
    elements = (
        [ensure_ast(elem) for elem in state.stack[-tuple_size:]]
        if tuple_size > 0
        else []
    )
    new_stack = state.stack[:-tuple_size] if tuple_size > 0 else state.stack

    # Create tuple AST
    tuple_node = ast.Tuple(elts=elements, ctx=ast.Load())
    new_stack = new_stack + [tuple_node]
    return replace(state, stack=new_stack)


@register_handler("BUILD_CONST_KEY_MAP")
def handle_build_const_key_map(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # BUILD_CONST_KEY_MAP builds a dictionary with constant keys
    # The keys are in a tuple on TOS, values are on the stack below
    assert instr.arg is not None
    assert isinstance(state.stack[-1], ast.Tuple), "Expected a tuple of keys"
    map_size: int = instr.arg
    # Pop the keys tuple and values
    keys_tuple: ast.Tuple = state.stack[-1]
    keys: list[ast.expr | None] = [ensure_ast(key) for key in keys_tuple.elts]
    values = [ensure_ast(val) for val in state.stack[-map_size - 1 : -1]]
    new_stack = state.stack[: -map_size - 1]

    # Create dictionary AST
    dict_node = ast.Dict(keys=keys, values=values)
    new_stack = new_stack + [dict_node]
    return replace(state, stack=new_stack)


@register_handler("LIST_EXTEND")
def handle_list_extend(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LIST_EXTEND extends the list at TOS-1 with the iterable at TOS
    # initially recognized as list comp

    # The list being extended is actually in state.result instead of the stack
    # because it was initially recognized as a list comprehension in BUILD_LIST,
    # while the actual result expression is in the stack where the list "should be"
    # and needs to be put back into the state result slot
    assert isinstance(state.result, ast.ListComp) and not state.result.generators
    assert isinstance(state.stack[-1], ast.Tuple | ast.List)
    prev_result = state.stack[-instr.argval - 1]

    new_val = ast.List(
        elts=[ensure_ast(e) for e in state.stack[-1].elts], ctx=ast.Load()
    )
    new_stack = state.stack[:-2] + [new_val]

    return replace(state, stack=new_stack, result=prev_result)


@register_handler("SET_UPDATE")
def handle_set_update(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # The set being extended is actually in state.result instead of the stack
    # because it was initially recognized as a list comprehension in BUILD_SET,
    # while the actual result expression is in the stack where the set "should be"
    # and needs to be put back into the state result slot
    assert isinstance(state.result, ast.SetComp) and not state.result.generators
    assert isinstance(state.stack[-1], ast.Tuple | ast.List | ast.Set)
    prev_result = state.stack[-instr.argval - 1]

    new_val = ast.Set(elts=[ensure_ast(e) for e in state.stack[-1].elts])
    new_stack = state.stack[:-2] + [new_val]

    return replace(state, stack=new_stack, result=prev_result)


@register_handler("DICT_UPDATE")
def handle_dict_update(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # The dict being extended is actually in state.result instead of the stack
    # because it was initially recognized as a list comprehension in BUILD_MAP,
    # while the actual result expression is in the stack where the dict "should be"
    # and needs to be put back into the state result slot
    assert isinstance(state.result, ast.DictComp) and not state.result.generators
    assert isinstance(state.stack[-1], ast.Dict)
    prev_result = state.stack[-instr.argval - 1]

    new_val = ast.Dict(
        keys=[ensure_ast(e) for e in state.stack[-1].keys],
        values=[ensure_ast(e) for e in state.stack[-1].values],
    )
    new_stack = state.stack[:-2] + [new_val]

    return replace(state, stack=new_stack, result=prev_result)


# ============================================================================
# CONDITIONAL JUMP HANDLERS
# ============================================================================


# Python 3.10 version
@register_handler("POP_JUMP_IF_FALSE", version=PythonVersion.PY_310)
def handle_pop_jump_if_false_310(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # POP_JUMP_IF_FALSE pops a value from the stack and jumps if it's false
    # In comprehensions, this is used for filter conditions
    condition = ensure_ast(state.stack[-1])
    new_stack = state.stack[:-1]

    if instr.argval < instr.offset:
        # Jumping backward to loop start - this is a condition
        # When POP_JUMP_IF_FALSE jumps back, it means "if false, skip this item"
        assert isinstance(state.result, CompExp) and state.result.generators
        new_result = copy.deepcopy(state.result)
        new_result.generators[-1].ifs.append(condition)
        return replace(state, stack=new_stack, result=new_result)
    else:
        raise NotImplementedError("Lazy and+or behavior not implemented yet")


# Python 3.13 version
@register_handler("POP_JUMP_IF_FALSE", version=PythonVersion.PY_313)
def handle_pop_jump_if_false(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # POP_JUMP_IF_FALSE pops a value from the stack and jumps if it's false
    # In comprehensions, this is used for filter conditions
    condition = ensure_ast(state.stack[-1])
    new_stack = state.stack[:-1]

    if isinstance(state.result, CompExp) and state.result.generators:
        # In Python 3.13, when POP_JUMP_IF_FALSE jumps forward to the yield,
        # it means "if condition is False, then yield the item"
        # So we need to negate the condition: we want items where NOT condition
        negated_condition = ast.UnaryOp(op=ast.Not(), operand=condition)
        new_result = copy.deepcopy(state.result)
        new_result.generators[-1].ifs.append(negated_condition)
        return replace(state, stack=new_stack, result=new_result)
    else:
        # Not in a comprehension context - might be boolean logic
        raise NotImplementedError("Lazy and+or behavior not implemented yet")


# Python 3.10 version
@register_handler("POP_JUMP_IF_TRUE", version=PythonVersion.PY_310)
def handle_pop_jump_if_true_310(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # POP_JUMP_IF_TRUE pops a value from the stack and jumps if it's true
    # This can be:
    # 1. Part of an OR expression (jump to YIELD_VALUE)
    # 2. A negated condition like "not x % 2" (jump back to loop start)
    condition = ensure_ast(state.stack[-1])
    new_stack = state.stack[:-1]

    if instr.argval < instr.offset:
        # Jumping backward to loop start - this is a negated condition
        # When POP_JUMP_IF_TRUE jumps back, it means "if false, skip this item"
        # So we need to negate the condition to get the filter condition
        assert isinstance(state.result, CompExp) and state.result.generators
        negated_condition = ast.UnaryOp(op=ast.Not(), operand=condition)
        new_result = copy.deepcopy(state.result)
        new_result.generators[-1].ifs.append(negated_condition)
        return replace(state, stack=new_stack, result=new_result)
    else:
        raise NotImplementedError("Lazy and+or behavior not implemented yet")


# Python 3.13 version
@register_handler("POP_JUMP_IF_TRUE", version=PythonVersion.PY_313)
def handle_pop_jump_if_true(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # POP_JUMP_IF_TRUE pops a value from the stack and jumps if it's true
    # In Python 3.13, this is used for filter conditions where True means continue
    condition = ensure_ast(state.stack[-1])
    new_stack = state.stack[:-1]

    # In Python 3.13, if we have a comprehension and generators, this is likely a filter
    if isinstance(state.result, CompExp) and state.result.generators:
        # For POP_JUMP_IF_TRUE in filters, we want the condition to be true to continue
        # So we add the condition directly (no negation needed)
        new_result = copy.deepcopy(state.result)
        new_result.generators[-1].ifs.append(condition)
        return replace(state, stack=new_stack, result=new_result)
    else:
        # Not in a comprehension context - might be boolean logic
        raise NotImplementedError("Lazy and+or behavior not implemented yet")


@register_handler("POP_JUMP_IF_NONE", version=PythonVersion.PY_313)
def handle_pop_jump_if_none(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # POP_JUMP_IF_NONE pops a value and jumps if it's None
    condition = ensure_ast(state.stack[-1])
    new_stack = state.stack[:-1]

    if isinstance(state.result, CompExp) and state.result.generators:
        # Create "x is None" condition
        none_condition = ast.Compare(
            left=condition, ops=[ast.Is()], comparators=[ast.Constant(value=None)]
        )
        new_result = copy.deepcopy(state.result)
        new_result.generators[-1].ifs.append(none_condition)
        return replace(state, stack=new_stack, result=new_result)
    else:
        raise NotImplementedError("Lazy and+or behavior not implemented yet")


@register_handler("POP_JUMP_IF_NOT_NONE", version=PythonVersion.PY_313)
def handle_pop_jump_if_not_none(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # POP_JUMP_IF_NOT_NONE pops a value and jumps if it's not None
    condition = ensure_ast(state.stack[-1])
    new_stack = state.stack[:-1]

    if isinstance(state.result, CompExp) and state.result.generators:
        # Create "x is not None" condition
        not_none_condition = ast.Compare(
            left=condition, ops=[ast.IsNot()], comparators=[ast.Constant(value=None)]
        )
        new_result = copy.deepcopy(state.result)
        new_result.generators[-1].ifs.append(not_none_condition)
        return replace(state, stack=new_stack, result=new_result)
    else:
        raise NotImplementedError("Lazy and+or behavior not implemented yet")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


@functools.singledispatch
def ensure_ast(value) -> ast.expr:
    """Ensure value is an AST node"""
    raise TypeError(f"Cannot convert {type(value)} to AST node")


@ensure_ast.register
def _ensure_ast_ast(value: ast.expr) -> ast.expr:
    """If already an AST node, return it as is"""
    return value


@ensure_ast.register(int)
@ensure_ast.register(float)
@ensure_ast.register(str)
@ensure_ast.register(bytes)
@ensure_ast.register(bool)
@ensure_ast.register(complex)
@ensure_ast.register(type(None))
def _ensure_ast_constant(value) -> ast.Constant:
    return ast.Constant(value=value)


@ensure_ast.register
def _ensure_ast_tuple(value: tuple) -> ast.Tuple:
    """Convert tuple to AST - special handling for dict items"""
    if len(value) > 0 and value[0] == "dict_item":
        return ast.Tuple(
            elts=[ensure_ast(value[1]), ensure_ast(value[2])], ctx=ast.Load()
        )
    else:
        return ast.Tuple(elts=[ensure_ast(v) for v in value], ctx=ast.Load())


@ensure_ast.register(type(iter((1,))))
def _ensure_ast_tuple_iterator(value: Iterator) -> ast.Tuple:
    return ensure_ast(tuple(value.__reduce__()[1][0]))  # type: ignore


@ensure_ast.register
def _ensure_ast_list(value: list) -> ast.List:
    return ast.List(elts=[ensure_ast(v) for v in value], ctx=ast.Load())


@ensure_ast.register(type(iter([1])))
def _ensure_ast_list_iterator(value: Iterator) -> ast.List:
    return ensure_ast(list(value.__reduce__()[1][0]))  # type: ignore


@ensure_ast.register(set)
@ensure_ast.register(frozenset)
def _ensure_ast_set(value: set | frozenset) -> ast.Set:
    return ast.Set(elts=[ensure_ast(v) for v in value])


@ensure_ast.register(type(iter({1})))
def _ensure_ast_set_iterator(value: Iterator) -> ast.Set:
    return ensure_ast(set(value.__reduce__()[1][0]))  # type: ignore


@ensure_ast.register
def _ensure_ast_dict(value: dict) -> ast.Dict:
    return ast.Dict(
        keys=[ensure_ast(k) for k in value.keys()],
        values=[ensure_ast(v) for v in value.values()],
    )


@ensure_ast.register(type(iter({1: 2})))
def _ensure_ast_dict_iterator(value: Iterator) -> ast.expr:
    return ensure_ast(value.__reduce__()[1][0])


@ensure_ast.register
def _ensure_ast_range(value: range) -> ast.Call:
    return ast.Call(
        func=ast.Name(id="range", ctx=ast.Load()),
        args=[ensure_ast(value.start), ensure_ast(value.stop), ensure_ast(value.step)],
        keywords=[],
    )


@ensure_ast.register(type(iter(range(1))))
def _ensure_ast_range_iterator(value: Iterator) -> ast.Call:
    return ensure_ast(value.__reduce__()[1][0])  # type: ignore


@ensure_ast.register
def _ensure_ast_codeobj(value: types.CodeType) -> ast.Lambda | CompLambda:
    assert inspect.iscode(value), "Input must be a code object"

    name: str = value.co_name.split(".")[-1]

    # Check preconditions
    if name in {"<genexpr>", "<dictcomp>", "<listcomp>", "<setcomp>"}:
        assert name == "<genexpr>" or sys.version_info < (3, 13)
        assert name != "<genexpr>" or value.co_flags & inspect.CO_GENERATOR
        assert value.co_flags & inspect.CO_NEWLOCALS
        assert value.co_argcount == 1
        assert value.co_kwonlyargcount == value.co_posonlyargcount == 0
        assert DummyIterName().id in value.co_varnames
    elif name == "<lambda>":
        assert not value.co_flags & inspect.CO_GENERATOR
        assert value.co_flags & inspect.CO_NEWLOCALS
        assert DummyIterName().id not in value.co_varnames
    else:
        raise TypeError(f"Unsupported code object type: {value.co_name}")

    # Symbolic execution to reconstruct the AST
    state = ReconstructionState()
    for instr in dis.get_instructions(value):
        state = OP_HANDLERS[instr.opname](state, instr)
    result: ast.expr = state.result

    # Check postconditions
    assert not any(isinstance(x, ast.stmt) for x in ast.walk(result)), (
        "Final return value must not contain statement nodes"
    )
    assert not any(
        isinstance(x, Placeholder | Null | CompLambda) for x in ast.walk(result)
    ), "Final return value must not contain temporary nodes"
    assert not any(x.arg == ".0" for x in ast.walk(result) if isinstance(x, ast.arg)), (
        "Final return value must not contain .0 argument"
    )
    assert not any(
        isinstance(x, ast.Name) and x.id == ".0"
        for x in ast.walk(result)
        if not isinstance(x, DummyIterName)
    ), "Final return value must not contain .0 names"
    assert sum(1 for x in ast.walk(result) if isinstance(x, DummyIterName)) <= 1, (
        "Final return value must contain at most 1 dummy iterator names"
    )
    assert all(x.generators for x in ast.walk(result) if isinstance(x, CompExp)), (
        "Return value must have generators if not a lambda"
    )

    if name == "<lambda>" and isinstance(result, ast.expr):
        args = ast.arguments(
            posonlyargs=[
                ast.arg(arg=arg)
                for arg in value.co_varnames[: value.co_posonlyargcount]
            ],
            args=[
                ast.arg(arg=arg)
                for arg in value.co_varnames[
                    value.co_posonlyargcount : value.co_argcount
                ]
            ],
            kwonlyargs=[
                ast.arg(arg=arg)
                for arg in value.co_varnames[
                    value.co_argcount : value.co_argcount + value.co_kwonlyargcount
                ]
            ],
            kw_defaults=[],
            defaults=[],
        )
        return ast.Lambda(args=args, body=result)
    elif name == "<genexpr>" and isinstance(result, ast.GeneratorExp):
        return CompLambda(body=result)
    elif name == "<dictcomp>" and isinstance(result, ast.DictComp):
        return CompLambda(body=result)
    elif name == "<listcomp>" and isinstance(result, ast.ListComp):
        return CompLambda(body=result)
    elif name == "<setcomp>" and isinstance(result, ast.SetComp):
        return CompLambda(body=result)
    else:
        raise TypeError(f"Invalid result for type {name}: {result}")


@ensure_ast.register
def _ensure_ast_lambda(value: types.LambdaType) -> ast.Lambda:
    assert inspect.isfunction(value) and value.__name__.endswith("<lambda>"), (
        "Input must be a lambda function"
    )
    code: types.CodeType = value.__code__
    result = ensure_ast(code)
    assert isinstance(result, ast.Lambda), "Lambda body must be an AST Lambda node"
    assert not isinstance(result, CompLambda), "Lambda must not be a CompLambda"
    return result


@ensure_ast.register
def _ensure_ast_genexpr(genexpr: types.GeneratorType) -> ast.GeneratorExp:
    assert inspect.isgenerator(genexpr), "Input must be a generator expression"
    assert inspect.getgeneratorstate(genexpr) == inspect.GEN_CREATED, (
        "Generator must be in created state"
    )
    genexpr_ast = ensure_ast(genexpr.gi_code)
    assert isinstance(genexpr_ast, CompLambda)
    geniter_ast = ensure_ast(genexpr.gi_frame.f_locals[".0"])
    result = genexpr_ast.inline(geniter_ast)
    assert isinstance(result, ast.GeneratorExp)
    assert inspect.getgeneratorstate(genexpr) == inspect.GEN_CREATED, (
        "Generator must stay in created state"
    )
    return result


# ============================================================================
# MAIN RECONSTRUCTION FUNCTION
# ============================================================================


def reconstruct(genexpr: Generator[object, None, None]) -> ast.Expression:
    """
    Reconstruct an AST from a generator expression's bytecode.

    This function analyzes the bytecode of a generator object and reconstructs
    an abstract syntax tree (AST) that represents the original comprehension
    expression. The reconstruction process simulates the Python VM's execution
    of the bytecode, building AST nodes instead of executing operations.

    The reconstruction handles complex comprehension features including:
    - Multiple nested loops
    - Filter conditions (if clauses)
    - Complex expressions in the yield/result part
    - Tuple unpacking in loop variables
    - Various operators and function calls

    Args:
        genexpr (Generator[object, None, None]): The generator object to analyze.
            Must be a freshly created generator that has not been iterated yet
            (in 'GEN_CREATED' state).

    Returns:
        ast.Expression: An AST node representing the reconstructed comprehension.

    Raises:
        AssertionError: If the input is not a generator or if the generator
            has already been started (not in 'GEN_CREATED' state).

    Example:
        >>> # Generator expression
        >>> g = (x * 2 for x in range(10) if x % 2 == 0)
        >>> ast_node = reconstruct(g)
        >>> isinstance(ast_node, ast.Expression)
        True

        >>> # The reconstructed AST can be compiled and evaluated
        >>> import ast
        >>> code = compile(ast_node, '<string>', 'eval')
        >>> result = eval(code)
        >>> list(result)
        [0, 4, 8, 12, 16]

    Note:
        The reconstruction is based on bytecode analysis and may not perfectly
        preserve the original source code formatting or variable names in all
        cases. However, the semantic behavior of the reconstructed AST should
        match the original comprehension.
    """
    assert inspect.isgenerator(genexpr), "Input must be a generator expression"
    return ast.fix_missing_locations(ast.Expression(ensure_ast(genexpr)))


class NameToCall(ast.NodeTransformer):
    """
    Transform variable names into calls to those variables.
    This transformer replaces occurrences of specified variable names in an AST
    with calls to those variables. For example, if the variable name is 'x',
    it will replace 'x' with 'x()'.
    """

    varnames: set[str]

    def __init__(self, varnames: set[str]):
        self.varnames = varnames

    def visit_Name(self, node: ast.Name) -> ast.Call | ast.Name:
        if node.id in self.varnames and isinstance(node.ctx, ast.Load):
            return ast.Call(node, args=[], keywords=[])
        else:
            return node


class GeneratorExpToForexpr(ast.NodeTransformer):
    """
    Transform generator expressions into calls to `forexpr`.
    This transformer converts generator expressions of the form:

        (expr for var in iter)
    into calls to `forexpr`:
        forexpr(expr, {var: lambda: iter})

    It supports:
    - Multiple nested loops
    - Filter conditions (if clauses) - converted to filtered generator expressions
    - Tuple unpacking in loop variables
    - Variables are correctly transformed into calls within the expression and iterators

    Examples:
        >>> import ast
        >>> source = "(x * 2 for x in range(10))"
        >>> tree = ast.parse(source, mode='eval')
        >>> transformer = GeneratorExpToForexpr()
        >>> transformed = transformer.visit(tree)
        >>> ast.unparse(transformed)
        'forexpr(x() * 2, {x: lambda: range(10)})'

        >>> source = "(x for x in range(10) if x % 2 == 0)"
        >>> tree = ast.parse(source, mode='eval')
        >>> transformed = transformer.visit(tree)
        >>> ast.unparse(transformed)
        'forexpr(x(), {x: (x for x in range(10) if x % 2 == 0)})'

        >>> source = "((x, y) for x, y in pairs)"
        >>> tree = ast.parse(source, mode='eval')
        >>> transformed = transformer.visit(tree)
        >>> ast.unparse(transformed)
        'forexpr((x(), y()), {(x, y): lambda: pairs})'

    """

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.Call:
        # Check for unsupported features
        for gen in node.generators:
            if not isinstance(gen.target, ast.Name) and not isinstance(
                gen.target, ast.Tuple
            ):
                raise NotImplementedError(
                    f"Unsupported target type: {type(gen.target)}"
                )

        # Get all variable names from all targets (including unpacked tuples)
        def get_names_from_target(target):
            if isinstance(target, ast.Name):
                return [target.id]
            elif isinstance(target, ast.Tuple):
                names = []
                for elt in target.elts:
                    names.extend(get_names_from_target(elt))
                return names
            else:
                raise NotImplementedError(
                    f"Unsupported target type in unpacking: {type(target)}"
                )

        streams = ast.Dict(keys=[], values=[])
        all_var_names: set[str] = set()

        for gen in node.generators:
            # Collect variable names from previous generators
            prev_var_names = set(all_var_names)

            # Add current target variables to the set
            target_names = get_names_from_target(gen.target)
            all_var_names.update(target_names)

            # Create the value for this generator
            value: ast.expr  # TODO : Specify type more precisely
            if gen.ifs:
                # If there are filters, create a generator expression for the filtered iterator
                # Note: In the filter conditions, we need to transform previous loop variables
                # but NOT the current loop variable
                filtered_gen = ast.GeneratorExp(
                    elt=gen.target if isinstance(gen.target, ast.Name) else gen.target,
                    generators=[
                        ast.comprehension(
                            target=gen.target,
                            iter=self.visit(NameToCall(prev_var_names).visit(gen.iter)),
                            ifs=[
                                self.visit(NameToCall(prev_var_names).visit(if_clause))
                                for if_clause in gen.ifs
                            ],
                            is_async=gen.is_async,
                        )
                    ],
                )
                value = filtered_gen
            else:
                # No filters, create a lambda
                value = ast.Lambda(
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=self.visit(NameToCall(prev_var_names).visit(gen.iter)),
                )

            streams.keys.append(gen.target)
            streams.values.append(value)

        # Transform the body expression
        # First apply NameToCall, then recursively visit for nested generators
        body: ast.expr = NameToCall(all_var_names).visit(node.elt)
        body = self.visit(body)  # Recursively transform nested generator expressions

        return ast.Call(
            func=ast.Name(id="forexpr", ctx=ast.Load()),
            args=[body, streams],
            keywords=[],
        )
