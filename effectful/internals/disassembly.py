"""
Generator expression bytecode reconstruction module.

This module provides functionality to reconstruct AST representations from compiled
generator expressions by analyzing their bytecode. The primary use case is to recover
the original structure of generator comprehensions from their compiled form.

The only public-facing interface is the `disassemble()` function, which takes a
generator object and returns an AST node representing the original comprehension.
All other functions and classes in this module are internal implementation details.

Example:
    >>> g = (x * 2 for x in range(10) if x % 2 == 0)
    >>> ast_node = disassemble(g)
    >>> # ast_node is now an ast.Expression representing the original expression
"""

import ast
import collections
import collections.abc
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


class Skipped(ast.Name):
    """Placeholder for skipped branches in if-expressions."""

    def __init__(self, id: str, ctx=ast.Load()):
        super().__init__(id=id, ctx=ctx)


class Null(ast.Constant):
    """Placeholder for NULL values generated in bytecode."""

    def __init__(self, value=None):
        super().__init__(value=value)


class ConvertedValue(ast.expr):
    """Wrapper for values that have been converted with CONVERT_VALUE."""

    def __init__(self, value: ast.expr, conversion: int):
        self.value = value
        self.conversion = conversion
        # Map CONVERT_VALUE args to ast.FormattedValue conversion values
        # CONVERT_VALUE: 0=None, 1=str, 2=repr, 3=ascii
        # ast.FormattedValue: -1=none, 115=str, 114=repr, 97=ascii
        conversion_map = {0: -1, 1: 115, 2: 114, 3: 97}
        self.ast_conversion = conversion_map.get(conversion, -1)


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


class ReplacePlaceholder(ast.NodeTransformer):
    def __init__(self, value: ast.expr):
        self.value = value
        self._done = False
        super().__init__()

    def visit(self, node):
        if isinstance(node, Placeholder) and not self._done:
            self._done = True
            return self.value
        else:
            return self.generic_visit(node)


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
        code: The compiled code object from which the bytecode is being processed.
              This is typically obtained from a generator function or comprehension.

        stack: Simulates the Python VM's value stack. Contains AST nodes or
               values that would be on the stack during execution. Operations
               like LOAD_FAST push to this stack, while operations like
               BINARY_ADD pop operands and push results.
    """

    code: types.CodeType
    instruction: dis.Instruction

    stack: list[ast.expr] = field(default_factory=list)
    result: ast.expr = field(default_factory=Placeholder)

    loops: dict[int, int] = field(default_factory=collections.Counter)
    finished: bool = field(default=False)

    branches: dict[int, int] = field(default_factory=collections.Counter)

    @property
    def instructions(self) -> collections.OrderedDict[int, dis.Instruction]:
        """Get the bytecode instructions for the current code object."""
        return collections.OrderedDict(
            (instr.offset, instr) for instr in dis.get_instructions(self.code)
        )

    @property
    def next_instructions(self) -> collections.abc.Mapping[int, dis.Instruction]:
        instrs_list = list(self.instructions.values())
        return {i1.offset: i2 for i1, i2 in zip(instrs_list[:-1], instrs_list[1:])}

    @property
    def is_filter(self) -> bool:
        """Check if an instruction is a filter clause in a comprehension"""
        return (
            self.instruction.opname in BRANCH_OPS
            and self.next_instructions[self.instruction.offset].opname
            == "JUMP_BACKWARD"
            and self.instructions[
                self.next_instructions[self.instruction.offset].argval
            ].opname
            in LOOP_OPS
        )

    @property
    def is_branch(self) -> bool:
        """Check if an instruction is a branch in an if-expression"""
        return self.instruction.opname in BRANCH_OPS and not self.is_filter


# Python version enum for version-specific handling
class PythonVersion(enum.Enum):
    PY_312 = 12
    PY_313 = 13


# Global handler registry
OpHandler = Callable[[ReconstructionState, dis.Instruction], ReconstructionState]

OP_HANDLERS: dict[str, OpHandler] = {}


@typing.overload
def register_handler(
    opname: str, *, version: PythonVersion
) -> Callable[[OpHandler], OpHandler]: ...


@typing.overload
def register_handler(
    opname: str,
    handler: OpHandler,
    *,
    version: PythonVersion,
) -> OpHandler: ...


def register_handler(
    opname: str,
    handler=None,
    *,
    version: PythonVersion,
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

    if dis.opmap[opname] in dis.hasjrel:
        assert opname in LOOP_OPS | BRANCH_OPS | JUMP_OPS
    else:
        assert opname not in LOOP_OPS | BRANCH_OPS | JUMP_OPS

    @functools.wraps(handler)
    def _wrapper(
        state: ReconstructionState,
        instr: dis.Instruction,
    ) -> ReconstructionState:
        assert instr.opname == opname, (
            f"Handler for '{opname}' called with wrong instruction"
        )
        assert not state.finished, "Cannot process instruction on finished state"

        new_state = handler(state, instr)

        jump: bool | None  # argument to dis.stack_effect
        if instr.opname in LOOP_OPS:
            if state.loops[instr.offset] > 0:
                new_state = replace(
                    new_state, instruction=state.instructions[instr.argval]
                )
                jump = True
            else:
                new_state = replace(
                    new_state, instruction=state.next_instructions[instr.offset]
                )
                new_state.loops[instr.offset] += 1
                jump = False
        elif instr.opname in BRANCH_OPS:
            if state.branches.get(instr.offset, 0):
                new_state = replace(
                    new_state, instruction=state.next_instructions[instr.offset]
                )
                jump = False
            else:
                new_state = replace(
                    new_state, instruction=state.instructions[instr.argval]
                )
                jump = True
        elif instr.opname in JUMP_OPS:
            new_state = replace(new_state, instruction=state.instructions[instr.argval])
            jump = True
        elif instr.opname not in RETURN_OPS and instr.offset in state.next_instructions:
            new_state = replace(
                new_state, instruction=state.next_instructions[instr.offset]
            )
            jump = None
        else:
            new_state = replace(new_state, finished=True)
            jump = None

        # post-condition: check stack effect
        expected_stack_effect = dis.stack_effect(instr.opcode, instr.arg, jump=jump)
        actual_stack_effect = len(new_state.stack) - len(state.stack)
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


LOOP_OPS = {"FOR_ITER"}
BRANCH_OPS = {
    "POP_JUMP_IF_TRUE",
    "POP_JUMP_IF_FALSE",
    "POP_JUMP_IF_NOT_NONE",
    "POP_JUMP_IF_NONE",
}
RETURN_OPS = {"RETURN_VALUE", "RETURN_CONST"}
JUMP_OPS = {dis.opname[d] for d in dis.hasjrel} - LOOP_OPS - BRANCH_OPS - RETURN_OPS


def _symbolic_exec(code: types.CodeType) -> ast.expr:
    """Execute bytecode symbolically, following control flow."""
    continuations: list[ReconstructionState] = [
        ReconstructionState(
            code=code,
            instruction=next(iter(dis.get_instructions(code))),
            stack=[Placeholder(), Placeholder()]
            if PythonVersion(sys.version_info.minor) == PythonVersion.PY_312
            else [Placeholder()],
        )
    ]

    results: list[ast.expr] = []

    while continuations:
        state = continuations.pop()
        while not state.finished:
            if state.is_branch and not state.branches.get(state.instruction.offset, 0):
                continuations.append(
                    replace(
                        state, branches=state.branches | {state.instruction.offset: 1}
                    )
                )
            state = OP_HANDLERS[state.instruction.opname](state, state.instruction)
        results.append(state.result)

    assert results, "No results from symbolic execution"
    return functools.reduce(
        lambda a, b: _MergeBranches(a).visit(b), reversed(results[:-1]), results[-1]
    )


class _MergeBranches(ast.NodeTransformer):
    def __init__(self, node_with_orelse: ast.expr):
        self._orelses = {
            n.body.id: n.orelse
            for n in ast.walk(node_with_orelse)
            if isinstance(n, ast.IfExp) and isinstance(n.body, Skipped)
        }
        assert self._orelses, "No orelse branches to merge"
        super().__init__()

    def visit_IfExp(self, node: ast.IfExp):
        if isinstance(node.orelse, Skipped) and node.orelse.id in self._orelses:
            return ast.IfExp(
                test=node.test, body=node.body, orelse=self._orelses[node.orelse.id]
            )
        else:
            return self.generic_visit(node)


# ============================================================================
# GENERATOR COMPREHENSION HANDLERS
# ============================================================================


@register_handler("RETURN_GENERATOR", version=PythonVersion.PY_312)
def handle_return_generator_312(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # RETURN_GENERATOR is the first instruction in generator expressions in Python 3.13+
    assert len(state.stack) == 2 and all(
        isinstance(x, Null | Placeholder) for x in state.stack
    ), "RETURN_GENERATOR must be the first instruction"
    new_result = ast.GeneratorExp(elt=Placeholder(), generators=[])
    return replace(state, stack=[new_result, Null()])


@register_handler("RETURN_GENERATOR", version=PythonVersion.PY_313)
def handle_return_generator(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # RETURN_GENERATOR is the first instruction in generator expressions in Python 3.13+
    assert len(state.stack) == 1 and isinstance(state.stack[0], Null | Placeholder), (
        "RETURN_GENERATOR must be the first instruction"
    )
    return replace(
        state, stack=[ast.GeneratorExp(elt=Placeholder(), generators=[]), Null()]
    )


@register_handler("YIELD_VALUE", version=PythonVersion.PY_312)
@register_handler("YIELD_VALUE", version=PythonVersion.PY_313)
def handle_yield_value(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # YIELD_VALUE pops a value from the stack and yields it
    # This is the expression part of the generator
    assert isinstance(state.result, Placeholder)
    new_result = copy.deepcopy(state.stack[0])
    assert isinstance(new_result, ast.GeneratorExp), (
        "YIELD_VALUE must be called after RETURN_GENERATOR"
    )
    assert len(new_result.generators) > 0, "YIELD_VALUE should have generators"
    assert any(isinstance(x, Placeholder) for x in ast.walk(new_result.elt))
    new_result.elt = ReplacePlaceholder(ensure_ast(state.stack[-1])).visit(
        new_result.elt
    )
    new_stack = [new_result] + state.stack[1:]
    return replace(state, stack=new_stack, result=new_result)


# ============================================================================
# LIST COMPREHENSION HANDLERS
# ============================================================================


@register_handler("BUILD_LIST", version=PythonVersion.PY_312)
@register_handler("BUILD_LIST", version=PythonVersion.PY_313)
def handle_build_list(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    assert instr.arg is not None
    size: int = instr.arg

    if size == 0:
        # Check if this looks like the start of a list comprehension pattern
        # In nested comprehensions, BUILD_LIST(0) starts a new list comprehe
        new_ret = ast.ListComp(elt=Placeholder(), generators=[])
        new_stack = state.stack + [new_ret]
        return replace(state, stack=new_stack)
    else:
        # BUILD_LIST with elements - create a regular list
        elements = [ensure_ast(elem) for elem in state.stack[-size:]]
        new_stack = state.stack[:-size]
        elt_node = ast.List(elts=elements, ctx=ast.Load())
        new_stack = new_stack + [elt_node]
        return replace(state, stack=new_stack)


@register_handler("LIST_APPEND", version=PythonVersion.PY_312)
@register_handler("LIST_APPEND", version=PythonVersion.PY_313)
def handle_list_append(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    assert isinstance(state.stack[-instr.argval - 1], ast.ListComp)

    # add the body to the comprehension
    comp: ast.ListComp = copy.deepcopy(state.stack[-instr.argval - 1])
    assert any(isinstance(x, Placeholder) for x in ast.walk(comp.elt))
    comp.elt = ReplacePlaceholder(state.stack[-1]).visit(comp.elt)

    # swap the return value
    new_stack = state.stack[:-1]
    new_stack[-instr.argval] = comp

    return replace(state, stack=new_stack)


# ============================================================================
# SET COMPREHENSION HANDLERS
# ============================================================================


@register_handler("BUILD_SET", version=PythonVersion.PY_312)
@register_handler("BUILD_SET", version=PythonVersion.PY_313)
def handle_build_set(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    assert instr.arg is not None
    size: int = instr.arg

    if size == 0:
        new_result = ast.SetComp(elt=Placeholder(), generators=[])
        new_stack = state.stack + [new_result]
        return replace(state, stack=new_stack)
    else:
        elements = [ensure_ast(elem) for elem in state.stack[-size:]]
        new_stack = state.stack[:-size]
        elt_node = ast.Set(elts=elements)
        new_stack = new_stack + [elt_node]
        return replace(state, stack=new_stack)


@register_handler("SET_ADD", version=PythonVersion.PY_312)
@register_handler("SET_ADD", version=PythonVersion.PY_313)
def handle_set_add(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    assert isinstance(state.stack[-instr.argval - 1], ast.SetComp)

    # add the body to the comprehension
    comp: ast.SetComp = copy.deepcopy(state.stack[-instr.argval - 1])
    assert any(isinstance(x, Placeholder) for x in ast.walk(comp.elt))
    comp.elt = ReplacePlaceholder(state.stack[-1]).visit(comp.elt)

    # swap the return value
    new_stack = state.stack[:-1]
    new_stack[-instr.argval] = comp

    return replace(state, stack=new_stack)


# ============================================================================
# DICT COMPREHENSION HANDLERS
# ============================================================================


@register_handler("BUILD_MAP", version=PythonVersion.PY_312)
@register_handler("BUILD_MAP", version=PythonVersion.PY_313)
def handle_build_map(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    assert instr.arg is not None
    size: int = instr.arg

    if size == 0:
        new_result = ast.DictComp(key=Placeholder(), value=Placeholder(), generators=[])
        new_stack = state.stack + [new_result]
        return replace(state, stack=new_stack)
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


@register_handler("MAP_ADD", version=PythonVersion.PY_312)
@register_handler("MAP_ADD", version=PythonVersion.PY_313)
def handle_map_add(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    assert isinstance(state.stack[-instr.argval - 2], ast.DictComp)

    # add the body to the comprehension
    comp: ast.DictComp = copy.deepcopy(state.stack[-instr.argval - 2])
    assert any(isinstance(x, Placeholder) for x in ast.walk(comp.key))
    assert any(isinstance(x, Placeholder) for x in ast.walk(comp.value))
    comp.key = ReplacePlaceholder(state.stack[-2]).visit(comp.key)
    comp.value = ReplacePlaceholder(state.stack[-1]).visit(comp.value)

    # swap the return value
    new_stack = state.stack[:-2]
    new_stack[-instr.argval] = comp

    return replace(state, stack=new_stack)


# ============================================================================
# LOOP CONTROL HANDLERS
# ============================================================================


@register_handler("RETURN_VALUE", version=PythonVersion.PY_312)
@register_handler("RETURN_VALUE", version=PythonVersion.PY_313)
def handle_return_value(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    assert isinstance(state.result, Placeholder)
    assert len(state.stack) == 2
    new_result = ReplacePlaceholder(ensure_ast(state.stack[-1])).visit(state.stack[-2])
    new_stack = state.stack[:-1]
    return replace(state, stack=new_stack, result=new_result)


@register_handler("RETURN_CONST", version=PythonVersion.PY_312)
@register_handler("RETURN_CONST", version=PythonVersion.PY_313)
def handle_return_const(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # RETURN_CONST returns a constant value (replaces some LOAD_CONST + RETURN_VALUE patterns)
    # Similar to RETURN_VALUE but with a constant
    if isinstance(state.result, Placeholder):
        return replace(state, result=ensure_ast(instr.argval))
    else:
        assert instr.argval is None
        return state


@register_handler("FOR_ITER", version=PythonVersion.PY_312)
@register_handler("FOR_ITER", version=PythonVersion.PY_313)
def handle_for_iter(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # FOR_ITER pops an iterator from the stack and pushes the next item
    # If the iterator is exhausted, it jumps to the target instruction
    assert len(state.stack) > 0, "FOR_ITER must have an iterator on the stack"

    if state.loops[instr.offset] > 0:
        return replace(state, stack=state.stack + [Null()])

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

    for pos, item in zip(reversed(range(len(state.stack))), reversed(state.stack)):
        if isinstance(item, CompExp) and isinstance(
            getattr(item, "elt", getattr(item, "key", None)), Placeholder
        ):
            new_result = copy.deepcopy(item)
            new_result.generators.append(loop_info)
            new_stack = (
                state.stack[:pos]
                + [new_result]
                + state.stack[pos + 1 :]
                + [loop_info.target]
            )
            return replace(state, stack=new_stack)

    raise TypeError("FOR_ITER did not find partial comprehension on stack")


@register_handler("GET_ITER", version=PythonVersion.PY_312)
@register_handler("GET_ITER", version=PythonVersion.PY_313)
def handle_get_iter(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # GET_ITER converts the top stack item to an iterator
    # For AST reconstruction, we typically don't need to change anything
    # since the iterator will be used directly in the comprehension
    return state


@register_handler("JUMP_FORWARD", version=PythonVersion.PY_312)
@register_handler("JUMP_FORWARD", version=PythonVersion.PY_313)
def handle_jump_forward(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # JUMP_FORWARD is used to jump forward in the code
    # In generator expressions, this is often used to skip code in conditional logic
    return state


@register_handler("JUMP_BACKWARD", version=PythonVersion.PY_312)
@register_handler("JUMP_BACKWARD", version=PythonVersion.PY_313)
def handle_jump_backward(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # JUMP_BACKWARD is used to jump back to the beginning of a loop (replaces JUMP_ABSOLUTE in 3.13)
    # In generator expressions, this typically indicates the end of the loop body
    return state


@register_handler("RESUME", version=PythonVersion.PY_312)
@register_handler("RESUME", version=PythonVersion.PY_313)
def handle_resume(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # RESUME is used for resuming execution after yield/await - mostly no-op for AST reconstruction
    return state


@register_handler("END_FOR", version=PythonVersion.PY_312)
def handle_end_for_312(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # END_FOR marks the end of a for loop, followed by POP_TOP (in 3.12)
    new_stack = state.stack[:-2]
    return replace(state, stack=new_stack)


@register_handler("END_FOR", version=PythonVersion.PY_313)
def handle_end_for(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # END_FOR marks the end of a for loop - no action needed for AST reconstruction
    new_stack = state.stack[:-1]
    return replace(state, stack=new_stack)


@register_handler("RERAISE", version=PythonVersion.PY_312)
@register_handler("RERAISE", version=PythonVersion.PY_313)
def handle_reraise(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # RERAISE re-raises an exception - generally ignore for AST reconstruction
    return state


# ============================================================================
# VARIABLE OPERATIONS HANDLERS
# ============================================================================


@register_handler("LOAD_FAST", version=PythonVersion.PY_312)
@register_handler("LOAD_FAST", version=PythonVersion.PY_313)
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


@register_handler("LOAD_DEREF", version=PythonVersion.PY_312)
@register_handler("LOAD_DEREF", version=PythonVersion.PY_313)
def handle_load_deref(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LOAD_DEREF loads a value from a closure variable
    var_name = instr.argval
    new_stack = state.stack + [ast.Name(id=var_name, ctx=ast.Load())]
    return replace(state, stack=new_stack)


@register_handler("LOAD_CLOSURE", version=PythonVersion.PY_312)
@register_handler("LOAD_CLOSURE", version=PythonVersion.PY_313)
def handle_load_closure(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LOAD_CLOSURE loads a closure variable
    var_name = instr.argval
    new_stack = state.stack + [ast.Name(id=var_name, ctx=ast.Load())]
    return replace(state, stack=new_stack)


@register_handler("LOAD_CONST", version=PythonVersion.PY_312)
@register_handler("LOAD_CONST", version=PythonVersion.PY_313)
def handle_load_const(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    const_value = instr.argval
    new_stack = state.stack + [ensure_ast(const_value)]
    return replace(state, stack=new_stack)


@register_handler("LOAD_GLOBAL", version=PythonVersion.PY_312)
@register_handler("LOAD_GLOBAL", version=PythonVersion.PY_313)
def handle_load_global(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    global_name = instr.argval

    if instr.argrepr.endswith(" + NULL"):
        new_stack = state.stack + [ast.Name(id=global_name, ctx=ast.Load()), Null()]
    elif instr.argrepr.startswith("NULL + "):
        new_stack = state.stack + [Null(), ast.Name(id=global_name, ctx=ast.Load())]
    else:
        new_stack = state.stack + [ast.Name(id=global_name, ctx=ast.Load())]
    return replace(state, stack=new_stack)


@register_handler("LOAD_NAME", version=PythonVersion.PY_312)
@register_handler("LOAD_NAME", version=PythonVersion.PY_313)
def handle_load_name(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LOAD_NAME is similar to LOAD_GLOBAL but for names in the global namespace
    name = instr.argval
    new_stack = state.stack + [ast.Name(id=name, ctx=ast.Load())]
    return replace(state, stack=new_stack)


@register_handler("STORE_DEREF", version=PythonVersion.PY_312)
@register_handler("STORE_DEREF", version=PythonVersion.PY_313)
def handle_store_deref(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # STORE_DEREF stores a value into a closure variable
    # For AST reconstruction, we treat this the same as STORE_FAST
    return handle_store_fast(state, instr)


@register_handler("STORE_FAST", version=PythonVersion.PY_312)
@register_handler("STORE_FAST", version=PythonVersion.PY_313)
def handle_store_fast(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    if isinstance(state.stack[-1], ast.Name) and state.stack[-1].id == instr.argval:
        # If the variable is already on the stack, we can skip adding it again
        # This is common in nested comprehensions where the same variable is reused
        return replace(state, stack=state.stack[:-1])

    assert isinstance(state.stack[-1], Placeholder)
    for pos, item in zip(reversed(range(len(state.stack))), reversed(state.stack)):
        if isinstance(item, CompExp) and item.generators[-1].target == state.stack[-1]:
            new_result = copy.deepcopy(item)
            new_result.generators[-1].target = ast.Name(
                id=instr.argval, ctx=ast.Store()
            )
            new_stack = state.stack[:pos] + [new_result] + state.stack[pos + 1 : -1]
            return replace(state, stack=new_stack)

    raise TypeError("STORE_FAST did not find matching Placeholder")


@register_handler("STORE_FAST_LOAD_FAST", version=PythonVersion.PY_313)
def handle_store_fast_load_fast(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # STORE_FAST_LOAD_FAST stores and then loads the same variable (optimization)
    # The instruction has two names: store_name and load_name
    # In Python 3.13, this is often used for loop variables

    # In Python 3.13, the instruction argument contains both names
    # argval should be a tuple (store_name, load_name)
    assert isinstance(instr.argval, tuple)
    store_name, load_name = instr.argval

    assert isinstance(state.stack[-1], Placeholder)
    for pos, item in zip(reversed(range(len(state.stack))), reversed(state.stack)):
        if isinstance(item, CompExp) and item.generators[-1].target == state.stack[-1]:
            new_result = copy.deepcopy(item)
            new_result.generators[-1].target = ast.Name(id=store_name, ctx=ast.Store())
            new_var = ast.Name(id=load_name, ctx=ast.Load())
            new_stack = (
                state.stack[:pos] + [new_result] + state.stack[pos + 1 : -1] + [new_var]
            )
            return replace(state, stack=new_stack)

    raise TypeError("STORE_FAST_LOAD_FAST did not find matching Placeholder")


@register_handler("LOAD_FAST_AND_CLEAR", version=PythonVersion.PY_312)
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


@register_handler("MAKE_CELL", version=PythonVersion.PY_312)
@register_handler("MAKE_CELL", version=PythonVersion.PY_313)
def handle_make_cell(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # MAKE_CELL creates a new cell in slot i for closure variables
    # This is used when variables from outer scopes are captured by inner scopes
    # For AST reconstruction purposes, this is just a variable scoping mechanism
    # that we can ignore since the AST doesn't track low-level closure details
    return state


@register_handler("COPY_FREE_VARS", version=PythonVersion.PY_312)
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


@register_handler("POP_TOP", version=PythonVersion.PY_312)
@register_handler("POP_TOP", version=PythonVersion.PY_313)
def handle_pop_top(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # POP_TOP removes the top item from the stack
    # In generators, often used after YIELD_VALUE
    # Also used to clean up the duplicated middle value in failed chained comparisons
    new_stack = state.stack[:-1]
    return replace(state, stack=new_stack)


# Python 3.13 replacement for stack manipulation
@register_handler("SWAP", version=PythonVersion.PY_312)
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

    # For other depths, swap TOS with the item at specified depth
    assert depth <= stack_size, f"SWAP depth {depth} exceeds stack size {stack_size}"
    idx = stack_size - depth
    new_stack = state.stack.copy()
    new_stack[-1], new_stack[idx] = new_stack[idx], new_stack[-1]
    return replace(state, stack=new_stack)


@register_handler("COPY", version=PythonVersion.PY_312)
@register_handler("COPY", version=PythonVersion.PY_313)
def handle_copy(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # COPY duplicates the item at the specified depth
    assert instr.arg is not None
    depth = instr.arg
    stack_size = len(state.stack)
    if depth > stack_size:
        raise ValueError(f"COPY depth {depth} exceeds stack size {stack_size}")
    idx = stack_size - depth
    copied_item = state.stack[idx]
    new_stack = state.stack + [copied_item]
    return replace(state, stack=new_stack)


@register_handler("PUSH_NULL", version=PythonVersion.PY_312)
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


# Python 3.12+ BINARY_OP handler
@register_handler("BINARY_OP", version=PythonVersion.PY_312)
@register_handler("BINARY_OP", version=PythonVersion.PY_313)
def handle_binary_op(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # BINARY_OP in Python 3.12+ consolidates all binary operations
    # The operation type is determined by the instruction argument
    assert instr.arg is not None

    # Map argument values to AST operators based on Python 3.12+ implementation
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
    "UNARY_NEGATIVE",
    functools.partial(handle_unary_op, ast.USub()),
    version=PythonVersion.PY_312,
)
handle_unary_negative = register_handler(
    "UNARY_NEGATIVE",
    functools.partial(handle_unary_op, ast.USub()),
    version=PythonVersion.PY_313,
)
handle_unary_invert = register_handler(
    "UNARY_INVERT",
    functools.partial(handle_unary_op, ast.Invert()),
    version=PythonVersion.PY_312,
)
handle_unary_invert = register_handler(
    "UNARY_INVERT",
    functools.partial(handle_unary_op, ast.Invert()),
    version=PythonVersion.PY_313,
)
handle_unary_not = register_handler(
    "UNARY_NOT",
    functools.partial(handle_unary_op, ast.Not()),
    version=PythonVersion.PY_312,
)
handle_unary_not = register_handler(
    "UNARY_NOT",
    functools.partial(handle_unary_op, ast.Not()),
    version=PythonVersion.PY_313,
)


@register_handler("CONVERT_VALUE", version=PythonVersion.PY_313)
def handle_convert_value(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # CONVERT_VALUE applies a conversion to the value on top of stack
    # Used for f-string conversions like !r, !s, !a
    # The conversion type is stored in instr.arg:
    # 0 = None, 1 = str (!s), 2 = repr (!r), 3 = ascii (!a)
    assert len(state.stack) > 0, "CONVERT_VALUE requires a value on stack"
    assert instr.arg is not None, "CONVERT_VALUE requires conversion type"

    # Wrap the value with conversion information
    value = state.stack[-1]
    converted = ConvertedValue(value, instr.arg)
    new_stack = state.stack[:-1] + [converted]

    return replace(state, stack=new_stack)


@register_handler("CALL_INTRINSIC_1", version=PythonVersion.PY_312)
@register_handler("CALL_INTRINSIC_1", version=PythonVersion.PY_313)
def handle_call_intrinsic_1(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # CALL_INTRINSIC_1 calls an intrinsic function with one argument
    if instr.argrepr == "INTRINSIC_LIST_TO_TUPLE":
        assert isinstance(state.stack[-1], ast.List), (
            "Expected a list for LIST_TO_TUPLE"
        )
        tuple_node = ast.Tuple(elts=state.stack[-1].elts, ctx=ast.Load())
        return replace(state, stack=state.stack[:-1] + [tuple_node])
    elif instr.argrepr == "INTRINSIC_UNARY_POSITIVE":
        assert len(state.stack) > 0
        new_val = ast.UnaryOp(op=ast.UAdd(), operand=state.stack[-1])
        return replace(state, stack=state.stack[:-1] + [new_val])
    elif instr.argrepr == "INTRINSIC_STOPITERATION_ERROR":
        return state
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


@register_handler("COMPARE_OP", version=PythonVersion.PY_312)
@register_handler("COMPARE_OP", version=PythonVersion.PY_313)
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


@register_handler("CONTAINS_OP", version=PythonVersion.PY_312)
@register_handler("CONTAINS_OP", version=PythonVersion.PY_313)
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


@register_handler("IS_OP", version=PythonVersion.PY_312)
@register_handler("IS_OP", version=PythonVersion.PY_313)
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


@register_handler("CALL", version=PythonVersion.PY_312)
def handle_call_312(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # CALL in Python 3.12 handles both function and method calls
    # Stack layout: [..., callable or self, callable or NULL]
    assert instr.arg is not None
    arg_count: int = instr.arg

    # Check if this is a method call (no NULL on top)
    if isinstance(state.stack[-arg_count - 2], Null):
        # Regular function call: [..., NULL, callable, *args]
        func = ensure_ast(state.stack[-arg_count - 1])
        args = (
            [ensure_ast(arg) for arg in state.stack[-arg_count:]]
            if arg_count > 0
            else []
        )
        new_stack = state.stack[: -arg_count - 2]
    else:
        # Method call: [..., callable, self, *args]
        func = ensure_ast(state.stack[-arg_count - 2])
        self_arg = ensure_ast(state.stack[-arg_count - 1])
        remaining_args = (
            [ensure_ast(arg) for arg in state.stack[-arg_count:]]
            if arg_count > 0
            else []
        )
        args = [self_arg] + remaining_args
        new_stack = state.stack[: -arg_count - 2]

    if isinstance(func, CompLambda):
        assert len(args) == 1
        return replace(state, stack=new_stack + [func.inline(args[0])])
    else:
        # Create function call AST
        call_node = ast.Call(func=func, args=args, keywords=[])
        new_stack = new_stack + [call_node]
        return replace(state, stack=new_stack)


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


@register_handler("CALL_KW", version=PythonVersion.PY_313)
def handle_call_kw(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # CALL_KW pops function, arguments, and keyword names from stack
    assert instr.arg is not None
    arg_count: int = instr.arg

    func = ensure_ast(state.stack[-arg_count - 3])
    kw_names = state.stack[-1]
    assert isinstance(kw_names, ast.Tuple), "Expected a tuple of keyword names"

    # Pop arguments, function, and keyword names
    args = (
        [ensure_ast(arg) for arg in state.stack[-arg_count - 2 : -1]]
        if arg_count > 0
        else []
    )
    if not isinstance(state.stack[-arg_count - 3], Null):
        args = [ensure_ast(state.stack[-arg_count - 3])] + args

    keywords = []
    for i, kw in enumerate(reversed(kw_names.elts)):
        kw_name = (
            kw.s if isinstance(kw, ast.Constant) and isinstance(kw.s, str) else None
        )
        if kw_name is None:
            raise TypeError("Keyword names must be strings")
        kw_value = ensure_ast(state.stack[-1 - i])
        keywords.append(ast.keyword(arg=kw_name, value=kw_value))
    keywords.reverse()

    new_stack = state.stack[: -arg_count - 3]
    if isinstance(func, CompLambda):
        assert len(args) == 1 and len(keywords) == 0
        return replace(state, stack=new_stack + [func.inline(args[0])])
    else:
        # Create function call AST
        call_node = ast.Call(func=func, args=args, keywords=keywords)
        new_stack = new_stack + [call_node]
        return replace(state, stack=new_stack)


@register_handler("MAKE_FUNCTION", version=PythonVersion.PY_312)
def handle_make_function_312(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # MAKE_FUNCTION in Python 3.12 uses flags to determine stack consumption
    # Unlike 3.10, no qualified name on stack
    # Unlike 3.13, uses flags instead of SET_FUNCTION_ATTRIBUTE
    assert instr.arg is not None
    assert isinstance(state.stack[-1], ast.Lambda | CompLambda), (
        "Expected a function object (Lambda or CompLambda) on the stack."
    )
    if instr.argrepr == "closure":
        # This is a closure, remove the environment tuple from the stack for AST purposes
        new_stack = state.stack[:-2]
    elif instr.argrepr == "":
        new_stack = state.stack[:-1]
    else:
        raise NotImplementedError(
            "MAKE_FUNCTION with defaults or annotations not implemented."
        )

    # For comprehensions, we only care about the function object
    func = state.stack[-1]
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


@register_handler("LOAD_ATTR", version=PythonVersion.PY_312)
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
    elif instr.argrepr.startswith("NULL|self + "):
        new_stack = state.stack[:-1] + [Null(), attr_node]
    else:
        new_stack = state.stack[:-1] + [attr_node]
    return replace(state, stack=new_stack)


@register_handler("BINARY_SUBSCR", version=PythonVersion.PY_312)
@register_handler("BINARY_SUBSCR", version=PythonVersion.PY_313)
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


@register_handler("BINARY_SLICE", version=PythonVersion.PY_312)
@register_handler("BINARY_SLICE", version=PythonVersion.PY_313)
def handle_binary_slice(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # BINARY_SLICE implements obj[start:end] - pops start, end, and obj from stack
    end = ensure_ast(state.stack[-1])
    start = ensure_ast(state.stack[-2])
    container = ensure_ast(state.stack[-3])  # Object is below start and end
    sliced = ast.Subscript(
        value=container,
        slice=ast.Slice(lower=start, upper=end, step=None),
        ctx=ast.Load(),
    )
    new_stack = state.stack[:-3] + [sliced]
    return replace(state, stack=new_stack)


# ============================================================================
# OTHER CONTAINER BUILDING HANDLERS
# ============================================================================


@register_handler("UNPACK_SEQUENCE", version=PythonVersion.PY_312)
@register_handler("UNPACK_SEQUENCE", version=PythonVersion.PY_313)
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


@register_handler("BUILD_TUPLE", version=PythonVersion.PY_312)
@register_handler("BUILD_TUPLE", version=PythonVersion.PY_313)
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


@register_handler("BUILD_SLICE", version=PythonVersion.PY_312)
@register_handler("BUILD_SLICE", version=PythonVersion.PY_313)
def handle_build_slice(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # BUILD_SLICE creates a slice object from the top of the stack
    # The number of elements to pop is determined by the instruction argument
    assert instr.arg is not None
    slice_size: int = instr.arg

    if slice_size == 2:
        # Slice with start and end: [start, end]
        end = ensure_ast(state.stack[-1])
        start = ensure_ast(state.stack[-2])
        new_stack = state.stack[:-2]
        slice_node = ast.Slice(lower=start, upper=end, step=None)
    elif slice_size == 3:
        # Slice with start, end, and step: [start, end, step]
        step = ensure_ast(state.stack[-1])
        end = ensure_ast(state.stack[-2])
        start = ensure_ast(state.stack[-3])
        new_stack = state.stack[:-3]
        slice_node = ast.Slice(lower=start, upper=end, step=step)
    else:
        raise ValueError(f"Unsupported slice size: {slice_size}")

    # Create slice AST
    new_stack = new_stack + [slice_node]
    return replace(state, stack=new_stack)


@register_handler("BUILD_CONST_KEY_MAP", version=PythonVersion.PY_312)
@register_handler("BUILD_CONST_KEY_MAP", version=PythonVersion.PY_313)
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


@register_handler("LIST_EXTEND", version=PythonVersion.PY_312)
@register_handler("LIST_EXTEND", version=PythonVersion.PY_313)
def handle_list_extend(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LIST_EXTEND extends the list at TOS-1 with the iterable at TOS
    # initially recognized as list comp

    # The list being extended is actually in state.result instead of the stack
    # because it was initially recognized as a list comprehension in BUILD_LIST,
    # while the actual result expression is in the stack where the list "should be"
    # and needs to be put back into the state result slot
    assert isinstance(state.stack[-1], ast.Tuple | ast.List)
    assert isinstance(state.stack[-instr.argval - 1], ast.ListComp)

    new_val = ast.List(
        elts=[ensure_ast(e) for e in state.stack[-1].elts], ctx=ast.Load()
    )
    new_stack = state.stack[:-2] + [new_val]

    return replace(state, stack=new_stack)


@register_handler("SET_UPDATE", version=PythonVersion.PY_312)
@register_handler("SET_UPDATE", version=PythonVersion.PY_313)
def handle_set_update(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # The set being extended is actually in state.result instead of the stack
    # because it was initially recognized as a list comprehension in BUILD_SET,
    # while the actual result expression is in the stack where the set "should be"
    # and needs to be put back into the state result slot
    assert isinstance(state.stack[-instr.argval - 1], ast.SetComp)
    assert isinstance(state.stack[-1], ast.Tuple | ast.List | ast.Set)

    new_val = ast.Set(elts=[ensure_ast(e) for e in state.stack[-1].elts])
    new_stack = state.stack[:-2] + [new_val]

    return replace(state, stack=new_stack)


@register_handler("DICT_UPDATE", version=PythonVersion.PY_312)
@register_handler("DICT_UPDATE", version=PythonVersion.PY_313)
def handle_dict_update(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # The dict being extended is actually in state.result instead of the stack
    # because it was initially recognized as a list comprehension in BUILD_MAP,
    # while the actual result expression is in the stack where the dict "should be"
    # and needs to be put back into the state result slot
    assert isinstance(state.stack[-instr.argval - 1], ast.DictComp)
    assert isinstance(state.stack[-1], ast.Dict)

    new_val = ast.Dict(
        keys=[ensure_ast(e) for e in state.stack[-1].keys],
        values=[ensure_ast(e) for e in state.stack[-1].values],
    )
    new_stack = state.stack[:-2] + [new_val]

    return replace(state, stack=new_stack)


@register_handler("BUILD_STRING", version=PythonVersion.PY_312)
@register_handler("BUILD_STRING", version=PythonVersion.PY_313)
def handle_build_string(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # BUILD_STRING concatenates strings from the stack
    # For f-strings, it combines FormattedValue and Constant nodes
    assert instr.arg is not None
    string_size: int = instr.arg

    if string_size == 0:
        # Empty string case
        new_stack = state.stack + [ast.Constant(value="")]
        return replace(state, stack=new_stack)

    # Pop elements for the string
    elements = [ensure_ast(elem) for elem in state.stack[-string_size:]]
    new_stack = state.stack[:-string_size]

    # Check if this is an f-string build (has FormattedValue nodes)
    # or a regular string concatenation
    if any(isinstance(elem, ast.JoinedStr) for elem in elements):
        # This is an f-string - create JoinedStr
        values = []
        for elem in elements:
            if isinstance(elem, ast.JoinedStr):
                values.extend(elem.values)
            else:
                values.append(elem)
        return replace(state, stack=new_stack + [ast.JoinedStr(values=values)])
    elif all(isinstance(elem, ast.Constant) for elem in elements):
        # This is regular string concatenation or format spec building
        # If all elements are constants, we might be building a format spec
        # Concatenate the constant strings
        assert all(
            isinstance(elem, ast.Constant) and isinstance(elem.value, str)
            for elem in elements
        )
        concat_str = "".join(
            elem.value
            for elem in elements
            if isinstance(elem, ast.Constant) and isinstance(elem.value, str)
        )
        return replace(state, stack=new_stack + [ast.Constant(value=concat_str)])
    else:
        raise TypeError("Should not be here?")


@register_handler("FORMAT_VALUE", version=PythonVersion.PY_312)
def handle_format_value(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # FORMAT_VALUE formats a string with a value in Python 3.12
    # Flag bits: (flags & 0x03) = conversion, (flags & 0x04) = has format spec
    assert instr.arg is not None, "FORMAT_VALUE requires flags argument"
    assert len(state.stack) >= 1, "Not enough items on stack for FORMAT_VALUE"

    flags = instr.arg

    # Check if there's a format specification
    has_format_spec = bool(flags & 0x04)

    if has_format_spec:
        # Pop format spec and value
        assert len(state.stack) >= 2, (
            "FORMAT_VALUE with format spec needs 2 stack items"
        )
        format_spec = ensure_ast(state.stack[-1])
        value = ensure_ast(state.stack[-2])
        new_stack = state.stack[:-2]

        # Wrap format spec in JoinedStr if it's a constant
        if isinstance(format_spec, ast.Constant):
            format_spec_node = ast.JoinedStr(values=[format_spec])
        else:
            assert isinstance(format_spec, ast.JoinedStr)
            format_spec_node = format_spec
    else:
        # Just pop the value
        value = ensure_ast(state.stack[-1])
        new_stack = state.stack[:-1]
        format_spec_node = None

    # Determine conversion type from flags
    conversion_flags = flags & 0x03
    conversion_map = {
        0: -1,  # No conversion
        1: 115,  # str (!s)
        2: 114,  # repr (!r)
        3: 97,  # ascii (!a)
    }
    conversion = conversion_map[conversion_flags]

    # Create formatted value AST
    formatted_node = ast.FormattedValue(
        value=value, conversion=conversion, format_spec=format_spec_node
    )
    new_stack = new_stack + [ast.JoinedStr(values=[formatted_node])]
    return replace(state, stack=new_stack)


@register_handler("FORMAT_SIMPLE", version=PythonVersion.PY_313)
def handle_format_simple(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # FORMAT_SIMPLE formats a string with a single value
    # Pops the value and the format string from the stack
    assert len(state.stack) >= 1, "Not enough items on stack for FORMAT_SIMPLE"
    value = state.stack[-1]
    new_stack = state.stack[:-1]

    # Check if the value was converted
    if isinstance(value, ConvertedValue):
        conversion = value.ast_conversion
        value = value.value
    else:
        conversion = -1
        value = ensure_ast(value)

    # Create formatted string AST
    formatted_node = ast.FormattedValue(
        value=value, conversion=conversion, format_spec=None
    )
    new_stack = new_stack + [ast.JoinedStr(values=[formatted_node])]
    return replace(state, stack=new_stack)


@register_handler("FORMAT_WITH_SPEC", version=PythonVersion.PY_313)
def handle_format_with_spec(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # FORMAT_WITH_SPEC formats a value with a format specifier
    # Stack order in Python 3.13: format_spec on top, value below
    assert len(state.stack) >= 2, "Not enough items on stack for FORMAT_WITH_SPEC"
    format_spec = ensure_ast(state.stack[-1])  # Format spec is on top
    value = state.stack[-2]  # Value is below
    new_stack = state.stack[:-2]

    # Check if the value was converted
    if isinstance(value, ConvertedValue):
        conversion = value.ast_conversion
        value = value.value
    else:
        conversion = -1
        value = ensure_ast(value)

    # Create formatted string AST with specifier
    # The format_spec should be wrapped in a JoinedStr if it's a simple constant
    if isinstance(format_spec, ast.Constant):
        format_spec_node = ast.JoinedStr(values=[format_spec])
    else:
        # Already a JoinedStr from nested formatting
        assert isinstance(format_spec, ast.JoinedStr)
        format_spec_node = format_spec

    formatted_node = ast.FormattedValue(
        value=value, conversion=conversion, format_spec=format_spec_node
    )
    new_stack = new_stack + [ast.JoinedStr(values=[formatted_node])]
    return replace(state, stack=new_stack)


# ============================================================================
# CONDITIONAL JUMP HANDLERS
# ============================================================================


def _handle_pop_jump_if(
    f_condition: Callable[[ast.expr], ast.expr],
    state: ReconstructionState,
    instr: dis.Instruction,
) -> ReconstructionState:
    # Generic handler for POP_JUMP_IF_* instructions
    # Pops a value from the stack and jumps if the condition is met
    condition = f_condition(ensure_ast(state.stack[-1]))

    if state.is_filter:
        for pos, item in zip(reversed(range(len(state.stack))), reversed(state.stack)):
            if isinstance(item, CompExp) and isinstance(
                getattr(item, "elt", getattr(item, "key", None)), Placeholder
            ):
                new_result = copy.deepcopy(item)
                new_result.generators[-1].ifs.append(condition)
                new_stack = state.stack[:pos] + [new_result] + state.stack[pos + 1 : -1]
                return replace(state, stack=new_stack)
        raise TypeError("No comprehension context found for filter condition")
    else:
        for pos, item in zip(reversed(range(len(state.stack))), reversed(state.stack)):
            if any(isinstance(x, Placeholder) for x in ast.walk(item)):
                body: Skipped | Placeholder
                orelse: Skipped | Placeholder
                if state.branches.get(instr.offset, 0):
                    # we don't jump, so we're in the orelse branch
                    body, orelse = Skipped(id=f".SKIPPED_{instr.offset}"), Placeholder()
                else:
                    # we jump, so we're in the body branch
                    body, orelse = Placeholder(), Skipped(id=f".SKIPPED_{instr.offset}")
                new_ifexp = ast.IfExp(test=condition, body=body, orelse=orelse)
                new_result = ReplacePlaceholder(new_ifexp).visit(copy.deepcopy(item))
                new_stack = state.stack[:pos] + [new_result] + state.stack[pos + 1 : -1]
                return replace(state, stack=new_stack)
        raise TypeError("No placeholder found for conditional expression")


@register_handler("POP_JUMP_IF_TRUE", version=PythonVersion.PY_312)
@register_handler("POP_JUMP_IF_TRUE", version=PythonVersion.PY_313)
def handle_pop_jump_if_true(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # POP_JUMP_IF_TRUE pops a value from the stack and jumps if it's true
    # In Python 3.13, this is used for filter conditions where True means continue
    return _handle_pop_jump_if(lambda c: c, state, instr)


@register_handler("POP_JUMP_IF_FALSE", version=PythonVersion.PY_312)
@register_handler("POP_JUMP_IF_FALSE", version=PythonVersion.PY_313)
def handle_pop_jump_if_false(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # POP_JUMP_IF_FALSE pops a value from the stack and jumps if it's false
    # In comprehensions, this is used for filter conditions
    return _handle_pop_jump_if(
        lambda c: ast.UnaryOp(op=ast.Not(), operand=c), state, instr
    )


@register_handler("POP_JUMP_IF_NONE", version=PythonVersion.PY_312)
@register_handler("POP_JUMP_IF_NONE", version=PythonVersion.PY_313)
def handle_pop_jump_if_none(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # POP_JUMP_IF_NONE pops a value and jumps if it's None
    return _handle_pop_jump_if(
        lambda c: ast.Compare(
            left=c, ops=[ast.Is()], comparators=[ast.Constant(value=None)]
        ),
        state,
        instr,
    )


@register_handler("POP_JUMP_IF_NOT_NONE", version=PythonVersion.PY_312)
@register_handler("POP_JUMP_IF_NOT_NONE", version=PythonVersion.PY_313)
def handle_pop_jump_if_not_none(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # POP_JUMP_IF_NOT_NONE pops a value and jumps if it's not None
    return _handle_pop_jump_if(
        lambda c: ast.Compare(
            left=c, ops=[ast.IsNot()], comparators=[ast.Constant(value=None)]
        ),
        state,
        instr,
    )


@register_handler("SEND", version=PythonVersion.PY_312)
@register_handler("SEND", version=PythonVersion.PY_313)
def handle_send(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    raise TypeError("SEND instruction should not appear in generator comprehensions")


@register_handler("JUMP_BACKWARD_NO_INTERRUPT", version=PythonVersion.PY_312)
@register_handler("JUMP_BACKWARD_NO_INTERRUPT", version=PythonVersion.PY_313)
def handle_jump_backward_no_interrupt(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    raise TypeError(
        "JUMP_BACKWARD_NO_INTERRUPT instruction should not appear in generator comprehensions"
    )


@register_handler("JUMP", version=PythonVersion.PY_312)
@register_handler("JUMP", version=PythonVersion.PY_313)
def handle_jump(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    raise TypeError("JUMP instruction should not appear in generator comprehensions")


@register_handler("JUMP_NO_INTERRUPT", version=PythonVersion.PY_312)
@register_handler("JUMP_NO_INTERRUPT", version=PythonVersion.PY_313)
def handle_jump_no_interrupt(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    raise TypeError(
        "JUMP_NO_INTERRUPT instruction should not appear in generator comprehensions"
    )


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
    result: ast.expr = _symbolic_exec(value)

    # Check postconditions
    assert not any(isinstance(x, ast.stmt) for x in ast.walk(result)), (
        "Final return value must not contain statement nodes"
    )
    assert not any(
        isinstance(x, Placeholder | Skipped | Null | CompLambda | ConvertedValue)
        for x in ast.walk(result)
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


def disassemble(genexpr: Generator[object, None, None]) -> ast.Expression:
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
        >>> ast_node = disassemble(g)
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
