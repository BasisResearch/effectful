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
import functools
import inspect
import types
import typing
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field, replace

CompExp = ast.GeneratorExp | ast.ListComp | ast.SetComp | ast.DictComp


class Placeholder(ast.Name):
    """Placeholder for AST nodes that are not yet resolved."""
    def __init__(self):
        super().__init__(id=".PLACEHOLDER", ctx=ast.Load())


class DummyIterName(ast.Name):
    """Dummy name for the iterator variable in generator expressions."""
    def __init__(self):
        super().__init__(id=".0", ctx=ast.Load())


class CompLambda(ast.Lambda):
    """Placeholder AST node representing a lambda function used in comprehensions."""
    def __init__(self, body: CompExp):
        assert sum(1 for x in ast.walk(body) if isinstance(x, DummyIterName)) == 1
        assert len(body.generators) > 0
        assert isinstance(body.generators[0].iter, DummyIterName)
        super().__init__(
            args=ast.arguments(
                posonlyargs=[ast.arg(DummyIterName().id)],
                args=[],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]
            ),
            body=body
        )

    def inline(self, iterator: ast.expr) -> CompExp:
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
    result: CompExp | ast.Lambda | Placeholder = field(default_factory=Placeholder)
    stack: list[ast.expr] = field(default_factory=list)


# Global handler registry
OpHandler = Callable[[ReconstructionState, dis.Instruction], ReconstructionState]

OP_HANDLERS: dict[str, OpHandler] = {}


@typing.overload
def register_handler(opname: str) -> Callable[[OpHandler], OpHandler]:
    ...

@typing.overload  
def register_handler(opname: str, handler: OpHandler) -> OpHandler:
    ...

def register_handler(opname: str, handler = None):
    """Register a handler for a specific operation name"""
    if handler is None:
        return functools.partial(register_handler, opname)
    
    assert opname in dis.opmap, f"Invalid operation name: '{opname}'"

    if opname in OP_HANDLERS:
        raise ValueError(f"Handler for '{opname}' already exists.")

    @functools.wraps(handler)
    def _wrapper(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
        assert instr.opname == opname, f"Handler for '{opname}' called with wrong instruction"
        return handler(copy.deepcopy(state), instr)
    
    OP_HANDLERS[opname] = _wrapper
    return _wrapper


# ============================================================================
# GENERATOR COMPREHENSION HANDLERS
# ============================================================================

@register_handler('GEN_START')
def handle_gen_start(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # GEN_START is typically the first instruction in generator expressions
    # It initializes the generator
    assert isinstance(state.result, Placeholder), "GEN_START must be the first instruction"
    return replace(state, result=ast.GeneratorExp(elt=Placeholder(), generators=[]))


@register_handler('YIELD_VALUE')
def handle_yield_value(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # YIELD_VALUE pops a value from the stack and yields it
    # This is the expression part of the generator
    assert isinstance(state.result, ast.GeneratorExp), "YIELD_VALUE must be called after GEN_START"
    assert isinstance(state.result.elt, Placeholder), "YIELD_VALUE must be called before yielding"
    assert len(state.result.generators) > 0, "YIELD_VALUE should have generators"

    new_stack = state.stack[:-1]
    ret = ast.GeneratorExp(
        elt=ensure_ast(state.stack[-1]),
        generators=state.result.generators,
    )
    return replace(state, stack=new_stack, result=ret)


# ============================================================================
# LIST COMPREHENSION HANDLERS
# ============================================================================

@register_handler('BUILD_LIST')
def handle_build_list(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    if isinstance(state.result, Placeholder) and len(state.stack) == 0:
        # This BUILD_LIST is the start of a list comprehension
        # Initialize the result as a ListComp with a placeholder element
        ret = ast.ListComp(elt=Placeholder(), generators=[])
        new_stack = state.stack + [ret]
        return replace(state, stack=new_stack, result=ret)
    else:
        size: int = instr.arg
        # Pop elements for the list
        elements = [ensure_ast(elem) for elem in state.stack[-size:]] if size > 0 else []
        new_stack = state.stack[:-size] if size > 0 else state.stack
   
        # Create list AST
        elt_node = ast.List(elts=elements, ctx=ast.Load())
        new_stack = new_stack + [elt_node]
        return replace(state, stack=new_stack)


@register_handler('LIST_APPEND')
def handle_list_append(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    assert isinstance(state.result, ast.ListComp), "LIST_APPEND must be called within a ListComp context"
    new_stack = state.stack[:-1]
    new_ret = ast.ListComp(
        elt=ensure_ast(state.stack[-1]),
        generators=state.result.generators,
    )
    return replace(state, stack=new_stack, result=new_ret)


# ============================================================================
# SET COMPREHENSION HANDLERS
# ============================================================================

@register_handler('BUILD_SET')
def handle_build_set(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    if isinstance(state.result, Placeholder) and len(state.stack) == 0:
        # This BUILD_SET is the start of a list comprehension
        # Initialize the result as a ListComp with a placeholder element
        ret = ast.SetComp(elt=Placeholder(), generators=[])
        new_stack = state.stack + [ret]
        return replace(state, stack=new_stack, result=ret)
    else:
        size: int = instr.arg
        # Pop elements for the set
        elements = [ensure_ast(elem) for elem in state.stack[-size:]] if size > 0 else []
        new_stack = state.stack[:-size] if size > 0 else state.stack
   
        # Create set AST
        elt_node = ast.Set(elts=elements)
        new_stack = new_stack + [elt_node]
        return replace(state, stack=new_stack)


@register_handler('SET_ADD')
def handle_set_add(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    assert isinstance(state.result, ast.SetComp), "SET_ADD must be called after BUILD_SET"
    new_stack = state.stack[:-1]
    new_ret = ast.SetComp(
        elt=ensure_ast(state.stack[-1]),
        generators=state.result.generators,
    )
    return replace(state, stack=new_stack, result=new_ret)


# ============================================================================
# DICT COMPREHENSION HANDLERS
# ============================================================================

@register_handler('BUILD_MAP')
def handle_build_map(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    if isinstance(state.result, Placeholder) and len(state.stack) == 0:
        # This is the start of a comprehension
        # Initialize the result with a placeholder element
        ret = ast.DictComp(key=Placeholder(), value=Placeholder(), generators=[])
        new_stack = state.stack + [ret]
        return replace(state, stack=new_stack, result=ret)
    else:
        size: int = instr.arg
        # Pop key-value pairs for the dict
        keys = [ensure_ast(state.stack[-2*i-2]) for i in range(size)]
        values = [ensure_ast(state.stack[-2*i-1]) for i in range(size)]
        new_stack = state.stack[:-2*size] if size > 0 else state.stack

        # Create dict AST
        dict_node = ast.Dict(keys=keys, values=values)
        new_stack = new_stack + [dict_node]
        return replace(state, stack=new_stack)


@register_handler('MAP_ADD')
def handle_map_add(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    assert isinstance(state.result, ast.DictComp), "MAP_ADD must be called after BUILD_MAP"
    new_stack = state.stack[:-2]
    new_ret = ast.DictComp(
        key=ensure_ast(state.stack[-2]),
        value=ensure_ast(state.stack[-1]),
        generators=state.result.generators,
    )
    return replace(state, stack=new_stack, result=new_ret)


# ============================================================================
# LOOP CONTROL HANDLERS
# ============================================================================

@register_handler('RETURN_VALUE')
def handle_return_value(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # RETURN_VALUE ends the generator
    # Usually preceded by LOAD_CONST None
    if isinstance(state.result, CompExp):
        return replace(state, stack=state.stack[:-1])
    elif isinstance(state.result, Placeholder) and len(state.stack) == 1:
        return replace(state, stack=state.stack[:-1], result=ensure_ast(state.stack[-1]))
    else:
        raise TypeError("Unexpected RETURN_VALUE in reconstruction")


@register_handler('FOR_ITER')
def handle_for_iter(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # FOR_ITER pops an iterator from the stack and pushes the next item
    # If the iterator is exhausted, it jumps to the target instruction
    assert len(state.stack) > 0, "FOR_ITER must have an iterator on the stack"
    assert isinstance(state.result, CompExp), "FOR_ITER must be called within a comprehension context"

    # The iterator should be on top of stack
    # Create new stack without the iterator
    new_stack = state.stack[:-1]
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
    new_ret: ast.GeneratorExp | ast.ListComp | ast.SetComp | ast.DictComp
    if isinstance(state.result, ast.DictComp):
        # If it's a DictComp, we need to ensure the loop is added to the dict comprehension
        new_ret = ast.DictComp(
            key=state.result.key,
            value=state.result.value,
            generators=state.result.generators + [loop_info],
        )
    else:
        new_ret = type(state.result)(
            elt=state.result.elt,
            generators=state.result.generators + [loop_info],
        )

    return replace(state, stack=new_stack, result=new_ret)


@register_handler('GET_ITER')
def handle_get_iter(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # GET_ITER converts the top stack item to an iterator
    # For AST reconstruction, we typically don't need to change anything
    # since the iterator will be used directly in the comprehension
    return state


@register_handler('JUMP_ABSOLUTE')
def handle_jump_absolute(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # JUMP_ABSOLUTE is used to jump back to the beginning of a loop
    # In generator expressions, this typically indicates the end of the loop body
    return state


@register_handler('JUMP_FORWARD')
def handle_jump_forward(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # JUMP_FORWARD is used to jump forward in the code
    # In generator expressions, this is often used to skip code in conditional logic
    return state


@register_handler('UNPACK_SEQUENCE')
def handle_unpack_sequence(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # UNPACK_SEQUENCE unpacks a sequence into multiple values
    # arg is the number of values to unpack
    unpack_count = instr.arg
    sequence = ensure_ast(state.stack[-1])
    new_stack = state.stack[:-1]
    
    # For tuple unpacking in comprehensions, we typically see patterns like:
    # ((k, v) for k, v in items) where items is unpacked into k and v
    # Create placeholder variables for the unpacked values
    for i in range(unpack_count):
        var_name = f'_unpack_{i}'
        new_stack = new_stack + [ast.Name(id=var_name, ctx=ast.Load())]
    
    return replace(state, stack=new_stack)


# ============================================================================
# VARIABLE OPERATIONS HANDLERS
# ============================================================================

@register_handler('LOAD_FAST')
def handle_load_fast(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    var_name: str = instr.argval
    
    if var_name == '.0':
        # Special handling for .0 variable (the iterator)
        new_stack = state.stack + [DummyIterName()]
    else:
        # Regular variable load
        new_stack = state.stack + [ast.Name(id=var_name, ctx=ast.Load())]
    
    return replace(state, stack=new_stack)


@register_handler('STORE_FAST')
def handle_store_fast(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    assert isinstance(state.result, CompExp), "STORE_FAST must be called within a comprehension context"
    var_name = instr.argval
    
    # Update the most recent loop's target variable
    assert len(state.result.generators) > 0, "STORE_FAST must be within a loop context"

    # Create a new LoopInfo with updated target
    updated_loop = ast.comprehension(
        target=ast.Name(id=var_name, ctx=ast.Store()),
        iter=state.result.generators[-1].iter,
        ifs=state.result.generators[-1].ifs,
        is_async=state.result.generators[-1].is_async
    )

    # Update the last loop in the generators list
    if isinstance(state.result, ast.DictComp):
        new_ret = ast.DictComp(
            key=state.result.key,
            value=state.result.value,
            generators=state.result.generators[:-1] + [updated_loop],
        )
    else:
        new_ret = type(state.result)(
            elt=state.result.elt,
            generators=state.result.generators[:-1] + [updated_loop],
        )

    # Create new loops list with the updated loop
    return replace(state, result=new_ret)


@register_handler('LOAD_CONST')
def handle_load_const(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    const_value = instr.argval
    new_stack = state.stack + [ensure_ast(const_value)]
    return replace(state, stack=new_stack)


@register_handler('LOAD_GLOBAL')
def handle_load_global(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    global_name = instr.argval
    new_stack = state.stack + [ast.Name(id=global_name, ctx=ast.Load())]
    return replace(state, stack=new_stack)


@register_handler('LOAD_NAME')
def handle_load_name(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # LOAD_NAME is similar to LOAD_GLOBAL but for names in the global namespace
    name = instr.argval
    new_stack = state.stack + [ast.Name(id=name, ctx=ast.Load())]
    return replace(state, stack=new_stack)


@register_handler('STORE_DEREF')
def handle_store_deref(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # STORE_DEREF stores a value into a closure variable
    assert isinstance(state.result, CompExp), "STORE_DEREF must be called within a comprehension context"
    var_name = instr.argval
    
    # Update the most recent loop's target variable
    assert len(state.result.generators) > 0, "STORE_DEREF must be within a loop context"

    # Create a new LoopInfo with updated target
    updated_loop = ast.comprehension(
        target=ast.Name(id=var_name, ctx=ast.Store()),
        iter=state.result.generators[-1].iter,
        ifs=state.result.generators[-1].ifs,
        is_async=state.result.generators[-1].is_async
    )

    # Update the last loop in the generators list
    if isinstance(state.result, ast.DictComp):
        new_ret = ast.DictComp(
            key=state.result.key,
            value=state.result.value,
            generators=state.result.generators[:-1] + [updated_loop],
        )
    else:
        new_ret = type(state.result)(
            elt=state.result.elt,
            generators=state.result.generators[:-1] + [updated_loop],
        )

    # Create new loops list with the updated loop
    return replace(state, result=new_ret)


@register_handler('LOAD_DEREF')
def handle_load_deref(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # LOAD_DEREF loads a value from a closure variable
    var_name = instr.argval
    new_stack = state.stack + [ast.Name(id=var_name, ctx=ast.Load())]
    return replace(state, stack=new_stack)


@register_handler('LOAD_CLOSURE')
def handle_load_closure(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # LOAD_CLOSURE loads a closure variable
    var_name = instr.argval
    new_stack = state.stack + [ast.Name(id=var_name, ctx=ast.Load())]
    return replace(state, stack=new_stack)


# ============================================================================
# STACK MANAGEMENT HANDLERS
# ============================================================================

@register_handler('POP_TOP')
def handle_pop_top(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # POP_TOP removes the top item from the stack
    # In generators, often used after YIELD_VALUE
    # Also used to clean up the duplicated middle value in failed chained comparisons
    new_stack = state.stack[:-1]
    return replace(state, stack=new_stack)


@register_handler('DUP_TOP')
def handle_dup_top(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # DUP_TOP duplicates the top stack item
    top_item = state.stack[-1]
    new_stack = state.stack + [top_item]
    return replace(state, stack=new_stack)


@register_handler('ROT_TWO')
def handle_rot_two(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # ROT_TWO swaps the top two stack items
    new_stack = state.stack[:-2] + [state.stack[-1], state.stack[-2]]
    return replace(state, stack=new_stack)


@register_handler('ROT_THREE')
def handle_rot_three(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # ROT_THREE rotates the top three stack items
    # TOS -> TOS1, TOS1 -> TOS2, TOS2 -> TOS
    new_stack = state.stack[:-3] + [state.stack[-2], state.stack[-1], state.stack[-3]]
    
    # Check if the top two items are the same (from DUP_TOP)
    # This heuristic indicates we're setting up for a chained comparison
    if len(state.stack) >= 3 and state.stack[-1] == state.stack[-2]:
        raise NotImplementedError("Chained comparison not implemented yet")
    
    return replace(state, stack=new_stack)


@register_handler('ROT_FOUR')
def handle_rot_four(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # ROT_FOUR rotates the top four stack items
    # TOS -> TOS1, TOS1 -> TOS2, TOS2 -> TOS3, TOS3 -> TOS
    new_stack = state.stack[:-4] + [state.stack[-2], state.stack[-1], state.stack[-4], state.stack[-3]]
    return replace(state, stack=new_stack)


# ============================================================================
# BINARY ARITHMETIC/LOGIC OPERATION HANDLERS
# ============================================================================

def handle_binop(op: ast.operator, state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    right = ensure_ast(state.stack[-1])
    left = ensure_ast(state.stack[-2])
    new_stack = state.stack[:-2] + [ast.BinOp(left=left, op=op, right=right)]
    return replace(state, stack=new_stack)


handler_binop_add = register_handler('BINARY_ADD', functools.partial(handle_binop, ast.Add()))
handler_binop_subtract = register_handler('BINARY_SUBTRACT', functools.partial(handle_binop, ast.Sub()))
handler_binop_multiply = register_handler('BINARY_MULTIPLY', functools.partial(handle_binop, ast.Mult()))
handler_binop_true_divide = register_handler('BINARY_TRUE_DIVIDE', functools.partial(handle_binop, ast.Div()))
handler_binop_floor_divide = register_handler('BINARY_FLOOR_DIVIDE', functools.partial(handle_binop, ast.FloorDiv()))
handler_binop_modulo = register_handler('BINARY_MODULO', functools.partial(handle_binop, ast.Mod()))
handler_binop_power = register_handler('BINARY_POWER', functools.partial(handle_binop, ast.Pow()))
handler_binop_lshift = register_handler('BINARY_LSHIFT', functools.partial(handle_binop, ast.LShift()))
handler_binop_rshift = register_handler('BINARY_RSHIFT', functools.partial(handle_binop, ast.RShift()))
handler_binop_or = register_handler('BINARY_OR', functools.partial(handle_binop, ast.BitOr()))
handler_binop_xor = register_handler('BINARY_XOR', functools.partial(handle_binop, ast.BitXor()))
handler_binop_and = register_handler('BINARY_AND', functools.partial(handle_binop, ast.BitAnd()))


# ============================================================================
# UNARY OPERATION HANDLERS
# ============================================================================

def handle_unary_op(op: ast.unaryop, state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    operand = ensure_ast(state.stack[-1])
    new_stack = state.stack[:-1] + [ast.UnaryOp(op=op, operand=operand)]
    return replace(state, stack=new_stack)


handle_unary_negative = register_handler('UNARY_NEGATIVE', functools.partial(handle_unary_op, ast.USub()))
handle_unary_positive = register_handler('UNARY_POSITIVE', functools.partial(handle_unary_op, ast.UAdd()))
handle_unary_invert = register_handler('UNARY_INVERT', functools.partial(handle_unary_op, ast.Invert()))
handle_unary_not = register_handler('UNARY_NOT', functools.partial(handle_unary_op, ast.Not()))


# ============================================================================
# COMPARISON OPERATION HANDLERS
# ============================================================================

CMP_OPMAP: dict[str, ast.cmpop] = {
    '<': ast.Lt(),
    '<=': ast.LtE(),
    '>': ast.Gt(),
    '>=': ast.GtE(),
    '==': ast.Eq(),
    '!=': ast.NotEq(),
}


@register_handler('COMPARE_OP')
def handle_compare_op(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    assert dis.cmp_op[instr.arg] == instr.argval, f"Unsupported comparison operation: {instr.argval}"

    right = ensure_ast(state.stack[-1])
    left = ensure_ast(state.stack[-2])
    
    # Map comparison operation codes to AST operators
    op_name = instr.argval
    compare_node = ast.Compare(
        left=left,
        ops=[CMP_OPMAP[op_name]],
        comparators=[right]
    )
    new_stack = state.stack[:-2] + [compare_node]
    return replace(state, stack=new_stack)


@register_handler('CONTAINS_OP')
def handle_contains_op(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    right = ensure_ast(state.stack[-1])  # Container
    left = ensure_ast(state.stack[-2])   # Item to check
    
    # instr.arg determines if it's 'in' (0) or 'not in' (1)
    op = ast.NotIn() if instr.arg else ast.In()
    
    compare_node = ast.Compare(
        left=left,
        ops=[op],
        comparators=[right]
    )
    new_stack = state.stack[:-2] + [compare_node]
    return replace(state, stack=new_stack)


@register_handler('IS_OP')
def handle_is_op(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    right = ensure_ast(state.stack[-1])
    left = ensure_ast(state.stack[-2])
    
    # instr.arg determines if it's 'is' (0) or 'is not' (1)
    op = ast.IsNot() if instr.arg else ast.Is()
    
    compare_node = ast.Compare(
        left=left,
        ops=[op],
        comparators=[right]
    )
    new_stack = state.stack[:-2] + [compare_node]
    return replace(state, stack=new_stack)


# ============================================================================
# FUNCTION CALL HANDLERS
# ============================================================================

@register_handler('CALL_FUNCTION')
def handle_call_function(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # CALL_FUNCTION pops function and arguments from stack
    arg_count: int = instr.arg
    # Pop arguments and function
    args = [ensure_ast(arg) for arg in state.stack[-arg_count:]] if arg_count > 0 else []
    func = ensure_ast(state.stack[-arg_count - 1])
    new_stack = state.stack[:-arg_count - 1]

    if isinstance(func, CompLambda):
        assert len(args) == 1
        return replace(state, stack=new_stack + [func.inline(args[0])])
    else:
        # Create function call AST
        call_node = ast.Call(func=func, args=args, keywords=[])
        new_stack = new_stack + [call_node]
        return replace(state, stack=new_stack)
    

@register_handler('LOAD_METHOD')
def handle_load_method(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # LOAD_METHOD loads a method from an object
    # It pushes the bound method and the object (for the method call)
    obj = ensure_ast(state.stack[-1])
    method_name = instr.argval
    new_stack = state.stack[:-1]
    
    # Create method access as an attribute
    method_attr = ast.Attribute(value=obj, attr=method_name, ctx=ast.Load())
    
    # For LOAD_METHOD, we push both the method and the object
    # But for AST purposes, we just need the method attribute
    new_stack = new_stack + [method_attr]
    return replace(state, stack=new_stack)


@register_handler('CALL_METHOD')
def handle_call_method(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # CALL_METHOD calls a method - similar to CALL_FUNCTION but for methods
    arg_count = instr.arg
    # Pop arguments and method
    args = [ensure_ast(arg) for arg in state.stack[-arg_count:]] if arg_count > 0 else []
    method = ensure_ast(state.stack[-arg_count - 1])
    new_stack = state.stack[:-arg_count - 1]
    
    # Create method call AST
    call_node = ast.Call(func=method, args=args, keywords=[])
    new_stack = new_stack + [call_node]
    return replace(state, stack=new_stack)


@register_handler('MAKE_FUNCTION')
def handle_make_function(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # MAKE_FUNCTION creates a function from code object and name on stack
    assert isinstance(state.stack[-1], ast.Constant) and isinstance(state.stack[-1].value, str), "Function name must be a constant string."
    if instr.argrepr == 'closure':
        # This is a closure, remove the environment tuple from the stack for AST purposes
        new_stack = state.stack[:-3]
    elif instr.argrepr == '':
        new_stack = state.stack[:-2]
    else:
        raise NotImplementedError("MAKE_FUNCTION with defaults or annotations not implemented.")

    body: ast.expr = state.stack[-2]
    name: str = state.stack[-1].value

    if isinstance(body, CompExp) and sum(1 for x in ast.walk(body) if isinstance(x, DummyIterName)) == 1:
        return replace(state, stack=new_stack + [CompLambda(body)])
    else:
        raise NotImplementedError("Lambda reconstruction not implemented yet")


# ============================================================================
# OBJECT ACCESS HANDLERS  
# ============================================================================

@register_handler('LOAD_ATTR')
def handle_load_attr(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # LOAD_ATTR loads an attribute from the object on top of stack
    obj = ensure_ast(state.stack[-1])
    attr_name = instr.argval
    new_stack = state.stack[:-1]
    
    # Create attribute access AST
    attr_node = ast.Attribute(value=obj, attr=attr_name, ctx=ast.Load())
    new_stack = new_stack + [attr_node]
    return replace(state, stack=new_stack)


@register_handler('BINARY_SUBSCR')
def handle_binary_subscr(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # BINARY_SUBSCR implements obj[index] - pops index and obj from stack
    index = ensure_ast(state.stack[-1])  # Index is on top
    obj = ensure_ast(state.stack[-2])    # Object is below index
    new_stack = state.stack[:-2]
    
    # Create subscript access AST
    subscr_node = ast.Subscript(value=obj, slice=index, ctx=ast.Load())
    new_stack = new_stack + [subscr_node]
    return replace(state, stack=new_stack)


# ============================================================================
# OTHER CONTAINER BUILDING HANDLERS
# ============================================================================

@register_handler('BUILD_TUPLE')
def handle_build_tuple(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    tuple_size: int = instr.arg
    # Pop elements for the tuple
    elements = [ensure_ast(elem) for elem in state.stack[-tuple_size:]] if tuple_size > 0 else []
    new_stack = state.stack[:-tuple_size] if tuple_size > 0 else state.stack
    
    # Create tuple AST
    tuple_node = ast.Tuple(elts=elements, ctx=ast.Load())
    new_stack = new_stack + [tuple_node]
    return replace(state, stack=new_stack)


@register_handler('LIST_TO_TUPLE')
def handle_list_to_tuple(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # LIST_TO_TUPLE converts a list on the stack to a tuple
    list_obj = ensure_ast(state.stack[-1])
    assert isinstance(list_obj, ast.List), "Expected a list for LIST_TO_TUPLE"
    
    # Create tuple AST from the list's elements
    tuple_node = ast.Tuple(elts=list_obj.elts, ctx=ast.Load())
    new_stack = state.stack[:-1] + [tuple_node]
    return replace(state, stack=new_stack)


@register_handler('LIST_EXTEND')
def handle_list_extend(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # LIST_EXTEND extends the list at TOS-1 with the iterable at TOS
    iterable = ensure_ast(state.stack[-1])
    list_obj = state.stack[-2]  # This should be a list from BUILD_LIST
    new_stack = state.stack[:-2]
    
    # If the list is empty and we're extending with a tuple/iterable,
    # we can convert this to a simple list of the iterable's elements
    if isinstance(list_obj, ast.List) and len(list_obj.elts) == 0:
        # If extending with a constant tuple, expand it to list elements
        if isinstance(iterable, ast.Constant) and isinstance(iterable.value, tuple):
            elements = [ast.Constant(value=elem) for elem in iterable.value]
            list_node = ast.List(elts=elements, ctx=ast.Load())
            new_stack = new_stack + [list_node]
            return replace(state, stack=new_stack)
    
    # Fallback: create a list from the iterable using list() constructor
    list_call = ast.Call(
        func=ast.Name(id='list', ctx=ast.Load()),
        args=[iterable],
        keywords=[]
    )
    new_stack = new_stack + [list_call]
    return replace(state, stack=new_stack)


@register_handler('BUILD_CONST_KEY_MAP')
def handle_build_const_key_map(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # BUILD_CONST_KEY_MAP builds a dictionary with constant keys
    # The keys are in a tuple on TOS, values are on the stack below
    map_size: int = instr.arg
    # Pop the keys tuple and values
    keys_tuple: ast.Tuple = state.stack[-1]
    keys = [ensure_ast(key) for key in keys_tuple.elts]
    values = [ensure_ast(val) for val in state.stack[-map_size-1:-1]]
    new_stack = state.stack[:-map_size-1]
    
    # Create dictionary AST
    dict_node = ast.Dict(keys=keys, values=values)
    new_stack = new_stack + [dict_node]
    return replace(state, stack=new_stack)


# ============================================================================
# CONDITIONAL JUMP HANDLERS
# ============================================================================

@register_handler('POP_JUMP_IF_FALSE')
def handle_pop_jump_if_false(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # POP_JUMP_IF_FALSE pops a value from the stack and jumps if it's false
    # In comprehensions, this is used for filter conditions
    condition = ensure_ast(state.stack[-1])
    new_stack = state.stack[:-1]

    if instr.argval < instr.offset:
        # Jumping backward to loop start - this is a condition
        # When POP_JUMP_IF_FALSE jumps back, it means "if false, skip this item"
        # So we need to negate the condition to get the filter condition
        assert isinstance(state.result, CompExp) and state.result.generators
        updated_loop = ast.comprehension(
            target=state.result.generators[-1].target,
            iter=state.result.generators[-1].iter,
            ifs=state.result.generators[-1].ifs + [condition],
            is_async=state.result.generators[-1].is_async,
        )
        if isinstance(state.result, ast.DictComp):
            new_ret = ast.DictComp(
                key=state.result.key,
                value=state.result.value,
                generators=state.result.generators[:-1] + [updated_loop],
            )
        else:
            new_ret = type(state.result)(
                elt=state.result.elt,
                generators=state.result.generators[:-1] + [updated_loop],
            )
        return replace(state, stack=new_stack, result=new_ret)
    else:
        raise NotImplementedError("Lazy and+or behavior not implemented yet")


@register_handler('POP_JUMP_IF_TRUE')
def handle_pop_jump_if_true(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
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
        # negate the condition
        condition = ast.UnaryOp(op=ast.Not(), operand=condition)
        updated_loop = ast.comprehension(
            target=state.result.generators[-1].target,
            iter=state.result.generators[-1].iter,
            ifs=state.result.generators[-1].ifs + [condition],
            is_async=state.result.generators[-1].is_async,
        )
        if isinstance(state.result, ast.DictComp):
            new_ret = ast.DictComp(
                key=state.result.key,
                value=state.result.value,
                generators=state.result.generators[:-1] + [updated_loop],
            )
        else:
            new_ret = type(state.result)(
                elt=state.result.elt,
                generators=state.result.generators[:-1] + [updated_loop],
            )
        return replace(state, stack=new_stack, result=new_ret)
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
    if len(value) > 0 and value[0] == 'dict_item':
        return ast.Tuple(
            elts=[ensure_ast(value[1]), ensure_ast(value[2])],
            ctx=ast.Load()
        )
    else:
        return ast.Tuple(elts=[ensure_ast(v) for v in value], ctx=ast.Load())


@ensure_ast.register(type(iter((1,))))
def _ensure_ast_tuple_iterator(value: Iterator) -> ast.Tuple:
    return ensure_ast(tuple(value.__reduce__()[1][0]))


@ensure_ast.register
def _ensure_ast_list(value: list) -> ast.List:
    return ast.List(elts=[ensure_ast(v) for v in value], ctx=ast.Load())


@ensure_ast.register(type(iter([1])))
def _ensure_ast_list_iterator(value: Iterator) -> ast.List:
    return ensure_ast(list(value.__reduce__()[1][0]))


@ensure_ast.register
def _ensure_ast_set(value: set) -> ast.Set:
    return ast.Set(elts=[ensure_ast(v) for v in value])


@ensure_ast.register(type(iter({1})))
def _ensure_ast_set_iterator(value: Iterator) -> ast.Set:
    return ensure_ast(set(value.__reduce__()[1][0]))


@ensure_ast.register
def _ensure_ast_dict(value: dict) -> ast.Dict:
    return ast.Dict(
        keys=[ensure_ast(k) for k in value.keys()],
        values=[ensure_ast(v) for v in value.values()]
    )


@ensure_ast.register(type(iter({1: 2})))
def _ensure_ast_dict_iterator(value: Iterator) -> ast.Dict:
    return ensure_ast(value.__reduce__()[1][0])


@ensure_ast.register
def _ensure_ast_range(value: range) -> ast.Call:
    return ast.Call(
        func=ast.Name(id='range', ctx=ast.Load()),
        args=[ensure_ast(value.start), ensure_ast(value.stop), ensure_ast(value.step)],
        keywords=[]
    )


@ensure_ast.register(type(iter(range(1))))
def _ensure_ast_range_iterator(value: Iterator) -> ast.Call:
    return ensure_ast(value.__reduce__()[1][0])


@ensure_ast.register
def _ensure_ast_codeobj(value: types.CodeType) -> ast.expr:
    # Symbolic execution to reconstruct the AST
    state = ReconstructionState()
    for instr in dis.get_instructions(value):
        state = OP_HANDLERS[instr.opname](state, instr)

    # Check postconditions
    assert not any(isinstance(x, Placeholder) for x in ast.walk(state.result)), "Return value must not contain placeholders"
    assert not isinstance(state.result, CompExp) or len(state.result.generators) > 0, "Return value must have generators if not a lambda"
    return state.result


@ensure_ast.register
def _ensure_ast_lambda(value: types.LambdaType) -> ast.Lambda:
    assert inspect.isfunction(value) and value.__name__.endswith("<lambda>"), "Input must be a lambda function"

    code: types.CodeType = value.__code__
    body: ast.expr = ensure_ast(code)
    args = ast.arguments(
        posonlyargs=[ast.arg(arg=arg) for arg in code.co_varnames[:code.co_posonlyargcount]],
        args=[ast.arg(arg=arg) for arg in code.co_varnames[code.co_posonlyargcount:code.co_argcount]],
        kwonlyargs=[ast.arg(arg=arg) for arg in code.co_varnames[code.co_argcount:code.co_argcount + code.co_kwonlyargcount]],
        kw_defaults=[],
        defaults=[],
    )
    return ast.Lambda(args=args, body=body)


@ensure_ast.register
def _ensure_ast_genexpr(genexpr: types.GeneratorType) -> ast.GeneratorExp:
    assert inspect.isgenerator(genexpr), "Input must be a generator expression"
    assert inspect.getgeneratorstate(genexpr) == inspect.GEN_CREATED, "Generator must be in created state"
    genexpr_ast: ast.GeneratorExp = ensure_ast(genexpr.gi_code)
    geniter_ast: ast.expr = ensure_ast(genexpr.gi_frame.f_locals['.0'])
    result = CompLambda(genexpr_ast).inline(geniter_ast)
    assert inspect.getgeneratorstate(genexpr) == inspect.GEN_CREATED, "Generator must stay in created state"
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
