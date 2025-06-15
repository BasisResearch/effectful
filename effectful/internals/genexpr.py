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
    >>> # ast_node is now an ast.GeneratorExp representing the original expression
"""

import ast
import dis
import functools
import inspect
import types
import typing
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace


CompExp = ast.GeneratorExp | ast.ListComp | ast.SetComp | ast.DictComp


class Placeholder(ast.Name):
    """Placeholder for AST nodes that are not yet resolved."""
    def __init__(self):
        super().__init__(id=".PLACEHOLDER", ctx=ast.Load())


class IterDummyName(ast.Name):
    """Dummy name for the iterator variable in generator expressions."""
    def __init__(self):
        super().__init__(id=".0", ctx=ast.Load())


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
        stack: Simulates the Python VM's value stack. Contains AST nodes or
               values that would be on the stack during execution. Operations
               like LOAD_FAST push to this stack, while operations like
               BINARY_ADD pop operands and push results.
               
        pending_conditions: Filter conditions that haven't been assigned to
                           a loop yet. Some bytecode patterns require collecting
                           conditions before knowing which loop they belong to.
                           
        or_conditions: Conditions that are part of an OR expression. These
                      need to be combined with ast.BoolOp(op=ast.Or()).
    """
    ret: ast.Lambda | ast.GeneratorExp | ast.ListComp | ast.SetComp | ast.DictComp
    stack: list[ast.expr] = field(default_factory=list)  # Stack of AST nodes or values
    pending_conditions: list[ast.expr] = field(default_factory=list)
    or_conditions: list[ast.expr] = field(default_factory=list)


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
    
    if opname in OP_HANDLERS:
        raise ValueError(f"Handler for '{opname}' already exists.")

    @functools.wraps(handler)
    def _wrapper(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
        assert instr.opname == opname, f"Handler for '{opname}' called with wrong instruction"
        return handler(state, instr)
    
    OP_HANDLERS[opname] = _wrapper
    return _wrapper


# ============================================================================
# GENERATOR COMPREHENSION HANDLERS
# ============================================================================

@register_handler('GEN_START')
def handle_gen_start(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # GEN_START is typically the first instruction in generator expressions
    # It initializes the generator
    assert isinstance(state.ret, ast.GeneratorExp)
    assert isinstance(state.ret.elt, Placeholder), "GEN_START must be called before yielding"
    assert len(state.ret.generators) == 0, "GEN_START should not have generators yet"
    return state


@register_handler('YIELD_VALUE')
def handle_yield_value(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # YIELD_VALUE pops a value from the stack and yields it
    # This is the expression part of the generator
    assert isinstance(state.ret, ast.GeneratorExp), "YIELD_VALUE must be called after GEN_START"
    assert isinstance(state.ret.elt, Placeholder), "YIELD_VALUE must be called before yielding"
    assert len(state.ret.generators) > 0, "YIELD_VALUE should have generators"

    new_stack = state.stack[:-1]
    ret = ast.GeneratorExp(
        elt=ensure_ast(state.stack[-1]),
        generators=state.ret.generators,
    )
    return replace(state, stack=new_stack, ret=ret)


# ============================================================================
# LIST COMPREHENSION HANDLERS
# ============================================================================

@register_handler('BUILD_LIST')
def handle_build_list(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    list_size: int = instr.arg
    # Pop elements for the list
    elements = [ensure_ast(elem) for elem in state.stack[-list_size:]] if list_size > 0 else []
    new_stack = state.stack[:-list_size] if list_size > 0 else state.stack
    
    # Create list AST
    list_node = ast.List(elts=elements, ctx=ast.Load())
    new_stack = new_stack + [list_node]
    return replace(state, stack=new_stack)


@register_handler('LIST_APPEND')
def handle_list_append(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    assert isinstance(state.ret, ast.ListComp), "LIST_APPEND must be called within a ListComp context"
    new_stack = state.stack[:-1]
    new_ret = ast.ListComp(
        elt=ensure_ast(state.stack[-1]),
        generators=state.ret.generators,
    )
    return replace(state, stack=new_stack, ret=new_ret)


# ============================================================================
# SET COMPREHENSION HANDLERS
# ============================================================================

@register_handler('BUILD_SET')
def handle_build_set(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    raise NotImplementedError("BUILD_SET not implemented yet")  # TODO


@register_handler('SET_ADD')
def handle_set_add(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    assert isinstance(state.ret, ast.SetComp), "SET_ADD must be called after BUILD_SET"
    new_stack = state.stack[:-1]
    new_ret = ast.SetComp(
        elt=ensure_ast(state.stack[-1]),
        generators=state.ret.generators,
    )
    return replace(state, stack=new_stack, ret=new_ret)


# ============================================================================
# DICT COMPREHENSION HANDLERS
# ============================================================================

@register_handler('BUILD_MAP')
def handle_build_map(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    raise NotImplementedError("BUILD_MAP not implemented yet")  # TODO


@register_handler('MAP_ADD')
def handle_map_add(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    assert isinstance(state.ret, ast.DictComp), "MAP_ADD must be called after BUILD_MAP"
    new_stack = state.stack[:-2]
    new_ret = ast.DictComp(
        key=ensure_ast(state.stack[-2]),
        value=ensure_ast(state.stack[-1]),
        generators=state.ret.generators,
    )
    return replace(state, stack=new_stack, ret=new_ret)


# ============================================================================
# LOOP CONTROL HANDLERS
# ============================================================================

@register_handler('RETURN_VALUE')
def handle_return_value(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # RETURN_VALUE ends the generator
    # Usually preceded by LOAD_CONST None
    if isinstance(state.ret, CompExp):
        return replace(state, stack=state.stack[:-1])
    elif isinstance(state.ret, ast.Lambda):
        raise NotImplementedError("Lambda reconstruction not implemented yet")


@register_handler('FOR_ITER')
def handle_for_iter(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # FOR_ITER pops an iterator from the stack and pushes the next item
    # If the iterator is exhausted, it jumps to the target instruction
    assert len(state.stack) > 0, "FOR_ITER must have an iterator on the stack"
    assert isinstance(state.ret, CompExp), "FOR_ITER must be called within a comprehension context"

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
    if isinstance(state.ret, ast.DictComp):
        # If it's a DictComp, we need to ensure the loop is added to the dict comprehension
        new_ret = ast.DictComp(
            key=state.ret.key,
            value=state.ret.value,
            generators=state.ret.generators + [loop_info],
        )
    else:
        new_ret = type(state.ret)(
            elt=state.ret.elt,
            generators=state.ret.generators + [loop_info],
        )

    return replace(state, stack=new_stack, ret=new_ret)


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


# ============================================================================
# VARIABLE OPERATIONS HANDLERS
# ============================================================================

@register_handler('LOAD_FAST')
def handle_load_fast(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    var_name: str = instr.argval
    
    if var_name == '.0':
        # Special handling for .0 variable (the iterator)
        new_stack = state.stack + [IterDummyName()]
    else:
        # Regular variable load
        new_stack = state.stack + [ast.Name(id=var_name, ctx=ast.Load())]
    
    return replace(state, stack=new_stack)


@register_handler('STORE_FAST')
def handle_store_fast(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    assert isinstance(state.ret, CompExp), "STORE_FAST must be called within a comprehension context"
    var_name = instr.argval
    
    # Update the most recent loop's target variable
    assert len(state.ret.generators) > 0, "STORE_FAST must be within a loop context"

    # Create a new LoopInfo with updated target
    updated_loop = ast.comprehension(
        target=ast.Name(id=var_name, ctx=ast.Store()),
        iter=state.ret.generators[-1].iter,
        ifs=state.ret.generators[-1].ifs,
        is_async=state.ret.generators[-1].is_async
    )

    # Update the last loop in the generators list
    if isinstance(state.ret, ast.DictComp):
        new_ret = ast.DictComp(
            key=state.ret.key,
            value=state.ret.value,
            generators=state.ret.generators[:-1] + [updated_loop],
        )
    else:
        new_ret = type(state.ret)(
            elt=state.ret.elt,
            generators=state.ret.generators[:-1] + [updated_loop],
        )

    # Create new loops list with the updated loop
    return replace(state, ret=new_ret)


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
# ARITHMETIC/LOGIC HANDLERS
# ============================================================================

@register_handler('BINARY_ADD')
def handle_binary_add(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    right = ensure_ast(state.stack[-1])
    left = ensure_ast(state.stack[-2])
    new_stack = state.stack[:-2] + [ast.BinOp(left=left, op=ast.Add(), right=right)]
    return replace(state, stack=new_stack)


@register_handler('BINARY_SUBTRACT')
def handle_binary_subtract(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    right = ensure_ast(state.stack[-1])
    left = ensure_ast(state.stack[-2])
    new_stack = state.stack[:-2] + [ast.BinOp(left=left, op=ast.Sub(), right=right)]
    return replace(state, stack=new_stack)


@register_handler('BINARY_MULTIPLY')
def handle_binary_multiply(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    right = ensure_ast(state.stack[-1])
    left = ensure_ast(state.stack[-2])
    new_stack = state.stack[:-2] + [ast.BinOp(left=left, op=ast.Mult(), right=right)]
    return replace(state, stack=new_stack)


@register_handler('BINARY_TRUE_DIVIDE')
def handle_binary_true_divide(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    right = ensure_ast(state.stack[-1])
    left = ensure_ast(state.stack[-2])
    new_stack = state.stack[:-2] + [ast.BinOp(left=left, op=ast.Div(), right=right)]
    return replace(state, stack=new_stack)


@register_handler('BINARY_FLOOR_DIVIDE')
def handle_binary_floor_divide(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    right = ensure_ast(state.stack[-1])
    left = ensure_ast(state.stack[-2])
    new_stack = state.stack[:-2] + [ast.BinOp(left=left, op=ast.FloorDiv(), right=right)]
    return replace(state, stack=new_stack)


@register_handler('BINARY_MODULO')
def handle_binary_modulo(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    right = ensure_ast(state.stack[-1])
    left = ensure_ast(state.stack[-2])
    new_stack = state.stack[:-2] + [ast.BinOp(left=left, op=ast.Mod(), right=right)]
    return replace(state, stack=new_stack)


@register_handler('BINARY_POWER')
def handle_binary_power(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    right = ensure_ast(state.stack[-1])
    left = ensure_ast(state.stack[-2])
    new_stack = state.stack[:-2] + [ast.BinOp(left=left, op=ast.Pow(), right=right)]
    return replace(state, stack=new_stack)


@register_handler('BINARY_LSHIFT')
def handle_binary_lshift(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    right = ensure_ast(state.stack[-1])
    left = ensure_ast(state.stack[-2])
    new_stack = state.stack[:-2] + [ast.BinOp(left=left, op=ast.LShift(), right=right)]
    return replace(state, stack=new_stack)


@register_handler('BINARY_RSHIFT')
def handle_binary_rshift(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    right = ensure_ast(state.stack[-1])
    left = ensure_ast(state.stack[-2])
    new_stack = state.stack[:-2] + [ast.BinOp(left=left, op=ast.RShift(), right=right)]
    return replace(state, stack=new_stack)


@register_handler('BINARY_OR')
def handle_binary_or(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    right = ensure_ast(state.stack[-1])
    left = ensure_ast(state.stack[-2])
    new_stack = state.stack[:-2] + [ast.BinOp(left=left, op=ast.BitOr(), right=right)]
    return replace(state, stack=new_stack)


@register_handler('BINARY_XOR')
def handle_binary_xor(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    right = ensure_ast(state.stack[-1])
    left = ensure_ast(state.stack[-2])
    new_stack = state.stack[:-2] + [ast.BinOp(left=left, op=ast.BitXor(), right=right)]
    return replace(state, stack=new_stack)


@register_handler('BINARY_AND')
def handle_binary_and(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    right = ensure_ast(state.stack[-1])
    left = ensure_ast(state.stack[-2])
    new_stack = state.stack[:-2] + [ast.BinOp(left=left, op=ast.BitAnd(), right=right)]
    return replace(state, stack=new_stack)


# ============================================================================
# UNARY OPERATION HANDLERS
# ============================================================================

@register_handler('UNARY_NEGATIVE')
def handle_unary_negative(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    operand = ensure_ast(state.stack[-1])
    new_stack = state.stack[:-1] + [ast.UnaryOp(op=ast.USub(), operand=operand)]
    return replace(state, stack=new_stack)


@register_handler('UNARY_POSITIVE')
def handle_unary_positive(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    operand = ensure_ast(state.stack[-1])
    new_stack = state.stack[:-1] + [ast.UnaryOp(op=ast.UAdd(), operand=operand)]
    return replace(state, stack=new_stack)


@register_handler('UNARY_INVERT')
def handle_unary_invert(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    operand = ensure_ast(state.stack[-1])
    new_stack = state.stack[:-1] + [ast.UnaryOp(op=ast.Invert(), operand=operand)]
    return replace(state, stack=new_stack)


@register_handler('UNARY_NOT')
def handle_unary_not(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    operand = ensure_ast(state.stack[-1])
    new_stack = state.stack[:-1] + [ast.UnaryOp(op=ast.Not(), operand=operand)]
    return replace(state, stack=new_stack)


# ============================================================================
# FUNCTION CALL HANDLERS
# ============================================================================

@register_handler('CALL_FUNCTION')
def handle_call_function(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # CALL_FUNCTION pops function and arguments from stack
    arg_count = instr.arg
    # Pop arguments and function
    args = [ensure_ast(arg) for arg in state.stack[-arg_count:]] if arg_count > 0 else []
    func = ensure_ast(state.stack[-arg_count - 1])
    new_stack = state.stack[:-arg_count - 1]
    
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
    assert instr.arg == 0, "MAKE_FUNCTION with defaults or annotations not allowed."

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
# COMPARISON HANDLERS
# ============================================================================

@register_handler('COMPARE_OP')
def handle_compare_op(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    right = ensure_ast(state.stack[-1])
    left = ensure_ast(state.stack[-2])
    
    # Map comparison operation codes to AST operators
    op_map = {
        '<': ast.Lt(),
        '<=': ast.LtE(),
        '>': ast.Gt(),
        '>=': ast.GtE(),
        '==': ast.Eq(),
        '!=': ast.NotEq(),
        'in': ast.In(),
        'not in': ast.NotIn(),
        'is': ast.Is(),
        'is not': ast.IsNot(),
    }
    assert instr.argval in op_map, f"Unsupported comparison operation: {instr.argval}"
    
    op_name = instr.argval
    compare_node = ast.Compare(
        left=left,
        ops=[op_map[op_name]],
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
# CONDITIONAL JUMP HANDLERS
# ============================================================================

@register_handler('POP_JUMP_IF_FALSE')
def handle_pop_jump_if_false(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # POP_JUMP_IF_FALSE pops a value from the stack and jumps if it's false
    # In comprehensions, this is used for filter conditions
    condition = ensure_ast(state.stack[-1])
    new_stack = state.stack[:-1]
    
    # If we have pending OR conditions, this is the final condition in an OR expression
    if state.or_conditions:
        # Combine all OR conditions into a single BoolOp
        all_or_conditions = state.or_conditions + [condition]
        combined_condition = ast.BoolOp(op=ast.Or(), values=all_or_conditions)
        
        # Add the combined condition to the loop and clear OR conditions
        if isinstance(state.ret, CompExp) and state.ret.generators:
            updated_loop = ast.comprehension(
                target=state.ret.generators[-1].target,
                iter=state.ret.generators[-1].iter,
                ifs=state.ret.generators[-1].ifs + [combined_condition],
                is_async=state.ret.generators[-1].is_async,
            )
            if isinstance(state.ret, ast.DictComp):
                new_ret = ast.DictComp(
                    key=state.ret.key,
                    value=state.ret.value,
                    generators=state.ret.generators[:-1] + [updated_loop],
                )
            else:
                new_ret = type(state.ret)(
                    elt=state.ret.elt,
                    generators=state.ret.generators[:-1] + [updated_loop],
                )
            return replace(state, stack=new_stack, ret=new_ret, or_conditions=[])
        else:
            new_pending = state.pending_conditions + [combined_condition]
            return replace(state, stack=new_stack, pending_conditions=new_pending, or_conditions=[])
    else:
        # Regular condition - add to the most recent loop
        if isinstance(state.ret, CompExp) and state.ret.generators:
            updated_loop = ast.comprehension(
                target=state.ret.generators[-1].target,
                iter=state.ret.generators[-1].iter,
                ifs=state.ret.generators[-1].ifs + [condition],
                is_async=state.ret.generators[-1].is_async,
            )
            if isinstance(state.ret, ast.DictComp):
                new_ret = ast.DictComp(
                    key=state.ret.key,
                    value=state.ret.value,
                    generators=state.ret.generators[:-1] + [updated_loop],
                )
            else:
                new_ret = type(state.ret)(
                    elt=state.ret.elt,
                    generators=state.ret.generators[:-1] + [updated_loop],
                )
            return replace(state, stack=new_stack, ret=new_ret)
        else:
            # If no loops yet, add to pending conditions
            new_pending = state.pending_conditions + [condition]
            return replace(state, stack=new_stack, pending_conditions=new_pending)


@register_handler('POP_JUMP_IF_TRUE')
def handle_pop_jump_if_true(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # POP_JUMP_IF_TRUE pops a value from the stack and jumps if it's true
    # This can be:
    # 1. Part of an OR expression (jump to YIELD_VALUE)
    # 2. A negated condition like "not x % 2" (jump back to loop start)
    condition = ensure_ast(state.stack[-1])
    new_stack = state.stack[:-1]
    
    # Check if this jumps forward (to YIELD_VALUE - OR pattern) vs back to loop (NOT pattern)
    # In OR: POP_JUMP_IF_TRUE jumps forward to yield the value
    # In NOT: POP_JUMP_IF_TRUE jumps back to skip this iteration
    if instr.argval > instr.offset:
        # Jumping forward - part of an OR expression
        new_or_conditions = state.or_conditions + [condition]
        return replace(state, stack=new_stack, or_conditions=new_or_conditions)
    else:
        # Jumping backward to loop start - this is a negated condition
        # When POP_JUMP_IF_TRUE jumps back, it means "if true, skip this item"
        # So we need to negate the condition to get the filter condition
        negated_condition = ast.UnaryOp(op=ast.Not(), operand=condition)
        
        if isinstance(state.ret, CompExp) and state.ret.generators:
            updated_loop = ast.comprehension(
                target=state.ret.generators[-1].target,
                iter=state.ret.generators[-1].iter,
                ifs=state.ret.generators[-1].ifs + [negated_condition],
                is_async=state.ret.generators[-1].is_async,
            )
            if isinstance(state.ret, ast.DictComp):
                new_ret = ast.DictComp(
                    key=state.ret.key,
                    value=state.ret.value,
                    generators=state.ret.generators[:-1] + [updated_loop],
                )
            else:
                new_ret = type(state.ret)(
                    elt=state.ret.elt,
                    generators=state.ret.generators[:-1] + [updated_loop],
                )
            return replace(state, stack=new_stack, ret=new_ret)
        else:
            new_pending = state.pending_conditions + [negated_condition]
            return replace(state, stack=new_stack, pending_conditions=new_pending)


# ============================================================================
# UNPACKING HANDLERS
# ============================================================================

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
def _ensure_ast_tuple_iterator(value: Iterator) -> ast.expr:
    return ensure_ast(tuple(value.__reduce__()[1][0]))


@ensure_ast.register
def _ensure_ast_list(value: list) -> ast.List:
    return ast.List(elts=[ensure_ast(v) for v in value], ctx=ast.Load())


@ensure_ast.register(type(iter([1])))
def _ensure_ast_list_iterator(value: Iterator) -> ast.expr:
    return ensure_ast(list(value.__reduce__()[1][0]))


@ensure_ast.register
def _ensure_ast_set(value: set) -> ast.Set:
    return ast.Set(elts=[ensure_ast(v) for v in value])


@ensure_ast.register(type(iter({1})))
def _ensure_ast_set_iterator(value: Iterator) -> ast.expr:
    return ensure_ast(set(value.__reduce__()[1][0]))


@ensure_ast.register
def _ensure_ast_dict(value: dict) -> ast.Dict:
    return ast.Dict(
        keys=[ensure_ast(k) for k in value.keys()],
        values=[ensure_ast(v) for v in value.values()]
    )


@ensure_ast.register(type(iter({1: 2})))
def _ensure_ast_dict_iterator(value: Iterator) -> ast.expr:
    return ensure_ast(value.__reduce__()[1][0])


@ensure_ast.register
def _ensure_ast_range(value: range) -> ast.Call:
    return ast.Call(
        func=ast.Name(id='range', ctx=ast.Load()),
        args=[ensure_ast(value.start), ensure_ast(value.stop), ensure_ast(value.step)],
        keywords=[]
    )


@ensure_ast.register(type(iter(range(1))))
def _ensure_ast_range_iterator(value: Iterator) -> ast.expr:
    return ensure_ast(value.__reduce__()[1][0])


@ensure_ast.register
def _ensure_ast_codeobj(value: types.CodeType) -> CompExp | ast.Lambda:
    # Determine return type based on the first instruction
    ret: CompExp | ast.Lambda
    instructions = list(dis.get_instructions(value))
    if instructions[0].opname == 'GEN_START' and instructions[1].opname == 'LOAD_FAST' and instructions[1].argval == '.0':
        ret = ast.GeneratorExp(elt=Placeholder(), generators=[])
    elif instructions[0].opname == 'BUILD_LIST' and instructions[1].opname == 'LOAD_FAST' and instructions[1].argval == '.0':
        ret = ast.ListComp(elt=Placeholder(), generators=[])
    elif instructions[0].opname == 'BUILD_SET' and instructions[1].opname == 'LOAD_FAST' and instructions[1].argval == '.0':
        ret = ast.SetComp(elt=Placeholder(), generators=[])
    elif instructions[0].opname == 'BUILD_MAP' and instructions[1].opname == 'LOAD_FAST' and instructions[1].argval == '.0':
        ret = ast.DictComp(key=Placeholder(), value=Placeholder(), generators=[])
    elif instructions[0].opname in {'BUILD_LIST', 'BUILD_SET', 'BUILD_MAP'}:
        raise NotImplementedError("Unpacking construction not implemented yet")
    elif instructions[-1].opname == 'RETURN_VALUE':
        # not a comprehension, assume it's a lambda
        ret = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                vararg=None,
                kwonlyargs=[],
                kwarg=None,
                defaults=[],
                kw_defaults=[],
            ),
            body=Placeholder()
        )
    else:
        raise TypeError("Code type from unsupported source")

    # Symbolic execution to reconstruct the AST
    state = ReconstructionState(ret=ret)
    for instr in instructions:
        state = OP_HANDLERS[instr.opname](state, instr)

    # Check postconditions
    assert not any(isinstance(x, Placeholder) for x in ast.walk(state.ret)), "Return value must not contain placeholders"
    assert isinstance(state.ret, ast.Lambda) or len(state.ret.generators) > 0, "Return value must have generators if not a lambda"
    return state.ret


# ============================================================================
# MAIN RECONSTRUCTION FUNCTION
# ============================================================================

@ensure_ast.register
def _ensure_ast_lambda(value: types.LambdaType) -> ast.Lambda:
    assert inspect.isfunction(value), "Input must be a lambda function"
    raise NotImplementedError("Lambda reconstruction not implemented yet")


@ensure_ast.register
def reconstruct(genexpr: types.GeneratorType) -> ast.GeneratorExp:
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
        genexpr (GeneratorType): The generator object to analyze. Must be
            a freshly created generator that has not been iterated yet
            (in 'GEN_CREATED' state).
    
    Returns:
        ast.GeneratorExp: An AST node representing the reconstructed comprehension.
            The specific type depends on the original comprehension:
    
    Raises:
        AssertionError: If the input is not a generator or if the generator
            has already been started (not in 'GEN_CREATED' state).
    
    Example:
        >>> # Generator expression
        >>> g = (x * 2 for x in range(10) if x % 2 == 0)
        >>> ast_node = reconstruct(g)
        >>> isinstance(ast_node, ast.GeneratorExp)
        True
        
        >>> # The reconstructed AST can be compiled and evaluated
        >>> import ast
        >>> code = compile(ast.Expression(body=ast_node), '<string>', 'eval')
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
    assert inspect.getgeneratorstate(genexpr) == inspect.GEN_CREATED, "Generator must be in created state"
    genexpr_ast: ast.GeneratorExp = ensure_ast(genexpr.gi_code)
    assert isinstance(genexpr_ast.generators[0].iter, IterDummyName)
    assert len([x for x in ast.walk(genexpr_ast) if isinstance(x, IterDummyName)]) == 1
    genexpr_ast.generators[0].iter = ensure_ast(genexpr.gi_frame.f_locals['.0'])
    return genexpr_ast
