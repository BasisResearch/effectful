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
from types import GeneratorType, FunctionType
from typing import Callable, Any, List, Dict, Iterator, Optional, Union
from dataclasses import dataclass, field, replace


# Categories for organizing opcodes
CATEGORIES = {
    'Core Generator': {
        'GEN_START', 'YIELD_VALUE', 'RETURN_VALUE'
    },
    
    'Loop Control': {
        'GET_ITER', 'FOR_ITER', 'JUMP_ABSOLUTE', 'JUMP_FORWARD',
        'POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE'
    },
    
    'Variable Operations': {
        'LOAD_FAST', 'STORE_FAST', 'LOAD_GLOBAL', 'LOAD_DEREF', 'STORE_DEREF',
        'LOAD_CONST', 'LOAD_NAME', 'STORE_NAME'
    },
    
    'Arithmetic/Logic': {
        'BINARY_ADD', 'BINARY_SUBTRACT', 'BINARY_MULTIPLY', 'BINARY_TRUE_DIVIDE',
        'BINARY_FLOOR_DIVIDE', 'BINARY_MODULO', 'BINARY_POWER', 'BINARY_LSHIFT',
        'BINARY_RSHIFT', 'BINARY_OR', 'BINARY_XOR', 'BINARY_AND',
        'UNARY_POSITIVE', 'UNARY_NEGATIVE', 'UNARY_NOT', 'UNARY_INVERT'
    },
    
    'Comparisons': {
        'COMPARE_OP'
    },
    
    'Object Access': {
        'LOAD_ATTR', 'BINARY_SUBSCR', 'BUILD_SLICE', 'STORE_ATTR',
        'STORE_SUBSCR', 'DELETE_SUBSCR'
    },
    
    'Function Calls': {
        'CALL_FUNCTION', 'CALL_FUNCTION_KW', 'CALL_FUNCTION_EX', 'CALL_METHOD',
        'LOAD_METHOD', 'CALL', 'PRECALL'
    },
    
    'Container Building': {
        'BUILD_TUPLE', 'BUILD_LIST', 'BUILD_SET', 'BUILD_MAP',
        'BUILD_STRING', 'FORMAT_VALUE', 'LIST_APPEND', 'SET_ADD', 'MAP_ADD',
        'BUILD_CONST_KEY_MAP'
    },
    
    'Stack Management': {
        'POP_TOP', 'DUP_TOP', 'ROT_TWO', 'ROT_THREE', 'ROT_FOUR',
        'COPY', 'SWAP'
    },
    
    'Unpacking': {
        'UNPACK_SEQUENCE', 'UNPACK_EX'
    },

    'Other': {
        'NOP', 'EXTENDED_ARG', 'CACHE', 'RESUME', 'MAKE_CELL'
    }
}

OP_CATEGORIES: dict[str, str] = {op: category for category, ops in CATEGORIES.items() for op in ops}


@dataclass(frozen=True)
class LoopInfo:
    """Information about a single loop in a comprehension.
    
    This class stores all the components needed to reconstruct a single 'for' clause
    in a comprehension expression. In Python, comprehensions can have multiple
    nested loops, and each loop can have zero or more filter conditions.
    
    For example, in the comprehension:
        [x*y for x in range(3) for y in range(4) if x < y if x + y > 2]
    
    There would be two LoopInfo objects:
    1. First loop: target='x', iter_ast=range(3), conditions=[]
    2. Second loop: target='y', iter_ast=range(4), conditions=[x < y, x + y > 2]
    
    Attributes:
        target: The loop variable(s) as an AST node. Usually an ast.Name node
                (e.g., 'x'), but can also be a tuple for unpacking 
                (e.g., '(i, j)' in 'for i, j in pairs').
        iter_ast: The iterator expression as an AST node. This is what comes
                  after 'in' in the for clause (e.g., range(3), list_var, etc).
        conditions: List of filter expressions (if clauses) that apply to this
                    loop level. Each condition is an AST node representing a
                    boolean expression.
    """
    target: ast.AST  # The loop variable(s) as AST node
    iter_ast: ast.AST  # The iterator as AST node
    conditions: List[ast.AST] = field(default_factory=list)  # if conditions as AST nodes


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
               
        loops: List of LoopInfo objects representing the comprehension's loops.
               Built up as FOR_ITER instructions are encountered. The order
               matters - outer loops come before inner loops.
               
        comprehension_type: Type of comprehension being built. Defaults to
                           'generator' but can be 'list', 'set', or 'dict'.
                           This affects which AST node type is ultimately created.
                           
        expression: The main expression that gets yielded/collected. For example,
                   in '[x*2 for x in items]', this would be the AST for 'x*2'.
                   Captured when YIELD_VALUE is encountered.
                   
        key_expression: For dict comprehensions only - the key part of the
                       key:value pair. In '{k: v for k,v in items}', this
                       would be the AST for 'k'.
                       
        code_obj: The code object being analyzed (from generator.gi_code).
                 Contains the bytecode and other metadata like variable names.
                 
        frame: The generator's frame object (from generator.gi_frame).
               Provides access to the runtime state, including local variables
               like the '.0' iterator variable.
               
        current_loop_var: Name of the most recently stored loop variable.
                         Helps track which variable is being used in the
                         current loop context.
                         
        pending_conditions: Filter conditions that haven't been assigned to
                           a loop yet. Some bytecode patterns require collecting
                           conditions before knowing which loop they belong to.
                           
        or_conditions: Conditions that are part of an OR expression. These
                      need to be combined with ast.BoolOp(op=ast.Or()).
    """
    stack: List[Any] = field(default_factory=list)  # Stack of AST nodes or values
    loops: List[LoopInfo] = field(default_factory=list)
    comprehension_type: str = 'generator'  # 'generator', 'list', 'set', 'dict'
    expression: Optional[ast.AST] = None  # Main expression being yielded
    key_expression: Optional[ast.AST] = None  # For dict comprehensions
    code_obj: Any = None
    frame: Any = None
    current_loop_var: Optional[str] = None  # Track current loop variable
    pending_conditions: List[ast.AST] = field(default_factory=list)
    or_conditions: List[ast.AST] = field(default_factory=list)


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
# CORE GENERATOR HANDLERS
# ============================================================================

@register_handler('GEN_START')
def handle_gen_start(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # GEN_START is typically the first instruction in generator expressions
    # It initializes the generator
    return state


# ============================================================================
# LOOP CONTROL HANDLERS
# ============================================================================

@register_handler('FOR_ITER')
def handle_for_iter(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # FOR_ITER pops an iterator from the stack and pushes the next item
    # If the iterator is exhausted, it jumps to the target instruction
    # The iterator should be on top of stack
    # Create new stack without the iterator
    new_stack = state.stack[:-1]
    iterator = state.stack[-1]
    
    # Create a new loop variable - we'll get the actual name from STORE_FAST
    # For now, use a placeholder
    loop_info = LoopInfo(
        target=ast.Name(id='_temp', ctx=ast.Store()),
        iter_ast=ensure_ast(iterator)
    )
    
    # Create new loops list with the new loop info
    new_loops = state.loops + [loop_info]
    
    return replace(state, stack=new_stack, loops=new_loops)


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
    
    # Special handling for .0 variable (the iterator)
    if var_name[0] == '.':
        # This is loading the iterator passed to the generator
        # We need to reconstruct what it represents
        if not state.frame or var_name not in state.frame.f_locals:
            raise ValueError(f"Iterator variable '{var_name}' not found in frame locals.")

        new_stack = state.stack + [ensure_ast(state.frame.f_locals[var_name])]
    else:
        # Regular variable load
        new_stack = state.stack + [ast.Name(id=var_name, ctx=ast.Load())]
    
    return replace(state, stack=new_stack)


@register_handler('STORE_FAST')
def handle_store_fast(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    var_name = instr.argval
    
    # Update the most recent loop's target variable
    if state.loops:
        # Create a new LoopInfo with updated target
        updated_loop = replace(
            state.loops[-1],
            target=ast.Name(id=var_name, ctx=ast.Store())
        )
        # Create new loops list with the updated loop
        new_loops = state.loops[:-1] + [updated_loop]
        return replace(state, loops=new_loops, current_loop_var=var_name)
    
    return replace(state, current_loop_var=var_name)


@register_handler('LOAD_CONST')
def handle_load_const(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    const_value = instr.argval
    new_stack = state.stack + [ast.Constant(value=const_value)]
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


@register_handler('STORE_NAME') 
def handle_store_name(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # STORE_NAME stores to a name in the global namespace
    # In generator expressions, this is uncommon but we'll handle it like STORE_FAST
    name = instr.argval
    return replace(state, current_loop_var=name)


# ============================================================================
# CORE GENERATOR HANDLERS (continued)
# ============================================================================

@register_handler('YIELD_VALUE')
def handle_yield_value(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # YIELD_VALUE pops a value from the stack and yields it
    # This is the expression part of the generator
    expression = ensure_ast(state.stack[-1])
    new_stack = state.stack[:-1]
    return replace(state, stack=new_stack, expression=expression)


@register_handler('RETURN_VALUE')
def handle_return_value(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # RETURN_VALUE ends the generator
    # Usually preceded by LOAD_CONST None
    new_stack = state.stack[:-1]  # Remove the None
    return replace(state, stack=new_stack)


# ============================================================================
# STACK MANAGEMENT HANDLERS
# ============================================================================

@register_handler('POP_TOP')
def handle_pop_top(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # POP_TOP removes the top item from the stack
    # In generators, often used after YIELD_VALUE
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


@register_handler('CALL')
def handle_call(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # CALL is the newer unified call instruction (Python 3.11+)
    # Similar to CALL_FUNCTION but with a different calling convention
    arg_count = instr.arg

    # Pop arguments and function
    args = [ensure_ast(arg) for arg in state.stack[-arg_count:]] if arg_count > 0 else []
    func = ensure_ast(state.stack[-arg_count - 1])
    new_stack = state.stack[:-arg_count - 1]
    
    # Create function call AST
    call_node = ast.Call(func=func, args=args, keywords=[])
    new_stack = new_stack + [call_node]
    return replace(state, stack=new_stack)


@register_handler('PRECALL')
def handle_precall(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # PRECALL is used to prepare for a function call (Python 3.11+)
    # Usually followed by CALL, so we don't need to do much here
    return state


@register_handler('MAKE_FUNCTION')
def handle_make_function(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # MAKE_FUNCTION creates a function from code object and name on stack
    # For lambda functions, we need to reconstruct the lambda expression
    code_obj: ast.Constant = state.stack[-2]  # Code object
    lambda_code: types.CodeType = code_obj.value
    
    # For lambda functions, try to reconstruct the lambda expression
    lambda_code = code_obj.value
    
    # Simple lambda reconstruction - try to extract the basic pattern
    # This is a simplified approach for common lambda patterns
    # Get the lambda's bytecode instructions
    lambda_instructions = list(dis.get_instructions(lambda_code))

    # Map binary operations
    op_map = {
        'BINARY_MULTIPLY': ast.Mult(),
        'BINARY_ADD': ast.Add(),
        'BINARY_SUBTRACT': ast.Sub(),
        'BINARY_TRUE_DIVIDE': ast.Div(),
        'BINARY_FLOOR_DIVIDE': ast.FloorDiv(),
        'BINARY_MODULO': ast.Mod(),
        'BINARY_POWER': ast.Pow(),
    }
 
    # For simple lambdas like "lambda y: y * 2"
    # Look for pattern: LOAD_FAST, LOAD_CONST, BINARY_OP, RETURN_VALUE
    if (len(lambda_instructions) == 4 and
        lambda_instructions[0].opname == 'LOAD_FAST' and
        lambda_instructions[1].opname == 'LOAD_CONST' and
        lambda_instructions[2].opname in op_map and
        lambda_instructions[3].opname == 'RETURN_VALUE'):
        
        param_name = lambda_instructions[0].argval
        const_value = lambda_instructions[1].argval
        op_name = lambda_instructions[2].opname
               
        # Create lambda AST: lambda param: param op constant
        lambda_ast = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=param_name, annotation=None)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=ast.BinOp(
                left=ast.Name(id=param_name, ctx=ast.Load()),
                op=op_map[op_name],
                right=ast.Constant(value=const_value)
            )
        )
        new_stack = state.stack[:-2] + [lambda_ast]
        return replace(state, stack=new_stack)
    else:
        raise NotImplementedError("Complex lambda reconstruction not implemented yet.")


@register_handler('GET_ITER')
def handle_get_iter(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # GET_ITER converts the top stack item to an iterator
    # For AST reconstruction, we typically don't need to change anything
    # since the iterator will be used directly in the comprehension
    return state


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
# CONTAINER BUILDING HANDLERS
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


@register_handler('BUILD_LIST')
def handle_build_list(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    list_size = instr.arg
    # Pop elements for the list
    elements = [ensure_ast(elem) for elem in state.stack[-list_size:]] if list_size > 0 else []
    new_stack = state.stack[:-list_size] if list_size > 0 else state.stack
    
    # Create list AST
    list_node = ast.List(elts=elements, ctx=ast.Load())
    new_stack = new_stack + [list_node]
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
    map_size = instr.arg
    # Pop the keys tuple and values
    keys_tuple = state.stack[-1]
    values = [ensure_ast(val) for val in state.stack[-map_size-1:-1]]
    new_stack = state.stack[:-map_size-1]
    
    # Extract keys from the constant tuple
    if isinstance(keys_tuple, ast.Constant) and isinstance(keys_tuple.value, tuple):
        keys = [ast.Constant(value=key) for key in keys_tuple.value]
    else:
        # Fallback if keys are not in expected format
        keys = [ast.Constant(value=f'key_{i}') for i in range(len(values))]
    
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
    
    op_name = instr.argval
    if op_name in op_map:
        compare_node = ast.Compare(
            left=left,
            ops=[op_map[op_name]],
            comparators=[right]
        )
        new_stack = state.stack[:-2] + [compare_node]
        return replace(state, stack=new_stack)
    else:
        raise TypeError(f"Unsupported comparison operation: {op_name}")


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
        if state.loops:
            updated_loop = replace(
                state.loops[-1],
                conditions=state.loops[-1].conditions + [combined_condition]
            )
            new_loops = state.loops[:-1] + [updated_loop]
            return replace(state, stack=new_stack, loops=new_loops, or_conditions=[])
        else:
            new_pending = state.pending_conditions + [combined_condition]
            return replace(state, stack=new_stack, pending_conditions=new_pending, or_conditions=[])
    else:
        # Regular condition - add to the most recent loop
        if state.loops:
            updated_loop = replace(
                state.loops[-1],
                conditions=state.loops[-1].conditions + [condition]
            )
            new_loops = state.loops[:-1] + [updated_loop]
            return replace(state, stack=new_stack, loops=new_loops)
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
        
        if state.loops:
            updated_loop = replace(
                state.loops[-1],
                conditions=state.loops[-1].conditions + [negated_condition]
            )
            new_loops = state.loops[:-1] + [updated_loop]
            return replace(state, stack=new_stack, loops=new_loops)
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
# SIMPLE/UTILITY OPCODE HANDLERS
# ============================================================================

@register_handler('NOP')
def handle_nop(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # NOP does nothing
    return state


@register_handler('CACHE')  
def handle_cache(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # CACHE is used for optimization caching, no effect on AST
    return state


@register_handler('RESUME')
def handle_resume(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # RESUME is used for resuming generators, no effect on AST reconstruction
    return state


@register_handler('EXTENDED_ARG')
def handle_extended_arg(state: ReconstructionState, instr: dis.Instruction) -> ReconstructionState:
    # EXTENDED_ARG extends the argument of the next instruction, no direct effect
    return state


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@functools.singledispatch
def ensure_ast(value) -> ast.AST:
    """Ensure value is an AST node"""
    raise TypeError(f"Cannot convert {type(value)} to AST node")


@ensure_ast.register
def _ensure_ast_ast(value: ast.AST) -> ast.AST:
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
def _ensure_ast_tuple_iterator(value: Iterator) -> ast.AST:
    return ensure_ast(tuple(value.__reduce__()[1][0]))


@ensure_ast.register
def _ensure_ast_list(value: list) -> ast.List:
    return ast.List(elts=[ensure_ast(v) for v in value], ctx=ast.Load())


@ensure_ast.register(type(iter([1])))
def _ensure_ast_list_iterator(value: Iterator) -> ast.AST:
    return ensure_ast(list(value.__reduce__()[1][0]))


@ensure_ast.register
def _ensure_ast_set(value: set) -> ast.Set:
    return ast.Set(elts=[ensure_ast(v) for v in value])


@ensure_ast.register(type(iter({1})))
def _ensure_ast_set_iterator(value: Iterator) -> ast.AST:
    return ensure_ast(set(value.__reduce__()[1][0]))


@ensure_ast.register
def _ensure_ast_dict(value: dict) -> ast.Dict:
    return ast.Dict(
        keys=[ensure_ast(k) for k in value.keys()],
        values=[ensure_ast(v) for v in value.values()]
    )


@ensure_ast.register(type(iter({1: 2})))
def _ensure_ast_dict_iterator(value: Iterator) -> ast.AST:
    # TODO figure out how to handle dict iterators
    raise TypeError("dict key iterator not yet supported")


@ensure_ast.register
def _ensure_ast_range(value: range) -> ast.Call:
    return ast.Call(
        func=ast.Name(id='range', ctx=ast.Load()),
        args=[ensure_ast(value.start), ensure_ast(value.stop), ensure_ast(value.step)],
        keywords=[]
    )


@ensure_ast.register(type(iter(range(1))))
def _ensure_ast_range_iterator(value: Iterator) -> ast.AST:
    return ensure_ast(value.__reduce__()[1][0])


def build_comprehension_ast(state: ReconstructionState) -> ast.AST:
    """Build the final comprehension AST from the state"""
    # Build comprehension generators
    generators = []
    
    for loop in state.loops:
        comp = ast.comprehension(
            target=loop.target,
            iter=loop.iter_ast,
            ifs=loop.conditions,
            is_async=0
        )
        generators.append(comp)
    
    # Add any pending conditions to the last loop
    if state.pending_conditions and generators:
        generators[-1].ifs.extend(state.pending_conditions)
    
    # Determine the main expression
    if state.expression:
        elt = state.expression
    elif state.stack:
        elt = ensure_ast(state.stack[-1])
    else:
        elt = ast.Name(id='item', ctx=ast.Load())
    
    # Build the appropriate comprehension type
    if state.comprehension_type == 'dict' and state.key_expression:
        return ast.DictComp(
            key=state.key_expression,
            value=elt,
            generators=generators
        )
    elif state.comprehension_type == 'list':
        return ast.ListComp(
            elt=elt,
            generators=generators
        )
    elif state.comprehension_type == 'set':
        return ast.SetComp(
            elt=elt,
            generators=generators
        )
    else:  # generator
        return ast.GeneratorExp(
            elt=elt,
            generators=generators
        )


# ============================================================================
# MAIN RECONSTRUCTION FUNCTION
# ============================================================================

def reconstruct(genexpr: GeneratorType) -> ast.GeneratorExp:
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
    assert inspect.getgeneratorstate(genexpr) == 'GEN_CREATED', "Generator must be in created state"
    
    # Initialize reconstruction state
    state = ReconstructionState(
        code_obj=genexpr.gi_code,
        frame=genexpr.gi_frame
    )
    
    # Process each instruction
    for instr in dis.get_instructions(genexpr.gi_code):
        # Call the handler
        state = OP_HANDLERS[instr.opname](state, instr)
    
    # Build and return the final AST
    return build_comprehension_ast(state)
