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
import builtins
import collections
import collections.abc
import copy
import dis
import enum
import functools
import inspect
import itertools
import sys
import types
import typing
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field, replace

CompExp = ast.GeneratorExp | ast.ListComp | ast.SetComp | ast.DictComp


class Placeholder(ast.Name):
    """Placeholder for AST nodes that are not yet resolved."""

    def __init__(
        self,
        id: typing.Literal[".PLACEHOLDER"] = ".PLACEHOLDER",
        ctx: ast.Load = ast.Load(),
    ):
        super().__init__(id=id, ctx=ctx)


class DummyIterName(ast.Name):
    """Dummy name for the iterator variable in generator expressions."""

    def __init__(self, id: typing.Literal[".0"] = ".0", ctx: ast.Load = ast.Load()):
        super().__init__(id=id, ctx=ctx)


class Skipped(ast.Name):
    """Placeholder for skipped branches in if-expressions.

    ``id`` is defaulted so that ``copy.deepcopy`` can reconstruct the node: on
    Python 3.12 ``ast.AST.__reduce__`` supplies no positional arguments (3.13+
    supplies one per field), so the constructor must be callable with none.
    """

    def __init__(self, id: str = "", ctx: ast.Load = ast.Load()):
        super().__init__(id=id, ctx=ctx)


class CommonConstant(ast.Name):
    """A constant pushed by 3.14's LOAD_COMMON_CONSTANT.

    3.14 can inline `any()`/`all()` over a generator, guarding the fast path
    with `loaded_name is <the builtin>`. Marking the builtin lets that guard be
    recognised so the generic call path is followed instead, which still spells
    out the original call.
    """

    def __init__(self, id: str = "", ctx: ast.Load = ast.Load()):
        super().__init__(id=id, ctx=ctx)


class TargetHole(ast.Name):
    """Placeholder for a comprehension loop target that is not yet named.

    ``FOR_ITER`` knows that a loop target exists but not what it is called; the
    name only arrives with the ``STORE_FAST``/``UNPACK_SEQUENCE`` instructions
    that follow. Each hole carries a unique ``id`` so that it can be located
    again inside a comprehension after the surrounding state has been copied,
    which matters for unpacking targets where several holes are live at once.
    """

    _counter: typing.ClassVar[Iterator[int]] = itertools.count()

    def __init__(self, id: str = "", ctx: ast.Store = ast.Store()):
        super().__init__(id=id or f".TARGET_{next(TargetHole._counter)}", ctx=ctx)


class ReplaceTargetHole(ast.NodeTransformer):
    """Replace the uniquely-identified :class:`TargetHole` ``id`` with ``replacement``."""

    id: str
    replacement: ast.expr

    def __init__(self, id: str, replacement: ast.expr):
        self.id = id
        self.replacement = replacement
        super().__init__()

    def visit_TargetHole(self, node: TargetHole) -> ast.expr:
        return self.replacement if node.id == self.id else node


def _bind_target_hole(
    stack: list[ast.expr], hole: ast.expr, replacement: ast.expr
) -> list[ast.expr]:
    """Fill the loop-target hole ``hole`` of the innermost matching comprehension.

    Returns a new stack; the hole itself is left in place for the caller to pop.
    """
    assert isinstance(hole, TargetHole), f"Expected a loop target hole, got {hole}"
    for pos, item in zip(reversed(range(len(stack))), reversed(stack)):
        if not isinstance(item, CompExp) or not item.generators:
            continue
        if not any(
            isinstance(n, TargetHole) and n.id == hole.id
            for n in ast.walk(item.generators[-1].target)
        ):
            continue
        new_comp = ReplaceTargetHole(hole.id, replacement).visit(copy.deepcopy(item))
        return stack[:pos] + [new_comp] + stack[pos + 1 :]

    raise TypeError(f"No comprehension found with loop target hole {hole.id}")


class Null(ast.Constant):
    """Placeholder for NULL values generated in bytecode."""

    def __init__(self, value: None = None):
        super().__init__(value=value)


class ConvertedValue(ast.expr):
    """Wrapper for values that have been converted with CONVERT_VALUE."""

    value: ast.expr
    conversion: int
    ast_conversion: int

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
    value: ast.expr
    _done: bool

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


class ReplaceSkipped(ast.NodeTransformer):
    id: str
    replacement: ast.expr

    def __init__(self, id: str, replacement: ast.expr):
        self.id = id
        self.replacement = copy.deepcopy(replacement)
        super().__init__()

    def visit_IfExp(self, node: ast.IfExp):
        if isinstance(node.body, Skipped) and node.body.id == self.id:
            return ast.IfExp(test=node.test, body=self.replacement, orelse=node.orelse)
        elif isinstance(node.orelse, Skipped) and node.orelse.id == self.id:
            return ast.IfExp(test=node.test, body=node.body, orelse=self.replacement)
        else:
            return self.generic_visit(node)


class BranchState(typing.NamedTuple):
    testval: bool
    value: ast.expr


class BranchIdentifier(ast.NodeVisitor):
    branching: collections.abc.MutableMapping[str, BranchState]
    filter_lengths: list[int]

    def __init__(self):
        self.branching = {}
        self.filter_lengths = []
        super().__init__()

    def visit_IfExp(self, node: ast.IfExp):
        if isinstance(node.body, Skipped):
            self.branching[node.body.id] = BranchState(
                testval=False, value=copy.deepcopy(node.orelse)
            )
        elif isinstance(node.orelse, Skipped):
            self.branching[node.orelse.id] = BranchState(
                testval=True, value=copy.deepcopy(node.body)
            )
        return self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension):
        self.filter_lengths.append(len(node.ifs))
        return self.generic_visit(node)


@functools.cache
def _instructions(
    code: types.CodeType,
) -> collections.abc.Mapping[int, dis.Instruction]:
    """Decode a code object once; every state derived from it shares the result."""
    return collections.OrderedDict(
        (instr.offset, instr) for instr in dis.get_instructions(code)
    )


@functools.cache
def _next_instructions(
    code: types.CodeType,
) -> collections.abc.Mapping[int, dis.Instruction]:
    """Map each instruction offset to the instruction that follows it."""
    ordered = list(_instructions(code).values())
    return {before.offset: after for before, after in zip(ordered[:-1], ordered[1:])}


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

    # How many times each FOR_ITER has been entered on this path.
    loops: dict[int, int] = field(default_factory=dict)
    finished: bool = field(default=False)

    # Which edge each already-resolved conditional jump took on this path.
    branches: "dict[int, BranchEdge]" = field(default_factory=dict)

    # Locals bound to a known expression rather than to a loop target. 3.14
    # unrolls a single-iteration loop over a literal, storing its targets
    # directly, so those names have to be substituted back at their uses.
    bindings: dict[str, ast.expr] = field(default_factory=dict)

    # Set by KW_NAMES (Python 3.12 only) and consumed by the following CALL.
    # KW_NAMES has no stack effect, so the names cannot live on `stack`.
    kw_names: tuple[str, ...] | None = field(default=None)

    @property
    def instructions(self) -> collections.abc.Mapping[int, dis.Instruction]:
        """The bytecode instructions of the current code object, by offset."""
        return _instructions(self.code)

    @property
    def next_instructions(self) -> collections.abc.Mapping[int, dis.Instruction]:
        return _next_instructions(self.code)


# Python version enum for version-specific handling
class PythonVersion(enum.IntEnum):
    PY_312 = 12
    PY_313 = 13
    PY_314 = 14


def current_version() -> PythonVersion:
    """The bytecode dialect of the running interpreter.

    Raises on a Python this module has not been taught, rather than guessing
    that the previous release's opcodes still mean what they used to.
    """
    try:
        return PythonVersion(sys.version_info.minor)
    except ValueError as e:
        supported = ", ".join(f"3.{v.value}" for v in PythonVersion)
        raise NotImplementedError(
            f"effectful.internals.disassembly supports {supported}, "
            f"not 3.{sys.version_info.minor}"
        ) from e


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
    """Register a handler for one opcode in one Python bytecode dialect.

    Every dialect a handler applies to is named explicitly. Opcodes are not
    assumed to carry forward: a release can keep an opcode's name while changing
    what it does, so applicability to a new Python is a decision to make per
    opcode rather than a default.
    """
    if handler is None:
        return functools.partial(register_handler, opname, version=version)

    # Skip registration if version doesn't match current version
    if version != current_version():
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
            if state.loops.get(instr.offset, 0) > 0:
                new_state = replace(
                    new_state, instruction=state.instructions[instr.argval]
                )
                jump = True
            else:
                # Copy rather than mutate: continuations forked from this state
                # share the mapping and must not see each other's loop counts.
                new_state = replace(
                    new_state,
                    instruction=state.next_instructions[instr.offset],
                    loops={
                        **state.loops,
                        instr.offset: state.loops.get(instr.offset, 0) + 1,
                    },
                )
                jump = False
        elif instr.opname in BRANCH_OPS:
            if new_state.branches.get(instr.offset) == BranchEdge.FALL_THROUGH:
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


LOOP_OPS: set[typing.Literal["FOR_ITER"]] = {"FOR_ITER"}

BRANCH_OPS: set[
    typing.Literal[
        "POP_JUMP_IF_TRUE",
        "POP_JUMP_IF_FALSE",
        "POP_JUMP_IF_NOT_NONE",
        "POP_JUMP_IF_NONE",
    ]
] = {
    "POP_JUMP_IF_TRUE",
    "POP_JUMP_IF_FALSE",
    "POP_JUMP_IF_NOT_NONE",
    "POP_JUMP_IF_NONE",
}

RETURN_OPS: set[typing.Literal["RETURN_VALUE", "RETURN_CONST"]] = {
    "RETURN_VALUE",
    "RETURN_CONST",
}

JUMP_OPS = {dis.opname[d] for d in dis.hasjrel} - LOOP_OPS - BRANCH_OPS - RETURN_OPS


# Instructions that emit an element of the comprehension being built. Reaching
# one of these means the current iteration was *not* filtered out.
PRODUCE_OPS = {"YIELD_VALUE", "LIST_APPEND", "SET_ADD", "MAP_ADD"}


def _successor_offsets(state: ReconstructionState, instr: dis.Instruction) -> list[int]:
    """Offsets control can transfer to from ``instr``, ignoring exception edges."""
    following = state.next_instructions.get(instr.offset)
    if instr.opname in BRANCH_OPS | LOOP_OPS:
        return [instr.argval] + ([following.offset] if following else [])
    elif instr.opname in JUMP_OPS:
        return [instr.argval]
    elif instr.opname in RETURN_OPS:
        return []
    else:
        return [following.offset] if following else []


def _reachable_outcomes(state: ReconstructionState, start: int) -> tuple[bool, bool]:
    """From ``start``, can the iteration be skipped, and can an element be produced?

    Returns ``(can_skip, can_produce)``. "Skip" means reaching the loop
    back-edge without emitting an element, i.e. being filtered out.
    """
    seen: set[int] = set()
    pending = [start]
    can_skip = can_produce = False

    while pending:
        offset = pending.pop()
        if offset in seen or offset not in state.instructions:
            continue
        seen.add(offset)

        instr = state.instructions[offset]
        if instr.opname in PRODUCE_OPS:
            can_produce = True
            continue
        if (
            instr.opname == "JUMP_BACKWARD"
            and state.instructions[instr.argval].opname in LOOP_OPS
        ):
            can_skip = True
            continue

        pending.extend(_successor_offsets(state, instr))

    return can_skip, can_produce


class BranchEdge(enum.IntEnum):
    """Which way a conditional jump was resolved on the path being explored."""

    TAKE_JUMP = 1
    FALL_THROUGH = 2


class BranchKind(enum.Enum):
    """What role a conditional jump plays in a comprehension.

    TERNARY
        A conditional expression: the arms reconverge having each pushed a
        value, and both are spliced back together into an ``ast.IfExp``.
    FILTER
        Part of a filter's condition. The condition is consumed rather than
        producing a value, so each surviving path records the conjunction of
        tests that got it to the element, and the filter as a whole is the
        disjunction of those conjunctions.
    """

    TERNARY = enum.auto()
    FILTER = enum.auto()


@functools.cache
def _stack_depths(code: types.CodeType) -> collections.abc.Mapping[int, int]:
    """VM stack depth on entry to each reachable instruction.

    Depths are relative to the start of the code object, which is all that is
    needed to tell a value-producing branch from a control-flow one.
    """
    instructions = collections.OrderedDict(
        (i.offset, i) for i in dis.get_instructions(code)
    )
    ordered = list(instructions.values())
    following = {a.offset: b.offset for a, b in zip(ordered[:-1], ordered[1:])}

    depths: dict[int, int] = {ordered[0].offset: 0}
    pending = collections.deque([ordered[0].offset])
    while pending:
        offset = pending.popleft()
        instr, depth = instructions[offset], depths[offset]
        if instr.opname in RETURN_OPS:
            continue

        edges: list[tuple[int, bool | None]] = []
        if instr.opname in BRANCH_OPS | LOOP_OPS:
            edges = [(instr.argval, True)]
            if offset in following:
                edges.append((following[offset], False))
        elif instr.opname in JUMP_OPS:
            edges = [(instr.argval, True)]
        elif offset in following:
            edges = [(following[offset], None)]

        for target, jump in edges:
            if target in instructions and target not in depths:
                depths[target] = depth + dis.stack_effect(
                    instr.opcode, instr.arg, jump=jump
                )
                pending.append(target)

    return depths


def _forward_reachable(state: ReconstructionState, start: int) -> set[int]:
    """Offsets reachable from ``start`` without producing or looping back."""
    seen: set[int] = set()
    pending = [start]
    while pending:
        offset = pending.pop()
        if offset in seen or offset not in state.instructions:
            continue
        seen.add(offset)

        instr = state.instructions[offset]
        if instr.opname in PRODUCE_OPS:
            continue
        if (
            instr.opname == "JUMP_BACKWARD"
            and state.instructions[instr.argval].opname in LOOP_OPS
        ):
            continue

        pending.extend(_successor_offsets(state, instr))

    return seen


def _is_conditional_expression(
    state: ReconstructionState, instr: dis.Instruction
) -> bool:
    """Do the two edges of ``instr`` reconverge one stack slot deeper?

    That is the signature of a conditional expression: each arm leaves a value
    behind and control rejoins to consume it. A filter's condition is consumed
    by the jump itself, so wherever its edges meet -- if they meet at all -- the
    stack is no deeper than it was.
    """
    following = state.next_instructions.get(instr.offset)
    if following is None:
        return False

    common = _forward_reachable(state, instr.argval) & _forward_reachable(
        state, following.offset
    )
    if not common:
        return False  # the edges never rejoin, so nothing was left on the stack

    depths = _stack_depths(state.code)
    join = min(common)  # arms are laid out contiguously, so the join comes first
    if join not in depths or following.offset not in depths:
        return False
    return depths[join] == depths[following.offset] + 1


def _classify_branch(
    state: ReconstructionState, instr: dis.Instruction
) -> tuple[BranchKind, list[BranchEdge]]:
    """Classify a conditional jump and list the edges worth exploring."""
    both = [BranchEdge.TAKE_JUMP, BranchEdge.FALL_THROUGH]
    following = state.next_instructions.get(instr.offset)
    if following is None:
        return BranchKind.TERNARY, both

    jump_skip, _ = _reachable_outcomes(state, instr.argval)
    fall_skip, _ = _reachable_outcomes(state, following.offset)

    # Neither edge can drop the current iteration -- because there is no loop at
    # all (a lambda body) or because the element is produced regardless (a
    # conditional in the element expression). Either way nothing is filtered.
    if not jump_skip and not fall_skip:
        return BranchKind.TERNARY, both

    # Otherwise the branch could be a filter, or a conditional expression that
    # merely happens to sit inside one. Only the latter leaves a value behind.
    if _is_conditional_expression(state, instr):
        return BranchKind.TERNARY, both

    # An edge that cannot reach an element contributes nothing to the filter, so
    # there is no point walking it. Pruning those edges is also what keeps the
    # executor out of the operand-cleanup block on a chained comparison's
    # failing edge.
    live = [
        edge
        for edge, start in (
            (BranchEdge.TAKE_JUMP, instr.argval),
            (BranchEdge.FALL_THROUGH, following.offset),
        )
        if _reachable_outcomes(state, start)[1]
    ]
    return BranchKind.FILTER, live or [BranchEdge.TAKE_JUMP]


def _negate(condition: ast.expr) -> ast.expr:
    """Logical negation, cancelling a `not` rather than stacking another one."""
    if isinstance(condition, ast.UnaryOp) and isinstance(condition.op, ast.Not):
        return condition.operand
    return ast.UnaryOp(op=ast.Not(), operand=condition)


def _conjoin(conditions: list[ast.expr]) -> ast.expr | None:
    """Combine the entries of a ``comprehension.ifs`` list into one expression."""
    if not conditions:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return ast.BoolOp(op=ast.And(), values=list(conditions))


def _disjoin(left: ast.expr, right: ast.expr) -> ast.expr:
    """Combine two conditions with ``or``, flattening nested disjunctions.

    Two rewrites are applied while combining. Duplicate disjuncts are dropped,
    because paths through independent filters repeat them. And ``X or (not X and
    Y)`` becomes ``X or Y``: enumerating paths records the negation of every
    test a path declined, so a later disjunct restates the negation of an
    earlier one. Dropping it is not merely tidier -- `or` short-circuits, so the
    earlier disjunct has already been evaluated, and leaving the negation in
    would evaluate it a second time, which is visibly wrong when the condition
    contains an assignment expression.

    Conditions are keyed by ``ast.dump`` exactly once each: these lists get long
    and the expressions large, so re-dumping per comparison dominates.
    """
    values: list[ast.expr] = []
    for side in (left, right):
        if isinstance(side, ast.BoolOp) and isinstance(side.op, ast.Or):
            values.extend(side.values)
        else:
            values.append(side)

    unique: list[ast.expr] = []
    seen: set[str] = set()
    for value in values:
        key = ast.dump(value)
        if key in seen:
            continue
        seen.add(key)

        # Absorb the negations of the disjuncts already accepted.
        if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.And):
            kept = [
                conjunct
                for conjunct in value.values
                if not (
                    isinstance(conjunct, ast.UnaryOp)
                    and isinstance(conjunct.op, ast.Not)
                    and ast.dump(conjunct.operand) in seen
                )
            ]
            if kept and len(kept) < len(value.values):
                conjoined = _conjoin(kept)
                assert conjoined is not None
                value = conjoined

        unique.append(value)

    return unique[0] if len(unique) == 1 else ast.BoolOp(op=ast.Or(), values=unique)


def _merge_filters_into(
    node: typing.Any, other: typing.Any, mutate: bool = True
) -> bool:
    """Walk two results in parallel, OR-ing the filters where they disagree.

    Everything outside a ``comprehension.ifs`` has to match exactly; the ifs are
    where the paths are allowed to differ, and are combined rather than
    compared. Filters are not recursed into, so a nested comprehension inside a
    filter is treated as part of that filter's condition.

    Returns False if the two results differ somewhere they may not. With
    ``mutate=False`` nothing is written, which allows compatibility to be tested
    before paying for a deep copy -- most candidate pairs do not merge, and the
    copy dominates otherwise.
    """
    if type(node) is not type(other):
        return False

    if isinstance(node, ast.comprehension):
        if ast.dump(node.target) != ast.dump(other.target):
            return False
        if not _merge_filters_into(node.iter, other.iter, mutate):
            return False
        if not mutate:
            return True

        guard, other_guard = _conjoin(node.ifs), _conjoin(other.ifs)
        if guard is None or other_guard is None:
            # One path reached the element unconditionally, so the filter as a
            # whole is unconditional at this generator.
            node.ifs = []
        elif ast.dump(guard) != ast.dump(other_guard):
            node.ifs = [_disjoin(guard, other_guard)]
        return True

    if not isinstance(node, ast.AST):
        return bool(node == other)

    for name in node._fields:
        mine, theirs = getattr(node, name, None), getattr(other, name, None)
        if isinstance(mine, list) or isinstance(theirs, list):
            if not isinstance(mine, list) or not isinstance(theirs, list):
                return False
            if len(mine) != len(theirs):
                return False
            if not all(_merge_filters_into(a, b, mutate) for a, b in zip(mine, theirs)):
                return False
        elif isinstance(mine, ast.AST) or isinstance(theirs, ast.AST):
            if not isinstance(mine, ast.AST) or not isinstance(theirs, ast.AST):
                return False
            if not _merge_filters_into(mine, theirs, mutate):
                return False
        elif mine != theirs:
            return False

    return True


def _merge_filters(left: ast.expr, right: ast.expr) -> ast.expr | None:
    """Combine two paths that differ only in which filter conditions they met.

    Each path through a filter records the conjunction that got it to the
    element; the filter as a whole is the disjunction over all such paths.
    Returns ``None`` when the results differ by more than their filters.
    """
    # A marker anywhere means some conditional expression is still unresolved,
    # and unresolved arms must be spliced before anything can be OR-ed.
    if any(isinstance(n, Skipped) for n in ast.walk(left)):
        return None
    if any(isinstance(n, Skipped) for n in ast.walk(right)):
        return None

    if not _merge_filters_into(left, right, mutate=False):
        return None

    merged = copy.deepcopy(left)
    return merged if _merge_filters_into(merged, right) else None


def _skipped_offset(key: str) -> int:
    """Sort key for `.SKIPPED_<offset>` markers, so merging is deterministic."""
    return int(key.rsplit("_", 1)[-1])


def _merge_at_ifexp(left: ast.expr, right: ast.expr) -> ast.expr:
    """
    Merge two expression ASTs obtained from two branches of symbolic execution.
    """
    if isinstance(left, ast.Constant) and left.value is None:
        return copy.deepcopy(right)
    elif isinstance(right, ast.Constant) and right.value is None:
        return copy.deepcopy(left)

    assert type(left) == type(right)

    lb, rb = BranchIdentifier(), BranchIdentifier()
    lb.visit(left)
    rb.visit(right)

    # A conditional expression: each path filled in one arm and left a marker in
    # the other, so splice the two together. Sorted for determinism -- set
    # iteration order over the marker names varies with PYTHONHASHSEED.
    common_keys = set(lb.branching) & set(rb.branching)
    differing = [
        key
        for key in sorted(common_keys, key=_skipped_offset)
        if lb.branching[key].testval != rb.branching[key].testval
    ]

    # Only copy once it is known there is something to splice; this runs for
    # every candidate pair of paths, most of which have nothing in common.
    merged: ast.expr = copy.deepcopy(left) if differing else left
    for key in differing:
        visited = ReplaceSkipped(key, rb.branching[key].value).visit(merged)
        assert isinstance(visited, ast.expr)
        merged = visited
    spliced = bool(differing)

    # The paths may *also* have satisfied different filter conditions on the way
    # to the element, so combine those too rather than picking one arbitrarily.
    combined = _merge_filters(merged, right)
    if combined is not None:
        return combined
    if spliced:
        return merged

    if ast.dump(left) == ast.dump(right):
        return copy.deepcopy(left)

    raise ValueError("No differing branches found to merge")


def _specialization_guard_edge(
    state: ReconstructionState, instr: dis.Instruction
) -> BranchEdge | None:
    """The edge past a 3.14 inlined-builtin guard, or None if this isn't one.

    3.14 may inline `any()`/`all()` over a generator, guarding the inlined code
    with `loaded_name is <the builtin>` and keeping an ordinary call on the
    other edge. Only that other edge still contains the call to reconstruct, so
    the guard is treated as though the identity test failed.
    """
    if not state.stack:
        return None
    condition = state.stack[-1]
    if not isinstance(condition, ast.Compare):
        return None
    if not any(isinstance(c, CommonConstant) for c in condition.comparators):
        return None

    # Follow the edge taken when the identity test is false.
    if instr.opname == "POP_JUMP_IF_FALSE":
        return BranchEdge.TAKE_JUMP
    elif instr.opname == "POP_JUMP_IF_TRUE":
        return BranchEdge.FALL_THROUGH
    else:
        return None


def _merge_all(results: list[ast.expr]) -> ast.expr:
    """Combine every path's result into one expression.

    Merging is not associative: a path that still carries an unfilled
    conditional-expression arm can only combine with the path that took the
    other arm, which need not be its neighbour. So rather than folding in
    order, repeatedly merge whichever pair actually combines.
    """
    pending = list(results)
    while len(pending) > 1:
        for i, j in itertools.combinations(range(len(pending)), 2):
            try:
                merged = _merge_at_ifexp(pending[i], pending[j])
            except (ValueError, AssertionError):
                continue
            pending = [merged] + [p for k, p in enumerate(pending) if k not in (i, j)]
            break
        else:
            raise ValueError("Could not merge the paths of symbolic execution")

    return pending[0]


def _decide_branch(
    state: ReconstructionState, instr: dis.Instruction, edge: BranchEdge
) -> ReconstructionState:
    """Record which edge of ``instr`` the path being explored takes."""
    return replace(state, branches={**state.branches, instr.offset: edge})


def _symbolic_exec(code: types.CodeType) -> ast.expr:
    """Execute bytecode symbolically, following control flow."""
    continuations: list[ReconstructionState] = [
        ReconstructionState(
            code=code,
            instruction=next(iter(dis.get_instructions(code))),
            stack=[Placeholder(), Placeholder()]
            if current_version() == PythonVersion.PY_312
            and code.co_flags & inspect.CO_GENERATOR
            else [Placeholder()],
        )
    ]

    results: list[ast.expr] = []

    while continuations:
        state = continuations.pop()
        while not state.finished:
            instr = state.instruction
            if instr.opname in BRANCH_OPS and instr.offset not in state.branches:
                forced = _specialization_guard_edge(state, instr)
                if forced is not None:
                    state = _decide_branch(state, instr, forced)
                else:
                    _, live = _classify_branch(state, instr)
                    # Explore the first live edge now; queue the rest for later.
                    continuations.extend(
                        _decide_branch(state, instr, edge) for edge in live[1:]
                    )
                    state = _decide_branch(state, instr, live[0])

            state = OP_HANDLERS[state.instruction.opname](state, state.instruction)
        results.append(state.result)

    assert results, "No results from symbolic execution"
    result = _merge_all(results)
    assert not any(isinstance(n, Skipped) for n in ast.walk(result)), (
        "Every conditional expression arm must have been filled in"
    )
    return result


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
@register_handler("RETURN_GENERATOR", version=PythonVersion.PY_314)
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
@register_handler("YIELD_VALUE", version=PythonVersion.PY_314)
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
@register_handler("BUILD_LIST", version=PythonVersion.PY_314)
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
@register_handler("LIST_APPEND", version=PythonVersion.PY_314)
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
@register_handler("BUILD_SET", version=PythonVersion.PY_314)
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
@register_handler("SET_ADD", version=PythonVersion.PY_314)
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
@register_handler("BUILD_MAP", version=PythonVersion.PY_314)
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
@register_handler("MAP_ADD", version=PythonVersion.PY_314)
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


def _unyielded_comprehension(state: ReconstructionState) -> CompExp | None:
    """The comprehension of a body the compiler proved unreachable, if any.

    An always-false filter lets the compiler drop the whole body: the loop is
    still walked, but nothing is ever yielded or appended, so no element
    expression survives. The partly built comprehension still carries its
    generators, so it can be rebuilt with a filter that is never satisfied --
    which iterates exactly as the original did and produces nothing.
    """
    for item in reversed(state.stack):
        if not isinstance(item, CompExp) or not item.generators:
            continue

        element = item.value if isinstance(item, ast.DictComp) else item.elt
        if not isinstance(element, Placeholder):
            continue

        unreachable = copy.deepcopy(item)
        never = ast.Constant(value=None)
        if isinstance(unreachable, ast.DictComp):
            unreachable.key, unreachable.value = never, copy.deepcopy(never)
        else:
            unreachable.elt = never
        unreachable.generators[-1].ifs = [ast.Constant(value=False)]
        return unreachable

    return None


@register_handler("RETURN_VALUE", version=PythonVersion.PY_314)
def handle_return_value_314(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # Two things changed in 3.14. RETURN_CONST is gone, so a generator's
    # trailing `return None` now arrives as LOAD_CONST + RETURN_VALUE; and
    # RETURN_VALUE's stack effect is 0 rather than -1, the returned value being
    # discarded along with the frame. The value therefore stays on the stack.
    if not isinstance(state.result, Placeholder):
        assert (
            isinstance(state.stack[-1], ast.Constant) and state.stack[-1].value is None
        ), "A generator may only fall off the end returning None"
        return state

    unreachable = _unyielded_comprehension(state)
    if unreachable is not None:
        return replace(state, result=unreachable)

    assert len(state.stack) == 2
    new_result = ReplacePlaceholder(ensure_ast(state.stack[-1])).visit(state.stack[-2])
    return replace(state, result=new_result)


@register_handler("RETURN_CONST", version=PythonVersion.PY_312)
@register_handler("RETURN_CONST", version=PythonVersion.PY_313)
def handle_return_const(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # RETURN_CONST returns a constant value (replaces some LOAD_CONST + RETURN_VALUE patterns)
    # Similar to RETURN_VALUE but with a constant
    if isinstance(state.result, Placeholder):
        unreachable = _unyielded_comprehension(state)
        if unreachable is not None:
            return replace(state, result=unreachable)
        return replace(state, result=ensure_ast(instr.argval))
    else:
        assert instr.argval is None
        return state


@register_handler("FOR_ITER", version=PythonVersion.PY_312)
@register_handler("FOR_ITER", version=PythonVersion.PY_313)
@register_handler("FOR_ITER", version=PythonVersion.PY_314)
def handle_for_iter(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # FOR_ITER pops an iterator from the stack and pushes the next item
    # If the iterator is exhausted, it jumps to the target instruction
    assert len(state.stack) > 0, "FOR_ITER must have an iterator on the stack"

    if state.loops.get(instr.offset, 0) > 0:
        return replace(state, stack=state.stack + [Null()])

    # The iterator should be on top of stack
    iterator: ast.expr = state.stack[-1]

    for pos, item in zip(reversed(range(len(state.stack))), reversed(state.stack)):
        if not isinstance(item, CompExp):
            continue

        element = item.value if isinstance(item, ast.DictComp) else item.elt
        new_result = copy.deepcopy(item)

        if isinstance(element, Placeholder):
            loop_iter = ensure_ast(iterator)
        elif isinstance(element, ast.IfExp) and any(
            isinstance(x, Placeholder) for x in ast.walk(element)
        ):
            # A conditional expression was being built up in the element slot,
            # but it turned out to be this loop's iterable, as in
            # `for y in (a if c else b)`. Move it back out and plug the value
            # this path produced into the arm still awaiting one.
            if isinstance(new_result, ast.DictComp):
                new_result.key, new_result.value = Placeholder(), Placeholder()
            else:
                new_result.elt = Placeholder()

            plugged = ReplacePlaceholder(ensure_ast(iterator)).visit(
                copy.deepcopy(element)
            )
            assert isinstance(plugged, ast.expr)
            loop_iter = plugged
        else:
            continue

        # The loop target is not named until the STORE_* that follows.
        loop_info = ast.comprehension(
            target=TargetHole(), iter=loop_iter, ifs=[], is_async=0
        )
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
@register_handler("GET_ITER", version=PythonVersion.PY_314)
def handle_get_iter(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # GET_ITER converts the top stack item to an iterator
    # For AST reconstruction, we typically don't need to change anything
    # since the iterator will be used directly in the comprehension
    return state


@register_handler("END_FOR", version=PythonVersion.PY_312)
def handle_end_for_312(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # END_FOR marks the end of a for loop, followed by POP_TOP (in 3.12)
    new_stack = state.stack[:-2]
    return replace(state, stack=new_stack)


@register_handler("END_FOR", version=PythonVersion.PY_313)
@register_handler("END_FOR", version=PythonVersion.PY_314)
def handle_end_for(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # END_FOR marks the end of a for loop - no action needed for AST reconstruction
    new_stack = state.stack[:-1]
    return replace(state, stack=new_stack)


@register_handler("RERAISE", version=PythonVersion.PY_312)
@register_handler("RERAISE", version=PythonVersion.PY_313)
@register_handler("RERAISE", version=PythonVersion.PY_314)
def handle_reraise(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # RERAISE re-raises an exception - generally ignore for AST reconstruction
    return state


# ============================================================================
# VARIABLE OPERATIONS HANDLERS
# ============================================================================


def _literal_elements(value: ast.expr, count: int | None = None) -> list[ast.expr]:
    """The elements of a literal sequence, for destructuring a known value."""
    assert isinstance(value, ast.Tuple | ast.List), (
        f"Cannot unpack {type(value).__name__}; expected a literal sequence"
    )
    assert count is None or len(value.elts) == count, (
        f"Expected {count} values to unpack, got {len(value.elts)}"
    )
    return [ensure_ast(element) for element in value.elts]


def _bind_local(
    state: ReconstructionState, var_name: str, value: ast.expr
) -> ReconstructionState:
    """Record that a local now stands for ``value``, popping it off the stack.

    Reached when a store is not filling in a loop target: 3.14 unrolls a
    single-iteration loop over a literal, assigning its targets outright. The
    loop is gone from the bytecode, so the names it bound are not in scope in
    the reconstruction and their uses are substituted instead.
    """
    bindings = {**state.bindings, var_name: ensure_ast(value)}
    return replace(state, stack=state.stack[:-1], bindings=bindings)


def _read_local(state: ReconstructionState, var_name: str) -> ast.expr:
    """The expression a local name stands for at this point on this path."""
    if var_name == DummyIterName().id:
        return DummyIterName()
    elif var_name in state.bindings:
        # Bound to a known expression rather than by a loop, so the name itself
        # is not in scope in the reconstruction; use what it was bound to.
        return copy.deepcopy(state.bindings[var_name])
    else:
        return ast.Name(id=var_name, ctx=ast.Load())


@register_handler("LOAD_FAST", version=PythonVersion.PY_312)
@register_handler("LOAD_FAST", version=PythonVersion.PY_313)
@register_handler("LOAD_FAST", version=PythonVersion.PY_314)
@register_handler("LOAD_FAST_CHECK", version=PythonVersion.PY_312)
@register_handler("LOAD_FAST_CHECK", version=PythonVersion.PY_313)
@register_handler("LOAD_FAST_CHECK", version=PythonVersion.PY_314)
def handle_load_fast(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LOAD_FAST_CHECK differs only in raising when the local is unbound, which
    # says nothing about the expression being reconstructed.
    return replace(state, stack=state.stack + [_read_local(state, instr.argval)])


@register_handler("LOAD_DEREF", version=PythonVersion.PY_312)
@register_handler("LOAD_DEREF", version=PythonVersion.PY_313)
@register_handler("LOAD_DEREF", version=PythonVersion.PY_314)
def handle_load_deref(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LOAD_DEREF loads a value from a closure variable
    var_name = instr.argval
    new_stack = state.stack + [ast.Name(id=var_name, ctx=ast.Load())]
    return replace(state, stack=new_stack)


@register_handler("LOAD_CLOSURE", version=PythonVersion.PY_312)
@register_handler("LOAD_CLOSURE", version=PythonVersion.PY_313)
@register_handler("LOAD_CLOSURE", version=PythonVersion.PY_314)
def handle_load_closure(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LOAD_CLOSURE loads a closure variable
    var_name = instr.argval
    new_stack = state.stack + [ast.Name(id=var_name, ctx=ast.Load())]
    return replace(state, stack=new_stack)


@register_handler("LOAD_CONST", version=PythonVersion.PY_312)
@register_handler("LOAD_CONST", version=PythonVersion.PY_313)
@register_handler("LOAD_CONST", version=PythonVersion.PY_314)
def handle_load_const(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    const_value = instr.argval
    new_stack = state.stack + [ensure_ast(const_value)]
    return replace(state, stack=new_stack)


@register_handler("LOAD_GLOBAL", version=PythonVersion.PY_312)
@register_handler("LOAD_GLOBAL", version=PythonVersion.PY_313)
@register_handler("LOAD_GLOBAL", version=PythonVersion.PY_314)
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
@register_handler("LOAD_NAME", version=PythonVersion.PY_314)
def handle_load_name(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LOAD_NAME is similar to LOAD_GLOBAL but for names in the global namespace
    name = instr.argval
    new_stack = state.stack + [ast.Name(id=name, ctx=ast.Load())]
    return replace(state, stack=new_stack)


def _is_assignment_expression(state: ReconstructionState) -> bool:
    """Is the top of the stack a COPY of the value beneath it?

    That duplication is how an assignment expression keeps its value after
    binding it: `COPY 1` then a `STORE_*`. `handle_copy` pushes the very same
    node, so identity is what distinguishes it from two equal-looking values.
    """
    return len(state.stack) >= 2 and state.stack[-1] is state.stack[-2]


def _handle_assignment_expression(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    """Rebuild `(name := value)` from the COPY/STORE pair that implements it."""
    target = ast.Name(id=instr.argval, ctx=ast.Store())
    named = ast.NamedExpr(target=target, value=ensure_ast(state.stack[-1]))
    return replace(state, stack=state.stack[:-2] + [named])


@register_handler("STORE_GLOBAL", version=PythonVersion.PY_312)
@register_handler("STORE_GLOBAL", version=PythonVersion.PY_313)
@register_handler("STORE_GLOBAL", version=PythonVersion.PY_314)
def handle_store_global(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # A comprehension has no globals of its own, so the only way it stores one
    # is an assignment expression, which binds in the enclosing scope.
    assert _is_assignment_expression(state), (
        "STORE_GLOBAL outside an assignment expression"
    )
    return _handle_assignment_expression(state, instr)


@register_handler("STORE_DEREF", version=PythonVersion.PY_312)
@register_handler("STORE_DEREF", version=PythonVersion.PY_313)
@register_handler("STORE_DEREF", version=PythonVersion.PY_314)
def handle_store_deref(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # STORE_DEREF stores into a closure variable: either an assignment
    # expression binding in an enclosing function, or a loop target that an
    # inner comprehension captures.
    if _is_assignment_expression(state):
        return _handle_assignment_expression(state, instr)
    return handle_store_fast(state, instr)


@register_handler("STORE_FAST", version=PythonVersion.PY_312)
@register_handler("STORE_FAST", version=PythonVersion.PY_313)
@register_handler("STORE_FAST", version=PythonVersion.PY_314)
def handle_store_fast(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    if _is_assignment_expression(state):
        # An assignment expression whose target is local to this code object,
        # as in a comprehension inlined into the lambda that binds the name.
        return _handle_assignment_expression(state, instr)

    if isinstance(state.stack[-1], ast.Name) and state.stack[-1].id == instr.argval:
        # If the variable is already on the stack, we can skip adding it again
        # This is common in nested comprehensions where the same variable is reused
        return replace(state, stack=state.stack[:-1])

    if not isinstance(state.stack[-1], TargetHole):
        return _bind_local(state, instr.argval, state.stack[-1])

    new_stack = _bind_target_hole(
        state.stack, state.stack[-1], ast.Name(id=instr.argval, ctx=ast.Store())
    )
    return replace(state, stack=new_stack[:-1])


@register_handler("STORE_FAST_LOAD_FAST", version=PythonVersion.PY_313)
@register_handler("STORE_FAST_LOAD_FAST", version=PythonVersion.PY_314)
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

    if _is_assignment_expression(state):
        # `(name := value)` whose result is read straight back, as in
        # `(z := w) + z`. The duplicate becomes the assignment expression and
        # the reload becomes a plain reference to the name just bound.
        named = ast.NamedExpr(
            target=ast.Name(id=store_name, ctx=ast.Store()),
            value=ensure_ast(state.stack[-1]),
        )
        reload = ast.Name(id=load_name, ctx=ast.Load())
        return replace(state, stack=state.stack[:-2] + [named, reload])

    if not isinstance(state.stack[-1], TargetHole):
        # A plain assignment followed by a load, as 3.14 emits when it unrolls
        # a single-iteration loop over a literal.
        bound = _bind_local(state, store_name, state.stack[-1])
        return replace(bound, stack=bound.stack + [_read_local(bound, load_name)])

    new_stack = _bind_target_hole(
        state.stack, state.stack[-1], ast.Name(id=store_name, ctx=ast.Store())
    )
    new_var = ast.Name(id=load_name, ctx=ast.Load())
    return replace(state, stack=new_stack[:-1] + [new_var])


@register_handler("STORE_FAST_STORE_FAST", version=PythonVersion.PY_313)
@register_handler("STORE_FAST_STORE_FAST", version=PythonVersion.PY_314)
def handle_store_fast_store_fast(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # STORE_FAST_STORE_FAST stores STACK[-1] into the first named variable and
    # STACK[-2] into the second. It is emitted for unpacking targets, so both
    # values are loop-target holes belonging to the same comprehension.
    assert isinstance(instr.argval, tuple)
    first_name, second_name = instr.argval

    if not isinstance(state.stack[-1], TargetHole):
        # Not loop targets: a pair of plain assignments, as 3.14 emits when it
        # unrolls a single-iteration loop over a literal.
        bound = _bind_local(state, first_name, state.stack[-1])
        return _bind_local(bound, second_name, bound.stack[-1])

    new_stack = _bind_target_hole(
        state.stack, state.stack[-1], ast.Name(id=first_name, ctx=ast.Store())
    )
    new_stack = _bind_target_hole(
        new_stack, new_stack[-2], ast.Name(id=second_name, ctx=ast.Store())
    )
    return replace(state, stack=new_stack[:-2])


@register_handler("LOAD_FAST_AND_CLEAR", version=PythonVersion.PY_312)
@register_handler("LOAD_FAST_AND_CLEAR", version=PythonVersion.PY_313)
@register_handler("LOAD_FAST_AND_CLEAR", version=PythonVersion.PY_314)
def handle_load_fast_and_clear(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LOAD_FAST_AND_CLEAR pushes a local variable onto the stack and clears it
    # For AST reconstruction, we treat this the same as LOAD_FAST
    return replace(state, stack=state.stack + [_read_local(state, instr.argval)])


@register_handler("LOAD_FAST_LOAD_FAST", version=PythonVersion.PY_313)
@register_handler("LOAD_FAST_LOAD_FAST", version=PythonVersion.PY_314)
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

    new_stack = state.stack + [_read_local(state, var1), _read_local(state, var2)]

    return replace(state, stack=new_stack)


@register_handler("MAKE_CELL", version=PythonVersion.PY_312)
@register_handler("MAKE_CELL", version=PythonVersion.PY_313)
@register_handler("MAKE_CELL", version=PythonVersion.PY_314)
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
@register_handler("COPY_FREE_VARS", version=PythonVersion.PY_314)
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
@register_handler("POP_TOP", version=PythonVersion.PY_314)
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
@register_handler("SWAP", version=PythonVersion.PY_314)
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
@register_handler("COPY", version=PythonVersion.PY_314)
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
@register_handler("PUSH_NULL", version=PythonVersion.PY_314)
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
    op_map: collections.abc.Mapping[int, ast.operator] = {
        0: ast.Add(),  # +
        1: ast.BitAnd(),  # &
        2: ast.FloorDiv(),  # //
        3: ast.LShift(),  # <<
        4: ast.MatMult(),  # @
        5: ast.Mult(),  # *
        6: ast.Mod(),  # %
        7: ast.BitOr(),  # |
        8: ast.Pow(),  # **
        9: ast.RShift(),  # >>
        10: ast.Sub(),  # -
        11: ast.Div(),  # /
        12: ast.BitXor(),  # ^
    }

    op = op_map.get(instr.arg)
    if op is None:
        raise TypeError(f"Unknown binary operation: {instr.arg}")

    return handle_binop(op, state, instr)


# 3.14 folded subscripting into BINARY_OP; `dis._nb_ops` names the oparg
# NB_SUBSCR. Looked up rather than hard-coded, since it sits past the in-place
# operators and so moves whenever one is added.
_NB_OPS: list[tuple[str, str]] = getattr(dis, "_nb_ops", [])
NB_SUBSCR: int | None = next(
    (i for i, (name, _) in enumerate(_NB_OPS) if name == "NB_SUBSCR"), None
)


@register_handler("BINARY_OP", version=PythonVersion.PY_314)
def handle_binary_op_314(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # As in 3.13, except that BINARY_OP now also implements `a[b]`, which used
    # to be its own BINARY_SUBSCR opcode.
    if instr.arg is not None and instr.arg == NB_SUBSCR:
        return handle_binary_subscr(state, instr)
    return handle_binary_op(state, instr)


@register_handler("LOAD_SMALL_INT", version=PythonVersion.PY_314)
def handle_load_small_int(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LOAD_SMALL_INT pushes an int in range(256) held in the oparg itself,
    # rather than going through co_consts.
    assert isinstance(instr.argval, int)
    return replace(state, stack=state.stack + [ensure_ast(instr.argval)])


@register_handler("LOAD_FAST_BORROW", version=PythonVersion.PY_314)
def handle_load_fast_borrow(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # A borrowed reference differs only in ownership, which the AST does not
    # model, so this is LOAD_FAST as far as reconstruction is concerned.
    return handle_load_fast(state, instr)


@register_handler("LOAD_FAST_BORROW_LOAD_FAST_BORROW", version=PythonVersion.PY_314)
def handle_load_fast_borrow_load_fast_borrow(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    return handle_load_fast_load_fast(state, instr)


@register_handler("LOAD_COMMON_CONSTANT", version=PythonVersion.PY_314)
def handle_load_common_constant(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # Pushes one of a small hardcoded set of constants. In a comprehension this
    # only shows up in the guard of an inlined builtin; see CommonConstant.
    name = getattr(instr.argval, "__name__", str(instr.argval))
    return replace(state, stack=state.stack + [CommonConstant(id=name)])


@register_handler("NOT_TAKEN", version=PythonVersion.PY_314)
def handle_not_taken(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # A no-op marking the not-taken edge of a branch for sys.monitoring.
    return state


@register_handler("POP_ITER", version=PythonVersion.PY_314)
def handle_pop_iter(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # POP_ITER discards the exhausted iterator that FOR_ITER left behind. In
    # 3.13 the same cleanup was spelled END_FOR followed by POP_TOP.
    return replace(state, stack=state.stack[:-1])


# ============================================================================
# UNARY OPERATION HANDLERS
# ============================================================================


def handle_unary_op(
    op: ast.unaryop, state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    operand = ensure_ast(state.stack[-1])
    new_stack = state.stack[:-1] + [ast.UnaryOp(op=op, operand=operand)]
    return replace(state, stack=new_stack)


UNARY_OPS: dict[str, ast.unaryop] = {
    "UNARY_NEGATIVE": ast.USub(),
    "UNARY_INVERT": ast.Invert(),
    "UNARY_NOT": ast.Not(),
}

# These three behave identically in every dialect this module supports; 3.13's
# "requires an exact bool operand" note on UNARY_NOT constrains the operand, not
# the reconstruction.
for _opname, _op in UNARY_OPS.items():
    for _version in (
        PythonVersion.PY_312,
        PythonVersion.PY_313,
        PythonVersion.PY_314,
    ):
        register_handler(
            _opname, functools.partial(handle_unary_op, _op), version=_version
        )


@register_handler("CONVERT_VALUE", version=PythonVersion.PY_313)
@register_handler("CONVERT_VALUE", version=PythonVersion.PY_314)
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
@register_handler("CALL_INTRINSIC_1", version=PythonVersion.PY_314)
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
@register_handler("TO_BOOL", version=PythonVersion.PY_314)
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
@register_handler("COMPARE_OP", version=PythonVersion.PY_314)
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
@register_handler("CONTAINS_OP", version=PythonVersion.PY_314)
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
@register_handler("IS_OP", version=PythonVersion.PY_314)
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


@register_handler("KW_NAMES", version=PythonVersion.PY_312)
def handle_kw_names(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # KW_NAMES names the trailing arguments of the CALL that follows it.
    # Python 3.13 replaced this pair with a single CALL_KW instruction.
    assert isinstance(instr.argval, tuple), "KW_NAMES requires a tuple of names"
    assert all(isinstance(name, str) for name in instr.argval)
    assert state.kw_names is None, "KW_NAMES must be consumed by the following CALL"
    return replace(state, kw_names=instr.argval)


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

    # A preceding KW_NAMES names the trailing `len(kw_names)` positional slots.
    keywords: list[ast.keyword] = []
    if state.kw_names is not None:
        assert 0 < len(state.kw_names) <= arg_count
        keywords = [
            ast.keyword(arg=name, value=value)
            for name, value in zip(state.kw_names, args[-len(state.kw_names) :])
        ]
        args = args[: -len(state.kw_names)]

    if isinstance(func, CompLambda):
        assert len(args) == 1 and not keywords
        return replace(state, stack=new_stack + [func.inline(args[0])], kw_names=None)
    else:
        # Create function call AST
        call_node = ast.Call(func=func, args=args, keywords=keywords)
        new_stack = new_stack + [call_node]
        return replace(state, stack=new_stack, kw_names=None)


@register_handler("CALL", version=PythonVersion.PY_313)
@register_handler("CALL", version=PythonVersion.PY_314)
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
@register_handler("CALL_KW", version=PythonVersion.PY_314)
def handle_call_kw(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # CALL_KW pops function, arguments, and keyword names from stack
    assert instr.arg is not None
    arg_count: int = instr.arg
    assert arg_count > 0, "CALL_KW requires at least one argument"

    func = ensure_ast(state.stack[-arg_count - 3])
    assert not isinstance(func, CompLambda | Null)

    kw_names = state.stack[-1]
    assert isinstance(kw_names, ast.Tuple), "Expected a tuple of keyword names"
    assert len(kw_names.elts) > 0, "Expected at least one keyword name"

    # Pop arguments, function, and keyword names
    keywords = []
    for i, kw in enumerate(reversed(kw_names.elts)):
        assert isinstance(kw, ast.Constant) and isinstance(kw.value, str)
        keywords += [ast.keyword(arg=kw.value, value=ensure_ast(state.stack[-2 - i]))]
    keywords.reverse()

    args = [ensure_ast(a) for a in state.stack[-arg_count - 1 : -len(keywords) - 1]]
    if not isinstance(state.stack[-arg_count - 2], Null):
        args = [ensure_ast(state.stack[-arg_count - 2])] + args

    # Create function call AST
    call_node = ast.Call(func=func, args=args, keywords=keywords)
    new_stack = state.stack[: -arg_count - 3] + [call_node]
    return replace(state, stack=new_stack)


# Flags shared by MAKE_FUNCTION (3.12) and SET_FUNCTION_ATTRIBUTE (3.13)
MAKE_FUNCTION_DEFAULTS = 0x01
MAKE_FUNCTION_KWDEFAULTS = 0x02
MAKE_FUNCTION_ANNOTATIONS = 0x04
MAKE_FUNCTION_CLOSURE = 0x08
MAKE_FUNCTION_ANNOTATE = 0x10  # added in 3.14
MAKE_FUNCTION_FLAGS = (
    MAKE_FUNCTION_DEFAULTS,
    MAKE_FUNCTION_KWDEFAULTS,
    MAKE_FUNCTION_ANNOTATIONS,
    MAKE_FUNCTION_CLOSURE,
)


def _apply_function_attribute(
    func: ast.Lambda | CompLambda, flag: int, value: ast.expr
) -> ast.Lambda | CompLambda:
    """Attach one function attribute to a reconstructed lambda."""
    if flag == MAKE_FUNCTION_CLOSURE:
        # Free variables are already spelled by name in the reconstructed body.
        return func
    if flag == MAKE_FUNCTION_ANNOTATE:
        # A lambda has no annotations, and the AST does not carry the lazy
        # annotate function 3.14 attaches to annotated functions.
        return func

    assert isinstance(func, ast.Lambda) and not isinstance(func, CompLambda), (
        "Only lambdas carry defaults; comprehensions take exactly one argument"
    )

    if flag == MAKE_FUNCTION_DEFAULTS:
        # A tuple of defaults for the *trailing* positional parameters.
        assert isinstance(value, ast.Tuple), "Expected a tuple of default values"
        func.args.defaults = list(value.elts)
    elif flag == MAKE_FUNCTION_KWDEFAULTS:
        # A dict mapping keyword-only parameter names to their defaults.
        assert isinstance(value, ast.Dict), "Expected a dict of keyword defaults"
        by_name = {
            key.value: val
            for key, val in zip(value.keys, value.values)
            if isinstance(key, ast.Constant)
        }
        func.args.kw_defaults = [by_name.get(a.arg) for a in func.args.kwonlyargs]
    else:
        raise NotImplementedError("Function annotations are not supported")

    return func


def _split_callable(
    first: ast.expr, second: ast.expr
) -> tuple[ast.expr, ast.expr | None]:
    """Separate the callable from the NULL-or-self slot beside it.

    Which of the two comes first varies: LOAD_GLOBAL and LOAD_ATTR report the
    order in their argrepr, and it differs between 3.13 and 3.14.
    """
    if isinstance(first, Null):
        return second, None
    elif isinstance(second, Null):
        return first, None
    else:
        return first, second


def _build_variadic_call(
    func: ast.expr,
    self_arg: ast.expr | None,
    positional: ast.expr,
    keyword_mapping: ast.expr | None,
) -> ast.Call:
    """Assemble `func(*positional, **keyword_mapping)`.

    CALL_FUNCTION_EX receives its arguments already collected into a sequence
    and a mapping, with the original mix of plain and starred arguments no
    longer distinguishable. Spelling every argument as unpacked reproduces the
    call exactly, even where the source did not use `*` for all of them.
    """
    args: list[ast.expr] = [] if self_arg is None else [ensure_ast(self_arg)]
    args.append(ast.Starred(value=ensure_ast(positional), ctx=ast.Load()))
    keywords = (
        []
        if keyword_mapping is None
        else [ast.keyword(arg=None, value=ensure_ast(keyword_mapping))]
    )
    return ast.Call(func=ensure_ast(func), args=args, keywords=keywords)


@register_handler("CALL_FUNCTION_EX", version=PythonVersion.PY_312)
@register_handler("CALL_FUNCTION_EX", version=PythonVersion.PY_313)
def handle_call_function_ex(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # Stack: callable and NULL-or-self, the positional sequence, and -- only
    # when the low bit of the oparg is set -- the keyword mapping.
    size = 4 if instr.arg else 3
    keyword_mapping = state.stack[-1] if instr.arg else None
    positional = state.stack[-2] if instr.arg else state.stack[-1]
    func, self_arg = _split_callable(state.stack[-size], state.stack[-size + 1])

    call = _build_variadic_call(func, self_arg, positional, keyword_mapping)
    return replace(state, stack=state.stack[:-size] + [call])


@register_handler("CALL_FUNCTION_EX", version=PythonVersion.PY_314)
def handle_call_function_ex_314(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # 3.14 always reserves the keyword-mapping slot, pushing NULL into it when
    # the call has no `**` argument, so the layout is a fixed four slots.
    keyword_mapping = None if isinstance(state.stack[-1], Null) else state.stack[-1]
    func, self_arg = _split_callable(state.stack[-4], state.stack[-3])

    call = _build_variadic_call(func, self_arg, state.stack[-2], keyword_mapping)
    return replace(state, stack=state.stack[:-4] + [call])


@register_handler("MAKE_FUNCTION", version=PythonVersion.PY_312)
def handle_make_function_312(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # MAKE_FUNCTION in Python 3.12 uses flags to determine stack consumption.
    # Unlike 3.10 there is no qualified name on the stack, and unlike 3.13 the
    # extra attributes travel with this instruction rather than with a following
    # SET_FUNCTION_ATTRIBUTE. They are pushed in ascending flag order, below the
    # code object.
    assert instr.arg is not None
    assert isinstance(state.stack[-1], ast.Lambda | CompLambda), (
        "Expected a function object (Lambda or CompLambda) on the stack."
    )

    set_flags = [flag for flag in MAKE_FUNCTION_FLAGS if instr.arg & flag]
    attributes = state.stack[-1 - len(set_flags) : -1]

    func = copy.deepcopy(state.stack[-1])
    for flag, value in zip(set_flags, attributes):
        func = _apply_function_attribute(func, flag, value)

    new_stack = state.stack[: -1 - len(set_flags)] + [func]
    return replace(state, stack=new_stack)


# Python 3.13 version
@register_handler("MAKE_FUNCTION", version=PythonVersion.PY_313)
@register_handler("MAKE_FUNCTION", version=PythonVersion.PY_314)
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
@register_handler("SET_FUNCTION_ATTRIBUTE", version=PythonVersion.PY_314)
def handle_set_function_attribute(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # SET_FUNCTION_ATTRIBUTE sets one attribute on a function object. Python
    # 3.13 uses it in place of the MAKE_FUNCTION flags; the stack holds the
    # attribute value below the function, and only the function is left behind.
    assert instr.arg is not None
    assert isinstance(state.stack[-1], ast.Lambda | CompLambda), (
        "Expected a function object (Lambda or CompLambda) on the stack."
    )

    func = _apply_function_attribute(
        copy.deepcopy(state.stack[-1]), instr.arg, state.stack[-2]
    )
    return replace(state, stack=state.stack[:-2] + [func])


# ============================================================================
# OBJECT ACCESS HANDLERS
# ============================================================================


@register_handler("LOAD_ATTR", version=PythonVersion.PY_312)
@register_handler("LOAD_ATTR", version=PythonVersion.PY_313)
@register_handler("LOAD_ATTR", version=PythonVersion.PY_314)
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
@register_handler("BINARY_SLICE", version=PythonVersion.PY_314)
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
@register_handler("UNPACK_SEQUENCE", version=PythonVersion.PY_314)
def handle_unpack_sequence(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # UNPACK_SEQUENCE splits a comprehension loop target into `arg` sub-targets,
    # as in ((k, v) for k, v in items). The names are not known yet, so the
    # single target hole is refined into a tuple of fresh holes, which the
    # following STORE_* instructions bind one at a time.
    #
    # CPython pushes the unpacked values right-to-left, so element 0 ends up on
    # top of the stack and is consumed by the first STORE_*.
    assert instr.arg is not None
    unpack_count: int = instr.arg

    if not isinstance(state.stack[-1], TargetHole):
        # Destructuring a known value rather than a loop target, as 3.14 emits
        # when it unrolls a single-iteration loop over a literal.
        elements = _literal_elements(state.stack[-1], unpack_count)
        return replace(state, stack=state.stack[:-1] + list(reversed(elements)))

    holes = [TargetHole() for _ in range(unpack_count)]
    new_stack = _bind_target_hole(
        state.stack, state.stack[-1], ast.Tuple(elts=list(holes), ctx=ast.Store())
    )
    return replace(state, stack=new_stack[:-1] + list(reversed(holes)))


@register_handler("UNPACK_EX", version=PythonVersion.PY_312)
@register_handler("UNPACK_EX", version=PythonVersion.PY_313)
@register_handler("UNPACK_EX", version=PythonVersion.PY_314)
def handle_unpack_ex(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # UNPACK_EX handles a starred target, as in ((a, b) for a, *b in pairs).
    # The low byte of the argument counts the targets before the starred one and
    # the high byte counts those after it; the starred target itself collects
    # whatever is left over. As with UNPACK_SEQUENCE the values are pushed
    # right-to-left, so the first target ends up on top of the stack.
    assert instr.arg is not None
    before, after = instr.arg & 0xFF, instr.arg >> 8

    if not isinstance(state.stack[-1], TargetHole):
        # Destructuring a known value; the starred target collects the middle.
        elements = _literal_elements(state.stack[-1])
        assert len(elements) >= before + after, "Too few values to unpack"
        middle = elements[before : len(elements) - after]
        unpacked: list[ast.expr] = [
            *elements[:before],
            ast.List(elts=list(middle), ctx=ast.Load()),
            *elements[len(elements) - after :],
        ]
        return replace(state, stack=state.stack[:-1] + list(reversed(unpacked)))

    holes = [TargetHole() for _ in range(before + 1 + after)]
    elts: list[ast.expr] = list(holes)
    elts[before] = ast.Starred(value=holes[before], ctx=ast.Store())

    new_stack = _bind_target_hole(
        state.stack, state.stack[-1], ast.Tuple(elts=elts, ctx=ast.Store())
    )
    return replace(state, stack=new_stack[:-1] + list(reversed(holes)))


@register_handler("BUILD_TUPLE", version=PythonVersion.PY_312)
@register_handler("BUILD_TUPLE", version=PythonVersion.PY_313)
@register_handler("BUILD_TUPLE", version=PythonVersion.PY_314)
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
@register_handler("BUILD_SLICE", version=PythonVersion.PY_314)
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
@register_handler("LIST_EXTEND", version=PythonVersion.PY_314)
def handle_list_extend(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # LIST_EXTEND appends the contents of the iterable at TOS to the list
    # further down the stack. That list is either the empty ListComp that
    # BUILD_LIST(0) optimistically created -- a list display, not a
    # comprehension after all -- or a partly built argument list for a call
    # with a starred argument.
    update = state.stack[-1]
    target = state.stack[-instr.argval - 1]

    # A literal iterable contributes its elements directly; anything else has to
    # stay unpacked, as in `[*whatever]`.
    elements: list[ast.expr]
    if isinstance(update, ast.Tuple | ast.List):
        elements = [ensure_ast(e) for e in update.elts]
    else:
        elements = [ast.Starred(value=ensure_ast(update), ctx=ast.Load())]

    if isinstance(target, ast.ListComp) and not target.generators:
        merged = ast.List(elts=elements, ctx=ast.Load())
    else:
        assert isinstance(target, ast.List), "LIST_EXTEND expects a list to extend"
        merged = ast.List(elts=list(target.elts) + elements, ctx=ast.Load())

    new_stack = state.stack[:-1]
    new_stack[-instr.argval] = merged
    return replace(state, stack=new_stack)


@register_handler("DICT_MERGE", version=PythonVersion.PY_312)
@register_handler("DICT_MERGE", version=PythonVersion.PY_313)
@register_handler("DICT_MERGE", version=PythonVersion.PY_314)
def handle_dict_merge(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # DICT_MERGE folds the mapping at TOS into the one below it, rejecting
    # duplicate keys. It assembles the keyword arguments of a call using `**`.
    update = state.stack[-1]
    target = state.stack[-instr.argval - 1]

    # An `ast.Dict` entry with a key of None is `**value`, which is how a
    # mapping that is not a literal has to be spliced in.
    def entries(node: ast.expr) -> tuple[list[ast.expr | None], list[ast.expr]]:
        if isinstance(node, ast.Dict):
            return list(node.keys), list(node.values)
        return [None], [ensure_ast(node)]

    if isinstance(target, ast.DictComp) and not target.generators:
        # BUILD_MAP(0) guessed at a dict comprehension; it was a `**` argument.
        keys, values = entries(update)
    else:
        assert isinstance(target, ast.Dict), "DICT_MERGE expects a dict to merge into"
        target_keys, target_values = entries(target)
        update_keys, update_values = entries(update)
        keys, values = target_keys + update_keys, target_values + update_values

    new_stack = state.stack[:-1]
    new_stack[-instr.argval] = ast.Dict(keys=keys, values=values)
    return replace(state, stack=new_stack)


@register_handler("SET_UPDATE", version=PythonVersion.PY_312)
@register_handler("SET_UPDATE", version=PythonVersion.PY_313)
@register_handler("SET_UPDATE", version=PythonVersion.PY_314)
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
@register_handler("DICT_UPDATE", version=PythonVersion.PY_314)
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
@register_handler("BUILD_STRING", version=PythonVersion.PY_314)
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
@register_handler("FORMAT_SIMPLE", version=PythonVersion.PY_314)
def handle_format_simple(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # FORMAT_SIMPLE formats a string with a single value
    # Pops the value and the format string from the stack
    assert len(state.stack) >= 1, "Not enough items on stack for FORMAT_SIMPLE"
    value = state.stack[-1]

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
    new_stack = state.stack[:-1] + [ast.JoinedStr(values=[formatted_node])]
    return replace(state, stack=new_stack)


@register_handler("FORMAT_WITH_SPEC", version=PythonVersion.PY_313)
@register_handler("FORMAT_WITH_SPEC", version=PythonVersion.PY_314)
def handle_format_with_spec(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # FORMAT_WITH_SPEC formats a value with a format specifier
    # Stack order in Python 3.13: format_spec on top, value below
    assert len(state.stack) >= 2, "Not enough items on stack for FORMAT_WITH_SPEC"
    format_spec = ensure_ast(state.stack[-1])  # Format spec is on top
    value = state.stack[-2]  # Value is below

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
    new_stack = state.stack[:-2] + [ast.JoinedStr(values=[formatted_node])]
    return replace(state, stack=new_stack)


# ============================================================================
# CONDITIONAL JUMP HANDLERS
# ============================================================================


def _handle_pop_jump_if(
    f_condition: Callable[[ast.expr], ast.expr],
    state: ReconstructionState,
    instr: dis.Instruction,
) -> ReconstructionState:
    # Generic handler for POP_JUMP_IF_* instructions. Pops a value from the
    # stack; `condition` is true exactly when the jump is taken.
    condition: ast.expr = f_condition(ensure_ast(state.stack[-1]))

    # An inlined-builtin guard is an implementation detail of the interpreter,
    # not part of the comprehension: drop it rather than record it as a filter.
    if _specialization_guard_edge(state, instr) is not None:
        return replace(state, stack=state.stack[:-1])

    kind, _ = _classify_branch(state, instr)
    edge = state.branches.get(instr.offset, BranchEdge.TAKE_JUMP)

    if kind is BranchKind.TERNARY:
        return _handle_conditional_expression(state, instr, condition, edge)

    # A filter. The guard is the condition under which *this* path carries on
    # toward the element, so it is negated when the path falls through.
    guard = condition if edge is BranchEdge.TAKE_JUMP else _negate(condition)
    return _attach_filter(state, guard)


def _attach_filter(
    state: ReconstructionState, guard: ast.expr | None
) -> ReconstructionState:
    """Conjoin ``guard`` to the filters of the innermost unfinished comprehension."""
    for pos, item in zip(reversed(range(len(state.stack))), reversed(state.stack)):
        if not isinstance(item, CompExp):
            continue

        elt: ast.expr = item.value if isinstance(item, ast.DictComp) else item.elt
        new_result: CompExp = copy.deepcopy(item)

        if isinstance(elt, Placeholder):
            resolved = guard
        elif isinstance(elt, ast.IfExp) and any(
            isinstance(x, Placeholder) for x in ast.walk(elt)
        ):
            # A conditional expression was being built up in the element slot,
            # but it turned out to be part of this filter's condition. Move it
            # back out, plugging the guard into the arm still awaiting a value.
            if isinstance(new_result, ast.DictComp):
                new_result.key, new_result.value = Placeholder(), Placeholder()
            else:
                new_result.elt = Placeholder()

            if guard is None:
                resolved = None
            else:
                plugged = ReplacePlaceholder(guard).visit(copy.deepcopy(elt))
                assert isinstance(plugged, ast.expr)
                resolved = plugged
        else:
            continue

        if resolved is not None:
            ifs = new_result.generators[-1].ifs
            combined = _conjoin(ifs + [resolved])
            assert combined is not None
            new_result.generators[-1].ifs = [combined]

        new_stack = state.stack[:pos] + [new_result] + state.stack[pos + 1 : -1]
        return replace(state, stack=new_stack)

    raise TypeError("No comprehension context found for filter condition")


def _handle_conditional_expression(
    state: ReconstructionState,
    instr: dis.Instruction,
    condition: ast.expr,
    edge: BranchEdge,
) -> ReconstructionState:
    """Start an ``ast.IfExp``, marking the arm this path did not take."""
    for pos, item in zip(reversed(range(len(state.stack))), reversed(state.stack)):
        if any(isinstance(x, Placeholder) for x in ast.walk(item)):
            body: Skipped | Placeholder
            orelse: Skipped | Placeholder
            skipped = Skipped(id=f".SKIPPED_{instr.offset}")
            if edge is BranchEdge.FALL_THROUGH:
                body, orelse = skipped, Placeholder()
            else:
                body, orelse = Placeholder(), skipped

            new_ifexp = ast.IfExp(test=condition, body=body, orelse=orelse)
            new_result = ReplacePlaceholder(new_ifexp).visit(copy.deepcopy(item))
            new_stack = state.stack[:pos] + [new_result] + state.stack[pos + 1 : -1]
            return replace(state, stack=new_stack)

    raise TypeError("No placeholder found for conditional expression")


@register_handler("POP_JUMP_IF_TRUE", version=PythonVersion.PY_312)
@register_handler("POP_JUMP_IF_TRUE", version=PythonVersion.PY_313)
@register_handler("POP_JUMP_IF_TRUE", version=PythonVersion.PY_314)
def handle_pop_jump_if_true(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # POP_JUMP_IF_TRUE pops a value from the stack and jumps if it's true
    # In Python 3.13, this is used for filter conditions where True means continue
    return _handle_pop_jump_if(lambda c: c, state, instr)


@register_handler("POP_JUMP_IF_FALSE", version=PythonVersion.PY_312)
@register_handler("POP_JUMP_IF_FALSE", version=PythonVersion.PY_313)
@register_handler("POP_JUMP_IF_FALSE", version=PythonVersion.PY_314)
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
@register_handler("POP_JUMP_IF_NONE", version=PythonVersion.PY_314)
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
@register_handler("POP_JUMP_IF_NOT_NONE", version=PythonVersion.PY_314)
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


@register_handler("JUMP_FORWARD", version=PythonVersion.PY_312)
@register_handler("JUMP_FORWARD", version=PythonVersion.PY_313)
@register_handler("JUMP_FORWARD", version=PythonVersion.PY_314)
def handle_jump_forward(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # JUMP_FORWARD is used to jump forward in the code
    # In generator expressions, this is often used to skip code in conditional logic
    return state


@register_handler("JUMP_BACKWARD", version=PythonVersion.PY_312)
@register_handler("JUMP_BACKWARD", version=PythonVersion.PY_313)
@register_handler("JUMP_BACKWARD", version=PythonVersion.PY_314)
def handle_jump_backward(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # JUMP_BACKWARD is used to jump back to the beginning of a loop (replaces JUMP_ABSOLUTE in 3.13)
    # In generator expressions, this typically indicates the end of the loop body
    return state


@register_handler("JUMP_BACKWARD_NO_INTERRUPT", version=PythonVersion.PY_312)
@register_handler("JUMP_BACKWARD_NO_INTERRUPT", version=PythonVersion.PY_313)
@register_handler("JUMP_BACKWARD_NO_INTERRUPT", version=PythonVersion.PY_314)
def handle_jump_backward_no_interrupt(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    raise TypeError(
        "JUMP_BACKWARD_NO_INTERRUPT instruction should not appear in generator comprehensions"
    )


@register_handler("JUMP_NO_INTERRUPT", version=PythonVersion.PY_312)
@register_handler("JUMP_NO_INTERRUPT", version=PythonVersion.PY_313)
@register_handler("JUMP_NO_INTERRUPT", version=PythonVersion.PY_314)
def handle_jump_no_interrupt(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    raise TypeError(
        "JUMP_NO_INTERRUPT instruction should not appear in generator comprehensions"
    )


@register_handler("JUMP", version=PythonVersion.PY_312)
@register_handler("JUMP", version=PythonVersion.PY_313)
@register_handler("JUMP", version=PythonVersion.PY_314)
def handle_jump(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    raise TypeError("JUMP instruction should not appear in generator comprehensions")


@register_handler("EXTENDED_ARG", version=PythonVersion.PY_312)
@register_handler("EXTENDED_ARG", version=PythonVersion.PY_313)
@register_handler("EXTENDED_ARG", version=PythonVersion.PY_314)
def handle_extended_arg(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # EXTENDED_ARG prefixes an instruction whose argument does not fit in a
    # byte. `dis` has already folded it into the following instruction's `arg`,
    # so there is nothing left to do here.
    return state


@register_handler("RESUME", version=PythonVersion.PY_312)
@register_handler("RESUME", version=PythonVersion.PY_313)
@register_handler("RESUME", version=PythonVersion.PY_314)
def handle_resume(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    # RESUME is used for resuming execution after yield/await - mostly no-op for AST reconstruction
    return state


@register_handler("SEND", version=PythonVersion.PY_312)
@register_handler("SEND", version=PythonVersion.PY_313)
@register_handler("SEND", version=PythonVersion.PY_314)
def handle_send(
    state: ReconstructionState, instr: dis.Instruction
) -> ReconstructionState:
    raise TypeError("SEND instruction should not appear in generator comprehensions")


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


def _unconsumed(value: Iterator) -> typing.Any:
    """Return the items an iterator has not yet yielded, as a concrete sequence.

    Built-in sequence iterators pickle as ``(iter, (underlying,), index)``, where
    ``index`` is how far the iterator has advanced (absent or ``None`` when it
    does not apply). ``reversed`` objects pickle with ``reversed`` as the
    callable and count *down* from the end of the underlying sequence.
    """
    reduced = value.__reduce__()
    assert isinstance(reduced, tuple) and len(reduced) >= 2, (
        f"Cannot recover the contents of {type(value)}"
    )
    if not reduced[1]:  # an exhausted iterator pickles with no arguments
        return ()

    underlying = reduced[1][0]
    index = reduced[2] if len(reduced) > 2 and reduced[2] is not None else 0
    return underlying[index::-1] if reduced[0] is reversed else underlying[index:]


@ensure_ast.register(type(iter((1,))))
def _ensure_ast_tuple_iterator(value: Iterator) -> ast.Tuple:
    return ensure_ast(tuple(_unconsumed(value)))  # type: ignore


@ensure_ast.register
def _ensure_ast_list(value: list) -> ast.List:
    return ast.List(elts=[ensure_ast(v) for v in value], ctx=ast.Load())


@ensure_ast.register(type(iter([1])))
@ensure_ast.register(type(iter({1: 2}.values())))
@ensure_ast.register(type(iter({1: 2}.items())))
@ensure_ast.register(type(iter(reversed([1]))))
@ensure_ast.register(reversed)
def _ensure_ast_list_iterator(value: Iterator) -> ast.List:
    return ensure_ast(list(_unconsumed(value)))  # type: ignore


@ensure_ast.register(type(iter("ab")))  # str_ascii_iterator
@ensure_ast.register(type(iter("\xe9b")))  # str_iterator
@ensure_ast.register(type(iter(b"ab")))
@ensure_ast.register(type(iter(bytearray(b"ab"))))
def _ensure_ast_str_iterator(value: Iterator) -> ast.Constant:
    remainder = _unconsumed(value)
    # bytearray iteration yields ints, exactly as bytes iteration does
    return ensure_ast(  # type: ignore
        bytes(remainder) if isinstance(remainder, bytearray) else remainder
    )


@ensure_ast.register(set)
@ensure_ast.register(frozenset)
def _ensure_ast_set(value: set | frozenset) -> ast.Set:
    return ast.Set(elts=[ensure_ast(v) for v in value])


@ensure_ast.register(type(iter({1})))
def _ensure_ast_set_iterator(value: Iterator) -> ast.Set:
    return ensure_ast(set(_unconsumed(value)))  # type: ignore


@ensure_ast.register
def _ensure_ast_dict(value: dict) -> ast.Dict:
    return ast.Dict(
        keys=[ensure_ast(k) for k in value.keys()],
        values=[ensure_ast(v) for v in value.values()],
    )


@ensure_ast.register(type(iter({1: 2})))
def _ensure_ast_dict_iterator(value: Iterator) -> ast.expr:
    return ensure_ast(_unconsumed(value))


@ensure_ast.register(types.BuiltinFunctionType)
@ensure_ast.register(type)
def _ensure_ast_builtin(value: typing.Callable) -> ast.Name:
    """A built-in callable is referred to by name, which resolves via builtins.

    Covers both built-in functions (``abs``) and built-in types used as
    callables (``bool``), which appear as the predicate of a ``filter`` or the
    function of a ``map``.
    """
    name = getattr(value, "__name__", None)
    assert name and getattr(builtins, name, None) is value, (
        f"Cannot reference non-builtin callable {value!r}"
    )
    return ast.Name(id=name, ctx=ast.Load())


@ensure_ast.register(zip)
@ensure_ast.register(enumerate)
@ensure_ast.register(map)
@ensure_ast.register(filter)
def _ensure_ast_iterator_adaptor(value: Iterator) -> ast.Call:
    """Rebuild zip/enumerate/map/filter from the arguments they pickle with.

    These wrap other iterators rather than a concrete sequence, so unlike a list
    or range iterator they cannot be materialised -- but ``__reduce__`` hands
    back their constituent parts, each of which ``ensure_ast`` can handle in
    turn. Any already-consumed prefix is reflected in the inner iterators.
    """
    reduced = value.__reduce__()
    if isinstance(reduced, str):
        raise TypeError(f"Cannot convert {type(value)} to AST node")
    func, args = reduced[:2]
    return ast.Call(
        func=ast.Name(id=func.__name__, ctx=ast.Load()),
        args=[ensure_ast(arg) for arg in args],
        keywords=[],
    )


@ensure_ast.register
def _ensure_ast_slice(value: slice) -> ast.Slice:
    """A constant slice, as 3.14 emits for `s[1:3]` alongside BINARY_OP/NB_SUBSCR."""
    return ast.Slice(
        lower=None if value.start is None else ensure_ast(value.start),
        upper=None if value.stop is None else ensure_ast(value.stop),
        step=None if value.step is None else ensure_ast(value.step),
    )


@ensure_ast.register
def _ensure_ast_range(value: range) -> ast.Call:
    return ast.Call(
        func=ast.Name(id="range", ctx=ast.Load()),
        args=[ensure_ast(value.start), ensure_ast(value.stop), ensure_ast(value.step)],
        keywords=[],
    )


@ensure_ast.register(type(iter(range(1))))
def _ensure_ast_range_iterator(value: Iterator) -> ast.Call:
    return ensure_ast(_unconsumed(value))  # type: ignore


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
        isinstance(
            x,
            Placeholder
            | Skipped
            | TargetHole
            | CommonConstant
            | Null
            | CompLambda
            | ConvertedValue,
        )
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
        # co_varnames lists parameters first: positional, keyword-only, then
        # *args and **kwargs if present. Default values are not part of the code
        # object -- they are pushed by the caller and attached by MAKE_FUNCTION
        # (3.12) or SET_FUNCTION_ATTRIBUTE (3.13).
        names = value.co_varnames
        n_args, n_kwonly = value.co_argcount, value.co_kwonlyargcount
        n_params = n_args + n_kwonly

        vararg = kwarg = None
        if value.co_flags & inspect.CO_VARARGS:
            vararg = ast.arg(arg=names[n_params])
            n_params += 1
        if value.co_flags & inspect.CO_VARKEYWORDS:
            kwarg = ast.arg(arg=names[n_params])

        args = ast.arguments(
            posonlyargs=[ast.arg(arg=arg) for arg in names[: value.co_posonlyargcount]],
            args=[ast.arg(arg=arg) for arg in names[value.co_posonlyargcount : n_args]],
            vararg=vararg,
            kwonlyargs=[ast.arg(arg=arg) for arg in names[n_args : n_args + n_kwonly]],
            kw_defaults=[None] * n_kwonly,
            kwarg=kwarg,
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
    assert genexpr.gi_frame is not None, "Generator must not be exhausted"
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


def disassemble(
    genexpr: Generator[typing.Any, typing.Any, typing.Any],
) -> ast.Expression:
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
