"""Naming the operation a type checker's error about ``Operation.__call__`` concerns.

Both checkers report a wrong call to an operation against ``Operation.__call__``,
whose parameters are the generic ``*args: Q.args, **kwargs: Q.kwargs``, so the
operation's own name and signature never reach the code's author. Both checkers do
know the signature, and reveal it for an operation-typed expression. What is shared
between them lives here: finding the call an error is about, and building the copy
of the source in which each such callee is probed with ``typing.reveal_type``. Each
checker reads its own output and renders its own wording.
"""

import ast
import re

# The module alias the probe reaches `typing.reveal_type` through; lengthened until
# it appears nowhere in the source, so no binding there can intercept the probe.
PROBE_ALIAS = "_effectful_type_probe"


def character_column(lines: list[str], line: int, byte_column: int) -> int:
    """The 0-based character column of `byte_column` (as ``ast`` counts, in UTF-8
    bytes) on the 1-based `line` of `lines`."""
    return len(lines[line - 1].encode()[:byte_column].decode())


def owner(
    tree: ast.Module, source: str, line: int, column: int, *, argument: bool
) -> ast.Call | None:
    """The call an error at `line` and 0-based character `column` is about.

    `argument` says the checker placed the error at an argument of the call, as
    both do for a wrongly typed argument, rather than at the call itself. The
    innermost call around the position is not the owner: a wrong nested call
    ``op(other(x))`` is reported at ``other(x)``, which belongs to ``op``.
    """
    lines = source.splitlines()

    def starts_at(node: ast.expr | ast.keyword) -> bool:
        return (
            node.lineno == line
            and character_column(lines, node.lineno, node.col_offset) == column
        )

    def arguments(call: ast.Call) -> list[ast.expr | ast.keyword]:
        return [*call.args, *call.keywords]

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    if argument:
        return next(
            (call for call in calls for arg in arguments(call) if starts_at(arg)),
            None,
        )
    return max(
        (call for call in calls if starts_at(call)),
        key=lambda call: (call.end_lineno or 0, call.end_col_offset or 0),
        default=None,
    )


def probed(
    source: str, tree: ast.Module, calls: list[ast.Call]
) -> tuple[str, dict[tuple[int, int], tuple[int, int]]]:
    """`source` with each call's callee ``f`` replaced by ``<alias>.reveal_type(f)``,
    and where each callee now starts, keyed by its original position.

    `reveal_type` returns its argument, so the copy binds and evaluates exactly as
    the original, including names bound by a comprehension or a parameter. The
    ``import typing as <alias>`` line goes after any docstring and ``__future__``
    imports, which must stay first. Positions are 1-based lines and 0-based
    character columns in the copy.
    """
    alias = PROBE_ALIAS
    while alias in source:
        alias += "_"
    funcs = {(call.func.lineno, call.func.col_offset): call.func for call in calls}
    probe = f"{alias}.reveal_type(".encode()
    # (line, byte column, text) of every insertion, in original coordinates.
    insertions = [
        insertion
        for func in funcs.values()
        for insertion in (
            (func.lineno, func.col_offset, probe),
            (func.end_lineno or func.lineno, func.end_col_offset or 0, b")"),
        )
    ]
    lines = [line.encode() for line in source.splitlines(keepends=True)]
    for line, col, text in sorted(insertions, reverse=True):
        lines[line - 1] = lines[line - 1][:col] + text + lines[line - 1][col:]
    header = 0
    for index, statement in enumerate(tree.body):
        docstring = (
            index == 0
            and isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
        future = (
            isinstance(statement, ast.ImportFrom) and statement.module == "__future__"
        )
        if not (docstring or future):
            break
        header = statement.end_lineno or statement.lineno
    positions = {}
    for key, func in funcs.items():
        shift = sum(
            len(text)
            for line, col, text in insertions
            if line == func.lineno and col <= func.col_offset
        )
        prefix = lines[func.lineno - 1][: func.col_offset + shift]
        # The alias import adds one line above every probe.
        positions[key] = (func.lineno + 1, len(prefix.decode()))
    copy = [line.decode() for line in lines]
    copy.insert(header, f"import typing as {alias}\n")
    return "".join(copy), positions


def callee(source: str, call: ast.Call) -> str:
    """The callee expression as written."""
    return ast.get_source_segment(source, call.func) or ast.unparse(call.func)


def missing_parameters_declared(message: str, params: str) -> bool:
    """Whether every parameter `message` names as missing appears in `params`, a
    revealed signature; a missing parameter it does not declare means the owner
    was misidentified."""
    match = re.search(
        r"(?:parameters?|argument) ((?:[`\"][^`\"]+[`\"](?:, )?)+)", message
    )
    if (
        match is None
        or "missing" not in message.lower()
        and "No argument" not in message
    ):
        return True
    return all(
        re.search(rf"\b{re.escape(name)}\b", params)
        for name in re.findall(r"[`\"]([^`\"]+)[`\"]", match[1])
    )
