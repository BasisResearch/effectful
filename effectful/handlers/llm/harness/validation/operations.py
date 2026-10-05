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
_PROBE_ALIAS = "_effectful_type_probe"


def character_column(lines: list[str], line: int, byte_column: int) -> int:
    """The 0-based character column of `byte_column` (as ``ast`` counts, in UTF-8
    bytes) on the 1-based `line` of `lines`."""
    return len(lines[line - 1].encode()[:byte_column].decode())


def candidates(
    tree: ast.Module, source: str, line: int, column: int, *, argument: bool
) -> list[ast.Call]:
    """The calls an error at `line` and 0-based character `column` may be about.

    `argument` says the checker placed the error at an argument of the call, as
    both do for a wrongly typed argument, rather than at the call itself. The
    innermost call around the position is not the owner: a wrong nested call
    ``op(other(x))`` is reported at ``other(x)``, which belongs to ``op``. A
    call-level position does not pick one call either: ``op(1).upper()`` and
    ``make().op(1)`` each start two calls there. The checker settles it by which
    candidate's callee it types as an ``Operation``.
    """
    lines = source.splitlines()

    def starts_at(node: ast.expr | ast.keyword) -> bool:
        # A keyword argument also starts at its value, where mypy reports a wrongly
        # typed one.
        starts: list[ast.expr | ast.keyword] = (
            [node, node.value] if isinstance(node, ast.keyword) else [node]
        )
        return any(
            start.lineno == line
            and character_column(lines, start.lineno, start.col_offset) == column
            for start in starts
        )

    def arguments(call: ast.Call) -> list[ast.expr | ast.keyword]:
        return [*call.args, *call.keywords]

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    if argument:
        return [
            call for call in calls if any(starts_at(arg) for arg in arguments(call))
        ]
    return [call for call in calls if starts_at(call)]


def probed(
    source: str, tree: ast.Module, calls: list[ast.Call]
) -> tuple[str, dict[ast.expr, tuple[int, int]]]:
    """`source` with each call's callee ``f`` replaced by ``<alias>.reveal_type(f)``,
    and where each callee now starts, keyed by the callee's node: nested callees
    (``op`` inside ``op(1).upper``) start at the same position.

    `reveal_type` returns its argument, so the copy binds and evaluates exactly as
    the original, including names bound by a comprehension or a parameter. Callees
    may nest (``make`` inside ``make().op``); the outer probe then encloses the
    inner one. The ``import typing as <alias>`` line goes after any docstring and
    ``__future__`` imports, which must stay first. Positions are where each probe's
    argument starts in the copy, as both checkers report a revealed type: 1-based
    lines and 0-based character columns.
    """
    alias = _PROBE_ALIAS
    while alias in source:
        alias += "_"

    def extent(func: ast.expr) -> tuple[int, int, int, int]:
        return (
            func.lineno,
            func.col_offset,
            func.end_lineno or func.lineno,
            func.end_col_offset or func.col_offset,
        )

    # Outer before inner: by start, then the larger extent first.
    funcs = sorted(
        {extent(call.func): call.func for call in calls}.items(),
        key=lambda item: (item[0][0], item[0][1], -item[0][2], -item[0][3]),
    )
    probe = f"{alias}.reveal_type(".encode()
    # (line, byte column, order, text): at one column, a `)` closing an earlier
    # expression comes first, then the probes from outer to inner.
    insertions: list[tuple[int, int, int, bytes]] = []
    for rank, ((line, col, end_line, end_col), _) in enumerate(funcs):
        insertions.append((line, col, 1 + rank, probe))
        insertions.append((end_line, end_col, 0, b")"))
    lines = [line.encode() for line in source.splitlines(keepends=True)]
    # Inserting right to left keeps every earlier column valid; at one column the
    # last insertion ends up leftmost.
    for line, col, _, inserted in sorted(insertions, reverse=True):
        lines[line - 1] = lines[line - 1][:col] + inserted + lines[line - 1][col:]
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
    copy = [line.decode() for line in lines]
    copy.insert(header, f"import typing as {alias}\n")
    text = "".join(copy)
    # The probes, in the copy's order, match `funcs` outer before inner.
    copy_lines = text.splitlines()
    arguments = sorted(
        (
            node.args[0].lineno,
            character_column(copy_lines, node.args[0].lineno, node.args[0].col_offset),
        )
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reveal_type"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == alias
    )
    positions = {
        func: position for (_, func), position in zip(funcs, arguments, strict=True)
    }
    return text, positions


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
