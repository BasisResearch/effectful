"""Type checking of generated code by shelling out to ty.

`TyTypeChecker` is interchangeable with
`~effectful.handlers.llm.harness.validation.mypy.MypyTypeChecker` -- same
operation, same contract, a different checker behind it. ty is a compiled binary
that needs no per-call cache and builds no module graph in this process, so a
check costs milliseconds where mypy's costs seconds, and on the failure path it
reports the offending line with ty's own hints rather than a line of JSON. Prefer
it unless a stack specifically needs mypy's analysis.

Either checker is independent of any executor: it says how generated code is
*checked*, not how it is parsed, compiled or run, so it is installed alongside
whichever of those handlers a stack uses::

    handler(TyTypeChecker()), handler(BuiltinExecutor())

rather than being part of one.
"""

import ast
import dataclasses
import functools
import os
import re
import shutil
import subprocess
import sys
import tempfile

import ty

from effectful.handlers.llm.harness.hooks import PromptInjectingInterpretation
from effectful.handlers.llm.harness.validation.hooks import type_check
from effectful.ops.syntax import implements


@dataclasses.dataclass
class TyTypeChecker(PromptInjectingInterpretation):
    """Python you write is type-checked before it is run, by the ty type
    checker. Code that fails the check does not execute at all: you get ty's
    diagnostics back -- the message, the offending line, its hints -- and the
    turn is yours again to fix them.

    Treat that as a fast, free reviewer rather than an obstacle. Annotate what
    you write, use the types the surrounding code declares, and read a
    diagnostic as a claim about your code that is usually correct. Silencing
    one with `typing.Any` or a blanket `# type: ignore` will pass the check and
    then fail at runtime, where the error costs a whole turn instead of none.

    Only the code you generate is checked; errors elsewhere in the module you
    are working in are not yours to fix and will not block you.
    """

    #: Rules ignored under ``lenient=True``. Deliberately short: ty already
    #: grants most of that leniency unasked -- see `type_check`.
    lenient_ignored_rules: tuple[str, ...] = ("conflicting-declarations",)

    @functools.cached_property
    def _header(self) -> re.Pattern[str]:
        """The line that opens a diagnostic: ``severity[rule]: message``, at column
        zero. ty has no JSON output, so its default rendering is what gets parsed."""
        return re.compile(r"^(?P<severity>error|warning)\[(?P<rule>[\w-]+)\]:")

    @functools.cached_property
    def _location(self) -> re.Pattern[str]:
        """A location marker within a diagnostic: ``   --> path:line:col``. Matched on
        the trailing ``:line:col`` rather than the leading path, which ty prints
        relative to its working directory and which may itself contain colons."""
        return re.compile(r"^\s*--> (?P<path>.*?):(?P<line>\d+):(?P<col>\d+)$")

    @functools.cached_property
    def _summary(self) -> re.Pattern[str]:
        """ty's closing tally, which it always prints and offers no flag to suppress
        (``--quiet`` drops the diagnostics and keeps this). Belongs to no diagnostic."""
        return re.compile(r"^(Found \d+ diagnostic|All checks passed)")

    @staticmethod
    def _in_region(line: int | None, lo: int | None, hi: int | None) -> bool:
        """Whether a diagnostic reported at `line` falls inside ``[lo, hi]`` -- the
        spliced region. An open bound (``None``) is unbounded on that side, so
        ``lo=hi=None`` accepts every line; a diagnostic carrying no line at all can't
        be attributed to the region and is rejected.
        """
        return (
            line is not None
            and (lo is None or lo <= line)
            and (hi is None or line <= hi)
        )

    def _diagnostics(self, stdout: str) -> list[tuple[str, int | None, str]]:
        """ty's diagnostics as ``(severity, line, rendered)``, in reported order.

        A diagnostic runs from one header to the next, and its ``line`` is the *first*
        ``-->`` marker it carries -- the primary location -- since a diagnostic may
        carry further markers for secondary annotations (``info: Method defined here``)
        pointing elsewhere in the file. ``None`` when it carries no marker at all.

        `rendered` is ty's own text for the diagnostic, kept verbatim so the report
        raised below carries the source excerpt, carets and ``info:`` notes that make
        it worth handing back to a model.
        """
        diagnostics: list[tuple[str, int | None, str]] = []
        current: list[str] = []
        severity: str | None = None
        line: int | None = None

        def flush() -> None:
            if current and severity is not None:
                diagnostics.append((severity, line, "\n".join(current).rstrip()))

        for text in stdout.splitlines():
            header = self._header.match(text)
            if header is not None:
                flush()
                current, severity, line = [text], header["severity"], None
                continue
            if not current or self._summary.match(text):
                continue
            current.append(text)
            if line is None:
                location = self._location.match(text)
                if location is not None:
                    line = int(location["line"])
        flush()
        return diagnostics

    @implements(type_check)
    def type_check(
        self,
        source: str,
        lo: int | None = None,
        hi: int | None = None,
        *,
        lenient: bool = False,
    ) -> None:
        """Run ty on `source` and raise ``TypeError`` if any error diagnostic falls
        within ``[lo, hi]``; raise ``RuntimeError`` if ty itself fails to run.

        Applies ty to whatever source it's given -- spliced or otherwise -- and
        reports only the region's errors (the whole source when the region is
        omitted), so pre-existing errors elsewhere in `source` never block synthesis.

        ``lenient`` disables far less here than under `MypyTypeChecker`, because ty
        grants most of that leniency unasked. A variable may be redefined with a new
        type across cells and ty narrows to the latest binding, and a def/class/import
        may be redefined -- no flag needed for either. A body that doesn't return the
        Skill's declared type is reported against the *signature* line, while a
        body that returns the wrong type is reported against the ``return`` statement,
        so the region filter tells those two apart on its own; splitting them by
        position rather than by flag is what keeps ``lenient`` from also waiving a
        genuine wrong-return-type error. That leaves `no-redef`'s counterpart, kept for
        faithfulness to ty's own mapping though it fires on none of the redefinition
        shapes a REPL produces.
        """
        stdout, stderr, status = self._run(
            source,
            "full",
            *(
                arg
                for rule in (self.lenient_ignored_rules if lenient else ())
                for arg in ("--ignore", rule)
            ),
        )
        # Exit status >= 2 means ty itself failed (2: usage/config/IO, 101: internal
        # panic) -- a tool failure, not a type error -- so raise `RuntimeError` rather
        # than read a verdict out of output it never produced. Status 1 is the ordinary
        # "found diagnostics" case, filtered below; 0 is clean.
        if status >= 2:
            raise RuntimeError(f"ty could not check the source:\n{stdout}{stderr}")
        diagnostics = self._diagnostics(stdout)
        # ty says it found something but none of it parsed: its rendering has moved
        # under us. Say so, rather than read the silence as an empty region and let
        # ill-typed code through.
        if status == 1 and not diagnostics:
            raise RuntimeError(f"ty reported unparseable diagnostics:\n{stdout}")
        errors = [
            rendered
            for severity, line, rendered in diagnostics
            if severity == "error" and self._in_region(line, lo, hi)
        ]
        if errors:
            # Not the source: it's large and the model already has the generated code.
            raise TypeError(
                "ty type check failed:\n"
                + "\n\n".join(self._name_operations(source, errors))
            )

    def _run(
        self, source: str, output_format: str, *extra: str
    ) -> tuple[str, str, int]:
        """Run ty on `source` in an isolated temp project; return its stdout, stderr
        and exit status."""
        tmpdir = tempfile.mkdtemp(prefix="effectful_typecheck_")
        # Read before the subprocess is handed `cwd=tmpdir`, which is set only so ty
        # cites the temp file by bare name in the report.
        cwd = os.getcwd()
        try:
            tf_path = os.path.join(tmpdir, "_synthesized.py")
            with open(tf_path, "w", encoding="utf-8") as f:
                f.write(source)
            # Pass a file, not the source: ty has no `--command`. Each call gets an
            # isolated temp dir, which doubles as the project root so no stray
            # `ty.toml`/`[tool.ty]` near the caller can change the verdict. (Unlike
            # mypy, ty needs no cache dir: it has no on-disk cache to isolate.)
            proc = subprocess.run(
                [
                    ty.find_ty_bin(),
                    "check",
                    os.path.basename(tf_path),
                    "--project",
                    tmpdir,
                    # Third-party imports resolve out of the environment this process
                    # runs in, as `sys.executable -m mypy` implicitly does; first-party
                    # ones out of its working directory, where mypy also finds them --
                    # the source being checked is a Skill's own module, so those are
                    # exactly the imports the check is *for*, and an editable install is
                    # not reachable through site-packages alone. Deliberately *not*
                    # `--extra-search-path` over all of `sys.path`: handing ty the
                    # stdlib directory makes it read CPython's sources instead of its
                    # vendored typeshed and panic, and entries that aren't directories
                    # (zips, editable-install path hooks) are a usage error.
                    "--python",
                    sys.prefix,
                    "--extra-search-path",
                    cwd,
                    "--color",
                    "never",
                    "--output-format",
                    output_format,
                    # Matches mypy's `--ignore-missing-imports`: an unresolved module
                    # becomes `Unknown` and stays gradual, as mypy's becomes `Any`.
                    "--ignore",
                    "unresolved-import",
                    *extra,
                ],
                capture_output=True,
                text=True,
                cwd=tmpdir,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        return proc.stdout, proc.stderr, proc.returncode

    def _name_operations(self, source: str, errors: list[str]) -> list[str]:
        """`errors` with each diagnostic about an operation call rewritten to name the
        operation and show its signature.

        ty reports a wrong call to an operation against ``Operation.__call__``, whose
        parameters are the generic ``*args: Q.args, **kwargs: Q.kwargs``, and points
        into this package's source. The signature the caller needed is the operation's
        own, which ty knows: one more run reveals the type of each such call's callee,
        probed in place so every name in scope resolves as it did in the check. A
        callee ty does not type as an ``Operation`` keeps ty's diagnostic as written.
        """
        if not any(_OPERATION_CALL in rendered for rendered in errors):
            return errors
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # ty recovers from a syntax error outside the checked region and still
            # reports the region's errors; without a tree there is no call to name,
            # so its wording stands.
            return errors
        owners = [self._owner(tree, source, rendered) for rendered in errors]
        calls = sorted(
            {(call.func.lineno, call.func.col_offset): call for call in owners if call}
        )
        if not calls:
            return errors
        by_position = dict(
            zip(
                calls,
                self._revealed(source, tree, [owner for owner in owners if owner]),
            )
        )
        return [
            rendered
            if call is None
            else self._rename(
                rendered,
                ast.get_source_segment(source, call.func) or ast.unparse(call.func),
                by_position[(call.func.lineno, call.func.col_offset)],
            )
            for rendered, call in zip(errors, owners)
        ]

    def _owner(self, tree: ast.Module, source: str, rendered: str) -> ast.Call | None:
        """The call a diagnostic about ``Operation.__call__`` is about, or ``None``.

        ty places argument errors (a wrong type, an unknown keyword, one positional
        too many) at the offending argument, and a missing argument at the call. The
        innermost call around the position is therefore not the owner: a wrong nested
        call ``op(other(x))`` is reported at ``other(x)``, which belongs to ``op``.
        """
        header = self._header.match(rendered)
        location = next(
            (
                match
                for text in rendered.splitlines()
                if (match := self._location.match(text))
            ),
            None,
        )
        if header is None or location is None or _OPERATION_CALL not in rendered:
            return None
        line, col = int(location["line"]), int(location["col"])
        lines = source.splitlines()

        def starts_at(node: ast.expr | ast.keyword) -> bool:
            # ast offsets count UTF-8 bytes; ty's columns count characters.
            text = lines[node.lineno - 1].encode()[: node.col_offset].decode()
            return (node.lineno, len(text) + 1) == (line, col)

        def arguments(call: ast.Call) -> list[ast.expr | ast.keyword]:
            return [*call.args, *call.keywords]

        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        if header["rule"] in _ARGUMENT_RULES:
            return next(
                (
                    call
                    for call in calls
                    for argument in arguments(call)
                    if starts_at(argument)
                ),
                None,
            )
        if header["rule"] in _CALL_RULES:
            starting = [call for call in calls if starts_at(call)]
            return max(
                starting,
                key=lambda call: (call.end_lineno or 0, call.end_col_offset or 0),
                default=None,
            )
        return None

    def _revealed(
        self, source: str, tree: ast.Module, calls: list[ast.Call]
    ) -> list[str]:
        """ty's type for each call's callee, in source order, from one run over a copy
        of `source` in which each callee ``f`` becomes ``reveal_type(f)``.

        `reveal_type` returns its argument, so the copy binds and evaluates exactly as
        the original. It is reached through a module alias the copy imports, named so
        it appears nowhere in `source` and no binding there can intercept the probe;
        the import follows any docstring and ``__future__`` imports, which must stay
        first. The splices are applied from the end of the source backwards so earlier
        positions stay valid, and each report is matched to its probe by position.
        """
        unique = {(c.func.lineno, c.func.col_offset): c.func for c in calls}
        alias = _PROBE_ALIAS
        while alias in source:
            alias += "_"
        probe = f"{alias}.reveal_type(".encode()
        # (line, byte column, text) of every insertion, in original coordinates.
        insertions = [
            insertion
            for func in unique.values()
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
                isinstance(statement, ast.ImportFrom)
                and statement.module == "__future__"
            )
            if not (docstring or future):
                break
            header = statement.end_lineno or statement.lineno

        def probe_position(func: ast.expr) -> tuple[int, int]:
            # Where ty reports the probe: the callee's start inside its probe call,
            # shifted by every insertion before it on its line and by the alias
            # import's line, as a 1-based character column.
            shift = sum(
                len(text)
                for line, col, text in insertions
                if line == func.lineno and col <= func.col_offset
            )
            prefix = lines[func.lineno - 1][: func.col_offset + shift]
            return func.lineno + 1, len(prefix.decode()) + 1

        probed = [line.decode() for line in lines]
        probed.insert(header, f"import typing as {alias}\n")
        stdout, stderr, status = self._run("".join(probed), "concise")
        if status >= 2:
            raise RuntimeError(
                f"ty could not check the probed source:\n{stdout}{stderr}"
            )
        reports = {
            (int(match["line"]), int(match["col"])): match["type"]
            for match in _REVEALED.finditer(stdout)
        }
        revealed = []
        for position in sorted(unique):
            expected = probe_position(unique[position])
            # Reports at other positions come from the module's own reveal_type
            # calls; a probe without a report means the splice is wrong, not the
            # code, and guessing its signature would mislead.
            if expected not in reports:
                raise RuntimeError(f"ty reported no type for the probe at {expected}")
            revealed.append(reports[expected])
        return revealed

    def _rename(self, rendered: str, callee: str, revealed: str) -> str:
        """`rendered` naming operation `callee` with its signature, if `revealed` is
        an ``Operation`` type; otherwise `rendered` unchanged."""
        operation = _OPERATION_TYPE.fullmatch(revealed)
        if operation is None:
            return rendered
        params, result = operation["params"], operation["result"]
        missing = _MISSING_PARAMETERS.search(rendered)
        # A missing parameter the operation does not declare means the owner is wrong.
        if missing and not all(
            re.search(rf"\b{re.escape(name)}\b", params)
            for name in re.findall(r"`([^`]+)`", missing["names"])
        ):
            return rendered
        header, *body = rendered.splitlines()
        header = header.replace(_OPERATION_CALL, f"operation `{callee}`")
        # ty counts the bound `self` of `Operation.__call__` among the positionals.
        header = _POSITIONAL_COUNT.sub(
            lambda m: f"expected {int(m['expected']) - 1}, got {int(m['got']) - 1}",
            header,
        )
        kept: list[str] = []
        skipping = False
        for index, text in enumerate(body):
            if text.startswith(("info:", "help:", "note:")):
                following = body[index + 1] if index + 1 < len(body) else ""
                location = self._location.match(following)
                skipping = location is not None and location["path"].endswith(
                    _OPERATIONS_MODULE
                )
            if not skipping:
                kept.append(text)
        kept.append(f"info: `{callee}` is called as {callee}({params}) -> {result}")
        return "\n".join([header, *kept])


# ty names a wrong operation call by the method it dispatches through.
_OPERATION_CALL = "bound method `Operation.__call__`"
_OPERATIONS_MODULE = os.path.join("effectful", "ops", "types.py")
# The module alias the probe reaches `typing.reveal_type` through.
_PROBE_ALIAS = "_effectful_ty_probe"
_ARGUMENT_RULES = frozenset(
    {"invalid-argument-type", "unknown-argument", "too-many-positional-arguments"}
)
_CALL_RULES = frozenset({"missing-argument"})
_REVEALED = re.compile(
    r"^\S+?:(?P<line>\d+):(?P<col>\d+): info\[revealed-type\] "
    r"Revealed type: `(?P<type>.*)`$",
    re.M,
)
_OPERATION_TYPE = re.compile(r"Operation\[\((?P<params>.*)\), (?P<result>.*)\]")
_MISSING_PARAMETERS = re.compile(r"required parameters? (?P<names>(`[^`]+`(, )?)+)")
_POSITIONAL_COUNT = re.compile(r"expected (?P<expected>\d+), got (?P<got>\d+)")
