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
import subprocess
import sys
import tempfile

import ty

from effectful.handlers.llm.harness.hooks import PromptInjectingInterpretation
from effectful.handlers.llm.harness.validation import operations
from effectful.handlers.llm.harness.validation.hooks import type_check
from effectful.ops.syntax import implements


@dataclasses.dataclass(frozen=True)
class _Diagnostic:
    """One diagnostic in ty's full output: its header fields, its primary location
    (``None`` when it carries no ``-->`` marker), and ty's own text for it.

    `rendered` is kept verbatim so the report carries the source excerpt, carets and
    ``info:`` notes that make it worth handing back to a model.
    """

    severity: str
    rule: str
    line: int | None
    column: int | None
    rendered: str


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

    def _diagnostics(self, stdout: str) -> list[_Diagnostic]:
        """ty's diagnostics, in reported order.

        A diagnostic runs from one header to the next, and its location is the *first*
        ``-->`` marker it carries -- the primary location -- since a diagnostic may
        carry further markers for secondary annotations (``info: Method defined here``)
        pointing elsewhere in the file.
        """
        diagnostics: list[_Diagnostic] = []
        current: list[str] = []
        header: re.Match[str] | None = None
        location: re.Match[str] | None = None

        def flush() -> None:
            if current and header is not None:
                diagnostics.append(
                    _Diagnostic(
                        severity=header["severity"],
                        rule=header["rule"],
                        line=int(location["line"]) if location else None,
                        column=int(location["col"]) - 1 if location else None,
                        rendered="\n".join(current).rstrip(),
                    )
                )

        for text in stdout.splitlines():
            if (match := self._header.match(text)) is not None:
                flush()
                current, header, location = [text], match, None
                continue
            if not current or self._summary.match(text):
                continue
            current.append(text)
            if location is None:
                location = self._location.match(text)
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
        stdout, status = self._run(
            source,
            "full",
            *(
                arg
                for rule in (self.lenient_ignored_rules if lenient else ())
                for arg in ("--ignore", rule)
            ),
        )
        diagnostics = self._diagnostics(stdout)
        # ty says it found something but none of it parsed: its rendering has moved
        # under us. Say so, rather than read the silence as an empty region and let
        # ill-typed code through.
        if status == 1 and not diagnostics:
            raise RuntimeError(f"ty reported unparseable diagnostics:\n{stdout}")
        errors = [
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.severity == "error"
            and self._in_region(diagnostic.line, lo, hi)
        ]
        if errors:
            # Not the source: it's large and the model already has the generated code.
            raise TypeError(
                "ty type check failed:\n"
                + "\n\n".join(self._name_operations(source, errors))
            )

    def _run(self, source: str, output_format: str, *extra: str) -> tuple[str, int]:
        """Run ty on `source` in an isolated temp project; return its stdout and exit
        status, which is 0 when clean and 1 when it found diagnostics.

        Exit status >= 2 means ty itself failed (2: usage/config/IO, 101: internal
        panic) -- a tool failure, not a type error -- so raise `RuntimeError` rather
        than read a verdict out of output it never produced.
        """
        # Read before the subprocess is handed `cwd=tmpdir`, which is set only so ty
        # cites the temp file by bare name in the report.
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory(
            prefix="effectful_typecheck_", ignore_cleanup_errors=True
        ) as tmpdir:
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
        if proc.returncode >= 2:
            raise RuntimeError(
                f"ty could not check the source:\n{proc.stdout}{proc.stderr}"
            )
        return proc.stdout, proc.returncode

    def _name_operations(self, source: str, errors: list[_Diagnostic]) -> list[str]:
        """`errors` with each diagnostic about an operation call rewritten to name the
        operation and show its signature.

        One more ty run reveals the type of each such call's callee, probed in place
        (see `operations.probed`). A callee ty does not type as an ``Operation``, or
        a module that does not parse, keeps ty's wording.
        """
        rendered = [error.rendered for error in errors]
        if not any(_OPERATION_CALL in text for text in rendered):
            return rendered
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # ty recovers from a syntax error outside the checked region and still
            # reports the region's errors; without a tree there is no call to name.
            return rendered
        candidates = [self._candidates(tree, source, error) for error in errors]
        calls = [call for found in candidates for call in found]
        if not calls:
            return rendered
        revealed = self._revealed(source, tree, calls)
        renamed = []
        for error, found in zip(errors, candidates, strict=True):
            operation = next(
                (
                    (call, signature)
                    for call in found
                    if (signature := operations.signature(revealed[call.func], "("))
                    # Of `make()(1)` and `make()`, both operations, the one whose
                    # signature declares the missing parameters owns the error.
                    and operations.missing_parameters_declared(
                        error.rendered, signature[0]
                    )
                ),
                None,
            )
            if operation is None:
                renamed.append(error.rendered)
                continue
            call, (params, result) = operation
            renamed.append(
                self._rename(
                    error.rendered, operations.callee(source, call), params, result
                )
            )
        return renamed

    def _candidates(
        self, tree: ast.Module, source: str, error: _Diagnostic
    ) -> list[ast.Call]:
        """The calls a diagnostic about ``Operation.__call__`` may be about.

        ty places argument errors (a wrong type, an unknown keyword, one positional
        too many, a parameter given twice) at the offending argument, and a missing
        argument at the call. Errors in the checked region are always located.
        """
        if (
            _OPERATION_CALL not in error.rendered
            or error.rule not in _ARGUMENT_RULES | _CALL_RULES
            or error.line is None
            or error.column is None
        ):
            return []
        return operations.candidates(
            tree,
            source,
            error.line,
            error.column,
            argument=error.rule in _ARGUMENT_RULES,
        )

    def _revealed(
        self, source: str, tree: ast.Module, calls: list[ast.Call]
    ) -> dict[ast.expr, str]:
        """ty's revealed type for each call's callee, keyed by the callee's node.

        Reports at other positions come from the module's own ``reveal_type`` calls;
        a probe without a report means the probe is wrong, not the code, and a
        guessed signature would mislead.
        """
        copy, positions = operations.probed(source, tree, calls)
        stdout, _ = self._run(copy, "concise")
        reports = {
            (int(match["line"]), int(match["col"]) - 1): match["type"]
            for match in _REVEALED.finditer(stdout)
        }
        revealed = {}
        for func, position in positions.items():
            if position not in reports:
                raise RuntimeError(f"ty reported no type for the probe at {position}")
            revealed[func] = reports[position]
        return revealed

    def _rename(self, rendered: str, callee: str, params: str, result: str) -> str:
        """`rendered` naming operation `callee` and showing its signature
        ``callee(params) -> result``."""
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
_ARGUMENT_RULES = frozenset(
    {
        "invalid-argument-type",
        "unknown-argument",
        "too-many-positional-arguments",
        "parameter-already-assigned",
    }
)
_CALL_RULES = frozenset({"missing-argument"})
_REVEALED = re.compile(
    r"^\S+?:(?P<line>\d+):(?P<col>\d+): info\[revealed-type\] "
    r"Revealed type: `(?P<type>.*)`$",
    re.M,
)
_POSITIONAL_COUNT = re.compile(r"expected (?P<expected>\d+), got (?P<got>\d+)")
