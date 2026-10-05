"""Type checking of generated code by shelling out to mypy.

`MypyTypeChecker` is independent of any executor: it says how generated code is
*checked*, not how it is parsed, compiled or run, so it is installed alongside
whichever of those handlers a stack uses::

    handler(MypyTypeChecker()), handler(BuiltinExecutor())

rather than being part of one.

It is interchangeable with
`~effectful.handlers.llm.harness.validation.ty.TyTypeChecker`, which implements
the same operation with the same contract and is substantially faster; that
module's docstring compares the two.
"""

import ast
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import typing

# trigger mypy installation errors early
import mypy.api  # noqa: F401

from effectful.handlers.llm.harness.hooks import PromptInjectingInterpretation
from effectful.handlers.llm.harness.validation import operations
from effectful.handlers.llm.harness.validation.hooks import type_check
from effectful.ops.syntax import implements


@dataclasses.dataclass
class MypyTypeChecker(PromptInjectingInterpretation):
    """Python you write is type-checked before it is run, by mypy. Code that
    fails the check does not execute at all: you get mypy's diagnostics back --
    the message, the error code, the offending line -- and the turn is yours
    again to fix them.

    Treat that as a fast, free reviewer rather than an obstacle. Annotate what
    you write, use the types the surrounding code declares, and read a
    diagnostic as a claim about your code that is usually correct. Silencing one
    with `typing.Any` or a blanket `# type: ignore` will pass the check and then
    fail at runtime, where the error costs a whole turn instead of none.

    Only the code you generate is checked; errors elsewhere in the module you
    are working in are not yours to fix and will not block you.
    """

    #: Flags added under ``lenient=True`` to waive the diagnostics that a REPL
    #: transcript or a spliced function body provokes by construction -- see
    #: `type_check` for what each one is for.
    lenient_flags: tuple[str, ...] = (
        "--allow-redefinition-new",
        "--local-partial-types",
        "--disable-error-code=no-redef",
        "--disable-error-code=return",
        "--disable-error-code=empty-body",
    )

    # Strict and lenient checks use cache-affecting options, so each gets its own
    # incremental cache for this handler's lifetime. `TemporaryDirectory` removes
    # each cache when the handler is discarded.
    _strict_cache: tempfile.TemporaryDirectory = dataclasses.field(
        default_factory=lambda: tempfile.TemporaryDirectory(
            prefix="effectful_mypy_strict_cache_"
        ),
        init=False,
        repr=False,
        compare=False,
    )
    _lenient_cache: tempfile.TemporaryDirectory = dataclasses.field(
        default_factory=lambda: tempfile.TemporaryDirectory(
            prefix="effectful_mypy_lenient_cache_"
        ),
        init=False,
        repr=False,
        compare=False,
    )

    # A handler may be shared by concurrent Skill calls. Independent mypy processes
    # must not write the same incremental cache at once, so serialize checks that
    # share a mode. Strict and lenient calls use different caches and can still run
    # concurrently. The synthesized source gets its own directory below as well.
    _strict_lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )
    _lenient_lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    @staticmethod
    def _region_errors(
        stdout: str, lo: int | None, hi: int | None
    ) -> list[dict[str, typing.Any]]:
        """mypy ``--output=json`` diagnostics of severity ``error`` whose reported
        line falls within ``[lo, hi]`` -- the spliced region. An open bound (``None``)
        is unbounded on that side, so ``lo=hi=None`` reports every error.

        ``--output=json`` emits one JSON object per diagnostic carrying mypy's own
        ``severity`` and ``line`` fields, so we filter on those directly rather than
        parsing (and risking mis-parsing) its human-readable format.  Only reached
        for exit status < 2; a fatal status emits text, not JSON, and is handled by
        the caller before this runs.
        """
        errors: list[dict[str, typing.Any]] = []
        for line in stdout.splitlines():
            if not line.strip():
                continue
            diag = json.loads(line)
            if diag["severity"] != "error":
                continue
            if (lo is None or lo <= diag["line"]) and (
                hi is None or diag["line"] <= hi
            ):
                errors.append(diag)
        return errors

    @implements(type_check)
    def type_check(
        self,
        source: str,
        lo: int | None = None,
        hi: int | None = None,
        *,
        lenient: bool = False,
    ) -> None:
        """Run mypy on `source` and raise ``TypeError`` if any error diagnostic falls
        within ``[lo, hi]``; raise ``RuntimeError`` if mypy itself fails to run.

        Applies mypy to whatever source it's given -- spliced or otherwise -- and
        reports only the region's errors (the whole source when the region is
        omitted), so pre-existing errors elsewhere in `source` never block synthesis.

        When ``lenient`` (for REPL code spliced into a Skill body): allow a variable to be
        redefined with a new type across cells (``--allow-redefinition-new``, which supersedes
        the narrower ``--allow-redefinition`` and requires ``--local-partial-types``), a
        def/class/import to be redefined (``no-redef``), and the body not to return the
        Skill's declared type (``return``/``empty-body``). All normal for an
        incrementally-built REPL, not real errors.
        """
        stdout = self._run(source, lenient)
        errors = self._region_errors(stdout, lo, hi)
        if errors:
            # Not the source: it's large and the model already has the generated code.
            report = "\n".join(
                json.dumps(e) for e in self._name_operations(source, errors, lenient)
            )
            raise TypeError("mypy type check failed:\n" + report)

    def _run(self, source: str, lenient: bool) -> str:
        """Run mypy on `source` with the cache for its mode; return its JSON output.

        Exit status >= 2 means mypy itself failed (fatal/usage/internal/syntax) -- a
        tool failure, not a type error -- and it emits text rather than JSON, so
        raise `RuntimeError` rather than parse or silently pass.
        """
        tmpdir = tempfile.mkdtemp(prefix="effectful_typecheck_")
        try:
            tf_path = os.path.join(tmpdir, "_synthesized.py")
            with open(tf_path, "w", encoding="utf-8") as f:
                f.write(source)
            # Keep the source path unique per call even though the cache persists.
            # Mypy can accept a same-path, same-size source from its mtime fast path;
            # changing the path makes it validate the source hash before reuse.
            cache = self._lenient_cache if lenient else self._strict_cache
            lock = self._lenient_lock if lenient else self._strict_lock
            with lock:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "mypy",
                        tf_path,
                        "--cache-dir",
                        cache.name,
                        "--no-error-summary",
                        "--output=json",
                        "--ignore-missing-imports",
                        "--disable-error-code=import-untyped",
                        # Imported modules are typed but their own diagnostics
                        # stay out: they are not the model's to fix, and their
                        # line numbers would be read as the region's.
                        "--follow-imports=silent",
                        *(self.lenient_flags if lenient else []),
                    ],
                    capture_output=True,
                    text=True,
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        if proc.returncode >= 2:
            raise RuntimeError(
                f"mypy could not check the source:\n{proc.stdout}{proc.stderr}"
            )
        return proc.stdout

    def _name_operations(
        self, source: str, errors: list[dict[str, typing.Any]], lenient: bool
    ) -> list[dict[str, typing.Any]]:
        """`errors` with each diagnostic about an operation call rewritten to name the
        operation, with its signature as the hint.

        mypy places ``arg-type`` at the argument, and ``call-arg`` (missing,
        unexpected or too many arguments) and ``misc`` (a parameter given twice) at
        the call. One more mypy run reveals each candidate callee's type, probed in
        place (see `operations.probed`); an error none of whose candidates mypy
        types as an ``Operation`` keeps mypy's wording. The source parses: mypy
        exits with status 2 on a syntax error, which `_run` raises.
        """
        if not any(_OPERATION_CALL in error["message"] for error in errors):
            return errors
        tree = ast.parse(source)
        # mypy's columns match character columns on ASCII lines only; past non-ASCII
        # text they count neither characters nor bytes, so no candidate is found
        # there and the error keeps mypy's wording.
        candidates = [
            operations.candidates(
                tree,
                source,
                error["line"],
                error["column"],
                argument=error["code"] == "arg-type",
            )
            if _OPERATION_CALL in error["message"]
            and error["code"] in {"arg-type", "call-arg", "misc"}
            else []
            for error in errors
        ]
        calls = [call for found in candidates for call in found]
        if not calls:
            return errors
        copy, positions = operations.probed(source, tree, calls)
        stdout = self._run(copy, lenient)
        # A revealed type is its own note, or, when an error sits at the same
        # position (a probe around a wrong call), folded into that error's hint.
        reports = {
            (diagnostic["line"], diagnostic["column"]): match["type"]
            for diagnostic in (json.loads(line) for line in stdout.splitlines() if line)
            for text in (
                diagnostic["message"]
                if diagnostic["severity"] == "note"
                else diagnostic["hint"] or ""
            ).splitlines()
            if (match := _REVEALED.fullmatch(text))
        }
        # A probe without a report means the probe is wrong, not the code.
        missing = [p for p in positions.values() if p not in reports]
        if missing:
            raise RuntimeError(f"mypy reported no type for the probes at {missing}")
        renamed = []
        for error, found in zip(errors, candidates, strict=True):
            operation = next(
                (
                    (call, match)
                    for call in found
                    if (
                        match := _OPERATION_TYPE.fullmatch(
                            reports[positions[call.func]]
                        )
                    )
                ),
                None,
            )
            if operation is None:
                renamed.append(error)
                continue
            call, match = operation
            renamed.append(
                self._rename(
                    error,
                    operations.callee(source, call),
                    match["params"],
                    match["result"],
                )
            )
        return renamed

    @staticmethod
    def _rename(
        error: dict[str, typing.Any], callee: str, params: str, result: str
    ) -> dict[str, typing.Any]:
        """`error` naming operation `callee`, with its signature
        ``callee(params) -> result`` as the hint."""
        # mypy prints a parameter's default as a bare trailing ``=``.
        params = re.sub(r" =(?=,|$)", " = ...", params)
        if not operations.missing_parameters_declared(error["message"], params):
            return error
        return {
            **error,
            "message": error["message"].replace(
                _OPERATION_CALL, f'operation "{callee}"'
            ),
            "hint": f'"{callee}" is called as {callee}({params}) -> {result}',
        }


# mypy names a wrong operation call by the method it dispatches through.
_OPERATION_CALL = '"__call__" of "Operation"'
_REVEALED = re.compile(r'Revealed type is "(?P<type>.*)"')
_OPERATION_TYPE = re.compile(
    r"(?:\w+\.)*Operation\[\[(?P<params>.*)\], (?P<result>.*)\]"
)
