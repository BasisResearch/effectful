import json
import os
import shutil
import subprocess
import sys
import tempfile
import typing

# trigger mypy installation errors early
import mypy.api  # noqa: F401

from effectful.handlers.llm.harness.validation.hooks import type_check
from effectful.ops.syntax import ObjectInterpretation, implements


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
        if (lo is None or lo <= diag["line"]) and (hi is None or diag["line"] <= hi):
            errors.append(diag)
    return errors


def _mypy_check_region(
    source: str,
    lo: int | None = None,
    hi: int | None = None,
    lenient: bool = False,
) -> None:
    """Run mypy on `source` and raise ``TypeError`` if any error diagnostic falls
    within ``[lo, hi]``; raise ``RuntimeError`` if mypy itself fails to run.

    Applies mypy to whatever source it's given -- spliced or otherwise -- and
    reports only the region's errors (the whole source when the region is
    omitted), so pre-existing errors elsewhere in `source` never block synthesis.

    When ``lenient`` (for REPL code spliced into a Template body): allow a variable to be
    redefined with a new type across cells (``--allow-redefinition-new``, which supersedes
    the narrower ``--allow-redefinition`` and requires ``--local-partial-types``), a
    def/class/import to be redefined (``no-redef``), and the body not to return the
    Template's declared type (``return``/``empty-body``). All normal for an
    incrementally-built REPL, not real errors.
    """
    lenient_flags = (
        [
            "--allow-redefinition-new",
            "--local-partial-types",
            "--disable-error-code=no-redef",
            "--disable-error-code=return",
            "--disable-error-code=empty-body",
        ]
        if lenient
        else []
    )
    # Run mypy as a subprocess, not the in-process `mypy.api.run`: the API builds
    # typeshed and a full module graph inside this process and never returns that
    # memory, so under a test/agent session doing many checks it accumulates to many
    # GB (OOM). A subprocess reclaims all of it on exit. Pass a file (not --command:
    # it hits an argv length limit on large modules); each call gets an isolated temp
    # dir + cache so parallel decodes don't share -- and deadlock on -- mypy's cache.
    tmpdir = tempfile.mkdtemp(prefix="effectful_typecheck_")
    try:
        tf_path = os.path.join(tmpdir, "_synthesized.py")
        with open(tf_path, "w", encoding="utf-8") as f:
            f.write(source)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                tf_path,
                "--cache-dir",
                os.path.join(tmpdir, "cache"),
                "--no-error-summary",
                "--output=json",
                "--ignore-missing-imports",
                "--disable-error-code=import-untyped",
                *lenient_flags,
            ],
            capture_output=True,
            text=True,
        )
        stdout, stderr, status = proc.stdout, proc.stderr, proc.returncode
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    # Exit status >= 2 means mypy itself failed (fatal/usage/internal/syntax) -- a
    # tool failure, not a type error -- and it emits text rather than JSON, so
    # raise `RuntimeError` rather than parse or silently pass.
    if status >= 2:
        raise RuntimeError(
            f"mypy could not check the source:\n{(stdout or '') + (stderr or '')}"
        )
    errors = _region_errors(stdout or "", lo, hi)
    if errors:
        # Not the source: it's large and the model already has the generated code.
        report = "\n".join(json.dumps(e) for e in errors)
        raise TypeError("mypy type check failed:\n" + report)


class MypyTypeChecker(ObjectInterpretation):
    """Handler that handles type_check by shelling out to mypy.

    Independent of any executor: it says how generated code is *checked*, not how
    it is parsed, compiled or run, so it is installed alongside whichever of those
    handlers a stack uses (``handler(MypyTypeChecker()), handler(BuiltinExecutor())``)
    rather than being part of one.
    """

    @implements(type_check)
    def type_check(
        self,
        source: str,
        lo: int | None = None,
        hi: int | None = None,
        *,
        lenient: bool = False,
    ) -> None:
        _mypy_check_region(source, lo, hi, lenient)
