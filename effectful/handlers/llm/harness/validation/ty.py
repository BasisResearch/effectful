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
    """Handler that handles type_check by shelling out to ty.

    Interchangeable with
    `~effectful.handlers.llm.harness.validation.mypy.MypyTypeChecker` -- same
    operation, same contract, a different checker behind it. ty is a compiled binary
    that needs no per-call cache and builds no module graph in this process, so a
    check costs milliseconds where mypy's costs seconds, and on the failure path it
    reports the offending line with ty's own hints rather than a line of JSON.

    Independent of any executor: it says how generated code is *checked*, not how
    it is parsed, compiled or run, so it is installed alongside whichever of those
    handlers a stack uses (``handler(TyTypeChecker()), handler(BuiltinExecutor())``)
    rather than being part of one.
    """

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
                    "full",
                    # Matches mypy's `--ignore-missing-imports`: an unresolved module
                    # becomes `Unknown` and stays gradual, as mypy's becomes `Any`.
                    "--ignore",
                    "unresolved-import",
                    *(
                        arg
                        for rule in (self.lenient_ignored_rules if lenient else ())
                        for arg in ("--ignore", rule)
                    ),
                ],
                capture_output=True,
                text=True,
                cwd=tmpdir,
            )
            stdout, stderr, status = proc.stdout, proc.stderr, proc.returncode
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
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
            raise TypeError("ty type check failed:\n" + "\n\n".join(errors))
