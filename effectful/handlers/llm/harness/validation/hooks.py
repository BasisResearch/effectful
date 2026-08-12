import collections.abc
import doctest
import types
import typing

import effectful.handlers.llm.harness.execution.hooks
from effectful.ops.types import Operation


@Operation.define
def type_check(
    source: str,
    lo: int | None = None,
    hi: int | None = None,
    *,
    lenient: bool = False,
) -> None:
    """
    Type check a module source, reporting only diagnostics inside a line region.

    source: A complete module source to check (e.g. produced by
        ``splice_into_source``, which splices generated code into a Template's real
        module source).
    lo, hi: Inclusive line range within ``source`` to report errors from; when
        omitted, the whole source is in scope. Errors outside the region are
        ignored so unrelated pre-existing code never blocks synthesis.
    lenient: when True, relax the check for incrementally-built REPL code spliced into
        a Template body -- allow redefinition (a cell may rebind or redefine a name)
        and don't require the body to satisfy the Template's return type. Off (strict)
        for a synthesized ``Callable`` or ``TemplateBody``, which must honor its
        signature and gets no redefinition slack. How much slack this buys is up to
        the handler: it is a list of disabled mypy error codes under `MypyTypeChecker`
        and close to a no-op under `TyTypeChecker`, which is already this permissive.

    Returns if the source type-checks, raises TypeError on an in-region failure.

    Unlike `parse`/`compile`/`exec`, which have no meaning without a provider,
    type checking is an optional layer over them: the default rule below passes
    everything, so a stack with no type checker installed runs generated code
    unchecked rather than refusing to run it at all. `MypyTypeChecker` and
    `TyTypeChecker` are the handlers that make the check real.
    """
    return None


@Operation.define
def run_doctests(
    obj: collections.abc.Callable | type | types.ModuleType,
    globs: collections.abc.Mapping[str, typing.Any],
) -> None:
    """Run the doctests found in a synthesized object's docstring.

    obj: The synthesized object (typically a function) whose docstring may
        contain interactive ``>>>`` examples.
    globs: The namespace the examples execute in (typically the exec namespace,
        which already contains the function plus its lexical context).

    Returns None, raises TypeError if any doctest example fails. A docstring
    with no examples is a no-op (passes trivially).

    Unlike the other operations here, this one carries its mechanics in its
    default rule: finding the examples and reporting their failures is the same
    work whatever provider is installed. What differs is how each example is
    compiled and executed, and that is delegated to the `compile` and `exec`
    operations -- so a docstring's examples run under exactly the provider that
    runs the code they document, and no provider at all is an error here just as
    it is there.
    """
    assert hasattr(obj, "__name__")
    finder = doctest.DocTestFinder(recurse=False)
    runner = doctest.DocTestRunner(verbose=False)
    # `doctest.DocTestRunner` runs every example as a bare
    # ``exec(compile(source, filename, "single", flags, True), test.globs)``, with
    # no hook to supply a different compiler or a different way to execute the
    # result. Both names resolve as globals of the `doctest` module before falling
    # back to builtins, so binding the operations there for the duration of the run
    # redirects example compilation and execution without reimplementing (and
    # having to track) the runner. `compile`'s signature is `builtins.compile`'s, so
    # it takes the runner's call as written; `exec` takes the runner's ``(code,
    # globs)`` and, under a provider that sandboxes it, supplies its own namespace.
    #
    # This rebinds module state, so it is not safe against another thread running
    # doctests concurrently under a *different* interpretation; the block is short
    # and holds only for the examples of one synthesized object.
    doctest.compile = effectful.handlers.llm.harness.execution.hooks.compile  # type: ignore[attr-defined]
    doctest.exec = effectful.handlers.llm.harness.execution.hooks.exec  # type: ignore[attr-defined]
    # Collect each example's want/got report via `out=...` and read failure
    # counts from `run`'s return value, avoiding `summarize`, which would print
    # to stdout instead of returning the report.
    output: list[str] = []
    failed = attempted = 0
    try:
        for test in finder.find(obj, name=obj.__name__, globs=dict(globs)):
            results = runner.run(test, out=output.append)
            failed += results.failed
            attempted += results.attempted
    finally:
        # `doctest` defines neither name itself, so removing the bindings restores
        # it exactly as we found it: both resolve to the builtin again.
        doctest.__dict__.pop("compile", None)
        doctest.__dict__.pop("exec", None)
    if failed:
        report = "".join(output).strip()
        if not report:
            report = f"{failed} doctest(s) failed out of {attempted} attempted."
        raise TypeError(f"doctest failed:\n{report}")
    return None
