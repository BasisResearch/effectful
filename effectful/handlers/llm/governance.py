"""Static tool-graph governance for LLM :class:`~effectful.handlers.llm.template.Template` s.

A :class:`~effectful.handlers.llm.template.Tool` **is** an
:class:`~effectful.ops.types.Operation` (``class Tool(Operation)``,
``class Template(Tool)``), so a template's tools *are* part of its effect row. These
compute the tool graph without ever calling the LLM:

* :func:`toolsof` — the transitive tool graph reachable from a tool/template via ``.tools``.
  Fully static: it reads lexically-captured ``.tools`` mappings, nothing is executed.
* :func:`reachable_tools` — the tools a zero-arg function can reach, *through* templates,
  by reifying it to a :class:`~effectful.ops.types.Term` (never running the LLM or any
  tool body) and walking it for the tools it mentions, then expanding via :func:`toolsof`.
* :func:`check_tools` — the leak check ``reachable_tools(fn) - allowed``.

**Soundness precondition (important).** :func:`reachable_tools` obtains the term by
*running ``fn``'s own Python body* under a reifying interpretation. Operation calls become
term nodes, but **native Python control flow in ``fn`` is resolved at analysis time** — an
untaken ``if``/``for``/``try`` branch contributes no tools. So ``check_tools(fn) == set()``
is a tool-safety guarantee **only for a straight-line ``fn``** (or one whose branching is
expressed with reifying conditional *operations*, which fold both arms). It is not a proof
over arbitrary Python control flow; for a branchy ``fn`` it is the set reached *on this
reification*, which may under-approximate. Keep governed entry points straight-line.
"""

import collections.abc
from collections.abc import Callable
from typing import Any

from effectful.handlers.llm.completions import _LexicalVariableTool
from effectful.handlers.llm.template import Tool
from effectful.internals.runtime import interpreter
from effectful.ops.semantics import apply
from effectful.ops.syntax import defdata
from effectful.ops.types import Term

__all__ = ["toolsof", "reachable_tools", "check_tools"]


def _is_governed(tool: Any) -> bool:
    """A ``Tool`` the governance graph should count. Excludes synthetic
    :class:`~effectful.handlers.llm.completions._LexicalVariableTool` readers — they are
    prompt-variable plumbing auto-wrapped from lexical values by ``LexicalReaders``, not
    tools an agent "reaches", so treating them as reachable would flag plumbing as a leak.
    """
    return isinstance(tool, Tool) and not isinstance(tool, _LexicalVariableTool)


def _tools_in(term: Any) -> frozenset[Tool]:
    """Every governed :class:`Tool` appearing *anywhere* in a reified ``term`` — as an
    operation or as an argument. ``Tool`` / ``Template`` subclass
    :class:`~effectful.ops.types.Operation` and define their own ``__apply__``, so a called
    tool sits in the ``args`` of an ``apply`` node rather than being the node's ``op``; a
    structural walk catches both. Synthetic lexical readers are excluded (:func:`_is_governed`).
    """
    found: set[Tool] = set()

    def walk(x: Any) -> None:
        if _is_governed(x):
            found.add(x)
        if isinstance(x, Term):
            walk(x.op)
            for a in x.args:
                walk(a)
            for v in x.kwargs.values():
                walk(v)
        elif isinstance(x, collections.abc.Mapping):
            for k, v in x.items():
                walk(k)
                walk(v)
        elif isinstance(x, (list, tuple, set, frozenset)):
            for e in x:
                walk(e)

    walk(term)
    return frozenset(found)


def toolsof(tool: Tool) -> frozenset[Tool]:
    """The tools transitively reachable from ``tool`` through its ``.tools`` graph
    (a template's tools are themselves tools, so this closes over sub-agents too).

    Fully static — it reads the lexically-captured ``.tools`` mapping, never calls the
    LLM. ``tool`` itself is *not* included (it is the root, not something it reaches), and
    synthetic lexical readers are excluded (:func:`_is_governed`).
    """
    seen: set[Tool] = set()
    stack: list[Tool] = [tool]
    while stack:
        cur = stack.pop()
        for sub in getattr(cur, "tools", {}).values():
            if _is_governed(sub) and sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return frozenset(seen)


def reachable_tools(fn: Callable[[], Any]) -> frozenset[Tool]:
    """Every tool a zero-arg ``fn`` can reach, *including through templates it calls*,
    without ever running a tool body or the LLM.

    ``fn`` is reified to a :class:`~effectful.ops.types.Term` under ``defdata`` — so even
    tools with real implementations become term nodes rather than executing — and the term
    is walked for the :class:`Tool` s it mentions (:func:`_tools_in`). Those are the
    *directly* reached tools; :func:`toolsof` expands each to the tools it in turn reaches.
    A template's body is ``raise NotHandled`` so it performs no tools directly, but the
    tools it captured lexically are recovered statically through :func:`toolsof`.

    The reifier uses ``interpreter`` (*replace*), never ``handler`` (*merge*): merging
    would let an ambient ``Tool.__apply__`` handler win dispatch, so the tool would run
    concretely instead of reifying and the walk would silently miss it — the static
    guarantee must hold regardless of what is installed at the call site.
    """
    with interpreter({apply: defdata}):
        term = fn()  # reify — no tool body runs, no LLM call, ambient handlers ignored
    direct = _tools_in(term)
    return direct.union(*(toolsof(t) for t in direct))


def check_tools(fn: Callable[[], Any], *allowed: Tool) -> frozenset[Tool]:
    """Tools ``fn`` can reach that are not in the ``allowed`` set — the static tool-safety
    leak check, computed with **no LLM call**. Empty == ``fn`` reaches no tool outside
    ``allowed`` *on this reification* (including through nested templates); this is a
    guarantee only for a straight-line ``fn`` — see :func:`reachable_tools` for the
    precondition (a branchy ``fn`` may under-approximate).

    The tool-graph analogue of :func:`~effectful.ops.effects.check_uses`:
    ``reachable_tools(fn) - allowed``.
    """
    return reachable_tools(fn) - frozenset(allowed)
