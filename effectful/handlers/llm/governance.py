"""Static tool-graph governance for LLM :class:`~effectful.handlers.llm.template.Template` s.

A :class:`~effectful.handlers.llm.template.Tool` **is** an
:class:`~effectful.ops.types.Operation` (``class Tool(Operation)``,
``class Template(Tool)``), so a template's tools *are* part of its effect row. These are
the ``ε``-engine siblings specialised to the tool graph — the **static, sound** core of
§5 tool governance, computable with **no LLM call**:

* :func:`toolsof` — the transitive tool graph reachable from a tool/template via ``.tools``.
* :func:`reachable_tools` — the tools a zero-arg function can reach, *through* templates,
  by reifying it to a :class:`~effectful.ops.types.Term` (never running the LLM) and
  folding it with :func:`~effectful.ops.effects.usesof`.

``reachable_tools(fn) <= declared`` is then a compile-time tool-safety check (L2): an
agent provably cannot reach a dangerous tool, even through nested sub-agents. This is
*more* tractable than the general effect fold — a static named-op graph, so none of the
higher-order/reifiability caveats of the general case apply.
"""

import collections.abc
from typing import Any, Callable

from effectful.handlers.llm.template import Tool
from effectful.internals.runtime import interpreter
from effectful.ops.semantics import apply
from effectful.ops.syntax import defdata
from effectful.ops.types import Term

__all__ = ["toolsof", "reachable_tools"]


def _tools_in(term: Any) -> frozenset[Tool]:
    """Every :class:`Tool` appearing *anywhere* in a reified ``term`` — as an operation
    or as an argument. ``Tool`` / ``Template`` subclass :class:`~effectful.ops.types.Operation`
    and define their own ``__apply__``, so a called tool sits in the ``args`` of an
    ``apply`` node rather than being the node's ``op``; a structural walk catches both.
    """
    found: set[Tool] = set()

    def walk(x: Any) -> None:
        if isinstance(x, Tool):
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
    LLM. ``tool`` itself is *not* included (it is the root, not something it reaches).
    """
    seen: set[Tool] = set()
    stack: list[Tool] = [tool]
    while stack:
        cur = stack.pop()
        for sub in getattr(cur, "tools", {}).values():
            if sub not in seen:
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
