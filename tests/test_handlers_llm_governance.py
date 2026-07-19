"""L0 static tool governance: ``toolsof`` / ``reachable_tools`` — no LLM call.

These check the static tool graph (§5): which tools a template, or a function that calls
one, can reach — computed by reifying to a Term and folding, never running the LLM.
"""

from typing import Annotated

from effectful.handlers.llm.governance import check_tools, reachable_tools, toolsof
from effectful.handlers.llm.template import Template, Tool
from effectful.internals.runtime import interpreter
from effectful.ops.semantics import apply
from effectful.ops.syntax import Uses


def _trip_planner():
    """Build a template with a captured tool graph and a caller of it.

    Returns ``(suggest_city, delete_everything, my_fn)`` where ``suggest_city`` is a
    template that lexically captures ``cities``/``weather``/``delete_everything``, and
    ``my_fn`` is a plain function that calls the template.
    """

    @Tool.define
    def cities() -> list[str]:
        """Return a list of cities."""
        return ["Chicago", "Barcelona"]

    @Tool.define
    def weather(city: str) -> str:
        """Return the weather in a city."""
        return "sunny"

    @Tool.define
    def delete_everything() -> None:
        """Dangerous: wipe all state."""
        raise RuntimeError("boom")

    @Template.define
    def suggest_city() -> str:
        """Use the `cities` and `weather` tools to suggest a city."""
        raise NotImplementedError

    def my_fn() -> str:
        return suggest_city()

    return suggest_city, delete_everything, my_fn


def test_toolsof_is_the_static_tool_graph():
    suggest_city, delete_everything, _ = _trip_planner()
    reached = toolsof(suggest_city)
    # every lexically-captured tool is reachable, including the dangerous one
    assert delete_everything in reached
    # the root itself is not one of the tools it reaches
    assert suggest_city not in reached


def test_reachable_tools_sees_through_a_template_without_calling_the_llm():
    suggest_city, delete_everything, my_fn = _trip_planner()
    reached = reachable_tools(my_fn)
    # the template it calls, and (transitively) that template's captured tools
    assert suggest_city in reached
    assert delete_everything in reached
    assert toolsof(suggest_city) <= reached


def test_reachable_tools_is_the_leak_check():
    # `reachable_tools(fn) <= declared` is the static tool-safety guarantee.
    suggest_city, delete_everything, my_fn = _trip_planner()
    declared = {suggest_city} | toolsof(suggest_city) - {delete_everything}
    leak = reachable_tools(my_fn) - declared
    assert leak == frozenset({delete_everything})  # flagged, LLM never called


def test_uses_restricts_template_tools_to_the_allow_list():
    # L1: a `Uses[...]` return annotation makes `.tools` an allow-list. With cities,
    # weather and delete_everything all in scope, a template declaring `Uses[cities]`
    # is offered only cities — the dangerous tool is never even presented to the LLM.
    @Tool.define
    def cities() -> list[str]:
        """Return a list of cities."""
        return ["Chicago"]

    @Tool.define
    def weather(city: str) -> str:
        """Return the weather."""
        return "sunny"

    @Tool.define
    def delete_everything() -> None:
        """Dangerous."""
        raise RuntimeError

    @Template.define
    def restricted() -> Annotated[str, Uses[cities]]:
        """Suggest a city using only `cities`."""
        raise NotImplementedError

    @Template.define
    def unrestricted() -> str:
        """Suggest a city."""
        raise NotImplementedError

    assert set(restricted.tools.values()) == {cities}
    # the restriction flows transitively — the dangerous tool drops out of reachability
    assert delete_everything not in toolsof(restricted)
    assert delete_everything not in reachable_tools(lambda: restricted())
    # default (no Uses) is unchanged: full lexical capture, dangerous tool included
    assert {cities, weather, delete_everything} <= set(unrestricted.tools.values())


def test_check_tools_flags_the_leak():
    # check_tools = reachable_tools - allowed; the L2 tool-safety check (no LLM).
    suggest_city, delete_everything, my_fn = _trip_planner()
    allowed = {suggest_city} | toolsof(suggest_city) - {delete_everything}
    assert check_tools(my_fn, *allowed) == frozenset({delete_everything})
    # allowing everything reachable -> no leak
    assert check_tools(my_fn, *reachable_tools(my_fn)) == frozenset()


def test_reachable_tools_ignores_ambient_apply_handler():
    # Soundness law: static reachability must not depend on what is installed at the call
    # site. With `handler` (merge) an ambient apply interpretation would win dispatch, so a
    # called tool runs concretely instead of reifying and is silently missed. The reifier
    # uses `interpreter` (replace), so the row is identical either way and the reified
    # tool ops never reach the ambient handler.
    suggest_city, delete_everything, my_fn = _trip_planner()
    baseline = reachable_tools(my_fn)

    ran = []

    def ambient(op, *a, **k):  # a valid ambient interpretation (concrete execution)
        ran.append(op)
        return op.__default_rule__(*a, **k)

    with interpreter({apply: ambient}):
        under_ambient = reachable_tools(my_fn)

    assert under_ambient == baseline  # row unaffected by the ambient handler
    assert delete_everything in under_ambient
    # reification isolated the tool calls — none of the trip tools executed concretely
    assert not ({suggest_city, delete_everything} & set(ran))
