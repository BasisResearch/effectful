"""L0 static tool governance: ``toolsof`` / ``reachable_tools`` — no LLM call.

These check the static tool graph (§5): which tools a template, or a function that calls
one, can reach — computed by reifying to a Term and folding, never running the LLM.
"""

from effectful.handlers.llm.governance import reachable_tools, toolsof
from effectful.handlers.llm.template import Template, Tool


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
    names = {t.__name__ for t in reached}
    assert {"cities", "weather", "delete_everything"} <= names
    assert delete_everything in reached
    # the root itself is not one of the tools it reaches
    assert suggest_city not in reached


def test_reachable_tools_sees_through_a_template_without_calling_the_llm():
    suggest_city, delete_everything, my_fn = _trip_planner()
    reached = reachable_tools(my_fn)
    # the template it calls, and (transitively) that template's tools
    assert suggest_city in reached
    assert delete_everything in reached
    assert {"suggest_city", "cities", "weather", "delete_everything"} <= {
        t.__name__ for t in reached
    }


def test_reachable_tools_is_the_leak_check():
    # `reachable_tools(fn) <= declared` is the static tool-safety guarantee.
    suggest_city, delete_everything, my_fn = _trip_planner()
    declared = {suggest_city} | toolsof(suggest_city) - {delete_everything}
    leak = reachable_tools(my_fn) - declared
    assert leak == frozenset({delete_everything})  # flagged, LLM never called
