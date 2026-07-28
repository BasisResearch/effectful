"""Tests for `effectful.handlers.llm.choreographies` -- choreographic endpoint projection.

No real LLM is involved: `MockLLM` and friends implement `Template.__apply__`
directly, so what is under test is the choreography -- step allocation, result
sharing, scatter distribution -- rather than any completion logic.

Every test runs under a timeout, because the failure mode of a concurrency bug
here is a hang rather than a wrong answer.
"""

import asyncio
import concurrent.futures
import gc
import logging
import threading
from typing import Any

import pytest

from effectful.handlers.llm import Agent, Template
from effectful.handlers.llm.choreographies import (
    Choreography,
    ChoreographyError,
    call,
    scatter,
    step,
)
from effectful.ops.semantics import handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled

TIMEOUT = 10
"""Seconds any single choreography may take. Concurrency bugs show up as hangs."""


def run(coro, timeout: float = TIMEOUT) -> Any:
    """Run *coro* to completion under a timeout."""

    async def _main():
        return await asyncio.wait_for(coro, timeout)

    return asyncio.run(_main())


# ── Test doubles ──────────────────────────────────────────────────


def _key(template) -> str:
    """``agent-id.template-name`` for a bound template, else its name."""
    agent = getattr(template, "__agent__", None)
    name = template.__name__
    return f"{agent.__agent_id__}.{name}" if agent is not None else name


class MockLLM(ObjectInterpretation):
    """Answers template calls from a canned mapping.

    Keys are matched most-specific first: ``agent-id.template-name``, then
    ``template-name``. A value may be a callable ``(template, args) -> result``.
    """

    def __init__(self, responses: dict[str, Any]):
        self._responses = responses
        self._lock = threading.Lock()
        self.calls: list[str] = []

    @implements(Template.__apply__)
    def _call(self, template, *args, **kwargs):
        key = _key(template)
        with self._lock:
            self.calls.append(key)
        if key in self._responses:
            response = self._responses[key]
        elif template.__name__ in self._responses:
            response = self._responses[template.__name__]
        else:
            response = f"mock-{template.__name__}"
        return response(template, args) if callable(response) else response

    def calls_for(self, agent_id: str) -> list[str]:
        with self._lock:
            return [c for c in self.calls if c.startswith(f"{agent_id}.")]


class FailingMockLLM(MockLLM):
    """A `MockLLM` that raises on specific ``agent-id.template-name`` keys."""

    def __init__(self, responses: dict[str, Any], fail_on: set[str]):
        super().__init__(responses)
        self._fail_on = fail_on

    @implements(Template.__apply__)
    def _call(self, template, *args, **kwargs):
        if _key(template) in self._fail_on:
            raise RuntimeError(f"Simulated failure on {_key(template)}")
        return super()._call(template, *args, **kwargs)


# ── Agents ────────────────────────────────────────────────────────


class Architect(Agent):
    """Plans modules."""

    @Template.define
    def plan(self, spec: str) -> str:
        """Plan modules for: {spec}"""
        raise NotHandled


class Coder(Agent):
    """Writes code."""

    @Template.define
    def implement(self, spec: str) -> str:
        """Implement: {spec}"""
        raise NotHandled


class Reviewer(Agent):
    """Reviews code."""

    @Template.define
    def review(self, code: str) -> str:
        """Review: {code}"""
        raise NotHandled


class Verifier(Agent):
    """Writes tests."""

    @Template.define
    def write_tests(self, spec: str) -> str:
        """Write tests for: {spec}"""
        raise NotHandled


@Template.define
def announce(text: str) -> str:
    """Announce: {text}"""
    raise NotHandled


# ── Helpers ───────────────────────────────────────────────────────


def choreograph(program, agents, responses, *, mock=None, **kwargs):
    """Run *program* over *agents* with a mock LLM; return ``(result, mock)``."""
    mock = mock if mock is not None else MockLLM(responses)
    choreo = Choreography(program, agents=agents, handlers=[mock])
    return run(choreo.run_async(**kwargs)), mock


async def _plan_then_implement(architect, coder):
    plan = await step(architect.plan, "spec")
    return await step(coder.implement, plan)


# ── Tests ─────────────────────────────────────────────────────────


class TestChoreography:
    def test_each_agent_executes_only_its_own_steps(self):
        architect, coder = Architect(agent_id="arch"), Coder(agent_id="coder")

        async def program(spec, architect, coder):
            plan = await step(architect.plan, spec)
            return await step(coder.implement, plan)

        result, mock = choreograph(
            program,
            [architect, coder],
            {"plan": "the plan", "implement": "the code"},
            spec="build it",
            architect=architect,
            coder=coder,
        )

        assert result == "the code"
        assert mock.calls_for("arch") == ["arch.plan"]
        assert mock.calls_for("coder") == ["coder.implement"]

    def test_every_agent_computes_the_same_result(self):
        agents = [Architect(agent_id="a"), Coder(agent_id="c"), Reviewer(agent_id="r")]
        architect, coder, reviewer = agents

        seen: list[Any] = []

        async def program(architect, coder, reviewer):
            plan = await step(architect.plan, "spec")
            code = await step(coder.implement, plan)
            verdict = await step(reviewer.review, code)
            seen.append((plan, code, verdict))
            return verdict

        result, _ = choreograph(
            program,
            agents,
            {"plan": "P", "implement": "C", "review": "PASS"},
            architect=architect,
            coder=coder,
            reviewer=reviewer,
        )

        assert result == "PASS"
        assert len(seen) == 3
        assert all(view == seen[0] for view in seen)

    def test_control_flow_stays_in_sync(self):
        """A loop whose trip count depends on a peer's result: every agent must
        take the same branch, because every agent sees the same results."""
        architect = Architect(agent_id="arch")
        coder = Coder(agent_id="coder")
        reviewer = Reviewer(agent_id="rev")

        reviews = iter(["RETRY", "RETRY", "PASS"])
        lock = threading.Lock()

        def next_review(template, args):
            with lock:
                return next(reviews)

        async def program(architect, coder, reviewer):
            plan = await step(architect.plan, "spec")
            code = await step(coder.implement, plan)
            rounds = 0
            while True:
                rounds += 1
                verdict = await step(reviewer.review, code)
                if verdict == "PASS":
                    return rounds
                code = await step(coder.implement, verdict)

        result, mock = choreograph(
            program,
            [architect, coder, reviewer],
            {"plan": "P", "implement": "C", "review": next_review},
            architect=architect,
            coder=coder,
            reviewer=reviewer,
        )

        assert result == 3
        assert mock.calls_for("rev") == ["rev.review"] * 3
        assert mock.calls_for("coder") == ["coder.implement"] * 3

    def test_a_step_returning_none_does_not_hang(self):
        """``None`` is a result like any other, and has to be distinguishable
        from a step that has not finished."""
        architect, coder = Architect(agent_id="arch"), Coder(agent_id="coder")

        async def program(architect, coder):
            plan = await step(architect.plan, "spec")
            assert plan is None
            return await step(coder.implement, "go")

        result, _ = choreograph(
            program,
            [architect, coder],
            {"plan": None, "implement": "code"},
            architect=architect,
            coder=coder,
        )
        assert result == "code"

    def test_unbound_template_runs_on_every_agent(self):
        architect, coder = Architect(agent_id="arch"), Coder(agent_id="coder")

        async def program(architect, coder):
            headline = await step(announce, "starting")
            plan = await step(architect.plan, headline)
            return await step(coder.implement, plan)

        _, mock = choreograph(
            program,
            [architect, coder],
            {"announce": "news", "plan": "P", "implement": "C"},
            architect=architect,
            coder=coder,
        )
        assert mock.calls.count("announce") == 2

    def test_single_agent(self):
        coder = Coder(agent_id="solo")

        async def program(coder):
            return await step(coder.implement, "spec")

        result, _ = choreograph(program, [coder], {"implement": "code"}, coder=coder)
        assert result == "code"

    def test_a_step_owned_by_an_outsider_is_rejected(self):
        """Nobody would ever run it, so every agent would wait for a result
        that cannot arrive. Fail loudly rather than hang."""
        coder = Coder(agent_id="coder")
        stranger = Reviewer(agent_id="not-in-this-choreography")

        async def program(coder, stranger):
            code = await step(coder.implement, "spec")
            return await step(stranger.review, code)

        choreo = Choreography(program, agents=[coder], handlers=[MockLLM({})])
        with pytest.raises(ChoreographyError, match="not part of this choreography"):
            run(choreo.run_async(coder=coder, stranger=stranger))

    def test_agent_failure_becomes_a_choreography_error(self):
        architect, coder = Architect(agent_id="arch"), Coder(agent_id="coder")
        choreo = Choreography(
            _plan_then_implement,
            agents=[architect, coder],
            handlers=[FailingMockLLM({"plan": "P"}, fail_on={"coder.implement"})],
        )

        with pytest.raises(ChoreographyError, match="'coder' failed"):
            run(choreo.run_async(architect=architect, coder=coder))

    def test_a_failed_run_leaves_no_unretrieved_futures(self, caplog):
        """The agents waiting on a failed step are cancelled by the task group
        before they read its exception. Asyncio complains about an exception
        nobody retrieved unless the failing side reads it first."""
        architect, coder = Architect(agent_id="arch"), Coder(agent_id="coder")
        choreo = Choreography(
            _plan_then_implement,
            agents=[architect, coder],
            handlers=[FailingMockLLM({}, fail_on={"arch.plan"})],
        )

        with caplog.at_level(logging.DEBUG, logger="asyncio"):
            with pytest.raises(ChoreographyError):
                run(choreo.run_async(architect=architect, coder=coder))
            gc.collect()

        assert "never retrieved" not in caplog.text

    def test_repeated_runs_are_deterministic(self):
        architect, coder = Architect(agent_id="arch"), Coder(agent_id="coder")
        choreo = Choreography(
            _plan_then_implement,
            agents=[architect, coder],
            handlers=[(mock := MockLLM({"plan": "P", "implement": "C"}))],
        )

        results = [
            run(choreo.run_async(architect=architect, coder=coder)) for _ in range(5)
        ]

        assert results == ["C"] * 5
        # Results live for the duration of a run, so each run re-executes.
        assert mock.calls_for("arch") == ["arch.plan"] * 5


class TestScatter:
    def test_results_come_back_in_item_order(self):
        architect = Architect(agent_id="arch")
        coders = [Coder(agent_id=f"coder-{i}") for i in range(3)]

        async def program(architect, coder):
            await step(architect.plan, "spec")
            return await scatter(
                ["a", "b", "c", "d", "e"],
                coder,
                lambda c, item: call(c.implement, item),
            )

        result, mock = choreograph(
            program,
            [architect, *coders],
            {"plan": "P", "implement": lambda template, args: f"code({args[0]})"},
            architect=architect,
            coder=coders,
        )

        assert result == [f"code({item})" for item in "abcde"]
        # Each item was implemented exactly once, by exactly one coder.
        assert len([c for c in mock.calls if c.endswith(".implement")]) == 5

    def test_work_is_shared_between_agents(self):
        """Two coders, two items, and a barrier only both of them can clear.

        If one coder took both items the other would never arrive and the
        barrier would time out -- which is the point: this asserts real
        concurrent pull, not just that the work got done.
        """
        barrier = threading.Barrier(2, timeout=TIMEOUT / 2)
        coders = [Coder(agent_id="coder-1"), Coder(agent_id="coder-2")]

        def implement(template, args):
            barrier.wait()
            return f"code({args[0]})"

        async def program(coder):
            return await scatter(
                ["x", "y"], coder, lambda c, item: call(c.implement, item)
            )

        result, mock = choreograph(
            program, coders, {"implement": implement}, coder=coders
        )

        assert result == ["code(x)", "code(y)"]
        assert mock.calls_for("coder-1") and mock.calls_for("coder-2")

    def test_empty_scatter(self):
        coder = Coder(agent_id="coder")

        async def program(coder):
            return await scatter([], coder, lambda c, item: call(c.implement, item))

        result, _ = choreograph(program, [coder], {}, coder=coder)
        assert result == []

    def test_concurrent_scatters_via_gather(self):
        """`asyncio.gather` over several scatters lets agents in different
        groups work at the same time."""
        coder = Coder(agent_id="coder")
        tester = Verifier(agent_id="tester")

        # Neither group can finish until both have started.
        barrier = threading.Barrier(2, timeout=TIMEOUT / 2)

        def gated(prefix):
            def _run(template, args):
                barrier.wait()
                return f"{prefix}({args[0]})"

            return _run

        async def program(coder, tester):
            return await asyncio.gather(
                scatter(["a"], coder, lambda c, item: call(c.implement, item)),
                scatter(["b"], tester, lambda t, item: call(t.write_tests, item)),
            )

        (code, tests), _ = choreograph(
            program,
            [coder, tester],
            {"implement": gated("code"), "write_tests": gated("tests")},
            coder=coder,
            tester=tester,
        )

        assert code == ["code(a)"]
        assert tests == ["tests(b)"]

    def test_step_inside_scatter_is_rejected(self):
        """Per-item work is not a choreography step: allocating step IDs from
        inside a scatter would desynchronise the agents."""
        coder = Coder(agent_id="coder")
        reviewer = Reviewer(agent_id="rev")

        async def program(coder, reviewer):
            return await scatter(
                ["a"], coder, lambda c, item: step(reviewer.review, item)
            )

        choreo = Choreography(program, agents=[coder, reviewer], handlers=[MockLLM({})])

        with pytest.raises(ChoreographyError, match="cannot be used inside scatter"):
            run(choreo.run_async(coder=coder, reviewer=reviewer))

    def test_failure_inside_scatter_propagates(self):
        coders = [Coder(agent_id="coder-1"), Coder(agent_id="coder-2")]

        async def program(coder):
            return await scatter(
                ["a", "b"], coder, lambda c, item: call(c.implement, item)
            )

        choreo = Choreography(
            program,
            agents=coders,
            handlers=[FailingMockLLM({}, fail_on={"coder-1.implement"})],
        )

        with pytest.raises(ChoreographyError, match="Simulated failure"):
            run(choreo.run_async(coder=coders))

    def test_scatter_outside_a_choreography_is_sequential(self):
        coders = [Coder(agent_id="coder-1"), Coder(agent_id="coder-2")]
        seen: list[str] = []

        async def fake_call(agent, item):
            seen.append(f"{agent.__agent_id__}:{item}")
            return item.upper()

        result = run(scatter(["a", "b", "c"], coders, fake_call))
        assert result == ["A", "B", "C"]
        assert seen == ["coder-1:a", "coder-2:b", "coder-1:c"]


class TestManualProjection:
    """Driving projections directly, without `Choreography.run_async`."""

    def test_agents_outnumbering_worker_threads_still_finish(self):
        """Four agents, one worker thread, a chain in which each waits on the
        previous one. A waiting agent is a suspended coroutine, so it holds no
        thread -- run agent-per-thread instead and this deadlocks."""
        agents = [Coder(agent_id=f"coder-{i}") for i in range(4)]
        mock = MockLLM({"implement": lambda template, args: args[0] + "!"})
        choreo = Choreography(_chain, agents=agents)

        async def go():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return await asyncio.gather(
                    *(_drive(choreo, a, mock, pool, agents) for a in agents)
                )

        assert run(go()) == ["x!!!!"] * 4

    def test_a_waiting_agent_sees_the_owners_failure(self):
        """`asyncio.gather` does not cancel siblings when one raises, so a
        waiting agent only learns of a failure if the failed step carries it."""
        agents = [Coder(agent_id=f"coder-{i}") for i in range(2)]
        mock = FailingMockLLM({}, fail_on={"coder-0.implement"})
        choreo = Choreography(_chain, agents=agents)

        async def go():
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                return await asyncio.gather(
                    *(_drive(choreo, a, mock, pool, agents) for a in agents),
                    return_exceptions=True,
                )

        outcomes = run(go())
        assert [type(o) for o in outcomes] == [RuntimeError, RuntimeError]


async def _chain(coders):
    value = "x"
    for coder in coders:
        value = await step(coder.implement, value)
    return value


async def _drive(choreo, agent, mock, pool, coders):
    projection = choreo.projection(agent, executor=pool)
    with handler(mock), projection.activate():
        return await _chain(coders)


class TestHandlerPropagation:
    """`effectful`'s interpretation lives in a `contextvars.ContextVar`, which
    is what makes this design work: each agent task gets its own copy, and
    each blocking call carries that copy into its worker thread."""

    def test_handlers_installed_around_run_reach_the_worker_threads(self):
        """Handlers do not have to be passed to `Choreography` -- anything
        installed around the call (as `effectful.handlers.llm.harness` does)
        is inherited by every agent task and by the threads they call into."""
        architect, coder = Architect(agent_id="arch"), Coder(agent_id="coder")
        mock = MockLLM({"plan": "P", "implement": "C"})
        choreo = Choreography(_plan_then_implement, agents=[architect, coder])

        with handler(mock):
            assert run(choreo.run_async(architect=architect, coder=coder)) == "C"
        assert sorted(mock.calls) == ["arch.plan", "coder.implement"]
