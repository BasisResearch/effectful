"""Tests for ``docs/source/llm_examples/choreographies/library.py`` -- choreographic
endpoint projection.

No real LLM is involved: `MockLLM` and friends implement `Skill.__apply__`
directly, so what is under test is the choreography -- step allocation, result
sharing, scatter distribution -- rather than any completion logic.

Every test runs under a timeout, because the failure mode of a concurrency bug
here is a hang rather than a wrong answer.
"""

import asyncio
import concurrent.futures
import dataclasses
import gc
import logging
import threading
from typing import Any

import pytest

from docs.source.llm_examples.choreographies.library import (
    Choreography,
    ChoreographyError,
    EndpointProjection,
    _Steps,
    scatter,
    step,
)
from effectful.handlers.llm import Agent, Skill
from effectful.ops.semantics import fwd, handler
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


def _key(skill) -> str:
    """``agent-id.skill-name`` for a bound skill, else its name."""
    agent = getattr(skill, "__self__", None)
    name = skill.__name__
    return f"{agent.__agent_id__}.{name}" if agent is not None else name


class MockLLM(ObjectInterpretation):
    """Answers skill calls from a canned mapping.

    Keys are matched most-specific first: ``agent-id.skill-name``, then
    ``skill-name``. A value may be a callable ``(skill, args) -> result``.
    """

    def __init__(self, responses: dict[str, Any]):
        self._responses = responses
        self._lock = threading.Lock()
        self.calls: list[str] = []

    @implements(Skill.__apply__)
    def _call(self, skill, *args, **kwargs):
        key = _key(skill)
        with self._lock:
            self.calls.append(key)
        if key in self._responses:
            response = self._responses[key]
        elif skill.__name__ in self._responses:
            response = self._responses[skill.__name__]
        else:
            response = f"mock-{skill.__name__}"
        return response(skill, args) if callable(response) else response

    def calls_for(self, agent_id: str) -> list[str]:
        with self._lock:
            return [c for c in self.calls if c.startswith(f"{agent_id}.")]


class FailingMockLLM(MockLLM):
    """A `MockLLM` that raises on specific ``agent-id.skill-name`` keys."""

    def __init__(self, responses: dict[str, Any], fail_on: set[str]):
        super().__init__(responses)
        self._fail_on = fail_on

    @implements(Skill.__apply__)
    def _call(self, skill, *args, **kwargs):
        if _key(skill) in self._fail_on:
            raise RuntimeError(f"Simulated failure on {_key(skill)}")
        return super()._call(skill, *args, **kwargs)


# ── Agents ────────────────────────────────────────────────────────


class Architect(Agent):
    """Plans modules."""

    @Skill.define
    def plan(self, spec: str) -> str:
        """Plan modules for: {spec}"""
        raise NotHandled


class Coder(Agent):
    """Writes code."""

    @Skill.define
    def implement(self, spec: str) -> str:
        """Implement: {spec}"""
        raise NotHandled


class Reviewer(Agent):
    """Reviews code."""

    @Skill.define
    def review(self, code: str) -> str:
        """Review: {code}"""
        raise NotHandled


class Verifier(Agent):
    """Writes tests."""

    @Skill.define
    def write_tests(self, spec: str) -> str:
        """Write tests for: {spec}"""
        raise NotHandled


@Skill.define
def announce(text: str) -> str:
    """Announce: {text}"""
    raise NotHandled


# ── Helpers ───────────────────────────────────────────────────────


def choreograph(program, agents, responses, *, mock=None, log=None, **kwargs):
    """Run *program* over *agents* with a mock LLM; return ``(result, mock)``."""
    mock = mock if mock is not None else MockLLM(responses)
    choreo = Choreography(program, agents=agents, log=log)
    with handler(mock):
        return run(choreo.run_async(**kwargs)), mock


async def _plan_then_implement(architect, coder):
    plan = await step(architect.plan, "spec")
    return await step(coder.implement, plan)


# ── Tests ─────────────────────────────────────────────────────────


class TestChoreography:
    def test_a_choreography_is_called_like_its_program(self):
        """`Choreography` is the program made runnable: same arguments,
        positional or keyword, and the same result -- awaited for you."""
        architect, coder = Architect(__agent_id__="arch"), Coder(__agent_id__="coder")

        async def program(spec: str, architect: Architect, coder: Coder) -> str:
            plan = await step(architect.plan, spec)
            return await step(coder.implement, plan)

        choreo = Choreography(program, agents=[architect, coder])
        with handler(MockLLM({"plan": "P", "implement": "C"})):
            positional = choreo("build it", architect, coder)
            by_keyword = choreo("build it", architect=architect, coder=coder)

        assert positional == by_keyword == "C"
        # mypy infers `str` here from the program's return type; the assert
        # documents it, and the call above would not type check with the
        # wrong arguments.
        assert isinstance(positional, str)

    def test_each_agent_executes_only_its_own_steps(self):
        architect, coder = Architect(__agent_id__="arch"), Coder(__agent_id__="coder")

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
        agents = [
            Architect(__agent_id__="a"),
            Coder(__agent_id__="c"),
            Reviewer(__agent_id__="r"),
        ]
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
        architect = Architect(__agent_id__="arch")
        coder = Coder(__agent_id__="coder")
        reviewer = Reviewer(__agent_id__="rev")

        reviews = iter(["RETRY", "RETRY", "PASS"])
        lock = threading.Lock()

        def next_review(skill, args):
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
        architect, coder = Architect(__agent_id__="arch"), Coder(__agent_id__="coder")

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

    def test_unbound_skill_runs_on_every_agent(self):
        architect, coder = Architect(__agent_id__="arch"), Coder(__agent_id__="coder")

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
        coder = Coder(__agent_id__="solo")

        async def program(coder):
            return await step(coder.implement, "spec")

        result, _ = choreograph(program, [coder], {"implement": "code"}, coder=coder)
        assert result == "code"

    def test_a_step_owned_by_an_outsider_is_rejected(self):
        """Nobody would ever run it, so every agent would wait for a result
        that cannot arrive. Fail loudly rather than hang."""
        coder = Coder(__agent_id__="coder")
        stranger = Reviewer(__agent_id__="not-in-this-choreography")

        async def program(coder, stranger):
            code = await step(coder.implement, "spec")
            return await step(stranger.review, code)

        choreo = Choreography(program, agents=[coder])
        with (
            handler(MockLLM({})),
            pytest.raises(ChoreographyError, match="not part of this choreography"),
        ):
            run(choreo.run_async(coder=coder, stranger=stranger))

    def test_agent_failure_becomes_a_choreography_error(self):
        architect, coder = Architect(__agent_id__="arch"), Coder(__agent_id__="coder")
        choreo = Choreography(_plan_then_implement, agents=[architect, coder])

        with (
            handler(FailingMockLLM({"plan": "P"}, fail_on={"coder.implement"})),
            pytest.raises(ChoreographyError, match="'coder' failed"),
        ):
            run(choreo.run_async(architect=architect, coder=coder))

    def test_a_failed_run_leaves_no_unretrieved_futures(self, caplog):
        """The agents waiting on a failed step are cancelled by the task group
        before they read its exception. Asyncio complains about an exception
        nobody retrieved unless the failing side reads it first."""
        architect, coder = Architect(__agent_id__="arch"), Coder(__agent_id__="coder")
        choreo = Choreography(_plan_then_implement, agents=[architect, coder])

        with caplog.at_level(logging.DEBUG, logger="asyncio"):
            with (
                handler(FailingMockLLM({}, fail_on={"arch.plan"})),
                pytest.raises(ChoreographyError),
            ):
                run(choreo.run_async(architect=architect, coder=coder))
            gc.collect()

        assert "never retrieved" not in caplog.text

    def test_repeated_runs_are_deterministic(self):
        architect, coder = Architect(__agent_id__="arch"), Coder(__agent_id__="coder")
        choreo = Choreography(_plan_then_implement, agents=[architect, coder])
        mock = MockLLM({"plan": "P", "implement": "C"})

        with handler(mock):
            results = [
                run(choreo.run_async(architect=architect, coder=coder))
                for _ in range(5)
            ]

        assert results == ["C"] * 5
        # Results live for the duration of a run, so each run re-executes.
        assert mock.calls_for("arch") == ["arch.plan"] * 5


class TestScatter:
    def test_results_come_back_in_item_order(self):
        architect = Architect(__agent_id__="arch")
        coders = [Coder(__agent_id__=f"coder-{i}") for i in range(3)]

        async def program(architect, coder):
            await step(architect.plan, "spec")
            return await scatter(
                ["a", "b", "c", "d", "e"],
                coder,
                lambda c, item: step(c.implement, item),
            )

        result, mock = choreograph(
            program,
            [architect, *coders],
            {"plan": "P", "implement": lambda skill, args: f"code({args[0]})"},
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
        coders = [Coder(__agent_id__="coder-1"), Coder(__agent_id__="coder-2")]

        def implement(skill, args):
            barrier.wait()
            return f"code({args[0]})"

        async def program(coder):
            return await scatter(
                ["x", "y"], coder, lambda c, item: step(c.implement, item)
            )

        result, mock = choreograph(
            program, coders, {"implement": implement}, coder=coders
        )

        assert result == ["code(x)", "code(y)"]
        assert mock.calls_for("coder-1") and mock.calls_for("coder-2")

    def test_empty_scatter(self):
        coder = Coder(__agent_id__="coder")

        async def program(coder):
            return await scatter([], coder, lambda c, item: step(c.implement, item))

        result, _ = choreograph(program, [coder], {}, coder=coder)
        assert result == []

    def test_concurrent_scatters_via_gather(self):
        """`asyncio.gather` over several scatters lets agents in different
        groups work at the same time."""
        coder = Coder(__agent_id__="coder")
        tester = Verifier(__agent_id__="tester")

        # Neither group can finish until both have started.
        barrier = threading.Barrier(2, timeout=TIMEOUT / 2)

        def gated(prefix):
            def _run(skill, args):
                barrier.wait()
                return f"{prefix}({args[0]})"

            return _run

        async def program(coder, tester):
            return await asyncio.gather(
                scatter(["a"], coder, lambda c, item: step(c.implement, item)),
                scatter(["b"], tester, lambda t, item: step(t.write_tests, item)),
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

    def test_a_step_inside_an_item_allocates_no_step_id(self):
        """The item already is a step -- it has its own ID and its own place in
        the log -- so a step inside it is just the call, and the choreography's
        step counter is untouched by however many the item makes."""
        coder = Coder(__agent_id__="coder")

        async def two_calls(c, item):
            first = await step(c.implement, item)
            return await step(c.implement, first)

        async def program(coder):
            scattered = await scatter(["a", "b"], coder, two_calls)
            after = await step(coder.implement, "after")
            return scattered, after

        (scattered, after), mock = choreograph(
            program,
            [coder],
            {"implement": lambda skill, args: f"<{args[0]}>"},
            coder=coder,
        )

        assert scattered == ["<<a>>", "<<b>>"]
        assert after == "<after>"
        # Four calls inside the scatter, one step after it: the scatter is
        # step-0000 and the step that follows is step-0001, not step-0005.
        assert len(mock.calls) == 5

    def test_an_item_may_not_step_on_another_agent(self):
        """A scatter item is work one agent took on alone, so its skills
        have to be its own -- nobody else is waiting to run them."""
        coder = Coder(__agent_id__="coder")
        reviewer = Reviewer(__agent_id__="rev")

        async def program(coder, reviewer):
            return await scatter(
                ["a"], coder, lambda c, item: step(reviewer.review, item)
            )

        choreo = Choreography(program, agents=[coder, reviewer])

        with (
            handler(MockLLM({})),
            pytest.raises(ChoreographyError, match="belongs to 'rev'"),
        ):
            run(choreo.run_async(coder=coder, reviewer=reviewer))

    def test_failure_inside_scatter_propagates(self):
        coders = [Coder(__agent_id__="coder-1"), Coder(__agent_id__="coder-2")]

        async def program(coder):
            return await scatter(
                ["a", "b"], coder, lambda c, item: step(c.implement, item)
            )

        choreo = Choreography(program, agents=coders)

        with (
            handler(FailingMockLLM({}, fail_on={"coder-1.implement"})),
            pytest.raises(ChoreographyError, match="Simulated failure"),
        ):
            run(choreo.run_async(coder=coders))


async def _chain(coders):
    value = "x"
    for coder in coders:
        value = await step(coder.implement, value)
    return value


async def _drive(agents, steps, mock, pool):
    """Run `_chain` as every agent at once, over one shared step state.

    This is what `Choreography.run_async` does; spelling it out here is how
    these tests get to choose the thread pool.
    """

    async def as_agent(agent):
        projection = EndpointProjection(agent, steps, executor=pool)
        with handler(mock), handler(projection):
            return await _chain(agents)

    return await asyncio.gather(*(as_agent(a) for a in agents), return_exceptions=True)


class TestProjection:
    """`EndpointProjection` on its own, driven without a `Choreography`."""

    def test_agents_outnumbering_worker_threads_still_finish(self):
        """Four agents, one worker thread, a chain in which each waits on the
        previous one. A waiting agent is a suspended coroutine, so it holds no
        thread -- run agent-per-thread instead and this deadlocks."""
        agents = [Coder(__agent_id__=f"coder-{i}") for i in range(4)]
        mock = MockLLM({"implement": lambda skill, args: args[0] + "!"})

        async def go():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return await _drive(agents, _Steps(), mock, pool)

        assert run(go()) == ["x!!!!"] * 4

    def test_a_waiting_agent_sees_the_owners_failure(self):
        """`asyncio.gather` does not cancel siblings when one raises, so a
        waiting agent only learns of a failure if the failed step carries it.

        `Choreography` runs its agents in a task group, which does cancel, so
        this is the mechanism underneath rather than what a run relies on."""
        agents = [Coder(__agent_id__=f"coder-{i}") for i in range(2)]
        mock = FailingMockLLM({}, fail_on={"coder-0.implement"})

        async def go():
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                return await _drive(agents, _Steps(), mock, pool)

        assert [type(o) for o in run(go())] == [RuntimeError, RuntimeError]


@dataclasses.dataclass(frozen=True)
class _Verdict:
    """A non-JSON result, to pin that the log is not limited to JSON types."""

    passed: bool
    note: str


class Judge(Agent):
    """Returns a dataclass."""

    @Skill.define
    def rule(self, case: str) -> _Verdict:
        """Rule on: {case}"""
        raise NotHandled


class TestResume:
    """A run given a log path replays what an earlier run finished."""

    def test_a_second_run_calls_no_model_at_all(self, tmp_path):
        architect, coder = Architect(__agent_id__="arch"), Coder(__agent_id__="coder")
        log = tmp_path / "steps.db"

        first, _ = choreograph(
            _plan_then_implement,
            [architect, coder],
            {"plan": "P", "implement": "C"},
            log=log,
            architect=architect,
            coder=coder,
        )
        second, mock = choreograph(
            _plan_then_implement,
            [architect, coder],
            {},
            mock=FailingMockLLM({}, fail_on={"arch.plan", "coder.implement"}),
            log=log,
            architect=architect,
            coder=coder,
        )

        assert first == second == "C"
        assert mock.calls == []

    def test_resume_picks_up_after_the_last_completed_step(self, tmp_path):
        """The canonical case: a run dies partway through, and the next one
        re-uses what finished and redoes only what didn't."""
        architect, coder = Architect(__agent_id__="arch"), Coder(__agent_id__="coder")
        log = tmp_path / "steps.db"

        failing = Choreography(_plan_then_implement, agents=[architect, coder], log=log)
        with (
            handler(FailingMockLLM({"plan": "P"}, fail_on={"coder.implement"})),
            pytest.raises(ChoreographyError),
        ):
            run(failing.run_async(architect=architect, coder=coder))

        result, mock = choreograph(
            _plan_then_implement,
            [architect, coder],
            {"implement": "C at last"},
            log=log,
            architect=architect,
            coder=coder,
        )

        assert result == "C at last"
        assert mock.calls_for("arch") == []
        assert mock.calls_for("coder") == ["coder.implement"]

    def test_resume_reads_a_log_written_by_another_instance(self, tmp_path):
        """Resumption is across processes, so nothing may be carried in memory
        from the run that wrote the log."""
        architect, coder = Architect(__agent_id__="arch"), Coder(__agent_id__="coder")
        path = tmp_path / "steps.db"

        choreograph(
            _plan_then_implement,
            [architect, coder],
            {"plan": "P", "implement": "C"},
            log=path,
            architect=architect,
            coder=coder,
        )
        result, mock = choreograph(
            _plan_then_implement,
            [architect, coder],
            {},
            mock=FailingMockLLM({}, fail_on={"arch.plan", "coder.implement"}),
            log=path,
            architect=architect,
            coder=coder,
        )

        assert result == "C"
        assert mock.calls == []

    def test_a_scatter_resumes_item_by_item(self, tmp_path):
        """Interrupt a scatter over five modules and the next run implements
        only the ones that never finished."""
        coders = [Coder(__agent_id__=f"coder-{i}") for i in range(2)]
        log = tmp_path / "steps.db"

        async def prior_run():
            steps = _Steps(log)
            for index, value in [(1, "cached b"), (3, "cached d")]:
                steps.resolve(f"step-0000:{index}", value)

        run(prior_run())

        async def program(coder):
            return await scatter(
                list("abcde"), coder, lambda c, item: step(c.implement, item)
            )

        result, mock = choreograph(
            program,
            coders,
            {"implement": lambda skill, args: f"code({args[0]})"},
            log=log,
            coder=coders,
        )

        assert result == ["code(a)", "cached b", "code(c)", "cached d", "code(e)"]
        assert len([c for c in mock.calls if c.endswith(".implement")]) == 3

    def test_a_failed_step_runs_again_rather_than_replaying(self, tmp_path):
        """Nothing is recorded for a step that raised, so the next run has no
        cached answer to reach for and simply retries it."""
        coder = Coder(__agent_id__="coder")
        log = tmp_path / "steps.db"

        async def program(coder):
            return await step(coder.implement, "spec")

        choreo = Choreography(program, agents=[coder], log=log)
        with (
            handler(FailingMockLLM({}, fail_on={"coder.implement"})),
            pytest.raises(ChoreographyError),
        ):
            run(choreo.run_async(coder=coder))

        result, mock = choreograph(
            program, [coder], {"implement": "C"}, log=log, coder=coder
        )
        assert result == "C"
        assert mock.calls == ["coder.implement"]

    def test_a_dataclass_result_survives_a_resume(self, tmp_path):
        """Results are pickled rather than JSON-encoded, so a step may return
        anything a skill can decode to."""
        judge = Judge(__agent_id__="judge")
        coder = Coder(__agent_id__="coder")
        log = tmp_path / "steps.db"
        verdict = _Verdict(passed=False, note="needs work")

        async def program(judge, coder):
            ruling = await step(judge.rule, "the case")
            assert ruling == verdict
            return await step(coder.implement, ruling.note)

        choreograph(
            program,
            [judge, coder],
            {"rule": verdict, "implement": "C"},
            log=log,
            judge=judge,
            coder=coder,
        )
        # The second run gets its ruling from the log, dataclass and all.
        result, mock = choreograph(
            program,
            [judge, coder],
            {},
            mock=FailingMockLLM({}, fail_on={"judge.rule", "coder.implement"}),
            log=log,
            judge=judge,
            coder=coder,
        )
        assert result == "C"
        assert mock.calls == []

    def test_a_log_in_a_directory_that_does_not_exist_yet(self, tmp_path):
        architect, coder = Architect(__agent_id__="arch"), Coder(__agent_id__="coder")
        log = tmp_path / "nested" / "deeper" / "steps.db"

        result, _ = choreograph(
            _plan_then_implement,
            [architect, coder],
            {"plan": "P", "implement": "C"},
            log=log,
            architect=architect,
            coder=coder,
        )
        assert result == "C"
        assert log.exists()

    def test_deleting_the_log_starts_over(self, tmp_path):
        architect, coder = Architect(__agent_id__="arch"), Coder(__agent_id__="coder")
        log = tmp_path / "steps.db"

        choreograph(
            _plan_then_implement,
            [architect, coder],
            {"plan": "P", "implement": "C"},
            log=log,
            architect=architect,
            coder=coder,
        )
        log.unlink()
        _, mock = choreograph(
            _plan_then_implement,
            [architect, coder],
            {"plan": "P2", "implement": "C2"},
            log=log,
            architect=architect,
            coder=coder,
        )

        assert sorted(mock.calls) == ["arch.plan", "coder.implement"]

    def test_a_step_result_of_none_replays_as_none(self, tmp_path):
        """A recorded ``None`` has to come back as a completed step, not as a
        step with nothing recorded for it."""
        architect, coder = Architect(__agent_id__="arch"), Coder(__agent_id__="coder")
        log = tmp_path / "steps.db"

        async def program(architect, coder):
            plan = await step(architect.plan, "spec")
            assert plan is None
            return await step(coder.implement, "go")

        choreograph(
            program,
            [architect, coder],
            {"plan": None, "implement": "C"},
            log=log,
            architect=architect,
            coder=coder,
        )
        result, mock = choreograph(
            program,
            [architect, coder],
            {},
            mock=FailingMockLLM({}, fail_on={"arch.plan", "coder.implement"}),
            log=log,
            architect=architect,
            coder=coder,
        )

        assert result == "C"
        assert mock.calls == []


class TestPrimitivesAreOperations:
    """`step`, `call` and `scatter` are ordinary `Operation`s, so they have
    default rules outside a choreography and can be handled like anything
    else inside one."""

    def test_step_outside_a_choreography_just_calls(self):
        bot = Coder(__agent_id__="solo")

        async def go():
            with handler(MockLLM({"implement": "code"})):
                return await step(bot.implement, "spec")

        assert run(go()) == "code"

    def test_a_step_handler_can_forward(self):
        """`fwd` works from a handler that returns an awaitable rather than
        being a coroutine function, which is the shape every implementation of
        these operations has to take -- `effectful` binds `fwd` around the
        synchronous call, so a deferred body would run after it was gone."""
        bot = Coder(__agent_id__="solo")
        seen: list[str] = []

        class Tracer(ObjectInterpretation):
            @implements(step)
            def _step(self, skill, *args, **kwargs):
                pending = fwd()

                async def traced():
                    result = await pending
                    seen.append(f"{skill.__name__} -> {result}")
                    return result

                return traced()

        async def go():
            with handler(MockLLM({"implement": "code"})), handler(Tracer()):
                return await step(bot.implement, "spec")

        assert run(go()) == "code"
        assert seen == ["implement -> code"]

    def test_scatter_outside_a_choreography_is_sequential(self):
        coders = [Coder(__agent_id__="coder-1"), Coder(__agent_id__="coder-2")]
        seen: list[str] = []

        async def fake_call(agent, item):
            seen.append(f"{agent.__agent_id__}:{item}")
            return item.upper()

        assert run(scatter(["a", "b", "c"], coders, fake_call)) == ["A", "B", "C"]
        assert seen == ["coder-1:a", "coder-2:b", "coder-1:c"]


class TestHandlerPropagation:
    """`effectful`'s interpretation lives in a `contextvars.ContextVar`, which
    is what makes this design work: each agent task gets its own copy, and
    each blocking call carries that copy into its worker thread."""

    def test_handlers_installed_around_run_reach_the_worker_threads(self):
        """Handlers do not have to be passed to `Choreography` -- anything
        installed around the call (as `effectful.handlers.llm.harness` does)
        is inherited by every agent task and by the threads they call into."""
        architect, coder = Architect(__agent_id__="arch"), Coder(__agent_id__="coder")
        mock = MockLLM({"plan": "P", "implement": "C"})
        choreo = Choreography(_plan_then_implement, agents=[architect, coder])

        with handler(mock):
            assert run(choreo.run_async(architect=architect, coder=coder)) == "C"
        assert sorted(mock.calls) == ["arch.plan", "coder.implement"]
