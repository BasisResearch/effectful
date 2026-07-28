"""Tests for `effectful.handlers.llm.multi` -- choreographic EPP with a TaskQueue.

No real LLM is involved: `MockLLM` and friends implement `Template.__apply__`
directly, so what is under test is the choreography -- step allocation, claim
based distribution, crash recovery -- rather than any completion logic.

Every test runs under a timeout, because the failure mode of a concurrency bug
here is a hang rather than a wrong answer.
"""

import asyncio
import concurrent.futures
import sqlite3
import threading
from typing import Any

import pytest

from effectful.handlers.llm import Agent, Template
from effectful.handlers.llm.multi import (
    MISSING,
    Choreography,
    ChoreographyError,
    InMemoryTaskQueue,
    PersistentTaskQueue,
    TaskStatus,
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


# ── Queue tests ───────────────────────────────────────────────────


@pytest.fixture(params=["in_memory", "persistent"])
def queue(request, tmp_path):
    q = (
        InMemoryTaskQueue()
        if request.param == "in_memory"
        else PersistentTaskQueue(tmp_path / "queue.db")
    )
    yield q
    run(q.close())


class TestTaskQueue:
    def test_submit_and_claim(self, queue):
        async def go():
            await queue.submit("work", {"n": 1}, task_id="t1")
            task = await queue.claim_id("t1", "worker-1")
            assert task is not None
            assert task["id"] == "t1"
            assert task["payload"] == {"n": 1}
            assert task["owner"] == "worker-1"
            # A claimed task cannot be claimed again.
            assert await queue.claim_id("t1", "worker-2") is None

        run(go())

    def test_claim_missing_task(self, queue):
        assert run(queue.claim_id("nope", "worker-1")) is None

    def test_submit_is_idempotent(self, queue):
        async def go():
            await queue.submit("work", {"n": 1}, task_id="t1")
            await queue.submit("work", {"n": 2}, task_id="t1")
            task = await queue.claim_id("t1", "worker-1")
            assert task["payload"] == {"n": 1}
            assert await queue.pending_count() == 0

        run(go())

    def test_get_result_is_missing_until_done(self, queue):
        async def go():
            await queue.submit("work", {}, task_id="t1")
            assert await queue.get_result("t1") is MISSING
            await queue.claim_id("t1", "w")
            assert await queue.get_result("t1") is MISSING
            await queue.complete("t1", "w", {"answer": 42})
            assert await queue.get_result("t1") == {"answer": 42}

        run(go())

    @pytest.mark.parametrize("result", [None, 0, "", [], False])
    def test_falsy_results_are_distinguishable_from_absence(self, queue, result):
        """A completed step whose result is falsy -- ``None`` above all -- must
        not read back as 'not done yet', or a poll loop waits forever."""

        async def go():
            await queue.submit("work", {}, task_id="t1")
            await queue.claim_id("t1", "w")
            await queue.complete("t1", "w", result)
            assert await queue.get_result("t1") == result
            assert await queue.get_result("t1") is not MISSING
            assert await queue.get_result("absent") is MISSING

        run(go())

    def test_claim_prefix_takes_lowest_id_first(self, queue):
        async def go():
            for i in (2, 0, 1):
                await queue.submit("work", {"i": i}, task_id=f"s:{i:04d}")
            await queue.submit("work", {}, task_id="other:0000")

            claimed = []
            while (task := await queue.claim_prefix("s:", "w")) is not None:
                claimed.append(task["id"])
            assert claimed == ["s:0000", "s:0001", "s:0002"]
            # The unrelated prefix is untouched.
            assert await queue.claim_id("other:0000", "w") is not None

        run(go())

    def test_release_stale_claims_releases_claimed(self, queue):
        async def go():
            await queue.submit("work", {}, task_id="t1")
            await queue.claim_id("t1", "w1")
            assert await queue.release_stale_claims("w2") == 0
            assert await queue.release_stale_claims("w1") == 1
            assert await queue.pending_count() == 1
            assert await queue.claim_id("t1", "w2") is not None

        run(go())

    def test_release_stale_claims_releases_failed(self, queue):
        """A failed task has to go back to pending on restart. Left failed, it
        is a step whose result never arrives and whose owner never retries."""

        async def go():
            await queue.submit("work", {}, task_id="t1")
            await queue.claim_id("t1", "w1")
            await queue.fail("t1", "w1", "boom")
            assert await queue.release_stale_claims("w1") == 1
            assert await queue.pending_count() == 1
            assert await queue.get_result("t1") is MISSING
            assert await queue.claim_id("t1", "w1") is not None

        run(go())

    def test_release_stale_claims_leaves_done_alone(self, queue):
        async def go():
            await queue.submit("work", {}, task_id="t1")
            await queue.claim_id("t1", "w1")
            await queue.complete("t1", "w1", "value")
            assert await queue.release_stale_claims("w1") == 0
            assert await queue.get_result("t1") == "value"

        run(go())

    def test_all_done_and_pending_count(self, queue):
        async def go():
            assert await queue.all_done()
            await queue.submit("work", {}, task_id="t1")
            assert await queue.pending_count() == 1
            assert not await queue.all_done()
            await queue.claim_id("t1", "w")
            assert await queue.pending_count() == 0
            assert not await queue.all_done()
            await queue.complete("t1", "w", "x")
            assert await queue.all_done()

        run(go())

    def test_complete_requires_a_claim(self, queue):
        async def go():
            await queue.submit("work", {}, task_id="t1")
            await queue.complete("t1", "w", "value")  # never claimed
            assert await queue.get_result("t1") is MISSING

        run(go())


class TestPersistentTaskQueue:
    def test_wal_mode_enabled(self, tmp_path):
        q = PersistentTaskQueue(tmp_path / "q.db")
        try:
            with sqlite3.connect(q.db_path) as conn:
                assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        finally:
            run(q.close())

    def test_state_survives_a_new_instance(self, tmp_path):
        db = tmp_path / "q.db"
        first = PersistentTaskQueue(db)

        async def go():
            await first.submit("work", {"n": 7}, task_id="t1")
            await first.claim_id("t1", "w")
            await first.complete("t1", "w", "done")

        run(go())
        run(first.close())

        second = PersistentTaskQueue(db)
        try:
            assert run(second.get_result("t1")) == "done"
        finally:
            run(second.close())

    def test_concurrent_claims_never_double_claim(self, tmp_path):
        """Two queue objects on one database stand in for two processes.

        The claim is a read followed by a write; without a transaction that
        spans both, two claimers can read the same pending row and both take
        it. An in-process lock would not catch this -- these are separate
        connections on separate threads.
        """
        db = tmp_path / "q.db"
        left, right = PersistentTaskQueue(db), PersistentTaskQueue(db)
        n = 25

        async def drain(q: PersistentTaskQueue, owner: str) -> list[str]:
            claimed = []
            while (task := await q.claim_prefix("s:", owner)) is not None:
                claimed.append(task["id"])
                await asyncio.sleep(0)  # interleave with the other claimer
            return claimed

        async def go():
            for i in range(n):
                await left.submit("work", {"i": i}, task_id=f"s:{i:04d}")
            return await asyncio.gather(drain(left, "a"), drain(right, "b"))

        try:
            a, b = run(go())
        finally:
            run(left.close())
            run(right.close())

        assert sorted(a + b) == [f"s:{i:04d}" for i in range(n)]
        assert not set(a) & set(b)


# ── Choreography tests ────────────────────────────────────────────


def choreograph(program, agents, responses, *, queue=None, mock=None, **kwargs):
    """Run *program* over *agents* with a mock LLM; return ``(result, mock)``."""
    mock = mock if mock is not None else MockLLM(responses)
    choreo = Choreography(
        program,
        agents=agents,
        queue=queue if queue is not None else InMemoryTaskQueue(),
        handlers=[mock],
        poll_interval=0.01,
    )
    return run(choreo.run_async(**kwargs)), mock


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
        """``None`` is a result like any other. Reading it back as 'not done'
        would leave every other agent polling forever."""
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

    def test_agent_failure_becomes_a_choreography_error(self):
        architect, coder = Architect(agent_id="arch"), Coder(agent_id="coder")

        async def program(architect, coder):
            plan = await step(architect.plan, "spec")
            return await step(coder.implement, plan)

        choreo = Choreography(
            program,
            agents=[architect, coder],
            queue=InMemoryTaskQueue(),
            handlers=[FailingMockLLM({"plan": "P"}, fail_on={"coder.implement"})],
            poll_interval=0.01,
        )

        with pytest.raises(ChoreographyError, match="'coder' failed"):
            run(choreo.run_async(architect=architect, coder=coder))

    def test_repeated_runs_are_deterministic(self):
        architect, coder = Architect(agent_id="arch"), Coder(agent_id="coder")

        async def program(architect, coder):
            plan = await step(architect.plan, "spec")
            return await step(coder.implement, plan)

        results = [
            choreograph(
                program,
                [architect, coder],
                {"plan": "P", "implement": "C"},
                architect=architect,
                coder=coder,
            )[0]
            for _ in range(5)
        ]
        assert results == ["C"] * 5


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

        If one coder claimed both items the other would never arrive and the
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
        """`asyncio.gather` over several scatters is what `fan_out` used to be:
        agents in different groups work at the same time."""
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

        choreo = Choreography(
            program,
            agents=[coder, reviewer],
            queue=InMemoryTaskQueue(),
            handlers=[MockLLM({})],
            poll_interval=0.01,
        )

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
            queue=InMemoryTaskQueue(),
            handlers=[FailingMockLLM({}, fail_on={"coder-1.implement"})],
            poll_interval=0.01,
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

        async def program(architect, coder):
            plan = await step(architect.plan, "spec")
            return await step(coder.implement, plan)

        choreo = Choreography(
            program,
            agents=[architect, coder],
            queue=InMemoryTaskQueue(),
            handlers=[],
            poll_interval=0.01,
        )

        with handler(mock):
            assert run(choreo.run_async(architect=architect, coder=coder)) == "C"
        assert sorted(mock.calls) == ["arch.plan", "coder.implement"]

    def test_agents_outnumbering_worker_threads_still_finish(self):
        """Four agents, one worker thread, a chain in which each waits on the
        previous one. A waiting agent is a suspended coroutine, so it holds no
        thread -- run agent-per-thread instead and this deadlocks."""
        agents = [Coder(agent_id=f"coder-{i}") for i in range(4)]
        mock = MockLLM({"implement": lambda template, args: args[0] + "!"})
        choreo = Choreography(
            _chain, agents=agents, queue=InMemoryTaskQueue(), poll_interval=0.01
        )

        async def drive(agent, pool):
            projection = choreo.projection(agent, executor=pool)
            with handler(mock), projection.activate():
                return await _chain(agents)

        async def go():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return await asyncio.gather(*(drive(a, pool) for a in agents))

        assert run(go()) == ["x!!!!"] * 4


async def _chain(coders):
    value = "x"
    for coder in coders:
        value = await step(coder.implement, value)
    return value


class TestCrashRecovery:
    """Restart semantics: the program re-runs from the top, and steps that
    already finished return their recorded result instead of calling a model."""

    def _program(self):
        async def program(architect, coder):
            plan = await step(architect.plan, "spec")
            return await step(coder.implement, plan)

        return program

    def test_completed_steps_are_not_re_executed(self):
        architect, coder = Architect(agent_id="arch"), Coder(agent_id="coder")
        queue = InMemoryTaskQueue()

        async def prior_run():
            await queue.submit("plan", {"agent": "arch"}, task_id="step-0000")
            await queue.claim_id("step-0000", "arch")
            await queue.complete("step-0000", "arch", "cached plan")

        run(prior_run())

        result, mock = choreograph(
            self._program(),
            [architect, coder],
            {"plan": "SHOULD NOT RUN", "implement": "fresh code"},
            queue=queue,
            architect=architect,
            coder=coder,
        )

        assert result == "fresh code"
        assert mock.calls_for("arch") == []
        assert mock.calls_for("coder") == ["coder.implement"]

    def test_a_second_run_calls_no_model_at_all(self):
        architect, coder = Architect(agent_id="arch"), Coder(agent_id="coder")
        queue = InMemoryTaskQueue()
        program = self._program()

        first, _ = choreograph(
            program,
            [architect, coder],
            {"plan": "P", "implement": "C"},
            queue=queue,
            architect=architect,
            coder=coder,
        )
        second, mock = choreograph(
            program,
            [architect, coder],
            {},
            queue=queue,
            mock=FailingMockLLM({}, fail_on={"arch.plan", "coder.implement"}),
            architect=architect,
            coder=coder,
        )

        assert first == second == "C"
        assert mock.calls == []

    def test_a_failed_step_is_retried_on_restart(self):
        """A failure aborts the run with the step marked failed. The next run
        has to release it -- otherwise the step is neither done nor claimable
        and every agent waits for a result that cannot arrive."""
        architect, coder = Architect(agent_id="arch"), Coder(agent_id="coder")
        queue = InMemoryTaskQueue()
        program = self._program()

        failing = Choreography(
            program,
            agents=[architect, coder],
            queue=queue,
            handlers=[FailingMockLLM({"plan": "P"}, fail_on={"coder.implement"})],
            poll_interval=0.01,
        )
        with pytest.raises(ChoreographyError):
            run(failing.run_async(architect=architect, coder=coder))
        assert run(queue.get_result("step-0001")) is MISSING

        result, mock = choreograph(
            program,
            [architect, coder],
            {"implement": "code at last"},
            queue=queue,
            architect=architect,
            coder=coder,
        )

        assert result == "code at last"
        # The architect's step survived the failed run and was not re-run.
        assert mock.calls_for("arch") == []

    def test_scatter_resumes_from_partial_results(self):
        coders = [Coder(agent_id="coder-1"), Coder(agent_id="coder-2")]
        queue = InMemoryTaskQueue()

        async def program(coder):
            return await scatter(
                ["a", "b", "c"], coder, lambda c, item: call(c.implement, item)
            )

        async def prior_run():
            await queue.submit("scatter-step-0000", {"item_index": 1}, "step-0000:0001")
            await queue.claim_id("step-0000:0001", "coder-1")
            await queue.complete("step-0000:0001", "coder-1", "cached b")

        run(prior_run())

        result, mock = choreograph(
            program,
            coders,
            {"implement": lambda template, args: f"code({args[0]})"},
            queue=queue,
            coder=coders,
        )

        assert result == ["code(a)", "cached b", "code(c)"]
        assert len([c for c in mock.calls if c.endswith(".implement")]) == 2

    def test_restart_with_a_persistent_queue(self, tmp_path):
        architect, coder = Architect(agent_id="arch"), Coder(agent_id="coder")
        program = self._program()
        db = tmp_path / "queue.db"

        first = PersistentTaskQueue(db)
        result, _ = choreograph(
            program,
            [architect, coder],
            {"plan": "P", "implement": "C"},
            queue=first,
            architect=architect,
            coder=coder,
        )
        run(first.close())
        assert result == "C"

        # A fresh process: new queue object, same database.
        second = PersistentTaskQueue(db)
        try:
            resumed, mock = choreograph(
                program,
                [architect, coder],
                {},
                queue=second,
                mock=FailingMockLLM({}, fail_on={"arch.plan", "coder.implement"}),
                architect=architect,
                coder=coder,
            )
        finally:
            run(second.close())

        assert resumed == "C"
        assert mock.calls == []

    def test_stale_claims_are_released_on_restart(self, tmp_path):
        architect, coder = Architect(agent_id="arch"), Coder(agent_id="coder")
        queue = PersistentTaskQueue(tmp_path / "queue.db")

        async def crashed_run():
            # A previous process claimed step 0 and died before finishing it.
            await queue.submit("plan", {"agent": "arch"}, task_id="step-0000")
            await queue.claim_id("step-0000", "arch")

        run(crashed_run())

        try:
            result, mock = choreograph(
                self._program(),
                [architect, coder],
                {"plan": "P", "implement": "C"},
                queue=queue,
                architect=architect,
                coder=coder,
            )
            assert run(queue.all_done())
        finally:
            run(queue.close())

        assert result == "C"
        assert mock.calls_for("arch") == ["arch.plan"]

    def test_queue_records_final_state(self, tmp_path):
        architect, coder = Architect(agent_id="arch"), Coder(agent_id="coder")
        queue = PersistentTaskQueue(tmp_path / "queue.db")
        try:
            choreograph(
                self._program(),
                [architect, coder],
                {"plan": "P", "implement": "C"},
                queue=queue,
                architect=architect,
                coder=coder,
            )
            with sqlite3.connect(queue.db_path) as conn:
                rows = conn.execute("SELECT id, status, owner FROM tasks").fetchall()
        finally:
            run(queue.close())

        assert sorted(rows) == [
            ("step-0000", TaskStatus.DONE, "arch"),
            ("step-0001", TaskStatus.DONE, "coder"),
        ]
