"""Tests for effectful.handlers.llm.multi — choreographic EPP with TaskQueue."""

import itertools
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from effectful.handlers.llm import Template
from effectful.handlers.llm.multi import (
    Choreography,
    ChoreographyError,
    EndpointProjection,
    TaskQueue,
    TaskStatus,
    scatter,
)
from effectful.handlers.llm.persistence import PersistentAgent
from effectful.handlers.llm.template import get_bound_agent
from effectful.ops.semantics import handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import NotHandled

# ── Fixtures and helpers ──────────────────────────────────────────

STATE_DIR = Path("/tmp/test_multi_epp")

# Default timeout for all concurrent tests (seconds).
# Concurrency bugs often manifest as hangs — this catches them.
THREAD_TIMEOUT = 10


@pytest.fixture(autouse=True)
def clean_state():
    if STATE_DIR.exists():
        shutil.rmtree(STATE_DIR)
    STATE_DIR.mkdir(parents=True)
    yield
    if STATE_DIR.exists():
        shutil.rmtree(STATE_DIR)


class MockLLM(ObjectInterpretation):
    """Mock LLM handler that returns canned responses."""

    def __init__(self, responses: dict[str, Any]):
        self._responses = responses
        self.calls: list[str] = []

    @implements(Template.__apply__)
    def _call(self, template, *args, **kwargs):
        bound = get_bound_agent(template)
        key = (
            f"{bound.__agent_id__}.{template.__name__}" if bound else template.__name__
        )
        self.calls.append(key)
        return self._responses.get(
            key, self._responses.get(template.__name__, f"mock-{template.__name__}")
        )


class DelayedMockLLM(ObjectInterpretation):
    """Mock LLM that introduces per-agent delays to force scheduling orderings.

    ``delays`` maps agent_id to a sleep duration (seconds) applied before
    each template call for that agent.  This lets tests deterministically
    force one thread to run before another.
    """

    def __init__(self, responses: dict[str, Any], delays: dict[str, float]):
        self._responses = responses
        self._delays = delays
        self.calls: list[str] = []
        self._lock = threading.Lock()

    @implements(Template.__apply__)
    def _call(self, template, *args, **kwargs):
        bound = get_bound_agent(template)
        agent_id = bound.__agent_id__ if bound else None
        if agent_id and agent_id in self._delays:
            time.sleep(self._delays[agent_id])
        key = (
            f"{bound.__agent_id__}.{template.__name__}" if bound else template.__name__
        )
        with self._lock:
            self.calls.append(key)
        return self._responses.get(
            key, self._responses.get(template.__name__, f"mock-{template.__name__}")
        )


class FailingMockLLM(ObjectInterpretation):
    """Mock LLM that raises on specific agent.template keys."""

    def __init__(self, responses: dict[str, Any], fail_on: set[str]):
        self._responses = responses
        self._fail_on = fail_on

    @implements(Template.__apply__)
    def _call(self, template, *args, **kwargs):
        bound = get_bound_agent(template)
        key = (
            f"{bound.__agent_id__}.{template.__name__}" if bound else template.__name__
        )
        if key in self._fail_on:
            raise RuntimeError(f"Simulated failure on {key}")
        return self._responses.get(
            key, self._responses.get(template.__name__, f"mock-{template.__name__}")
        )


def _run_threads_with_timeout(targets, timeout=THREAD_TIMEOUT):
    """Start threads and join with timeout.  Raises if any thread hangs."""
    threads = [threading.Thread(target=t, daemon=True) for t in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)
        if t.is_alive():
            raise TimeoutError(
                f"Thread {t.name} did not finish within {timeout}s — "
                "possible deadlock or infinite poll"
            )


# ── Agent definitions ─────────────────────────────────────────────


class Architect(PersistentAgent):
    """Plans modules."""

    @Template.define
    def plan(self, spec: str) -> str:
        """Plan modules for: {spec}"""
        raise NotHandled


class Coder(PersistentAgent):
    """Writes code."""

    @Template.define
    def implement(self, spec: str) -> str:
        """Implement: {spec}"""
        raise NotHandled


class Reviewer(PersistentAgent):
    """Reviews code."""

    @Template.define
    def review(self, code: str) -> str:
        """Review: {code}"""
        raise NotHandled


# ── TaskQueue tests ───────────────────────────────────────────────


class TestTaskQueue:
    def test_submit_and_claim(self):
        tq = TaskQueue(STATE_DIR / "q")
        tid = tq.submit("code", {"file": "main.py"}, task_id="t1")
        assert tid == "t1"

        task = tq.claim("code", "worker1")
        assert task is not None
        assert task["id"] == "t1"
        assert task["status"] == TaskStatus.CLAIMED

        # Can't claim again
        assert tq.claim("code", "worker2") is None

    def test_idempotent_submit(self):
        tq = TaskQueue(STATE_DIR / "q")
        tq.submit("code", {}, task_id="t1")
        tq.submit("code", {}, task_id="t1")  # no-op
        assert tq.pending_count() == 1

    def test_complete_and_get_result(self):
        tq = TaskQueue(STATE_DIR / "q")
        tq.submit("code", {}, task_id="t1")
        tq.claim("code", "w1")
        tq.complete("t1", "w1", {"output": "hello"})
        assert tq.get_result("t1") == {"output": "hello"}

    def test_release_stale_claims(self):
        tq = TaskQueue(STATE_DIR / "q")
        tq.submit("code", {}, task_id="t1")
        tq.claim("code", "crashed_worker")
        assert tq.pending_count() == 0

        released = tq.release_stale_claims("crashed_worker")
        assert released == 1
        assert tq.pending_count() == 1

        # Can re-claim
        task = tq.claim("code", "new_worker")
        assert task is not None

    def test_claim_by_prefix(self):
        tq = TaskQueue(STATE_DIR / "q")
        tq.submit("scatter", {}, task_id="step-0001:0000")
        tq.submit("scatter", {}, task_id="step-0001:0001")
        tq.submit("other", {}, task_id="step-0002")

        task = tq.claim_by_prefix("step-0001:", "w1")
        assert task is not None
        assert task["id"].startswith("step-0001:")

        task2 = tq.claim_by_prefix("step-0001:", "w1")
        assert task2 is not None
        assert task2["id"] != task["id"]

        assert tq.claim_by_prefix("step-0001:", "w1") is None


# ── EPP tests ─────────────────────────────────────────────────────


class TestEndpointProjection:
    def _run_choreo(
        self,
        agents,
        choreo_fn,
        responses,
        *,
        mock_cls=None,
        timeout=THREAD_TIMEOUT,
        **kwargs,
    ):
        """Helper: run a choreography with mock LLM.

        *mock_cls* can be a callable ``(responses) -> ObjectInterpretation``
        to inject custom mock behaviour (e.g. delays).
        """
        tq = TaskQueue(STATE_DIR / "q")
        ids = frozenset(a.__agent_id__ for a in agents)
        results: dict[str, Any] = {}
        llm_calls: dict[str, list[str]] = {}
        errors: list[tuple[str, Exception]] = []
        lock = threading.Lock()

        def run_agent(agent):
            try:
                mock = mock_cls(responses) if mock_cls else MockLLM(responses)
                tq.release_stale_claims(agent.__agent_id__)
                epp = EndpointProjection(agent, tq, ids, poll_interval=0.02)
                with handler(mock), handler(epp):
                    r = choreo_fn(**kwargs)
                    with lock:
                        results[agent.__agent_id__] = r
                        llm_calls[agent.__agent_id__] = list(
                            mock.calls if hasattr(mock, "calls") else []
                        )
            except Exception as e:
                with lock:
                    errors.append((agent.__agent_id__, e))

        _run_threads_with_timeout(
            [lambda a=a: run_agent(a) for a in agents],
            timeout=timeout,
        )

        return results, llm_calls, errors

    def test_basic_sequential(self):
        """Two agents: planner plans, worker executes."""
        planner = Architect(agent_id="arch")
        worker = Coder(agent_id="coder")

        def choreo(spec, arch, coder):
            plan = arch.plan(spec)
            return coder.implement(plan)

        results, calls, errors = self._run_choreo(
            [planner, worker],
            choreo,
            {"plan": "the plan", "implement": "code"},
            spec="build it",
            arch=planner,
            coder=worker,
        )

        assert not errors, errors
        assert results["arch"] == "code"
        assert results["coder"] == "code"
        # Planner executed plan, worker executed implement
        assert "arch.plan" in calls["arch"]
        assert "coder.implement" in calls["coder"]

    def test_all_agents_same_result(self):
        """All agent threads produce the same result."""
        arch = Architect(agent_id="a")
        coder = Coder(agent_id="c")
        reviewer = Reviewer(agent_id="r")

        def choreo(arch, coder, reviewer):
            plan = arch.plan("spec")
            code = coder.implement(plan)
            return reviewer.review(code)

        results, _, errors = self._run_choreo(
            [arch, coder, reviewer],
            choreo,
            {"plan": "plan", "implement": "code", "review": "PASS"},
            arch=arch,
            coder=coder,
            reviewer=reviewer,
        )

        assert not errors
        assert results["a"] == results["c"] == results["r"] == "PASS"

    def test_while_loop(self):
        """Control flow: reviewer retries, then passes."""
        arch = Architect(agent_id="arch")
        coder = Coder(agent_id="coder")
        reviewer = Reviewer(agent_id="rev")

        review_count = {"n": 0}

        class LoopMock(ObjectInterpretation):
            def __init__(self):
                self.calls: list[str] = []

            @implements(Template.__apply__)
            def _call(self, template, *args, **kwargs):
                self.calls.append(template.__name__)
                if template.__name__ == "plan":
                    return "plan"
                elif template.__name__ == "implement":
                    return f"code-v{len(self.calls)}"
                elif template.__name__ == "review":
                    review_count["n"] += 1
                    return "RETRY" if review_count["n"] <= 1 else "PASS"
                return "?"

        def choreo(arch, coder, reviewer):
            plan = arch.plan("spec")
            code = coder.implement(plan)
            while True:
                verdict = reviewer.review(code)
                if verdict == "PASS":
                    return code
                code = coder.implement(verdict)

        tq = TaskQueue(STATE_DIR / "q")
        ids = frozenset(["arch", "coder", "rev"])
        results: dict[str, Any] = {}
        errors: list = []
        lock = threading.Lock()

        def run(agent):
            try:
                mock = LoopMock()
                epp = EndpointProjection(agent, tq, ids, poll_interval=0.02)
                with handler(mock), handler(epp):
                    r = choreo(arch=arch, coder=coder, reviewer=reviewer)
                    with lock:
                        results[agent.__agent_id__] = r
            except Exception as e:
                with lock:
                    errors.append(e)

        _run_threads_with_timeout([lambda a=a: run(a) for a in [arch, coder, reviewer]])

        assert not errors, errors
        assert all(r == results["arch"] for r in results.values())
        # Should have 5 persisted steps: plan, implement, review, implement, review
        done_files = list(tq.queue_dir.glob(f"*.{TaskStatus.DONE}.json"))
        assert len(done_files) == 5

    def test_crash_recovery(self):
        """Pre-cache step 0, restart: step 0 from cache, step 1 fresh."""
        tq = TaskQueue(STATE_DIR / "q")
        arch = Architect(agent_id="arch")
        coder = Coder(agent_id="coder")

        # Simulate prior run: step 0 done
        tq.submit("plan", {"agent": "arch"}, task_id="step-0000")
        tq.claim("plan", "arch")
        tq.complete("step-0000", "arch", "cached-plan")

        def choreo(arch, coder):
            plan = arch.plan("spec")
            return coder.implement(plan)

        ids = frozenset(["arch", "coder"])
        results: dict[str, Any] = {}
        llm_calls: dict[str, list[str]] = {}
        lock = threading.Lock()
        errors: list = []

        def run(agent):
            try:
                mock = MockLLM({"plan": "SHOULD NOT RUN", "implement": "fresh-code"})
                tq.release_stale_claims(agent.__agent_id__)
                epp = EndpointProjection(agent, tq, ids, poll_interval=0.02)
                with handler(mock), handler(epp):
                    r = choreo(arch=arch, coder=coder)
                    with lock:
                        results[agent.__agent_id__] = r
                        llm_calls[agent.__agent_id__] = list(mock.calls)
            except Exception as e:
                with lock:
                    errors.append(e)

        _run_threads_with_timeout([lambda a=a: run(a) for a in [arch, coder]])

        assert not errors, errors
        assert results["arch"] == "fresh-code"
        assert results["coder"] == "fresh-code"
        # arch should NOT have called LLM for plan
        assert "arch.plan" not in llm_calls.get("arch", [])
        # coder should have called implement
        assert "coder.implement" in llm_calls.get("coder", [])


# ── Ordering permutation tests ────────────────────────────────────


class TestOrderingPermutations:
    """Run choreographies under every possible thread-scheduling order.

    Uses controlled delays to deterministically force one agent to
    execute before another.  For N agents there are N! orderings;
    all must produce the same result.
    """

    def _run_with_ordering(self, agents, choreo_fn, responses, ordering, **kwargs):
        """Run *choreo_fn* with agents delayed so they execute in *ordering*."""
        # Give each agent a staggered delay: first in ordering gets 0,
        # second gets a small delay, etc.
        delays = {agent.__agent_id__: i * 0.03 for i, agent in enumerate(ordering)}
        tq = TaskQueue(STATE_DIR / "q")
        ids = frozenset(a.__agent_id__ for a in agents)
        results: dict[str, Any] = {}
        errors: list[tuple[str, Exception]] = []
        lock = threading.Lock()

        def run_agent(agent):
            try:
                mock = DelayedMockLLM(responses, delays)
                tq.release_stale_claims(agent.__agent_id__)
                epp = EndpointProjection(agent, tq, ids, poll_interval=0.02)
                with handler(mock), handler(epp):
                    r = choreo_fn(**kwargs)
                    with lock:
                        results[agent.__agent_id__] = r
            except Exception as e:
                with lock:
                    errors.append((agent.__agent_id__, e))

        _run_threads_with_timeout([lambda a=a: run_agent(a) for a in agents])
        return results, errors

    def test_two_agent_all_orderings(self):
        """Two agents, two orderings — both must agree on the result."""
        arch = Architect(agent_id="arch")
        coder = Coder(agent_id="coder")
        agents = [arch, coder]
        responses = {"plan": "the-plan", "implement": "the-code"}

        def choreo(arch, coder):
            plan = arch.plan("spec")
            return coder.implement(plan)

        all_results = []
        for perm in itertools.permutations(agents):
            # Each permutation needs a fresh queue directory
            q_dir = STATE_DIR / "q"
            if q_dir.exists():
                shutil.rmtree(q_dir)
            results, errors = self._run_with_ordering(
                agents,
                choreo,
                responses,
                perm,
                arch=arch,
                coder=coder,
            )
            assert not errors, f"Ordering {[a.__agent_id__ for a in perm]}: {errors}"
            all_results.append(results)

        # All orderings must produce the same result for every agent
        for r in all_results:
            assert r["arch"] == r["coder"] == "the-code"

    def test_three_agent_all_orderings(self):
        """Three agents, six orderings — all must agree."""
        arch = Architect(agent_id="a")
        coder = Coder(agent_id="c")
        reviewer = Reviewer(agent_id="r")
        agents = [arch, coder, reviewer]
        responses = {"plan": "plan", "implement": "code", "review": "PASS"}

        def choreo(arch, coder, reviewer):
            plan = arch.plan("spec")
            code = coder.implement(plan)
            return reviewer.review(code)

        expected = "PASS"
        for perm in itertools.permutations(agents):
            q_dir = STATE_DIR / "q"
            if q_dir.exists():
                shutil.rmtree(q_dir)
            results, errors = self._run_with_ordering(
                agents,
                choreo,
                responses,
                perm,
                arch=arch,
                coder=coder,
                reviewer=reviewer,
            )
            assert not errors, f"Ordering {[a.__agent_id__ for a in perm]}: {errors}"
            for aid, r in results.items():
                assert r == expected, (
                    f"Agent {aid} got {r!r} != {expected!r} "
                    f"with ordering {[a.__agent_id__ for a in perm]}"
                )

    def test_scatter_all_orderings(self):
        """Scatter with two coders + reviewer, all orderings agree."""
        c1 = Coder(agent_id="c1")
        c2 = Coder(agent_id="c2")
        rev = Reviewer(agent_id="rev")
        agents = [c1, c2, rev]
        responses = {"implement": "code", "review": "ok"}

        def choreo(items, coders, reviewer):
            codes = scatter(items, coders, lambda c, m: c.implement(m))
            return [reviewer.review(c) for c in codes]

        for perm in itertools.permutations(agents):
            q_dir = STATE_DIR / "q"
            if q_dir.exists():
                shutil.rmtree(q_dir)
            results, errors = self._run_with_ordering(
                agents,
                choreo,
                responses,
                perm,
                items=["A", "B"],
                coders=[c1, c2],
                reviewer=rev,
            )
            assert not errors, f"Ordering {[a.__agent_id__ for a in perm]}: {errors}"
            for aid, r in results.items():
                assert r == ["ok", "ok"], (
                    f"Agent {aid} got {r!r} with ordering "
                    f"{[a.__agent_id__ for a in perm]}"
                )


# ── Race condition tests ──────────────────────────────────────────


class TestRaceConditions:
    def test_concurrent_claims_no_double_execution(self):
        """Multiple workers racing to claim the same task type —
        exactly one wins, no double execution."""
        tq = TaskQueue(STATE_DIR / "q")
        tq.submit("work", {"data": "x"}, task_id="t1")

        claimed_by: list[str] = []
        lock = threading.Lock()
        barrier = threading.Barrier(5)

        def try_claim(worker_id):
            barrier.wait()  # all threads start claiming simultaneously
            task = tq.claim("work", worker_id)
            if task is not None:
                with lock:
                    claimed_by.append(worker_id)

        _run_threads_with_timeout([lambda w=f"w{i}": try_claim(w) for i in range(5)])
        assert len(claimed_by) == 1, f"Double-claim: {claimed_by}"

    def test_concurrent_submit_idempotent(self):
        """Multiple threads submitting the same task_id — only one task created."""
        tq = TaskQueue(STATE_DIR / "q")
        barrier = threading.Barrier(5)

        def submit(i):
            barrier.wait()
            tq.submit("work", {"thread": i}, task_id="same-id")

        _run_threads_with_timeout([lambda i=i: submit(i) for i in range(5)])
        assert tq.pending_count() == 1

    def test_step_counters_stay_in_sync(self):
        """All agent threads must assign the same step ID to each
        choreographic statement, even under different scheduling."""
        arch = Architect(agent_id="arch")
        coder = Coder(agent_id="coder")

        step_ids_seen: dict[str, list[str]] = {"arch": [], "coder": []}
        lock = threading.Lock()

        class StepTrackingMock(ObjectInterpretation):
            @implements(Template.__apply__)
            def _call(self, template, *args, **kwargs):
                return "result"

        tq = TaskQueue(STATE_DIR / "q")
        ids = frozenset(["arch", "coder"])

        def run_agent(agent):
            mock = StepTrackingMock()
            tq.release_stale_claims(agent.__agent_id__)
            epp = EndpointProjection(agent, tq, ids, poll_interval=0.02)
            with handler(mock), handler(epp):
                arch.plan("spec")
                coder.implement("plan")
            # After execution, check the step counter
            with lock:
                step_ids_seen[agent.__agent_id__].append(epp._step)

        _run_threads_with_timeout([lambda a=a: run_agent(a) for a in [arch, coder]])
        # Both agents should have advanced through the same number of steps
        assert step_ids_seen["arch"] == step_ids_seen["coder"]

    def test_many_agents_many_steps(self):
        """Stress test: 5 coders scatter over 20 items."""
        coders = [Coder(agent_id=f"c{i}") for i in range(5)]
        items = list(range(20))
        responses = {"implement": "done"}

        def choreo(items, coders):
            return scatter(items, coders, lambda c, m: c.implement(str(m)))

        mock = MockLLM(responses)
        c = Choreography(
            choreo,
            agents=coders,
            state_dir=STATE_DIR,
            handlers=[mock],
            poll_interval=0.02,
        )
        result = c.run(items=items, coders=coders)
        assert result == ["done"] * 20


# ── Edge case tests ───────────────────────────────────────────────


class TestEdgeCases:
    def test_single_agent_choreography(self):
        """A choreography with only one agent works without deadlock."""
        coder = Coder(agent_id="solo")

        def choreo(coder):
            return coder.implement("just me")

        mock = MockLLM({"implement": "solo-code"})
        c = Choreography(
            choreo,
            agents=[coder],
            state_dir=STATE_DIR,
            handlers=[mock],
            poll_interval=0.02,
        )
        result = c.run(coder=coder)
        assert result == "solo-code"

    def test_empty_scatter(self):
        """Scatter over an empty list returns [] without hanging."""
        c1 = Coder(agent_id="c1")
        c2 = Coder(agent_id="c2")

        def choreo(items, coders):
            return scatter(items, coders, lambda c, m: c.implement(m))

        mock = MockLLM({"implement": "code"})
        c = Choreography(
            choreo,
            agents=[c1, c2],
            state_dir=STATE_DIR,
            handlers=[mock],
            poll_interval=0.02,
        )
        result = c.run(items=[], coders=[c1, c2])
        assert result == []

    def test_agent_error_propagates(self):
        """An exception in one agent's template propagates as ChoreographyError."""
        arch = Architect(agent_id="arch")
        coder = Coder(agent_id="coder")

        def choreo(arch, coder):
            plan = arch.plan("spec")
            return coder.implement(plan)

        mock = FailingMockLLM(
            responses={"implement": "code"},
            fail_on={"arch.plan"},
        )
        c = Choreography(
            choreo,
            agents=[arch, coder],
            state_dir=STATE_DIR,
            handlers=[mock],
            poll_interval=0.02,
        )
        with pytest.raises(ChoreographyError, match="arch"):
            c.run(arch=arch, coder=coder)

    def test_scatter_single_worker(self):
        """Scatter with one worker still completes all items."""
        coder = Coder(agent_id="c1")

        call_count = {"n": 0}
        call_lock = threading.Lock()

        class CountingMock(ObjectInterpretation):
            @implements(Template.__apply__)
            def _call(self, template, *args, **kwargs):
                with call_lock:
                    call_count["n"] += 1
                return f"result-{call_count['n']}"

        def choreo(items, coders):
            return scatter(items, coders, lambda c, m: c.implement(m))

        c = Choreography(
            choreo,
            agents=[coder],
            state_dir=STATE_DIR,
            handlers=[CountingMock()],
            poll_interval=0.02,
        )
        result = c.run(items=["a", "b", "c"], coders=[coder])
        assert len(result) == 3
        # All results should be non-None
        assert all(r is not None for r in result)

    def test_scatter_error_propagates(self):
        """An error inside a scatter item propagates as ChoreographyError."""
        c1 = Coder(agent_id="c1")

        class ScatterFailMock(ObjectInterpretation):
            @implements(Template.__apply__)
            def _call(self, template, *args, **kwargs):
                raise RuntimeError("scatter item failed")

        def choreo(items, coders):
            return scatter(items, coders, lambda c, m: c.implement(m))

        c = Choreography(
            choreo,
            agents=[c1],
            state_dir=STATE_DIR,
            handlers=[ScatterFailMock()],
            poll_interval=0.02,
        )
        with pytest.raises(ChoreographyError):
            c.run(items=["x"], coders=[c1])

    def test_result_none_vs_not_done(self):
        """get_result returns None for non-existent tasks, not for tasks
        whose result *is* None — ensuring poll loops don't confuse the two."""
        tq = TaskQueue(STATE_DIR / "q")
        # Non-existent task
        assert tq.get_result("nonexistent") is None

        # Task with an actual result of a falsy value
        tq.submit("work", {}, task_id="t1")
        tq.claim("work", "w1")
        tq.complete("t1", "w1", 0)  # result is 0 (falsy but not None)
        assert tq.get_result("t1") == 0

    def test_repeated_runs_deterministic(self):
        """Running the same choreography 5 times gives identical results."""
        arch = Architect(agent_id="arch")
        coder = Coder(agent_id="coder")

        def choreo(arch, coder):
            plan = arch.plan("spec")
            return coder.implement(plan)

        results = []
        for i in range(5):
            q_dir = STATE_DIR / "q"
            if q_dir.exists():
                shutil.rmtree(q_dir)
            mock = MockLLM({"plan": "plan", "implement": "code"})
            c = Choreography(
                choreo,
                agents=[arch, coder],
                state_dir=STATE_DIR,
                handlers=[mock],
                poll_interval=0.02,
            )
            results.append(c.run(arch=arch, coder=coder))

        assert all(r == results[0] for r in results)


# ── Scatter tests ─────────────────────────────────────────────────


class TestScatter:
    def test_scatter_distributes_work(self):
        """Scatter distributes items across agents via claim mechanism."""
        c1 = Coder(agent_id="c1")
        c2 = Coder(agent_id="c2")
        rev = Reviewer(agent_id="rev")

        tq = TaskQueue(STATE_DIR / "q")
        ids = frozenset(["c1", "c2", "rev"])
        items = ["A", "B", "C", "D"]

        def choreo(items, coders, reviewer):
            codes = scatter(items, coders, lambda c, m: c.implement(m))
            return [reviewer.review(code) for code in codes]

        results: dict[str, Any] = {}
        llm_calls: dict[str, list[str]] = {}
        errors: list = []
        lock = threading.Lock()

        def run(agent):
            try:
                mock = MockLLM({"implement": "code", "review": "PASS"})
                tq.release_stale_claims(agent.__agent_id__)
                epp = EndpointProjection(agent, tq, ids, poll_interval=0.02)
                with handler(mock), handler(epp):
                    r = choreo(items, [c1, c2], rev)
                    with lock:
                        results[agent.__agent_id__] = r
                        llm_calls[agent.__agent_id__] = list(mock.calls)
            except Exception as e:
                with lock:
                    errors.append(e)

        _run_threads_with_timeout([lambda a=a: run(a) for a in [c1, c2, rev]])

        assert not errors, errors
        assert results["c1"] == results["c2"] == results["rev"]
        # Total execute calls across coders should be 4
        c1_impl = [c for c in llm_calls.get("c1", []) if "implement" in c]
        c2_impl = [c for c in llm_calls.get("c2", []) if "implement" in c]
        assert len(c1_impl) + len(c2_impl) == 4
        # Reviewer did all 4 reviews
        rev_reviews = [c for c in llm_calls.get("rev", []) if "review" in c]
        assert len(rev_reviews) == 4

    def test_scatter_crash_recovery(self):
        """Scatter with some items cached: only remaining items executed."""
        tq = TaskQueue(STATE_DIR / "q")
        c1 = Coder(agent_id="c1")
        c2 = Coder(agent_id="c2")

        items = ["A", "B", "C", "D"]

        # Pre-cache items 0 and 1
        for i in range(2):
            step = f"step-0000:{i:04d}"
            done = tq._task_path(step, TaskStatus.DONE)
            done.write_text(
                json.dumps(
                    {
                        "id": step,
                        "type": "scatter-step-0000",
                        "status": "done",
                        "result": f"cached-{items[i]}",
                        "payload": {"item_index": i},
                    }
                )
            )

        def choreo(items, coders):
            return scatter(items, coders, lambda c, m: c.implement(m))

        ids = frozenset(["c1", "c2"])
        results: dict[str, Any] = {}
        llm_calls: dict[str, list[str]] = {}
        errors: list = []
        lock = threading.Lock()

        def run(agent):
            try:
                mock = MockLLM({"implement": "fresh"})
                tq.release_stale_claims(agent.__agent_id__)
                epp = EndpointProjection(agent, tq, ids, poll_interval=0.02)
                with handler(mock), handler(epp):
                    r = choreo(items, [c1, c2])
                    with lock:
                        results[agent.__agent_id__] = r
                        llm_calls[agent.__agent_id__] = list(mock.calls)
            except Exception as e:
                with lock:
                    errors.append(e)

        _run_threads_with_timeout([lambda a=a: run(a) for a in [c1, c2]])

        assert not errors, errors
        r = results["c1"]
        assert r[0] == "cached-A"
        assert r[1] == "cached-B"
        assert r[2] == "fresh"
        assert r[3] == "fresh"

        # Only 2 LLM calls total (items 2 and 3)
        total = sum(len(calls) for calls in llm_calls.values())
        assert total == 2


# ── Choreography runner tests ─────────────────────────────────────


class TestChoreography:
    def test_run(self):
        """High-level Choreography.run() orchestrates everything."""
        arch = Architect(agent_id="arch")
        coder = Coder(agent_id="coder")

        def choreo(arch, coder):
            plan = arch.plan("spec")
            return coder.implement(plan)

        mock = MockLLM({"plan": "plan", "implement": "code"})
        c = Choreography(
            choreo,
            agents=[arch, coder],
            state_dir=STATE_DIR,
            handlers=[mock],
            poll_interval=0.02,
        )
        result = c.run(arch=arch, coder=coder)
        assert result == "code"

    def test_run_restart(self):
        """Run, restart, verify cached results are used."""
        arch = Architect(agent_id="arch")
        coder = Coder(agent_id="coder")

        def choreo(arch, coder):
            plan = arch.plan("spec")
            return coder.implement(plan)

        mock1 = MockLLM({"plan": "plan-v1", "implement": "code-v1"})
        c1 = Choreography(
            choreo,
            agents=[arch, coder],
            state_dir=STATE_DIR,
            handlers=[mock1],
            poll_interval=0.02,
        )
        result1 = c1.run(arch=arch, coder=coder)
        assert result1 == "code-v1"

        # "Restart" — new Choreography, same state_dir
        # Even with different responses, should use cached
        mock2 = MockLLM({"plan": "plan-v2", "implement": "code-v2"})
        c2 = Choreography(
            choreo,
            agents=[arch, coder],
            state_dir=STATE_DIR,
            handlers=[mock2],
            poll_interval=0.02,
        )
        result2 = c2.run(arch=arch, coder=coder)
        # Both steps were cached from first run
        assert result2 == "code-v1"

    def test_scatter_with_choreography(self):
        """Choreography with scatter."""
        c1 = Coder(agent_id="c1")
        c2 = Coder(agent_id="c2")

        def choreo(items, coders):
            return scatter(items, coders, lambda c, m: c.implement(m))

        mock = MockLLM({"implement": "code"})
        c = Choreography(
            choreo,
            agents=[c1, c2],
            state_dir=STATE_DIR,
            handlers=[mock],
            poll_interval=0.02,
        )
        result = c.run(items=["A", "B", "C"], coders=[c1, c2])
        assert result == ["code", "code", "code"]
