"""Tests for effectful.handlers.llm.multi — choreographic EPP with TaskQueue."""

import itertools
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
    InMemoryTaskQueue,
    PersistentTaskQueue,
    TaskStatus,
    fan_out,
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


class TesterAgent(PersistentAgent):
    """Writes tests."""

    @Template.define
    def write_tests(self, spec: str) -> str:
        """Write tests for: {spec}"""
        raise NotHandled


class Prover(PersistentAgent):
    """Proves theorems."""

    @Template.define
    def prove(self, spec: str) -> str:
        """Prove: {spec}"""
        raise NotHandled


# ── TaskQueue tests ───────────────────────────────────────────────

# Counter to avoid directory collisions between parametrized persistent tests.
_ptq_counter = 0


def _make_persistent_queue():
    global _ptq_counter
    _ptq_counter += 1
    return PersistentTaskQueue(STATE_DIR / f"q-{_ptq_counter}.db")


@pytest.fixture(params=["persistent", "in_memory"])
def make_queue(request):
    """Parametrized fixture — runs each test against both queue backends."""
    if request.param == "persistent":
        return _make_persistent_queue
    else:
        return InMemoryTaskQueue


class TestTaskQueue:
    def test_submit_and_claim(self, make_queue):
        tq = make_queue()
        tid = tq.submit("code", {"file": "main.py"}, task_id="t1")
        assert tid == "t1"

        task = tq.claim("code", "worker1")
        assert task is not None
        assert task["id"] == "t1"
        assert task["status"] == TaskStatus.CLAIMED

        # Can't claim again
        assert tq.claim("code", "worker2") is None

    def test_idempotent_submit(self, make_queue):
        tq = make_queue()
        tq.submit("code", {}, task_id="t1")
        tq.submit("code", {}, task_id="t1")  # no-op
        assert tq.pending_count() == 1

    def test_complete_and_get_result(self, make_queue):
        tq = make_queue()
        tq.submit("code", {}, task_id="t1")
        tq.claim("code", "w1")
        tq.complete("t1", "w1", {"output": "hello"})
        assert tq.get_result("t1") == {"output": "hello"}

    def test_release_stale_claims(self, make_queue):
        tq = make_queue()
        tq.submit("code", {}, task_id="t1")
        tq.claim("code", "crashed_worker")
        assert tq.pending_count() == 0

        released = tq.release_stale_claims("crashed_worker")
        assert released == 1
        assert tq.pending_count() == 1

        # Can re-claim
        task = tq.claim("code", "new_worker")
        assert task is not None

    def test_claim_by_prefix(self, make_queue):
        tq = make_queue()
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

    def test_all_done(self, make_queue):
        tq = make_queue()
        assert tq.all_done()

        tq.submit("code", {}, task_id="t1")
        assert not tq.all_done()

        tq.claim("code", "w1")
        assert not tq.all_done()  # claimed but not done

        tq.complete("t1", "w1", "result")
        assert tq.all_done()

    def test_fail(self, make_queue):
        tq = make_queue()
        tq.submit("code", {}, task_id="t1")
        tq.claim("code", "w1")
        tq.fail("t1", "w1", "boom")
        # Failed tasks are not pending/claimed, so all_done is True
        assert tq.all_done()
        # get_result returns None for failed tasks
        assert tq.get_result("t1") is None

    def test_concurrent_claims(self, make_queue):
        """Multiple threads claiming — no double claims."""
        tq = make_queue()
        n_tasks = 20
        for i in range(n_tasks):
            tq.submit("work", {"i": i}, task_id=f"t{i:03d}")

        claimed: list[dict] = []
        lock = threading.Lock()

        def claimer(owner):
            while True:
                task = tq.claim("work", owner)
                if task is None:
                    break
                with lock:
                    claimed.append(task)

        threads = [threading.Thread(target=claimer, args=(f"w{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=THREAD_TIMEOUT)

        # Each task claimed exactly once
        ids = [t["id"] for t in claimed]
        assert len(ids) == n_tasks
        assert len(set(ids)) == n_tasks


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
        tq = InMemoryTaskQueue()
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

        tq = InMemoryTaskQueue()
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

    def test_crash_recovery(self):
        """Pre-cache step 0, restart: step 0 from cache, step 1 fresh."""
        tq = InMemoryTaskQueue()
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
        tq = InMemoryTaskQueue()
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
        tq = InMemoryTaskQueue()
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
        tq = InMemoryTaskQueue()
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

        tq = InMemoryTaskQueue()
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
            queue=InMemoryTaskQueue(),
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
            queue=InMemoryTaskQueue(),
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
            queue=InMemoryTaskQueue(),
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
            queue=InMemoryTaskQueue(),
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
            queue=InMemoryTaskQueue(),
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
            queue=InMemoryTaskQueue(),
            handlers=[ScatterFailMock()],
            poll_interval=0.02,
        )
        with pytest.raises(ChoreographyError):
            c.run(items=["x"], coders=[c1])

    def test_result_none_vs_not_done(self):
        """get_result returns None for non-existent tasks, not for tasks
        whose result *is* None — ensuring poll loops don't confuse the two."""
        tq = InMemoryTaskQueue()
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
                queue=InMemoryTaskQueue(),
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

        tq = InMemoryTaskQueue()
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
        tq = InMemoryTaskQueue()
        c1 = Coder(agent_id="c1")
        c2 = Coder(agent_id="c2")

        items = ["A", "B", "C", "D"]

        # Pre-cache items 0 and 1 via submit/claim/complete
        for i in range(2):
            step = f"step-0000:{i:04d}"
            tq.submit("scatter-step-0000", {"item_index": i}, task_id=step)
            tq.claim("scatter-step-0000", "prior-worker")
            tq.complete(step, "prior-worker", f"cached-{items[i]}")

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
            queue=InMemoryTaskQueue(),
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

        # Shared persistent queue survives across Choreography instances
        shared_queue = PersistentTaskQueue(STATE_DIR / "restart_q.db")

        mock1 = MockLLM({"plan": "plan-v1", "implement": "code-v1"})
        c1 = Choreography(
            choreo,
            agents=[arch, coder],
            queue=shared_queue,
            handlers=[mock1],
            poll_interval=0.02,
        )
        result1 = c1.run(arch=arch, coder=coder)
        assert result1 == "code-v1"

        # "Restart" — new Choreography, same persistent queue
        # Even with different responses, should use cached
        mock2 = MockLLM({"plan": "plan-v2", "implement": "code-v2"})
        c2 = Choreography(
            choreo,
            agents=[arch, coder],
            queue=shared_queue,
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
            queue=InMemoryTaskQueue(),
            handlers=[mock],
            poll_interval=0.02,
        )
        result = c.run(items=["A", "B", "C"], coders=[c1, c2])
        assert result == ["code", "code", "code"]


# ── SQLite crash tolerance tests ─────────────────────────────────


class TestSQLiteTaskQueueCrashTolerance:
    """Tests that verify SQLite-specific crash tolerance properties."""

    def test_wal_mode_enabled(self):
        """WAL journal mode is enabled for crash tolerance."""
        import sqlite3

        tq = PersistentTaskQueue(STATE_DIR / "wal_test.db")
        tq.submit("work", {}, task_id="t1")

        conn = sqlite3.connect(str(tq.db_path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"

    def test_db_integrity_after_operations(self):
        """Database passes integrity check after various operations."""
        import sqlite3

        tq = PersistentTaskQueue(STATE_DIR / "integrity_test.db")

        # Submit, claim, complete, fail
        tq.submit("work", {"data": "a"}, task_id="t1")
        tq.submit("work", {"data": "b"}, task_id="t2")
        tq.claim("work", "w1")
        tq.complete("t1", "w1", {"result": "done"})
        tq.claim("work", "w1")
        tq.fail("t2", "w1", "boom")

        conn = sqlite3.connect(str(tq.db_path))
        result = conn.execute("PRAGMA integrity_check").fetchone()[0]
        conn.close()
        assert result == "ok"

    def test_persistence_across_instances(self):
        """Data survives creating a new PersistentTaskQueue on same db."""
        db_path = STATE_DIR / "persist_test.db"

        # Instance 1: submit and complete
        tq1 = PersistentTaskQueue(db_path)
        tq1.submit("work", {"key": "value"}, task_id="t1")
        tq1.claim("work", "w1")
        tq1.complete("t1", "w1", "result-1")

        # Instance 2: verify result survives
        tq2 = PersistentTaskQueue(db_path)
        assert tq2.get_result("t1") == "result-1"
        assert tq2.all_done()

    def test_stale_claims_survive_restart(self):
        """Claimed-but-not-completed tasks are recoverable after restart."""
        db_path = STATE_DIR / "stale_test.db"

        # Instance 1: submit and claim (simulating crash before completion)
        tq1 = PersistentTaskQueue(db_path)
        tq1.submit("work", {"data": "x"}, task_id="t1")
        tq1.submit("work", {"data": "y"}, task_id="t2")
        tq1.claim("work", "crashed_worker")
        tq1.claim("work", "crashed_worker")

        # Instance 2: restart, release stale claims, re-claim
        tq2 = PersistentTaskQueue(db_path)
        assert tq2.pending_count() == 0  # both are claimed
        released = tq2.release_stale_claims("crashed_worker")
        assert released == 2
        assert tq2.pending_count() == 2

        # Can re-claim and complete
        task = tq2.claim("work", "new_worker")
        assert task is not None
        tq2.complete(task["id"], "new_worker", "recovered")
        assert tq2.get_result(task["id"]) == "recovered"

    def test_partial_choreography_restart(self):
        """Choreography can resume from a partially completed state."""
        db_path = STATE_DIR / "partial_choreo.db"
        arch = Architect(agent_id="arch")
        coder = Coder(agent_id="coder")

        # "Run 1": complete step 0 (plan), simulate crash before step 1
        tq1 = PersistentTaskQueue(db_path)
        tq1.submit("plan", {"agent": "arch"}, task_id="step-0000")
        tq1.claim("plan", "arch")
        tq1.complete("step-0000", "arch", "the-plan")
        # step 1 never submitted (crash)

        # "Run 2": restart with same db — step 0 cached, step 1 fresh
        tq2 = PersistentTaskQueue(db_path)

        def choreo(arch, coder):
            plan = arch.plan("spec")
            return coder.implement(plan)

        ids = frozenset(["arch", "coder"])
        results: dict[str, Any] = {}
        errors: list = []
        lock = threading.Lock()

        def run(agent):
            try:
                mock = MockLLM({"plan": "SHOULD-NOT-RUN", "implement": "fresh-code"})
                tq2.release_stale_claims(agent.__agent_id__)
                epp = EndpointProjection(agent, tq2, ids, poll_interval=0.02)
                with handler(mock), handler(epp):
                    r = choreo(arch=arch, coder=coder)
                    with lock:
                        results[agent.__agent_id__] = r
            except Exception as e:
                with lock:
                    errors.append(e)

        _run_threads_with_timeout([lambda a=a: run(a) for a in [arch, coder]])

        assert not errors, errors
        # Step 0 was cached as "the-plan", step 1 ran fresh
        assert results["arch"] == "fresh-code"
        assert results["coder"] == "fresh-code"

    def test_concurrent_claims_across_connections(self):
        """Multiple threads with separate connections cannot double-claim."""
        db_path = STATE_DIR / "concurrent_conn.db"
        tq = PersistentTaskQueue(db_path)

        n_tasks = 20
        for i in range(n_tasks):
            tq.submit("work", {"i": i}, task_id=f"t{i:03d}")

        claimed: list[dict] = []
        lock = threading.Lock()

        def claimer(owner):
            while True:
                task = tq.claim("work", owner)
                if task is None:
                    break
                with lock:
                    claimed.append(task)

        _run_threads_with_timeout([lambda w=f"w{i}": claimer(w) for i in range(5)])

        ids = [t["id"] for t in claimed]
        assert len(ids) == n_tasks
        assert len(set(ids)) == n_tasks

    def test_scatter_crash_recovery_sqlite(self):
        """Scatter with SQLite queue: cached items survive restart."""
        db_path = STATE_DIR / "scatter_crash.db"

        # "Run 1": complete scatter items 0 and 1
        tq1 = PersistentTaskQueue(db_path)
        for i in range(2):
            step = f"step-0000:{i:04d}"
            tq1.submit("scatter-step-0000", {"item_index": i}, task_id=step)
            tq1.claim("scatter-step-0000", "prior-worker")
            tq1.complete(step, "prior-worker", f"cached-{i}")

        # "Run 2": scatter should pick up cached items 0,1 and run 2,3 fresh
        tq2 = PersistentTaskQueue(db_path)
        c1 = Coder(agent_id="c1")
        c2 = Coder(agent_id="c2")
        items = ["A", "B", "C", "D"]

        def choreo(items, coders):
            return scatter(items, coders, lambda c, m: c.implement(m))

        ids = frozenset(["c1", "c2"])
        results: dict[str, Any] = {}
        errors: list = []
        lock = threading.Lock()

        def run(agent):
            try:
                mock = MockLLM({"implement": "fresh"})
                tq2.release_stale_claims(agent.__agent_id__)
                epp = EndpointProjection(agent, tq2, ids, poll_interval=0.02)
                with handler(mock), handler(epp):
                    r = choreo(items, [c1, c2])
                    with lock:
                        results[agent.__agent_id__] = r
            except Exception as e:
                with lock:
                    errors.append(e)

        _run_threads_with_timeout([lambda a=a: run(a) for a in [c1, c2]])

        assert not errors, errors
        r = results["c1"]
        assert r[0] == "cached-0"
        assert r[1] == "cached-1"
        assert r[2] == "fresh"
        assert r[3] == "fresh"

    def test_idempotent_submit_across_restarts(self):
        """Re-submitting existing task_id after restart is a no-op."""
        db_path = STATE_DIR / "idempotent_restart.db"

        tq1 = PersistentTaskQueue(db_path)
        tq1.submit("work", {"original": True}, task_id="t1")

        tq2 = PersistentTaskQueue(db_path)
        tq2.submit("work", {"duplicate": True}, task_id="t1")  # should be ignored
        assert tq2.pending_count() == 1

        # Verify original payload preserved
        task = tq2.claim("work", "w1")
        assert task is not None
        assert task["payload"] == {"original": True}


# ── fan_out tests ─────────────────────────────────────────────────


class TestFanOutDefault:
    """fan_out without EPP handler — sequential fallback."""

    def test_default_sequential(self):
        """Default fan_out runs groups sequentially."""
        coder = Coder(agent_id="c1")
        tester = TesterAgent(agent_id="t1")

        mock = MockLLM(
            {
                "c1.implement": "code-result",
                "t1.write_tests": "test-result",
            }
        )
        with handler(mock):
            results = fan_out(
                [
                    (["a", "b"], coder, lambda c, m: c.implement(m)),
                    (["x"], tester, lambda t, m: t.write_tests(m)),
                ]
            )
        assert len(results) == 2
        assert results[0] == ["code-result", "code-result"]
        assert results[1] == ["test-result"]

    def test_default_empty_groups(self):
        """fan_out with empty item lists returns empty lists."""
        coder = Coder(agent_id="c1")
        results = fan_out(
            [
                ([], coder, lambda c, m: c.implement(m)),
            ]
        )
        assert results == [[]]

    def test_default_no_groups(self):
        """fan_out with no groups returns empty list."""
        results = fan_out([])
        assert results == []


class TestFanOutEPP:
    """fan_out under EndpointProjection — concurrent execution."""

    def test_concurrent_different_agent_types(self):
        """Three different agent types work concurrently via fan_out."""
        coder = Coder(agent_id="c1")
        tester = TesterAgent(agent_id="t1")
        prover = Prover(agent_id="p1")

        mock = MockLLM(
            {
                "c1.implement": "impl-result",
                "t1.write_tests": "test-result",
                "p1.prove": "proof-result",
            }
        )
        queue = InMemoryTaskQueue()
        agents = [coder, tester, prover]
        ids = frozenset(a.__agent_id__ for a in agents)

        results_map: dict[str, Any] = {}
        errors: list[Exception] = []
        lock = threading.Lock()

        def choreo(c, t, p):
            return fan_out(
                [
                    (["a", "b"], c, lambda c, m: c.implement(m)),
                    (["x", "y", "z"], t, lambda t, m: t.write_tests(m)),
                    (["th1"], p, lambda p, m: p.prove(m)),
                ]
            )

        def run(agent):
            try:
                epp = EndpointProjection(agent, queue, ids, poll_interval=0.02)
                with handler(mock), handler(epp):
                    r = choreo(coder, tester, prover)
                    with lock:
                        results_map[agent.__agent_id__] = r
            except Exception as e:
                with lock:
                    errors.append(e)

        _run_threads_with_timeout([lambda a=a: run(a) for a in agents])

        assert not errors, errors
        # All agents compute same result
        for aid in ["c1", "t1", "p1"]:
            r = results_map[aid]
            assert r[0] == ["impl-result", "impl-result"]
            assert r[1] == ["test-result", "test-result", "test-result"]
            assert r[2] == ["proof-result"]

    def test_fan_out_with_multiple_workers_per_group(self):
        """fan_out with multiple coders in one group, single tester in another."""
        c1 = Coder(agent_id="c1")
        c2 = Coder(agent_id="c2")
        t1 = TesterAgent(agent_id="t1")

        execution_log: list[str] = []
        log_lock = threading.Lock()

        class TrackingMockLLM(ObjectInterpretation):
            @implements(Template.__apply__)
            def _call(self, template, *args, **kwargs):
                bound = get_bound_agent(template)
                key = f"{bound.__agent_id__}.{template.__name__}" if bound else ""
                with log_lock:
                    execution_log.append(key)
                return f"result-{key}"

        mock = TrackingMockLLM()
        queue = InMemoryTaskQueue()
        agents = [c1, c2, t1]
        ids = frozenset(a.__agent_id__ for a in agents)

        results_map: dict[str, Any] = {}
        errors: list[Exception] = []
        lock = threading.Lock()

        def choreo(coders, tester):
            return fan_out(
                [
                    (["m1", "m2", "m3", "m4"], coders, lambda c, m: c.implement(m)),
                    (["t1", "t2"], tester, lambda t, m: t.write_tests(m)),
                ]
            )

        def run(agent):
            try:
                epp = EndpointProjection(agent, queue, ids, poll_interval=0.02)
                with handler(mock), handler(epp):
                    r = choreo([c1, c2], t1)
                    with lock:
                        results_map[agent.__agent_id__] = r
            except Exception as e:
                with lock:
                    errors.append(e)

        _run_threads_with_timeout([lambda a=a: run(a) for a in agents])

        assert not errors, errors
        # Coders split the 4 items; tester handles 2 items
        r = results_map["c1"]
        assert len(r[0]) == 4  # 4 code results
        assert len(r[1]) == 2  # 2 test results
        # Both coders should have executed some implement calls
        coder_calls = [c for c in execution_log if c.endswith(".implement")]
        tester_calls = [c for c in execution_log if c.endswith(".write_tests")]
        assert len(coder_calls) == 4
        assert len(tester_calls) == 2

    def test_fan_out_empty_group_no_hang(self):
        """Empty items in one group don't cause hangs."""
        coder = Coder(agent_id="c1")
        tester = TesterAgent(agent_id="t1")

        mock = MockLLM({"c1.implement": "code", "t1.write_tests": "test"})
        queue = InMemoryTaskQueue()
        agents = [coder, tester]
        ids = frozenset(a.__agent_id__ for a in agents)

        results_map: dict[str, Any] = {}
        errors: list[Exception] = []
        lock = threading.Lock()

        def choreo(c, t):
            return fan_out(
                [
                    ([], c, lambda c, m: c.implement(m)),  # empty!
                    (["x"], t, lambda t, m: t.write_tests(m)),
                ]
            )

        def run(agent):
            try:
                epp = EndpointProjection(agent, queue, ids, poll_interval=0.02)
                with handler(mock), handler(epp):
                    r = choreo(coder, tester)
                    with lock:
                        results_map[agent.__agent_id__] = r
            except Exception as e:
                with lock:
                    errors.append(e)

        _run_threads_with_timeout([lambda a=a: run(a) for a in agents])

        assert not errors, errors
        r = results_map["c1"]
        assert r[0] == []
        assert r[1] == ["test"]

    def test_fan_out_step_counter_sync(self):
        """fan_out consumes exactly one step ID across all agent threads."""
        coder = Coder(agent_id="c1")
        reviewer = Reviewer(agent_id="r1")

        mock = MockLLM(
            {
                "c1.implement": "code",
                "r1.review": "lgtm",
            }
        )
        queue = InMemoryTaskQueue()
        agents = [coder, reviewer]
        ids = frozenset(a.__agent_id__ for a in agents)

        results_map: dict[str, Any] = {}
        errors: list[Exception] = []
        lock = threading.Lock()

        def choreo(c, r):
            # fan_out uses 1 step, then review uses 1 step
            fan_results = fan_out(
                [
                    (["a"], c, lambda c, m: c.implement(m)),
                ]
            )
            verdict = r.review(str(fan_results))
            return verdict

        def run(agent):
            try:
                epp = EndpointProjection(agent, queue, ids, poll_interval=0.02)
                with handler(mock), handler(epp):
                    r = choreo(coder, reviewer)
                    with lock:
                        results_map[agent.__agent_id__] = r
            except Exception as e:
                with lock:
                    errors.append(e)

        _run_threads_with_timeout([lambda a=a: run(a) for a in agents])

        assert not errors, errors
        # Both agents compute the same final result
        assert results_map["c1"] == "lgtm"
        assert results_map["r1"] == "lgtm"

    def test_fan_out_error_in_one_group(self):
        """Error in one group's fn propagates as ChoreographyError."""
        coder = Coder(agent_id="c1")
        tester = TesterAgent(agent_id="t1")

        class ErrorMockLLM(ObjectInterpretation):
            @implements(Template.__apply__)
            def _call(self, template, *args, **kwargs):
                bound = get_bound_agent(template)
                key = f"{bound.__agent_id__}.{template.__name__}" if bound else ""
                if key == "t1.write_tests":
                    raise RuntimeError("test failure!")
                return "ok"

        mock = ErrorMockLLM()

        def program(c, t):
            return fan_out(
                [
                    (["a"], c, lambda c, m: c.implement(m)),
                    (["x"], t, lambda t, m: t.write_tests(m)),
                ]
            )

        choreo = Choreography(
            program,
            agents=[coder, tester],
            handlers=[mock],
        )
        with pytest.raises(ChoreographyError, match="test failure"):
            choreo.run(c=coder, t=tester)

    def test_fan_out_all_orderings_agree(self):
        """All thread scheduling orderings produce the same result."""
        for delays in [
            {"c1": 0.0, "t1": 0.05},
            {"c1": 0.05, "t1": 0.0},
            {"c1": 0.0, "t1": 0.0},
        ]:
            coder = Coder(agent_id="c1")
            tester = TesterAgent(agent_id="t1")

            mock = DelayedMockLLM(
                {"c1.implement": "code", "t1.write_tests": "test"},
                delays,
            )
            queue = InMemoryTaskQueue()
            agents = [coder, tester]
            ids = frozenset(a.__agent_id__ for a in agents)

            results_map: dict[str, Any] = {}
            errors: list[Exception] = []
            lock = threading.Lock()

            def choreo(c, t):
                return fan_out(
                    [
                        (["a", "b"], c, lambda c, m: c.implement(m)),
                        (["x"], t, lambda t, m: t.write_tests(m)),
                    ]
                )

            def run(agent):
                try:
                    epp = EndpointProjection(agent, queue, ids, poll_interval=0.02)
                    with handler(mock), handler(epp):
                        r = choreo(coder, tester)
                        with lock:
                            results_map[agent.__agent_id__] = r
                except Exception as e:
                    with lock:
                        errors.append(e)

            _run_threads_with_timeout([lambda a=a: run(a) for a in agents])

            assert not errors, errors
            assert results_map["c1"] == results_map["t1"]


class TestFanOutChoreography:
    """fan_out via the high-level Choreography runner."""

    def test_choreography_fan_out(self):
        """fan_out works through Choreography.run()."""
        coder = Coder(agent_id="c1")
        tester = TesterAgent(agent_id="t1")

        mock = MockLLM({"c1.implement": "code-out", "t1.write_tests": "test-out"})

        def program(c, t):
            return fan_out(
                [
                    (["a", "b"], c, lambda c, m: c.implement(m)),
                    (["x"], t, lambda t, m: t.write_tests(m)),
                ]
            )

        choreo = Choreography(
            program,
            agents=[coder, tester],
            handlers=[mock],
        )
        result = choreo.run(c=coder, t=tester)
        assert result == [["code-out", "code-out"], ["test-out"]]

    def test_choreography_fan_out_with_scatter(self):
        """fan_out and scatter can be mixed in the same choreography."""
        architect = Architect(agent_id="arch")
        coder = Coder(agent_id="c1")
        tester = TesterAgent(agent_id="t1")
        reviewer = Reviewer(agent_id="r1")

        mock = MockLLM(
            {
                "arch.plan": "plan-result",
                "c1.implement": "code",
                "t1.write_tests": "test",
                "r1.review": "lgtm",
            }
        )

        def program(arch, c, t, r):
            plan = arch.plan("project")
            # fan_out: code and test in parallel
            code_results, test_results = fan_out(
                [
                    (["m1", "m2"], c, lambda c, m: c.implement(m)),
                    (["t1"], t, lambda t, m: t.write_tests(m)),
                ]
            )
            # Then scatter reviews sequentially
            reviews = scatter(code_results, r, lambda r, code: r.review(code))
            return {"plan": plan, "reviews": reviews, "tests": test_results}

        choreo = Choreography(
            program,
            agents=[architect, coder, tester, reviewer],
            handlers=[mock],
        )
        result = choreo.run(arch=architect, c=coder, t=tester, r=reviewer)
        assert result["plan"] == "plan-result"
        assert result["reviews"] == ["lgtm", "lgtm"]
        assert result["tests"] == ["test"]


class TestFanOutCrashTolerance:
    """fan_out crash recovery with PersistentTaskQueue."""

    def test_fan_out_crash_recovery_with_cached_results(self):
        """Pre-cached fan_out items are reused on restart."""
        db_path = STATE_DIR / "fan_out_crash.db"

        # "Run 1": simulate partial completion
        tq1 = PersistentTaskQueue(db_path)
        # Pre-cache group 0 items
        tq1.submit(
            "fan-step-0000:g0",
            {"group": 0, "item_index": 0},
            task_id="step-0000:g0:0000",
        )
        tq1.claim_by_prefix("step-0000:g0:0000", "prior-worker")
        tq1.complete("step-0000:g0:0000", "prior-worker", "cached-code")

        tq1.submit(
            "fan-step-0000:g1",
            {"group": 1, "item_index": 0},
            task_id="step-0000:g1:0000",
        )
        tq1.claim_by_prefix("step-0000:g1:0000", "prior-worker")
        tq1.complete("step-0000:g1:0000", "prior-worker", "cached-test")

        # "Run 2": restart with one uncached item per group
        tq2 = PersistentTaskQueue(db_path)
        c1 = Coder(agent_id="c1")
        t1 = TesterAgent(agent_id="t1")

        mock = MockLLM({"c1.implement": "fresh-code", "t1.write_tests": "fresh-test"})

        def choreo(c, t):
            return fan_out(
                [
                    (["A", "B"], c, lambda c, m: c.implement(m)),
                    (["X", "Y"], t, lambda t, m: t.write_tests(m)),
                ]
            )

        agents = [c1, t1]
        ids = frozenset(a.__agent_id__ for a in agents)
        results_map: dict[str, Any] = {}
        errors: list[Exception] = []
        lock = threading.Lock()

        def run(agent):
            try:
                tq2.release_stale_claims(agent.__agent_id__)
                epp = EndpointProjection(agent, tq2, ids, poll_interval=0.02)
                with handler(mock), handler(epp):
                    r = choreo(c1, t1)
                    with lock:
                        results_map[agent.__agent_id__] = r
            except Exception as e:
                with lock:
                    errors.append(e)

        _run_threads_with_timeout([lambda a=a: run(a) for a in agents])

        assert not errors, errors
        r = results_map["c1"]
        assert r[0][0] == "cached-code"
        assert r[0][1] == "fresh-code"
        assert r[1][0] == "cached-test"
        assert r[1][1] == "fresh-test"

    def test_fan_out_concurrent_no_double_execution(self):
        """No item is executed twice even under concurrent claiming."""
        c1 = Coder(agent_id="c1")
        c2 = Coder(agent_id="c2")
        t1 = TesterAgent(agent_id="t1")
        t2 = TesterAgent(agent_id="t2")

        executed: list[str] = []
        exec_lock = threading.Lock()

        class TrackingMock(ObjectInterpretation):
            @implements(Template.__apply__)
            def _call(self, template, *args, **kwargs):
                bound = get_bound_agent(template)
                key = f"{bound.__agent_id__}.{template.__name__}" if bound else ""
                with exec_lock:
                    executed.append(key)
                return f"result-{key}"

        mock = TrackingMock()
        queue = InMemoryTaskQueue()
        agents = [c1, c2, t1, t2]
        ids = frozenset(a.__agent_id__ for a in agents)

        results_map: dict[str, Any] = {}
        errors: list[Exception] = []
        lock = threading.Lock()

        items_code = [f"mod-{i}" for i in range(10)]
        items_test = [f"test-{i}" for i in range(8)]

        def choreo(coders, testers):
            return fan_out(
                [
                    (items_code, coders, lambda c, m: c.implement(m)),
                    (items_test, testers, lambda t, m: t.write_tests(m)),
                ]
            )

        def run(agent):
            try:
                epp = EndpointProjection(agent, queue, ids, poll_interval=0.02)
                with handler(mock), handler(epp):
                    r = choreo([c1, c2], [t1, t2])
                    with lock:
                        results_map[agent.__agent_id__] = r
            except Exception as e:
                with lock:
                    errors.append(e)

        _run_threads_with_timeout([lambda a=a: run(a) for a in agents])

        assert not errors, errors
        # Exactly 10 implement + 8 write_tests calls
        impl_calls = [c for c in executed if ".implement" in c]
        test_calls = [c for c in executed if ".write_tests" in c]
        assert len(impl_calls) == 10
        assert len(test_calls) == 8
        # All agents see the same result
        for aid in ["c1", "c2", "t1", "t2"]:
            assert results_map[aid] == results_map["c1"]
