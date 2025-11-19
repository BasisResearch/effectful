"""
Tests for the futures handler (effectful.handlers.futures).

This module tests the integration of concurrent.futures with effectful,
including context preservation across thread boundaries.
"""

import time
from concurrent.futures import Future
from threading import RLock

import effectful.handlers.futures as futures
from effectful.handlers.futures import (
    Executor,
    ThreadPoolFuturesInterpretation,
)
from effectful.ops.semantics import NotHandled, defop, evaluate, handler
from effectful.ops.types import Term


@defop
def add(x: int, y: int) -> int:
    raise NotHandled


@defop
def a_mul(x: int, y: int) -> Future[int]:
    raise NotHandled


@defop
def a_div(x: int, y: int) -> Future[int]:
    raise NotHandled


@defop
def a_fac(n: int) -> Future[int]:
    raise NotHandled


def test_uninterp_async():
    """calling async func without interpretation returns term"""
    t = a_div(10, 20)
    assert isinstance(t, Term)


def test_mutual_exclusion():
    """Handler execution is not mutually exclusive by default, just
    like any other object call. As in python, if you call a function
    that may have some shared state, you must lock it as a client.

    Without mutual exclusion, the race condition in add_interp would cause
    add_calls to be less than 10. With mutual exclusion, we're guaranteed
    to get exactly 10 calls.

    """
    add_calls = 0

    def add_interp(x: int, y: int) -> int:
        nonlocal add_calls
        no_calls = add_calls
        time.sleep(0.001)
        add_calls = no_calls + 1
        return x + y

    client_lock = RLock()

    def client(x: int):
        # hey, I'm running a function that may have shared state, let me lock it
        with client_lock:
            res = add(x, x)
        return res

    with (
        handler(ThreadPoolFuturesInterpretation(max_workers=4)),
        handler({add: add_interp}),
    ):
        _ = sum(Executor.map(client, list(range(10))))
        # With mutual exclusion, we're guaranteed to get exactly 10
        assert add_calls == 10


def test_concurrent_client_execution():
    add_calls = 0
    add_calls_interp = 0

    def add_interp(x: int, y: int) -> int:
        nonlocal add_calls
        no_calls = add_calls
        time.sleep(0.001)
        add_calls = no_calls + 1
        return x + y

    def client(x: int):
        # clients submitted to the executor ARE NOT synchronous
        nonlocal add_calls_interp
        no_calls = add_calls_interp
        time.sleep(0.001)
        add_calls_interp = no_calls + 1
        return add(x, x)

    with (
        handler(ThreadPoolFuturesInterpretation(max_workers=4)),
        handler({add: add_interp}),
    ):
        _ = sum(Executor.map(client, list(range(10))))
        # Without mutual exclusion, we're not guaranteed to get exactly 10
        assert add_calls != 10
        # client is not synchronous so no guarantees.
        assert add_calls_interp != 10


def test_wait_several_futures():
    def client_code():
        results = []
        for fut in futures.wait([a_div(3, 4), a_mul(4, 5)]).done:
            results.append(fut.result())  # noqa: PERF401
        return results

    def a_div_interp(x, y):
        return Executor.submit(lambda x, y: x / y, x, y)

    def a_mul_interp(x, y):
        return Executor.submit(lambda x, y: x * y, x, y)

    with (
        handler(ThreadPoolFuturesInterpretation()),
        handler({a_div: a_div_interp, a_mul: a_mul_interp}),
    ):
        assert set(client_code()) == {3 / 4, 4 * 5}


def test_eval_of_concurrent_terms():
    def client_code():
        # spawn two tasks in parallel
        r1 = a_div(3, 4)
        r2 = a_mul(3, 4)
        return r1.result() + r2.result()

    def a_div_interp(x, y):
        return Executor.submit(lambda x, y: x / y, x, y)

    def a_mul_interp(x, y):
        return Executor.submit(lambda x, y: x * y, x, y)

    res_stx = client_code()
    assert isinstance(res_stx, Term)

    with (
        handler(ThreadPoolFuturesInterpretation()),
        handler({a_div: a_div_interp, a_mul: a_mul_interp}),
    ):
        res = client_code()
        assert res == (3 / 4 + 3 * 4)
        res = evaluate(res)
        assert res == (3 / 4 + 3 * 4)


def test_context_captured_at_submission():
    def submit_work():
        return Executor.submit(lambda: add(3, 4))

    def add_interp(x, y):
        return x + y

    def add_as_mul_interp(x, y):
        return x * y

    with handler(ThreadPoolFuturesInterpretation()):
        with handler({add: add_interp}):
            future = submit_work()

        # Retrieve result in a different context
        with handler({add: add_as_mul_interp}):
            result = future.result()

        # The result should be 7 (from submission context), not 12
        assert result == 7

    # Also test retrieving result completely outside any interpretation
    with (
        handler(ThreadPoolFuturesInterpretation()),
        handler({add: add_interp}),
    ):
        future = submit_work()

    # Retrieve result outside the handler context entirely
    result = future.result()
    assert result == 7


def test_concurrent_execution_faster_than_sequential():
    sleep_duration = 0.001  # 50ms per task

    def add_with_sleep(x, y):
        # important: we must release lock here to allow concurrency
        start = time.time()
        time.sleep(sleep_duration)
        return time.time() - start

    with (
        handler(ThreadPoolFuturesInterpretation(max_workers=3)),
        handler({add: add_with_sleep}),
    ):
        start = time.time()

        # Submit three tasks concurrently
        f1 = Executor.submit(lambda: add(1, 2))
        f2 = Executor.submit(lambda: add(3, 4))
        f3 = Executor.submit(lambda: add(5, 6))

        # Get all results
        sequential_time = f1.result() + f2.result() + f3.result()
        elapsed = time.time() - start

        assert elapsed < sequential_time
