"""
Tests for the futures handler (effectful.handlers.futures).

This module tests the integration of concurrent.futures with effectful,
including context preservation across thread boundaries.
"""

import time
from concurrent.futures import Future

import effectful.handlers.futures as futures
from effectful.handlers.futures import (
    Executor,
    ThreadPoolFuturesInterpretation,
)
from effectful.internals.runtime import release_handler_lock
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
    """Test that handler execution is mutually exclusive by default.

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

    def client(x: int):
        return add(x, x)

    with (
        handler(ThreadPoolFuturesInterpretation(max_workers=4)),
        handler({add: add_interp}),
    ):
        _ = sum(Executor.map(client, list(range(10))))
        # With mutual exclusion, we're guaranteed to get exactly 10
        assert add_calls == 10


def test_release_lock_for_concurrent_io():
    """Test that release_handler_lock allows concurrent I/O operations.

    This demonstrates the pattern for handlers that perform I/O and want
    to allow other handlers to run concurrently during the I/O wait.
    """
    from effectful.internals.runtime import release_handler_lock

    io_calls = 0
    concurrent_ios = 0
    max_concurrent = 0

    @defop
    def io_operation(x: int) -> int:
        """Simulates a slow io operation"""
        raise NotHandled

    def io_interp(x: int) -> int:
        nonlocal io_calls, concurrent_ios, max_concurrent
        # update state using lock
        io_calls += 1

        # release lock for IO
        with release_handler_lock():
            concurrent_ios += 1
            max_concurrent = max(max_concurrent, concurrent_ios)
            time.sleep(0.01)  # Simulate I/O wait
            concurrent_ios -= 1
        return x * 2

    def client(x: int):
        return io_operation(x)

    with (
        handler(ThreadPoolFuturesInterpretation(max_workers=4)),
        handler({io_operation: io_interp}),
    ):
        results = list(Executor.map(client, list(range(10))))

        assert io_calls == 10
        assert results == [x * 2 for x in range(10)]
        # With release_handler_lock, multiple I/O operations can run concurrently
        assert max_concurrent > 1, (
            f"Expected concurrent I/O, got max_concurrent={max_concurrent}"
        )


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
        with release_handler_lock():
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
