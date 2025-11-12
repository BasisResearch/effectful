import asyncio
import time

import pytest

from effectful.ops.semantics import evaluate, handler
from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled, Term


# Test 1: Basic async operation without handler (should return a coroutine that returns a Term)
@defop
async def async_add(a: int, b: int) -> int:
    raise NotHandled


# Test 2: Async operation with handler
@defop
async def async_multiply(a: int, b: int) -> int:
    raise NotHandled


@defop
def sync_double(x: int) -> int:
    raise NotHandled


@defop
async def async_square(x: int) -> int:
    raise NotHandled


async def async_multiply_handler(a: int, b: int) -> int:
    await asyncio.sleep(0.1)
    return a * b


def double_handler(x: int) -> int:
    return x * 2


async def square_handler(x: int) -> int:
    await asyncio.sleep(0.1)
    return x * x


@pytest.mark.asyncio
async def test_async_without_handler():
    """Test that async operations without handlers return coroutines that yield Terms"""

    # Calling async operation without a handler should return a coroutine
    result = async_add(1, 2)
    assert asyncio.iscoroutine(result), (
        "Calling async operation without a handler should return a coroutine"
    )

    # When we await it, we should get a Term
    term = await result
    assert isinstance(term, Term), "When we await it, we should get a Term"


@pytest.mark.asyncio
async def test_async_with_handler():
    """Test that async operations with handlers work correctly"""

    with handler({async_multiply: async_multiply_handler}):
        result = async_multiply(3, 4)
        assert asyncio.iscoroutine(result), (
            "Calling async operations with handlers should return a coroutine"
        )

        value = await result
        assert value == 12, "Async operations should call their async handler"


@pytest.mark.asyncio
async def test_nested_async():
    """Test nested async operations"""

    with handler({async_multiply: async_multiply_handler}):
        # Create nested async calls
        expr = async_multiply(await async_multiply(2, 3), 4)

        result = await expr
        assert result == 24, "Nested operations work: 24 (2 * 3 * 4)"


@defop
async def async_sleep_and_return(duration: float, value: int) -> float:
    """Async operation that sleeps then returns the duration"""
    raise NotHandled


async def async_sleep_handler(duration: float, _value: int) -> float:
    """Handler that actually performs async sleep"""
    start_time = time.time()
    await asyncio.sleep(duration)
    return time.time() - start_time


@pytest.mark.asyncio
async def test_concurrent_execution():
    """Test that async operations can run concurrently"""
    with handler({async_sleep_and_return: async_sleep_handler}):
        start = time.time()
        # Create 5 concurrent tasks
        tasks = [async_sleep_and_return(1.0, i) for i in range(5)]
        res = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        assert elapsed < sum(res), "Operations ran concurrently!"


@pytest.mark.asyncio
async def test_mixed_sync_async_basic():
    """Test that sync and async operations can coexist"""

    with handler({sync_double: double_handler, async_square: square_handler}):
        # Mix sync and async operations
        doubled = sync_double(5)  # Returns 10 immediately
        assert isinstance(doubled, int), (
            "Synchronous operations in an async context return immediately"
        )
        assert doubled == 10, (
            "Synchronous operations are interpreted by their synchronous handlers."
        )

        squared = await async_square(3)  # Returns 9 after await
        assert squared == 9, (
            "Asynchronous operations in an async context are interpreted by their asynchronous handlers."
        )


@pytest.mark.asyncio
async def test_mixed_sync_async_advanced():
    t_coro = async_square(sync_double(2) + sync_double(3))
    assert asyncio.iscoroutine(t_coro), "mixed async sync operations return coroutines."
    t = await t_coro
    assert isinstance(t, Term), (
        "mixed async sync operations coroutines return terms once completed."
    )

    t_sync = sync_double(20) + sync_double(30)
    assert isinstance(t, Term), (
        "sync operations still return terms in an async context."
    )

    with handler({sync_double: double_handler, async_square: square_handler}):
        t_eval = evaluate(t)
        assert asyncio.iscoroutine(t_eval), (
            "Evaluating a term with async operations returns a couroutine"
        )

        res = await t_eval
        assert res == 100, (
            "evaluating mixed async sync terms redirects to their corresponding implementations"
        )

        res = evaluate(t_sync)
        assert isinstance(res, int), (
            "terms consisting of only sync operations behave as normal in an async context."
        )

@pytest.mark.asyncio
async def test_awaited_uninterpreted_terms_contain_await():
    """Test that awaited uninterpreted async terms produce terms with await_ inside"""
    from effectful.ops.semantics import await_

    coro = async_add(10, 20)
    assert asyncio.iscoroutine(coro), "Async operation without handler returns coroutine"

    term = await coro
    assert isinstance(term, Term), "Awaiting uninterpreted async operation returns a Term"
    assert term.op == await_, "The term is an await_ term"
    assert term.args[0] == async_add, "await_ term contains the async operation"
    assert term.args[1] == 10, "await_ term contains first argument"
    assert term.args[2] == 20, "await_ term contains second argument"

    async def add_handler(a: int, b: int) -> int:
        await asyncio.sleep(0.01)
        return a + b

    with handler({async_add: add_handler}):
        result_coro = evaluate(term)
        assert asyncio.iscoroutine(result_coro), "Evaluating await_ term returns coroutine"
        result = await result_coro
        assert result == 30, "Evaluating the await_ term produces the correct result"

