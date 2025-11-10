import asyncio
import time

import pytest

from effectful.ops.semantics import evaluate, handler, apply, async_apply
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
    from effectful.ops.syntax import await_

    # Call an async operation without a handler - returns a coroutine
    coro = async_add(10, 20)
    assert asyncio.iscoroutine(coro), "Async operation without handler returns coroutine"

    # Await it - should get an await_ term
    term = await coro
    assert isinstance(term, Term), "Awaiting uninterpreted async operation returns a Term"
    assert term.op == await_, "The term is an await_ term"
    assert term.args[0] == async_add, "await_ term contains the async operation"
    assert term.args[1] == 10, "await_ term contains first argument"
    assert term.args[2] == 20, "await_ term contains second argument"

    # The term itself should be evaluate-able with a handler
    async def add_handler(a: int, b: int) -> int:
        await asyncio.sleep(0.01)
        return a + b

    with handler({async_add: add_handler}):
        result_coro = evaluate(term)
        assert asyncio.iscoroutine(result_coro), "Evaluating await_ term returns coroutine"
        result = await result_coro
        assert result == 30, "Evaluating the await_ term produces the correct result"


@pytest.mark.asyncio
async def test_handlers_captured_at_construction_not_await():
    """Test that handlers are bound at construction time, not at await time"""

    @defop
    async def get_value() -> str:
        raise NotHandled

    # Handler 1: returns "construction"
    async def handler1() -> str:
        await asyncio.sleep(0.01)
        return "construction"

    # Handler 2: returns "await_time"
    async def handler2() -> str:
        await asyncio.sleep(0.01)
        return "await_time"

    # Construct the coroutine under handler1
    with handler({get_value: handler1}):
        coro = get_value()
        assert asyncio.iscoroutine(coro), "get_value() returns a coroutine"

    # Now await it under handler2
    with handler({get_value: handler2}):
        result = await coro

    # Result should be from handler1 (construction time), not handler2 (await time)
    assert result == "construction", (
        "Handler from construction time is used, not handler from await time"
    )


@pytest.mark.asyncio
async def test_contextvar_preserves_async_context():
    """Test that ContextVar preserves handler context across async boundaries"""

    @defop
    async def async_identity(x: int) -> int:
        raise NotHandled

    async def identity_plus_one(x: int) -> int:
        await asyncio.sleep(0.01)
        return x + 1

    async def identity_plus_ten(x: int) -> int:
        await asyncio.sleep(0.01)
        return x + 10

    async def task1():
        """Task running with handler 1"""
        with handler({async_identity: identity_plus_one}):
            # This should use identity_plus_one
            result = await async_identity(5)
            return result

    async def task2():
        """Task running with handler 2"""
        with handler({async_identity: identity_plus_ten}):
            # This should use identity_plus_ten
            result = await async_identity(5)
            return result

    # Run both tasks concurrently - each should maintain its own handler context
    results = await asyncio.gather(task1(), task2())

    assert results[0] == 6, "Task 1 used identity_plus_one handler (5 + 1)"
    assert results[1] == 15, "Task 2 used identity_plus_ten handler (5 + 10)"


@pytest.mark.asyncio
async def test_override_apply_handles_all_operations():
    """Test that overriding apply alone handles both sync and async ops"""

    @defop
    def sync_op(x: int) -> int:
        raise NotHandled

    @defop
    async def async_op(x: int) -> int:
        raise NotHandled

    # Track what gets called
    apply_calls = []

    # Override apply to capture all operations
    def custom_apply(op, *args, **kwargs):
        apply_calls.append((op.__name__, args))
        # Return different values based on op
        if op == sync_op:
            return args[0] * 2
        elif op == async_op:
            # For async ops, we see await_ being applied
            # await_ returns a coroutine, so we need to return one too
            async def async_result():
                await asyncio.sleep(0.01)
                return args[1] * 3  # args[0] is the async_op, args[1] is the actual arg
            return async_result()
        elif op.__name__ == 'await_':
            # This is the await_ operation wrapping our async_op
            # args[0] is the async operation, args[1:] are the actual arguments
            async def await_result():
                await asyncio.sleep(0.01)
                return args[1] * 3  # args[1] is the first real argument (5)
            return await_result()
        raise NotHandled

    with handler({apply: custom_apply}):
        # Call sync operation
        sync_result = sync_op(5)
        assert sync_result == 10, "Sync operation handled by apply"

        # Call async operation - apply sees await_, not async_op directly
        async_coro = async_op(5)
        assert asyncio.iscoroutine(async_coro), "Async operation returns coroutine from apply"
        async_result = await async_coro
        assert async_result == 15, "Async operation handled by apply via await_"

    # Check that apply saw both operations
    assert len(apply_calls) == 2
    assert apply_calls[0] == ("sync_op", (5,)), "Sync op went through apply"
    assert apply_calls[1] == ("await_", (async_op, 5)), "Async op went through apply as await_"


@pytest.mark.asyncio
async def test_apply_can_delegate_to_async_apply():
    """Test that custom apply can delegate async ops to async_apply

    This test shows that when you override apply, you can check if the operation
    is await_ and delegate to async_apply, which will then handle the actual
    async operation. This allows separating sync and async handling logic.
    """

    @defop
    def sync_op(x: int) -> int:
        raise NotHandled

    @defop
    async def async_op(x: int) -> int:
        raise NotHandled

    apply_calls = []
    async_apply_calls = []

    # Custom apply that delegates await_ operations to async_apply
    def custom_apply(op, *args, **kwargs):
        apply_calls.append(op.__name__)

        # Check if this is the await_ operation
        from effectful.ops.syntax import await_

        if op == await_:
            # For await_ operations, delegate to async_apply
            # This lets async_apply handle the async operation
            return async_apply(op, *args, **kwargs)
        else:
            # Handle sync operations directly
            return args[0] * 2

    # Custom async_apply that tracks calls and applies custom logic
    async def custom_async_apply(op, *args, **kwargs):
        async_apply_calls.append(op.__name__)
        await asyncio.sleep(0.01)

        # For await_ operations, extract the actual operation and apply custom logic
        from effectful.ops.syntax import await_
        if op == await_:
            # args[0] is the actual async operation (async_op)
            # args[1:] are the arguments to that operation
            actual_op = args[0]
            actual_args = args[1:]

            # Apply custom logic based on the actual operation
            if actual_op.__name__ == 'async_op':
                return actual_args[0] * 3

        # Fallback: raise NotHandled to construct a term
        raise NotHandled

    with handler({apply: custom_apply, async_apply: custom_async_apply}):
        # Sync operation goes through apply only
        sync_result = sync_op(5)
        assert sync_result == 10, "Sync op handled by custom apply"

        # Async operation goes through apply (sees await_), which delegates to async_apply
        async_coro = async_op(5)
        assert asyncio.iscoroutine(async_coro), "async_apply returns coroutine"
        async_result = await async_coro
        assert async_result == 15, "Async op handled by custom async_apply"

    # Check call tracking
    assert "sync_op" in apply_calls, "Sync op went through apply"
    assert "await_" in apply_calls, "Async op went through apply as await_"
    assert "await_" in async_apply_calls, "await_ was delegated to async_apply"



