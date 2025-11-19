"""
Futures handler for effectful - provides integration with concurrent.futures.

This module provides operations for working with concurrent.futures, allowing
effectful operations to be executed asynchronously in thread pools with
automatic preservation of interpretation context.
"""

import concurrent.futures as futures
import functools
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from effectful.ops.semantics import defop
from effectful.ops.syntax import ObjectInterpretation, defdata, implements
from effectful.ops.types import NotHandled, Term


class Executor:
    """Namespace for executor-related operations."""

    @staticmethod
    @defop  # type: ignore
    def submit[**P, T](
        task: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        """
        Submit a task for asynchronous execution.

        This operation should be handled by providing a FuturesInterpretation
        which automatically preserves the interpretation context across thread boundaries.

        :param task: The callable to execute asynchronously
        :param args: Positional arguments for the task
        :param kwargs: Keyword arguments for the task
        :return: A Future representing the asynchronous computation

        Example:
            >>> from concurrent.futures import ThreadPoolExecutor
            >>> from effectful.handlers.futures import ThreadPoolFuturesInterpretation
            >>> from effectful.ops.semantics import handler
            >>>
            >>> pool = ThreadPoolExecutor()
            >>> with handler(ThreadPoolFuturesInterpretation(pool)):
            >>>     future = Executor.submit(my_function, arg1, arg2)
        """
        raise NotHandled

    @staticmethod
    @defop
    def map[T, R](
        func: Callable[[T], R],
        *iterables: Iterable[T],
        timeout: float | None = None,
        chunksize: int = 1,
    ) -> Iterable[R]:
        """
        Map a function over iterables, executing asynchronously.

        Returns an iterator yielding results as they complete. Equivalent to
        map(func, *iterables) but executes asynchronously.

        This operation should be handled by providing a FuturesInterpretation
        which automatically preserves the interpretation context across thread boundaries.

        :param func: The function to map over the iterables
        :param iterables: One or more iterables to map over
        :param timeout: Maximum time to wait for a result (default: None)
        :param chunksize: Size of chunks for ProcessPoolExecutor (default: 1)
        :return: An iterator yielding results

        Example:
            >>> from effectful.handlers.futures import ThreadPoolFuturesInterpretation
            >>> from effectful.ops.semantics import handler
            >>>
            >>> def square(x):
            >>>     return x ** 2
            >>>
            >>> with handler(ThreadPoolFuturesInterpretation()):
            >>>     results = list(Executor.map(square, range(10)))
            >>>     print(results)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
        """
        raise NotHandled


class FuturesInterpretation(ObjectInterpretation):
    """
    Base interpretation for concurrent.futures executors.

    This interpretation automatically preserves the effectful interpretation context
    when submitting tasks to worker threads, ensuring that effectful operations
    work correctly across thread boundaries.
    """

    def __init__(self, executor: futures.Executor):
        """
        Initialize the futures interpretation.

        :param executor: The executor to use (ThreadPoolExecutor or ProcessPoolExecutor)
        """
        super().__init__()
        self.executor: futures.Executor = executor

    def shutdown(self, *args, **kwargs):
        self.executor.shutdown(*args, **kwargs)

    @implements(Executor.submit)
    def submit(self, task: Callable, *args, **kwargs) -> Future:
        """
        Submit a task to the executor with automatic context preservation.

        Captures the current interpretation context and ensures it is restored
        in the worker thread before executing the task.
        """
        from effectful.internals.runtime import get_interpretation, interpreter

        # Capture the current interpretation context
        context = get_interpretation()

        # Submit the wrapped task to the underlying executor
        return self.executor.submit(interpreter(context)(task), *args, **kwargs)

    @implements(Executor.map)
    def map(self, func: Callable, *iterables, timeout=None, chunksize=1):
        """
        Map a function over iterables with automatic context preservation.

        Captures the current interpretation context and ensures it is restored
        in each worker thread before executing the function.
        """
        from effectful.internals.runtime import get_interpretation, interpreter

        # Capture the current interpretation context
        context = get_interpretation()

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            # Restore the interpretation context in the worker thread
            with interpreter(context):
                return func(*args, **kwargs)

        # Call the executor's map with the wrapped function
        return self.executor.map(
            wrapped_func, *iterables, timeout=timeout, chunksize=chunksize
        )


class ThreadPoolFuturesInterpretation(FuturesInterpretation):
    """
    Interpretation for ThreadPoolExecutor with automatic context preservation.

    Example:
        >>> from concurrent.futures import ThreadPoolExecutor, Future
        >>> from effectful.ops.semantics import defop, handler
        >>> from effectful.handlers.futures import Executor, ThreadPoolFuturesInterpretation
        >>>
        >>> @defop
        >>> def async_pow(n: int, k: int) -> Future[int]:
        >>>     return Executor.submit(pow, n, k)
        >>>
        >>> pool = ThreadPoolExecutor()
        >>> with handler(ThreadPoolFuturesInterpretation(pool)):
        >>>     result = async_pow(2, 10).result()
        >>>     print(result)  # 1024
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize with a ThreadPoolExecutor.

        :param max_workers: Maximum number of worker threads (default: None, uses default from ThreadPoolExecutor)
        """
        super().__init__(ThreadPoolExecutor(*args, **kwargs))


type ReturnOptions = Literal["All_COMPLETED", "FIRST_COMPLETED", "FIRST_EXCEPTION"]


@dataclass(frozen=True)
class DoneAndNotDoneFutures[T]:
    done: set[Future[T]]
    not_done: set[Future[T]]


@defdata.register(DoneAndNotDoneFutures)
class _DoneAndNotDoneFuturesTerm[T](Term[DoneAndNotDoneFutures[T]]):
    """Term representing a DoneAndNotDoneFutures result."""

    def __init__(self, op, *args, **kwargs):
        self._op = op
        self._args = args
        self._kwargs = kwargs

    @property
    def op(self):
        return self._op

    @property
    def args(self):
        return self._args

    @property
    def kwargs(self):
        return self._kwargs

    @defop  # type: ignore[prop-decorator]
    @property
    def done(self) -> set[Future[T]]:
        """Get the set of done futures."""
        if not isinstance(self, Term):
            return self.done
        else:
            raise NotHandled

    @defop  # type: ignore[prop-decorator]
    @property
    def not_done(self) -> set[Future[T]]:
        """Get the set of not done futures."""
        if not isinstance(self, Term):
            return self.not_done
        else:
            raise NotHandled


@defop
def wait[T](
    fs: Iterable[Future[T]],
    timeout: int | None = None,
    return_when: ReturnOptions = futures.ALL_COMPLETED,  # type: ignore
) -> DoneAndNotDoneFutures[T]:
    if (
        isinstance(timeout, Term)
        or isinstance(return_when, Term)
        or any(not isinstance(t, Future) for t in fs)
    ):
        raise NotHandled
    return futures.wait(fs, timeout, return_when)  # type: ignore


@defop
def as_completed[T](
    fs: Iterable[Future[T]],
    timeout: int | None = None,
) -> Iterable[Future[T]]:
    if isinstance(timeout, Term) or any(isinstance(t, Term) for t in fs):
        raise NotHandled
    return futures.as_completed(fs, timeout)
