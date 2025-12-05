import functools
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent import futures
from concurrent.futures.thread import ThreadPoolExecutor

from effectful.handlers.llm import Template
from effectful.handlers.llm.providers import completion, tool_call
from effectful.internals.runtime import get_interpretation, interpreter
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, implements


class KAheadSampler[**P, T](ObjectInterpretation):
    no_voters: int
    k: int
    """Number of votes ahead before an answer is accepted"""
    votes: Counter[T] = Counter()

    def __init__(self, no_voters: int = 6, k: int = 3):
        self.no_voters = no_voters
        self.k = k

    @implements(Template.__call__)
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        executor = ThreadPoolExecutor()
        intp = get_interpretation()
        tasks = [
            executor.submit(interpreter(intp)(fwd), *args, **kwargs)
            for _ in range(self.no_voters)
        ]

        def n_votes_ahead():
            match self.votes.most_common(2):
                case [[_, v1], [_, v2]]:
                    return v1 >= v2 + self.k
                case [[_, v1]]:
                    return v1 >= self.k
                case _:
                    return False

        while not n_votes_ahead():
            done, remain = futures.wait(tasks, return_when=futures.FIRST_COMPLETED)
            tasks = list(remain)
            for fut in done:
                res = fut.result()
                self.votes[res] += 1
                tasks.append(executor.submit(interpreter(intp)(fwd), *args, **kwargs))
        executor.shutdown()
        return self.votes.most_common(1)[0][0]


def sample[**P, T](template: Template[P, T], n: int) -> Callable[P, Sequence[T]]:
    """sample returns a function with the same signature as `template` except
    that `n` samples are returned.

    When computing a batch of samples, calls to `completion` (and handlers of
    `completion`) proceed in parallel, but other calls (e.g. tool calls) proceed
    synchronously.

    """
    lock = threading.Lock()

    def _completion(*args, **kwargs):
        lock.release()
        result = fwd()
        lock.acquire()
        return result

    def _tool_call(*args, **kwargs):
        lock.acquire()
        try:
            result = fwd()
        except Exception as e:
            lock.release()
            raise e
        return result

    @functools.wraps(template)
    @handler({completion: _completion, tool_call: _tool_call})
    def wrapper(*args, **kwargs):
        with ThreadPoolExecutor() as executor:

            @interpreter(get_interpretation())
            def do_work():
                lock.acquire()
                try:
                    result = template(*args, **kwargs)
                finally:
                    assert lock.locked()
                    lock.release()
                return result

            tasks = [executor.submit(do_work) for _ in range(n)]
            completed = futures.wait(tasks, return_when=futures.ALL_COMPLETED)
            return [t.result() for t in completed.done]

    return wrapper
