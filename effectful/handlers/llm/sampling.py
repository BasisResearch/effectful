from collections import Counter
from concurrent import futures
from concurrent.futures.thread import ThreadPoolExecutor

from effectful.handlers.llm import Template
from effectful.internals.runtime import get_interpretation, interpreter
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements


class KAheadSampler[**P, T](ObjectInterpretation):
    no_voters: int
    k: int
    """Number of votes ahead before an answer is accepted"""
    votes: Counter[T] = Counter()

    def __init__(self, no_voters: int = 6, k: int = 3):
        self.no_voters = no_voters
        self.k = k

    @implements(Template.__apply__)
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
