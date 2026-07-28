"""Choreographic programming for multi-agent LLM systems.

Write a single ``async`` function describing how agents interact from a global
perspective, then run it with automatic endpoint projection (EPP). Every agent
runs that same function as its own `asyncio.Task`, and inter-agent
communication falls out of ordinary asyncio primitives.

## How it works

Each `step` in the choreography is assigned an incrementing step ID. Because
every agent runs the same program, and every step's result is shared, all
agents allocate the same IDs in the same order. Each ID names an
`asyncio.Future`; for a given step, `EndpointProjection` either:

- **executes** it and resolves the future, if the step's template belongs to
  this agent; or
- **awaits** that future, if it belongs to another agent.

That is the whole coordination mechanism. A future is exactly a write-once,
read-by-many cell, which is what a step result is: the architect computes step
0 once, and every other agent reads it. Waiting is event-driven -- nothing
polls, and there is no interval to tune.

`scatter` needs the one thing futures don't provide, namely handing each item
to exactly one of several workers. That is a work queue, so it uses one:
an `asyncio.Queue` of item indices that the agents in the pool drain with
`get_nowait`. Whoever is free takes the next item, which balances load by
construction.

Results live in memory, so by default an interrupted run starts over. Give
`Choreography` a `StepLog` and each step is written to SQLite as it completes;
a later run replays those results and resumes at the first step that never
finished. (Agent *history* is a separate matter -- give an
`~effectful.handlers.llm.template.Agent` an ``agent_id`` and install
`~effectful.handlers.llm.completions.SQLitePersister` to checkpoint it.)

## Why the program is async, and where the threads went

`effectful`'s handler stack is synchronous, top to bottom: `Template.__apply__`,
`fwd`, and everything in `effectful.handlers.llm.completions` down to
`litellm.completion` are blocking calls. Two consequences shape this module.

*A handler's body must run synchronously -- but it may return an awaitable.*
`coproduct` wraps every handler in a synchronous continuation (see
`effectful.internals.runtime._set_prompt`), and `Operation.__call__` binds
`~effectful.ops.semantics.fwd` around the call itself, so an ``async def``
handler would return an un-awaited coroutine whose body later ran outside both
bindings. `step`, `call` and `scatter` are therefore ordinary `Operation`s
whose implementations return coroutines rather than being coroutines, which is
also what lets a step ID be allocated while `step` is being called. It is why
the choreography spells its steps out with ``await step(...)`` rather than
calling ``architect.plan(spec)`` directly.

*A template call must still run on a thread.* `call` hands the blocking call to
a worker thread and awaits it, so an agent waiting on a peer costs a suspended
coroutine rather than a parked thread. Note that the naive alternative --
wrapping each agent's whole program in `asyncio.to_thread` -- deadlocks here:
agents block on each other, `asyncio.to_thread` draws from a default executor
of ``min(32, cpu_count + 4)`` workers, and waiting agents hold workers that the
agents they wait for can never get. `Choreography` sizes its own executor to
the number of agents for the same reason.

## Primitives

`step`
    A choreography step: one template call, executed by its owner and shared
    with everyone else.
`call`
    Execution on a worker thread with no coordination. Use it for work that is
    already covered by an enclosing step -- notably inside `scatter`.
`scatter`
    Distribute items across a pool of same-role agents, each item going to
    whichever agent is free.

Several scatters run concurrently with `asyncio.gather`; agents belonging to
more than one group work on all of them at once::

    specs, tests, proofs = await asyncio.gather(
        scatter(blocks, spec_writer, lambda w, b: call(w.write_spec, b)),
        scatter(blocks, tester, lambda t, b: call(t.write_tests, b)),
        scatter(blocks, prover, lambda p, b: call(p.prove, b)),
    )

Step IDs are allocated when `step`/`scatter` is *called*, not when the returned
awaitable is *awaited*, so the IDs in a `asyncio.gather` are deterministic and
agree across agents.

## Example -- sequential choreography with a review loop

::

    from typing import Literal, TypedDict

    from effectful.handlers.llm import Agent, Template
    from effectful.handlers.llm.completions import LiteLLMProvider, RetryLLMHandler
    from effectful.handlers.llm.choreographies import Choreography, call, scatter, step
    from effectful.ops.semantics import handler

    class ReviewResult(TypedDict):
        verdict: Literal["PASS", "NEEDS_FIXES"]
        feedback: str

    class Architect(Agent):
        \"\"\"You are a software architect.\"\"\"

        @Template.define
        def plan_modules(self, project_spec: str) -> str:
            \"\"\"Break this project into modules: {project_spec}\"\"\"

    class Coder(Agent):
        \"\"\"You are a Python developer.\"\"\"

        @Template.define
        def implement_module(self, spec: str) -> str:
            \"\"\"Implement the module: {spec}\"\"\"

    class Reviewer(Agent):
        \"\"\"You are a code reviewer.\"\"\"

        @Template.define
        def review_code(self, code: str) -> ReviewResult:
            \"\"\"Review this code: {code}\"\"\"

    async def build_codebase(project_spec, architect, coder, reviewer):
        plan = await step(architect.plan_modules, project_spec)
        code = await step(coder.implement_module, plan)
        while True:
            result = await step(reviewer.review_code, code)
            if result["verdict"] == "PASS":
                return code
            code = await step(coder.implement_module, result["feedback"])

    architect = Architect(agent_id="architect")
    coder = Coder(agent_id="coder")
    reviewer = Reviewer(agent_id="reviewer")

    choreo = Choreography(build_codebase, agents=[architect, coder, reviewer])

    # Handlers come from the enclosing context -- every agent task inherits
    # them, and so does every worker thread they call into.
    with handler(LiteLLMProvider(model="gpt-4o-mini")), handler(RetryLLMHandler()):
        result = choreo.run(
            project_spec="Build a URL slugify library",
            architect=architect,
            coder=coder,
            reviewer=reviewer,
        )

## Example -- parallel scatter across multiple coders

::

    async def build_parallel(project_spec, architect, coder, reviewer):
        plan = await step(architect.plan_modules, project_spec)
        # Each module goes to whichever coder is free.
        codes = await scatter(
            plan["modules"], coder,
            lambda c, mod: call(c.implement_module, str(mod)),
        )
        return [await step(reviewer.review_code, code) for code in codes]

    choreo = Choreography(
        build_parallel, agents=[architect, coder1, coder2, coder3, reviewer]
    )
    with handler(LiteLLMProvider(model="gpt-4o-mini")), handler(RetryLLMHandler()):
        # Pass a list for a role -- scatter distributes across all three coders.
        reviews = choreo.run(
            project_spec="Build textkit with slugify, wrap, and redact modules",
            architect=architect,
            coder=[coder1, coder2, coder3],
            reviewer=reviewer,
        )

"""

import asyncio
import concurrent.futures
import contextlib
import contextvars
import functools
import os
import pathlib
import pickle
import sqlite3
import typing
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from effectful.handlers.llm.template import Agent
from effectful.ops.semantics import handler
from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.types import Operation


class ChoreographyError(Exception):
    """Raised when a choreography fails because one of its agents failed."""


# ── Resumption ────────────────────────────────────────────────────


class StepLog:
    """A durable record of completed steps, so an interrupted run can resume.

    Pass one to `Choreography` and each step is written to SQLite as it
    finishes. A later run over the same log replays those results instead of
    calling the model again, and picks up at the first step that never
    completed::

        choreo = Choreography(
            build_codebase,
            agents=[architect, coder, reviewer],
            handlers=[LiteLLMProvider(model="gpt-4o-mini")],
            log=StepLog("./state/steps.db"),
        )

    Only successful steps are recorded, so a step that failed or was
    interrupted simply runs again. Scatter items are recorded individually:
    interrupt a scatter over ten modules after six and the next run implements
    the remaining four.

    .. warning::

        Steps are identified by position, so a log only makes sense for the
        program that wrote it. Editing the choreography shifts the step IDs
        and the recorded results land on the wrong steps -- use `clear`, or a
        fresh path, whenever the program changes.

    Results are pickled, which is what lets a step return a dataclass or any
    other decoded value rather than only JSON. The log is a cache of your own
    run and is read back with the same trust as
    `~effectful.handlers.llm.completions.SQLitePersister`'s checkpoints.

    The recorded steps are readable on their own, which is the easiest way to
    see how far a run got:

    >>> import pathlib, tempfile
    >>> log = StepLog(pathlib.Path(tempfile.mkdtemp()) / "steps.db")
    >>> log.load()
    {}
    >>> log.record("step-0000", {"verdict": "PASS"})
    >>> log.load()
    {'step-0000': {'verdict': 'PASS'}}
    >>> log.clear()
    >>> log.load()
    {}

    Args:
        path: Path to the SQLite database file.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS steps ("
                "  id TEXT PRIMARY KEY, result BLOB NOT NULL"
                ")"
            )

    def _connect(self) -> contextlib.AbstractContextManager[sqlite3.Connection]:
        """A connection that closes on exit, in autocommit mode.

        Autocommit because every write here is a single statement: there is
        nothing to group into a transaction, and a step's result is durable the
        moment it is written.
        """
        return contextlib.closing(sqlite3.connect(str(self.path), isolation_level=None))

    def load(self) -> dict[str, Any]:
        """Every recorded step, by step ID."""
        with self._connect() as conn:
            rows = conn.execute("SELECT id, result FROM steps").fetchall()
        return {step_id: pickle.loads(blob) for step_id, blob in rows}

    def record(self, step_id: str, result: Any) -> None:
        """Record *result* as the outcome of *step_id*."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO steps (id, result) VALUES (?, ?)",
                (step_id, pickle.dumps(result)),
            )

    def clear(self) -> None:
        """Forget every recorded step, so the next run starts over."""
        with self._connect() as conn:
            conn.execute("DELETE FROM steps")


# ── Shared step state ─────────────────────────────────────────────


class _Steps:
    """The shared state of one choreography run.

    Two dictionaries, keyed by step ID: a `asyncio.Future` per step, holding
    the result its owner computes and every other agent awaits, and a
    `asyncio.Queue` per scatter, holding the item indices its pool drains.

    Both accessors are get-or-create, and neither awaits, so concurrent agents
    cannot interleave inside them: whichever agent reaches a step first creates
    its cell and the rest find it. That also means one of these belongs to one
    event loop, which is why `Choreography` makes a fresh one per run.
    """

    def __init__(self, log: StepLog | None = None) -> None:
        self._results: dict[str, asyncio.Future] = {}
        self._work: dict[str, asyncio.Queue[int]] = {}
        self._log = log

    def result(self, step_id: str) -> asyncio.Future:
        """The future holding *step_id*'s result."""
        future = self._results.get(step_id)
        if future is None:
            future = self._results[step_id] = asyncio.get_running_loop().create_future()
        return future

    def resolve(self, step_id: str, value: Any) -> None:
        """Publish *value* as *step_id*'s result, recording it first.

        Recording before publishing keeps the log ahead of the run: a crash
        between the two costs one step's re-execution on the next run, whereas
        the other order would report a step as done that no later run knows
        about.
        """
        if self._log is not None:
            self._log.record(step_id, value)
        self.result(step_id).set_result(value)

    def replay(self) -> int:
        """Pre-resolve the steps an earlier run recorded, and return how many."""
        if self._log is None:
            return 0
        recorded = self._log.load()
        for step_id, value in recorded.items():
            future = self.result(step_id)
            if not future.done():
                future.set_result(value)
        return len(recorded)

    def work(self, step_id: str, results: Sequence[asyncio.Future]) -> asyncio.Queue:
        """The queue of item indices for the scatter at *step_id*.

        Items already resolved -- replayed from a previous run -- are left out,
        so a resumed scatter only distributes what is still outstanding.
        """
        queue = self._work.get(step_id)
        if queue is None:
            queue = self._work[step_id] = asyncio.Queue()
            for index, result in enumerate(results):
                if not result.done():
                    queue.put_nowait(index)
        return queue


def _fail(future: asyncio.Future, error: BaseException) -> None:
    """Fail *future*, so agents awaiting it see the error instead of hanging."""
    if future.done():
        return
    future.set_exception(error)
    # The agents that would have retrieved this are normally cancelled by the
    # task group before they get the chance, and asyncio complains at
    # collection time about an exception nobody read. Read it here: the error
    # still reaches any live waiter, and the failure is reported by the agent
    # that actually raised it.
    future.exception()


# ── Endpoint projection ───────────────────────────────────────────


@Operation.define
def call[**P, T](fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> Awaitable[T]:
    """Await a blocking call -- usually a `Template` -- on a worker thread.

    No coordination: the call is not a choreography step, and no other agent
    learns its result. Use it for work already covered by an enclosing step,
    most importantly for the per-item function passed to `scatter`.

    Unhandled, this is `asyncio.to_thread`. `EndpointProjection` handles it by
    using the choreography's own thread pool instead.
    """
    return asyncio.to_thread(fn, *args, **kwargs)


@Operation.define
def step[**P, T](
    template: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> Awaitable[T]:
    """Take one step of a choreography, and return an awaitable for its result.

    Under `EndpointProjection`, the agent that owns *template* executes the
    step while the others await its result; a step recorded by an earlier run
    (see `StepLog`) returns without calling the model at all. A template bound
    to no agent is executed by every agent.

    The step ID is allocated when `step` is called, not when its result is
    awaited, so concurrent steps still get the same IDs in the same order on
    every agent.

    Unhandled, this is just `call`.
    """
    return call(template, *args, **kwargs)


@Operation.define
def scatter[A: Agent, T, U](
    items: Sequence[T],
    agent: A | Sequence[A],
    fn: Callable[[A, T], Awaitable[U]],
) -> Awaitable[list[U]]:
    """Distribute *items* over *agent* by calling ``await fn(agent, item)``.

    *agent* may be a single agent or a pool of same-role agents. Under
    `EndpointProjection` the pool draws items from a shared `asyncio.Queue`
    until it is empty, which balances load by construction: a fast agent takes
    more items.

    Results come back in *items* order, whoever computed them.

    Unhandled, items are processed sequentially, round-robin over the pool.

    .. warning::

        *fn* should only touch the agent it is handed, with `call`. A `step`
        inside *fn* is rejected, and other agents' templates are not supported.
    """
    return _scatter_sequentially(items, agent, fn)


async def _scatter_sequentially[A: Agent, T, U](
    items: Sequence[T],
    agent: A | Sequence[A],
    fn: Callable[[A, T], Awaitable[U]],
) -> list[U]:
    agents = [agent] if isinstance(agent, Agent) else list(agent)
    return [await fn(agents[i % len(agents)], item) for i, item in enumerate(items)]


def _no_step_inside_scatter(template: Callable, *args, **kwargs):
    """The interpretation of `step` in force while a scatter item is running."""
    raise RuntimeError(
        "step() cannot be used inside scatter(): its per-item work is not a "
        "choreography step, and allocating step IDs there would put the agents "
        "out of sync. Use call() instead."
    )


class EndpointProjection(ObjectInterpretation):
    """Projects a choreographic program onto a single agent.

    Install it like any other handler -- ``with handler(projection):`` -- and
    `step`, `call` and `scatter` route through it for the current task.

    Each implementation runs synchronously and *returns* an awaitable rather
    than being a coroutine function itself. That is what keeps step IDs in
    lockstep, since the ID is allocated while `step` is being called, and it
    is also what keeps `~effectful.ops.semantics.fwd` meaningful: `effectful`
    binds it around the synchronous call, so a handler that returned an
    un-awaited coroutine would run its body after that binding was gone.

    Args:
        agent: The agent this projection speaks for.
        steps: The run's shared step state. Every agent in a choreography must
            be given the same one -- it is how they exchange results.
        agent_ids: The IDs of every agent in the run, used to reject a step
            belonging to an agent that is not participating. ``None`` skips
            the check.
        executor: Thread pool for blocking template calls. ``None`` uses
            asyncio's default executor, which is only safe when agents do not
            wait on each other -- `Choreography` always passes its own.
    """

    def __init__(
        self,
        agent: Agent,
        steps: "_Steps",
        agent_ids: frozenset[str] | None = None,
        executor: concurrent.futures.Executor | None = None,
    ) -> None:
        self._agent = agent
        self._agent_id = agent.__agent_id__
        self._steps = steps
        self._agent_ids = agent_ids
        self._executor = executor
        self._step = 0

    @property
    def agent(self) -> Agent:
        """The agent this projection speaks for."""
        return self._agent

    def _next_step(self) -> str:
        step_id = f"step-{self._step:04d}"
        self._step += 1
        return step_id

    @implements(call)
    def _call(self, fn: Callable, *args, **kwargs) -> Awaitable:
        return self._in_thread(fn, *args, **kwargs)

    @implements(step)
    def _step(self, template: Callable, *args, **kwargs) -> Awaitable:
        return self._run_step(self._next_step(), template, args, kwargs)

    @implements(scatter)
    def _scatter_items(self, items, agent, fn) -> Awaitable:
        return self._scatter(self._next_step(), items, agent, fn)

    async def _in_thread[T](self, fn: Callable[..., T], *args, **kwargs) -> T:
        """Await *fn* on a worker thread, carrying the current context along.

        The context copy is what puts the agent's `effectful` handler stack --
        provider, retries, persistence -- in scope inside the worker.
        """
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(
            self._executor, functools.partial(ctx.run, fn, *args, **kwargs)
        )

    async def _run_step(
        self, step_id: str, template: Callable, args: tuple, kwargs: dict
    ) -> Any:
        agent = getattr(template, "__agent__", None)

        if agent is None:
            # Unbound template: not owned by anyone, so every agent runs it.
            return await call(template, *args, **kwargs)

        if self._agent_ids is not None and agent.__agent_id__ not in self._agent_ids:
            raise ChoreographyError(
                f"{template.__name__}() belongs to agent "
                f"{agent.__agent_id__!r}, which is not part of this "
                f"choreography -- no one would ever run it."
            )

        result = self._steps.result(step_id)
        if agent.__agent_id__ != self._agent_id:
            return await result
        if result.done():
            # Recorded by an earlier run; see `StepLog`.
            return result.result()

        try:
            value = await call(template, *args, **kwargs)
        except Exception as e:
            _fail(result, e)
            raise
        self._steps.resolve(step_id, value)
        return value

    async def _scatter[A: Agent, T, U](
        self,
        step_id: str,
        items: Sequence[T],
        agent: A | Sequence[A],
        fn: Callable[[A, T], Awaitable[U]],
    ) -> list[U]:
        agents = [agent] if isinstance(agent, Agent) else list(agent)
        results = [self._steps.result(f"{step_id}:{i}") for i in range(len(items))]
        me = typing.cast(A, self._agent)

        if self._agent_id in {a.__agent_id__ for a in agents}:
            work = self._steps.work(step_id, results)
            while True:
                try:
                    index = work.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    # Rebinding `step` is what stops per-item work from
                    # allocating step IDs; the binding lasts exactly as long as
                    # the item does.
                    with handler({step: _no_step_inside_scatter}):
                        value = await fn(me, items[index])
                except Exception as e:
                    _fail(results[index], e)
                    raise
                self._steps.resolve(f"{step_id}:{index}", value)

        return [await result for result in results]


# ── Choreography runner ───────────────────────────────────────────


def _first_exception(group: BaseExceptionGroup) -> BaseException:
    """The first leaf exception of a possibly nested exception group."""
    exc = group.exceptions[0]
    return _first_exception(exc) if isinstance(exc, BaseExceptionGroup) else exc


class Choreography:
    """Run a choreographic program with endpoint projection.

    Every agent runs *program* as its own `asyncio.Task`; `EndpointProjection`
    is what makes each of those tasks behave differently. Blocking template
    calls go to a thread pool sized to the number of agents, so no agent can be
    starved by another's model call.

    The tasks run in an `asyncio.TaskGroup`, which supplies the parts the
    threaded version had to build by hand: the first failure cancels the other
    agents, and the failure propagates to the caller. Cancellation cannot
    interrupt an LLM call that is already in flight on a worker thread, so a
    failing run waits for those to return before it raises.

    Handlers are taken from the surrounding context, exactly as anywhere else
    in `effectful`: install them with `~effectful.ops.semantics.handler`
    around the run and every agent task inherits them, as does every worker
    thread the agents call into. Nothing needs to be handed to the
    choreography, which is also why a script run under
    `effectful.handlers.llm.harness` needs no handler code of its own.

    Without a *log*, each run starts from a clean slate: results live in
    memory for the duration of the run, so re-running a choreography
    re-executes it. With one, completed steps are replayed instead.

    Args:
        program: The choreographic ``async`` function. All agents run it.
        agents: The agents participating in the choreography.
        log: Where to record completed steps, so an interrupted run can be
            resumed by running it again. ``None`` keeps everything in memory.

    Example::

        choreo = Choreography(build_codebase, agents=[architect, coder, reviewer])

        with handler(LiteLLMProvider(model="gpt-4o-mini")), handler(RetryLLMHandler()):
            result = choreo.run(
                project_spec="Build a library...",
                architect=architect,
                coder=coder,
                reviewer=reviewer,
            )
    """

    def __init__(
        self,
        program: Callable[..., Awaitable[Any]],
        agents: Sequence[Agent],
        log: StepLog | None = None,
    ) -> None:
        self.program = program
        self.agents = list(agents)
        self.log = log
        self._steps = _Steps(log)

    def replay(self) -> int:
        """Pre-resolve the steps a previous run recorded in `log`.

        `run_async` calls this before starting the agents; call it yourself
        only when driving `projection` objects by hand. Must run inside the
        event loop the agents will use.
        """
        return self._steps.replay()

    def projection(
        self,
        agent: Agent,
        executor: concurrent.futures.Executor | None = None,
    ) -> EndpointProjection:
        """The `EndpointProjection` for one agent.

        Projections handed out between runs share this choreography's current
        step state, so they can drive a program together::

            choreo.replay()  # only if resuming from a StepLog

            async def drive(agent):
                with handler(choreo.projection(agent)):
                    return await choreo.program(**kwargs)

            async with asyncio.TaskGroup() as group:
                for agent in choreo.agents:
                    group.create_task(drive(agent))

        Doing it by hand is also how you get a handler *over* the projection --
        one that intercepts `step` and forwards with
        `~effectful.ops.semantics.fwd`. `run_async` installs the projection
        inside each agent task, where it wins over anything wrapped around the
        run, so a wrapping handler has to be installed after it.
        """
        return EndpointProjection(
            agent,
            self._steps,
            frozenset(a.__agent_id__ for a in self.agents),
            executor=executor,
        )

    async def _agent_main(
        self,
        agent: Agent,
        executor: concurrent.futures.Executor,
        kwargs: dict[str, Any],
    ) -> Any:
        with handler(self.projection(agent, executor=executor)):
            try:
                return await self.program(**kwargs)
            except (asyncio.CancelledError, ChoreographyError):
                raise
            except Exception as e:
                raise ChoreographyError(
                    f"Agent {agent.__agent_id__!r} failed: {e}"
                ) from e

    async def run_async(self, **kwargs: Any) -> Any:
        """Run the choreography to completion.

        Keyword arguments are forwarded to the choreographic function. All
        agents compute the same result; that result is returned.

        Raises:
            ChoreographyError: If any agent fails.
        """
        # Fresh state per run: futures belong to the loop that created them.
        self._steps = _Steps(self.log)
        self.replay()

        tasks: list[asyncio.Task] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(self.agents)), thread_name_prefix="choreo"
        ) as executor:
            try:
                async with asyncio.TaskGroup() as group:
                    tasks = [
                        group.create_task(
                            self._agent_main(agent, executor, kwargs),
                            name=f"choreo-{agent.__agent_id__}",
                        )
                        for agent in self.agents
                    ]
            except BaseExceptionGroup as group_error:
                raise _first_exception(group_error)

        return tasks[0].result()

    def run(self, **kwargs: Any) -> Any:
        """Run the choreography from synchronous code.

        Equivalent to ``asyncio.run(choreo.run_async(**kwargs))``; call
        `run_async` directly from inside a running event loop.
        """
        return asyncio.run(self.run_async(**kwargs))
