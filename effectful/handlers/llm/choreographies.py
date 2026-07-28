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

A choreography lives and dies with the process: results are in memory, so a
run that is interrupted starts over. (Agent *history* is a separate matter --
give an `~effectful.handlers.llm.template.Agent` an ``agent_id`` and install
`~effectful.handlers.llm.completions.SQLitePersister` to checkpoint it.)

## Why the program is async, and where the threads went

`effectful`'s handler stack is synchronous, top to bottom: `Template.__apply__`,
`fwd`, and everything in `effectful.handlers.llm.completions` down to
`litellm.completion` are blocking calls. Two consequences shape this module.

*A handler may never be a coroutine function.* `coproduct` wraps every handler
in a synchronous continuation (see `effectful.internals.runtime._set_prompt`),
so an ``async def`` handler would return an un-awaited coroutine, the wrapper's
``with handler(...)`` block would exit, and the body would later run outside the
interpretation that defines its `fwd`. `EndpointProjection` is therefore *not*
an `effectful.ops.syntax.ObjectInterpretation`; it is an ordinary object held
in a `contextvars.ContextVar` and driven explicitly by `step` and `scatter`.
That is also why the choreography spells its steps out with ``await`` rather
than calling ``architect.plan(spec)`` directly.

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

    choreo = Choreography(
        build_codebase,
        agents=[architect, coder, reviewer],
        handlers=[LiteLLMProvider(model="gpt-4o-mini"), RetryLLMHandler()],
    )
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
        build_parallel,
        agents=[architect, coder1, coder2, coder3, reviewer],
        handlers=[LiteLLMProvider(model="gpt-4o-mini"), RetryLLMHandler()],
    )
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
import typing
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from effectful.handlers.llm.template import Agent
from effectful.ops.semantics import handler
from effectful.ops.syntax import ObjectInterpretation
from effectful.ops.types import Interpretation


class ChoreographyError(Exception):
    """Raised when a choreography fails because one of its agents failed."""


# ── Shared step state ─────────────────────────────────────────────


class _StepLog:
    """The shared state of one choreography run.

    Two dictionaries, keyed by step ID: a `asyncio.Future` per step, holding
    the result its owner computes and every other agent awaits, and a
    `asyncio.Queue` per scatter, holding the item indices its pool drains.

    Both accessors are get-or-create, and neither awaits, so concurrent agents
    cannot interleave inside them: whichever agent reaches a step first creates
    its cell and the rest find it. That also means one log belongs to one event
    loop, which is why `Choreography` makes a fresh one per run.
    """

    def __init__(self) -> None:
        self._results: dict[str, asyncio.Future] = {}
        self._work: dict[str, asyncio.Queue[int]] = {}

    def result(self, step_id: str) -> asyncio.Future:
        """The future holding *step_id*'s result."""
        future = self._results.get(step_id)
        if future is None:
            future = self._results[step_id] = asyncio.get_running_loop().create_future()
        return future

    def work(self, step_id: str, count: int) -> asyncio.Queue[int]:
        """The queue of item indices for the scatter at *step_id*."""
        queue = self._work.get(step_id)
        if queue is None:
            queue = self._work[step_id] = asyncio.Queue()
            for index in range(count):
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


_PROJECTION: contextvars.ContextVar["EndpointProjection | None"] = (
    contextvars.ContextVar("effectful_choreography_projection", default=None)
)

_IN_SCATTER: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "effectful_choreography_in_scatter", default=False
)


class EndpointProjection:
    """Projects a choreographic program onto a single agent.

    Not an `effectful.ops.syntax.ObjectInterpretation`: it drives ``await``
    points, and `effectful` handlers must be synchronous (see the module
    docstring). Activate it for the current task with `activate`, after which
    `step` and `scatter` route through it.

    Args:
        agent: The agent this projection speaks for.
        steps: The run's shared `_StepLog`. Every agent in a choreography must
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
        steps: "_StepLog",
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

    @contextlib.contextmanager
    def activate(self):
        """Make this the active projection for the current task."""
        token = _PROJECTION.set(self)
        try:
            yield self
        finally:
            _PROJECTION.reset(token)

    def _next_step(self) -> str:
        step_id = f"step-{self._step:04d}"
        self._step += 1
        return step_id

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
            return await self._in_thread(template, *args, **kwargs)

        if self._agent_ids is not None and agent.__agent_id__ not in self._agent_ids:
            raise ChoreographyError(
                f"{template.__name__}() belongs to agent "
                f"{agent.__agent_id__!r}, which is not part of this "
                f"choreography -- no one would ever run it."
            )

        result = self._steps.result(step_id)
        if agent.__agent_id__ != self._agent_id:
            return await result

        try:
            value = await self._in_thread(template, *args, **kwargs)
        except Exception as e:
            _fail(result, e)
            raise
        result.set_result(value)
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
            work = self._steps.work(step_id, len(items))
            while True:
                try:
                    index = work.get_nowait()
                except asyncio.QueueEmpty:
                    break
                token = _IN_SCATTER.set(True)
                try:
                    value = await fn(me, items[index])
                except Exception as e:
                    _fail(results[index], e)
                    raise
                finally:
                    _IN_SCATTER.reset(token)
                results[index].set_result(value)

        return [await result for result in results]


# ── Choreography primitives ───────────────────────────────────────


def call[**P, T](fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> Awaitable[T]:
    """Await a blocking call -- usually a `Template` -- on a worker thread.

    No coordination: the call is not a choreography step, and no other agent
    learns its result. Use it for work already covered by an enclosing step,
    most importantly for the per-item function passed to `scatter`. Outside a
    choreography it is `asyncio.to_thread`.
    """
    projection = _PROJECTION.get()
    if projection is None:
        return asyncio.to_thread(fn, *args, **kwargs)
    return projection._in_thread(fn, *args, **kwargs)


def step[**P, T](
    template: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> Awaitable[T]:
    """Take one step of a choreography, and return an awaitable for its result.

    The step ID is allocated here, when `step` is called, rather than when the
    result is awaited, so that concurrent steps still get the same IDs in the
    same order on every agent.

    Under `EndpointProjection`, the agent that owns *template* executes the
    step while the others await its result. A template bound to no agent is
    executed by every agent.

    Outside a choreography this is just `call`.
    """
    projection = _PROJECTION.get()
    if projection is None:
        return call(template, *args, **kwargs)
    if _IN_SCATTER.get():
        raise RuntimeError(
            "step() cannot be used inside scatter(): its per-item work is not "
            "a choreography step, and allocating step IDs there would put the "
            "agents out of sync. Use call() instead."
        )
    return projection._run_step(projection._next_step(), template, args, kwargs)


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

    Outside a choreography, items are processed sequentially, round-robin over
    the pool.

    .. warning::

        *fn* should only touch the agent it is handed, with `call`. Steps
        (`step`) and other agents' templates inside *fn* are not supported.
    """
    projection = _PROJECTION.get()
    if projection is None:
        return _scatter_default(items, agent, fn)
    return projection._scatter(projection._next_step(), items, agent, fn)


async def _scatter_default[A: Agent, T, U](
    items: Sequence[T],
    agent: A | Sequence[A],
    fn: Callable[[A, T], Awaitable[U]],
) -> list[U]:
    agents = [agent] if isinstance(agent, Agent) else list(agent)
    return [await fn(agents[i % len(agents)], item) for i, item in enumerate(items)]


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

    Each run starts from a clean slate -- results live in memory for the
    duration of the run, so re-running a choreography re-executes it.

    Args:
        program: The choreographic ``async`` function. All agents run it.
        agents: The agents participating in the choreography.
        handlers: Handlers installed per agent beneath the projection (LLM
            provider, retries, persistence). Handlers already installed around
            the call -- by `effectful.handlers.llm.harness`, say -- are
            inherited and need not be repeated here.

    Example::

        choreo = Choreography(
            build_codebase,
            agents=[architect, coder, reviewer],
            handlers=[LiteLLMProvider(model="gpt-4o-mini"), RetryLLMHandler()],
        )
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
        handlers: Sequence[Interpretation | ObjectInterpretation] | None = None,
    ) -> None:
        self.program = program
        self.agents = list(agents)
        self.handlers = list(handlers or [])
        self._steps = _StepLog()

    def projection(
        self,
        agent: Agent,
        executor: concurrent.futures.Executor | None = None,
    ) -> EndpointProjection:
        """The `EndpointProjection` for one agent.

        Projections handed out between runs share this choreography's current
        step log, so they can drive a program together::

            async with asyncio.TaskGroup() as group:
                for agent in choreo.agents:
                    projection = choreo.projection(agent)
                    group.create_task(drive(projection, **kwargs))

        where ``drive`` installs the handlers and ``projection.activate()``
        around ``await choreo.program(**kwargs)``.
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
        projection = self.projection(agent, executor=executor)
        with contextlib.ExitStack() as stack:
            for h in self.handlers:
                stack.enter_context(handler(h))
            stack.enter_context(projection.activate())
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
        # A fresh log per run: futures belong to the loop that created them.
        self._steps = _StepLog()

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
