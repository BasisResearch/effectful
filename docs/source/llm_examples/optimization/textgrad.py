"""Textual gradients: backprop-style credit assignment over live skill calls.

Implements the core of TextGrad ("Automatic 'Differentiation' via Text",
arXiv:2406.07496), in the form popularized by optimizer/memory libraries such as
`strands-labs/ai-functions`: named text *parameters* feed forward passes, and after a
run an optimizer walks the computation graph backwards, splitting natural-language
feedback across each call's inputs ("textual gradients") and then folding the
gradients accumulated on each parameter into an improved value (`accumulate`).
The mental model is PyTorch autograd -- a `Parameter` is a learnable weight, a
traced run is a forward pass, and `TextGradOptimizer.step` is ``loss.backward()``
plus ``optimizer.step()``.

Where `library.py` (optimize_anything) improves an artifact by frontier *search* --
propose, evaluate, keep the Pareto survivors -- this module improves parameters by
*credit assignment*: one piece of downstream feedback is routed backwards through the
graph to whichever inputs earned it. The two are complementary faces of text
optimization, and each falls out of ordinary effectful idioms rather than a subsystem:

  * **The graph is recorded live by a handler, not reconstructed from an event log.**
    Reference implementations rebuild the computation graph post-hoc from a
    coordinator's event log, which forces a schema'd memory backend (values must
    serialize into recall events and rehydrate later). Here every skill call already
    flows through the interceptable operations of
    `effectful.handlers.llm.harness.hooks`: the optimizer implements `call_agent`
    to record `CallNode` s as calls happen, holding *live references*, and observes
    `call_user` / `call_assistant` / `call_tool` (forwarding untouched) to keep
    each node's transcript. Nesting gives parent/child edges for free (a skill
    called during another's completion loop is its child), and sibling dataflow --
    one call's output passed as another's argument -- is recovered by matching
    argument identity against recorded outputs.

  * **A parameter is just a mutable box.** With live references there is nothing to
    (de)serialize, so the whole memory-backend apparatus collapses into `Parameter`:
    ``value`` (≈ ``.data``, mutated in place by `accumulate`) plus ``gradients``
    (≈ ``.grad``). There is no name and no ``requires_grad``: a use site is named by
    the argument the box was passed as, and opting out of optimization is passing
    ``param.value`` instead of ``param`` -- ``.value`` is literally ``.detach()``.
    Boxes hosted as dataclass fields of a persistent
    `~effectful.handlers.llm.types.Agent` are checkpointed by the existing
    `~effectful.handlers.llm.harness.durability.persistence.SQLitePersister`
    with no code here.

  * **Backward-pass inputs are typed values, not rendered prompts.** The backward
    model is itself a `Skill`; its targets arrive as a ``list[Target]`` and the
    node's transcript as the captured messages, all through the `Encodable`
    bridge -- the "side information is just a typed value" idiom of `library.py`.

  * **Routing is certified at decode time.** The backward skill returns
    ``list[Feedback]``, and `Feedback.__post_init__` rejects any ``target`` naming
    no input of the node under distribution, so a hallucinated route raises during
    decoding and the harness's ``TenacityRetryer`` feeds the error back. Target
    names are ephemeral to a single backward call -- a join key, not a graph id.

  * **Dynamic scoping replaces "do not trace yourself".** Record the forward pass
    inside ``with handler(optimizer)`` and call ``step`` outside it, and the
    optimizer's internal skill calls never enter the graph; no special-case scope
    suppression exists. Likewise, *lexical* scoping keeps those internal skills out
    of everyone's toolset: they are methods (bound in no module scope, reachable
    through no `Agent`), so the harness never advertises them as callable tools.

The recording assumes a single-threaded forward pass, and results are matched to
arguments by object identity -- interpolating a result into an f-string before
passing it on still computes the right value but drops the dataflow edge.

Unlike its sibling examples, this module imports and intercepts harness
operations -- its subject *is* the agent loop. Every interception forwards, so
it supplements the launcher's stack rather than shadowing it.

See `guidelines.py` for the runnable demo. The two are separate modules
deliberately: the harness splices a skill's defining module into its system prompt,
so demo skills sharing this file would read these optimizer prompts and vice versa.
"""

import contextvars
import dataclasses
import graphlib
import inspect
import typing

import pydantic

from effectful.handlers.llm import Skill
from effectful.handlers.llm.harness.hooks import (
    Message,
    call_agent,
    call_assistant,
    call_tool,
    call_user,
)
from effectful.handlers.llm.types import Encodable
from effectful.internals.unification import nested_type
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements

# ── Parameters ────────────────────────────────────────────────────────────────


@dataclasses.dataclass(eq=False)
class Parameter[T]:
    """A learnable value: the store, the recalled view, and the graph node in one.

    Pass the box itself to a recorded skill call and the optimizer's tracing
    handler records the use (under the argument's name) and splices ``value``
    into the prompt; pass ``param.value`` instead to detach -- the value still
    flows, but no edge is recorded and no gradient will reach the box. After
    ``step``, accumulation rewrites ``value`` in place, so every later use
    sees the improved value. (Outside the optimizer's handler scope there is
    nothing to unwrap the box, so pass ``param.value`` there too.)

    ``description`` is an optional escape hatch: the backward and accumulation
    models already see the traced prompt in which the box's value appeared, so
    its role is usually inferable from context; when set, it is passed to both
    as explicit format/merge instructions.

    ``eq=False`` keeps identity semantics: two boxes are never "the same
    parameter" by value, deduplication is by ``id``, and the box stays hashable
    (so it may be a class-level default, shared across instances -- assigning
    one in a class body also triggers `__set_name__`, naming the box after the
    attribute).
    """

    value: T
    description: str = ""
    gradients: list[str] = dataclasses.field(default_factory=list)

    def __set_name__(self, owner: type, name: str) -> None:
        self.__name__ = name

    def __str__(self) -> str:
        """Render the wrapped value (note: f-stringing a box drops its edge)."""
        return str(self.value)


# ── Graph nodes and argument scanning ────────────────────────────────────────


@dataclasses.dataclass(eq=False)
class CallNode:
    """One traced skill call: its transcript, inputs, output, and children.

    Pure record of the forward pass -- the backward pass reads it and writes
    nothing here. The only mutable optimization state anywhere is
    `Parameter.gradients` / `Parameter.value`; feedback in flight between
    nodes lives in a map local to one `TextGradOptimizer.backward` call.
    """

    skill_name: str
    messages: list[Message] = dataclasses.field(default_factory=list)
    value: typing.Any = None
    parameters: list[tuple[str, "Parameter[typing.Any]"]] = dataclasses.field(
        default_factory=list
    )
    children: list["CallNode"] = dataclasses.field(default_factory=list)


def _scan(
    value: typing.Any,
    label: str,
    seen: set[int],
    on_parameter: typing.Callable[[str, "Parameter[typing.Any]"], None],
    on_value: typing.Callable[[typing.Any], None],
) -> None:
    """Walk an argument, reporting `Parameter` boxes and every reachable object.

    Descends through dicts, sequences, sets and dataclass instances (so a box
    hosted on an ``Agent`` passed as ``self`` is found under its field name).
    ``on_value`` is called with each object so the tracer can match it against
    recorded outputs; ``seen`` keeps shared/cyclic structures finite.
    """
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, Parameter):
        on_parameter(label, value)
        return
    on_value(value)
    if isinstance(value, dict):
        for item in value.values():
            _scan(item, label, seen, on_parameter, on_value)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _scan(item, label, seen, on_parameter, on_value)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            _scan(getattr(value, f.name), f.name, seen, on_parameter, on_value)


def _strip(value: typing.Any) -> typing.Any:
    """Replace `Parameter` boxes with their values, rebuilding containers."""
    if isinstance(value, Parameter):
        return value.value
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items()}
    if isinstance(value, tuple):
        items = (_strip(v) for v in value)
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    if isinstance(value, (list, set, frozenset)):
        return type(value)(_strip(v) for v in value)
    return value


# ── Model-boundary types of the backward pass ────────────────────────────────

_VALID_TARGETS: contextvars.ContextVar[frozenset[str] | None] = contextvars.ContextVar(
    "_VALID_TARGETS", default=None
)


@dataclasses.dataclass(frozen=True)
class Target:
    """One routable input of the node under distribution, as the model sees it.

    Ephemeral to a single backward call: ``name`` is a per-call join key (the
    argument name a box was passed as, or a child call's skill name), not a
    graph identifier, and nothing here is stored in the graph.

    Deliberately *not* a supertype of `Parameter`: a target is the rendered
    snapshot of an input at the moment of one backward call -- and half the
    inputs are not parameters at all (a child `CallNode` appears with kind
    ``result``). It carries exactly what the live objects don't: a per-call
    name and a ``str``-rendered value. In autograd terms, a `Target` is the
    argument the backward function is shown; a `Parameter` is the leaf the
    routed gradient lands on.
    """

    name: str
    kind: typing.Literal["parameter", "result"]
    description: str
    value: str


@dataclasses.dataclass(frozen=True)
class Updated:
    """The accumulation skill's structured answer: just the updated value.

    A one-field wrapper so the reply is decoded by schema rather than scraped
    from prose. This is load-bearing, not ceremony: with a bare ``str`` return
    there is no response format, and small models answer with their working
    shown -- "To update the parameter based on the accumulated feedback, we
    need to consider..." -- *as the new value*, or wrap it in the delimiters
    the prompt used to show it. A schema'd single field leaves no room for
    either.
    """

    value: str


@dataclasses.dataclass(frozen=True)
class Feedback:
    """One routed piece of feedback the backward model emits.

    ``target`` must name a `Target` of the same backward call; construction
    checks it against the routing map the optimizer put in scope, so a
    hallucinated target raises during decoding and the harness's retryer feeds
    the error back to the model.
    """

    target: str
    feedback: str

    def __post_init__(self) -> None:
        valid = _VALID_TARGETS.get()
        if valid is not None and self.target not in valid:
            raise ValueError(
                f"Feedback references unknown target {self.target!r}. "
                f"Valid targets: {sorted(valid)}. Only use names from the "
                f"listed inputs."
            )


# ── The optimizer ────────────────────────────────────────────────────────────


class TextGradOptimizer(ObjectInterpretation):
    """Record a forward pass as a computation graph, then optimize its parameters.

    One object plays both autograd roles: installed as a handler it is the
    *tape*, recording every skill call; afterwards ``step`` backpropagates
    feedback through the recorded graph and updates the parameters::

        optimizer = TextGradOptimizer()
        with handler(optimizer):                       # record
            draft = write(topic="cats", guidelines=guidelines)  # a Parameter
            final = polish(draft=draft)
        graph = optimizer.step("Too long; cut the preamble.")   # optimize

    **Recording.** Edges come from two sources: *nesting* (a skill called while
    another's completion loop is open becomes its child) and *sibling dataflow*
    (an argument that is -- by object identity -- the output of an earlier
    recorded call grafts that call in as a child, unlisting it as a root).
    `Parameter` boxes found in the arguments are recorded under the argument
    (or dataclass field) name and unwrapped to their values before the call
    proceeds. The ``call_user`` / ``call_assistant`` / ``call_tool`` handlers
    only observe: they forward untouched and append the produced messages to
    the innermost open node's transcript.

    Every call is recorded, learnable or not: a parameterless node may still be
    a conduit to boxes in nested calls below it, a detached (``.value``) branch
    is still part of what happened, and the tape doubles as an honest trace for
    inspection. Which subtrees a gradient can actually reach is decided
    retrospectively, by `_toposort`'s pruning at backward time -- recording
    never anticipates what the optimizer will find relevant.

    **Optimizing.** ``step`` (and the finer-grained ``backward`` /
    ``accumulate`` / ``zero_grad``, which operate on any `CallNode` graph)
    invoke the LLM through the skill methods below, so they must run under the
    harness but *outside* this object's own handler scope -- otherwise the
    optimization would record itself into the graph it is optimizing. The
    skills are *methods* rather than module-level functions so they appear in
    no other skill's lexical scope, and this class is not an `Agent` so there
    is no shared ``__history__``: every backward call is a fresh conversation.

    Recording is single-threaded, and roots accumulate for the lifetime of the
    instance -- use one optimizer per forward pass, or pass ``output=`` to
    ``step`` to select among several recorded roots.
    """

    def __init__(self) -> None:
        self._roots: list[CallNode] = []
        self._stack: list[CallNode] = []
        self._index: dict[int, CallNode] = {}

    # -- Recording (handler methods) ------------------------------------------

    @implements(call_agent)
    def _trace_call(self, skill, *args, **kwargs):
        node = CallNode(skill_name=skill.__name__)

        try:
            bound = inspect.signature(skill).bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
        except TypeError:  # let the call itself report the signature error
            arguments = {}

        def on_parameter(label: str, box: Parameter[typing.Any]) -> None:
            name = label or getattr(box, "__name__", "") or "parameter"
            node.parameters.append((name, box))

        def on_value(value: typing.Any) -> None:
            child = self._index.get(id(value))
            if child is None or child is node:
                return
            if any(c is child for c in node.children):
                return
            self._roots = [r for r in self._roots if r is not child]
            node.children.append(child)

        seen: set[int] = set()
        for name, value in arguments.items():
            _scan(value, name, seen, on_parameter, on_value)

        if node.parameters:
            args = tuple(_strip(a) for a in args)
            kwargs = {k: _strip(v) for k, v in kwargs.items()}

        parent = self._stack[-1] if self._stack else None
        self._stack.append(node)
        try:
            result = fwd(skill, *args, **kwargs)
        finally:
            self._stack.pop()

        node.value = result
        self._index[id(result)] = node
        if parent is not None:
            parent.children.append(node)
        else:
            self._roots.append(node)
        return result

    @implements(call_user)
    def _trace_user(self, *args, **kwargs):
        message = fwd(*args, **kwargs)
        if self._stack:
            self._stack[-1].messages.append(message)
        return message

    @implements(call_assistant)
    def _trace_assistant(self, *args, **kwargs):
        message, tool_calls, result = fwd(*args, **kwargs)
        if self._stack:
            self._stack[-1].messages.append(message)
        return (message, tool_calls, result)

    @implements(call_tool)
    def _trace_tool(self, *args, **kwargs):
        message, result, is_final = fwd(*args, **kwargs)
        if self._stack:
            self._stack[-1].messages.append(message)
        return (message, result, is_final)

    # -- Backprop -----------------------------------------------------------

    @staticmethod
    def _toposort(root: CallNode) -> list[CallNode]:
        """`CallNode` s in reverse topological order (root first), pruned.

        Consumers precede producers, so feedback distributed at a node is
        already complete when the walk reaches its children. Subtrees that
        reach no `Parameter` are skipped entirely -- the requires-grad
        reachability check: no gradient can land there, so no backward call
        should be spent there.

        The result answers every graph question the optimizer has.
        Distribution order is the list itself; "which children are routable
        targets" is membership (a pruned subtree is exactly one that reaches
        no parameter, so for any child of a kept node, kept ⟺ routable); and
        every reachable `Parameter` lives on a kept node, so `zero_grad` and
        `accumulate` need no unpruned walk.

        Collection and pruning are a plain reachability walk (`graphlib` has
        no notion of skipping subtrees), but the *ordering* is delegated to
        `graphlib.TopologicalSorter` -- a child produces a value its parent
        consumes, so children are the parent's dependencies, ``static_order``
        yields producers first, and the backward pass wants the reverse. A
        diamond that somehow degenerated into a cycle raises ``CycleError``
        here instead of walking in a silently wrong order.
        """
        reaches: dict[int, bool] = {}

        def _reaches_parameter(node: CallNode) -> bool:
            """Memoized; seeded ``False`` so a cycle back to ``node`` terminates."""
            nid = id(node)
            if nid in reaches:
                return reaches[nid]
            reaches[nid] = False
            reaches[nid] = bool(node.parameters) or any(
                _reaches_parameter(c) for c in node.children
            )
            return reaches[nid]

        edges: dict[CallNode, list[CallNode]] = {}
        stack = [root]
        while stack:
            node = stack.pop()
            if node in edges:
                continue
            edges[node] = [c for c in node.children if _reaches_parameter(c)]
            stack.extend(edges[node])
        return list(graphlib.TopologicalSorter(edges).static_order())[::-1]

    def _targets_of(
        self, node: CallNode, kept: set[int]
    ) -> tuple[list[Target], dict[str, "Parameter[typing.Any] | CallNode"]]:
        """The node's routable inputs, named uniquely for one backward call.

        ``kept`` is the pruned reachable set from `_toposort`: for a child of a
        node in it, membership is exactly "leads to a parameter", so children
        outside it are not offered as targets (no gradient could land there).
        """
        routing: dict[str, Parameter[typing.Any] | CallNode] = {}
        targets: list[Target] = []

        def unique(name: str) -> str:
            candidate, n = name, 1
            while candidate in routing:
                n += 1
                candidate = f"{name}_{n}"
            return candidate

        seen_boxes: set[int] = set()
        for label, box in node.parameters:
            if id(box) in seen_boxes:
                continue
            seen_boxes.add(id(box))
            name = unique(label)
            routing[name] = box
            targets.append(
                Target(
                    name=name,
                    kind="parameter",
                    description=box.description,
                    value=str(box.value),
                )
            )
        for child in node.children:
            if id(child) not in kept:
                continue
            name = unique(child.skill_name)
            routing[name] = child
            targets.append(
                Target(name=name, kind="result", description="", value=str(child.value))
            )
        return targets, routing

    @Skill.define
    def _compute_gradients(
        self,
        targets: list[Target],
        trace: list[Message],
        output: str,
        feedback: list[str],
    ) -> list[Feedback]:
        """You are an optimization agent analyzing one step of an AI workflow to decide
        how its inputs should change. You are given the step's routable inputs, the
        conversation trace of the step, the step's output, and feedback (issues) that
        this output contributed to. Distribute the feedback across the inputs.

        # Inputs

        {targets}

        # Conversation trace

        {trace}

        # Output

        {output}

        # Issues

        {feedback}

        # Rules

        1. For an input of kind "parameter", the value is a standing instruction that
           will be rewritten once using your feedback and then reused on different
           future inputs. Phrase feedback as general guidance in this bullet format:
           - add: text to add to the value
           - update: text to change in the value
           - delete: text to remove from the value
        2. For an input of kind "result", the value is the output of an upstream step
           that will receive your feedback and re-distribute it to its own inputs.
           Phrase feedback as: how this specific result should change to resolve the
           issues.
        3. Give feedback to as few inputs as possible: only those whose change would
           actually resolve an issue, and only feedback relevant to that input (and to
           its description, when present). It is fine to ignore issues that concern no
           input.
        4. Feedback for a "parameter" must be general and applicable to future inputs,
           never specific to one run's data.
        """

    def backward(self, root: CallNode, feedback: str) -> dict[CallNode, list[str]]:
        """Route ``feedback`` from ``root`` through the graph to the parameters.

        The only state this writes is `Parameter.gradients`, which accumulates
        deliberately (across ``backward`` calls, and across uses of one box in
        several nodes) until `zero_grad`. The per-node feedback in flight is
        working state local to this call, returned for inspection and otherwise
        discarded -- nothing on the graph itself changes.

        Each node with incoming feedback is distributed exactly once: the walk
        is reverse-topological, so by a node's turn every downstream consumer
        (all of them, in a diamond) has already routed its refined share here.
        """
        order = self._toposort(root)
        kept = {id(node) for node in order}
        grads: dict[CallNode, list[str]] = {root: [feedback]}

        for node in order:
            incoming = grads.get(node)
            if not incoming:
                continue
            targets, routing = self._targets_of(node, kept)
            if not routing:
                # Nothing routable here: pass the feedback through unchanged.
                for child in node.children:
                    grads.setdefault(child, []).extend(incoming)
                continue

            # One backward model call per node, certified against its targets.
            token = _VALID_TARGETS.set(frozenset(routing))
            try:
                feedbacks = self._compute_gradients(
                    targets=targets,
                    trace=list(node.messages),
                    output=str(node.value),
                    feedback=list(incoming),
                )
            finally:
                _VALID_TARGETS.reset(token)

            # Feedback routed to a Parameter lands on the box (the persistent
            # ``.grad``); feedback routed to a child call joins the in-flight
            # map, re-distributed when the walk reaches that child.
            for fb in feedbacks:
                target = routing[fb.target]
                if isinstance(target, Parameter):
                    target.gradients.append(fb.feedback)
                else:
                    grads.setdefault(target, []).append(fb.feedback)
        return grads

    # The str-typed `Updated.value` reply is a workaround for `Skill`
    # signatures not supporting type variables: the natural definition is
    # ``def _accumulate[T](self, parameter: Parameter[T]) -> T`` with ``T``
    # bound per call from the parameter's value, which would let structured
    # decoding produce the parameter's own type directly and delete both the
    # wrapper and the validate-back shim in `accumulate`. (A closure-captured
    # concrete type does not work either: the harness's synthesis path
    # type-checks model code against the skill's *source*, where a closure
    # variable in an annotation is statically invalid.) If generic skills
    # land, define `_accumulate` that way.
    @Skill.define
    def _accumulate(self, parameter: Parameter) -> Updated:
        """Update the parameter below by applying its accumulated feedback: its
        ``value`` is the current content, ``gradients`` are the feedback entries
        to apply, and ``description``, when present, says what the value must
        contain and how updates should be merged into it -- follow it.

        <parameter>
        {parameter}
        </parameter>

        Preserve whatever the feedback does not ask to change, and keep the
        format and shape of the current value. The updated value must contain
        only the parameter's new content -- no commentary about the update
        itself. Do not use any tools.
        """

    def accumulate(self, root: CallNode) -> None:
        """Fold each parameter's accumulated gradients into its value, in place.

        A box recorded by several nodes is updated exactly once, on the union
        of its gradients (deduplication is by box identity). The box itself is
        the skill's one argument -- value, description and gradients travel
        together through its `Encodable` encoding -- but the reply is an
        `Updated` wrapping a ``str``, so a non-``str`` value is validated back
        into its own type from the returned JSON text.
        """
        seen: set[int] = set()
        boxes: list[Parameter[typing.Any]] = []
        for node in self._toposort(root):
            for _, box in node.parameters:
                if id(box) not in seen:
                    seen.add(id(box))
                    boxes.append(box)
        for box in boxes:
            if not box.gradients:
                continue
            updated = self._accumulate(box).value
            adapter: pydantic.TypeAdapter[typing.Any] = pydantic.TypeAdapter(
                Encodable[nested_type(box.value).value]  # type: ignore
            )
            box.value = (
                updated
                if isinstance(box.value, str)
                else adapter.validate_json(updated)
            )

    def zero_grad(self, root: CallNode) -> None:
        """Clear the gradients of every `Parameter` reachable from ``root``.

        The pruned walk suffices: a pruned subtree is precisely one containing
        no parameters, so every reachable box lives on a kept node.
        """
        for node in self._toposort(root):
            for _, box in node.parameters:
                box.gradients.clear()

    def graph(self, output: typing.Any | None = None) -> CallNode:
        """The recorded graph rooted at the call that produced ``output``.

        With no argument, the sole recorded root -- after a linear pipeline,
        the final output's node. When several independent roots were recorded
        (a training batch, say), ``output`` selects among them by identity.

        This is what makes per-root `backward` composable beyond `step`'s
        one-shot form: a batch run records one root per example, calls
        ``backward(optimizer.graph(out_i), feedback_i)`` for each, and then
        folds every parameter's merged gradients in a single `accumulate`.
        """
        if output is not None:
            root = self._index.get(id(output))
            if root is None:
                raise ValueError(
                    "output was not produced by a call recorded on this optimizer"
                )
            return root
        if len(self._roots) == 1:
            return self._roots[0]
        raise ValueError(
            f"optimizer has recorded {len(self._roots)} roots; pass output= "
            f"to select the value the feedback is about"
        )

    def step(
        self, feedback: str, output: typing.Any | None = None
    ) -> tuple[CallNode, dict[CallNode, list[str]]]:
        """Backward + accumulate in one call.

        The entry point is `graph`'s resolution of ``output``. Returns that
        root together with `backward`'s per-node routed feedback, for
        inspection.
        """
        root = self.graph(output)
        grads = self.backward(root, feedback)
        self.accumulate(root)
        return root, grads
