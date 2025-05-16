import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import chex
import effectful.handlers.jax.numpy as jnp
import flax.nnx as nnx
import jax
import matplotlib.pyplot as plt
import optax
from effectful.handlers.jax import jax_getitem
from effectful.handlers.jax._handlers import _register_jax_op, is_eager_array
from effectful.ops.semantics import fwd, handler, typeof
from effectful.ops.syntax import ObjectInterpretation, deffn, defop, implements
from effectful.ops.types import Operation, Term
from jax import random
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from weighted.handlers.jax import D, ScanFold, key
from weighted.handlers.jax import interpretation as jax_intp
from weighted.ops.semiring import add, mul
from weighted.ops.sugar import ArgMax, Sum


@dataclass(frozen=True)
class State:
    """State of the environment.

    Attributes:
        agent_position: Position of the agent as [row, col].
        food_positions: Binary grid of food positions (1 where food exists, 0 otherwise).
    """

    agent_position: jax.Array
    food_positions: jax.Array

    @staticmethod
    def random(
        key: chex.PRNGKey, grid_size: int = 15, food_probability: float = 0.05
    ) -> "State":
        # Split the random key
        key1, key2 = random.split(key)

        # Randomly place the agent on the grid
        agent_position = random.randint(key1, (2,), 0, grid_size)

        food_positions = random.uniform(key2, (grid_size, grid_size)) < food_probability
        food_positions = food_positions.astype(jnp.int32)
        food_positions = food_positions.at[agent_position[0], agent_position[1]].set(0)

        return State(agent_position, food_positions)

    def render(self, ax):
        """Render the environment using matplotlib."""
        grid = self.food_positions
        grid = grid.at[self.agent_position[0], self.agent_position[1]].set(
            2
        )  # Mark agent position (2)

        # Create a color map: 0 -> white, 1 -> green (food), 2 -> blue (agent)
        cmap = plt.cm.colors.ListedColormap(["white", "green", "blue"])
        bounds = [0, 1, 2, 3]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        # Plot the grid
        ax.imshow(grid, cmap=cmap, norm=norm)

    def __hash__(self):
        """Hash function for the state."""
        return hash(
            (
                tuple(self.agent_position.tolist()),
                tuple(self.food_positions.flatten().tolist()),
            )
        )

    def __eq__(self, other):
        """Equality check for the state."""
        if not isinstance(other, State):
            return NotImplemented
        return jnp.array_equal(
            self.agent_position, other.agent_position
        ) and jnp.array_equal(self.food_positions, other.food_positions)

    @staticmethod
    def of_array(arr):
        dim = int(math.sqrt(arr.shape[0] - 2))
        return State(arr[:2], jnp.reshape(arr[2:], (dim, dim)))

    def to_array(self):
        """Convert the state to a flat array."""
        return jnp.concatenate(
            (self.agent_position, jnp.ravel(self.food_positions)), dtype=jnp.int32
        )


Action: TypeAlias = int


def dynamics_and_reward(
    arr: jax.Array, action: Action, step_penalty=-0.01, food_reward=1.0
) -> tuple[jax.Array, float]:
    assert arr.shape == (102,)

    state = State.of_array(arr)

    action_to_direction = jnp.array(
        [
            [-1, 0],  # up
            [0, 1],  # right
            [1, 0],  # down
            [0, -1],  # left
        ]
    )

    curr_position = state.agent_position
    direction = jnp.squeeze(jax_getitem(action_to_direction, [action]))
    grid_size = state.food_positions.shape[0]

    # Calculate new position
    new_position = curr_position + direction

    # Apply boundary constraints using clip to keep the agent within the grid
    new_position = jnp.clip(new_position, 0, grid_size - 1)

    reward = (
        jax_getitem(
            state.food_positions,
            [jax_getitem(new_position, [0]), jax_getitem(new_position, [1])],
        )
        * food_reward
        + step_penalty
    )

    # Remove food from the new position
    # TODO: Implement .at method for indexed jax arrays
    def _build_food_mask(pos):
        ret = jnp.ones(state.food_positions.shape).at[pos[0], pos[1]].set(0)
        return ret

    build_food_mask = _register_jax_op(_build_food_mask)
    food_mask = build_food_mask(new_position)
    new_food = state.food_positions * food_mask

    # Create new state
    new_state = State(agent_position=new_position, food_positions=new_food)
    assert new_state.agent_position.shape == (2,)
    assert (
        len(new_state.food_positions.shape) == 2
        and new_state.food_positions.shape[0] == new_state.food_positions.shape[1]
    )

    return new_state.to_array(), reward


actions: Operation[[], jax.Array]
states = defop(jax.Array, name="States")
actions = defop(jax.Array, name="Actions")  # type: ignore


def is_eager(x):
    return not isinstance(x, Term) or is_eager_array(x)


@defop
def reward(s: jax.Array, a: Action) -> float:
    if not (is_eager(s) and is_eager(a)):
        raise NotImplementedError
    return dynamics_and_reward(s, a)[1]


@defop
def dynamics(s: jax.Array, a: Action) -> jax.Array:
    if not (is_eager(s) and is_eager(a)):
        raise NotImplementedError
    return dynamics_and_reward(s, a)[0]


@defop
def tuple_getitem(x, i):
    if not (isinstance(x, tuple) and isinstance(i, int)):
        raise NotImplementedError
    return x[i]


def policy_of_value(
    value: Callable[[jax.Array], float], state: jax.Array, discount_factor: float
) -> Action:
    a = defop(Action, name="a")

    next_state = dynamics(state, a())
    next_value = value(next_state)
    discounted_value = mul(discount_factor, next_value)
    best_action = ArgMax({a: actions()}, (add(reward(state, a()), discounted_value), a()))
    result = tuple_getitem(best_action, 1)
    return result


def value_of_policy(
    policy: Callable[[jax.Array], Action],
    initial: jax.Array,
    horizon: int,
    discount_factor: float,
) -> float:
    states: list[Operation[[], jax.Array]] = [
        defop(jax.Array, name=f"s{i}")  # type: ignore
        for i in range(horizon)
    ]
    bindings = {states[0]: jnp.expand_dims(initial, 0)} | {
        states[i]: jnp.expand_dims(dynamics(states[i - 1](), policy(states[i - 1]())), 0)
        for i in range(1, horizon)
    }

    total_reward = functools.reduce(
        add,
        [
            mul(discount_factor**t, reward(states[t](), policy(states[t]())))
            for t in range(horizon)
        ],
    )
    value = Sum(bindings, total_reward)
    return value


def policy_iteration(discount_factor=0.01, steps=1, horizon=1):
    s = defop(jax.Array)

    policy = deffn(0, s)
    keys = jax.random.split(key(), steps)
    for i in tqdm(range(steps)):
        with handler({key: deffn(keys[i])}):
            value = deffn(value_of_policy(policy, s(), horizon, discount_factor), s)
            policy = deffn(policy_of_value(value, s(), discount_factor), s)
    return policy


grid_size = 2
all_actions = jnp.array([0, 1, 2, 3])
all_states = []
for i in range(grid_size):
    for j in range(grid_size):
        for g in range(2 ** (grid_size * grid_size)):
            agent_position = jnp.array([i, j])
            food_positions = jnp.array(
                [
                    (g >> (i * grid_size + j)) & 1
                    for i in range(grid_size)
                    for j in range(grid_size)
                ]
            ).reshape((grid_size, grid_size))
            if not food_positions[agent_position[0], agent_position[1]]:
                all_states.append(
                    State(agent_position=agent_position, food_positions=food_positions)
                )


class TabularValueFn(ObjectInterpretation):
    @implements(deffn)
    def deffn(self, body, *args, **kwargs):
        if len(args) == 1 and typeof(args[0]()) == State:
            values = Sum({args[0]: states()}, D((args[0](), body)))

            @defop
            def pfun(x):
                if isinstance(x, Term):
                    raise NotImplementedError
                return values[x]

            return pfun

        return fwd()

    @implements(states)
    def _states(self):
        return all_states


class ValueMLP(nnx.Module):
    """Simple MLP for value function approximation using NNX."""

    def __init__(self, width, rngs):
        super().__init__()
        self.dense1 = nnx.Linear(width, width, rngs=rngs)
        self.dense2 = nnx.Linear(width, 1, rngs=rngs)

    def __call__(self, x):
        x = self.dense1(x)
        x = jax.nn.relu(x)
        x = self.dense2(x)
        return x


class NNValueFn(ObjectInterpretation):
    def __init__(self, num_samples=100, learning_rate=0.01, epochs=100, grid_size=10):
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.grid_size = grid_size

    def _generate_training_data(self, value_fn):
        """Generate training data by sampling random states and evaluating them."""
        keys = jax.random.split(key(), self.num_samples)
        states = jax.vmap(lambda k: State.random(k, grid_size=self.grid_size).to_array())(
            keys
        )
        values = jax.vmap(value_fn)(states)
        assert isinstance(values, jax.Array) and values.shape == (self.num_samples,)
        return states, values

    @implements(deffn)
    def deffn(self, body, *args, **kwargs):
        if len(args) == 1 and typeof(args[0]()) == jax.Array and typeof(body) == float:
            # Get the existing value function computation
            exact_value_fn = fwd()
            exact_value_fn_jit = jax.jit(lambda *a, **k: exact_value_fn(*a, **k))

            # generate training data
            features, values = self._generate_training_data(exact_value_fn_jit)

            # Create the model with NNX
            model = ValueMLP(self.grid_size**2 + 2, rngs=nnx.Rngs(key()))

            # Define optimizer
            tx = optax.adam(self.learning_rate)
            optimizer = nnx.Optimizer(model, tx)

            # Define loss function
            def loss_fn(model, x, targets):
                preds = model(x)
                return optax.squared_error(preds.squeeze(), targets).mean()

            @nnx.jit
            def train_step(model, optimizer, states, values):
                grad_fn = nnx.value_and_grad(loss_fn)
                loss, grads = grad_fn(model, states, values)
                optimizer = optimizer.update(grads)

            for _ in range(self.epochs):
                train_step(model, optimizer, features, values)

            # Define prediction function
            @_register_jax_op
            @nnx.jit
            def predict(x):
                return model(x)

            return predict

        return fwd()


with (
    handler(jax_intp),
    handler(NNValueFn()),
    handler(ScanFold()),
    handler({actions: lambda: all_actions}),
):
    policy = policy_iteration(steps=30, horizon=5)

    def frames(n):
        state = State.random(jax.random.PRNGKey(43), grid_size=10)
        yield state
        for _ in tqdm(range(n)):
            action = policy(state.to_array())
            state = State.of_array(dynamics(state.to_array(), action))
            yield state

    fig, ax = plt.subplots(figsize=(10, 8))
    n_frames = 30
    fs = list(frames(n_frames))
    ani = FuncAnimation(fig, lambda state: state.render(ax), fs)
    ani.save("episode.mp4")
