from dataclasses import dataclass

import chex
import effectful.handlers.jax.numpy as np
import effectful.handlers.numbers
import matplotlib.pyplot as plt
from effectful.handlers.numbers import add, mul
from effectful.ops.syntax import defop, defterm
from jax import random
from tqdm import tqdm

from weighted.fold_lang_v1 import *


@dataclass(frozen=True)
class State:
    """State of the environment.

    Attributes:
        agent_position: Position of the agent as [row, col].
        food_positions: Binary grid of food positions (1 where food exists, 0 otherwise).
    """

    agent_position: chex.Array
    food_positions: chex.Array

    @staticmethod
    def random(key: chex.PRNGKey, grid_size: int = 15, food_probability: float = 0.05) -> "State":
        # Split the random key
        key1, key2 = random.split(key)

        # Randomly place the agent on the grid
        agent_position = random.randint(key1, (2,), 0, grid_size)

        food_positions = random.uniform(key2, (grid_size, grid_size)) < food_probability
        food_positions = food_positions.astype(np.int32)
        food_positions = food_positions.at[agent_position[0], agent_position[1]].set(0)

        return State(agent_position=agent_position, food_positions=food_positions)

    def render(self):
        """Render the environment using matplotlib."""
        grid = self.food_positions
        grid = grid.at[self.agent_position[0], self.agent_position[1]].set(2)  # Mark agent position (2)

        # Create a color map: 0 -> white, 1 -> green (food), 2 -> blue (agent)
        cmap = plt.cm.colors.ListedColormap(["white", "green", "blue"])
        bounds = [0, 1, 2, 3]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        # Plot the grid
        plt.imshow(grid, cmap=cmap, norm=norm)
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        plt.title("Grid World")
        plt.show()

    def __hash__(self):
        """Hash function for the state."""
        return hash((tuple(self.agent_position.tolist()), tuple(self.food_positions.flatten().tolist())))

    def __eq__(self, other):
        """Equality check for the state."""
        if not isinstance(other, State):
            return NotImplemented
        return np.array_equal(self.agent_position, other.agent_position) and np.array_equal(
            self.food_positions, other.food_positions
        )


Action: TypeAlias = int


def dynamics_and_reward(state: State, action: Action, step_penalty=-0.01, food_reward=1.0) -> tuple[list[State], float]:
    action_to_direction = np.array(
        [
            [-1, 0],  # up
            [0, 1],  # right
            [1, 0],  # down
            [0, -1],  # left
        ]
    )

    curr_position = state.agent_position
    direction = action_to_direction[action]
    grid_size = state.food_positions.shape[0]

    # Calculate new position
    new_position = curr_position + direction

    # Apply boundary constraints using clip to keep the agent within the grid
    new_position = np.clip(new_position, 0, grid_size - 1)

    reward = state.food_positions[new_position[0], new_position[1]] * food_reward + step_penalty

    # Remove food from the new position
    new_food = state.food_positions.at[new_position[0], new_position[1]].set(0)

    # Create new state
    new_state = State(agent_position=new_position, food_positions=new_food)

    return [new_state], reward


states = defop(list[State], name="States")
actions = defop(jax.Array, name="Actions")


@defop
def reward(s: State, a: Action) -> float:
    if not any(isinstance(x, Term) for x in [s, a]):
        return dynamics_and_reward(s, a)[1]
    raise NotImplementedError


@defop
def dynamics(s: State, a: Action) -> list[State]:
    if not any(isinstance(x, Term) for x in [s, a]):
        return dynamics_and_reward(s, a)[0]
    raise NotImplementedError


@defop
def tuple_getitem(x, i):
    if not any(isinstance(a, Term) for a in (x, i)):
        return x[i]
    raise NotImplementedError


def policy_of_value(value, discount_factor):
    s, sn = defop(State, name="s"), defop(State, name="sn")
    a = defop(Action, name="a")
    w = defop(float, name="w")

    discounted_value = mul(discount_factor, fold(LinAlg, {sn: dynamics(s(), a())}, value(sn())))
    return deffn(tuple_getitem(fold(ArgMaxAlg, {a: actions()}, (reward(s(), a()) + discounted_value, a())), 1), s)


def value_of_policy(policy: Callable[[State], Action], discount_factor, horizon=10) -> Callable[[State], float]:
    s, sn = defop(State, name="s"), defop(State, name="sn")
    w = defop(float, name="w")

    value = deffn(0, s)  # make free in s, rather than function
    for _ in range(horizon):
        prev_value = value
        future_payoff = fold(LinAlg, {sn: dynamics(s(), policy(s()))}, prev_value(sn()))
        value = deffn(add(reward(s(), policy(s())), mul(discount_factor, future_payoff)), s)

    return value


def value_of_policy_(policy: Callable[[State], Action], initial: State, horizon: int, discount_factor: float) -> float:
    if horizon == 0:
        return reward(initial, policy(initial))

    sn = defop(State)
    future_reward = fold(
        LinAlg, {sn: dynamics(initial, policy(initial))}, value_of_policy_(policy, sn(), horizon - 1, discount_factor)
    )
    return add(reward(initial, policy(initial)), mul(discount_factor, future_reward))


def value_of_policy__(policy: Callable[[State], Action], initial: State, horizon: int, discount_factor: float) -> float:
    states = [defop(State) for _ in range(horizon)]

    total_reward = 0.0
    for t in range(horizon):
        action = policy(states[t]())
        total_reward += reward(states[t + 1](), action) * discount_factor**t

    return fold(
        LinAlg, {states[t + 1]: dynamics(states[t](), policy(states[t]())) for t in range(horizon)}, total_reward
    )

    if horizon == 0:
        return reward(initial, policy(initial))

    sn = defop(State)
    future_reward = fold(
        LinAlg, {sn: dynamics(initial, policy(initial))}, value_of_policy_(policy, sn(), horizon - 1, discount_factor)
    )
    return add(reward(initial, policy(initial)), mul(discount_factor, future_reward))


def converged(p1, p2, epsilon=0.01):
    return np.max(np.abs(p1 - p2)) < epsilon


def policy_iteration(discount_factor=0.01, steps=1):
    s = defop(State)

    policy = deffn(0, s)
    for _ in tqdm(range(steps)):
        value = value_of_policy(policy, discount_factor)
        next_policy = policy_of_value(value, discount_factor)
        # for st in all_states:
        #     print("State:", st, "Policy:", policy(st), "Next Policy:", next_policy(st))
        policy = next_policy
    return policy


grid_size = 2
all_actions = np.array([0, 1, 2, 3])
all_states = []
for i in range(grid_size):
    for j in range(grid_size):
        for g in range(2 ** (grid_size * grid_size)):
            state = State(
                agent_position=np.array([i, j]),
                food_positions=np.array(
                    [(g >> (i * grid_size + j)) & 1 for i in range(grid_size) for j in range(grid_size)]
                ).reshape((grid_size, grid_size)),
            )
            if not state.food_positions[state.agent_position[0], state.agent_position[1]]:
                all_states.append(state)


def partial_eval_state_deffn(body, *args, **kwargs):
    if len(args) == 1 and typeof(args[0]()) == State:
        var = args[0]
        values = {}
        for s in states():
            with handler({var: lambda: s}):
                try:
                    values[s] = evaluate(body)
                except TypeError:
                    breakpoint()

        @defop
        def pfun(x):
            if isinstance(x, Term):
                raise NotImplementedError
            if not isinstance(x, State):
                breakpoint()
            assert isinstance(x, State)
            return values[x]

        return pfun
    return fwd()


def fold_conv(*args, **kwargs):
    ret = fwd()
    if isinstance(ret, dict) and () in ret:
        return ret[()]
    return ret


def term_to_json(term):
    def _op_to_json(op, *args, **kwargs):
        return {"name": op.__name__, "freshening": op._freshening}

    def _term_to_json(expr):
        expr = defterm(expr)
        if isinstance(expr, dict):
            return {"dict": [(_term_to_json(k), _term_to_json(v)) for (k, v) in expr.items()]}
        elif isinstance(expr, list) or isinstance(expr, tuple):
            return {"list": [_term_to_json(t) for t in expr]}
        elif isinstance(expr, Term):
            args = [_term_to_json(t) for t in expr.args]
            kwargs = [(_term_to_json(k), _term_to_json(v)) for (k, v) in expr.kwargs.items()]
            return {"op": _op_to_json(expr.op), "args": args, "kwargs": kwargs, "type": str(typeof(expr))}
        elif isinstance(expr, Semiring):
            semirings = [
                (LinAlg, "LinAlg"),
                (MinAlg, "MinAlg"),
                (MaxAlg, "MaxAlg"),
                (ArgMinAlg, "ArgMinAlg"),
                (ArgMaxAlg, "ArgMaxAlg"),
            ]
            for r, n in semirings:
                if expr is r:
                    return {"value": n}
            return {"value": str(expr)}
        else:
            return {"value": str(expr)}

    import json

    j = _term_to_json(term)
    return json.dumps(j)


with handler(simplify_intp):
    t = policy_iteration(steps=1)

print(str(t))
# s = defop(State)
# with (
#     handler(dense_fold_intp),
#     handler(
#         {actions: lambda: all_actions, states: lambda: all_states, deffn: partial_eval_state_deffn, fold: fold_conv}
#     ),
# ):
#     policy = policy_iteration(steps=5)

# state = State(np.array([0, 0]), np.array([[0, 0], [1, 1]]))
# state.render()
# for _ in range(5):
#     action = policy(state)
#     print("Action:", action)
#     state = dynamics(state, action)[0]
#     state.render()
