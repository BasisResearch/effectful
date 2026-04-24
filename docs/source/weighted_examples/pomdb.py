import jax
from effectful.handlers.weighted.optimization.plates import plated
from effectful.ops.weighted.sugar import ArgMax, Sum

from effectful.handlers.jax import jax_getitem
from effectful.handlers.jax import numpy as jnp
from effectful.handlers.weighted.jax import GradientOptimizationReduce, reals
from effectful.handlers.weighted.jax import interpretation as jax_intp
from effectful.handlers.weighted.optimization import ReduceReorderReduction
from effectful.ops.semantics import handler
from effectful.ops.syntax import defop

"""
Consider the following classic POMDP example [1].

    A tiger is put with equal probability behind one of two doors,
    while treasure is put behind the other one. You are standing
    in front of the two closed doors and need to decide which one
    to open. If you open the door with the tiger, you will get hurt
    (negative reward). But if you open the door with treasure, you 
    receive a positive reward. Instead of opening a door right away,
    you also have the option to wait and listen for tiger noises.
    But listening is neither free nor entirely accurate. You might
    hear the tiger behind the left door while it is actually behind
    the right door and vice versa.

[1] Kamalzadeh, Hossein, and Michael Hahsler.
"POMDP: introduction to partially observable Markov decision processes."
"""


def main():
    t = defop(int, name="t")()  # time plate
    noise = 0.15
    noise2 = 1e-9

    # state == 0 means the tiger is behind the left door
    # state == 1 means the tiger is behind the right door
    state = defop(jax.Array, name="state")()

    # action == 0  means you open the left door
    # action == 1  means you open the right door
    # action == 2  means you listen
    action = defop(jax.Array, name="action")()

    # observation == 0  means you hear a growl from the left door
    # observation == 1  means you hear a growl from the right door
    # observation == 2  means you hear nothing
    observation = defop(jax.Array, name="observation")()

    # reward: action -> state -> int
    reward_model = jnp.array(
        [[-100, 10], [10, -100], [-1, -1]]  # small penality for waiting to listen
    )
    reward = jax_getitem(reward_model, (action[t], state[t]))

    # environment: action -> state -> observation
    environment_model = jnp.array(
        [
            [[0, 0, 1], [0, 0, 1]],
            [[0, 0, 1], [0, 0, 1]],
            [[1 - noise, noise], [noise, 1 - noise]],
        ]
    )
    environment = jax_getitem(environment_model, (action[t], state[t], observation[t]))

    # transition: action -> state -> next_state
    transition_model = jnp.array(
        [
            [[0.5, 0.5], [0.5, 0.5]],  # reset when door is opened
            [[0.5, 0.5], [0.5, 0.5]],
            [[1 - noise2, noise2], [noise2, 1 - noise2]],
        ]
    )
    transition = jax_getitem(transition_model, (action[t], state[t], state[t + 1]))

    # policy: observation -> action
    # we take dummy policy that always takes a random action
    policy_model = defop(jax.Array, name="policy_model")()
    policy = jax_getitem(policy_model, (observation[t], action[t + 1]))
    random_policy = jnp.ones((3, 3)) / 3.0

    # you start uninformed about the tiger/treasure door
    initial_state = jax_getitem(jnp.array([0.5, 0.5]), (state[0],))

    # compute the utility of our policy over the first 100 time steps
    streams = {
        state.op: jnp.arange(2),
        action.op: jnp.arange(3),
        observation.op: jnp.arange(3),
    }
    time_stream = {t.op: jnp.arange(100)}

    with handler(jax_intp), handler(ReduceReorderReduction()):
        body = initial_state * reward * environment * policy * transition
        utility = plated(time_stream, Sum(streams, body))

    gradient_intp = GradientOptimizationReduce(
        lr=0.001, steps=1000, init={policy_model: random_policy}
    )
    with handler(gradient_intp):
        optim_policy = ArgMax({policy_model: reals()}, utility)
    print("Found policy:", optim_policy)


if __name__ == "__main__":
    main()
