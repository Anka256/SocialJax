"""Tests for the Recognition env: API shape, invariants, and every rule of the design doc.

Run with pytest, or standalone:
  export PYTHONPATH=$PWD:$PYTHONPATH
  JAX_PLATFORMS=cpu python tests/test_recognition.py
"""

import jax
import jax.numpy as jnp
import numpy as onp

import socialjax
from socialjax.environments.movement import positions_unique
from socialjax.environments.recognition.recognition import (
    OBS_K,
    OBS_OTHER,
    OBS_OWN_STATUS,
    OBS_SELF,
    OBS_STATUS,
    OBS_WALL,
    Actions,
)

NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3


def make(**kwargs):
    return socialjax.make("recognition", **kwargs)


def place(env, locs, K=None, M=None):
    """A state with agents at the given (row, col, heading) and the given recognition."""
    _, state = env.reset(jax.random.PRNGKey(0))
    n = env.num_agents
    K = jnp.zeros((n, n)) if K is None else jnp.asarray(K, jnp.float32)
    M = K if M is None else jnp.asarray(M, jnp.float32)
    return state.replace(agent_locs=jnp.asarray(locs, jnp.int32), K=K, M=M)


def stay(env, state, key=0):
    actions = [jnp.int32(Actions.stay)] * env.num_agents
    return env.step_env(jax.random.PRNGKey(key), state, actions)


def test_api_shapes():
    env = make()
    obs, state = env.reset(jax.random.PRNGKey(0))
    n = env.num_agents
    assert obs.shape == (n, *env.observation_space()[1])
    obs, state, rew, done, info = stay(env, state)
    assert rew.shape == (n,)
    assert set(done) == {str(a) for a in range(n)} | {"__all__"}
    for name, value in info.items():
        assert value.shape == (n,), name


def test_rollout_invariants():
    env = make()
    key = jax.random.PRNGKey(1)
    _, state = env.reset(key)
    step = jax.jit(env.step_env)
    for _ in range(300):
        key, k_act, k_step = jax.random.split(key, 3)
        actions = jax.random.randint(k_act, (env.num_agents,), 0, env.num_actions)
        _, state, _, _, _ = step(k_step, state, actions)
        K, M = onp.asarray(state.K), onp.asarray(state.M)
        assert (K >= 0).all() and (K <= 1).all()
        assert (M >= K - 1e-6).all(), "memory must never be below current recognition"
        assert onp.allclose(onp.diag(K), 0) and onp.allclose(onp.diag(M), 0)
        assert positions_unique(state.agent_locs)


def test_recognition_is_directed():
    # agent 1 stands 3 cells in front of agent 0; both face north, so 1 cannot see 0
    env = make(num_agents=2)
    state = place(env, [[12, 12, NORTH], [9, 12, NORTH]])
    _, state, _, _, _ = stay(env, state)
    g = 1 - 3 / (env.base_range + 1)  # closeness at distance 3, status 0
    assert jnp.isclose(state.K[0, 1], env.growth_rate * g)
    assert state.K[1, 0] == 0


def test_out_of_cone_is_not_seen():
    # agent 1 is 3 cells to the side of agent 0: outside the 90-degree front cone
    env = make(num_agents=2)
    state = place(env, [[12, 12, NORTH], [12, 15, NORTH]])
    _, state, _, _, _ = stay(env, state)
    assert state.K[0, 1] == 0


def test_attention_budget():
    # agents 1 and 2 stand 2 and 4 cells in front of agent 0
    locs = [[12, 12, NORTH], [10, 12, SOUTH], [8, 12, SOUTH]]
    for budget, expected in [(1, [True, False]), (2, [True, True]), (0, [True, True])]:
        env = make(num_agents=3, attention_budget=budget, base_range=5.0)
        _, state, _, _, _ = stay(env, place(env, locs))
        assert [bool(state.K[0, j] > 0) for j in (1, 2)] == expected, f"budget={budget}"


def test_attention_goes_to_the_nearest_not_the_lowest_index():
    # agent 1 at distance sqrt(17), agent 2 at distance 4: agent 2 is nearer despite its index
    env = make(num_agents=3, attention_budget=1, base_range=5.0)
    _, state, _, _, _ = stay(env, place(env, [[12, 12, NORTH], [8, 13, SOUTH], [8, 12, SOUTH]]))
    assert state.K[0, 1] == 0 and state.K[0, 2] > 0
    # exact tie (both at distance sqrt(5)): the lower index wins
    _, state, _, _, _ = stay(env, place(env, [[12, 12, NORTH], [10, 11, SOUTH], [10, 13, SOUTH]]))
    assert state.K[0, 1] > 0 and state.K[0, 2] == 0


def test_memory_speeds_up_recognition():
    env = make(num_agents=2)
    locs = [[12, 12, NORTH], [9, 12, NORTH]]
    fresh = stay(env, place(env, locs))[1].K[0, 1]
    remembered = stay(env, place(env, locs, M=[[0, 0.5], [0, 0]]))[1].K[0, 1]
    assert jnp.isclose(remembered, fresh * (1 + env.memory_boost * 0.5))


def test_decay_and_memory():
    # back to back, far apart: nobody sees anybody
    env = make(num_agents=2)
    K = [[0, 0.5], [0, 0]]
    state = place(env, [[5, 5, NORTH], [20, 20, SOUTH]], K=K)
    _, state, _, _, info = stay(env, state)
    decay = env.recognition_decay * (1 + 0.5 / env.recognition_capacity)  # load of agent 0 is 0.5
    assert jnp.isclose(state.K[0, 1], 0.5 * (1 - decay))
    assert jnp.isclose(state.M[0, 1], 0.5 * (1 - env.memory_decay))
    assert info["sleep_events"].sum() == 0


def test_knowing_more_agents_makes_each_fade_faster():
    # agent 0 knows agent 1 alone, or agents 1..3; nobody sees anybody
    locs = [[2, 2, NORTH], [22, 2, SOUTH], [2, 22, NORTH], [22, 22, SOUTH]]
    alone = onp.zeros((4, 4))
    alone[0, 1] = 0.6
    crowd = alone.copy()
    crowd[0, 2] = crowd[0, 3] = 0.6
    env = make(num_agents=4, recognition_capacity=3.0)
    k_alone = stay(env, place(env, locs, K=alone))[1].K[0, 1]
    after = stay(env, place(env, locs, K=crowd))[1]
    assert after.K[0, 1] < k_alone
    assert jnp.isclose(after.K[0, 1], 0.6 * (1 - env.recognition_decay * (1 + 1.8 / 3.0)))
    assert jnp.isclose(after.M[0, 1], 0.6 * (1 - env.memory_decay)), "memory is not affected by load"
    # capacity off: plain decay
    env = make(num_agents=4, recognition_capacity=0.0)
    assert jnp.isclose(stay(env, place(env, locs, K=crowd))[1].K[0, 1], 0.6 * (1 - env.recognition_decay))


def test_sleep_and_forget_events():
    env = make(num_agents=2)
    locs = [[5, 5, NORTH], [20, 20, SOUTH]]
    # K just above the sleep threshold falls below it after one decay step
    state = place(env, locs, K=[[0, 0.2001], [0, 0]])
    _, state, _, _, info = stay(env, state)
    assert info["sleep_events"][1] == 1 and info["forget_events"][1] == 0
    # memory just above the forget threshold: the bond is forgotten and reset
    state = place(env, locs, K=[[0, 0.01], [0, 0]], M=[[0, 0.05001], [0, 0]])
    _, state, _, _, info = stay(env, state)
    assert info["forget_events"][1] == 1
    assert state.K[0, 1] == 0 and state.M[0, 1] == 0


def test_reward_formula():
    # agent 0 is known by agent 2 (s_0 = 0.9) and looks at agent 1 (s_1 = 0);
    # agent 2 faces away, so its recognition of 0 decays
    env = make(num_agents=3)
    K = onp.zeros((3, 3))
    K[2, 0] = 0.9
    state = place(env, [[12, 12, NORTH], [9, 12, NORTH], [20, 20, SOUTH]], K=K)
    _, new, rew, _, info = stay(env, state)
    s0 = 0.9
    gain = new.K[0, 1]  # K[0, 1] was 0
    assert jnp.isclose(rew[1], (1 + s0) / (1 + 0) * gain), "recognised by a known agent: weight 1.9"
    loss = new.K[2, 0] - 0.9
    assert loss < 0
    assert jnp.isclose(rew[0], (1 + 0) / (1 + s0) * loss), "losing recognition is penalised"
    assert jnp.allclose(info["individual_rewards"], rew)


def test_shared_rewards():
    env = make(num_agents=3, shared_rewards=True)
    K = onp.zeros((3, 3))
    K[2, 0] = 0.9
    state = place(env, [[12, 12, NORTH], [9, 12, NORTH], [20, 20, SOUTH]], K=K)
    _, _, rew, _, info = stay(env, state)
    assert jnp.allclose(rew, info["individual_rewards"].sum())


def test_visibility_grows_with_fame():
    # agent 1 stands 5 cells ahead of agent 0, beyond the base range of 3
    locs = [[12, 12, NORTH], [7, 12, NORTH], [1, 1, NORTH], [1, 23, NORTH], [23, 1, SOUTH], [23, 23, SOUTH]]
    K = onp.zeros((6, 6))
    K[2:, 1] = 1.0  # agents 2..5 know agent 1 fully: s_1 = 4, R_1 = 3 + 1.5 ln 5 = 5.4
    for c, expect_seen in [(1.5, True), (0.0, False)]:
        env = make(num_agents=6, fame_visibility=c)
        _, state, _, _, _ = stay(env, place(env, locs, K=K))
        assert bool(state.K[0, 1] > 0) == expect_seen, f"fame_visibility={c}"


def test_observation_layout():
    env = make(num_agents=2)
    # agent 0 faces east, agent 1 stands 3 cells ahead of it
    state = place(env, [[12, 12, EAST], [12, 15, WEST]], K=[[0, 0.3], [0.6, 0]])
    obs = env.get_obs(state)
    fwd, side = env.view_forward, env.view_side
    assert obs[0, fwd, side, OBS_SELF] == 1 and obs[0, ..., OBS_SELF].sum() == 1
    assert obs[0, fwd - 3, side, OBS_OTHER] == 1 and obs[0, ..., OBS_OTHER].sum() == 1
    assert jnp.isclose(obs[0, fwd - 3, side, OBS_K], 0.3)
    assert jnp.isclose(obs[0, fwd - 3, side, OBS_STATUS], 0.3)  # s_1 = 0.3, N - 1 = 1
    assert jnp.allclose(obs[0, ..., OBS_OWN_STATUS], 0.6)
    # an agent in the top-left corner facing north sees wall everywhere ahead of it
    state = place(env, [[0, 0, NORTH], [20, 20, NORTH]])
    obs = env.get_obs(state)
    assert (obs[0, :fwd, :, OBS_WALL] == 1).all()
    assert obs[0, fwd, side, OBS_WALL] == 0


def test_unequal_start():
    env = make(famous_agents=2, famous_init_recognition=0.5)
    _, state = env.reset(jax.random.PRNGKey(0))
    s = state.K.sum(axis=0)
    assert jnp.allclose(s[:2], 0.5 * (env.num_agents - 1)) and jnp.allclose(s[2:], 0)


def test_episode_end_and_auto_reset():
    env = make(num_inner_steps=5)
    _, state = env.reset(jax.random.PRNGKey(0))
    actions = [jnp.int32(Actions.stay)] * env.num_agents
    for t in range(5):
        _, state, _, done, _ = env.step(jax.random.PRNGKey(t), state, actions)
    assert done["__all__"] and state.t == 0


def test_vmap_over_envs():
    env = make()
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    obs, state = jax.vmap(env.reset)(keys)
    actions = [jnp.zeros(4, jnp.int32)] * env.num_agents
    obs, state, rew, done, info = jax.vmap(env.step)(keys, state, actions)
    assert obs.shape[:2] == (4, env.num_agents) and rew.shape == (4, env.num_agents)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok: {name}")
    print("ALL RECOGNITION TESTS PASSED")
