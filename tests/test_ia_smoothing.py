"""Inequity-aversion reward smoothing must follow e_t = gamma * lambda * e_{t-1} + r_t.

Hughes et al. (2018) and Demir et al. (2026) smooth each agent's reward with a decay of
gamma * lambda before comparing agents. Several envs used 0.99 * 0.01 (~0.0099), which
discards the history almost entirely, and harvest / territory had no smooth_rewards field in
their State, so enabling smoothing crashed.

Run with pytest, or standalone:
  export PYTHONPATH=$PWD:$PYTHONPATH
  JAX_PLATFORMS=cpu python tests/test_ia_smoothing.py
"""

import jax
import numpy as onp

import socialjax

# env id -> decay gamma * lambda used by that env
DECAYS = {
    "clean_up": 0.99 * 0.9,
    "coin_game": 0.99 * 0.9,
    "harvest_common_open": 0.99 * 0.9,
    "mushrooms": 0.99 * 0.9,
    "territory_open": 0.99 * 0.9,
    "gift": 0.99 * 0.99,
    "coop_mining": 0.99 * 0.99,
}
STEPS = 30


def run(name, fn):
    fn()
    print(f"ok: {name}")


def check_env(env_id):
    env = socialjax.make(
        env_id, shared_rewards=False, inequity_aversion=True, enable_smooth_rewards=True
    )
    n = env.num_agents
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    step = jax.jit(env.step)
    prev = onp.zeros(n)
    for t in range(STEPS):
        key, k_act, k_step = jax.random.split(key, 3)
        acts = jax.random.randint(k_act, (n,), 0, env.action_space(0).n)
        _, state, _, _, info = step(k_step, state, [acts[i] for i in range(n)])
        smooth = onp.asarray(info["smooth_rewards"], dtype=onp.float64).reshape(-1)
        raw = onp.asarray(info["original_rewards"], dtype=onp.float64).reshape(-1)
        assert smooth.shape == (n,), f"{env_id} step {t}: smooth_rewards shape {smooth.shape}"
        expected = DECAYS[env_id] * prev + raw
        assert onp.allclose(smooth, expected, atol=1e-4), (
            f"{env_id} step {t}: smooth {smooth.tolist()} != expected {expected.tolist()}"
        )
        prev = smooth


def test_clean_up():
    check_env("clean_up")


def test_coin_game():
    check_env("coin_game")


def test_harvest_common_open():
    check_env("harvest_common_open")


def test_mushrooms():
    check_env("mushrooms")


def test_territory_open():
    check_env("territory_open")


def test_gift():
    check_env("gift")


def test_coop_mining():
    check_env("coop_mining")


if __name__ == "__main__":
    for env_id in DECAYS:
        run(f"IA smoothing: {env_id}", lambda: check_env(env_id))
    print("ALL IA SMOOTHING TESTS PASSED")
