"""IPPO on the Recognition env: shared or per-agent networks, several seeds, resumable.

    export PYTHONPATH=$PWD:$PYTHONPATH
    python algorithms/recognition/train_ippo.py                                 # individual reward
    python algorithms/recognition/train_ippo.py ENV_KWARGS.shared_rewards=true
    python algorithms/recognition/train_ippo.py PARAMETER_SHARING=false NUM_SEEDS=3
    python algorithms/recognition/train_ippo.py hydra.run.dir=<run dir> RESUME=true   # continue

Training runs in chunks of CHECKPOINT_EVERY updates; after each chunk the whole
training state (networks, optimiser, envs, RNG, metrics so far) is written to
<run dir>/checkpoint.msgpack, so an interrupted run continues where it stopped.

Everything a run produces lands in its run directory:
    config.json            the configuration the run was started with
    train_ippo.log         progress log
    metrics_seed<k>.csv    one row per update (losses, returns, inequality metrics)
    params_seed<k>.pkl     trained network parameters
    eval_curve_seed<k>.csv inequality over the steps of an episode, trained policy
    eval_seed<k>.gif       one evaluation episode
    summary.json           final values of the main metrics, per seed
"""

import csv
import json
import logging
import os
import time
from typing import NamedTuple

import distrax
import hydra
import jax
import jax.numpy as jnp
import numpy as onp
import optax
from flax import serialization
from flax.training.train_state import TrainState
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from PIL import Image

import socialjax
from algorithms.utils.io_utils import save_params
from algorithms.utils.networks import ActorCritic

# Hydra sends this logger to the console and to <run dir>/train_ippo.log as the run goes.
log = logging.getLogger(__name__)


class Transition(NamedTuple):
    obs: jnp.ndarray  # (..., H, W, C)
    action: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray


def masked_mean(x, mask):
    """Mean of x over entries where mask is set; NaN when nothing is set."""
    return jnp.where(mask.sum() > 0, (x * mask).sum() / jnp.maximum(mask.sum(), 1), jnp.nan)


def make_env(config):
    return socialjax.make("recognition", **config["ENV_KWARGS"])


class Policy:
    """The agents' networks: one shared network, or one per agent (params stacked on axis 0)."""

    def __init__(self, env, config):
        self.n = env.num_agents
        self.shared = config["PARAMETER_SHARING"]
        self.obs_shape = env.observation_space()[1]
        self.network = ActorCritic(env.num_actions, activation=config["ACTIVATION"])

    def logits_value(self, params, obs):
        """One network on a batch of observations (B, H, W, C) -> logits (B, A), value (B,)."""
        pi, value = self.network.apply(params, obs)
        return pi.logits, value

    def init(self, key):
        dummy = jnp.zeros((1, *self.obs_shape))
        if self.shared:
            return self.network.init(key, dummy)
        return jax.vmap(self.network.init, in_axes=(0, None))(jax.random.split(key, self.n), dummy)

    def __call__(self, params, obs):
        """obs (E, N, H, W, C) -> logits (E, N, A), value (E, N)."""
        num_envs = obs.shape[0]
        if self.shared:
            logits, value = self.logits_value(params, obs.reshape(num_envs * self.n, *self.obs_shape))
            return logits.reshape(num_envs, self.n, -1), value.reshape(num_envs, self.n)
        return jax.vmap(self.logits_value, in_axes=(0, 1), out_axes=1)(params, obs)


def make_train(config):
    """Returns init(rng) -> runner and run_chunk(runner, start, length) -> (runner, metrics)."""
    env = make_env(config)
    policy = Policy(env, config)
    n, num_envs, num_steps = env.num_agents, config["NUM_ENVS"], config["NUM_STEPS"]
    num_updates = int(config["TOTAL_TIMESTEPS"] // (num_envs * num_steps))
    famous = jnp.arange(n) < env.famous_agents

    def lr_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / num_updates
        return config["LR"] * frac

    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(lr_schedule if config["ANNEAL_LR"] else config["LR"], eps=1e-5),
    )

    def init(rng):
        rng, k_init, k_reset = jax.random.split(rng, 3)
        make_state = lambda params: TrainState.create(apply_fn=policy.network.apply, params=params, tx=tx)
        params = policy.init(k_init)
        # per-agent networks get per-agent optimiser states (and per-agent gradient clipping)
        train_state = make_state(params) if policy.shared else jax.vmap(make_state)(params)
        obs, env_state = jax.vmap(env.reset)(jax.random.split(k_reset, num_envs))
        return train_state, env_state, obs, jnp.zeros((num_envs, n)), rng

    def env_step(carry, _):
        train_state, env_state, obs, ep_return, rng = carry
        rng, k_act, k_step = jax.random.split(rng, 3)
        logits, value = policy(train_state.params, obs)
        pi = distrax.Categorical(logits=logits)
        action = pi.sample(seed=k_act)  # (E, N)
        next_obs, env_state, reward, done, info = jax.vmap(env.step)(
            jax.random.split(k_step, num_envs), env_state, action
        )
        ended = done["__all__"]  # (E,)
        ep_return = ep_return + info["individual_rewards"]
        stats = {
            "ended": ended,
            "final_return": ep_return,
            "final_status": info["status"],
            "final_status_gini": info["status_gini"][:, 0],
            "final_isolated": info["num_isolated"][:, 0],
            "status_gini": info["status_gini"][:, 0],
            "forget_events": info["forget_events"].sum(-1),
        }
        ep_return = jnp.where(ended[:, None], 0.0, ep_return)
        dones = jnp.broadcast_to(ended[:, None], (num_envs, n))
        transition = Transition(obs, action, pi.log_prob(action), value, reward, dones)
        return (train_state, env_state, next_obs, ep_return, rng), (transition, stats)

    def gae(traj, last_value):
        def step(carry, t):
            adv, next_value = carry
            delta = t.reward + config["GAMMA"] * next_value * (1 - t.done) - t.value
            adv = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - t.done) * adv
            return (adv, t.value), adv

        _, advantages = jax.lax.scan(step, (jnp.zeros_like(last_value), last_value), traj, reverse=True, unroll=16)
        return advantages, advantages + traj.value

    def loss_fn(params, traj, advantages, targets):
        logits, value = policy.logits_value(params, traj.obs)
        pi = distrax.Categorical(logits=logits)
        log_prob = pi.log_prob(traj.action)
        value_clipped = traj.value + (value - traj.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
        value_loss = 0.5 * jnp.maximum(jnp.square(value - targets), jnp.square(value_clipped - targets)).mean()
        ratio = jnp.exp(log_prob - traj.log_prob)
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor_loss = -jnp.minimum(
            ratio * adv, jnp.clip(ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"]) * adv
        ).mean()
        entropy = pi.entropy().mean()
        total = actor_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
        return total, (value_loss, actor_loss, entropy)

    def ppo_update(train_state, traj, advantages, targets, rng):
        """PPO epochs for one network; every leaf has leading dims (S, B)."""
        batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), (traj, advantages, targets))
        batch_size = advantages.size

        def epoch(carry, _):
            train_state, rng = carry
            rng, k_perm = jax.random.split(rng)
            perm = jax.random.permutation(k_perm, batch_size)
            minibatches = jax.tree.map(
                lambda x: x[perm].reshape(config["NUM_MINIBATCHES"], -1, *x.shape[1:]), batch
            )

            def step(train_state, mb):
                grads, aux = jax.grad(loss_fn, has_aux=True)(train_state.params, *mb)
                return train_state.apply_gradients(grads=grads), aux

            train_state, aux = jax.lax.scan(step, train_state, minibatches)
            return (train_state, rng), aux

        (train_state, _), aux = jax.lax.scan(epoch, (train_state, rng), None, config["UPDATE_EPOCHS"])
        return train_state, jax.tree.map(jnp.mean, aux)

    def update(runner, update_idx):
        (train_state, env_state, obs, ep_return, rng), (traj, stats) = jax.lax.scan(
            env_step, runner, None, num_steps
        )
        _, last_value = policy(train_state.params, obs)
        advantages, targets = gae(traj, last_value)  # (S, E, N)
        rng, k_update = jax.random.split(rng)
        if policy.shared:
            merge = lambda x: x.reshape(num_steps, num_envs * n, *x.shape[3:])
            train_state, (value_loss, actor_loss, entropy) = ppo_update(
                train_state, *jax.tree.map(merge, (traj, advantages, targets)), k_update
            )
        else:
            per_agent = lambda x: jnp.moveaxis(x, 2, 0)  # (N, S, E, ...)
            train_state, (value_loss, actor_loss, entropy) = jax.vmap(ppo_update)(
                train_state,
                *jax.tree.map(per_agent, (traj, advantages, targets)),
                jax.random.split(k_update, n),
            )

        ended = stats["ended"].astype(jnp.float32)  # (S, E)
        per_agent_mask = ended[..., None] * jnp.ones(n)  # (S, E, N)
        metrics = {
            "env_step": (update_idx + 1) * num_envs * num_steps,
            "value_loss": value_loss.mean(),
            "actor_loss": actor_loss.mean(),
            "entropy": entropy.mean(),
            "train_reward": traj.reward.mean(),
            "status_gini_running": stats["status_gini"].mean(),
            "forget_events_per_step": stats["forget_events"].mean(),
            "episodes_ended": ended.sum(),
            "episode_return": masked_mean(stats["final_return"], per_agent_mask),
            "episode_return_min_agent": masked_mean(stats["final_return"].min(-1), ended),
            "episode_return_max_agent": masked_mean(stats["final_return"].max(-1), ended),
            "status_gini": masked_mean(stats["final_status_gini"], ended),
            "num_isolated": masked_mean(stats["final_isolated"], ended),
            "status_mean": masked_mean(stats["final_status"], per_agent_mask),
        }
        if env.famous_agents > 0:
            famous_mask, others_mask = per_agent_mask * famous, per_agent_mask * ~famous
            metrics["famous_status"] = masked_mean(stats["final_status"], famous_mask)
            metrics["others_status"] = masked_mean(stats["final_status"], others_mask)
            metrics["famous_return"] = masked_mean(stats["final_return"], famous_mask)
            metrics["others_return"] = masked_mean(stats["final_return"], others_mask)
        return (train_state, env_state, obs, ep_return, rng), metrics

    def run_chunk(runner, start, length):
        return jax.lax.scan(update, runner, start + jnp.arange(length))

    return init, run_chunk, num_updates


def eval_curves(env, policy, params, key, num_episodes):
    """Play full episodes with the trained policy; per step, the status Gini, isolated agents,
    the status of the agents ranked 1..N (best known first) and, with an unequal start, the
    mean status of the famous-born agents and of the others; averaged over episodes."""
    keys = jax.random.split(key, num_episodes + 1)
    obs, state = jax.vmap(env.reset)(keys[1:])

    def step(carry, _):
        obs, state, key = carry
        key, k_act, k_step = jax.random.split(key, 3)
        logits, _ = policy(params, obs)
        action = distrax.Categorical(logits=logits).sample(seed=k_act)
        obs, state, _, _, info = jax.vmap(env.step_env)(
            jax.random.split(k_step, num_episodes), state, action
        )
        ranked = jnp.sort(info["status"], axis=-1)[:, ::-1]  # (episodes, N), descending
        out = {
            "gini_mean": info["status_gini"][:, 0].mean(),
            "gini_std": info["status_gini"][:, 0].std(),
            "isolated_mean": info["num_isolated"][:, 0].mean(),
            "status_by_rank": ranked.mean(0),
        }
        if env.famous_agents > 0:
            out["famous_status"] = info["status"][:, : env.famous_agents].mean()
            out["others_status"] = info["status"][:, env.famous_agents :].mean()
        return (obs, state, key), out

    _, curves = jax.lax.scan(step, (obs, state, keys[0]), None, env.num_inner_steps)
    return curves


def render_episode(env, policy, params, key, frames):
    """Frames of one episode with the trained (stochastic) policy."""

    @jax.jit
    def step(key, state, obs):
        k_act, k_step = jax.random.split(key)
        logits, _ = policy(params, obs[None])
        action = distrax.Categorical(logits=logits[0]).sample(seed=k_act)
        return env.step(k_step, state, action)

    obs, state = env.reset(key)
    images = [env.render(state)]
    for _ in range(frames):
        key, k = jax.random.split(key)
        obs, state, _, _, _ = step(k, state, obs)
        images.append(env.render(state))
    return images


def last_finite(values):
    finite = values[onp.isfinite(values)]
    return finite[-1] if finite.size else onp.nan


def save_checkpoint(path, runner, next_update, metrics):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(serialization.msgpack_serialize({
            "runner": serialization.to_state_dict(jax.device_get(runner)),
            "next_update": next_update,
            "metrics": metrics,
        }))
    os.replace(tmp, path)  # atomic: an interrupted save never corrupts the last checkpoint


def load_checkpoint(path, runner_template):
    with open(path, "rb") as f:
        data = serialization.msgpack_restore(f.read())
    runner = serialization.from_state_dict(runner_template, data["runner"])
    metrics = {k: onp.asarray(v) for k, v in data["metrics"].items()}
    return runner, int(data["next_update"]), metrics


def run(config, out_dir):
    ckpt_path = os.path.join(out_dir, "checkpoint.msgpack")
    config_path = os.path.join(out_dir, "config.json")
    if config["RESUME"]:
        with open(config_path) as f:
            saved = json.load(f)
        # the saved configuration wins; only the training length may be extended
        saved["TOTAL_TIMESTEPS"] = config["TOTAL_TIMESTEPS"]
        config = {**saved, "RESUME": True}
    else:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    config["ENV_KWARGS"]["grid_size"] = tuple(config["ENV_KWARGS"]["grid_size"])

    init, run_chunk, num_updates = make_train(config)
    seeds = jnp.arange(config["NUM_SEEDS"])
    runner = jax.jit(jax.vmap(init))(jax.vmap(jax.random.PRNGKey)(config["SEED"] + seeds))
    start, metrics = 0, {}
    if config["RESUME"] and os.path.exists(ckpt_path):
        runner, start, metrics = load_checkpoint(ckpt_path, runner)
        log.info(f"Resumed from update {start} of {num_updates}")
    log.info(
        f"{num_updates} updates x {config['NUM_ENVS']} envs x {config['NUM_STEPS']} steps, "
        f"{config['NUM_SEEDS']} seed(s), {'shared network' if config['PARAMETER_SHARING'] else 'one network per agent'}"
    )

    chunks = {}  # jitted chunk functions by length (the last chunk may be shorter)
    t0, done_at_t0 = time.time(), start
    while start < num_updates:
        length = min(config["CHECKPOINT_EVERY"], num_updates - start)
        if length not in chunks:
            # seeds run one after another (lax.map, not vmap): the GPU is already saturated
            # by one seed's envs, and this keeps memory to one seed's worth on an 8 GB card
            chunks[length] = jax.jit(
                lambda r, s, L=length: jax.lax.map(lambda seed_runner: run_chunk(seed_runner, s, L), r)
            )
        runner, chunk_metrics = chunks[length](runner, start)
        chunk_metrics = jax.device_get(chunk_metrics)  # (seeds, length) per metric
        metrics = {
            k: onp.concatenate([metrics[k], v], axis=1) if k in metrics else v
            for k, v in chunk_metrics.items()
        }
        start += length
        save_checkpoint(ckpt_path, runner, start, metrics)

        elapsed = time.time() - t0
        eta = elapsed / (start - done_at_t0) * (num_updates - start)
        for k in range(config["NUM_SEEDS"]):
            m = {name: last_finite(v[k]) for name, v in metrics.items()}
            log.info(
                f"[seed {k}] {start}/{num_updates} updates, step {int(m['env_step']):>10,} | "
                f"return {m['episode_return']:7.3f} | status gini {m['status_gini']:.3f} | "
                f"isolated {m['num_isolated']:4.1f} | entropy {m['entropy']:.3f} | "
                f"remaining ~{eta / 60:.0f} min"
            )
    return config, runner, metrics


def finish(config, runner, metrics, out_dir):
    env = make_env(config)
    policy = Policy(env, config)
    train_state = runner[0]
    summary = {}
    for k in range(config["NUM_SEEDS"]):
        seed_metrics = {name: v[k] for name, v in metrics.items()}
        with open(os.path.join(out_dir, f"metrics_seed{k}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(seed_metrics)
            writer.writerows(zip(*seed_metrics.values()))

        state_k = jax.tree.map(lambda x: x[k], train_state)
        save_params(state_k, os.path.join(out_dir, f"params_seed{k}.pkl"))
        summary[f"seed{k}"] = {name: float(last_finite(v)) for name, v in seed_metrics.items()}

        if config["EVAL_EPISODES"] > 0:
            curves = jax.device_get(
                jax.jit(eval_curves, static_argnums=(0, 1, 4))(
                    env, policy, state_k.params, jax.random.PRNGKey(2000 + k), config["EVAL_EPISODES"]
                )
            )
            ranks = [f"status_rank{r + 1}" for r in range(env.num_agents)]
            groups = ["famous_status", "others_status"] if env.famous_agents > 0 else []
            with open(os.path.join(out_dir, f"eval_curve_seed{k}.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "gini_mean", "gini_std", "isolated_mean", *ranks, *groups])
                for t in range(env.num_inner_steps):
                    writer.writerow([
                        t + 1, curves["gini_mean"][t], curves["gini_std"][t],
                        curves["isolated_mean"][t], *curves["status_by_rank"][t],
                        *(curves[g][t] for g in groups),
                    ])
            summary[f"seed{k}"]["eval_final_gini"] = float(curves["gini_mean"][-1])
            summary[f"seed{k}"]["eval_final_isolated"] = float(curves["isolated_mean"][-1])

        if config["GIF_FRAMES"] > 0:
            images = render_episode(env, policy, state_k.params, jax.random.PRNGKey(1000 + k), config["GIF_FRAMES"])
            frames = [Image.fromarray(img) for img in images]
            frames[0].save(
                os.path.join(out_dir, f"eval_seed{k}.gif"),
                save_all=True, append_images=frames[1:], duration=80, loop=0,
            )

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info(json.dumps(summary, indent=2))


@hydra.main(version_base=None, config_path="config", config_name="ippo")
def main(cfg):
    config = OmegaConf.to_container(cfg, resolve=True)
    out_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Run directory: {out_dir}")
    config, runner, metrics = run(config, out_dir)
    finish(config, runner, metrics, out_dir)


if __name__ == "__main__":
    main()
