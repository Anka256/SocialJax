"""IPPO (parameter-shared PPO) on the Recognition env, in a single jitted loop.

    export PYTHONPATH=$PWD:$PYTHONPATH
    python algorithms/recognition/train_ippo.py                              # individual reward
    python algorithms/recognition/train_ippo.py ENV_KWARGS.shared_rewards=true
    python algorithms/recognition/train_ippo.py TOTAL_TIMESTEPS=2e6 GIF_FRAMES=0  # smoke test

Everything a run produces lands in its Hydra output directory:
    metrics_seed<k>.csv   one row per update (losses, returns, inequality metrics)
    params_seed<k>.pkl    trained network parameters
    eval_seed<k>.gif      one evaluation episode with the trained policy
    summary.json          final values of the main metrics, per seed
"""

import csv
import json
import logging
import os
from typing import NamedTuple

import hydra
import jax
import jax.numpy as jnp
import numpy as onp
import optax
from flax.training.train_state import TrainState
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from PIL import Image

import socialjax
from algorithms.utils.io_utils import save_params
from algorithms.utils.networks import ActorCritic

# Hydra sends this logger to the console and to <output dir>/train_ippo.log as the run goes.
log = logging.getLogger(__name__)


class Transition(NamedTuple):
    obs: jnp.ndarray
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


def make_train(config, log_fn=None):
    env = make_env(config)
    n, num_envs, num_steps = env.num_agents, config["NUM_ENVS"], config["NUM_STEPS"]
    num_updates = int(config["TOTAL_TIMESTEPS"] // (num_envs * num_steps))
    batch_size = num_envs * num_steps * n
    assert batch_size % config["NUM_MINIBATCHES"] == 0
    obs_shape = env.observation_space()[1]
    famous = jnp.arange(n) < env.famous_agents
    network = ActorCritic(env.num_actions, activation=config["ACTIVATION"])

    def lr_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / num_updates
        return config["LR"] * frac

    def train(rng, seed_id):
        rng, k_init, k_reset = jax.random.split(rng, 3)
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(lr_schedule if config["ANNEAL_LR"] else config["LR"], eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network.init(k_init, jnp.zeros((1, *obs_shape))),
            tx=tx,
        )
        obs, env_state = jax.vmap(env.reset)(jax.random.split(k_reset, num_envs))
        ep_return = jnp.zeros((num_envs, n))

        def env_step(carry, _):
            train_state, env_state, obs, ep_return, rng = carry
            rng, k_act, k_step = jax.random.split(rng, 3)
            flat_obs = obs.reshape(num_envs * n, *obs_shape)
            pi, value = network.apply(train_state.params, flat_obs)
            action = pi.sample(seed=k_act)
            obs, env_state, reward, done, info = jax.vmap(env.step)(
                jax.random.split(k_step, num_envs), env_state, action.reshape(num_envs, n)
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
            transition = Transition(
                flat_obs, action, pi.log_prob(action), value, reward.reshape(-1), jnp.repeat(ended, n)
            )
            return (train_state, env_state, obs, ep_return, rng), (transition, stats)

        def gae(traj, last_value):
            def step(carry, t):
                adv, next_value = carry
                delta = t.reward + config["GAMMA"] * next_value * (1 - t.done) - t.value
                adv = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - t.done) * adv
                return (adv, t.value), adv

            _, advantages = jax.lax.scan(
                step, (jnp.zeros_like(last_value), last_value), traj, reverse=True, unroll=16
            )
            return advantages, advantages + traj.value

        def loss_fn(params, traj, advantages, targets):
            pi, value = network.apply(params, traj.obs)
            log_prob = pi.log_prob(traj.action)
            value_clipped = traj.value + (value - traj.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
            value_loss = 0.5 * jnp.maximum(
                jnp.square(value - targets), jnp.square(value_clipped - targets)
            ).mean()
            ratio = jnp.exp(log_prob - traj.log_prob)
            adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            actor_loss = -jnp.minimum(
                ratio * adv, jnp.clip(ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"]) * adv
            ).mean()
            entropy = pi.entropy().mean()
            total = actor_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
            return total, (value_loss, actor_loss, entropy)

        def update_epoch(carry, _):
            train_state, traj, advantages, targets, rng = carry
            rng, k_perm = jax.random.split(rng)
            perm = jax.random.permutation(k_perm, batch_size)
            shuffle = lambda x: x.reshape(batch_size, *x.shape[2:])[perm].reshape(
                config["NUM_MINIBATCHES"], -1, *x.shape[2:]
            )
            minibatches = jax.tree.map(shuffle, (traj, advantages, targets))

            def update_minibatch(train_state, mb):
                grads, aux = jax.grad(loss_fn, has_aux=True)(train_state.params, *mb)
                return train_state.apply_gradients(grads=grads), aux

            train_state, aux = jax.lax.scan(update_minibatch, train_state, minibatches)
            return (train_state, traj, advantages, targets, rng), aux

        def update(carry, update_idx):
            train_state, env_state, obs, ep_return, rng = carry
            (train_state, env_state, obs, ep_return, rng), (traj, stats) = jax.lax.scan(
                env_step, (train_state, env_state, obs, ep_return, rng), None, num_steps
            )
            _, last_value = network.apply(train_state.params, obs.reshape(num_envs * n, *obs_shape))
            advantages, targets = gae(traj, last_value)
            (train_state, *_, rng), (value_loss, actor_loss, entropy) = jax.lax.scan(
                update_epoch, (train_state, traj, advantages, targets, rng), None, config["UPDATE_EPOCHS"]
            )

            ended = stats["ended"].astype(jnp.float32)  # (S, E)
            per_agent = ended[..., None] * jnp.ones(n)  # (S, E, N)
            metrics = {
                "env_step": (update_idx + 1) * num_envs * num_steps,
                "value_loss": value_loss.mean(),
                "actor_loss": actor_loss.mean(),
                "entropy": entropy.mean(),
                "train_reward": traj.reward.mean(),
                "status_gini_running": stats["status_gini"].mean(),
                "forget_events_per_step": stats["forget_events"].mean(),
                "episodes_ended": ended.sum(),
                "episode_return": masked_mean(stats["final_return"], per_agent),
                "episode_return_min_agent": masked_mean(stats["final_return"].min(-1), ended),
                "episode_return_max_agent": masked_mean(stats["final_return"].max(-1), ended),
                "status_gini": masked_mean(stats["final_status_gini"], ended),
                "num_isolated": masked_mean(stats["final_isolated"], ended),
                "status_mean": masked_mean(stats["final_status"], per_agent),
            }
            if env.famous_agents > 0:
                famous_mask = per_agent * famous
                others_mask = per_agent * ~famous
                metrics["famous_status"] = masked_mean(stats["final_status"], famous_mask)
                metrics["others_status"] = masked_mean(stats["final_status"], others_mask)
                metrics["famous_return"] = masked_mean(stats["final_return"], famous_mask)
                metrics["others_return"] = masked_mean(stats["final_return"], others_mask)
            if log_fn is not None:
                jax.debug.callback(log_fn, seed_id, update_idx, num_updates, metrics)
            return (train_state, env_state, obs, ep_return, rng), metrics

        (train_state, *_), metrics = jax.lax.scan(
            update, (train_state, env_state, obs, ep_return, rng), jnp.arange(num_updates)
        )
        return {"train_state": train_state, "metrics": metrics}

    return train, num_updates


def evaluate(env, params, config, key, frames):
    """Run the trained (stochastic) policy for `frames` steps in one env; return frames and final info."""
    network = ActorCritic(env.num_actions, activation=config["ACTIVATION"])

    @jax.jit
    def policy_step(key, state, obs):
        k_act, k_step = jax.random.split(key)
        pi, _ = network.apply(params, obs)
        return env.step(k_step, state, pi.sample(seed=k_act))

    obs, state = env.reset(key)
    images, info = [env.render(state)], None
    for t in range(frames):
        key, k = jax.random.split(key)
        obs, state, _, _, info = policy_step(k, state, obs)
        images.append(env.render(state))
    return images, info


def make_logger(config):
    use_wandb = config["WANDB_MODE"] != "disabled"
    if use_wandb:
        import wandb

        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            mode=config["WANDB_MODE"],
        )

    def log_fn(seed_id, update_idx, num_updates, metrics):
        metrics = {k: float(v) for k, v in metrics.items()}
        if use_wandb:
            wandb.log({f"seed{int(seed_id)}/{k}": v for k, v in metrics.items()})
        every = max(int(num_updates) // 20, 1)
        if (int(update_idx) + 1) % every == 0:
            # episode metrics are NaN in updates where no episode ended; show "-" there
            fmt = lambda v, spec: "-".rjust(len(format(0.0, spec))) if onp.isnan(v) else format(v, spec)
            log.info(
                f"[seed {int(seed_id)}] step {int(metrics['env_step']):>10,} | "
                f"return {fmt(metrics['episode_return'], '7.3f')} | "
                f"status gini {fmt(metrics['status_gini'], '.3f')} | "
                f"isolated {fmt(metrics['num_isolated'], '4.1f')} | "
                f"entropy {metrics['entropy']:.3f}"
            )

    return log_fn


@hydra.main(version_base=None, config_path="config", config_name="ippo")
def main(cfg):
    config = OmegaConf.to_container(cfg, resolve=True)
    config["ENV_KWARGS"]["grid_size"] = tuple(config["ENV_KWARGS"]["grid_size"])
    out_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Output directory: {out_dir}")

    train, num_updates = make_train(config, make_logger(config))
    log.info(f"{num_updates} updates x {config['NUM_ENVS']} envs x {config['NUM_STEPS']} steps")
    seeds = jnp.arange(config["NUM_SEEDS"])
    rngs = jax.vmap(jax.random.PRNGKey)(config["SEED"] + seeds)
    out = jax.block_until_ready(jax.jit(jax.vmap(train))(rngs, seeds))

    env = make_env(config)
    summary = {}
    for k in range(config["NUM_SEEDS"]):
        metrics = {name: onp.asarray(v[k]) for name, v in out["metrics"].items()}
        with open(os.path.join(out_dir, f"metrics_seed{k}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(metrics)
            writer.writerows(zip(*metrics.values()))

        train_state = jax.tree.map(lambda x: x[k], out["train_state"])
        save_params(train_state, os.path.join(out_dir, f"params_seed{k}.pkl"))

        # last update in which an episode ended, so the summary is never NaN
        last = onp.flatnonzero(metrics["episodes_ended"] > 0)[-1]
        summary[f"seed{k}"] = {name: float(v[last]) for name, v in metrics.items()}

        if config["GIF_FRAMES"] > 0:
            images, info = evaluate(
                env, train_state.params, config, jax.random.PRNGKey(1000 + k), config["GIF_FRAMES"]
            )
            gif = os.path.join(out_dir, f"eval_seed{k}.gif")
            frames = [Image.fromarray(img) for img in images]
            frames[0].save(gif, save_all=True, append_images=frames[1:], duration=80, loop=0)
            summary[f"seed{k}"]["eval_status_gini"] = float(info["status_gini"][0])
            summary[f"seed{k}"]["eval_num_isolated"] = float(info["num_isolated"][0])

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
