"""Recognition: a multi-agent environment for studying the Matthew effect.

Agents walk around an open arena. An agent recognises the agents standing in
front of it, and is rewarded when others come to recognise it, more so when
the recogniser is better known than itself. There is no other task: the
scarce resource is attention, i.e. the cells in front of well-known agents.

Recognition is directed. For every ordered pair (i, j) the env tracks
  K[i, j] in [0, 1]  how well i recognises j right now (decays fast)
  M[i, j] in [0, 1]  how well i once knew j (decays slowly; speeds re-recognition)
and the status of agent j is s_j = sum_i K[i, j].

Per step, for every pair where i sees j:
  dK = eta * g(d_ij) * (1 + kappa * M[i, j]) * (1 - K[i, j])
otherwise K decays: K <- (1 - delta_K * (1 + L_i / C)) * K, where the load
L_i = sum_j K[i, j] is how much i currently recognises in total, so the more
agents i knows, the faster each of them fades (a soft capacity, C). Memory follows
M <- max((1 - delta_M) * M, K). A bond whose memory drops below
forget_threshold is forgotten: K and M reset to 0.

Agent j's reward is sum_i w_ij * (K_new[i, j] - K[i, j]) with
w_ij = (1 + s_i) / (1 + s_j), so being recognised by a well-known agent pays
more, and losing recognition costs. An agent's visibility range grows with its
status: R_j = min(R0 + c * ln(1 + s_j), R_max).

See the design document for the reasoning behind each rule.
"""

from enum import IntEnum
from functools import partial
from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as onp
from flax import struct
from PIL import Image, ImageDraw

from socialjax.environments import spaces
from socialjax.environments.movement import resolve_movement
from socialjax.environments.multi_agent_env import MultiAgentEnv


class Actions(IntEnum):
    turn_left = 0
    turn_right = 1
    forward = 2
    backward = 3
    step_left = 4
    step_right = 5
    stay = 6


# Heading h in {0: north, 1: east, 2: south, 3: west} -> (row, col) unit step.
HEADING_STEP = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=jnp.int32)

# Observation channels.
OBS_WALL = 0
OBS_SELF = 1
OBS_OTHER = 2
OBS_HEADING = 3  # 4 channels: other agent's heading relative to mine
OBS_STATUS = 7  # other agent's status / (N - 1)
OBS_K = 8  # how well I recognise it
OBS_M = 9  # how well I once knew it
OBS_OWN_STATUS = 10  # my own status / (N - 1), broadcast over the window
NUM_OBS_CHANNELS = 11


@struct.dataclass
class State:
    agent_locs: chex.Array  # (N, 3) int32: row, col, heading
    K: chex.Array  # (N, N) float32, current recognition, zero diagonal
    M: chex.Array  # (N, N) float32, long-term memory, zero diagonal
    t: chex.Array  # () int32, step within the episode


def gini(x: chex.Array) -> chex.Array:
    """Gini coefficient of a non-negative vector; 0 when all entries are 0."""
    n = x.shape[0]
    mean = jnp.mean(x)
    diffs = jnp.abs(x[:, None] - x[None, :]).sum()
    return jnp.where(mean > 0, diffs / (2 * n * n * jnp.maximum(mean, 1e-12)), 0.0)


def status(K: chex.Array) -> chex.Array:
    """s_j = sum_i K[i, j]: how much everyone else recognises j."""
    return K.sum(axis=0)


class Recognition(MultiAgentEnv):
    def __init__(
        self,
        num_agents: int = 10,
        grid_size: Tuple[int, int] = (25, 25),
        num_inner_steps: int = 1000,
        # view window in the agent's own frame, in cells
        view_forward: int = 9,
        view_backward: int = 1,
        view_side: int = 5,
        # visibility range R_j = min(R0 + c * ln(1 + s_j), R_max)
        base_range: float = 3.0,
        max_range: float = 8.0,
        fame_visibility: float = 1.5,
        # an agent recognises at most this many agents per step (the nearest ones); 0 = no limit
        attention_budget: int = 0,
        # recognition dynamics
        growth_rate: float = 0.05,
        memory_boost: float = 1.0,
        recognition_decay: float = 0.01,
        # soft capacity C: decay grows with the load as delta_K * (1 + load / C); 0 = off
        recognition_capacity: float = 3.0,
        memory_decay: float = 0.001,
        sleep_threshold: float = 0.2,
        forget_threshold: float = 0.05,
        # one-off penalties on bond transitions; off in the first version
        sleep_penalty: float = 0.0,
        forget_penalty: float = 0.0,
        # unequal start: agents 0..famous_agents-1 start known by everyone
        famous_agents: int = 0,
        famous_init_recognition: float = 0.5,
        shared_rewards: bool = False,
        isolation_threshold: float = 0.5,
    ):
        super().__init__(num_agents=num_agents)
        assert max_range <= view_forward, "recognition range must fit in the view window"
        assert 0 <= famous_agents <= num_agents
        assert attention_budget >= 0
        assert num_agents <= grid_size[0] * grid_size[1]

        self.agents = list(range(num_agents))
        self.grid_size = grid_size
        self.num_inner_steps = num_inner_steps
        self.view_forward = view_forward
        self.view_backward = view_backward
        self.view_side = view_side
        self.base_range = base_range
        self.max_range = max_range
        self.fame_visibility = fame_visibility
        self.attention_budget = attention_budget
        self.growth_rate = growth_rate
        self.memory_boost = memory_boost
        self.recognition_decay = recognition_decay
        self.recognition_capacity = recognition_capacity
        self.memory_decay = memory_decay
        self.sleep_threshold = sleep_threshold
        self.forget_threshold = forget_threshold
        self.sleep_penalty = sleep_penalty
        self.forget_penalty = forget_penalty
        self.famous_agents = famous_agents
        self.famous_init_recognition = famous_init_recognition
        self.shared_rewards = shared_rewards
        self.isolation_threshold = isolation_threshold

        self.view_height = view_forward + view_backward + 1
        self.view_width = 2 * view_side + 1
        # (forward, lateral) offset of every window cell; row 0 is the farthest ahead.
        fwd = jnp.arange(view_forward, -view_backward - 1, -1)
        lat = jnp.arange(-view_side, view_side + 1)
        self._window_fwd, self._window_lat = jnp.meshgrid(fwd, lat, indexing="ij")

    # ------------------------------------------------------------------ spaces

    def action_space(self, agent_id=None) -> spaces.Discrete:
        return spaces.Discrete(len(Actions))

    def observation_space(self, agent_id=None):
        """(Box, shape), matching the other SocialJax envs' convention."""
        shape = (self.view_height, self.view_width, NUM_OBS_CHANNELS)
        return spaces.Box(low=0.0, high=1.0, shape=shape, dtype=jnp.float32), shape

    @property
    def name(self) -> str:
        return "Recognition"

    @property
    def num_actions(self) -> int:
        return len(Actions)

    # ------------------------------------------------------------------- reset

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        n = self.num_agents
        rows, cols = self.grid_size
        k_pos, k_dir = jax.random.split(key)
        cells = jax.random.permutation(k_pos, rows * cols)[:n]
        headings = jax.random.randint(k_dir, (n,), 0, 4)
        agent_locs = jnp.stack([cells // cols, cells % cols, headings], axis=-1).astype(jnp.int32)

        famous = jnp.arange(n) < self.famous_agents
        K = jnp.where(famous[None, :], self.famous_init_recognition, 0.0)
        K = K * (1.0 - jnp.eye(n))
        state = State(agent_locs=agent_locs, K=K, M=K, t=jnp.int32(0))
        return self.get_obs(state), state

    # -------------------------------------------------------------------- step

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, key: chex.PRNGKey, state: State, actions, timestep: int = 0
    ) -> Tuple[chex.Array, State, chex.Array, Dict[str, chex.Array], Dict[str, chex.Array]]:
        actions = jnp.asarray(actions, dtype=jnp.int32).reshape(self.num_agents)
        agent_locs = self._move(key, state.agent_locs, actions)

        s_old = status(state.K)
        visibility = self._visibility(s_old)
        seen, dist = self._sight(agent_locs, visibility)
        K, M, slept, forgot = self._update_recognition(state.K, state.M, seen, dist, visibility)

        # Reward to the recognised agent j: sum_i w_ij * dK[i, j], weights from pre-step status.
        w = (1.0 + s_old[:, None]) / (1.0 + s_old[None, :])
        individual = (w * (K - state.K)).sum(axis=0)
        individual -= self.sleep_penalty * slept.sum(axis=0) + self.forget_penalty * forgot.sum(axis=0)
        rewards = (
            jnp.full((self.num_agents,), individual.sum()) if self.shared_rewards else individual
        )

        t = state.t + 1
        new_state = State(agent_locs=agent_locs, K=K, M=M, t=t)
        done = t >= self.num_inner_steps
        dones = {str(a): done for a in self.agents}
        dones["__all__"] = done

        s_new = status(K)
        n = self.num_agents
        info = {
            "individual_rewards": individual,
            "status": s_new,
            "status_gini": jnp.full((n,), gini(s_new)),
            "num_isolated": jnp.full((n,), (s_new < self.isolation_threshold).sum()),
            "sleep_events": slept.sum(axis=0),
            "forget_events": forgot.sum(axis=0),
        }
        return self.get_obs(new_state), new_state, rewards, dones, info

    def _move(self, key: chex.PRNGKey, locs: chex.Array, actions: chex.Array) -> chex.Array:
        heading = locs[:, 2]
        turn = jnp.where(actions == Actions.turn_left, -1, 0) + jnp.where(actions == Actions.turn_right, 1, 0)
        new_heading = (heading + turn) % 4

        # Movement direction relative to the current heading: forward, back, left, right.
        rel = jnp.select(
            [actions == Actions.forward, actions == Actions.backward,
             actions == Actions.step_left, actions == Actions.step_right],
            [0, 2, 3, 1],
            default=-1,
        )
        step = jnp.where((rel >= 0)[:, None], HEADING_STEP[(heading + rel) % 4], 0)
        rows, cols = self.grid_size
        rc = jnp.clip(locs[:, :2] + step, 0, jnp.array([rows - 1, cols - 1]))
        proposed = jnp.concatenate([rc, new_heading[:, None]], axis=-1).astype(jnp.int32)
        return resolve_movement(key, locs, proposed)

    def _relative(self, locs: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """forward[i, j], lateral[i, j]: where j stands in i's own frame."""
        delta = locs[None, :, :2] - locs[:, None, :2]  # (N, N, 2), j relative to i
        fwd_dir = HEADING_STEP[locs[:, 2]]  # (N, 2)
        right_dir = HEADING_STEP[(locs[:, 2] + 1) % 4]
        forward = (delta * fwd_dir[:, None, :]).sum(-1)
        lateral = (delta * right_dir[:, None, :]).sum(-1)
        return forward, lateral

    def _visibility(self, s: chex.Array) -> chex.Array:
        """R_j = min(R0 + c * ln(1 + s_j), R_max): range at which agent j can be recognised."""
        return jnp.minimum(self.base_range + self.fame_visibility * jnp.log1p(s), self.max_range)

    def _sight(self, locs: chex.Array, visibility: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """seen[i, j]: j stands in i's 90-degree front cone within j's visibility range,
        and is among the attention_budget nearest such agents (ties go to the lower index)."""
        forward, lateral = self._relative(locs)
        dist = jnp.sqrt((forward ** 2 + lateral ** 2).astype(jnp.float32))
        in_cone = (forward >= 1) & (jnp.abs(lateral) <= forward)
        seen = in_cone & (dist <= visibility[None, :])
        if self.attention_budget > 0:
            n = self.num_agents
            # exact order: integer squared distance first, agent index breaks ties
            dist_sq = forward ** 2 + lateral ** 2
            score = jnp.where(seen, dist_sq * n + jnp.arange(n)[None, :], jnp.iinfo(jnp.int32).max)
            rank = jnp.argsort(jnp.argsort(score, axis=1), axis=1)
            seen = seen & (rank < self.attention_budget)
        return seen, dist

    def _update_recognition(self, K, M, seen, dist, visibility):
        closeness = jnp.clip(1.0 - dist / (visibility[None, :] + 1.0), 0.0, 1.0)  # g(d)
        gain = self.growth_rate * closeness * (1.0 + self.memory_boost * M) * (1.0 - K)
        decay = self.recognition_decay
        if self.recognition_capacity > 0:
            load = K.sum(axis=1, keepdims=True)  # (N, 1): how much each observer recognises
            decay = decay * (1.0 + load / self.recognition_capacity)
        K_new = jnp.where(seen, K + gain, jnp.clip(1.0 - decay, 0.0, 1.0) * K)
        M_new = jnp.maximum((1.0 - self.memory_decay) * M, K_new)

        slept = (K >= self.sleep_threshold) & (K_new < self.sleep_threshold)
        forgot = (M >= self.forget_threshold) & (M_new < self.forget_threshold)
        K_new = jnp.where(forgot, 0.0, K_new)
        M_new = jnp.where(forgot, 0.0, M_new)
        return K_new, M_new, slept, forgot

    # ------------------------------------------------------------- observation

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: State) -> chex.Array:
        """(N, view_height, view_width, NUM_OBS_CHANNELS) egocentric views, forward = up."""
        n = self.num_agents
        locs = state.agent_locs
        s = status(state.K)
        s_norm = s / max(n - 1, 1)
        forward, lateral = self._relative(locs)
        rel_heading = (locs[None, :, 2] - locs[:, None, 2]) % 4  # (N, N)

        def one_view(i):
            obs = jnp.zeros((self.view_height, self.view_width, NUM_OBS_CHANNELS), jnp.float32)

            # walls: window cells outside the arena
            fwd_dir = HEADING_STEP[locs[i, 2]]
            right_dir = HEADING_STEP[(locs[i, 2] + 1) % 4]
            world = (
                locs[i, :2]
                + self._window_fwd[..., None] * fwd_dir
                + self._window_lat[..., None] * right_dir
            )
            rows, cols = self.grid_size
            outside = (world[..., 0] < 0) | (world[..., 0] >= rows) | (world[..., 1] < 0) | (world[..., 1] >= cols)
            obs = obs.at[..., OBS_WALL].set(outside.astype(jnp.float32))
            obs = obs.at[..., OBS_OWN_STATUS].set(s_norm[i])

            # every agent, including i itself, placed at its window cell (dropped if outside)
            r = self.view_forward - forward[i]
            c = lateral[i] + self.view_side
            is_self = jnp.arange(n) == i
            feats = jnp.zeros((n, NUM_OBS_CHANNELS), jnp.float32)
            feats = feats.at[:, OBS_SELF].set(is_self)
            feats = feats.at[:, OBS_OTHER].set(~is_self)
            heading_1h = jax.nn.one_hot(rel_heading[i], 4) * (~is_self)[:, None]
            feats = feats.at[:, OBS_HEADING:OBS_HEADING + 4].set(heading_1h)
            feats = feats.at[:, OBS_STATUS].set(jnp.where(is_self, 0.0, s_norm))
            feats = feats.at[:, OBS_K].set(state.K[i])
            feats = feats.at[:, OBS_M].set(state.M[i])
            feats = feats.at[:, OBS_OWN_STATUS].set(s_norm[i])
            inside = (r >= 0) & (r < self.view_height) & (c >= 0) & (c < self.view_width)
            r = jnp.where(inside, r, self.view_height)  # out-of-range index -> dropped
            return obs.at[r, c].set(feats, mode="drop")

        return jax.vmap(one_view)(jnp.arange(n))

    # --------------------------------------------------------------- rendering

    def render(self, state: State, cell: int = 24) -> onp.ndarray:
        """RGB frame: agents coloured by status (gold = best known now), status printed
        beside each agent, a dot for heading, and a blue line for every active bond."""
        rows, cols = self.grid_size
        img = Image.new("RGB", (cols * cell, rows * cell), (245, 243, 238))
        draw = ImageDraw.Draw(img)
        locs = onp.asarray(state.agent_locs)
        K = onp.asarray(state.K)
        s = K.sum(axis=0)
        fame = s / max(s.max(), 1.0)  # relative to the best-known agent in this frame
        centre = lambda a: ((locs[a, 1] + 0.5) * cell, (locs[a, 0] + 0.5) * cell)

        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j and K[i, j] >= self.sleep_threshold:
                    shade = int(200 * (1 - K[i, j]))
                    draw.line([centre(i), centre(j)], fill=(shade, shade, 255), width=2)
        r = cell * 0.4
        for a in range(self.num_agents):
            x, y = centre(a)
            colour = tuple(int(g * f + 150 * (1 - f)) for g, f in zip((230, 170, 20), (fame[a],) * 3))
            draw.ellipse([x - r, y - r, x + r, y + r], fill=colour, outline=(40, 40, 40))
            dy, dx = onp.asarray(HEADING_STEP)[locs[a, 2]]
            hx, hy = x + dx * r * 0.6, y + dy * r * 0.6
            draw.ellipse([hx - 3, hy - 3, hx + 3, hy + 3], fill=(20, 20, 20))
            draw.text((x + r + 2, y - r - 2), f"{s[a]:.1f}", fill=(60, 60, 60))
        return onp.asarray(img)
