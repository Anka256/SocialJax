# Recognition

A multi-agent environment for studying the Matthew effect: whether agents that
are already well known become better known while the others become isolated.

Agents walk around an open arena (25 × 25 by default). There is no task other
than being recognised. Agents recognise whoever stands in their front cone, and
an agent is rewarded when others come to recognise it. The reward is larger
when the recogniser is better known than the recognised agent.

## Dynamics

Recognition is directed. For every ordered pair (i, j):

- `K[i, j]` is how well i recognises j now. It decays quickly.
- `M[i, j]` is how well i once knew j. It decays slowly and speeds up re-recognition.

The status of agent j is `s_j = Σ_i K[i, j]`.

| Rule | Formula |
| --- | --- |
| Visibility range of j | `R_j = min(R0 + c·ln(1 + s_j), R_max)` |
| i sees j | j is in i's 90° front cone, `d_ij ≤ R_j`, and, if `attention_budget` > 0, among that many nearest such agents (default 0 = no limit) |
| Growth (i sees j) | `K += η · g(d) · (1 + κ·M) · (1 − K)`, with `g(d) = 1 − d / (R_j + 1)` |
| Decay (i does not see j) | `K ← (1 − δ_K·(1 + L_i / C))·K`, with load `L_i = Σ_j K[i, j]`: the more agents i knows, the faster each fades (soft capacity C, default 3; 0 = off) |
| Memory | `M ← max((1 − δ_M)·M, K)` |
| Forgetting | `M < θ_forget` resets K and M to 0 |
| Reward of j | `Σ_i w_ij · ΔK[i, j]`, with `w_ij = (1 + s_i) / (1 + s_j)` |

Losing recognition (ΔK < 0) is penalised, so an agent cannot farm reward by
drifting away and coming back. One-off penalties on bonds going dormant or being
forgotten exist (`sleep_penalty`, `forget_penalty`) but are 0 by default.

## Actions and observations

- 7 actions: turn left, turn right, forward, backward, step left, step right, stay.
- Each agent sees an 11 × 11 window rotated to its heading, 9 cells ahead and 1 behind.
- Observation shape is `(N, 11, 11, 11)` with float32 channels:

| Index | Channel |
| --- | --- |
| 0 | Wall |
| 1 | Self |
| 2 | Other agent |
| 3–6 | Other agent's relative heading (one-hot) |
| 7 | Other agent's status |
| 8 | My K for that agent |
| 9 | My M for that agent |
| 10 | My own status, broadcast over the window |

## Usage

```python
import jax, socialjax

env = socialjax.make("recognition", num_agents=10, famous_agents=2)
obs, state = env.reset(jax.random.PRNGKey(0))
obs, state, rewards, dones, info = env.step(key, state, actions)
```

`info` holds per-agent `individual_rewards`, `status`, `status_gini`,
`num_isolated`, `sleep_events` and `forget_events`.

Tests: `tests/test_recognition.py`.
