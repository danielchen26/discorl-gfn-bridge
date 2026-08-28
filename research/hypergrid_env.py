#!/usr/bin/env python3
"""A hypergrid DAG as a DiscoRL jittable environment.

The dossier's whole argument turns on multi-path structure: several action
sequences build the same object, which is the one place GFlowNets and MaxEnt RL
provably disagree.  Catch and Atari have no such structure -- their state graphs
are trees over the agent's own history -- so to probe DiscoRL against a flow
condition at all we need an environment whose state graph is a genuine DAG with
analytically known path multiplicities.

The grid is the standard hypergrid: from (i, j) you may step RIGHT, UP, or STOP.
Stopping emits reward R(i, j); every other transition emits zero.  Every
trajectory is a monotone lattice path followed by a stop, so the number of
distinct action sequences reaching (i, j) is exactly C(i+j, i), and the parent
set of a state is at most {left, down} -- both facts the probe needs and neither
of which the agent is ever told.

Interface follows disco_rl.environments.jittable_envs._SingleStreamCatch:
initial_state / episode_reset / step / is_terminal / reward / render.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from ml_collections import config_dict

from disco_rl.environments.wrappers import batched_jittable_env

RIGHT, UP, STOP = 0, 1, 2

# Same constants as research/cumulants.py, so the exact DP oracle in that file
# describes this environment and not merely something similar.
R0, R1, R2 = 1e-2, 0.5, 2.0


def grid_reward(i, j, h: int):
  """Standard hypergrid reward (Bengio et al. 2021), strictly positive."""
  ax = jnp.abs(i / (h - 1) - 0.5)
  ay = jnp.abs(j / (h - 1) - 0.5)
  r = jnp.float32(R0)
  r = r + jnp.where((ax > 0.25) & (ay > 0.25), R1, 0.0)
  r = r + jnp.where((ax > 0.3) & (ax < 0.4) & (ay > 0.3) & (ay < 0.4), R2, 0.0)
  return r.astype(jnp.float32)


class _SingleStreamHypergrid:
  """Hypergrid with lifetime reset.

  State is (i, j, stopped).  `stopped` is carried in the state rather than
  inferred, because the terminal reward has to survive into the timestep where
  `is_terminal` is read.
  """

  def __init__(self, height: int = 8):
    self._h = height

  @property
  def num_actions(self) -> int:
    return 3

  def initial_state(self, rng):
    del rng
    return jnp.array((0, 0, 0), dtype=jnp.int32)

  def episode_reset(self, rng, state):
    del state
    return self.initial_state(rng)

  def step(self, rng, state, action):
    del rng
    i, j = state[0], state[1]
    # At the far edges RIGHT and UP are unavailable; folding them into STOP
    # keeps the action space rectangular without inventing absorbing states,
    # and matches how the exact oracle enumerates trajectories.
    can_right = i + 1 < self._h
    can_up = j + 1 < self._h
    move_right = (action == RIGHT) & can_right
    move_up = (action == UP) & can_up
    stopped = jnp.where(move_right | move_up, 0, 1)
    return jnp.array(
        [i + move_right.astype(jnp.int32), j + move_up.astype(jnp.int32), stopped],
        dtype=jnp.int32,
    )

  def is_terminal(self, state):
    return state[2] == 1

  def reward(self, state):
    return jnp.where(state[2] == 1, grid_reward(state[0], state[1], self._h), 0.0)

  def render(self, state):
    """One-hot board. Deliberately gives away neither path count nor parents."""
    board = jnp.zeros((self._h, self._h), dtype=jnp.float32)
    board = board.at[state[0], state[1]].set(1.0)
    return board.reshape((self._h, self._h, 1))


class HypergridJittableEnvironment(batched_jittable_env.BatchedJittableEnvironment):
  """Batched hypergrid."""

  def __init__(self, batch_size: int, env_settings: config_dict.ConfigDict):
    super().__init__(
        env_class=_SingleStreamHypergrid,
        batch_size=batch_size,
        env_settings=env_settings,
    )

def get_config_hypergrid() -> config_dict.ConfigDict:
  return config_dict.ConfigDict(dict(height=8))


def exact_terminal_distribution(policy_fn, height: int = 8):
  """Exact p(x) under a given policy, by forward DP over the DAG.

  Sampling would need enormous batches to resolve the low-reward corners, and
  the whole point of the measurement is a log-scale regression, so the estimate
  has to be exact rather than merely unbiased.

  `policy_fn(i, j) -> (p_right, p_up, p_stop)`.
  """
  import numpy as np

  reach = np.zeros((height, height))
  reach[0, 0] = 1.0
  term = np.zeros((height, height))
  for s in range(2 * height - 1):
    for i in range(height):
      j = s - i
      if not (0 <= j < height):
        continue
      pr, pu, ps = policy_fn(i, j)
      can_r = i + 1 < height
      can_u = j + 1 < height
      # Unavailable moves fold into STOP, exactly as `step` does.
      ps = ps + (0.0 if can_r else pr) + (0.0 if can_u else pu)
      term[i, j] = reach[i, j] * ps
      if can_r:
        reach[i + 1, j] += reach[i, j] * pr
      if can_u:
        reach[i, j + 1] += reach[i, j] * pu
  return term
