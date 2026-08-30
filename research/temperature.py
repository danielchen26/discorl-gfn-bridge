#!/usr/bin/env python3
"""What effective entropy temperature does a discovered RL rule implement?

research/tiapkin.py verified, to machine precision, that GFlowNet training is
soft RL on a specific MDP -- rewards log P_B on interior transitions and log R
at termination -- provided the entropy coefficient is exactly one. It also
priced the coefficient: ten percent off costs a KL of 0.04 to the target
distribution, and a factor of two costs 1.13. Remark 3 of that paper says any
lambda != 1 biases the sampler, and Remark 4 reports exactly that failure for
SAC with an adaptive coefficient.

A discovered update rule has no entropy coefficient at all. Its hyper-parameters
are pi_cost, y_cost, z_cost, value_cost, aux_policy_cost, target_params_coeff,
value_fn_td_lambda and the discount -- we checked. Whatever regularisation it
implements is emergent from the meta-learned rule, so it cannot be read off and
has to be measured. In this MDP it can be: run the rule, take the terminal
distribution it converges to, and fit the coefficient of the exact soft family
computed by value iteration.

lambda_eff = 1 would mean the discovered rule is a GFlowNet on this problem.
Anything else says how far off it is and in which direction.

WHAT THIS SCRIPT DEFENDS AGAINST, each item having actually gone wrong here.

*   A statistic with no leverage. The canonical hypergrid reward is nearly
    binary, and every distribution spread over its plateau returns the same
    mean log reward, so a scalar readout could not separate arms that were
    plainly different. The reward here is smooth and multimodal with a log
    range of about five, and the fit uses the whole distribution rather than
    one moment.

*   A one-sided statistic. KL to the target is symmetric about lambda = 1 and
    gives only a magnitude. Fitting the soft family gives a signed answer and,
    as a bonus, a residual that says whether any coefficient describes the
    agent at all.

*   An origin trap. With rewards log P_B every step of travel costs about 0.69,
    so on the canonical grid the cold optimum was to stop immediately and every
    arm collapsed onto it, identically. Here the cold optimum is 2.285 against
    -2.000 for stopping at the source, so a cold agent has to go somewhere.

*   A learning check that had the sign backwards. Judging "did it learn" by KL
    to the target rejects exactly the runs that learned most decisively, since
    a cold agent raises its own objective while moving away from the target.
    The check is now on the objective each arm actually optimises.

*   Terminal states counted wrong. An episode ends on STOP or on a move into a
    wall. Counting only STOP left the empirical and DP distributions differing
    by 0.5 for any agent that learned to push against an edge.

*   An unvalidated instrument. Before any learned agent is read, the exact
    soft-optimal policy at known coefficients is pushed through the identical
    pipeline -- sample, bucket by state, forward DP, fit -- and has to come
    back with the coefficient it was given.

Usage:
  .venv/bin/python research/temperature.py --json research/temperature.json
"""

from __future__ import annotations

import argparse
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict

from disco_rl import agent as agent_lib
from disco_rl import types
from disco_rl.environments.wrappers import batched_jittable_env

import disco_probe

RIGHT, UP, STOP = 0, 1, 2
HEIGHT = 6

# Three modes of different heights on the unit square. No subsidy for travel:
# a cold sampler concentrating near the source would be the correct reading of
# a cold rule, and the point is to have a landscape on which the readout can
# tell that apart from anything else.
_MODES = ((5.5, 0.15, 0.85), (4.5, 0.85, 0.15), (6.0, 0.85, 0.85))
_WIDTH = 0.05
_OFFSET = 2.0


def log_reward_np(i: int, j: int, h: int = HEIGHT) -> float:
    x, y = i / (h - 1), j / (h - 1)
    return sum(
        a * math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / _WIDTH) for a, cx, cy in _MODES
    ) - _OFFSET


def n_parents_np(i: int, j: int) -> int:
    return (1 if i > 0 else 0) + (1 if j > 0 else 0)


# --------------------------------------------------------------------------
# The MDP of Theorem 1, as a disco_rl environment.
# --------------------------------------------------------------------------


def geometric_potential(h: int) -> np.ndarray:
    """Soft value at lambda = 1 of this DAG with the reward flattened away.

    This is the part of the value that comes from the shape of the graph and
    the backward policy alone, with no reference to R. Used as a shaping
    potential it cancels exactly the term that makes travel look bad -- every
    step into a two-parent state costs about 0.69 and pays nothing until the
    end -- while leaving the soft-optimal policy at every temperature
    untouched, since for r' = r + Phi(s') - Phi(s),

        lambda log sum_a exp((r' + V'(s'))/lambda) = V(s) - Phi(s),
        pi' ~ exp((r + V(s'))/lambda) = pi.

    Using log F itself would be circular. This potential never looks at R.
    """
    V = np.zeros(h * h)
    for s in reversed(topo(h)):
        i, j = s // h, s % h
        qs = [0.0]
        for di, dj in ((1, 0), (0, 1)):
            ci, cj = i + di, j + dj
            if ci < h and cj < h:
                qs.append(-math.log(n_parents_np(ci, cj)) + V[ci * h + cj])
        m = max(qs)
        V[s] = m + math.log(sum(math.exp(q - m) for q in qs))
    return V


class _SingleStreamTiapkin:
    """Grid DAG carrying the rewards Theorem 1 prescribes.

    State is (i, j, stopped, previous i, previous j). Arriving at c pays
    log P_B(s|c), which for a backward policy uniform over parents is
    -log |Pa(c)|; stopping at s pays log R(s).

    With pb_rewards off the interior payments vanish and this becomes the
    ordinary hypergrid, reward at the end and nothing on the way. That variant
    is not the reduction and no temperature can be read from it; it exists to
    answer a single question about the wiring, namely whether an agent that
    fails here fails because of the MDP or because of us.

    With shaping on, every transition additionally pays Phi(s') - Phi(s) for
    the geometric potential above, and termination pays -Phi(s) since the
    absorbing state has none. The sum telescopes to a constant, so returns
    shift by -Phi(source) and the soft-optimal policy is unchanged at every
    temperature -- which the script checks rather than assumes. The previous
    position rides in the state only so that this transition reward can be
    computed behind disco_rl's reward(state) interface.
    """

    def __init__(self, height: int = HEIGHT, pb_rewards: bool = True,
                 shaped: bool = False):
        self._h = height
        self._pb = pb_rewards
        self._phi = jnp.asarray(geometric_potential(height)) if shaped else None

    @property
    def num_actions(self) -> int:
        return 3

    def initial_state(self, rng):
        del rng
        return jnp.array((0, 0, 0, 0, 0), dtype=jnp.int32)

    def episode_reset(self, rng, state):
        del state
        return self.initial_state(rng)

    def step(self, rng, state, action):
        del rng
        i, j = state[0], state[1]
        mv_r = (action == RIGHT) & (i + 1 < self._h)
        mv_u = (action == UP) & (j + 1 < self._h)
        return jnp.array(
            [i + mv_r.astype(jnp.int32), j + mv_u.astype(jnp.int32),
             jnp.where(mv_r | mv_u, 0, 1), i, j],
            dtype=jnp.int32,
        )

    def is_terminal(self, state):
        return state[2] == 1

    def reward(self, state):
        i, j, stopped, pi_, pj_ = state[0], state[1], state[2], state[3], state[4]
        h = self._h
        x, y = i / (h - 1), j / (h - 1)
        log_r = -_OFFSET
        for a, cx, cy in _MODES:
            log_r = log_r + a * jnp.exp(-((x - cx) ** 2 + (y - cy) ** 2) / _WIDTH)
        n_par = (i > 0).astype(jnp.float32) + (j > 0).astype(jnp.float32)
        move_r = -jnp.log(jnp.maximum(n_par, 1.0))  # zero at the source
        if not self._pb:
            move_r = jnp.zeros_like(move_r)
        if self._phi is not None:
            here = self._phi[i * h + j]
            move_r = move_r + here - self._phi[pi_ * h + pj_]
            log_r = log_r - here
        return jnp.where(stopped == 1, log_r, move_r).astype(jnp.float32)

    def render(self, state):
        board = jnp.zeros((self._h, self._h), dtype=jnp.float32)
        return board.at[state[0], state[1]].set(1.0).reshape((self._h, self._h, 1))


class TiapkinEnvironment(batched_jittable_env.BatchedJittableEnvironment):
    def __init__(self, batch_size: int, env_settings: config_dict.ConfigDict):
        super().__init__(
            env_class=_SingleStreamTiapkin,
            batch_size=batch_size,
            env_settings=env_settings,
        )


def get_config(height: int = HEIGHT, pb_rewards: bool = True,
               shaped: bool = False) -> config_dict.ConfigDict:
    return config_dict.ConfigDict(
        dict(height=height, pb_rewards=pb_rewards, shaped=shaped)
    )


# --------------------------------------------------------------------------
# The exact soft family, by value iteration
# --------------------------------------------------------------------------


def topo(h: int):
    return sorted(range(h * h), key=lambda k: (k // h) + (k % h))


def soft_policy(h: int, lam: float, phi: np.ndarray | None = None) -> np.ndarray:
    """Soft-optimal policy at coefficient lam, over (RIGHT, UP, STOP).

    Passing a potential applies the shaped rewards. The result must be
    identical, and the script checks that rather than trusting the algebra.
    """
    n = h * h
    V = np.zeros(n)
    pol = np.zeros((n, 3))
    for s in reversed(topo(h)):
        i, j = s // h, s % h
        can_r, can_u = i + 1 < h, j + 1 < h
        here = 0.0 if phi is None else phi[s]
        q = [-np.inf, -np.inf, log_reward_np(i, j, h) - here]
        if can_r:
            c = (i + 1) * h + j
            q[RIGHT] = -math.log(n_parents_np(i + 1, j)) + V[c] + (
                0.0 if phi is None else phi[c] - here)
        if can_u:
            c = i * h + (j + 1)
            q[UP] = -math.log(n_parents_np(i, j + 1)) + V[c] + (
                0.0 if phi is None else phi[c] - here)
        qa = np.array(q) / lam
        m = qa.max()
        V[s] = lam * (m + math.log(np.exp(qa - m).sum()))
        p = np.exp(qa - m)
        pol[s] = p / p.sum()
    return pol


def forward_dp(pi: np.ndarray, h: int, pb: bool = True):
    """Terminal distribution, expected undiscounted return and expected
    trajectory entropy of a state-conditional policy.

    Unavailable moves are folded into stopping, exactly as the environment
    does; the entropy is over the agent's own three-way policy, since that is
    what -log pi(a|s) reads off the logits. With pb off the interior payments
    are dropped, matching the control variant of the environment.
    """
    n = h * h
    reach = np.zeros(n)
    reach[0] = 1.0
    term = np.zeros(n)
    for s in topo(h):
        i, j = s // h, s % h
        pr, pu, ps = pi[s]
        can_r, can_u = i + 1 < h, j + 1 < h
        term[s] = reach[s] * (ps + (0.0 if can_r else pr) + (0.0 if can_u else pu))
        if can_r:
            reach[(i + 1) * h + j] += reach[s] * pr
        if can_u:
            reach[i * h + (j + 1)] += reach[s] * pu

    G = np.zeros(n)
    H = np.zeros(n)
    for s in reversed(topo(h)):
        i, j = s // h, s % h
        pr, pu, ps = pi[s]
        can_r, can_u = i + 1 < h, j + 1 < h
        stop_p = ps + (0.0 if can_r else pr) + (0.0 if can_u else pu)
        g = stop_p * log_reward_np(i, j, h)
        e = float(-np.sum(pi[s] * np.log(np.maximum(pi[s], 1e-300))))
        if can_r:
            c = (i + 1) * h + j
            g += pr * ((-math.log(n_parents_np(i + 1, j)) if pb else 0.0) + G[c])
            e += pr * H[c]
        if can_u:
            c = i * h + (j + 1)
            g += pu * ((-math.log(n_parents_np(i, j + 1)) if pb else 0.0) + G[c])
            e += pu * H[c]
        G[s], H[s] = g, e
    return term, float(G[0]), float(H[0])


def best_return(h: int, pb: bool = True) -> float:
    """The best undiscounted return any policy can achieve.

    With the log P_B payments in place this is not the largest reward on the
    grid: travel costs about 0.69 for every step into a state with two parents
    and nothing along an axis, so the best route runs up an edge and turns once
    at the end. Comparing an agent's return against this number is the only way
    to tell a rule that implements a cold sampler from a rule that gave up.
    """
    V = np.full(h * h, -np.inf)
    for s in reversed(topo(h)):
        i, j = s // h, s % h
        v = log_reward_np(i, j, h)
        for di, dj in ((1, 0), (0, 1)):
            ci, cj = i + di, j + dj
            if ci < h and cj < h:
                step = -math.log(n_parents_np(ci, cj)) if pb else 0.0
                v = max(v, step + V[ci * h + cj])
        V[s] = v
    return float(V[0])


class SoftFamily:
    """The one-parameter family the measurement fits against."""

    def __init__(self, h: int, lams: np.ndarray):
        self.h = h
        self.lams = lams
        self.logR = np.array([log_reward_np(k // h, k % h, h) for k in range(h * h)])
        tgt = np.exp(self.logR)
        self.target = tgt / tgt.sum()
        self.table = np.stack([forward_dp(soft_policy(h, float(l)), h)[0] for l in lams])
        # A floor keeps the divergence finite where a cold member has no mass,
        # without which the fit could not even be evaluated at small lambda.
        self.safe = self.table + 1e-12
        self.safe /= self.safe.sum(1, keepdims=True)

    def kl_to_target(self, p: np.ndarray) -> float:
        m = p > 1e-15
        return float(np.sum(p[m] * (np.log(p[m]) - np.log(self.target[m]))))

    def _member(self, lam: float) -> np.ndarray:
        q = forward_dp(soft_policy(self.h, lam), self.h)[0] + 1e-12
        return q / q.sum()

    @staticmethod
    def _div(p: np.ndarray, q: np.ndarray, m: np.ndarray) -> float:
        return float(np.sum(p[m] * (np.log(p[m]) - np.log(q[m]))))

    def fit(self, p: np.ndarray) -> tuple[float, float, bool]:
        """Coefficient of the closest member of the soft family, and the
        residual divergence that no coefficient can remove.

        A coarse grid brackets the minimum and golden section on log lambda
        refines it against exactly recomputed members, so neither the answer
        nor the residual carries the grid's discretisation. That mattered: a
        parabolic refinement on the tabulated grid biased lambda = 1 to 1.014
        and left a residual of 3e-4 that was spacing, not misfit.

        A large residual means the agent is not in this family at all and no
        coefficient describes it, which is a different and more interesting
        failure than landing at the wrong one.
        """
        m = p > 1e-15
        kls = np.array([self._div(p, q, m) for q in self.safe])
        k = int(np.argmin(kls))
        edge = k == 0 or k == len(self.lams) - 1
        lo = math.log(self.lams[max(k - 1, 0)])
        hi = math.log(self.lams[min(k + 1, len(self.lams) - 1)])
        if edge:
            return float(self.lams[k]), float(kls[k]), False

        inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)
        fc = self._div(p, self._member(math.exp(c)), m)
        fd = self._div(p, self._member(math.exp(d)), m)
        for _ in range(40):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - inv_phi * (b - a)
                fc = self._div(p, self._member(math.exp(c)), m)
            else:
                a, c, fc = c, d, fd
                d = a + inv_phi * (b - a)
                fd = self._div(p, self._member(math.exp(d)), m)
            if b - a < 1e-9:
                break
        lam = math.exp(0.5 * (a + b))
        return lam, self._div(p, self._member(lam), m), True


# --------------------------------------------------------------------------
# Agents
def build_agent(rule: str, batch: int, height: int, entropy_cost: float,
                lr: float | None, pb: bool, shaped: bool):
    """An agent for this MDP.

    discount_factor is forced to one because Theorem 1 is undiscounted; the
    default 0.997 costs several percent over an episode, larger than the
    effects being read. The actor-critic default carries entropy_cost = 0.2,
    which we set explicitly so it is a dial rather than a hidden constant.

    Each rule keeps its own learning rate unless one is given. Overriding both
    to 1e-3 made every arm commit to a policy within a few hundred updates and
    stop at the source, four units of return below what this MDP allows, so
    the temperatures being read were those of agents that had not optimised.
    """
    env = TiapkinEnvironment(batch_size=batch,
                             env_settings=get_config(height, pb, shaped))
    settings = (
        agent_lib.get_settings_disco() if rule == "disco"
        else agent_lib.get_settings_actor_critic()
    )
    settings.hyper_params.discount_factor = 1.0
    if rule != "disco":
        settings.hyper_params.entropy_cost = entropy_cost
    settings.net_settings.name = "mlp"
    settings.net_settings.net_args = dict(
        dense=(128, 128), model_arch_name="lstm", head_w_init_std=1e-2,
        model_kwargs=dict(head_mlp_hiddens=(64,), lstm_size=64),
    )
    if lr is not None:
        settings.learning_rate = lr
    ag = agent_lib.Agent(
        agent_settings=settings,
        single_observation_spec=env.single_observation_spec(),
        single_action_spec=env.single_action_spec(),
        batch_axis_name=None,
    )
    return env, ag


def bucket(idx_acts, h: int):
    """Empirical state-conditional policy and terminal distribution.

    An episode ends on STOP or on a move into a wall, since the environment
    converts an unavailable move into termination. Counting only STOP misses
    every agent that learns to push against an edge, which drove the
    consistency check to an error of 0.5 before it was fixed.

    On a DAG whose trajectories all begin at the source, the forward DP under
    this bucketed policy reproduces the empirical visit frequencies exactly --
    reach(c) = sum_p [n(p)/N][n(p->c)/n(p)] = n(c)/N -- so the consistency
    check carries no sampling noise and measures dependence on history alone.
    """
    n = h * h
    counts = np.zeros((n, 3))
    emp = np.zeros(n)
    for idx, acts in idx_acts:
        ii, jj = idx // h, idx % h
        ends = ((acts == STOP) | ((acts == RIGHT) & (ii == h - 1))
                | ((acts == UP) & (jj == h - 1)))
        np.add.at(counts, (idx.ravel(), acts.ravel()), 1.0)
        np.add.at(emp, idx[ends], 1.0)
    seen = counts.sum(1) > 0
    pi = np.where(seen[:, None], counts / np.maximum(counts.sum(1, keepdims=True), 1), 1 / 3)
    return pi, emp / max(emp.sum(), 1.0), int(seen.sum())


def sample_exact(pi: np.ndarray, h: int, n_traj: int, rng: np.random.Generator):
    """Roll the given state-conditional policy, in the shape bucket() expects."""
    idx, acts = [], []
    for _ in range(n_traj):
        i = j = 0
        while True:
            s = i * h + j
            a = int(rng.choice(3, p=pi[s]))
            idx.append(s)
            acts.append(a)
            if a == STOP or (a == RIGHT and i == h - 1) or (a == UP and j == h - 1):
                break
            i, j = (i + 1, j) if a == RIGHT else (i, j + 1)
    return [(np.array(idx), np.array(acts))]


def run_arm(rule, alpha, entropy_cost, h, batch, rollout_len, steps, probe_every,
            collect, seed, meta_params, fam, lr, pb=True, shaped=False):
    """Train one arm, probing the terminal distribution as it goes.

    The probe at step zero is the untrained baseline. Without it there is no
    telling a rule that implements a hot temperature from a rule that has not
    learned, since both give a broad terminal distribution.
    """
    env, ag = build_agent(rule, batch, h, entropy_cost, lr, pb, shaped)
    mp = meta_params if rule == "disco" else ag.update_rule.init_params(jax.random.PRNGKey(0))[0]

    rng = jax.random.PRNGKey(seed)
    learner_state = ag.initial_learner_state(rng)
    actor_state = ag.initial_actor_state(rng)
    env_state, ts = env.reset(rng)

    @jax.jit
    def unroll(p, actor_state, ts, env_state, key):
        def _step(carry, k):
            env_state, ts, actor_state = carry
            at, actor_state = ag.actor_step(p, k, ts, actor_state)
            env_state, ts = env.step(env_state, at.actions)
            return (env_state, ts, actor_state), at

        (env_state, ts, actor_state), stacked = jax.lax.scan(
            _step, (env_state, ts, actor_state), jax.random.split(key, rollout_len)
        )
        return types.ActorRollout.from_timestep(stacked), actor_state, ts, env_state

    def augment(roll):
        if alpha == 0.0:
            return roll
        logp = jax.nn.log_softmax(roll.agent_outs["logits"], axis=-1)
        taken = jnp.take_along_axis(logp, roll.actions[..., None], axis=-1)[..., 0]
        return roll.replace(rewards=roll.rewards - alpha * taken)

    learner = jax.jit(ag.learner_step, static_argnums=(5,))
    curve = []

    def probe(step, actor_state, ts, env_state, rng):
        batches = []
        for _ in range(collect):
            rng, k = jax.random.split(rng)
            roll, actor_state, ts, env_state = unroll(
                learner_state.params, actor_state, ts, env_state, k
            )
            obs = np.asarray(roll.observations)
            batches.append((obs.reshape(obs.shape[0], obs.shape[1], -1).argmax(-1),
                            np.asarray(roll.actions)))
        pi, emp, nseen = bucket(batches, h)
        term, ret, htraj = forward_dp(pi, h, pb)
        lam, res, inside = fam.fit(term)
        curve.append({
            "step": step, "bucket_err": float(np.max(np.abs(term - emp))),
            "states_seen": nseen, "return": ret, "h_traj": htraj,
            "soft": ret + alpha * htraj + entropy_cost * htraj,
            "kl": fam.kl_to_target(term), "lambda_eff": lam,
            "residual": res, "inside_sweep": inside,
        })
        return actor_state, ts, env_state, rng

    actor_state, ts, env_state, rng = probe(0, actor_state, ts, env_state, rng)
    for step in range(1, steps + 1):
        rng, ka, kl = jax.random.split(rng, 3)
        roll, actor_state, ts, env_state = unroll(
            learner_state.params, actor_state, ts, env_state, ka
        )
        learner_state, actor_state, _ = learner(
            kl, augment(roll), learner_state, actor_state, mp, False
        )
        if step % probe_every == 0 or step == steps:
            actor_state, ts, env_state, rng = probe(step, actor_state, ts, env_state, rng)
    return curve


def gate(curve, tol_settle: float = 0.08) -> tuple[bool, str]:
    """Is this run's readout admissible?

    The bucketed policy must reproduce the rollouts, or the exact DP behind
    every number is not describing this agent. The agent must have improved the
    objective it is actually optimising -- an earlier version judged this by KL
    to the target and had the sign backwards, since a cold agent raises its own
    objective while moving away from the target. And the run must have settled,
    or the temperature is a snapshot of something still moving.
    """
    first, last = curve[0], curve[-1]
    if last["bucket_err"] >= 0.05:
        return False, "bucketing invalid"
    if last["soft"] <= first["soft"] + 0.05:
        return False, "did not learn"
    if len(curve) >= 3 and abs(last["lambda_eff"] - curve[-2]["lambda_eff"]) > tol_settle * max(
        1.0, curve[-2]["lambda_eff"]
    ):
        return False, "not settled"
    return True, "ok"


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rollout-len", type=int, default=16)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--only", type=str, default="", help="substring filter on arm names")
    ap.add_argument("--probe-every", type=int, default=1000)
    ap.add_argument("--collect", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--traj", type=int, default=200000)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    h = HEIGHT
    lams = np.exp(np.linspace(math.log(0.03), math.log(8.0), 121))
    fam = SoftFamily(h, lams)

    print("=" * 84)
    print("STEP 1  the exact soft family, and the landscape it lives on")
    print("=" * 84)
    L = fam.logR.reshape(h, h)
    print(f"  log R spans {L.min():.2f} to {L.max():.2f}; target entropy "
          f"{-(fam.target * np.log(fam.target)).sum():.3f} against "
          f"{math.log(h * h):.3f} for uniform")
    cold = forward_dp(soft_policy(h, float(lams[0])), h)
    print(f"  cold member return {cold[1]:.3f} against {fam.logR[0]:.3f} for stopping "
          f"at the source, so a cold agent has to travel")
    anchor = float(np.max(np.abs(forward_dp(soft_policy(h, 1.0), h)[0] - fam.target)))
    print(f"  member at lambda = 1 equals the target R/Z to {anchor:.3e}   "
          f"{'PASS' if anchor < 1e-12 else 'FAIL'}")
    if anchor >= 1e-12:
        return 1

    phi = geometric_potential(h)
    shift = max(
        float(np.max(np.abs(soft_policy(h, l, phi) - soft_policy(h, l))))
        for l in (0.1, 0.5, 1.0, 2.0, 5.0)
    )
    print(f"  shaping by the geometric potential moves the soft-optimal policy by "
          f"{shift:.3e}")
    print(f"  at every coefficient tried, so the family being fitted is the same one "
          f"{'PASS' if shift < 1e-12 else 'FAIL'}")
    if shift >= 1e-12:
        return 1

    print()
    print("  I1  the fit, fed exact family members with no sampling")
    print(f"      {'true':>8}{'fitted':>10}{'residual':>12}")
    i1 = []
    for lam in (0.1, 0.3, 0.7, 1.0, 1.5, 3.0, 6.0):
        p = forward_dp(soft_policy(h, lam), h)[0]
        got, res, _ = fam.fit(p)
        i1.append((lam, got, res))
        print(f"      {lam:>8.2f}{got:>10.4f}{res:>12.2e}")
    i1_ok = all(abs(g - t) / t < 0.005 and r < 1e-9 for t, g, r in i1)
    print(f"      recovers every coefficient to 0.5 percent, residual below 1e-9: "
          f"{'PASS' if i1_ok else 'FAIL'}")

    print()
    print(f"  I2  the whole pipeline -- sample {args.traj} trajectories from the exact")
    print("      policy, bucket by state, forward DP, fit")
    print("      the bucketing error carries no sampling noise: on a DAG whose")
    print("      trajectories all start at the source, the DP under the bucketed")
    print("      policy reproduces the empirical visit frequencies identically, so")
    print("      any departure from zero measures dependence on history alone")
    print(f"      {'true':>8}{'fitted':>10}{'residual':>12}{'bucket err':>13}{'seen':>7}")
    rng = np.random.default_rng(0)
    i2 = []
    for lam in (0.3, 0.7, 1.0, 1.5, 3.0):
        pol = soft_policy(h, lam)
        pi_hat, emp, nseen = bucket(sample_exact(pol, h, args.traj, rng), h)
        term, _, _ = forward_dp(pi_hat, h)
        got, res, _ = fam.fit(term)
        berr = float(np.max(np.abs(term - emp)))
        i2.append((lam, got, res, berr))
        print(f"      {lam:>8.2f}{got:>10.4f}{res:>12.2e}{berr:>13.2e}{nseen:>7}")
    i2_ok = all(abs(g - t) / t < 0.10 and b < 1e-9 for t, g, r, b in i2)
    print(f"      recovers every coefficient within 10 percent, bucketing exact: "
          f"{'PASS' if i2_ok else 'FAIL'}")

    if not (i1_ok and i2_ok):
        print("\n  the instrument is not valid; nothing it says about an agent can be read")
        return 1

    meta_params = disco_probe.unflatten_params(np.load(disco_probe.WEIGHTS))

    # The ladder runs shaped, because shaping provably leaves the soft family
    # pointwise unchanged while letting the agents actually optimise, and a
    # ladder of agents that have not optimised measures nothing.
    ladder = [(f"ladder ac c={c}", "actor_critic", 0.0, c, True, True)
              for c in (0.0, 0.3, 1.0, 3.0)]
    # An arm whose coefficient is exactly one by construction: the entropy is
    # inside the reward, so the critic bootstraps it and the arm is soft RL at
    # precisely alpha rather than approximately.
    exact_arm = [("ladder ac r-aug 1", "actor_critic", 1.0, 0.0, True, True)]
    probes = [("disco shaped", "disco", 0.0, 0.0, True, True),
              ("disco shaped r-aug 1", "disco", 1.0, 0.0, True, True)]
    # The same rule in the unshaped reduction, which is where it collapses.
    hostile = [("disco unshaped", "disco", 0.0, 0.0, True, False)]
    # Wiring control. Drops the log P_B payments, so it is the ordinary
    # hypergrid and not the reduction: no temperature can be read from it. It
    # answers only whether a rule that fails in the reduction fails because of
    # the MDP or because of how we built it.
    wiring = [("CONTROL no-pb disco", "disco", 0.0, 0.0, False, False)]
    arms = ladder + exact_arm + probes + hostile + wiring
    if args.only:
        arms = [a for a in arms if args.only in a[0]]

    print("\n" + "=" * 84)
    print(f"STEP 2  arms in the MDP, {args.seeds} seeds, {args.steps} updates, "
          f"probed every {args.probe_every}")
    print("=" * 84)
    hdr = (f"  {'arm':<22}{'sd':>3}{'step':>6}{'seen':>5}{'bkt':>9}{'return':>9}"
           f"{'H':>7}{'KL':>8}{'lam_eff':>9}{'resid':>9}")
    best = {True: best_return(h, True), False: best_return(h, False)}
    print(f"  the best undiscounted return available is {best[True]:.3f} with the "
          f"log P_B payments and {best[False]:.3f} without;")
    print(f"  stopping at the source pays {fam.logR[0]:.3f} either way, so an agent "
          f"sitting there has not optimised")
    results = []
    for name, rule, alpha, ecost, pb, shaped in arms:
        print()
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for seed in range(args.seeds):
            curve = run_arm(rule, alpha, ecost, h, args.batch, args.rollout_len,
                            args.steps, args.probe_every, args.collect, seed,
                            meta_params, fam, args.lr, pb, shaped)
            ok, why = gate(curve)
            for c in curve:
                mark = "" if c is not curve[-1] else ("   <- " + why)
                print(f"  {name:<22}{seed:>3}{c['step']:>6}{c['states_seen']:>5}"
                      f"{c['bucket_err']:>9.1e}{c['return']:>9.3f}{c['h_traj']:>7.3f}"
                      f"{c['kl']:>8.3f}{c['lambda_eff']:>9.3f}{c['residual']:>9.4f}{mark}")
            results.append({"arm": name, "rule": rule, "alpha": alpha, "pb": pb,
                            "shaped": shaped,
                            "entropy_cost": ecost, "seed": seed, "admissible": ok,
                            "reason": why, "curve": curve, "final": curve[-1]})

    print("\n" + "=" * 84)
    print("STEP 3  reading")
    print("=" * 84)
    summary = {}
    for name, rule, alpha, ecost, pb, shaped in arms:
        runs = [r for r in results if r["arm"] == name]
        if not runs:
            continue
        ret = np.array([r["final"]["return"] for r in runs])
        gaps = best[pb] - ret
        good = [r for r in runs if r["admissible"]]
        summary[name] = {
            "rule": rule, "alpha": alpha, "entropy_cost": ecost, "pb": pb,
            "shaped": shaped,
            "return_mean": float(ret.mean()), "gap_mean": float(gaps.mean()),
            "n_admissible": len(good),
        }
        if not good:
            summary[name]["reasons"] = sorted({r["reason"] for r in runs})
            print(f"  {name:<22} return {ret.mean():>7.3f} "
                  f"({gaps.mean():+.3f} from best)   no admissible seed "
                  f"({', '.join(summary[name]['reasons'])})")
            continue
        le = np.array([r["final"]["lambda_eff"] for r in good])
        rs = np.array([r["final"]["residual"] for r in good])
        kl = np.array([r["final"]["kl"] for r in good])
        summary[name].update({
            "lambda_eff_mean": float(le.mean()), "lambda_eff_std": float(le.std()),
            "residual_mean": float(rs.mean()), "kl_mean": float(kl.mean()),
        })
        tail = ("  [control, not the reduction, no temperature meaning]" if not pb
                else f"   lambda_eff = {le.mean():.3f} +/- {le.std():.3f}   "
                     f"residual = {rs.mean():.4f}   KL to R/Z = {kl.mean():.3f}")
        print(f"  {name:<22} return {ret.mean():>7.3f} "
              f"({gaps.mean():+.3f} from best)   n = {len(good)}/{args.seeds}{tail}")

    def lam(n):
        return summary.get(n, {}).get("lambda_eff_mean")

    def gap(n):
        return summary.get(n, {}).get("gap_mean")

    def res(n):
        return summary.get(n, {}).get("residual_mean")

    print()
    wire = gap("CONTROL no-pb disco")
    if wire is not None:
        print(f"  wiring control: with the log P_B payments removed, the discovered "
              f"rule ends {wire:.3f} from the best return, so it can optimise this "
              f"grid at all")
    hostile_gap = gap("disco unshaped")
    if hostile_gap is not None:
        print(f"  the same rule in the unshaped reduction ends {hostile_gap:.3f} from "
              f"the best return")
    shaped_gap = gap("disco shaped")
    if shaped_gap is not None:
        print(f"  and with the geometric potential shaped in, which leaves the soft "
              f"family pointwise unchanged, {shaped_gap:.3f} from it")

    lad = [(c, lam(n)) for n, _, _, c, _, _ in ladder if lam(n) is not None]
    mono = len(lad) >= 3 and all(lad[i][1] < lad[i + 1][1] for i in range(len(lad) - 1))
    if lad:
        print(f"  ladder monotone in the coefficient we set: "
              f"{'PASS' if mono else 'FAIL'}   ({len(lad)}/{len(ladder)} arms read)   "
              + "  ".join(f"c={c}->{v:.2f}" for c, v in lad))
    ra = lam("ladder ac r-aug 1")
    if ra is not None:
        print(f"  the arm whose coefficient is exactly 1 by construction reads "
              f"{ra:.3f}")

    d0, dres = lam("disco shaped"), res("disco shaped")
    print()
    if wire is not None and wire > 1.0:
        verdict = ("VOID -- the discovered rule does not reach the best return even "
                   "with the log P_B payments removed, so the failure is in how this "
                   "environment was built and not in the reduction")
    elif d0 is None:
        verdict = ("INCONCLUSIVE -- the discovered rule produced no admissible seed "
                   "in the shaped reduction")
    elif shaped_gap is not None and shaped_gap > 0.5:
        verdict = (f"INCONCLUSIVE -- even shaped, the discovered rule ends "
                   f"{shaped_gap:.2f} short of the best return, so its distribution "
                   f"is that of an agent that has not finished optimising")
    elif dres is not None and dres > 0.4:
        verdict = (f"OUTSIDE THE FAMILY -- no coefficient describes the discovered "
                   f"rule; the best fit leaves a residual of {dres:.3f}")
    elif not mono:
        verdict = ("VOID -- the readout is not monotone in a coefficient we set "
                   "ourselves, so nothing it says about a discovered rule can be read")
    elif abs(d0 - 1.0) < 0.15:
        verdict = (f"GFLOWNET HERE -- the discovered rule implements lambda_eff = "
                   f"{d0:.3f} unaided, and lambda = 1 is exactly a GFlowNet")
    else:
        side = "colder" if d0 < 1 else "hotter"
        verdict = (f"NOT A GFLOWNET, AND {1 / d0:.1f} TIMES {side.upper()} -- in the "
                   f"MDP whose soft-optimal policy at lambda = 1 is exactly a "
                   f"GFlowNet, the discovered rule solves the problem but settles at "
                   f"an effective entropy coefficient of {d0:.3f}, residual "
                   f"{dres:.3f}")
    print(f"Verdict: {verdict}")

    out = {
        "config": vars(args),
        "instrument": {"i1": i1, "i2": i2, "anchor": anchor},
        "per_run": results, "summary": summary, "verdict": verdict,
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
