#!/usr/bin/env python3
"""Does DiscoRL's y-target behave like a flow, or like a value?

This is the training-free replacement for the multi-path exponent experiment
that the first draft of the dossier proposed.  That experiment was ill-posed:
DiscoRL maximises return, so on a hypergrid its policy collapses onto argmax R
and the terminal distribution degenerates to a point mass, at which point the
exponent gamma in p(x) ~ R(x) n(x)^gamma is not defined.  Measuring it early in
training would confound "has not converged" with "structurally unbiased".

So probe the update rule itself rather than the distribution it converges to.

Detailed balance says a state quantity is reconstructed from its successor and
the local transition probabilities,

    log F(s) = log F(s') + log p_B(s|s') - log p_F(s'|s),

whereas a value bootstrap says

    V(s) = r + gamma V(s'),

with NO dependence on the probability of the action that was taken.  That single
difference is the whole discriminant.  Define, for the scalar readout phi that
DiscoRL itself applies to its 600-way categorical prediction before feeding it
back to the meta-network (`y_net`, a 600->16->1 MLP whose weights ship inside
disco_103.npz -- so this probe has zero free parameters):

    beta  =  d phi(y_hat_t) / d log pi(a_t | s_t)      <- 0 for a value rule
    alpha =  d phi(y_hat_t) / d phi(y_{t+1})           <- ~1 for both

alpha is a sanity check; beta is the measurement.  A meta-network whose y target
carries flow semantics must react to the log-probability of the transition it
just took.  One whose y target is a disguised value function cannot.

Controls: the identical measurement under randomly initialised meta-parameters
of the same architecture, over several seeds.  That fixes the scale of "no
structure" for this readout and this input distribution.

Usage:
  .venv/bin/python research/disco_probe.py --json research/disco_probe.json
"""

from __future__ import annotations

import argparse
import functools
import json

import chex
import jax
import jax.numpy as jnp
import numpy as np

from disco_rl import agent as agent_lib
from disco_rl import types

import hypergrid_env

PLACEBO_OFFSET = 3

WEIGHTS = "/tmp/disco_rl/disco_rl/update_rules/weights/disco_103.npz"


def unflatten_params(flat) -> dict:
    """npz stores 'module/w' and 'module/b' flat; haiku wants them nested."""
    out: dict = {}
    for key_wb in flat:
        key = "/".join(key_wb.split("/")[:-1])
        out[key] = {"b": flat[f"{key}/b"], "w": flat[f"{key}/w"]}
    return out


def build(height: int, batch_size: int, seed: int):
    env = hypergrid_env.HypergridJittableEnvironment(
        batch_size=batch_size,
        env_settings=hypergrid_env.get_config_hypergrid(),
    )
    settings = agent_lib.get_settings_disco()
    settings.net_settings.name = "mlp"
    settings.net_settings.net_args = dict(
        dense=(128, 128),
        model_arch_name="lstm",
        head_w_init_std=1e-2,
        model_kwargs=dict(head_mlp_hiddens=(64,), lstm_size=64),
    )
    settings.learning_rate = 1e-3
    ag = agent_lib.Agent(
        agent_settings=settings,
        single_observation_spec=env.single_observation_spec(),
        single_action_spec=env.single_action_spec(),
        batch_axis_name=None,  # single device; keeps the probe free of pmap
    )
    del height, seed
    return env, ag


def collect_rollout(env, ag, rollout_len: int, rng):
    """One on-policy rollout from a freshly initialised agent."""
    learner_state = ag.initial_learner_state(rng)
    actor_state = ag.initial_actor_state(rng)
    env_state, ts = env.reset(rng)

    def _step(carry, step_rng):
        env_state, ts, actor_state = carry
        actor_timestep, actor_state = ag.actor_step(
            learner_state.params, step_rng, ts, actor_state
        )
        env_state, ts = env.step(env_state, actor_timestep.actions)
        return (env_state, ts, actor_state), actor_timestep

    _, stacked = jax.lax.scan(
        _step, (env_state, ts, actor_state), jax.random.split(rng, rollout_len)
    )
    return types.ActorRollout.from_timestep(stacked), learner_state, actor_state


def readout_fn(meta_params: dict, which: str):
    """phi: DiscoRL's own scalar readout of a 600-way categorical prediction.

    `y_net` and `z_net` are created in that order inside meta_nets.LSTM.__call__,
    so haiku names them `mlp` and `mlp_1`. They are hk.nets.MLP((16, 1)) with the
    default relu between layers and no activation on the output.
    """
    stem = "lstm/mlp" if which == "y" else "lstm/mlp_1"
    w0, b0 = meta_params[f"{stem}/~/linear_0"]["w"], meta_params[f"{stem}/~/linear_0"]["b"]
    w1, b1 = meta_params[f"{stem}/~/linear_1"]["w"], meta_params[f"{stem}/~/linear_1"]["b"]

    def phi(logits):
        h = jax.nn.relu(jax.nn.softmax(logits, axis=-1) @ w0 + b0)
        return jnp.squeeze(h @ w1 + b1, axis=-1)

    return phi



def train(env, ag, meta_params, rollout_len: int, steps: int, rng):
    """Train the agent with Disco103 itself, then hand back a fresh rollout.

    Disco103 was meta-learned against agents that had already found structure.
    Feeding it a randomly initialised agent -- near-uniform policy, near-uniform
    600-way predictions -- is an off-distribution query, and its outputs collapse
    to a constant, which would make every sensitivity read as zero for reasons
    that have nothing to do with flows.
    """
    learner_state = ag.initial_learner_state(rng)
    actor_state = ag.initial_actor_state(rng)
    env_state, ts = env.reset(rng)

    @jax.jit
    def unroll(params, actor_state, ts, env_state, rng):
        def _step(carry, step_rng):
            env_state, ts, actor_state = carry
            at, actor_state = ag.actor_step(params, step_rng, ts, actor_state)
            env_state, ts = env.step(env_state, at.actions)
            return (env_state, ts, actor_state), at

        (env_state, ts, actor_state), stacked = jax.lax.scan(
            _step, (env_state, ts, actor_state), jax.random.split(rng, rollout_len)
        )
        return types.ActorRollout.from_timestep(stacked), actor_state, ts, env_state

    learner_step = jax.jit(ag.learner_step, static_argnums=(5,))

    ret = 0.0
    for step in range(steps):
        rng, r_act, r_learn = jax.random.split(rng, 3)
        rollout, actor_state, ts, env_state = unroll(
            learner_state.params, actor_state, ts, env_state, r_act
        )
        learner_state, actor_state, _ = learner_step(
            r_learn, rollout, learner_state, actor_state, meta_params, False
        )
        ret = 0.99 * ret + 0.01 * float(jnp.sum(rollout.rewards) / rollout.rewards.shape[1])
        if step % max(1, steps // 5) == 0:
            print(f"  train step {step:>5}  reward/episode EMA {ret / (1 - 0.99 ** (step + 1)):.4f}")

    rng, r_act = jax.random.split(rng)
    rollout, actor_state, ts, env_state = unroll(
        learner_state.params, actor_state, ts, env_state, r_act
    )
    return rollout, learner_state, actor_state

def make_meta_out_fn(ag, rollout, learner_state, actor_state):
    """Closes over everything fixed so only the perturbed tensors vary."""
    agent_net_state = actor_state
    hyper_params = ag.settings.hyper_params.to_dict()

    def meta_out_of(agent_out_override, rewards_override=None):
        rewards = rollout.rewards[1:] if rewards_override is None else rewards_override
        eta_inputs = types.UpdateRuleInputs(
            observations=rollout.observations,
            actions=rollout.actions,
            rewards=rewards,
            is_terminal=rollout.discounts[1:] == 0,
            behaviour_agent_out=rollout.agent_outs,
            agent_out=agent_out_override,
            value_out=None,
        )
        meta_out, _ = ag.update_rule.unroll_meta_net(
            meta_params=meta_out_of.meta_params,
            params=learner_state.params,
            state=agent_net_state,
            meta_state=learner_state.meta_state,
            rollout=eta_inputs,
            hyper_params=hyper_params,
            unroll_policy_fn=ag._network.unroll,  # pylint: disable=protected-access
            rng=jax.random.PRNGKey(0),
            axis_name=None,
        )
        return meta_out

    return meta_out_of


def sensitivities(ag, rollout, learner_state, actor_state, meta_params, t0: int, h: float):
    """Three response coefficients of the y-target at trajectory index t0.

    Central finite differences, NOT autodiff: the meta-network's policy input
    carries a `stop_grad` (disco.py:338, the transform tuple for
    agent_out/logits), so a reverse-mode derivative through that path is
    identically zero and says nothing about the forward map.  Forward values are
    untouched by stop_grad, so differencing measures the real response.

        alpha = d phi(y_hat_t) / d phi(y_{t+1})      successor prediction
        beta  = d phi(y_hat_t) / d log pi(a_t|s_t)   THE DISCRIMINANT
        rho   = d phi(y_hat_t) / d r_t               immediate reward

    Detailed balance needs beta of order 1; a value bootstrap forces beta = 0
    while leaving alpha and rho of order 1.  rho therefore certifies that a
    small beta means insensitivity to the POLICY in particular, rather than a
    dead readout.

    Batch elements couple only through the advantage/TD EMA normalisers, which
    take batch means, so cross-talk is O(1/B) and is reported, not removed.
    """
    phi_y = readout_fn(meta_params, "y")
    base_out, _ = ag.unroll_net(learner_state.params, actor_state, rollout)
    meta_out_of = make_meta_out_fn(ag, rollout, learner_state, actor_state)
    meta_out_of.meta_params = meta_params

    actions = rollout.actions[t0]  # [B]
    onehot = jax.nn.one_hot(actions, base_out["logits"].shape[-1])

    def phi_hat(out, rewards=None):
        return phi_y(meta_out_of(out, rewards)["y"][t0])

    def shifted(key, value):
        out = dict(base_out)
        out[key] = value
        return out

    # --- beta (local): perturb log pi at the SAME step whose target we read.
    def beta_at(offset: int):
        idx = t0 + offset
        acts = rollout.actions[idx]
        oh = jax.nn.one_hot(acts, base_out["logits"].shape[-1])
        lo = shifted("logits", base_out["logits"].at[idx].add(-h * oh))
        hi = shifted("logits", base_out["logits"].at[idx].add(h * oh))

        def log_pi(lg):
            return jnp.take_along_axis(
                jax.nn.log_softmax(lg[idx], axis=-1), acts[:, None], axis=-1
            )[:, 0]

        return np.asarray(
            (phi_hat(hi) - phi_hat(lo)) / (log_pi(hi["logits"]) - log_pi(lo["logits"]))
        )

    beta = beta_at(0)
    # Placebo: detailed balance is LOCAL, so the target at t should barely care
    # about the policy three steps later. The reverse recurrence smears
    # information along the suffix, and without this control a non-zero beta
    # could be that smear rather than a balance condition.
    beta_placebo = beta_at(PLACEBO_OFFSET)

    # --- alpha. A local gradient direction is useless here: y_net's relu is
    #     flat at the agent's own y for this environment, so the steepest-ascent
    #     step is exactly zero and every element divides by nothing. Use finite
    #     random directions in logit space, large enough to move the softmax,
    #     and keep the draws that actually move phi(y_{t+1}).
    y_next = base_out["y"][t0 + 1]  # [B, 600]
    a_num, a_den = [], []
    for k in range(8):
        u = jax.random.normal(jax.random.PRNGKey(9_000 + k), y_next.shape)
        u = u / jnp.linalg.norm(u, axis=-1, keepdims=True)
        d = 6.0 * u  # logit-space amplitude; softmax moves materially at this scale
        num = phi_hat(shifted("y", base_out["y"].at[t0 + 1].add(d))) - phi_hat(
            shifted("y", base_out["y"].at[t0 + 1].add(-d))
        )
        den = phi_y(y_next + d) - phi_y(y_next - d)
        a_num.append(np.asarray(num))
        a_den.append(np.asarray(den))
    a_num = np.concatenate(a_num)
    a_den = np.concatenate(a_den)
    valid = np.abs(a_den) > 1e-7
    alpha = np.where(valid, a_num / np.where(valid, a_den, 1.0), np.nan)

    # --- rho
    rewards = rollout.rewards[1:]
    r_scale = float(jnp.maximum(jnp.std(rewards), 1e-3))
    bump = jnp.zeros_like(rewards).at[t0].set(h * r_scale)
    rho = np.asarray(
        (phi_hat(dict(base_out), rewards + bump) - phi_hat(dict(base_out), rewards - bump))
        / (2.0 * h * r_scale)
    )

    spread = float(jnp.std(phi_hat(dict(base_out))))
    return np.asarray(beta), beta_placebo, alpha, rho, spread, float(np.mean(valid))


def summarise(name, betas, placebos, alphas, rhos, spreads, valids):
    b = np.concatenate(betas)
    p = np.concatenate(placebos)
    a = np.concatenate(alphas)
    r = np.concatenate(rhos)
    return {
        "arm": name,
        "n": int(b.size),
        "phi_spread": float(np.mean(spreads)),
        "alpha_valid_frac": float(np.mean(valids)),
        "beta_mean": float(b.mean()),
        "beta_absmean": float(np.abs(b).mean()),
        "beta_std": float(b.std()),
        "placebo_absmean": float(np.abs(p).mean()),
        "locality": float(np.abs(b).mean() / (np.abs(p).mean() + 1e-12)),
        "alpha_absmean": float(np.nanmean(np.abs(a))),
        "rho_absmean": float(np.abs(r).mean()),
        "beta_over_alpha": float(np.abs(b).mean() / (np.nanmean(np.abs(a)) + 1e-12)),
        "beta_over_rho": float(np.abs(b).mean() / (np.abs(r).mean() + 1e-12)),
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rollout-len", type=int, default=20)
    ap.add_argument("--taps", type=int, default=6, help="trajectory indices probed")
    ap.add_argument("--null-seeds", type=int, default=3)
    ap.add_argument("--h", type=float, default=0.5, help="central-difference step")
    ap.add_argument("--json", type=str, default="")
    ap.add_argument(
        "--train-steps",
        type=int,
        default=400,
        help="agent updates under Disco103 before probing; the meta-network was "
        "meta-trained against competent agents, so probing a freshly initialised "
        "one puts its inputs far off-distribution and squashes every response",
    )
    args = ap.parse_args()

    rng = jax.random.PRNGKey(0)
    env, ag = build(args.height, args.batch, 0)

    disco = unflatten_params(np.load(WEIGHTS))
    ref, _ = ag.update_rule.init_params(jax.random.PRNGKey(0))
    chex.assert_trees_all_equal_shapes_and_dtypes(ref, disco)

    print(f"hypergrid H={args.height}  batch={args.batch}  rollout={args.rollout_len}")
    rollout, learner_state, actor_state = train(
        env, ag, disco, args.rollout_len, args.train_steps, rng
    )

    # t0 + PLACEBO_OFFSET must stay in range: JAX silently DROPS out-of-bounds
    # scatter indices, which would make the placebo a no-op reading as nan.
    taps = np.linspace(2, args.rollout_len - PLACEBO_OFFSET - 2, args.taps).astype(int)
    print(f"probing trajectory indices {list(taps)}\n")

    def run(name, params):
        cols = ([], [], [], [], [], [])
        for t0 in taps:
            vals = sensitivities(
                ag, rollout, learner_state, actor_state, params, int(t0), args.h
            )
            for c, v in zip(cols, vals):
                c.append(v)
        return summarise(name, *cols)

    rows = [run("Disco103", disco)]
    for s in range(args.null_seeds):
        rnd, _ = ag.update_rule.init_params(jax.random.PRNGKey(1000 + s))
        rows.append(run(f"random-init seed {s}", jax.tree.map(np.asarray, rnd)))

    hdr = (
        f"{'arm':<21}{'|beta|':>10}{'|alpha|':>10}{'|rho|':>10}"
        f"{'beta/alpha':>12}{'placebo':>10}{'locality':>10}{'phi spread':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['arm']:<21}{r['beta_absmean']:>10.2e}{r['alpha_absmean']:>10.2e}"
            f"{r['rho_absmean']:>10.2e}{r['beta_over_alpha']:>12.4f}"
            f"{r['placebo_absmean']:>10.2e}{r['locality']:>10.2f}{r['phi_spread']:>12.2e}"
        )

    d = rows[0]
    null = np.array([r["beta_absmean"] for r in rows[1:]])
    ratio = d["beta_absmean"] / (null.mean() + 1e-12)
    print(
        f"\nDisco103 |beta| is {ratio:.1f}x the random-init null "
        f"({null.mean():.2e} +/- {null.std():.2e})"
    )
    print(
        "Detailed balance predicts |beta/alpha| ~ 1 and strong locality; "
        "a pure value bootstrap predicts |beta/alpha| = 0.\n"
        f"Measured: |beta/alpha| = {d['beta_over_alpha']:.4f}, "
        f"locality = {d['locality']:.2f}x "
        "(same-step response over the response to the policy three steps later)."
    )
    if not np.isfinite(d["locality"]):
        verdict = "INCONCLUSIVE -- locality control did not evaluate"
    elif d["locality"] < 1.5:
        verdict = "NOT LOCAL -- the response is suffix smear, not a balance condition"
    elif d["beta_over_alpha"] > 0.7:
        verdict = "FLOW-LIKE"
    elif d["beta_over_alpha"] > 0.15:
        verdict = "PARTIAL -- policy-sensitive and local, but well short of detailed balance"
    else:
        verdict = "VALUE-LIKE (policy-insensitive target)"
    print(f"Verdict: {verdict}")

    out = {
        "height": args.height,
        "batch": args.batch,
        "rollout_len": args.rollout_len,
        "taps": [int(t) for t in taps],
        "arms": rows,
        "null_absmean": float(null.mean()),
        "null_absstd": float(null.std()),
        "ratio_to_null": float(ratio),
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
