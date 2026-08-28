#!/usr/bin/env python3
"""Extension D: does a categorical (HL-Gauss style) log-flow head beat a scalar one?

The dossier's Layer-1 result is that a GFlowNet minimises Var[W] while DiscoRL
minimises a KL against a categorical target.  Extension D asks a narrower,
purely engineering question that falls out of that reading:

    A GFlowNet's state-flow head is a SCALAR regression in log space.  DiscoRL
    never regresses a scalar -- every prediction it makes is a categorical
    distribution over a fixed support, read out by taking its expectation, and
    trained by a KL.  If the scalar log-flow regression is the badly conditioned
    part of a GFlowNet, then swapping ONLY the readout for a categorical head
    over a fixed support should improve conditioning and convergence.

This script tests exactly that, and nothing else.  There is no DiscoRL code
here, no meta-learning, and no comparison against Disco103.

WHAT IS HELD FIXED
------------------
The two arms share, bit-for-bit:

  * the environment (the same 8x8 hypergrid DAG, reward, action set and
    uniform-over-parents backward policy as `research/cumulants.py`, imported
    from that module so the two scripts cannot drift apart),
  * the trunk MLP and the policy head, including their random initialisation
    (both arms draw from the same PRNG keys, in the same order),
  * the free scalar log Z parameter and its initial value,
  * the objective, the optimiser, both learning rates, the batch size, the
    number of steps, the evaluation schedule, and the list of seeds,
  * the initial log-flow FUNCTION: the scalar head's bias and the categorical
    head's logits are both zero-initialised, and the support is chosen
    symmetric about 0, so at step 0 both arms predict log F(s) = 0 for every s.

The ONLY difference is how the trunk's features are turned into log F(s):

  arm "scalar"       log F(s) = w . h(s) + b                      (1 output)
  arm "categorical"  log F(s) = softmax(W h(s) + b) . support     (K outputs)

The categorical arm is a *readout* reparameterisation, not a different loss:
its gradient still comes from the same squared flow-balance residual.  (A
two-hot cross-entropy target would additionally change the objective, which
would confound the comparison this script is trying to make.)

OBJECTIVE
---------
Plain trajectory balance contains no state flow at all -- the only flow term in
it is F(s_0) = Z -- so it cannot test a state-flow head.  We therefore apply the
trajectory-balance residual to every sub-trajectory of each sampled trajectory
(this is sub-trajectory balance with lambda = 1; the m = 0, n = end term is
literally the classical TB residual).  Writing the trajectory as the node
sequence s_0, s_1, ..., s_{n-1}, TERMINAL, with

    A_0 = log Z_theta                (the free scalar)
    A_t = log F(s_t)                 (the head under test, 1 <= t <= n-1)
    A_n = 0                          (the sink; log R(x) is folded into g below)
    g_t = log p_F(a_t|s_t) - log p_B(s_t|s_{t+1})   [ and -log R(x) on `stop` ]
    C_t = sum_{u<t} g_u,  D_t = A_t - C_t

every sub-trajectory residual is just D_m - D_n, and the loss is the mean of
(D_m - D_n)^2 over all m < n.  That collapses to N*sum(D^2) - (sum D)^2 over
N(N-1)/2 pairs, so it costs one pass, not a double loop.

MEASUREMENT
-----------
KL(p_terminal || R/Z) is computed EXACTLY by dynamic programming over the DAG
using `cumulants.work_stats` / `cumulants.kl_to_target` -- never by sampling --
so the reported convergence curve carries no Monte-Carlo noise of its own.  We
also track |log Z_theta - log Z| against the exact log Z = 2.8118094353930627,
and the per-step global gradient norm (mean and 99th percentile, pooled over
seeds) as the conditioning proxy.

The KL curve is exact, but the policy it measures is trained on sampled
trajectories, so past some point training stops reducing bias and merely
jitters.  `noise_floor_kl` names that level, and is read off the curve rather
than chosen: it is the largest arm-averaged KL seen at or after the first
checkpoint where the arm-averaged KL goes UP, i.e. the height of the bounce
once descent has stopped.  Checkpoints at or below it are marked as such,
`geomean_kl_ratio` is taken only over checkpoints strictly above it, and
`final_checkpoint_in_noise_floor` warns when the last checkpoint -- the one the
`final_kl_*` fields report -- is inside it and therefore decides nothing.

HONEST CAVEAT, STATED UP FRONT
------------------------------
The motivating complaint is that log F "spans tens of orders of magnitude".  On
this particular grid it does not: exact DP gives log F in [-2.710, 2.812], a
span of 5.5 nats.  So this is a weak-stress version of extension D, and a null
result here is evidence about *this* regime only.  Reported as-is either way;
nothing was tuned until one arm won.

Usage:  python research/logf_head.py [--json research/logf_head.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cumulants import Grid, kl_to_target, reward, work_stats  # noqa: E402

# Action encoding used everywhere below.
RIGHT, UP, STOP = 0, 1, 2
NEG_INF = -1e9


# --------------------------------------------------------------------------
# Static tables for the DAG.  There are only 64 states, so every quantity the
# rollout needs is a lookup, and the network is evaluated once per gradient
# step on ALL states rather than once per visited state.
# --------------------------------------------------------------------------


def build_tables(h: int):
    n = h * h
    enc = np.zeros((n, 2 * h), np.float32)
    mask = np.zeros((n, 3), bool)
    child = np.zeros((n, 3), np.int32)
    logpb = np.zeros((n, 3), np.float32)
    for i in range(h):
        for j in range(h):
            s = i * h + j
            enc[s, i] = 1.0
            enc[s, h + j] = 1.0
            child[s, :] = s  # self-loop placeholder; only read for real moves
            if i + 1 < h:
                mask[s, RIGHT] = True
                child[s, RIGHT] = (i + 1) * h + j
                # p_B is uniform over the parents of the CHILD (i+1, j).
                logpb[s, RIGHT] = -math.log(1 + (1 if j > 0 else 0))
            if j + 1 < h:
                mask[s, UP] = True
                child[s, UP] = i * h + (j + 1)
                logpb[s, UP] = -math.log((1 if i > 0 else 0) + 1)
            mask[s, STOP] = True
            # `stop` has no backward edge; cumulants.edge_work charges -log R(s)
            # there instead, which is what makes the full-trajectory residual
            # equal log Z + W(tau).
            logpb[s, STOP] = math.log(reward(i, j, h))
    return jnp.asarray(enc), jnp.asarray(mask), jnp.asarray(child), jnp.asarray(logpb)


def exact_flows(g: Grid) -> dict:
    """Same edge-flow recursion as cumulants.flow_matching_policy, but returns F."""
    f: dict = {}
    for s in g.states():
        acc = reward(*s, g.h)
        for _, c in g.children(*s):
            acc += f[c] / g.n_parents(*c)
        f[s] = acc
    return f


# --------------------------------------------------------------------------
# Network.  Explicit param dict rather than haiku: the whole point of the
# experiment is that two arms share parameters draw-for-draw, and that is
# easier to guarantee -- and to read -- when the draws are written out.
# --------------------------------------------------------------------------


def init_params(seed: int, h: int, hidden: int, n_bins: int, categorical: bool):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)  # k1,k2 trunk; k3 policy head
    d_in = 2 * h
    p = {
        "W1": jax.random.normal(k1, (d_in, hidden)) * math.sqrt(2.0 / d_in),
        "b1": jnp.zeros(hidden),
        "W2": jax.random.normal(k2, (hidden, hidden)) * math.sqrt(2.0 / hidden),
        "b2": jnp.zeros(hidden),
        "Wp": jax.random.normal(k3, (hidden, 3)) * math.sqrt(2.0 / hidden),
        "bp": jnp.zeros(3),
        "log_z": jnp.zeros(()),
    }
    # Zero-init in BOTH arms.  With a support symmetric about 0 the categorical
    # readout of uniform logits is also 0, so the two arms start from the same
    # log F function and any divergence is attributable to the readout alone.
    out = n_bins if categorical else 1
    p["Wf"] = jnp.zeros((hidden, out))
    p["bf"] = jnp.zeros(out)
    return p


def trunk(p, enc):
    x = jax.nn.relu(enc @ p["W1"] + p["b1"])
    return jax.nn.relu(x @ p["W2"] + p["b2"])


def log_policy(p, enc, mask):
    logits = trunk(p, enc) @ p["Wp"] + p["bp"]
    return jax.nn.log_softmax(jnp.where(mask, logits, NEG_INF), axis=-1)


def log_flow(p, enc, support, categorical: bool):
    out = trunk(p, enc) @ p["Wf"] + p["bf"]
    if categorical:
        return jax.nn.softmax(out, axis=-1) @ support
    return out[..., 0]


# --------------------------------------------------------------------------
# On-policy rollout + sub-trajectory-balance loss.
# --------------------------------------------------------------------------


def rollout_D(key, logp_all, logf_all, log_z, child, logpb, n_steps):
    """Return the per-node balance potentials D_t and their validity mask."""

    def body(carry, k):
        idx, active, C = carry
        a = jax.random.categorical(k, logp_all[idx])
        g = jnp.where(active, logp_all[idx, a] - logpb[idx, a], 0.0)
        nxt = child[idx, a]
        stopped = a == STOP
        C_next = C + g
        # The node reached by a `stop` edge is the sink, whose flow is fixed
        # to 1 (log 0) because log R(x) was already charged inside g.
        A_next = jnp.where(stopped, 0.0, logf_all[nxt])
        return (
            jnp.where(stopped, idx, nxt),
            active & ~stopped,
            C_next,
        ), (A_next - C_next, active)

    keys = jax.random.split(key, n_steps)
    _, (D_rest, valid_rest) = jax.lax.scan(body, (jnp.int32(0), True, 0.0), keys)
    D = jnp.concatenate([log_z[None], D_rest])
    valid = jnp.concatenate([jnp.array([True]), valid_rest])
    return D, valid


def traj_loss(D, valid):
    w = valid.astype(D.dtype)
    n = w.sum()
    s1 = (w * D).sum()
    s2 = (w * D * D).sum()
    # sum_{m<n} (D_m - D_n)^2 == n*sum(D^2) - (sum D)^2, over n(n-1)/2 pairs.
    return (n * s2 - s1 * s1) / (0.5 * n * (n - 1.0))


def make_step(enc, mask, child, logpb, support, categorical, batch, n_steps, opt):
    def loss_fn(p, key):
        logp_all = log_policy(p, enc, mask)
        logf_all = log_flow(p, enc, support, categorical)
        keys = jax.random.split(key, batch)
        D, valid = jax.vmap(rollout_D, in_axes=(0, None, None, None, None, None, None))(
            keys, logp_all, logf_all, p["log_z"], child, logpb, n_steps
        )
        return jax.vmap(traj_loss)(D, valid).mean()

    @jax.jit
    def step(p, opt_state, key):
        loss, grads = jax.value_and_grad(loss_fn)(p, key)
        updates, opt_state = opt.update(grads, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, loss, optax.global_norm(grads)

    return step


# --------------------------------------------------------------------------
# Exact evaluation: no sampling anywhere below this line.
# --------------------------------------------------------------------------


def exact_kl(g: Grid, probs: np.ndarray, log_z_true: float) -> float:
    pol = {}
    for i in range(g.h):
        for j in range(g.h):
            row = probs[i * g.h + j]
            d = {"stop": float(row[STOP])}
            if i + 1 < g.h:
                d["right"] = float(row[RIGHT])
            if j + 1 < g.h:
                d["up"] = float(row[UP])
            pol[(i, j)] = d
    return kl_to_target(g, work_stats(g, pol)["terminal"], log_z_true)


def run_arm(arm, seeds, g, tables, support, args, log_z_true):
    enc, mask, child, logpb = tables
    categorical = arm == "categorical"
    n_steps = 2 * g.h - 1  # at most 2H-2 moves, then a forced `stop`
    opt = optax.multi_transform(
        {"net": optax.adam(args.lr), "z": optax.adam(args.lr_z)},
        {k: ("z" if k == "log_z" else "net") for k in
         ("W1", "b1", "W2", "b2", "Wp", "bp", "Wf", "bf", "log_z")},
    )
    step = make_step(enc, mask, child, logpb, support, categorical,
                     args.batch, n_steps, opt)
    probs_fn = jax.jit(lambda p: jnp.exp(log_policy(p, enc, mask)))

    checkpoints = list(range(0, args.steps + 1, args.eval_every))
    if checkpoints[-1] != args.steps:
        checkpoints.append(args.steps)
    kls = np.zeros((len(seeds), len(checkpoints)))
    logz_err = np.zeros(len(seeds))
    grad_norms = []

    for si, seed in enumerate(seeds):
        p = init_params(seed, g.h, args.hidden, args.bins, categorical)
        opt_state = opt.init(p)
        key = jax.random.PRNGKey(10_000 + seed)
        ci = 0
        for t in range(args.steps + 1):
            if ci < len(checkpoints) and checkpoints[ci] == t:
                kls[si, ci] = exact_kl(g, np.asarray(probs_fn(p)), log_z_true)
                ci += 1
            if t == args.steps:
                break
            key, sub = jax.random.split(key)
            p, opt_state, _, gn = step(p, opt_state, sub)
            grad_norms.append(float(gn))
        logz_err[si] = abs(float(p["log_z"]) - log_z_true)
        print(f"    seed {seed}: KL={kls[si, -1]:.6e}  |dlogZ|={logz_err[si]:.6f}")

    gn = np.asarray(grad_norms)
    return {
        "arm": arm,
        "final_kl_mean": float(kls[:, -1].mean()),
        "final_kl_std": float(kls[:, -1].std()),
        "final_logz_err_mean": float(logz_err.mean()),
        "final_logz_err_std": float(logz_err.std()),
        "grad_norm_mean": float(gn.mean()),
        "grad_norm_p99": float(np.percentile(gn, 99)),
        "curve": [
            {"step": int(s), "kl_mean": float(kls[:, c].mean()), "kl_std": float(kls[:, c].std())}
            for c, s in enumerate(checkpoints)
        ],
    }


def summarise(scalar_arm, categorical_arm) -> dict:
    """Whole-run comparison statistics, including the on-policy noise floor.

    A single final checkpoint is a weak statistic here: both arms eventually
    stop reducing bias and just jitter around the level that minibatch noise
    sustains, and at that point which arm is "ahead" is a coin flip.  The floor
    is estimated from the curve itself rather than picked by hand -- see
    `noise_floor_kl` in the module docstring.
    """
    cs, cc = scalar_arm["curve"], categorical_arm["curve"]
    pairs = [(a, b) for a, b in zip(cs, cc) if a["step"] > 0]
    won = sum(1 for a, b in pairs if b["kl_mean"] < a["kl_mean"])

    pooled = [0.5 * (a["kl_mean"] + b["kl_mean"]) for a, b in zip(cs, cc)]
    rise = next((k for k in range(1, len(pooled)) if pooled[k] > pooled[k - 1]), None)
    floor = max(pooled[rise:]) if rise is not None else 0.0

    above = [(a, b) for a, b in pairs if min(a["kl_mean"], b["kl_mean"]) > floor]
    ref = above or pairs  # degenerate only if the run never left the floor
    gmean = math.exp(sum(math.log(a["kl_mean"] / b["kl_mean"]) for a, b in ref) / len(ref))

    return {
        "geomean_kl_ratio": gmean,
        "geomean_n_checkpoints": len(ref),
        "checkpoints_won_categorical": won,
        "checkpoints_total": len(pairs),
        "noise_floor_kl": floor,
        "final_checkpoint_in_noise_floor": max(cs[-1]["kl_mean"], cc[-1]["kl_mean"]) <= floor,
    }



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=8)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--bins", type=int, default=51)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-z", dest="lr_z", type=float, default=1e-1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=250)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    g = Grid(args.height)
    log_z_true = math.log(sum(reward(i, j, g.h) for i in range(g.h) for j in range(g.h)))
    flows = exact_flows(g)
    lo_exact = math.log(min(flows.values()))
    hi_exact = math.log(max(flows.values()))

    # The categorical support is fixed a priori from the exact DP flows with a
    # 2-nat margin on each side, then symmetrised about 0 so that zero logits
    # read out as log F = 0 -- identical to the zero-initialised scalar head.
    half = max(abs(math.floor(lo_exact) - 2.0), abs(math.ceil(hi_exact) + 2.0))
    support = jnp.linspace(-half, half, args.bins)

    seeds = list(range(args.seeds))
    tables = build_tables(g.h)

    print(f"hypergrid H={g.h}  |X|={g.h * g.h}   log Z (exact) = {log_z_true!r}")
    print(f"exact log F range = [{lo_exact:.6f}, {hi_exact:.6f}]  (span {hi_exact - lo_exact:.3f} nats)")
    print(f"categorical support = [{-half:.1f}, {half:.1f}] with K={args.bins} bins")
    print(f"steps={args.steps} batch={args.batch} hidden={args.hidden} "
          f"lr={args.lr} lr_z={args.lr_z} seeds={seeds}\n")

    t0 = time.time()
    arms = []
    for arm in ("scalar", "categorical"):
        print(f"--- arm: {arm}")
        arms.append(run_arm(arm, seeds, g, tables, support, args, log_z_true))
        print()

    summary = summarise(arms[0], arms[1])
    floor = summary["noise_floor_kl"]

    # The curve is printed BEFORE the final-checkpoint table on purpose: the
    # last checkpoint alone reverses the sign of the effect once both arms are
    # inside the noise floor, so stdout must not lead with it.
    print("=" * 78)
    print("exact KL(p_terminal || R/Z) vs training step, mean +/- std over seeds")
    print(f"{'step':>7}  {'scalar':>21}  {'categorical':>21}   leader")
    print("-" * 78)
    for a, b in zip(arms[0]["curve"], arms[1]["curve"]):
        if a["step"] == 0:
            tag = "tied (identical init)"
        elif min(a["kl_mean"], b["kl_mean"]) <= floor:
            tag = "-- noise floor --"
        else:
            tag = "categorical" if b["kl_mean"] < a["kl_mean"] else "scalar"
        print(f"{a['step']:>7}  {a['kl_mean']:>11.5e} +/-{a['kl_std']:>7.1e}  "
              f"{b['kl_mean']:>11.5e} +/-{b['kl_std']:>7.1e}   {tag}")
    print("=" * 78)

    w = 13
    print(f"\n{'arm':<12} {'final KL(p||R/Z)':>{w + 12}} {'|log Z err|':>{w + 9}} "
          f"{'grad norm':>{w + 8}}")
    print(f"{'':<12} {'mean +/- std':>{w + 12}} {'mean +/- std':>{w + 9}} "
          f"{'mean / p99':>{w + 8}}")
    print("-" * 78)
    for a in arms:
        print(f"{a['arm']:<12} "
              f"{a['final_kl_mean']:>11.5e} +/- {a['final_kl_std']:<9.2e} "
              f"{a['final_logz_err_mean']:>9.5f} +/- {a['final_logz_err_std']:<8.5f} "
              f"{a['grad_norm_mean']:>8.3f} / {a['grad_norm_p99']:<8.3f}")
    print("-" * 78)

    print(f"on-policy noise floor            KL ~ {floor:.3e}")
    print(f"final checkpoint in that floor?  {summary['final_checkpoint_in_noise_floor']}"
          "   <- if True, the two rows above are NOT a verdict")
    print(f"categorical ahead at             {summary['checkpoints_won_categorical']}"
          f"/{summary['checkpoints_total']} checkpoints")
    print(f"geometric-mean KL ratio          {summary['geomean_kl_ratio']:.3f}x "
          f"(scalar/categorical, over the {summary['geomean_n_checkpoints']} "
          "checkpoints above the floor)")
    print(f"\nwall time {time.time() - t0:.1f}s")

    out = {
        "height": g.h,
        "steps": args.steps,
        "seeds": seeds,
        "arms": arms,
        "log_z_true": log_z_true,
        **summary,
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
