#!/usr/bin/env python3
"""Does DiscoRL's y-prediction track a flow, or a value?

The beta probe (research/disco_probe.py) reports a ratio, |beta/alpha| = 0.32,
and a ratio is only meaningful if the quantity being differentiated means
something.  phi = y_net(softmax(y)) is DiscoRL's own scalar readout, but nothing
so far establishes that it tracks a log-flow at all.  If it does not, the ratio
compares two arbitrary Jacobian entries and says nothing.

On this hypergrid both candidate targets are computable in closed form:

    log F(s)   the GFlowNet state flow for reward R and a backward policy
               uniform over parents, from the edge-flow recursion
               F(s) = R(s) + sum_c F(c) / |parents(c)|

    V_pi(s)    the on-policy value of the agent that Disco103 actually trained,
               by dynamic programming over the empirical action distribution

    V_star(s)  the optimal value, max reward reachable from s

So regress the readout on each and compare.  Whichever explains more variance is
what the discovered prediction is really carrying.  A low R^2 against all three
means the probe is measuring noise and the reported ratio must be withdrawn.

The agent's torso is recurrent, so y depends on history rather than on the state
alone.  We therefore bucket every visit to a state and average -- the resulting
phi_bar(s) is the state-conditional mean, which is the right object to compare
against a state function.

Usage:
  .venv/bin/python research/calibrate.py --json research/calibrate.json
"""

from __future__ import annotations

import argparse
import json
import math

import jax
import jax.numpy as jnp
import numpy as np

import disco_probe
import hypergrid_env


def exact_log_flow(h: int) -> np.ndarray:
    """log F(s) for p_B uniform over parents.

    Edge flows, not the naive tree recursion: on a DAG a state has several
    parents and F(s -> c) = p_B(s|c) F(c), so the child's flow is split among
    its parents rather than claimed whole by each.
    """
    f = np.zeros((h, h))
    for s in range(2 * h - 2, -1, -1):
        for i in range(h):
            j = s - i
            if not (0 <= j < h):
                continue
            acc = float(hypergrid_env.grid_reward(i, j, h))
            for di, dj in ((1, 0), (0, 1)):
                ci, cj = i + di, j + dj
                if ci < h and cj < h:
                    n_par = (1 if ci > 0 else 0) + (1 if cj > 0 else 0)
                    acc += f[ci, cj] / n_par
            f[i, j] = acc
    return np.log(f)


def value_of_policy(pi: np.ndarray, h: int) -> np.ndarray:
    """V_pi(s) by DP. pi[i, j] = (p_right, p_up, p_stop), already renormalised.

    Unavailable moves fold into STOP, matching hypergrid_env.step.
    """
    v = np.zeros((h, h))
    for s in range(2 * h - 2, -1, -1):
        for i in range(h):
            j = s - i
            if not (0 <= j < h):
                continue
            pr, pu, ps = pi[i, j]
            can_r, can_u = i + 1 < h, j + 1 < h
            ps_eff = ps + (0.0 if can_r else pr) + (0.0 if can_u else pu)
            acc = ps_eff * float(hypergrid_env.grid_reward(i, j, h))
            if can_r:
                acc += pr * v[i + 1, j]
            if can_u:
                acc += pu * v[i, j + 1]
            v[i, j] = acc
    return v


def optimal_value(h: int) -> np.ndarray:
    v = np.zeros((h, h))
    for s in range(2 * h - 2, -1, -1):
        for i in range(h):
            j = s - i
            if not (0 <= j < h):
                continue
            best = float(hypergrid_env.grid_reward(i, j, h))
            for di, dj in ((1, 0), (0, 1)):
                ci, cj = i + di, j + dj
                if ci < h and cj < h:
                    best = max(best, v[ci, cj])
            v[i, j] = best
    return v


def fit(x: np.ndarray, y: np.ndarray) -> dict:
    """R^2 of an ordinary least-squares line, plus Spearman rank correlation.

    Rank correlation is reported alongside R^2 because the readout is only ever
    claimed to be monotone in the target, not affine in it.
    """
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 4 or np.std(x) < 1e-12:
        return {"n": int(x.size), "r2": float("nan"), "spearman": float("nan"), "slope": float("nan")}
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r2 = 1.0 - resid.var() / y.var() if y.var() > 0 else float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    sp = float(np.corrcoef(rx, ry)[0, 1])
    return {"n": int(x.size), "r2": float(r2), "spearman": sp, "slope": float(slope)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rollout-len", type=int, default=18)
    ap.add_argument("--train-steps", type=int, default=400)
    ap.add_argument("--collect", type=int, default=40, help="rollouts gathered after training")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    h = args.height
    rng = jax.random.PRNGKey(0)
    env, ag = disco_probe.build(h, args.batch, 0)
    disco = disco_probe.unflatten_params(np.load(disco_probe.WEIGHTS))

    print(f"hypergrid H={h}  batch={args.batch}  training {args.train_steps} steps with Disco103")
    rollout, learner_state, actor_state = disco_probe.train(
        env, ag, disco, args.rollout_len, args.train_steps, rng
    )

    phi_y = disco_probe.readout_fn(disco, "y")
    phi_z = disco_probe.readout_fn(disco, "z")

    n = h * h
    sum_y = np.zeros(n)
    sum_z = np.zeros(n)
    sum_q = np.zeros(n)
    cnt = np.zeros(n)
    act_cnt = np.zeros((n, 3))

    key = rng
    for _ in range(args.collect):
        key, k = jax.random.split(key)
        roll, _, _ = disco_probe.collect_rollout(env, ag, args.rollout_len, k)
        out, _ = ag.unroll_net(learner_state.params, actor_state, roll)
        # The observation is a one-hot board, so argmax recovers the state index.
        obs = np.asarray(roll.observations)
        idx = obs.reshape(obs.shape[0], obs.shape[1], -1).argmax(-1)  # [T, B]
        acts = np.asarray(roll.actions)
        py = np.asarray(phi_y(out["y"]))  # [T, B]
        # z is action-conditional; take the action actually played.
        pz_all = np.asarray(phi_z(out["z"]))  # [T, B, A]
        pz = np.take_along_axis(pz_all, acts[..., None], axis=-1)[..., 0]
        # The q head is a categorical distribution over a fixed return support,
        # so its expected bin index is monotone in the decoded value. That makes
        # it the natural positive control: whatever else is true, q must track a
        # value function.
        qs = np.asarray(jax.nn.softmax(out["q"], axis=-1))  # [T, B, A, bins]
        bins = np.arange(qs.shape[-1], dtype=np.float64)
        q_val = (qs * bins).sum(-1)  # [T, B, A]
        pq = np.take_along_axis(q_val, acts[..., None], axis=-1)[..., 0]
        for t in range(idx.shape[0]):
            np.add.at(sum_y, idx[t], py[t])
            np.add.at(sum_z, idx[t], pz[t])
            np.add.at(sum_q, idx[t], pq[t])
            np.add.at(cnt, idx[t], 1.0)
            np.add.at(act_cnt, (idx[t], acts[t]), 1.0)

    seen = cnt > 0
    phi_bar = np.where(seen, sum_y / np.maximum(cnt, 1), np.nan).reshape(h, h)
    phiz_bar = np.where(seen, sum_z / np.maximum(cnt, 1), np.nan).reshape(h, h)
    phiq_bar = np.where(seen, sum_q / np.maximum(cnt, 1), np.nan).reshape(h, h)

    pi_hat = act_cnt / np.maximum(act_cnt.sum(1, keepdims=True), 1)
    pi_hat = np.where(cnt[:, None] > 0, pi_hat, 1.0 / 3.0).reshape(h, h, 3)

    log_f = exact_log_flow(h)
    v_pi = value_of_policy(pi_hat, h)
    v_star = optimal_value(h)
    log_v_pi = np.log(np.maximum(v_pi, 1e-12))

    targets = {
        "log_F": log_f,
        "V_pi": v_pi,
        "log_V_pi": log_v_pi,
        "V_star": v_star,
    }

    print(f"\nstates visited: {int(seen.sum())}/{n}   rollouts: {args.collect}\n")
    hdr = f"{'readout ~ target':<26}{'n':>5}{'R^2':>10}{'Spearman':>11}{'slope':>10}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for name, arr in targets.items():
        for rname, r in (("phi(y)", phi_bar), ("phi(z)", phiz_bar), ("q head", phiq_bar)):
            res = fit(arr.reshape(-1), r.reshape(-1))
            res["target"] = name
            res["readout"] = rname
            rows.append(res)
            print(
                f"{rname + ' ~ ' + name:<26}{res['n']:>5}{res['r2']:>10.4f}"
                f"{res['spearman']:>11.4f}{res['slope']:>10.4f}"
            )

    y_rows = [r for r in rows if r["readout"] == "phi(y)"]
    best = max(y_rows, key=lambda r: (r["r2"] if np.isfinite(r["r2"]) else -9))
    flow = next(r for r in y_rows if r["target"] == "log_F")
    val = max(
        (r for r in y_rows if r["target"] in ("V_pi", "log_V_pi", "V_star")),
        key=lambda r: (r["r2"] if np.isfinite(r["r2"]) else -9),
    )
    print(f"\nbest explanation of phi(y): {best['target']}  (R^2 = {best['r2']:.4f})")
    print(f"  flow hypothesis  log_F : R^2 = {flow['r2']:.4f}, Spearman = {flow['spearman']:.4f}")
    print(f"  value hypothesis {val['target']:<6}: R^2 = {val['r2']:.4f}, Spearman = {val['spearman']:.4f}")

    if max(flow["r2"], val["r2"]) < 0.15:
        verdict = "NEITHER -- the readout tracks no state function; the beta ratio is uninterpretable"
    elif flow["r2"] > val["r2"] + 0.1:
        verdict = "FLOW -- phi(y) tracks log F better than any value function"
    elif val["r2"] > flow["r2"] + 0.1:
        verdict = "VALUE -- phi(y) tracks a value function, not a flow"
    else:
        verdict = "AMBIGUOUS -- log F and value are not separable on this environment"
    print(f"Verdict: {verdict}")

    # Positive control: whatever y is doing, the categorical q head is a value
    # head by construction, so it must track a value function. If it does not,
    # the bucketing or the training run is broken and nothing above is readable.
    q_rows = [r for r in rows if r["readout"] == "q head"]
    q_val = max(
        (r for r in q_rows if r["target"] in ("V_pi", "log_V_pi", "V_star")),
        key=lambda r: (r["r2"] if np.isfinite(r["r2"]) else -9),
    )
    q_flow = next(r for r in q_rows if r["target"] == "log_F")
    print(
        f"Positive control (q head): value R^2 = {q_val['r2']:.4f} ({q_val['target']}), "
        f"log_F R^2 = {q_flow['r2']:.4f}"
    )
    out_control = {"q_value_r2": q_val["r2"], "q_value_target": q_val["target"], "q_logF_r2": q_flow["r2"]}

    out_json = {
        "height": h,
        "train_steps": args.train_steps,
        "collect": args.collect,
        "states_visited": int(seen.sum()),
        "fits": rows,
        "verdict": verdict,
        "phi_y_logF": {k: flow[k] for k in ("r2", "spearman", "n")},
        "phi_y_value": {"target": val["target"], **{k: val[k] for k in ("r2", "spearman", "n")}},
        "control": out_control,
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out_json, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
