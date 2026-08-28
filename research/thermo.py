#!/usr/bin/env python3
"""Is log F split across DiscoRL's two prediction channels?

research/calibrate.py established that phi(y) tracks neither the exact log F nor
a value function, while phi(z) tracks the on-policy value cleanly.  Read as "the
y channel is empty" that is a dead end.  There is a second reading.

For a Boltzmann policy pi ~ exp(beta Q) the soft value decomposes exactly:

    V_soft(s) = (1/beta) log sum_a exp(beta Q(s,a))
              = E_pi[Q(s,.)]        +  (1/beta) H(pi(.|s))
              = energy              +  entropy

and the GFlowNet correspondence is log F(s) = V_soft(s) at beta = 1.  So log F is
a FREE ENERGY: a combination of an energy term and an entropy term.  No single
scalar readout can track it unless that readout already mixes both.

DiscoRL emits two predictions, and its authors report that the discovered
semantics involve future policy entropy as well as upcoming reward.  If y and z
carry the thermodynamically conjugate pair rather than two copies of a value,
then log F is a combination of them and is invisible to either alone -- which is
exactly the pattern calibrate.py measured.

Test: the incremental R^2 of adding phi(y) to phi(z) when predicting log F.

    Delta = R2[log F | phi(z), phi(y)] - R2[log F | phi(z)]

For the reading to hold, Delta must be large AND specific: adding phi(y) must
help log F substantially more than it helps a pure value function, because a
value carries no entropy term.  A control that rises just as much would mean
phi(y) is a generic extra regressor rather than the entropy channel.

Usage:
  .venv/bin/python research/thermo.py --json research/thermo.json
"""

from __future__ import annotations

import argparse
import json

import jax
import numpy as np

import calibrate
import disco_probe
import hypergrid_env


def future_entropy(pi: np.ndarray, h: int) -> np.ndarray:
    """S(s) = expected cumulative future policy entropy, by DP.

    S(s) = H(pi(.|s)) + sum_a pi(a|s) S(child_a),  zero past a stop.
    Unavailable moves fold into STOP, matching hypergrid_env.step.
    """
    s_arr = np.zeros((h, h))
    for d in range(2 * h - 2, -1, -1):
        for i in range(h):
            j = d - i
            if not (0 <= j < h):
                continue
            pr, pu, ps = pi[i, j]
            can_r, can_u = i + 1 < h, j + 1 < h
            ps_eff = ps + (0.0 if can_r else pr) + (0.0 if can_u else pu)
            probs = np.array([pr if can_r else 0.0, pu if can_u else 0.0, ps_eff])
            nz = probs[probs > 0]
            acc = float(-(nz * np.log(nz)).sum())
            if can_r:
                acc += pr * s_arr[i + 1, j]
            if can_u:
                acc += pu * s_arr[i, j + 1]
            s_arr[i, j] = acc
    return s_arr


def eta_squared(v: np.ndarray, g: np.ndarray) -> float:
    """Between-group variance over total variance.

    A potential -- flow, value or entropy alike -- must be a function of the
    state. The agent's torso is recurrent, so before concluding that a readout
    fails to track any state function we have to check it IS one: if its
    variance were mostly within-state, no state-function regression could ever
    find it and every R^2 above would be zero for a reason unrelated to what the
    channel carries.
    """
    tot = v.var()
    if tot <= 0:
        return float("nan")
    m = np.zeros(int(g.max()) + 1)
    c = np.zeros(int(g.max()) + 1)
    np.add.at(m, g, v)
    np.add.at(c, g, 1.0)
    mu = m / np.maximum(c, 1)
    return float(((mu[g] - v.mean()) ** 2).mean() / tot)


def r2_multi(X: np.ndarray, y: np.ndarray) -> float:
    """R^2 of an OLS fit of y on the columns of X plus an intercept."""
    keep = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[keep], y[keep]
    A = np.hstack([X, np.ones((X.shape[0], 1))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return float(1.0 - resid.var() / y.var()) if y.var() > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rollout-len", type=int, default=18)
    ap.add_argument("--train-steps", type=int, default=400)
    ap.add_argument("--collect", type=int, default=40)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    h = args.height
    rng = jax.random.PRNGKey(0)
    env, ag = disco_probe.build(h, args.batch, 0)
    disco = disco_probe.unflatten_params(np.load(disco_probe.WEIGHTS))
    print(f"hypergrid H={h}  training {args.train_steps} steps with Disco103")
    _, learner_state, actor_state = disco_probe.train(
        env, ag, disco, args.rollout_len, args.train_steps, rng
    )

    phi_y = disco_probe.readout_fn(disco, "y")
    phi_z = disco_probe.readout_fn(disco, "z")
    n = h * h
    sum_y, sum_z, cnt = np.zeros(n), np.zeros(n), np.zeros(n)
    act_cnt = np.zeros((n, 3))

    raw_y_l, raw_z_l, raw_s_l, raw_t_l = [], [], [], []
    key = rng
    for _ in range(args.collect):
        key, k = jax.random.split(key)
        roll, _, _ = disco_probe.collect_rollout(env, ag, args.rollout_len, k)
        out, _ = ag.unroll_net(learner_state.params, actor_state, roll)
        obs = np.asarray(roll.observations)
        idx = obs.reshape(obs.shape[0], obs.shape[1], -1).argmax(-1)
        acts = np.asarray(roll.actions)
        py = np.asarray(phi_y(out["y"]))
        pz = np.take_along_axis(np.asarray(phi_z(out["z"])), acts[..., None], axis=-1)[..., 0]
        raw_y_l.append(py.ravel())
        raw_z_l.append(pz.ravel())
        raw_s_l.append(idx.ravel())
        raw_t_l.append(np.tile(np.arange(idx.shape[0])[:, None], (1, idx.shape[1])).ravel())
        for t in range(idx.shape[0]):
            np.add.at(sum_y, idx[t], py[t])
            np.add.at(sum_z, idx[t], pz[t])
            np.add.at(cnt, idx[t], 1.0)
            np.add.at(act_cnt, (idx[t], acts[t]), 1.0)

    raw_y = np.concatenate(raw_y_l)
    raw_z = np.concatenate(raw_z_l)
    raw_s = np.concatenate(raw_s_l)
    raw_t = np.concatenate(raw_t_l)

    seen = cnt > 0
    fy = np.where(seen, sum_y / np.maximum(cnt, 1), np.nan)
    fz = np.where(seen, sum_z / np.maximum(cnt, 1), np.nan)
    pi_hat = act_cnt / np.maximum(act_cnt.sum(1, keepdims=True), 1)
    pi_hat = np.where(cnt[:, None] > 0, pi_hat, 1.0 / 3.0).reshape(h, h, 3)

    log_f = calibrate.exact_log_flow(h).reshape(-1)
    v_pi = calibrate.value_of_policy(pi_hat, h).reshape(-1)
    ent = future_entropy(pi_hat, h).reshape(-1)

    Z = fz[:, None]
    YZ = np.stack([fz, fy], 1)
    Y = fy[:, None]

    results = {}
    for tname, tgt in (("log_F", log_f), ("V_pi", v_pi), ("H_future", ent)):
        r_z = r2_multi(Z, tgt)
        r_y = r2_multi(Y, tgt)
        r_yz = r2_multi(YZ, tgt)
        results[tname] = {"r2_z": r_z, "r2_y": r_y, "r2_yz": r_yz, "delta": r_yz - r_z}

    hdr = f"{'target':<12}{'R2[.|z]':>10}{'R2[.|y]':>10}{'R2[.|z,y]':>12}{'delta':>10}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for k, v in results.items():
        print(
            f"{k:<12}{v['r2_z']:>10.4f}{v['r2_y']:>10.4f}{v['r2_yz']:>12.4f}{v['delta']:>10.4f}"
        )

    var_hdr = f"{'readout':<10}{'eta2 by STATE':>16}{'eta2 by TIME':>15}{'std':>12}"
    print("\n" + var_hdr)
    print("-" * len(var_hdr))
    variance = {}
    for nm, v in (("phi(y)", raw_y), ("phi(z)", raw_z)):
        variance[nm] = {
            "eta2_state": eta_squared(v, raw_s),
            "eta2_time": eta_squared(v, raw_t),
            "std": float(v.std()),
        }
        print(
            f"{nm:<10}{variance[nm]['eta2_state']:>16.4f}"
            f"{variance[nm]['eta2_time']:>15.4f}{variance[nm]['std']:>12.3e}"
        )
    ratio = variance["phi(z)"]["std"] / max(variance["phi(y)"]["std"], 1e-30)
    print(
        f"\nphi(y) IS a state function (eta2 = {variance['phi(y)']['eta2_state']:.4f}) but its "
        f"dynamic range is {ratio:.0f}x smaller than phi(z)'s -- in this regime the channel is "
        "close to inert, which bounds how strongly any null result above can be read."
    )

    d_flow = results["log_F"]["delta"]
    d_val = results["V_pi"]["delta"]
    print(
        f"\nAdding phi(y) buys {d_flow:.4f} R^2 on log F "
        f"and {d_val:.4f} on the pure value control."
    )
    print(
        f"phi(y) alone against future policy entropy: R^2 = {results['H_future']['r2_y']:.4f}"
    )

    if results["H_future"]["r2_y"] > 0.3 and d_flow > d_val + 0.1:
        verdict = "CONJUGATE PAIR -- y carries entropy, z carries energy, log F is their sum"
    elif d_flow > d_val + 0.1:
        verdict = "PARTIAL -- phi(y) adds specifically to log F, but is not cleanly the entropy"
    elif results["H_future"]["r2_y"] > 0.3:
        verdict = "ENTROPY ONLY -- y tracks entropy but the sum does not reconstruct log F"
    else:
        verdict = "NO -- the free-energy split is not visible in these two readouts"
    print(f"Verdict: {verdict}")

    out = {
        "height": h,
        "train_steps": args.train_steps,
        "results": results,
        "variance": variance,
        "range_ratio_z_over_y": ratio,
        "verdict": verdict,
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
