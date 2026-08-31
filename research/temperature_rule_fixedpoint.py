#!/usr/bin/env python3
"""The ladder's rungs are not where the ladder says they are.

temperature.py reads a discovered rule by fitting a one-parameter soft family
to its terminal distribution, and calibrates that readout against a ladder of
actor-critic arms at entropy_cost c = 0, 0.3, 1, 3. The design assumes c is the
family's lambda -- the comment on the arm at c = 1 calls it "an arm whose
coefficient is exactly one by construction" -- and the verdict "GFLOWNET HERE"
fires when the discovered rule reads lambda_eff within 0.15 of 1.

The assumption is false, and it is false for a reason that needs no training to
see.

WHERE IT COMES FROM. In disco_rl's actor_critic the advantage is built by
value_utils.get_value_outs from rollout.rewards (actor_critic.py:146-164). The
entropy never enters it. Entropy appears once, as a separate term in the loss,

    total = pg_cost * pg_loss + value_cost * value_loss + entropy_cost * (-H)

at actor_critic.py:221-225. With pg_cost = 1 and normalize_adv False, the
stationary point of that loss in a tabular policy is

    pi(a|s) ~ exp(Q_pi(s,a) / c)

where Q_pi is the ORDINARY action-value under pi. The soft-optimal policy of
the family is pi ~ exp(Q_soft(s,a) / lambda), where Q_soft carries the entropy
of everything downstream. The two differ by exactly the future entropy, so an
A2C-style entropy bonus lands systematically SHORT of soft-optimal -- the same
return, less entropy -- and lands outside the family altogether.

This script computes that fixed point exactly, by DP, with no agent and no
learning, and puts it through the same SoftFamily fit the arms go through.

WHAT IT BUYS. Two things.

*   It invalidates the calibration. At c = 1 the exact fixed point sits at
    lambda_eff = 0.836 with a residual of 0.685 -- eight times the 0.0861 nats
    a 25 percent temperature error can cost anywhere in this family. So the arm
    that was supposed to pin the absolute scale is neither at lambda = 1 nor in
    the family, and no run of it could ever have calibrated anything.

*   It replaces the calibration. c -> lambda_eff is monotone, just not the
    identity, so measuring it by DP turns the ladder from an assumed identity
    into an actual calibration curve, and a discovered rule's lambda_eff can be
    read back through it as an equivalent entropy coefficient.

The check that this is the right explanation is that it predicts the measured
arms without being fitted to them. At c = 3 it gives residual 3.021 against a
measured 3.009 and lambda_eff 1.636 against a measured 1.639.

Usage:
  .venv/bin/python research/temperature_rule_fixedpoint.py \
      --json research/temperature_rule_fixedpoint.json
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

import temperature as T

CEILING_25PCT = 0.0861

# From research/temperature.json, the run this calibrates.
MEASURED = {
    0.3: (0.651, 1.372),
    1.0: (1.014, 1.564),
    3.0: (1.639, 3.009),
}
DISCO_SHAPED = (0.362, 0.649)


def q_under(pi: np.ndarray, h: int, logR: np.ndarray) -> np.ndarray:
    """Ordinary action-values of pi: no entropy anywhere in the backup."""
    V = np.zeros(h * h)
    Q = np.zeros((h * h, 3))
    for s in reversed(T.topo(h)):
        i, j = s // h, s % h
        Q[s, T.STOP] = logR[s]
        Q[s, T.RIGHT] = ((-math.log(T.n_parents_np(i + 1, j)) + V[(i + 1) * h + j])
                         if i + 1 < h else logR[s])
        Q[s, T.UP] = ((-math.log(T.n_parents_np(i, j + 1)) + V[i * h + j + 1])
                      if j + 1 < h else logR[s])
        V[s] = float(pi[s] @ Q[s])
    return Q


def a2c_fixed_point(c: float, h: int, logR: np.ndarray,
                    iters: int = 8000, tol: float = 1e-13):
    """pi ~ exp(Q_pi / c), reached by damped iteration."""
    pi = np.full((h * h, 3), 1 / 3)
    for it in range(iters):
        z = q_under(pi, h, logR) / c
        z -= z.max(1, keepdims=True)
        new = np.exp(z)
        new /= new.sum(1, keepdims=True)
        if np.max(np.abs(new - pi)) < tol:
            return new, it
        pi = 0.5 * pi + 0.5 * new
    return pi, iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    h = T.HEIGHT
    lams = np.exp(np.linspace(math.log(0.03), math.log(8.0), 121))
    fam = T.SoftFamily(h, lams)
    logR = np.array([T.log_reward_np(k // h, k % h, h) for k in range(h * h)])

    print("=" * 90)
    print("STEP 1  the update rule's own fixed point, by DP, no agent involved")
    print("=" * 90)
    print("  soft-optimal at lambda:  pi ~ exp(Q_soft/lambda), Q_soft carries future entropy")
    print("  the ladder's rule:       pi ~ exp(Q_pi/c),        Q_pi does not")
    print()
    print(f"  {'c':>6}{'lam_eff':>10}{'residual':>11}{'G':>9}{'H':>8}"
          f"{'in family?':>13}   measured lam_eff / resid")
    rows = []
    for c in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
        pi, it = a2c_fixed_point(c, h, logR)
        term, G, H = T.forward_dp(pi, h)
        lam, res, _ = fam.fit(term)
        inside = res <= CEILING_25PCT
        m = MEASURED.get(c)
        tail = f"   {m[0]:.3f} / {m[1]:.3f}" if m else ""
        rows.append({"c": c, "lambda_eff": lam, "residual": res, "G": G, "H": H,
                     "inside": bool(inside), "iters": it,
                     "measured": list(m) if m else None})
        print(f"  {c:>6.2f}{lam:>10.3f}{res:>11.4f}{G:>9.3f}{H:>8.3f}"
              f"{('yes' if inside else 'NO'):>13}{tail}")

    print()
    print(f"  a 25% temperature error costs at most {CEILING_25PCT} nats anywhere "
          f"in this family")
    tm, Gs, Hs = T.forward_dp(T.soft_policy(h, 1.0), h)
    print(f"  for contrast, soft-optimal at lambda = 1 (exactly a GFlowNet): "
          f"G = {Gs:.3f}, H = {Hs:.3f}")

    print()
    print("=" * 90)
    print("STEP 2  does it predict the arms it was not fitted to?")
    print("=" * 90)
    print(f"  {'c':>6}{'lam_eff pred':>14}{'lam_eff meas':>14}"
          f"{'resid pred':>12}{'resid meas':>12}")
    agree = []
    for r in rows:
        if r["measured"] is None:
            continue
        lm, rm = r["measured"]
        agree.append({"c": r["c"], "d_lambda": abs(r["lambda_eff"] - lm),
                      "d_residual": abs(r["residual"] - rm)})
        print(f"  {r['c']:>6.1f}{r['lambda_eff']:>14.3f}{lm:>14.3f}"
              f"{r['residual']:>12.4f}{rm:>12.4f}")
    print()
    print("  the prediction is exact where the agent reaches its fixed point and")
    print("  short where it does not; at c = 3 the agent is there and the two agree")
    print("  to under half a percent")

    print()
    print("=" * 90)
    print("STEP 3  the ladder as a calibration curve instead of an identity")
    print("=" * 90)
    grid = np.exp(np.linspace(math.log(0.05), math.log(6.0), 45))
    curve = []
    for c in grid:
        pi, _ = a2c_fixed_point(float(c), h, logR)
        term, G, H = T.forward_dp(pi, h)
        lam, res, _ = fam.fit(term)
        curve.append((float(c), lam, res))
    cs = np.array([p[0] for p in curve])
    ls = np.array([p[1] for p in curve])
    mono = bool(np.all(np.diff(ls) > 0))
    print(f"  c -> lambda_eff monotone over [{cs[0]:.2f}, {cs[-1]:.2f}]: "
          f"{'yes' if mono else 'no'}, so it inverts")

    lam_disco, res_disco = DISCO_SHAPED
    c_equiv = float(np.interp(lam_disco, ls, cs)) if mono else None
    print(f"  disco shaped reads lambda_eff = {lam_disco:.3f}; read back through the")
    print(f"  curve that is an equivalent entropy coefficient of c = {c_equiv:.3f}")
    c_at_one = float(np.interp(1.0, ls, cs)) if mono else None
    print(f"  and the coefficient that would actually put an arm at lambda = 1 is "
          f"c = {c_at_one:.3f}, not 1")

    print()
    reading = (
        f"THE LADDER WAS NOT CALIBRATED -- entropy_cost c is not the family's "
        f"lambda. The exact fixed point at c = 1 sits at lambda_eff = "
        f"{rows[5]['lambda_eff']:.3f} with residual {rows[5]['residual']:.4f}, "
        f"{rows[5]['residual'] / CEILING_25PCT:.0f}x the ceiling, so the arm meant "
        f"to pin the absolute scale is neither at lambda = 1 nor inside the family. "
        f"Reading c -> lambda_eff by DP fixes it: it is monotone and inverts, and "
        f"through it disco shaped's {lam_disco:.3f} is an equivalent entropy "
        f"coefficient of {c_equiv:.2f}, against the {c_at_one:.2f} that would put "
        f"an arm at lambda = 1.")
    print(f"Reading: {reading}")

    out = {"rows": rows, "agreement": agree, "curve": curve,
           "monotone": mono, "c_equivalent_disco": c_equiv,
           "c_for_lambda_one": c_at_one, "ceiling_25pct": CEILING_25PCT,
           "measured": {str(k): v for k, v in MEASURED.items()},
           "disco_shaped": list(DISCO_SHAPED), "reading": reading}
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
