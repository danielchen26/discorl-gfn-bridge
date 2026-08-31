#!/usr/bin/env python3
"""How many nats is a wrong temperature actually worth?

research/temperature.py fits a one-parameter soft family to an agent's terminal
distribution and reports two numbers: the coefficient of the nearest member,
and the residual divergence that no coefficient removes. It then calls the
agent OUTSIDE THE FAMILY when that residual exceeds 0.4, and the run it
committed reads

    disco shaped   lambda_eff = 0.362   residual = 0.649

so the verdict turned on 0.649 > 0.4. That threshold was picked, not measured,
and a threshold picked in nats means nothing until the family's own scale in
nats is known. This script measures it.

THE QUESTION. If an agent were exactly a member of the family but at the wrong
coefficient, how large a residual could that produce? None, by construction --
the fit would find its coefficient and the residual would be zero. So the
honest version is the local one: how much divergence does a given fractional
error in the coefficient buy? That fixes the scale on which a residual is large
or small, and it is a property of the family alone, with no agent in it.

Two readings come out:

*   CEILING. The largest divergence a pure temperature error of a given size
    can produce, maximised over where in the family it happens. A residual
    above the ceiling for delta cannot be explained by a temperature error of
    delta anywhere.

*   DIAMETER. The largest divergence between any two members at all. A residual
    above the diameter means the agent is further from the family than the
    family's two most distant members are from each other -- at which point
    "its effective temperature is X" is a projection onto a set the agent is
    nowhere near, and reporting X as a temperature is the error the residual
    exists to catch.

Both are computed from the same SoftFamily and the same divergence temperature.py
fits with, imported rather than restated.

Usage:
  .venv/bin/python research/temperature_residual_scale.py \
      --json research/temperature_residual_scale.json
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

import temperature as T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", type=str, default="research/temperature.json",
                    help="run whose residuals are put on this scale")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    h = T.HEIGHT
    lams = np.exp(np.linspace(math.log(0.03), math.log(8.0), 121))
    fam = T.SoftFamily(h, lams)

    def member(lam: float) -> np.ndarray:
        q = T.forward_dp(T.soft_policy(h, lam), h)[0] + 1e-12
        return q / q.sum()

    def div(p: np.ndarray, q: np.ndarray) -> float:
        m = p > 1e-15
        return float(np.sum(p[m] * (np.log(p[m]) - np.log(q[m]))))

    # A coarse but wide grid: the ceiling is a max over the family, so it wants
    # coverage, not resolution.
    scan = np.exp(np.linspace(math.log(0.05), math.log(6.0), 40))
    cache = {float(l): member(float(l)) for l in scan}

    print("=" * 78)
    print("STEP 1  what a pure temperature error costs, per coefficient")
    print("=" * 78)
    deltas = (0.10, 0.25, 0.50, 1.00)
    print(f"  {'lambda':>8}" + "".join(f"{f'+{int(d*100)}%':>12}" for d in deltas)
          + f"{'H(member)':>12}")
    rows = []
    for lam in (0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 3.0):
        p = member(lam)
        ent = float(-(p[p > 1e-15] * np.log(p[p > 1e-15])).sum())
        costs = [div(member(lam * (1 + d)), p) for d in deltas]
        rows.append({"lam": lam, "costs": costs, "entropy": ent})
        print(f"  {lam:>8.2f}" + "".join(f"{c:>12.5f}" for c in costs)
              + f"{ent:>12.3f}")

    ceiling = {}
    for d in deltas:
        worst, at = 0.0, None
        for lam in scan:
            lam = float(lam)
            hi = lam * (1 + d)
            if hi > lams[-1]:
                continue
            c = div(member(hi), cache[lam])
            if c > worst:
                worst, at = c, lam
        ceiling[d] = (worst, at)
    print()
    for d in deltas:
        w, at = ceiling[d]
        print(f"  ceiling for a {int(d*100):>3}% temperature error: {w:.4f} nats "
              f"(worst at lambda = {at:.2f})")

    # Diameter. The divergence is not symmetric, so take the max over both
    # orders; the fit uses D(agent || member), which is this orientation.
    worst, pair = 0.0, None
    for a in scan:
        pa = cache[float(a)]
        for b in scan:
            if a is b:
                continue
            c = div(pa, cache[float(b)])
            if c > worst:
                worst, pair = c, (float(a), float(b))
    print(f"  diameter of the whole family:                {worst:.4f} nats "
          f"(lambda {pair[0]:.2f} against {pair[1]:.2f})")

    print()
    print("=" * 78)
    print("STEP 2  the measured residuals on that scale")
    print("=" * 78)
    try:
        run = json.load(open(args.read))
    except OSError:
        print(f"  {args.read} not readable; scale reported without a run")
        run = None

    placed = []
    if run is not None:
        print(f"  {'arm':<24}{'lam_eff':>9}{'residual':>10}   reading")
        for name, s in run["summary"].items():
            res = s.get("residual_mean")
            if res is None:
                continue
            lam = s.get("lambda_eff_mean")
            if res > worst:
                note = (f"beyond the family's diameter by "
                        f"{res / worst:.1f}x -- not a temperature")
            else:
                over = [d for d in deltas if res > ceiling[d][0]]
                note = (f"more than a {int(max(over)*100)}% temperature error can "
                        f"cost anywhere" if over
                        else f"within a {int(min(deltas)*100)}% temperature error")
            placed.append({"arm": name, "lambda_eff": lam, "residual": res,
                           "reading": note})
            print(f"  {name:<24}{lam:>9.3f}{res:>10.4f}   {note}")

    print()
    inside = [p for p in placed if p["residual"] <= ceiling[0.25][0]]
    print(f"  arms whose residual a 25% temperature error could account for: "
          f"{len(inside)}/{len(placed)}")
    if placed and not inside:
        print("  so no arm in that run is described by any coefficient, and every")
        print("  lambda_eff it reports is a projection rather than a temperature")

    out = {"ceiling": {str(d): {"nats": ceiling[d][0], "at_lambda": ceiling[d][1]}
                       for d in deltas},
           "diameter": {"nats": worst, "lambdas": pair},
           "per_lambda": rows, "placed": placed, "read": args.read}
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
