#!/usr/bin/env python3
"""Is the family misfit a misfit, or is it the optimisation gap?

research/temperature_family_compare.py fits two one-parameter families to the
terminal distribution a discovered rule converges to -- the soft family, whose
member at coefficient one is exactly a GFlowNet, and the fixed-point family of
the actor-critic's own update, pi ~ exp(Q_pi / c) with the ordinary Q_pi -- and
reports the residual divergence that no coefficient in each removes. It read

    6 seeds:  soft 0.4734    rule 0.2977    ceiling 0.0861

and concluded NEITHER FAMILY: the rule's distribution is not a tempered version
of either, so asking what regularisation it implements has no answer of this
shape.

Those are means over final-step admissible rows, and the per-seed numbers behind
them run from 0.041 to 1.423. That range is not noise. This script checks
whether it is the thing the verdict was about at all, by regressing each seed's
residual on how far that seed got from the return this MDP allows.

If the residual is a family misfit it should not care how well the agent
optimised. If it is the optimisation gap wearing a family's clothes, it should
track it, and should vanish into the ceiling as the gap goes to zero.

The answer is the second, so the verdict is withdrawn. Reading the same rows the
other way: once the rule actually optimises, its distribution is within a
25 percent temperature error of the soft family, at a coefficient near 0.2 --
about five times colder than the coefficient that would make it a GFlowNet.

Usage:
  .venv/bin/python research/temperature_gap_confound.py
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent

# The best undiscounted return available in the shaped reduction on this
# landscape, from research/temperature.py's own best_return.
BEST_RETURN = 2.2847

# The most a 25 percent temperature error can cost anywhere in the family,
# measured by research/temperature_residual_scale.py.
CEILING = 0.0861


def final_rows(path: pathlib.Path):
    """Final-step rows, which is what the stored summary averages."""
    doc = json.loads(path.read_text())
    rows = doc["rows"]
    last = max(r["step"] for r in rows)
    return doc, sorted((r for r in rows if r["step"] == last), key=lambda r: r["seed"])


def fit(gap, res):
    slope, intercept = np.polyfit(gap, res, 1)
    pred = slope * gap + intercept
    ss_res = float(((res - pred) ** 2).sum())
    ss_tot = float(((res - res.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2, float(np.corrcoef(gap, res)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str,
                    default="temperature_family_compare_6seed.json")
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    doc, rows = final_rows(HERE / args.source)
    print("=" * 80)
    print(f"STEP 1  the per-seed rows behind the stored summary  ({args.source})")
    print("=" * 80)
    print(f"  stored:  soft {doc['soft_residual']:.4f}   rule {doc['rule_residual']:.4f}"
          f"   ceiling {doc['ceiling_25pct']}")
    print(f"\n  {'seed':>5}{'adm':>5}{'return':>9}{'gap':>8}{'soft lam':>10}"
          f"{'soft res':>10}{'rule res':>10}   inside the ceiling")
    for r in rows:
        gap = BEST_RETURN - r["return"]
        inside = [n for n, v in (("soft", r["soft_residual"]),
                                 ("rule", r["rule_residual"])) if v <= CEILING]
        print(f"  {r['seed']:>5}{str(bool(r['admissible']))[0]:>5}{r['return']:>9.3f}"
              f"{gap:>8.3f}{r['soft_lambda']:>10.4f}{r['soft_residual']:>10.4f}"
              f"{r['rule_residual']:>10.4f}   {' and '.join(inside) if inside else 'neither'}")

    adm = [r for r in rows if r["admissible"]]
    gap = np.array([BEST_RETURN - r["return"] for r in adm])
    soft = np.array([r["soft_residual"] for r in adm])
    rule = np.array([r["rule_residual"] for r in adm])
    seeds = [r["seed"] for r in adm]

    print("\n" + "=" * 80)
    print("STEP 2  regress the residual on the optimisation gap")
    print("=" * 80)
    out = {"source": args.source, "best_return": BEST_RETURN, "ceiling": CEILING,
           "per_seed": [{"seed": r["seed"], "admissible": bool(r["admissible"]),
                         "gap": BEST_RETURN - r["return"],
                         "soft_lambda": r["soft_lambda"],
                         "soft_residual": r["soft_residual"],
                         "rule_residual": r["rule_residual"]} for r in rows]}
    for name, res in (("soft", soft), ("rule", rule)):
        s, b, r2, rho = fit(gap, res)
        print(f"  {name:>5} residual = {s:6.3f} * gap {b:+.4f}    "
              f"pearson {rho:+.3f}   R^2 {r2:.3f}")
        out[f"{name}_fit"] = {"slope": s, "intercept": b, "r2": r2, "pearson": rho}

    print(f"\n  leave-one-out on the soft fit, so that no single seed carries it")
    print(f"  {'dropped':>8}{'n':>4}{'pearson':>10}{'slope':>9}{'intercept':>11}"
          f"{'R^2':>8}   intercept vs ceiling")
    loo = []
    for drop in [None] + seeds:
        m = np.array([True] * len(seeds)) if drop is None else np.array(
            [s != drop for s in seeds])
        if m.sum() < 3:
            continue
        s, b, r2, rho = fit(gap[m], soft[m])
        loo.append({"dropped": drop, "n": int(m.sum()), "slope": s,
                    "intercept": b, "r2": r2, "pearson": rho})
        print(f"  {str(drop):>8}{m.sum():>4}{rho:>10.3f}{s:>9.3f}{b:>11.4f}{r2:>8.3f}"
              f"   {'inside' if b <= CEILING else 'OUTSIDE'}")
    out["leave_one_out"] = loo

    print("\n" + "=" * 80)
    print("STEP 3  reading")
    print("=" * 80)
    every_inside = all(x["intercept"] <= CEILING for x in loo)
    strong = all(x["pearson"] > 0.9 for x in loo)
    best = min(adm, key=lambda r: BEST_RETURN - r["return"])
    print(f"  the best optimising admissible seed is {best['seed']}, gap "
          f"{BEST_RETURN - best['return']:.3f}, soft residual "
          f"{best['soft_residual']:.4f}, rule residual {best['rule_residual']:.4f}")
    print(f"  its two residuals differ by "
          f"{abs(best['soft_residual'] - best['rule_residual']):.4f} nats, so this MDP "
          f"does not separate the two families")
    print(f"  every leave-one-out intercept inside the ceiling: {every_inside}")
    print(f"  every leave-one-out correlation above 0.9:        {strong}")

    if strong and every_inside:
        reading = (
            "THE MISFIT IS THE OPTIMISATION GAP -- the residual the NEITHER FAMILY "
            "verdict rested on is predicted by how far the seed got from the "
            f"achievable return, pearson {out['soft_fit']['pearson']:+.3f} and R^2 "
            f"{out['soft_fit']['r2']:.3f}, and extrapolates to "
            f"{out['soft_fit']['intercept']:.4f} nats at zero gap, inside the "
            f"{CEILING} ceiling. That verdict is withdrawn. Read the other way: once "
            "the rule optimises, its terminal distribution is within a 25 percent "
            f"temperature error of the soft family at a coefficient near "
            f"{best['soft_lambda']:.2f}, about five times colder than the "
            "coefficient that would make it a GFlowNet."
        )
    elif strong:
        reading = (
            "CONFOUNDED BUT NOT EXPLAINED -- the residual tracks the optimisation "
            "gap, so the family verdict cannot be read from these runs, but the fit "
            "does not extrapolate inside the ceiling and a misfit at zero gap is not "
            "excluded."
        )
    else:
        reading = (
            "NO CONFOUND FOUND -- the residual does not track the optimisation gap, "
            "so the family verdict stands as reported."
        )
    print(f"\nReading: {reading}")
    out["reading"] = reading

    if args.json:
        (HERE / args.json).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
