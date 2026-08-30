#!/usr/bin/env python3
"""Does the discovered rule fail because of the reduction, or because it needs shaping?

research/temperature.py runs a wiring control -- the discovered rule on the
grid with the log P_B payments removed -- and VOIDs the entire measurement when
that control ends more than one unit of return short of the best available. It
did: 1.636 short, averaged over three seeds, two of which stalled at 1.333.

The verdict it prints from that is

    VOID -- the discovered rule does not reach the best return even with the
    log P_B payments removed, so the failure is in how this environment was
    built and not in the reduction

and it fires ahead of every other branch, so it decides the experiment.

But the control changes two things at once. It removes the log P_B payments
AND the geometric shaping, while the arm it is meant to exonerate runs with
the shaping on -- and that arm, "disco shaped", ends 0.053 from the best
available return, i.e. it very nearly solves the reduction. Of the four cells
in (payments x shaping) the script runs three and omits the one that separates
the two explanations:

    payments  shaping   arm                    gap from best
    on        on        disco shaped           0.053     measured
    on        off       disco unshaped         measured by the hostile arm
    off       off       CONTROL no-pb disco    1.636     measured
    off       on        --  not run  --        this script

The shaping is a potential, so it is policy-invariant at the optimum by
construction and temperature.py checks that claim numerically (the soft-optimal
policy moves by 1.8e-15 at every coefficient tried). It cannot change what the
best return is; it only changes how dense the signal is on the way there.

So the missing cell is a clean read:

*   If it reaches the best return, the control's failure is the missing
    shaping. The VOID is then an artifact of a confounded control, the
    reduction is not implicated, and "disco shaped" -- 0.053 from best, fitting
    the soft family at lambda_eff = 0.250 +/- 0.039 -- is the arm to read.

*   If it fails too, the log P_B payments are not the issue either way and the
    rule simply cannot optimise this grid without shaping. The VOID stands, but
    its stated reason -- "how this environment was built" -- becomes precise:
    the reduction is fine, the density of the reward signal is not.

Everything here is temperature.py's own machinery, imported rather than
restated, so the numbers are comparable to the ones it printed.

Usage:
  .venv/bin/python research/temperature_wiring_probe.py \
      --json research/temperature_wiring_probe.json
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

import disco_probe
import temperature as T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rollout-len", type=int, default=16)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--probe-every", type=int, default=1000)
    ap.add_argument("--collect", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    h = T.HEIGHT
    lams = np.exp(np.linspace(math.log(0.03), math.log(8.0), 121))
    fam = T.SoftFamily(h, lams)
    meta_params = disco_probe.unflatten_params(np.load(disco_probe.WEIGHTS))

    best_nopb = T.best_return(h, False)
    best_pb = T.best_return(h, True)
    print("=" * 84)
    print("the missing cell: log P_B payments OFF, geometric shaping ON")
    print("=" * 84)
    print(f"  best return without the payments {best_nopb:.3f}, with them {best_pb:.3f};")
    print(f"  stopping at the source pays {fam.logR[0]:.3f} either way")
    print(f"  the shaping is a potential, so neither number moves when it is on")
    print()
    print("  for reference, from temperature.py's own run:")
    print("    disco shaped          gap 0.053 (best seed) / 0.459 (mean of 3)")
    print("    CONTROL no-pb disco   gap 1.636 (mean of 3), two seeds stalled at 1.333")
    print()

    hdr = (f"  {'arm':<24}{'sd':>3}{'step':>6}{'seen':>5}{'bkt':>9}{'return':>9}"
           f"{'H':>7}{'gap':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    name = "no-pb shaped disco"
    results = []
    for seed in range(args.seeds):
        curve = T.run_arm("disco", 0.0, 0.0, h, args.batch, args.rollout_len,
                          args.steps, args.probe_every, args.collect, seed,
                          meta_params, fam, args.lr, False, True)
        ok, why = T.gate(curve)
        for c in curve:
            mark = "" if c is not curve[-1] else ("   <- " + why)
            print(f"  {name:<24}{seed:>3}{c['step']:>6}{c['states_seen']:>5}"
                  f"{c['bucket_err']:>9.1e}{c['return']:>9.3f}{c['h_traj']:>7.3f}"
                  f"{best_nopb - c['return']:>8.3f}{mark}")
        print()
        results.append({"arm": name, "seed": seed, "admissible": ok, "reason": why,
                        "curve": curve, "final": curve[-1],
                        "gap": best_nopb - curve[-1]["return"]})

    gaps = np.array([r["gap"] for r in results])
    print("=" * 84)
    print(f"  return {np.mean([r['final']['return'] for r in results]):.3f}, "
          f"gap from best {gaps.mean():.3f} +/- {gaps.std():.3f}   "
          f"(best single seed {gaps.min():.3f})")

    # The control's own threshold, applied to the cell it omitted.
    if gaps.mean() <= 1.0:
        reading = (
            f"THE CONTROL WAS CONFOUNDED -- with the payments removed AND the "
            f"shaping on, the rule ends {gaps.mean():.3f} from the best return, "
            f"inside the same threshold the control failed at 1.636. What the "
            f"control measured is the missing shaping, not the log P_B payments, "
            f"so its VOID does not implicate the reduction.")
    else:
        reading = (
            f"THE CONTROL STANDS -- even with the shaping on, the rule ends "
            f"{gaps.mean():.3f} from the best return without the payments, so "
            f"removing them is not what rescues it and the failure is not "
            f"specific to the reduction.")
    print(f"Reading: {reading}")

    out = {"config": vars(args), "best_return_no_pb": best_nopb,
           "best_return_pb": best_pb, "per_run": results,
           "gap_mean": float(gaps.mean()), "gap_std": float(gaps.std()),
           "reading": reading}
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
