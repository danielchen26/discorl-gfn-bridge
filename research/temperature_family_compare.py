#!/usr/bin/env python3
"""If the discovered rule is not a GFlowNet, is it an entropy-bonus actor-critic?

temperature_rule_fixedpoint.py showed that the ladder's own update rule has an
exact fixed point, pi ~ exp(Q_pi/c) with the ordinary Q_pi, and that this is a
different one-parameter family from the soft one: it coincides with the soft
family below c ~ 0.5 and leaves it above, reaching a residual of 3.02 at c = 3.
It reproduces the measured arms it was never fitted to -- at c = 3, residual
3.021 against 3.009 and lambda_eff 1.636 against 1.639.

That gives a second yardstick, and a sharper question than "which temperature".
The soft family answers "is it a GFlowNet". This one answers "is it the thing
the ladder is". Fitting a distribution to both and comparing residuals is a
model comparison rather than a projection, and the loser is informative:

    small residual against the soft family      -> soft RL at that lambda
    small residual against the rule's family    -> an entropy-bonus actor-critic
    large residual against both                 -> neither, and the question of
                                                   "what regularisation does it
                                                   implement" has a negative
                                                   answer with teeth

The suggestive numbers so far: disco shaped fits the soft family at lambda_eff
0.250 (this machine) / 0.362 (the committed run) with residual 0.649, and runs
at a trajectory entropy near 0.35, while the rule's family at c ~ 0.22 carries
lambda_eff 0.21 and entropy 0.30. Close enough to be worth testing and not
close enough to assert.

WHY THIS NEEDS A RUN. temperature.py stores lambda_eff, residual, return and
entropy per probe, but not the 36-vector they were computed from, so the second
fit cannot be done from the JSON. Rather than change temperature.py, this
wraps SoftFamily.fit to record every distribution it is handed, and drives
run_arm unmodified -- so the training, bucketing and DP are the same code that
produced the committed numbers.

Usage:
  .venv/bin/python research/temperature_family_compare.py \
      --json research/temperature_family_compare.json
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

import disco_probe
import temperature as T
import temperature_rule_fixedpoint as F

CEILING_25PCT = 0.0861


class RuleFamily:
    """The fixed points of pi ~ exp(Q_pi/c), as a fittable one-parameter family."""

    def __init__(self, h: int, cs: np.ndarray, logR: np.ndarray):
        self.h = h
        self.cs = cs
        self.logR = logR
        self.table = np.stack(
            [T.forward_dp(F.a2c_fixed_point(float(c), h, logR)[0], h)[0] for c in cs])
        self.safe = self.table + 1e-12
        self.safe /= self.safe.sum(1, keepdims=True)

    def _member(self, c: float) -> np.ndarray:
        q = T.forward_dp(F.a2c_fixed_point(float(c), self.h, self.logR)[0],
                         self.h)[0] + 1e-12
        return q / q.sum()

    @staticmethod
    def _div(p, q, m):
        return float(np.sum(p[m] * (np.log(p[m]) - np.log(q[m]))))

    def fit(self, p: np.ndarray):
        """Same golden-section refinement on log c that SoftFamily.fit uses."""
        m = p > 1e-15
        kls = np.array([self._div(p, q, m) for q in self.safe])
        k = int(np.argmin(kls))
        if k == 0 or k == len(self.cs) - 1:
            return float(self.cs[k]), float(kls[k]), False
        a, b = math.log(self.cs[k - 1]), math.log(self.cs[k + 1])
        inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
        c1, d1 = b - inv_phi * (b - a), a + inv_phi * (b - a)
        fc = self._div(p, self._member(math.exp(c1)), m)
        fd = self._div(p, self._member(math.exp(d1)), m)
        for _ in range(30):
            if fc < fd:
                b, d1, fd = d1, c1, fc
                c1 = b - inv_phi * (b - a)
                fc = self._div(p, self._member(math.exp(c1)), m)
            else:
                a, c1, fc = c1, d1, fd
                d1 = a + inv_phi * (b - a)
                fd = self._div(p, self._member(math.exp(d1)), m)
            if b - a < 1e-7:
                break
        c_hat = math.exp(0.5 * (a + b))
        return c_hat, self._div(p, self._member(c_hat), m), True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rollout-len", type=int, default=16)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--probe-every", type=int, default=2000)
    ap.add_argument("--collect", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    h = T.HEIGHT
    lams = np.exp(np.linspace(math.log(0.03), math.log(8.0), 121))
    fam = T.SoftFamily(h, lams)
    logR = np.array([T.log_reward_np(k // h, k % h, h) for k in range(h * h)])
    meta_params = disco_probe.unflatten_params(np.load(disco_probe.WEIGHTS))

    print("building the rule's family by DP", flush=True)
    rule_fam = RuleFamily(h, np.exp(np.linspace(math.log(0.05), math.log(6.0), 60)),
                          logR)
    print("  done", flush=True)

    # Record every distribution SoftFamily.fit is handed, without touching
    # temperature.py: run_arm calls fam.fit(term) once per probe.
    seen = []
    original_fit = fam.fit
    fam.fit = lambda p: (seen.append(np.array(p)), original_fit(p))[1]

    print()
    print("=" * 96)
    print("disco shaped, the arm the whole experiment is about, fitted to BOTH families")
    print("=" * 96)
    print(f"  a 25% temperature error costs at most {CEILING_25PCT} nats in the soft family")
    print()
    hdr = (f"  {'sd':>3}{'step':>6}{'return':>9}{'H':>7}"
           f"{'soft lam':>10}{'soft res':>10}{'rule c':>9}{'rule res':>10}   verdict")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    for seed in range(args.seeds):
        seen.clear()
        curve = T.run_arm("disco", 0.0, 0.0, h, args.batch, args.rollout_len,
                          args.steps, args.probe_every, args.collect, seed,
                          meta_params, fam, None, True, True)
        ok, why = T.gate(curve)
        for c_pt, p in zip(curve, seen):
            c_hat, r_rule, _ = rule_fam.fit(p)
            better = ("rule" if r_rule < c_pt["residual"] else "soft")
            inside = (r_rule <= CEILING_25PCT or c_pt["residual"] <= CEILING_25PCT)
            tag = f"{better} wins" + ("" if inside else ", neither inside")
            rows.append({"seed": seed, "step": c_pt["step"], "admissible": ok,
                         "reason": why, "return": c_pt["return"],
                         "h_traj": c_pt["h_traj"],
                         "soft_lambda": c_pt["lambda_eff"],
                         "soft_residual": c_pt["residual"],
                         "rule_c": c_hat, "rule_residual": r_rule})
            last = c_pt is curve[-1]
            print(f"  {seed:>3}{c_pt['step']:>6}{c_pt['return']:>9.3f}"
                  f"{c_pt['h_traj']:>7.3f}{c_pt['lambda_eff']:>10.3f}"
                  f"{c_pt['residual']:>10.4f}{c_hat:>9.3f}{r_rule:>10.4f}"
                  f"   {tag}{'   <- ' + why if last else ''}")
        print()

    fam.fit = original_fit
    finals = [r for r in rows if r["step"] == args.steps and r["admissible"]]
    if not finals:
        finals = [r for r in rows if r["step"] == args.steps]
    soft_r = float(np.mean([r["soft_residual"] for r in finals]))
    rule_r = float(np.mean([r["rule_residual"] for r in finals]))
    rule_c = float(np.mean([r["rule_c"] for r in finals]))
    soft_l = float(np.mean([r["soft_lambda"] for r in finals]))

    print("=" * 96)
    print(f"  soft family:   lambda_eff = {soft_l:.3f}   residual = {soft_r:.4f}")
    print(f"  rule's family: c          = {rule_c:.3f}   residual = {rule_r:.4f}")
    print()
    if rule_r <= CEILING_25PCT:
        reading = (
            f"IT IS AN ENTROPY-BONUS ACTOR-CRITIC -- the discovered rule's "
            f"distribution sits inside the fixed-point family of pi ~ exp(Q_pi/c) "
            f"at c = {rule_c:.3f}, residual {rule_r:.4f}, while the soft family "
            f"cannot hold it at {soft_r:.4f}. It is not a GFlowNet and it is not "
            f"soft RL; it is the thing the ladder is, at a cold coefficient.")
    elif rule_r < soft_r / 2:
        reading = (
            f"CLOSER TO THE RULE THAN TO SOFT RL -- residual {rule_r:.4f} against "
            f"{soft_r:.4f}, so the entropy-bonus form describes it better, but "
            f"{rule_r:.4f} is still above {CEILING_25PCT} and neither family "
            f"contains it.")
    else:
        reading = (
            f"NEITHER FAMILY -- soft {soft_r:.4f}, rule {rule_r:.4f}, both far "
            f"above {CEILING_25PCT}. The discovered rule's distribution is not a "
            f"tempered version of either, so 'what regularisation does it "
            f"implement' has no answer of this shape on this MDP.")
    print(f"Reading: {reading}")

    out = {"config": vars(args), "rows": rows, "soft_residual": soft_r,
           "rule_residual": rule_r, "rule_c": rule_c, "soft_lambda": soft_l,
           "ceiling_25pct": CEILING_25PCT, "reading": reading}
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
