#!/usr/bin/env python3
"""Is the ladder's misfit the critic's value resolution?

After temperature.py's own run, one fact is left standing that nothing explains:
`ladder ac c=1.0` -- the arm whose coefficient is exactly the family's lambda,
since disco_rl's actor_critic puts entropy in the loss as entropy_cost * (-H)
with pg_cost = 1.0 and normalize_adv False -- reads lambda_eff = 1.014, the
right coefficient, while leaving a residual of 1.564. A 25 percent temperature
error costs at most 0.0861 nats anywhere in this family, so 1.564 is not a
temperature error. The location is right and the distribution is not.

Four explanations have been excluded:

*   Sampling noise in the bucketed policy. temperature_pipeline_check.py drives
    the exact lambda = 1 policy through the environment and reads it back at
    lambda_eff = 0.995, residual 0.0062.
*   The policy class. disco_rl's MLP computes logits from the current
    observation alone -- the LSTM in net_args is a Muesli-style action-
    conditional model feeding other heads, not policy memory -- and the
    observation is the one-hot board, so the policy is Markov on exactly the
    family's domain.
*   Training budget. 30000 updates moves the residual from 1.36 to 1.23 and
    lambda_eff from 0.99 to 0.96. It plateaus.
*   Optimisation failure. The arm sits at return -0.48 against the -0.379 that
    the lambda = 1 member itself earns, so it is not an agent that gave up.

What is left is the critic. As configured the value head is categorical over
601 bins spanning max_abs_value = 300 through a nonlinear transform, so near
zero one bin is worth about 0.111 in raw return -- while the entire span of
returns available in this MDP is 5.43, from -2.000 for stopping at the source
to 3.431 for the best terminal state. The soft-optimal policy is set by
advantage differences; quantising them at 0.111 perturbs every logit by that
much divided by lambda, at lambda = 1.

Two arms, against the unchanged one already measured:

    max_abs_value = 8      601 bins over the range that exists: 0.0134 per bin
    categorical_value off  scalar value head, no bins at all

If either drops the residual under 0.0861, the ladder's misfit is the critic
and every reading in temperature.py has to be retaken with the value head
sized to the problem. If neither does, entropy-regularised policy gradient
does not reach the soft-optimal distribution on this MDP, and fitting the soft
family to a trained agent is not a measurement that can be made here at all --
which decides whether the whole route has a future.

Usage:
  .venv/bin/python research/temperature_critic_ablation.py \
      --json research/temperature_critic_ablation.json
"""

from __future__ import annotations

import argparse
import json
import math

import jax
import numpy as np

from disco_rl import agent as agent_lib

import disco_probe
import temperature as T

CEILING_25PCT = 0.0861


def make_build(max_abs_value: float | None, categorical: bool):
    """temperature.build_agent with the value head resized.

    Mirrors the original line for line -- same env, same settings source, same
    discount, same net -- and changes only the two value-head fields, so the
    comparison against the measured arm is clean.
    """

    def build(rule, batch, height, entropy_cost, lr, pb, shaped):
        env = T.TiapkinEnvironment(
            batch_size=batch, env_settings=T.get_config(height, pb, shaped))
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
        settings.update_rule.categorical_value = categorical
        if max_abs_value is not None:
            settings.update_rule.max_abs_value = max_abs_value
        if lr is not None:
            settings.learning_rate = lr
        ag = agent_lib.Agent(
            agent_settings=settings,
            single_observation_spec=env.single_observation_spec(),
            single_action_spec=env.single_action_spec(),
            batch_axis_name=None,
        )
        return env, ag

    return build


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

    print("=" * 92)
    print("the ladder arm whose coefficient is exactly 1, with the value head resized")
    print("=" * 92)
    print(f"  measured already, max_abs_value = 300, categorical: "
          f"lambda_eff 1.014, residual 1.564")
    print(f"  a 25% temperature error costs at most {CEILING_25PCT} nats anywhere "
          f"in this family")
    print()

    variants = [
        ("mav=8 categorical", 8.0, True),
        ("scalar value head", None, False),
    ]
    hdr = (f"  {'variant':<20}{'sd':>3}{'step':>6}{'seen':>5}{'bkt':>9}{'return':>9}"
           f"{'H':>7}{'KL':>8}{'lam_eff':>9}{'resid':>9}")
    original = T.build_agent
    results = []
    try:
        for name, mav, categorical in variants:
            T.build_agent = make_build(mav, categorical)
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for seed in range(args.seeds):
                curve = T.run_arm("actor_critic", 0.0, 1.0, h, args.batch,
                                  args.rollout_len, args.steps, args.probe_every,
                                  args.collect, seed, meta_params, fam, args.lr,
                                  True, True)
                ok, why = T.gate(curve)
                for c in curve:
                    mark = "" if c is not curve[-1] else ("   <- " + why)
                    print(f"  {name:<20}{seed:>3}{c['step']:>6}{c['states_seen']:>5}"
                          f"{c['bucket_err']:>9.1e}{c['return']:>9.3f}"
                          f"{c['h_traj']:>7.3f}{c['kl']:>8.3f}"
                          f"{c['lambda_eff']:>9.3f}{c['residual']:>9.4f}{mark}")
                print()
                results.append({"variant": name, "max_abs_value": mav,
                                "categorical": categorical, "seed": seed,
                                "admissible": ok, "reason": why, "curve": curve,
                                "final": curve[-1]})
    finally:
        T.build_agent = original

    print("=" * 92)
    summary = {}
    for name, mav, categorical in variants:
        good = [r for r in results if r["variant"] == name and r["admissible"]]
        if not good:
            summary[name] = {"n_admissible": 0}
            print(f"  {name:<20} no admissible seed")
            continue
        le = np.array([r["final"]["lambda_eff"] for r in good])
        rs = np.array([r["final"]["residual"] for r in good])
        summary[name] = {"n_admissible": len(good),
                         "lambda_eff_mean": float(le.mean()),
                         "residual_mean": float(rs.mean()),
                         "residual_min": float(rs.min())}
        print(f"  {name:<20} lambda_eff = {le.mean():.3f}   residual = "
              f"{rs.mean():.4f} (best seed {rs.min():.4f})   n = {len(good)}/{args.seeds}")

    best = min((s.get("residual_mean", np.inf) for s in summary.values()),
               default=np.inf)
    print()
    if best <= CEILING_25PCT:
        reading = (
            f"THE CRITIC WAS THE FAULT -- resizing the value head brings the "
            f"correct-by-construction arm to residual {best:.4f}, inside the "
            f"{CEILING_25PCT} nats a 25% temperature error can cost. Every reading "
            f"in temperature.py was taken through a critic quantising return at "
            f"0.111 and has to be retaken.")
    elif best < 1.564 / 2:
        reading = (
            f"THE CRITIC IS PART OF IT -- residual falls from 1.564 to {best:.4f}, "
            f"a real improvement, but still above {CEILING_25PCT}, so the value "
            f"head is one cause and not the whole one.")
    else:
        reading = (
            f"NOT THE CRITIC -- the best resized variant still leaves {best:.4f}. "
            f"With sampling, policy class, budget and optimisation already "
            f"excluded, entropy-regularised policy gradient does not reach the "
            f"soft-optimal distribution on this MDP, and fitting the soft family "
            f"to a trained agent is not a measurement that can be made here.")
    print(f"Reading: {reading}")

    out = {"config": vars(args), "per_run": results, "summary": summary,
           "ceiling_25pct": CEILING_25PCT, "baseline_residual": 1.564,
           "reading": reading}
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
