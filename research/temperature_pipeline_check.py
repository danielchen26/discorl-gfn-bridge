#!/usr/bin/env python3
"""The instrument check that STEP 1 does not do: drive the environment.

temperature.py validates its readout twice before it reads an agent. I1 feeds
the fit exact family members. I2 feeds the whole chain -- sample, bucket by
state, forward DP, fit -- and recovers every coefficient within ten percent
with a bucketing error at 1e-16.

Both are blind to the same link. I2 samples with research/temperature.py's
`sample_exact`, which rolls the policy in numpy and stops the trajectory when
it terminates. The agents do not: they act in the disco_rl environment, and
`probe()` hands `roll.observations` and `roll.actions` to `bucket` wholesale,
with no filter on step type. So the env -> bucket link is the one link the
instrument never tests, and it is the link every arm goes through.

The consistency number cannot catch a fault there either, and provably so. The
docstring on `bucket` proves reach(c) = n(c)/N when the DP is run under the
same bucketed policy that produced the counts. Feed it contaminated counts and
both sides are contaminated identically: bucket_err stays at 1e-16 while the
distribution it certifies is wrong.

WHAT THE ENVIRONMENT ACTUALLY DOES. `_SingleStreamTiapkin.step` ignores the
stopped flag. So a state with stopped = 1 is terminal for `is_terminal` -- the
timestep is LAST and `reward` pays log R again -- but it is not absorbing: the
agent keeps being asked for actions there, each one is recorded against the
state it stopped in, each STOP among them is counted by `bucket` as a fresh
termination, and a move takes it back out. A single trace from the source:

    t  obs  step_type  reward   action
    0    0        MID  -0.000        2   <- STOP, the real terminating action
    1    0       LAST  -2.000        2   <- these four are recorded against
    2    0       LAST  -2.000        2      state 0 and counted as four more
    3    0       LAST  -2.000        2      terminations at state 0
    4    0       LAST  -2.000        1   <- and this one moves it back out
    5    1        MID  -0.000        0

THE CHECK. Take the soft-optimal policy at lambda = 1, which is exactly the
target R/Z and which I1 and I2 both recover to four decimals, and act with it
in the environment instead of in numpy. Bucket the result the way probe() does,
and again with post-terminal steps dropped. A sound pipeline returns 1.000
either way.

Usage:
  .venv/bin/python research/temperature_pipeline_check.py \
      --json research/temperature_pipeline_check.json
"""

from __future__ import annotations

import argparse
import json
import math

import dm_env
import jax
import jax.numpy as jnp
import numpy as np

import temperature as T

LAST = int(dm_env.StepType.LAST)


def roll_in_env(pi: np.ndarray, h: int, batch: int, steps: int, seed: int):
    """Act with a prescribed state-conditional policy inside the environment.

    Deliberately mirrors what an agent does: read the observation, sample from
    the policy at that observation, step. No knowledge that a state is terminal,
    because the agent has none either -- the observation is the one-hot board
    and renders identically whether or not the stopped flag is set.
    """
    env = T.TiapkinEnvironment(batch_size=batch,
                               env_settings=T.get_config(h, True, False))
    key = jax.random.PRNGKey(seed)
    env_state, ts = env.reset(key)
    idx, acts, kinds = [], [], []
    cum = np.cumsum(pi, axis=1)
    for _ in range(steps):
        key, sub = jax.random.split(key)
        obs = np.asarray(ts.observation).reshape(batch, -1).argmax(-1)
        u = np.asarray(jax.random.uniform(sub, (batch,)))
        a = (u[:, None] > cum[obs]).sum(1)
        idx.append(obs)
        acts.append(a)
        kinds.append(np.asarray(ts.step_type))
        env_state, ts = env.step(env_state, jnp.asarray(a, dtype=jnp.int32))
    return (np.concatenate(idx), np.concatenate(acts), np.concatenate(kinds))


def read(idx, acts, h, fam):
    pi_hat, emp, nseen = T.bucket([(idx, acts)], h)
    term, ret, htraj = T.forward_dp(pi_hat, h)
    lam, res, _ = fam.fit(term)
    return {"lambda_eff": lam, "residual": res, "states_seen": nseen,
            "bucket_err": float(np.max(np.abs(term - emp))),
            "kl_to_target": fam.kl_to_target(term), "return": ret,
            "h_traj": htraj}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    h = T.HEIGHT
    lams = np.exp(np.linspace(math.log(0.03), math.log(8.0), 121))
    fam = T.SoftFamily(h, lams)

    print("=" * 88)
    print(f"I3  the exact soft-optimal policy at lambda = {args.lam}, acting in the "
          f"environment")
    print("=" * 88)
    print("  I2 recovers this same policy to four decimals when it is rolled in numpy.")
    print("  The only thing changed here is that the environment does the stepping,")
    print("  which is what every arm in temperature.py does.")
    print()

    pi = T.soft_policy(h, args.lam)
    rows = []
    hdr = (f"  {'seed':>4}{'pairs':>8}{'post-term':>11}{'lam_eff':>10}{'residual':>11}"
           f"{'bkt err':>10}{'seen':>6}")
    print("  as probe() buckets it -- every recorded step, no filter")
    print(hdr)
    for seed in range(args.seeds):
        idx, acts, kinds = roll_in_env(pi, h, args.batch, args.steps, seed)
        frac = float((kinds == LAST).mean())
        r = read(idx, acts, h, fam)
        rows.append({"seed": seed, "filtered": False, "post_terminal_frac": frac, **r})
        print(f"  {seed:>4}{len(idx):>8}{frac:>10.1%}{r['lambda_eff']:>10.3f}"
              f"{r['residual']:>11.4f}{r['bucket_err']:>10.1e}{r['states_seen']:>6}")

    print()
    print("  with post-terminal steps dropped -- the only change")
    print(hdr)
    for seed in range(args.seeds):
        idx, acts, kinds = roll_in_env(pi, h, args.batch, args.steps, seed)
        keep = kinds != LAST
        frac = float((kinds == LAST).mean())
        r = read(idx[keep], acts[keep], h, fam)
        rows.append({"seed": seed, "filtered": True, "post_terminal_frac": frac, **r})
        print(f"  {seed:>4}{int(keep.sum()):>8}{frac:>10.1%}{r['lambda_eff']:>10.3f}"
              f"{r['residual']:>11.4f}{r['bucket_err']:>10.1e}{r['states_seen']:>6}")

    raw = [r for r in rows if not r["filtered"]]
    fil = [r for r in rows if r["filtered"]]
    lam_raw = float(np.mean([r["lambda_eff"] for r in raw]))
    lam_fil = float(np.mean([r["lambda_eff"] for r in fil]))
    res_raw = float(np.mean([r["residual"] for r in raw]))
    res_fil = float(np.mean([r["residual"] for r in fil]))

    print()
    print("=" * 88)
    print(f"  unfiltered: lambda_eff = {lam_raw:.3f}, residual = {res_raw:.4f}")
    print(f"  filtered:   lambda_eff = {lam_fil:.3f}, residual = {res_fil:.4f}")
    print(f"  the truth is {args.lam:.3f} with residual 0")

    # 0.0861 nats is the ceiling on what a 25 percent temperature error can cost
    # anywhere in this family, measured by temperature_residual_scale.py.
    ceiling = 0.0861
    if res_raw > ceiling and res_fil <= ceiling:
        reading = (
            f"THE PIPELINE IS THE FAULT -- a policy that IS the target reads "
            f"lambda_eff = {lam_raw:.3f} with residual {res_raw:.4f} through the "
            f"env path, above the {ceiling} nats a 25% temperature error can cost, "
            f"and drops to {res_fil:.4f} at lambda_eff = {lam_fil:.3f} when the "
            f"post-terminal steps are removed. Every residual temperature.py "
            f"reports carries this, so OUTSIDE THE FAMILY is a statement about "
            f"bucket(), not about any agent.")
    elif res_raw <= ceiling:
        reading = (
            f"THE PIPELINE IS SOUND -- the exact policy survives the env path at "
            f"lambda_eff = {lam_raw:.3f}, residual {res_raw:.4f}, so the residuals "
            f"the arms show are theirs and the reading stands.")
    else:
        reading = (
            f"CONTAMINATED BUT NOT ONLY BY THIS -- filtering moves the residual "
            f"from {res_raw:.4f} to {res_fil:.4f}, still above {ceiling}, so the "
            f"post-terminal steps are a fault but not the whole of it.")
    print(f"Reading: {reading}")

    out = {"config": vars(args), "rows": rows, "lambda_unfiltered": lam_raw,
           "lambda_filtered": lam_fil, "residual_unfiltered": res_raw,
           "residual_filtered": res_fil, "ceiling_25pct": ceiling,
           "reading": reading}
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
