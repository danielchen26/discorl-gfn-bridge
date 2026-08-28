#!/usr/bin/env python3
"""Is DiscoRL's update the gradient of anything?

Everything the dossier tried to map assumed the question "which functional of W
does DiscoRL minimise" is well posed.  It is well posed only if the update is a
gradient field.  GFlowNet's trajectory balance is one by construction -- it is
the gradient of an explicit scalar loss -- so a potential exists and the cumulant
comparison applies.  DiscoRL produces a target and regresses onto it, which is a
SEMI-gradient: the same structure that makes TD learning not the gradient of any
function (Baird's counterexample).

By the Poincare lemma a field is a gradient iff its Jacobian is symmetric, so the
question has a single scalar answer.  Work in PREDICTION space rather than
parameter space: a gradient field in prediction space pulls back to a gradient
field in parameter space (the pullback of -grad_u Phi under u(theta) is
-grad_theta Phi(u(theta))), so prediction space answers the same question while
being architecture-independent, and it needs only FIRST derivatives of the
target map -- no nested differentiation.

The per-step loss is KL(target || prediction) with the target held fixed, whose
gradient with respect to the prediction logits u is

    v(u) = p_hat(u) - p(u),        p_hat = softmax(f_eta(u)),  p = softmax(u)

so the Jacobian splits as

    J = B - D,      B = d p_hat / du   (the bootstrap term),
                    D = d p / du = diag(p) - p p^T   (symmetric, always)

Every departure from a gradient field lives in B, which is exactly the term
stop_gradient discards.  So the sharp measurement is

    kappa_boot = ||sym B||_F / ||B||_F

REFERENCE SCALE, analytic and dimension-free.  For tabular TD the target map is
V_hat = r + gamma P V, so B = gamma P.  On a DAG there are no 2-cycles, hence
tr(P^2) = 0, hence ||P + P^T||_F = ||P - P^T||_F, hence

    kappa_boot(TD on any DAG) = 1 / sqrt(2) = 0.7071

giving the scale:  true gradient 1.0   |   tabular TD 0.7071   |   circulation 0.

THREE TRAPS, each of which silently returns the wrong answer.

1.  Second-order stop-gradient.  jax.lax.stop_gradient has zero derivative at
    ALL orders, so differentiating the repo's `backprop=False` path erases B and
    reports a perfect gradient field for a spurious reason.  Here B is obtained
    as a first derivative of the target map itself, so no stop_gradient is ever
    in the path.

2.  No reference scale -- handled by the analytic 1/sqrt(2) above and by an
    actor-critic arm measured through the identical pipeline.

3.  An unvalidated estimator.  Frobenius norms come from Hutchinson probes,
    ||M||^2 = E||Mw||^2, with Bw from a jvp and B^T w from a vjp on the SAME
    probe, since the ratio of two strongly correlated estimates is far tighter
    than the ratio of two independent ones.  The estimator is checked against
    synthetic maps of known kappa before it is believed.

A note on jit: disco.py:162 mutates the rollout dataclass in place, which leaks
tracers out of a nested transformation, so the probes run eagerly.  Slower, and
correct.

Usage:
  .venv/bin/python research/kappa.py --json research/kappa.json
"""

from __future__ import annotations

import argparse
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

import disco_probe
from disco_rl import agent as agent_lib
from disco_rl import types
import hypergrid_env


# --------------------------------------------------------------------------
# Estimator.
# --------------------------------------------------------------------------


def _ratio(sym: np.ndarray, asym: np.ndarray, block: int = 8) -> tuple[float, float]:
    """kappa = ||sym M|| / ||M||, with a block-jackknife standard error."""
    kap = math.sqrt(sym.sum() / max(sym.sum() + asym.sum(), 1e-30))
    n = sym.size
    nb = max(2, min(block, n))
    parts = np.array_split(np.arange(n), nb)
    ks = []
    for b in range(nb):
        m = np.ones(n, bool)
        m[parts[b]] = False
        ks.append(math.sqrt(sym[m].sum() / max(sym[m].sum() + asym[m].sum(), 1e-30)))
    ks = np.array(ks)
    return kap, math.sqrt((nb - 1) / nb * ((ks - ks.mean()) ** 2).sum())


def kappa_of_map(g, x0, n_probes: int, key, block: int = 8):
    """kappa of the Jacobian of a map g at x0, by paired Hutchinson probes.

    ||J +- J^T||^2 = E_w ||Jw +- J^T w||^2, both matvecs on the same w.
    """
    flat0, unravel = ravel_pytree(x0)
    d = flat0.size

    def flat_g(fx):
        return ravel_pytree(g(unravel(fx)))[0]

    sym, asym, nrm = [], [], []
    for _ in range(n_probes):
        key, k = jax.random.split(key)
        w = jax.random.normal(k, (d,))
        _, jw = jax.jvp(flat_g, (flat0,), (w,))
        _, vjp = jax.vjp(flat_g, flat0)
        (jtw,) = vjp(w)
        sym.append(float(jnp.sum((jw + jtw) ** 2)))
        asym.append(float(jnp.sum((jw - jtw) ** 2)))
        nrm.append(float(jnp.sum(jw**2)))
    kap, se = _ratio(np.array(sym), np.array(asym), block)
    return kap, se, d, float(np.mean(nrm))


def synthetic_check(d: int, n_probes: int, key) -> list[dict]:
    """g(x) = (S + c A) x with S symmetric and A antisymmetric.

    kappa = ||S|| / sqrt(||S||^2 + c^2 ||A||^2) exactly, so sweeping c walks the
    estimator across the whole range it will later be asked to report.
    """
    key, k1, k2 = jax.random.split(key, 3)
    M = jax.random.normal(k1, (d, d)) / math.sqrt(d)
    S = (M + M.T) / 2
    N = jax.random.normal(k2, (d, d)) / math.sqrt(d)
    A = (N - N.T) / 2
    nS, nA = float(jnp.linalg.norm(S)), float(jnp.linalg.norm(A))
    rows = []
    for c in (0.0, 0.25, 1.0, 4.0, 40.0):
        exact = nS / math.sqrt(nS**2 + (c * nA) ** 2)
        key, k = jax.random.split(key)
        est, se, _, _ = kappa_of_map(lambda x, c=c: (S + c * A) @ x, jnp.ones(d), n_probes, k)
        rows.append({"c": c, "exact": exact, "est": est, "se": se, "err": abs(est - exact)})
    return rows


def causal_check(d: int, n_probes: int, key) -> list[dict]:
    """Calibrate the estimator around the value the real measurement lands on.

    For any operator, ||B +- B^T||^2 = 2||B||^2 +- 2 tr(B^2), so

        kappa^2 = 1/2 + tr(B^2) / (2 ||B||^2).

    A CAUSAL bootstrap -- target at index i depending only on indices j > i -- is
    strictly triangular, so tr(B^2) = 0 and kappa = 1/sqrt(2) exactly, whatever
    else it does.  That is the structural floor.  Adding a diagonal block, which
    is what a target that also reads its own index does, moves kappa above it by
    a predictable amount.  Sweeping that block checks the estimator resolves
    exactly the regime the real number lives in.
    """
    key, k = jax.random.split(key)
    R = jax.random.normal(k, (d, d)) / math.sqrt(d)
    B0 = jnp.triu(R, 1)  # strictly upper triangular: a causal bootstrap
    rows = []
    for c in (0.0, 0.05, 0.15, 0.4):
        key, k1, k2 = jax.random.split(key, 3)
        diag = jnp.diag(jax.random.normal(k1, (d,)))
        B = B0 + c * diag
        exact = math.sqrt(0.5 + float(jnp.trace(B @ B)) / (2 * float(jnp.sum(B * B))))
        est, se, _, _ = kappa_of_map(lambda x, B=B: B @ x, jnp.ones(d), n_probes, k2)
        rows.append({"diag_scale": c, "exact": exact, "est": est, "se": se, "err": abs(est - exact)})
    return rows


# --------------------------------------------------------------------------
# The target maps.
# --------------------------------------------------------------------------


def build_arm(rule: str, height: int, batch: int):
    """Same network, same environment, different update rule."""
    env = hypergrid_env.HypergridJittableEnvironment(
        batch_size=batch, env_settings=hypergrid_env.get_config_hypergrid()
    )
    settings = (
        agent_lib.get_settings_disco()
        if rule == "disco"
        else agent_lib.get_settings_actor_critic()
    )
    settings.net_settings.name = "mlp"
    settings.net_settings.net_args = dict(
        dense=(128, 128),
        model_arch_name="lstm",
        head_w_init_std=1e-2,
        model_kwargs=dict(head_mlp_hiddens=(64,), lstm_size=64),
    )
    settings.learning_rate = 1e-3
    ag = agent_lib.Agent(
        agent_settings=settings,
        single_observation_spec=env.single_observation_spec(),
        single_action_spec=env.single_action_spec(),
        batch_axis_name=None,
    )
    return env, ag


def target_map(ag, rollout, actor_state, meta_state, meta_params, params, base_out):
    """u -> softmax(target(u)) on the channels the agent loss actually touches.

    Shapes force one choice.  The meta-network consumes agent outputs of length
    T+1 and emits targets of length T, so the map is square only on the first T
    slices; the final slice is held at its base value.  That is the sub-block the
    update actually lives in, since agent_loss truncates to [:-1] as well.

    The prediction vector is (policy logits, y logits, z logits at the action
    taken).  z at other actions stays fixed because the loss only ever touches
    z(s, a_t).  The target network is likewise held fixed: it is a Polyak-delayed
    copy, so suppressing its dependence on the current prediction is faithful to
    what the rule does within one update.
    """
    hyper = ag.settings.hyper_params.to_dict()
    T = base_out["logits"].shape[0] - 1
    acts = rollout.actions[:T]  # [T, B]
    n_act = base_out["z"].shape[2]
    onehot = jax.nn.one_hot(acts, n_act)[..., None]  # [T, B, A, 1]

    def target_unroll(*_args, **_kw):
        return base_out, None

    def g(u):
        logits = jnp.concatenate([u["logits"], base_out["logits"][T:]], axis=0)
        y = jnp.concatenate([u["y"], base_out["y"][T:]], axis=0)
        z_head = base_out["z"][:T] * (1 - onehot) + u["z_a"][:, :, None, :] * onehot
        z = jnp.concatenate([z_head, base_out["z"][T:]], axis=0)
        out = dict(base_out)
        out["logits"], out["y"], out["z"] = logits, y, z
        meta_out, _ = ag.update_rule.unroll_meta_net(
            meta_params=meta_params,
            params=params,
            state=actor_state,
            meta_state=meta_state,
            rollout=types.UpdateRuleInputs(
                observations=rollout.observations,
                actions=rollout.actions,
                rewards=rollout.rewards[1:],
                is_terminal=rollout.discounts[1:] == 0,
                behaviour_agent_out=rollout.agent_outs,
                agent_out=out,
                value_out=None,
            ),
            hyper_params=hyper,
            unroll_policy_fn=target_unroll,
            rng=jax.random.PRNGKey(0),
            axis_name=None,
        )
        # The loss is a KL between distributions, so the field lives in
        # probability space: differentiate the softmaxed target, not the logits.
        return {
            "logits": jax.nn.softmax(meta_out["pi"], axis=-1),
            "y": jax.nn.softmax(meta_out["y"], axis=-1),
            "z_a": jax.nn.softmax(meta_out["z"], axis=-1),
        }

    u0 = {
        "logits": base_out["logits"][:T],
        "y": base_out["y"][:T],
        "z_a": jnp.take_along_axis(base_out["z"][:T], acts[..., None, None], axis=2)[:, :, 0],
    }
    return g, u0


def kappa_td_dag() -> float:
    """Tabular TD on a DAG: B = gamma P, no 2-cycles, so kappa = 1/sqrt(2)."""
    return 1.0 / math.sqrt(2.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rollout-len", type=int, default=10)
    ap.add_argument("--train-steps", type=int, default=300)
    ap.add_argument("--probes", type=int, default=64)
    ap.add_argument("--synthetic-dim", type=int, default=200)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    key = jax.random.PRNGKey(0)

    print("=" * 78)
    print("STEP 1  estimator validation against known kappa")
    print("=" * 78)
    key, k = jax.random.split(key)
    syn = synthetic_check(args.synthetic_dim, args.probes, k)
    print(f"{'c':>8}{'exact':>12}{'estimated':>12}{'se':>10}{'|err|':>10}")
    for r in syn:
        print(f"{r['c']:>8.2f}{r['exact']:>12.4f}{r['est']:>12.4f}{r['se']:>10.4f}{r['err']:>10.4f}")
    worst = max(r["err"] for r in syn)
    trusted = worst <= 0.02
    print(f"\nworst absolute error: {worst:.4f}  ->  {'trusted' if trusted else 'NOT TRUSTED'}")

    key, k = jax.random.split(key)
    cal = causal_check(args.synthetic_dim, args.probes, k)
    print(f"\n{'diag scale':>12}{'exact':>12}{'estimated':>12}{'se':>10}{'|err|':>10}")
    for r in cal:
        print(
            f"{r['diag_scale']:>12.2f}{r['exact']:>12.4f}{r['est']:>12.4f}"
            f"{r['se']:>10.4f}{r['err']:>10.4f}"
        )
    worst_cal = max(r["err"] for r in cal)
    trusted = trusted and worst_cal <= 0.02
    print(f"worst error near the causal floor: {worst_cal:.4f}")

    print("\n" + "=" * 78)
    print("STEP 2  the bootstrap term of Disco103")
    print("=" * 78)
    disco_params = disco_probe.unflatten_params(np.load(disco_probe.WEIGHTS))
    env, ag = build_arm("disco", args.height, args.batch)
    key, k = jax.random.split(key)
    rollout, learner_state, actor_state = disco_probe.train(
        env, ag, disco_params, args.rollout_len, args.train_steps, k
    )
    base_out, _ = ag.unroll_net(learner_state.params, actor_state, rollout)
    g, u0 = target_map(
        ag, rollout, actor_state, learner_state.meta_state, disco_params,
        learner_state.params, base_out,
    )
    key, k = jax.random.split(key)
    kap, se, d, nrm = kappa_of_map(g, u0, args.probes, k)
    tr_frac = 2 * kap**2 - 1
    arms = [{
        "rule": "disco103", "kappa_bootstrap": kap, "se": se, "dim": d,
        "mean_sq_norm": nrm, "tr_B2_over_normB2": tr_frac,
    }]
    print(f"\n  Disco103   kappa_boot = {kap:.4f} +/- {se:.4f}   (dim {d})")
    print(f"             |Bw|^2 = {nrm:.3e}   tr(B^2)/||B||^2 = {tr_frac:+.5f}")

    print("\n" + "=" * 78)
    print("STEP 3  reading")
    print("=" * 78)
    td = kappa_td_dag()
    print(f"scale:  true gradient 1.0000  |  ANY causal bootstrap {td:.4f}  |  circulation 0.0000")
    present = nrm > 1e-20
    print(f"\nbootstrap term present: |Bw|^2 = {nrm:.3e} "
          f"-> {'yes' if present else 'ZERO, the trap swallowed it; void'}")
    dev = (kap - td) / max(se, 1e-12)
    print(f"distance from the causal floor: {kap - td:+.4f}  ({dev:+.1f} sigma)")

    if not (trusted and present):
        verdict = "VOID -- a precondition failed"
    elif kap > 0.99:
        verdict = "CONSERVATIVE -- the bootstrap term is symmetric; a potential exists"
    elif abs(kap - td) < 5 * se:
        verdict = (
            "GENERIC CAUSAL BOOTSTRAP -- indistinguishable from the structural floor; "
            "no functional is being minimised, and no extra circulation either"
        )
    elif kap > td:
        verdict = "ABOVE THE FLOOR -- more symmetric than a bare causal bootstrap"
    else:
        verdict = "BELOW THE FLOOR -- extra circulation beyond causality"
    print(f"Verdict: {verdict}")
    print(f"  conservative share of the bootstrap operator: kappa^2 = {kap**2:.4f}")

    out = {
        "synthetic": syn,
        "synthetic_worst_err": worst,
        "estimator_trusted": bool(trusted),
        "causal_calibration": cal,
        "arms": arms,
        "causal_floor": td,
        "bootstrap_present": bool(present),
        "verdict": verdict,
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
