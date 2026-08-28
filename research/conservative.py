#!/usr/bin/env python3
"""Does it matter that DiscoRL is not a gradient method?

research/kappa.py established a diagnosis: Disco103's update has kappa = 0.7088,
so it is not the gradient of any functional. A diagnosis invites the obvious
follow-up, which the measurement alone cannot answer -- does the missing
potential cost anything, or is it incidental?

That question has a counterfactual, and the counterfactual is cheap. The rule is
a semi-gradient only because the target is held fixed when the agent's gradient
is taken. Let the gradient flow through the target as well and the update becomes
the gradient of an honest scalar, hence conservative by construction. Same
meta-parameters, same network, same optimiser, same seeds, same rollouts; one
line differs.

    semi   grad taken with the meta-network's output held constant  (as shipped)
    full   grad taken through the meta-network's output too          (kappa = 1)

IMPORTANT SCOPE. `full` is NOT `semi` with its curl removed. It is `semi` plus
the entire derivative through the target, which is a different algorithm rather
than a projection of the same one. So a performance difference between the arms
cannot be attributed to conservativity alone; it bounds the effect of the whole
intervention, and the antisymmetric part is only a fraction of that. Given that
the antisymmetric share of the update is under one percent, a large difference
here is evidence that the OTHER part of the intervention matters, not that the
curl does.

Predictions are not symmetric between the two outcomes, which is what makes this
worth running:

  * if the two arms match, the missing potential is incidental, and every
    stability worry inherited from TD is empty for this rule;
  * if `full` is worse, then the non-conservativity is DOING WORK -- the
    discovered rule is useful partly BECAUSE it left the gradient class, and
    "being a gradient method" is a cost that discovery pays away.

Classical RL theory predicts the second: differentiating through the target
minimises the Bellman residual rather than reaching the Bellman fixed point, and
the residual minimiser is the worse solution under function approximation
(Baird, 1995). If that carries over to a *discovered* rule, it says something
about what meta-learning is exploiting.

The manipulation is verified rather than assumed: kappa is measured for both
fields, and `full` must come back at exactly 1.

Usage:
  .venv/bin/python research/conservative.py --json research/conservative.json
"""

from __future__ import annotations

import argparse
import functools
import json

import jax

jax.config.update("jax_enable_x64", False)  # training runs in the shipped dtype

import jax.numpy as jnp
import numpy as np

import disco_probe
import kappa as kappa_mod
from disco_rl import types


def make_inputs(ag, rollout, params, actor_state):
    agent_out, _ = ag.unroll_net(params, actor_state, rollout)
    return types.UpdateRuleInputs(
        observations=rollout.observations,
        actions=rollout.actions,
        rewards=rollout.rewards[1:],
        is_terminal=rollout.discounts[1:] == 0,
        behaviour_agent_out=rollout.agent_outs,
        agent_out=agent_out,
        value_out=None,
    )


def make_learner(ag, meta_params, mode: str):
    """One agent-parameter update. `mode` is the entire experimental variable."""
    hyper = ag.settings.hyper_params.to_dict()

    def targets(params, actor_state, meta_state, rollout):
        meta_out, new_meta_state = ag.update_rule.unroll_meta_net(
            meta_params=meta_params,
            params=params,
            state=actor_state,
            meta_state=meta_state,
            rollout=make_inputs(ag, rollout, params, actor_state),
            hyper_params=hyper,
            unroll_policy_fn=ag._network.unroll,  # pylint: disable=protected-access
            rng=jax.random.PRNGKey(0),
            axis_name=None,
        )
        return meta_out, new_meta_state

    def loss_given(params, meta_out, actor_state, rollout, backprop: bool):
        per_step, _ = ag.update_rule.agent_loss(
            make_inputs(ag, rollout, params, actor_state), meta_out, hyper, backprop=backprop
        )
        return jnp.mean(per_step)

    if mode == "semi":

        def grads(params, actor_state, meta_state, rollout):
            # Target computed OUTSIDE the gradient: it is a constant to the
            # update, which is exactly what stop_gradient achieves in the
            # shipped code, and exactly what forbids a potential.
            meta_out, new_meta_state = targets(params, actor_state, meta_state, rollout)
            g = jax.grad(loss_given)(params, meta_out, actor_state, rollout, False)
            return g, new_meta_state

    else:

        def grads(params, actor_state, meta_state, rollout):
            # Target computed INSIDE the gradient, so the update is the gradient
            # of a scalar and is conservative by construction. The new meta state
            # is returned through has_aux rather than smuggled out of the trace
            # through a closure -- doing the latter leaks a tracer and haiku
            # reports it far from where it was caused.
            def total(p):
                meta_out, nms = targets(p, actor_state, meta_state, rollout)
                return loss_given(p, meta_out, actor_state, rollout, True), nms

            g, nms = jax.grad(total, has_aux=True)(params)
            return g, nms

    return grads


def adam_init(params):
    return jax.tree.map(jnp.zeros_like, params), jax.tree.map(jnp.zeros_like, params)


def adam_step(params, g, m, v, t, lr):
    m = jax.tree.map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
    v = jax.tree.map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
    mh = jax.tree.map(lambda a: a / (1 - 0.9**t), m)
    vh = jax.tree.map(lambda a: a / (1 - 0.999**t), v)
    params = jax.tree.map(
        lambda a, b, c: a - lr * b / (jnp.sqrt(c + 1e-30) + 1e-8), params, mh, vh
    )
    return params, m, v


def run_arm(env, ag, meta_params, mode, steps, rollout_len, lr, seed):
    """Train one arm and return the reward-per-rollout curve."""
    rng = jax.random.PRNGKey(seed)
    learner_state = ag.initial_learner_state(rng)
    actor_state = ag.initial_actor_state(rng)
    env_state, ts = env.reset(rng)
    params = learner_state.params
    meta_state = learner_state.meta_state
    m, v = adam_init(params)
    grads = make_learner(ag, meta_params, mode)

    @jax.jit
    def unroll(p, actor_state, ts, env_state, key):
        def _step(carry, k):
            env_state, ts, actor_state = carry
            at, actor_state = ag.actor_step(p, k, ts, actor_state)
            env_state, ts = env.step(env_state, at.actions)
            return (env_state, ts, actor_state), at

        (env_state, ts, actor_state), stacked = jax.lax.scan(
            _step, (env_state, ts, actor_state), jax.random.split(key, rollout_len)
        )
        return types.ActorRollout.from_timestep(stacked), actor_state, ts, env_state

    curve = []
    for t in range(1, steps + 1):
        rng, k = jax.random.split(rng)
        rollout, actor_state, ts, env_state = unroll(params, actor_state, ts, env_state, k)
        g, meta_state = grads(params, actor_state, meta_state, rollout)
        params, m, v = adam_step(params, g, m, v, t, lr)
        curve.append(float(jnp.mean(jnp.sum(rollout.rewards, axis=0))))
    return np.array(curve), params, actor_state, meta_state, rollout


def manipulation_check(ag, meta_params, rollout, params, actor_state, meta_state, probes, key):
    """kappa of each arm's field, in prediction space. `full` must return 1."""
    base_out, _ = ag.unroll_net(params, actor_state, rollout)
    g_target, u0 = kappa_mod.target_map(
        ag, rollout, actor_state, meta_state, meta_params, params, base_out
    )

    def field_semi(u):
        # p_hat(u) - p(u): the direction the agent is actually moved in.
        pu = jax.tree.map(lambda x: jax.nn.softmax(x, axis=-1), u)
        return jax.tree.map(lambda a, b: a - b, g_target(u), pu)

    def scalar(u):
        # The KL's Hessian carries 1/p terms, and a 600-way softmax after
        # training puts mass far below float32 resolution in most bins, so an
        # unclipped version returns nan for the second derivative while the
        # forward value looks fine. Clip both arguments to a floor that is above
        # the point where the curvature is numerically meaningless.
        floor = 1e-6
        phat = jax.tree.map(lambda x: jnp.clip(x, floor, 1.0), g_target(u))
        pu = jax.tree.map(
            lambda x: jnp.clip(jax.nn.softmax(x, axis=-1), floor, 1.0), u
        )
        tot = 0.0
        for k in phat:
            tot = tot + jnp.sum(phat[k] * (jnp.log(phat[k]) - jnp.log(pu[k])))
        return tot

    def field_full(u):
        return jax.tree.map(lambda x: -x, jax.grad(scalar)(u))

    k1, k2 = jax.random.split(key)
    ks, ses, _, _ = kappa_mod.kappa_of_map(field_semi, u0, probes, k1)
    kf, sef, _, _ = kappa_mod.kappa_of_map(field_full, u0, probes, k2)
    return {"kappa_semi": ks, "se_semi": ses, "kappa_full": kf, "se_full": sef}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default="catch", choices=("catch", "hypergrid"))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--rollout-len", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--probes", type=int, default=24)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    meta_params = disco_probe.unflatten_params(np.load(disco_probe.WEIGHTS))
    print(f"env={args.env}  steps={args.steps}  batch={args.batch}  seeds={args.seeds}")
    print("one variable: whether the gradient flows through the meta-network's target\n")

    curves = {"semi": [], "full": []}
    check = None
    for seed in range(args.seeds):
        for mode in ("semi", "full"):
            env, ag = kappa_mod.build_arm("disco", 8, args.batch, args.env)
            c, params, actor_state, meta_state, rollout = run_arm(
                env, ag, meta_params, mode, args.steps, args.rollout_len, args.lr, seed
            )
            curves[mode].append(c)
            tail = float(np.mean(c[-args.steps // 5 :]))
            print(f"  seed {seed}  {mode:<5}  final-fifth mean reward {tail:+.4f}")
            if check is None and mode == "semi":
                check = manipulation_check(
                    ag, meta_params, rollout, params, actor_state, meta_state,
                    args.probes, jax.random.PRNGKey(7),
                )

    print("\nmanipulation check (prediction space):")
    print(f"  kappa(semi) = {check['kappa_semi']:.4f} +/- {check['se_semi']:.4f}")
    print(f"  kappa(full) = {check['kappa_full']:.4f} +/- {check['se_full']:.4f}"
          f"   {'<- conservative, as intended' if abs(check['kappa_full'] - 1) < 5e-3 else '<- MANIPULATION FAILED'}")

    S = np.stack(curves["semi"])
    F = np.stack(curves["full"])
    tail = args.steps // 5
    s_tail = S[:, -tail:].mean(1)
    f_tail = F[:, -tail:].mean(1)
    print(f"\nfinal-fifth mean reward over {args.seeds} seeds")
    print(f"  semi (as shipped, kappa<1) : {s_tail.mean():+.4f} +/- {s_tail.std():.4f}")
    print(f"  full (forced conservative) : {f_tail.mean():+.4f} +/- {f_tail.std():.4f}")
    gap = s_tail.mean() - f_tail.mean()
    pooled = np.sqrt(s_tail.var() / max(len(s_tail), 1) + f_tail.var() / max(len(f_tail), 1)) + 1e-12
    print(f"  gap (semi - full)          : {gap:+.4f}   ({gap/pooled:+.1f} sigma)")

    ok = abs(check["kappa_full"] - 1) < 5e-3
    if not ok:
        verdict = "VOID -- the manipulation did not produce a conservative field"
    elif gap > 2 * pooled:
        verdict = ("NON-CONSERVATIVITY DOES WORK -- forcing a potential costs performance, "
                   "so the discovered rule is useful partly because it left the gradient class")
    elif gap < -2 * pooled:
        verdict = "CONSERVATIVE IS BETTER -- the shipped semi-gradient is leaving performance behind"
    else:
        verdict = "INCIDENTAL -- no measurable cost to forcing a potential here"
    print(f"\nVerdict: {verdict}")

    out = {
        "env": args.env, "steps": args.steps, "seeds": args.seeds, "lr": args.lr,
        "manipulation": check,
        "semi_tail": s_tail.tolist(), "full_tail": f_tail.tolist(),
        "gap": float(gap), "sigma": float(gap / pooled),
        "semi_curve_mean": S.mean(0).tolist(), "full_curve_mean": F.mean(0).tolist(),
        "verdict": verdict,
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
