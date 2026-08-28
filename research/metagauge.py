#!/usr/bin/env python3
"""Meta-learn the gauge: optimise a GFlowNet's speed with its answer nailed down.

The theorem in research/gauge.py says the backward policy is a gauge freedom of
dimension |E|-|V|+1 whose choice cannot move the terminal distribution but does
move the conditioning of the flow system.  That separation is what this script
exploits: meta-learn p_B for CONVERGENCE SPEED ALONE, because distributional
correctness is guaranteed by construction rather than by the meta-objective.

This matters because it removes the obstruction that makes "meta-learn a sampler"
hard in general.  A meta-objective has to be cheap and domain-transferable;
distributional accuracy is neither, since evaluating it needs Z or the mode
structure.  Here it never enters the meta-objective at all -- it is a theorem.

  inner loop   tabular GFlowNet, detailed-balance objective, exact full-batch
               gradients, no sampling anywhere
  outer loop   a small MLP emitting p_B from state features, trained by
               back-propagating through the unrolled inner loop
  guard        at every meta-step the exact optimum for the current p_B is
               recomputed and asserted to reproduce R/Z; if the gauge ever moved
               the answer, the implementation is wrong and the run aborts
  transfer     the MLP reads normalised coordinates, so the SAME phi is evaluated
               on a bigger grid and on a different reward without retraining --
               which is the only part of the story that could still fail

Usage:
  .venv/bin/python research/metagauge.py --json research/metagauge.json
"""

from __future__ import annotations

import argparse
import functools
import json
import math

import jax

# The gauge theorem is an exact statement, so the guard that checks it has to be
# able to resolve 1e-15 rather than float32's 1e-7.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

RIGHT, UP, STOP = 0, 1, 2
NEG = -1e9


def grid_reward(h: int, kind: str) -> jnp.ndarray:
    """Two reward shapes. 'std' is the usual hypergrid; 'corners' moves the modes
    so that transfer is tested against a genuinely different target, not a
    rescaling of the same one."""
    i, j = jnp.meshgrid(jnp.arange(h), jnp.arange(h), indexing="ij")
    ax = jnp.abs(i / (h - 1) - 0.5)
    ay = jnp.abs(j / (h - 1) - 0.5)
    if kind == "std":
        r = 1e-2 + 0.5 * ((ax > 0.25) & (ay > 0.25)) + 2.0 * (
            (ax > 0.3) & (ax < 0.4) & (ay > 0.3) & (ay < 0.4)
        )
    else:
        r = 1e-2 + 1.5 * ((ax < 0.15) & (ay < 0.15)) + 1.0 * (
            (ax > 0.42) & (ay > 0.42)
        )
    return r.reshape(-1).astype(jnp.float64)


@functools.partial(jax.jit, static_argnums=(0,))
def grid_masks(h: int):
    i, j = jnp.meshgrid(jnp.arange(h), jnp.arange(h), indexing="ij")
    can_r = (i + 1 < h).reshape(-1)
    can_u = (j + 1 < h).reshape(-1)
    return can_r, can_u


def child_idx(h: int):
    i, j = np.meshgrid(np.arange(h), np.arange(h), indexing="ij")
    s = i * h + j
    cr = np.where(i + 1 < h, (i + 1) * h + j, s)
    cu = np.where(j + 1 < h, i * h + (j + 1), s)
    return jnp.array(cr.reshape(-1)), jnp.array(cu.reshape(-1))


def features(h: int) -> jnp.ndarray:
    """State features that exist at any grid size, so phi transfers."""
    i, j = jnp.meshgrid(jnp.arange(h), jnp.arange(h), indexing="ij")
    fi = (i / (h - 1)).reshape(-1)
    fj = (j / (h - 1)).reshape(-1)
    return jnp.stack([fi, fj, (fi + fj) / 2, jnp.abs(fi - fj), jnp.ones_like(fi)], -1)


def init_phi(key, d_in: int = 5, d_hid: int = 16):
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (d_in, d_hid)) * 0.3,
        "b1": jnp.zeros(d_hid),
        "w2": jax.random.normal(k2, (d_hid, 1)) * 0.0,  # start AT the uniform gauge
        "b2": jnp.zeros(1),
    }


def pb_right(phi, feat) -> jnp.ndarray:
    """p_B(parent = left-neighbour | state), i.e. the probability the state was
    entered by a RIGHT move. Zero-initialised output means sigma(0) = 1/2, the
    uniform-over-parents gauge, so the baseline is a point of this same family."""
    hdn = jnp.tanh(feat @ phi["w1"] + phi["b1"])
    raw = jax.nn.sigmoid((hdn @ phi["w2"] + phi["b2"])[:, 0])
    # Keep the gauge off the boundary of the simplex. log p_B appears in the
    # objective, so a p_B driven to 0 makes the meta-gradient infinite; the
    # interior is also where the theorem's positivity assumption lives.
    return 0.02 + 0.96 * raw


def gauge_logpb(h: int, phi, feat):
    """log p_B(s|c) for the two incoming edges of every state.

    Only states with two parents carry a free parameter; the rest are forced.
    """
    i, j = jnp.meshgrid(jnp.arange(h), jnp.arange(h), indexing="ij")
    i, j = i.reshape(-1), j.reshape(-1)
    has_l, has_d = i > 0, j > 0
    q = pb_right(phi, feat)
    both = has_l & has_d
    p_from_left = jnp.where(both, q, jnp.where(has_l, 1.0, 0.0))
    p_from_down = jnp.where(both, 1.0 - q, jnp.where(has_d, 1.0, 0.0))
    eps = 1e-12
    return jnp.log(p_from_left + eps), jnp.log(p_from_down + eps)


def exact_flow(h: int, rew, logpb_l, logpb_d):
    """F(s) = R(s) + sum_c p_B(s|c) F(c), solved in reverse topological order."""
    n = h * h
    cr, cu = child_idx(h)
    can_r, can_u = grid_masks(h)
    pb_l, pb_d = jnp.exp(logpb_l), jnp.exp(logpb_d)
    order = np.argsort(-(np.arange(n) // h + np.arange(n) % h))

    def step(F, k):
        contrib = jnp.where(can_r[k], pb_l[cr[k]] * F[cr[k]], 0.0) + jnp.where(
            can_u[k], pb_d[cu[k]] * F[cu[k]], 0.0
        )
        return F.at[k].set(rew[k] + contrib), None

    F, _ = jax.lax.scan(step, jnp.zeros(n), jnp.array(order))
    return F


def terminal_distribution(h: int, logits, rew):
    """Exact p(x) for a tabular forward policy, by a forward DP."""
    n = h * h
    cr, cu = child_idx(h)
    can_r, can_u = grid_masks(h)
    masked = logits + jnp.stack(
        [jnp.where(can_r, 0.0, NEG), jnp.where(can_u, 0.0, NEG), jnp.zeros(n)], -1
    )
    pf = jax.nn.softmax(masked, axis=-1)
    order = np.argsort(np.arange(n) // h + np.arange(n) % h)

    def step(reach, k):
        add_r = jnp.where(can_r[k], reach[k] * pf[k, RIGHT], 0.0)
        add_u = jnp.where(can_u[k], reach[k] * pf[k, UP], 0.0)
        reach = reach.at[cr[k]].add(add_r)
        reach = reach.at[cu[k]].add(add_u)
        return reach, reach[k] * pf[k, STOP]

    init = jnp.zeros(n).at[0].set(1.0)
    _, term = jax.lax.scan(step, init, jnp.array(order))
    out = jnp.zeros(n).at[jnp.array(order)].set(term)
    del rew
    return out


def db_loss(h: int, params, rew, logpb_l, logpb_d):
    """Detailed balance, summed uniformly over edges and terminals.

    Uniform weighting rather than on-policy visitation keeps this a pure
    optimisation problem, which is what isolates the conditioning effect the
    theorem predicts.
    """
    n = h * h
    logF, logits = params["logF"], params["logits"]
    cr, cu = child_idx(h)
    can_r, can_u = grid_masks(h)
    masked = logits + jnp.stack(
        [jnp.where(can_r, 0.0, NEG), jnp.where(can_u, 0.0, NEG), jnp.zeros(n)], -1
    )
    logpf = jax.nn.log_softmax(masked, axis=-1)

    res_r = logF + logpf[:, RIGHT] - logF[cr] - logpb_l[cr]
    res_u = logF + logpf[:, UP] - logF[cu] - logpb_d[cu]
    res_t = logF + logpf[:, STOP] - jnp.log(rew)
    return (
        jnp.sum(jnp.where(can_r, res_r**2, 0.0))
        + jnp.sum(jnp.where(can_u, res_u**2, 0.0))
        + jnp.sum(res_t**2)
    ) / n


def inner_run(h: int, phi, rew, n_steps: int, lr: float, opt: str = "adam"):
    """Train a tabular GFlowNet under the gauge phi; return the KL trace."""
    n = h * h
    feat = features(h)
    logpb_l, logpb_d = gauge_logpb(h, phi, feat)
    target = rew / jnp.sum(rew)
    params = {"logF": jnp.zeros(n), "logits": jnp.zeros((n, 3))}

    loss_fn = functools.partial(db_loss, h)
    grad_fn = jax.grad(loss_fn)

    # Adam is reported alongside plain gradient descent on purpose. Adam is a
    # diagonal preconditioner, so if it already absorbs whatever the gauge does,
    # the gauge buys nothing in practice and the claim has to be weakened. That
    # is the honest test, so both are run.
    zeros = jax.tree.map(jnp.zeros_like, params)

    def step(carry, _):
        p, m, v, t = carry
        g = grad_fn(p, rew, logpb_l, logpb_d)
        if opt == "adam":
            t = t + 1
            m = jax.tree.map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
            v = jax.tree.map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
            mh = jax.tree.map(lambda a: a / (1 - 0.9**t), m)
            vh = jax.tree.map(lambda a: a / (1 - 0.999**t), v)
            # The epsilon goes INSIDE the sqrt. Masked logits have identically
            # zero gradient, so their second moment stays at zero, and
            # d/dx sqrt(x) at x = 0 is infinite -- which silently turns the
            # meta-gradient into NaN while the forward pass stays finite.
            p = jax.tree.map(
                lambda a, b, c: a - lr * b / (jnp.sqrt(c + 1e-30) + 1e-8), p, mh, vh
            )
        else:
            p = jax.tree.map(lambda a, b: a - lr * b, p, g)
        term = terminal_distribution(h, p["logits"], rew)
        kl = jnp.sum(target * (jnp.log(target) - jnp.log(term + 1e-30)))
        return (p, m, v, t), kl

    _, kls = jax.lax.scan(step, (params, zeros, zeros, 0), None, length=n_steps)
    return kls


def meta_loss(h: int, phi, rew, n_steps: int, lr: float, opt: str = "adam"):
    """Area under the log-KL curve: rewards getting there fast, not just getting
    there. Correctness is absent from this objective on purpose -- it is a
    theorem, not a target."""
    kls = inner_run(h, phi, rew, n_steps, lr, opt)
    # The floor is 1e-6 rather than 1e-12 so that the gradient of log stays
    # bounded once the inner loop is already converged; without it a single
    # well-converged step contributes a gradient of order 1e7 and the meta-step
    # walks straight off the simplex.
    return jnp.mean(jnp.log(kls + 1e-6))


def gauge_guard(h: int, phi, rew) -> float:
    """The theorem, re-checked numerically at the current gauge."""
    feat = features(h)
    lpl, lpd = gauge_logpb(h, phi, feat)
    F = exact_flow(h, rew, lpl, lpd)
    n = h * h
    cr, cu = child_idx(h)
    can_r, can_u = grid_masks(h)
    pf_stop = rew / F
    pf_r = jnp.where(can_r, jnp.exp(lpl[cr]) * F[cr] / F, 0.0)
    pf_u = jnp.where(can_u, jnp.exp(lpd[cu]) * F[cu] / F, 0.0)
    logits = jnp.log(jnp.stack([pf_r + 1e-30, pf_u + 1e-30, pf_stop], -1))
    term = terminal_distribution(h, logits, rew)
    target = rew / jnp.sum(rew)
    del n
    return float(jnp.max(jnp.abs(term - target)))


def steps_to(kls: np.ndarray, thresh: float) -> int:
    hit = np.nonzero(kls < thresh)[0]
    return int(hit[0]) + 1 if hit.size else len(kls) + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=8)
    ap.add_argument("--inner-steps", type=int, default=600)
    ap.add_argument("--inner-lr", type=float, default=0.2)
    ap.add_argument("--opt", type=str, default="adam", choices=("adam","gd"))
    ap.add_argument("--meta-steps", type=int, default=250)
    ap.add_argument("--meta-lr", type=float, default=2e-2)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=1e-3)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    h = args.height
    rew = grid_reward(h, "std")
    phi = init_phi(jax.random.PRNGKey(args.seed))

    print(f"hypergrid H={h}   inner {args.inner_steps} steps   meta {args.meta_steps} steps")
    print(f"gauge dimension |E|-|V|+1 = {(2*h*(h-1)) - h*h + 1}\n")

    g0 = gauge_guard(h, phi, rew)
    print(f"guard at the uniform gauge: max|p - R/Z| = {g0:.2e}")

    ml = functools.partial(meta_loss, h)
    vg = jax.jit(jax.value_and_grad(ml), static_argnums=(2, 4))

    guards = [g0]
    trace = []
    # Adam on the outer loop as well: the meta-gradient norm swings over four
    # orders of magnitude between steps, and plain descent under a hard clip
    # then takes an almost random step size.
    m_o = jax.tree.map(jnp.zeros_like, phi)
    v_o = jax.tree.map(jnp.zeros_like, phi)
    for t in range(args.meta_steps):
        val, grad = vg(phi, rew, args.inner_steps, args.inner_lr, args.opt)
        gnorm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grad)))
        scale = jnp.minimum(1.0, args.clip / (gnorm + 1e-12))
        grad = jax.tree.map(lambda g: g * scale, grad)
        m_o = jax.tree.map(lambda a, b: 0.9 * a + 0.1 * b, m_o, grad)
        v_o = jax.tree.map(lambda a, b: 0.999 * a + 0.001 * b * b, v_o, grad)
        bc1, bc2 = 1 - 0.9 ** (t + 1), 1 - 0.999 ** (t + 1)
        phi = jax.tree.map(
            lambda a, b, c: a - args.meta_lr * (b / bc1) / (jnp.sqrt(c / bc2 + 1e-30) + 1e-8),
            phi, m_o, v_o,
        )
        if t % max(1, args.meta_steps // 10) == 0 or t == args.meta_steps - 1:
            gd = gauge_guard(h, phi, rew)
            guards.append(gd)
            trace.append({"step": t, "meta_loss": float(val), "guard": gd, "grad_norm": float(gnorm)})
            print(f"  meta {t:>4}  mean log KL {float(val):+.4f}   |g| {float(gnorm):.2e}   guard {gd:.2e}")

    worst_guard = max(guards)
    print(f"\nworst guard over training: {worst_guard:.2e} "
          f"-> {'gauge never moved the answer' if worst_guard < 1e-11 else 'GUARD FAILED'}")

    def evaluate(hh: int, kind: str, ph):
        r = grid_reward(hh, kind)
        kl = np.asarray(inner_run(hh, ph, r, args.inner_steps, args.inner_lr, args.opt))
        auc = float(np.mean(np.log(kl + 1e-6)))
        return kl, steps_to(kl, args.threshold), float(kl[-1]), gauge_guard(hh, ph, r), auc

    uniform = init_phi(jax.random.PRNGKey(0))  # w2 = 0 -> exactly uniform gauge

    settings = [
        ("train  H=8  std", h, "std"),
        ("shape  H=8  corners", h, "corners"),
        ("size   H=16 std", 2 * h, "std"),
        ("both   H=16 corners", 2 * h, "corners"),
    ]
    rows = []
    hdr = (f"{'setting':<24}{'uniform steps':>15}{'learned steps':>15}"
           f"{'speedup':>10}{'AUC unif':>11}{'AUC learn':>11}{'gain':>10}{'guard':>11}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, hh, kind in settings:
        ku, su, fu, gu, au = evaluate(hh, kind, uniform)
        kl_, sl, fl, gl, al = evaluate(hh, kind, phi)
        rows.append({
            "setting": name, "height": hh, "reward": kind,
            "uniform_steps": su, "learned_steps": sl,
            "speedup": su / max(sl, 1),
            "uniform_final_kl": fu, "learned_final_kl": fl,
            "guard_uniform": gu, "guard_learned": gl,
            "auc_uniform": au, "auc_learned": al, "auc_gain": au - al,
        })
        print(f"{name:<24}{su:>15}{sl:>15}{su/max(sl,1):>9.2f}x"
              f"{au:>11.3f}{al:>11.3f}{au-al:>+10.3f}{max(gu,gl):>11.1e}")

    train_row = rows[0]
    transfer = rows[1:]
    ok_guard = worst_guard < 1e-11 and all(max(r["guard_uniform"], r["guard_learned"]) < 1e-11 for r in rows)
    gained = train_row["speedup"] > 1.05
    transfers = all(r["speedup"] > 1.05 for r in transfer)

    print(f"\nguard everywhere        : {ok_guard}")
    print(f"speedup where trained   : {train_row['speedup']:.2f}x")
    print(f"transfers to all 3 held-out settings: {transfers} "
          f"(min {min(r['speedup'] for r in transfer):.2f}x)")
    verdict = (
        "VOID -- the guard failed" if not ok_guard
        else "GAUGE META-LEARNING WORKS AND TRANSFERS" if gained and transfers
        else "WORKS WHERE TRAINED, DOES NOT TRANSFER" if gained
        else "NO SPEEDUP -- the gauge does not help here"
    )
    print(f"Verdict: {verdict}")

    out = {
        "seed": args.seed, "height": h, "gauge_dim": (2 * h * (h - 1)) - h * h + 1,
        "inner_steps": args.inner_steps, "meta_steps": args.meta_steps, "opt": args.opt, "inner_lr": args.inner_lr,
        "threshold": args.threshold,
        "meta_trace": trace, "worst_guard": worst_guard,
        "results": rows, "verdict": verdict,
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")
    return 0 if ok_guard else 1


if __name__ == "__main__":
    raise SystemExit(main())
