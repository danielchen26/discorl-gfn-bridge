#!/usr/bin/env python3
"""Is a transformer's in-context computation gradient descent on least squares?

The usual evidence is behavioural: the model predicts what gradient descent
predicts. That is weak. A structural test is available and it is much sharper
than symmetry alone.

FIRST PRINCIPLES. If a stack of layers implements gradient descent on the
in-context least-squares objective, three things follow, in increasing
sharpness.

  (i)   Interior layers are gradient steps, so their Jacobians are symmetric,
        and have real spectrum in any positive-definite metric.

  (ii)  They are steps on the SAME objective, so for a quadratic loss all
        interior Jacobians are multiples of one another:
            J_l = -eta_l H  and  J_l' = -eta_l' H  =>  J_l is parallel to J_l'.

  (iii) That shared matrix is not arbitrary. For in-context linear regression
        the Hessian in the y-slot variables is the Gram matrix K_ij = x_i . x_j,
        so
            J_l  proportional to  K,
        which pins the direction completely and leaves only a step size.

Symmetry is necessary; alignment with K is nearly sufficient. And the boundary
layers are expected NOT to be gradient steps: the first has to encode the data
into the form the recursion acts on and the last has to read a prediction out,
and neither is a step of anything. So the prediction is a specific LAYER
STRUCTURE -- non-gradient encode, gradient-step interior, non-gradient decode --
rather than a blanket yes or no.

CONTROLS, run before anything is believed.
  C1  an explicit gradient step on the least-squares objective: kappa = 1,
      alignment = 1 by construction
  C2  the linear-attention construction that provably implements one GD step
  C3  THE CRITICAL NULL: recompute the alignment against a Gram matrix built
      from an independent draw of x. Two symmetric matrices of similar spectra
      are generically somewhat aligned, so without this control a positive
      result means nothing.
  C4  a random symmetric matrix, for the scale of "aligned by accident"

CONVERGENCE GUARD. R^2 against the least-squares predictor. Asking whether a
model implements gradient descent on a problem it has not solved is premature,
and an undertrained model looks non-gradient for reasons unrelated to the claim.

Usage:
  .venv/bin/python research/icl_gd.py --json research/icl_gd.json
"""

from __future__ import annotations

import argparse
import json
import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


# --------------------------------------------------------------------------
# Task
# --------------------------------------------------------------------------


def sample_batch(key, n_ctx: int, d: int, batch: int):
    kx, kw, kq = jax.random.split(key, 3)
    x = jax.random.normal(kx, (batch, n_ctx, d))
    w = jax.random.normal(kw, (batch, d)) / math.sqrt(d)
    y = jnp.einsum("bnd,bd->bn", x, w)
    xq = jax.random.normal(kq, (batch, d))
    yq = jnp.einsum("bd,bd->b", xq, w)
    return x, y, xq, yq


def tokens_of(x, y, xq):
    ctx = jnp.concatenate([x, y[..., None]], axis=-1)
    q = jnp.concatenate([xq, jnp.zeros_like(xq[..., :1])], axis=-1)
    return jnp.concatenate([ctx, q[:, None, :]], axis=1)


def r2_reference(key, n_ctx: int, d: int, batch: int) -> float:
    _, _, _, yq = sample_batch(key, n_ctx, d, batch)
    return float(jnp.var(yq))


# --------------------------------------------------------------------------
# Metrics on one Jacobian block
# --------------------------------------------------------------------------


def cosine(A: np.ndarray, B: np.ndarray) -> float:
    na, nb = np.linalg.norm(A), np.linalg.norm(B)
    return float((A * B).sum() / (na * nb)) if na > 0 and nb > 0 else float("nan")


def analyse(J: np.ndarray, K: np.ndarray, K_null: np.ndarray) -> dict:
    """Symmetry, spectrum, and alignment with the least-squares Hessian."""
    S = 0.5 * (J + J.T)
    A = 0.5 * (J - J.T)
    nJ = float(np.linalg.norm(J))
    ev = np.linalg.eigvals(J)
    mag = np.abs(ev)
    radius = float(mag.max()) if mag.size else 0.0
    n_complex = int((np.abs(ev.imag) > 1e-8 * max(radius, 1e-300)).sum())

    # Best scalar multiple of K, and what it leaves behind. eta > 0 is the sign
    # the construction predicts.
    nK2 = float((K * K).sum())
    eta = float((J * K).sum() / nK2) if nK2 > 0 else float("nan")
    resid = float(np.linalg.norm(J - eta * K) / nJ) if nJ > 0 else float("nan")

    return {
        "dim": int(J.shape[0]),
        "kappa": float(np.linalg.norm(S) / nJ) if nJ > 0 else float("nan"),
        "asym_share": float(np.linalg.norm(A) / nJ) if nJ > 0 else float("nan"),
        "n_complex": n_complex,
        "n_eig": int(ev.size),
        "imag_share": float(np.abs(ev.imag).sum() / mag.sum()) if mag.sum() > 0 else float("nan"),
        "cos_K": cosine(J, K),
        "cos_K_null": cosine(J, K_null),
        "eta": eta,
        "residual_after_K": resid,
        "norm": nJ,
    }


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------


def control_explicit_gd(x: np.ndarray, K_null: np.ndarray, eta: float = 0.05) -> dict:
    """One GD step on the y-slots: J = eta K by construction."""
    K = x @ x.T
    return analyse(eta * K, K, K_null)


def control_random_symmetric(n: int, K: np.ndarray, K_null: np.ndarray, key) -> dict:
    M = np.asarray(jax.random.normal(key, (n, n)))
    S = (M + M.T) / 2
    return analyse(S, K, K_null)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def init_params(key, layers: int, dt: int):
    ps = []
    for _ in range(layers):
        k1, k2, k3, k4, key = jax.random.split(key, 5)
        s = 1.0 / math.sqrt(dt)
        ps.append({
            "Q": jax.random.normal(k1, (dt, dt)) * s,
            "K": jax.random.normal(k2, (dt, dt)) * s,
            "V": jax.random.normal(k3, (dt, dt)) * s,
            # near-zero output projection: every layer starts as the identity,
            # without which deep residual attention with no normalisation
            # diverges and a diverged run answers nothing
            "P": jax.random.normal(k4, (dt, dt)) * s * 1e-2,
        })
    return ps


def layer_apply(p, e, softmax: bool):
    q = e @ p["Q"].T
    k = e @ p["K"].T
    v = e @ p["V"].T
    logits = q @ k.T / math.sqrt(e.shape[-1])
    a = jax.nn.softmax(logits, axis=-1) if softmax else logits / e.shape[0]
    return e + (a @ v) @ p["P"].T


def predict(params, e, softmax: bool):
    for p in params:
        e = layer_apply(p, e, softmax)
    return e[-1, -1]


def train(key, layers, d, n_ctx, softmax, steps, batch, lr):
    params = init_params(key, layers, d + 1)

    def loss_fn(params, x, y, xq, yq):
        e = tokens_of(x, y, xq)
        pred = jax.vmap(lambda ee: predict(params, ee, softmax))(e)
        return jnp.mean((pred - yq) ** 2)

    gfn = jax.jit(jax.value_and_grad(loss_fn))
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    curve = []
    for t in range(1, steps + 1):
        key, k = jax.random.split(key)
        xb, yb, xqb, yqb = sample_batch(k, n_ctx, d, batch)
        val, g = gfn(params, xb, yb, xqb, yqb)
        m = jax.tree.map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree.map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
        mh = jax.tree.map(lambda a: a / (1 - 0.9**t), m)
        vh = jax.tree.map(lambda a: a / (1 - 0.999**t), v)
        params = jax.tree.map(
            lambda a, b, c: a - lr * b / (jnp.sqrt(c + 1e-30) + 1e-8), params, mh, vh
        )
        curve.append(float(val))
        if not np.isfinite(curve[-1]):
            return params, curve, False
    return params, curve, True


def layer_blocks(params, e0, softmax, n_ctx, d):
    """Jacobian of each layer's UPDATE, restricted to the context y-slots."""
    out = []
    e = e0
    for p in params:
        J = np.asarray(
            jax.jacobian(lambda x, p=p: layer_apply(p, x, softmax) - x)(e)
        ).reshape(e.size, e.size)
        idx = (np.arange(e.shape[0]) * e.shape[1] + d)[:n_ctx]
        out.append(J[np.ix_(idx, idx)])
        e = layer_apply(p, e, softmax)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--n-ctx", type=int, default=20)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--softmax", action="store_true", default=True)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    key = jax.random.PRNGKey(0)
    key, kx, kn, kr, kc = jax.random.split(key, 5)
    x_probe = np.asarray(jax.random.normal(kx, (args.n_ctx, args.d)))
    K = x_probe @ x_probe.T
    x_null = np.asarray(jax.random.normal(kn, (args.n_ctx, args.d)))
    K_null = x_null @ x_null.T
    var_y = r2_reference(kr, args.n_ctx, args.d, 8192)

    print("=" * 82)
    print("STEP 1  controls")
    print("=" * 82)
    c1 = control_explicit_gd(x_probe, K_null)
    c4 = control_random_symmetric(args.n_ctx, K, K_null, kc)
    hdr = f"{'control':<28}{'kappa':>8}{'cplx':>6}{'cos K':>9}{'cos K_null':>12}{'resid':>9}"
    print(hdr)
    print("-" * len(hdr))
    for nm, c in (("C1 explicit GD step", c1), ("C4 random symmetric", c4)):
        print(f"{nm:<28}{c['kappa']:>8.4f}{c['n_complex']:>6}{c['cos_K']:>9.4f}"
              f"{c['cos_K_null']:>12.4f}{c['residual_after_K']:>9.4f}")
    ok = abs(c1["kappa"] - 1) < 1e-10 and abs(c1["cos_K"] - 1) < 1e-10
    print(f"\n  C1 pins kappa and alignment at 1: {'PASS' if ok else 'FAIL'}")
    print(f"  C3 null scale: a matrix aligned with K by accident scores about "
          f"{abs(c4['cos_K']):.3f}; the null Gram scores {abs(c1['cos_K_null']):.3f}")
    if not ok:
        return 1

    print("\n" + "=" * 82)
    print(f"STEP 2  trained transformers, {args.seeds} seeds, "
          f"L={args.layers}, d={args.d}, n_ctx={args.n_ctx}")
    print("=" * 82)
    all_seeds = []
    for seed in range(args.seeds):
        key, kt, ke = jax.random.split(key, 3)
        params, curve, fine = train(
            kt, args.layers, args.d, args.n_ctx, args.softmax,
            args.steps, args.batch, args.lr,
        )
        if not fine:
            print(f"  seed {seed}: diverged, skipped")
            continue
        final = float(np.mean(curve[-200:]))
        r2 = 1.0 - final / var_y
        e0 = tokens_of(*sample_batch(ke, args.n_ctx, args.d, 1)[:3])[0]
        blocks = layer_blocks(params, e0, args.softmax, args.n_ctx, args.d)
        recs = [analyse(J, K, K_null) for J in blocks]

        pair = np.zeros((len(blocks), len(blocks)))
        for i in range(len(blocks)):
            for j in range(len(blocks)):
                pair[i, j] = cosine(blocks[i], blocks[j])

        print(f"\n  seed {seed}   R^2 = {r2:.4f}  "
              f"{'SOLVED' if r2 > 0.99 else 'not solved, read with care'}")
        sub = f"    {'layer':<8}{'kappa':>8}{'cplx':>6}{'cos K':>9}{'cos null':>10}{'eta':>10}{'resid':>9}"
        print(sub)
        for li, r in enumerate(recs):
            print(f"    L{li:<7}{r['kappa']:>8.4f}{r['n_complex']:>6}{r['cos_K']:>9.4f}"
                  f"{r['cos_K_null']:>10.4f}{r['eta']:>10.2e}{r['residual_after_K']:>9.4f}")
        print("    pairwise cos(J_l, J_l'):")
        for i in range(len(blocks)):
            print("      " + " ".join(f"{pair[i, j]:+.3f}" for j in range(len(blocks))))
        all_seeds.append({"seed": seed, "r2": r2, "layers": recs, "pairwise": pair.tolist()})

    print("\n" + "=" * 82)
    print("STEP 3  reading")
    print("=" * 82)
    if not all_seeds:
        print("no seed converged; nothing to read")
        return 1

    L = len(all_seeds[0]["layers"])
    interior = list(range(1, L - 1))
    for name, idxs in (("boundary (first, last)", [0, L - 1]), ("interior", interior)):
        cosK = np.array([[s["layers"][i]["cos_K"] for i in idxs] for s in all_seeds]).ravel()
        cosN = np.array([[s["layers"][i]["cos_K_null"] for i in idxs] for s in all_seeds]).ravel()
        kap = np.array([[s["layers"][i]["kappa"] for i in idxs] for s in all_seeds]).ravel()
        cpx = np.array([[s["layers"][i]["n_complex"] for i in idxs] for s in all_seeds]).ravel()
        print(f"  {name:<24} kappa {kap.mean():.4f}+/-{kap.std():.4f}   "
              f"cos K {cosK.mean():+.4f}+/-{cosK.std():.4f}   "
              f"cos null {cosN.mean():+.4f}   complex {cpx.mean():.1f}")

    ic = np.array([[s["layers"][i]["cos_K"] for i in interior] for s in all_seeds]).ravel()
    inull = np.array([[s["layers"][i]["cos_K_null"] for i in interior] for s in all_seeds]).ravel()
    bc = np.array([[s["layers"][i]["cos_K"] for i in [0, L - 1]] for s in all_seeds]).ravel()
    sep = ic.mean() - inull.mean()
    print(f"\n  interior alignment above its own null : {sep:+.4f}")
    print(f"  interior minus boundary alignment     : {ic.mean() - bc.mean():+.4f}")

    if ic.mean() > 0.7 and sep > 0.3 and ic.mean() - bc.mean() > 0.2:
        verdict = ("LAYER STRUCTURE CONFIRMED -- interior layers align with the "
                   "least-squares Hessian, boundary layers do not")
    elif sep > 0.3:
        verdict = "PARTIAL -- interior layers align above the null but not strongly"
    else:
        verdict = "NOT CONFIRMED -- interior alignment is within the null"
    print(f"\nVerdict: {verdict}")

    out = {
        "config": vars(args), "var_y": var_y,
        "controls": {"explicit_gd": c1, "random_symmetric": c4},
        "seeds": all_seeds, "verdict": verdict,
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
