#!/usr/bin/env python3
"""Do transformers really perform gradient descent in context?

The claim is that a transformer layer, given in-context examples, implements a
step of gradient descent on an implicit objective. The evidence usually offered
is behavioural: the model's predictions track those of gradient descent. That is
weak, because two different algorithms can produce similar predictions.

A gradient step admits a structural test that behaviour does not. If a map is
v = -eta grad(Phi) then its Jacobian is -eta times a Hessian and is therefore
SYMMETRIC. And if it is a gradient step in some other metric, v = -M grad(Phi)
with M positive definite, then J = -M H is similar to -M^{1/2} H M^{1/2} and
therefore has REAL SPECTRUM. So:

    kappa = ||sym J||_F / ||J||_F = 1     necessary for a gradient step
    spectrum of J real                    necessary in ANY metric

The second is the stronger statement: complex eigenvalues cannot be argued away
by choosing a different parameterisation. Here the state space is small enough
to diagonalise exactly, so no stochastic estimator is needed.

TWO DESIGN DECISIONS, both of which change the answer.

1.  Which variable. The construction of von Oswald et al. is a gradient step in
    the y-slots of the CONTEXT tokens, with the x-slots held fixed and the query
    slot acting as a readout. Include the x-slots and the Jacobian acquires an
    asymmetric off-diagonal block that has nothing to do with the claim; include
    the query slot and the matrix picks up a zero column, since the query's own
    y never feeds back. We therefore measure on the context y-block and report
    the wider restrictions alongside, rather than picking whichever flatters.

2.  Magnitude as well as direction. In an earlier audit of a meta-learned RL
    rule we found kappa = 0.71 for the bootstrap term while the term itself was
    1% of the update, so the update was 0.77% non-conservative -- a two order of
    magnitude gap between what kappa says and what it sounds like. The same trap
    exists here, so every kappa is reported with the norm it applies to.

Positive controls run first and the script aborts if they fail:
    A  an explicit gradient step on a quadratic -- kappa must be exactly 1
    B  the linear-attention construction that provably implements GD -- same

Usage:
  .venv/bin/python research/icl_kappa.py --json research/icl_kappa.json
"""

from __future__ import annotations

import argparse
import functools
import json
import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


# --------------------------------------------------------------------------
# Analysis of a single exact Jacobian.
# --------------------------------------------------------------------------


def analyse(J: np.ndarray) -> dict:
    """kappa, spectrum, and the magnitudes both of them apply to.

    Exact: J is small enough to form and diagonalise, so nothing here is an
    estimate.
    """
    S = 0.5 * (J + J.T)
    A = 0.5 * (J - J.T)
    nJ = float(np.linalg.norm(J))
    kappa = float(np.linalg.norm(S) / nJ) if nJ > 0 else float("nan")

    ev = np.linalg.eigvals(J)
    im = np.abs(ev.imag)
    # A gradient flow in ANY positive-definite metric has real spectrum, so any
    # imaginary part at all is a metric-free refutation. Report the share of
    # eigenvalue magnitude that is imaginary, and how many eigenvalues carry it.
    mag = np.abs(ev)
    tot = float(mag.sum())
    imag_share = float(im.sum() / tot) if tot > 0 else float("nan")
    # Threshold against the spectral radius, not against each eigenvalue's own
    # real part. A rank-deficient symmetric matrix has near-zero eigenvalues
    # whose numerical imaginary parts are large relative to themselves and
    # meaningless relative to the matrix, and a per-eigenvalue relative test
    # reports those as complex.
    radius = float(mag.max()) if mag.size else 0.0
    n_complex = int((im > 1e-8 * max(radius, 1e-300)).sum())
    return {
        "dim": int(J.shape[0]),
        "kappa": kappa,
        "norm_J": nJ,
        "norm_asym": float(np.linalg.norm(A)),
        "asym_share": float(np.linalg.norm(A) / nJ) if nJ > 0 else float("nan"),
        "imag_share_of_spectrum": imag_share,
        "n_complex_eigenvalues": n_complex,
        "n_eigenvalues": int(ev.size),
        "max_abs_imag": float(im.max()) if im.size else 0.0,
        "spectral_radius": float(mag.max()) if mag.size else 0.0,
    }


def show(name: str, r: dict) -> None:
    print(
        f"  {name:<34} kappa={r['kappa']:.6f}  asym={r['asym_share']:.4%}  "
        f"complex {r['n_complex_eigenvalues']:>3}/{r['n_eigenvalues']:<3} "
        f"imag share {r['imag_share_of_spectrum']:.2e}  dim {r['dim']}"
    )


# --------------------------------------------------------------------------
# Task and controls.
# --------------------------------------------------------------------------


def sample_batch(key, n_ctx: int, d: int, batch: int):
    """In-context linear regression: y = w . x, w resampled per sequence."""
    kx, kw, kq = jax.random.split(key, 3)
    x = jax.random.normal(kx, (batch, n_ctx, d))
    w = jax.random.normal(kw, (batch, d)) / math.sqrt(d)
    y = jnp.einsum("bnd,bd->bn", x, w)
    xq = jax.random.normal(kq, (batch, d))
    yq = jnp.einsum("bd,bd->b", xq, w)
    return x, y, xq, yq


def tokens_of(x, y, xq):
    """Token j is (x_j, y_j); the query token carries y = 0."""
    ctx = jnp.concatenate([x, y[..., None]], axis=-1)
    q = jnp.concatenate([xq, jnp.zeros_like(xq[..., :1])], axis=-1)
    return jnp.concatenate([ctx, q[:, None, :]], axis=1)


def task_scale(key, n_ctx: int, d: int, batch: int, ridge: float = 1e-6):
    """Variance of the target, and the loss of the least-squares predictor.

    The convergence guard has to be scale-free. The task here is noiseless, so
    with n_ctx > d the least-squares predictor is exact and its loss is at
    machine precision; quoting a ratio against it makes every model look
    infinitely bad and says nothing. What is meaningful is the fraction of
    target variance a model leaves unexplained,

        R^2 = 1 - loss / Var(y_q),

    for which the least-squares predictor sits at 1. Asking whether a model
    implements gradient descent on a problem it has not solved is premature, so
    R^2 gates the structural question.
    """
    x, y, xq, yq = sample_batch(key, n_ctx, d, batch)
    xtx = jnp.einsum("bnd,bne->bde", x, x) + ridge * jnp.eye(d)
    xty = jnp.einsum("bnd,bn->bd", x, y)
    w = jnp.linalg.solve(xtx, xty[..., None]).squeeze(-1)
    pred = jnp.einsum("bd,bd->b", xq, w)
    return float(jnp.var(yq)), float(jnp.mean((pred - yq) ** 2))


def control_A(n: int, key) -> dict:
    """An explicit gradient step on a quadratic. kappa must be exactly 1."""
    M = jax.random.normal(key, (n, n))
    H = np.asarray((M + M.T) / 2 + n * jnp.eye(n))  # symmetric, well conditioned
    eta = 0.05
    return analyse(-eta * H)


def control_B(x: np.ndarray, eta: float = 0.05) -> dict:
    """The linear-attention construction that provably implements one GD step.

    With W_0 = 0 the update on the context y-slots is
        Delta y_j = eta * sum_i y_i (x_i . x_j),
    so the Jacobian on that block is eta K with K the Gram matrix -- symmetric
    and positive semidefinite by construction, hence kappa = 1 and real spectrum.
    This checks the instrument against a case where the claim is a theorem.
    """
    K = x @ x.T
    return analyse(eta * K)


# --------------------------------------------------------------------------
# A small transformer, trained.
# --------------------------------------------------------------------------


def init_params(key, layers: int, dt: int, softmax: bool):
    ps = []
    for _ in range(layers):
        k1, k2, k3, k4, key = jax.random.split(key, 5)
        s = 1.0 / math.sqrt(dt)
        ps.append({
            "Q": jax.random.normal(k1, (dt, dt)) * s,
            "K": jax.random.normal(k2, (dt, dt)) * s,
            "V": jax.random.normal(k3, (dt, dt)) * s,
            # The output projection starts near zero so every layer begins as
            # the identity. Deep residual linear attention without any
            # normalisation diverges otherwise, and a diverged run tells us
            # nothing about whether a converged one is a gradient step.
            "P": jax.random.normal(k4, (dt, dt)) * s * 1e-2,
        })
    del softmax
    return ps


def layer_apply(p, e, softmax: bool):
    """One residual self-attention layer. e: [T, dt]."""
    q = e @ p["Q"].T
    k = e @ p["K"].T
    v = e @ p["V"].T
    logits = q @ k.T / math.sqrt(e.shape[-1])
    a = jax.nn.softmax(logits, axis=-1) if softmax else logits / e.shape[0]
    return e + (a @ v) @ p["P"].T


def forward(params, e, softmax: bool):
    for p in params:
        e = layer_apply(p, e, softmax)
    return e


def predict(params, e, softmax: bool):
    return forward(params, e, softmax)[-1, -1]  # query token's y-slot


def train(key, layers: int, d: int, n_ctx: int, softmax: bool, steps: int, batch: int, lr: float):
    dt = d + 1
    params = init_params(key, layers, dt, softmax)

    def loss_fn(params, x, y, xq, yq):
        e = tokens_of(x, y, xq)
        pred = jax.vmap(lambda ee: predict(params, ee, softmax))(e)
        return jnp.mean((pred - yq) ** 2)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn), static_argnums=())
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    curve = []
    snaps = {}
    marks = {max(1, int(steps * f)) for f in (0.02, 0.1, 0.3, 0.6, 1.0)}
    for t in range(1, steps + 1):
        key, k = jax.random.split(key)
        xb, yb, xqb, yqb = sample_batch(k, n_ctx, d, batch)
        val, g = grad_fn(params, xb, yb, xqb, yqb)
        m = jax.tree.map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree.map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
        mh = jax.tree.map(lambda a: a / (1 - 0.9**t), m)
        vh = jax.tree.map(lambda a: a / (1 - 0.999**t), v)
        params = jax.tree.map(
            lambda a, b, c: a - lr * b / (jnp.sqrt(c + 1e-30) + 1e-8), params, mh, vh
        )
        curve.append(float(val))
        if not np.isfinite(curve[-1]):
            print(f"    diverged at step {t}; stopping. Lower --lr or --layers.")
            return params, curve, snaps
        if t in marks:
            snaps[t] = jax.tree.map(lambda a: a.copy(), params)
    return params, curve, snaps


def layer_jacobians(params, e0, softmax: bool, n_ctx: int, d: int) -> list[dict]:
    """Exact Jacobian of each layer, restricted three ways.

    ctx-y   the context y-slots: the variable the GD claim is actually about
    all-y   context plus query y-slot: adds a zero column, since the query's own
            y never feeds back into anything
    full    the entire residual stream: a strictly stronger requirement, since a
            gradient step on a latent variable is a gradient field on the stream
            only when read-in and write-out are transposes
    """
    out = []
    e = e0
    for li, p in enumerate(params):
        def step(x, p=p):
            return layer_apply(p, x, softmax) - x  # the update, not the map

        J_full = np.asarray(jax.jacobian(step)(e)).reshape(e.size, e.size)
        idx_y = np.arange(e.shape[0]) * e.shape[1] + d       # y-slot of each token
        idx_ctx = idx_y[:n_ctx]
        rec = {
            "layer": li,
            "ctx_y": analyse(J_full[np.ix_(idx_ctx, idx_ctx)]),
            "all_y": analyse(J_full[np.ix_(idx_y, idx_y)]),
            "full": analyse(J_full),
        }
        out.append(rec)
        e = layer_apply(p, e, softmax)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=8)
    ap.add_argument("--n-ctx", type=int, default=12)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    key = jax.random.PRNGKey(0)
    print("=" * 78)
    print("STEP 1  positive controls -- the instrument must pass these first")
    print("=" * 78)
    key, k1, k2 = jax.random.split(key, 3)
    a = control_A(args.n_ctx, k1)
    xs = np.asarray(jax.random.normal(k2, (args.n_ctx, args.d)))
    b = control_B(xs)
    show("A  explicit gradient step", a)
    show("B  linear-attention GD construction", b)
    ok = abs(a["kappa"] - 1) < 1e-10 and abs(b["kappa"] - 1) < 1e-10
    ok = ok and a["n_complex_eigenvalues"] == 0 and b["n_complex_eigenvalues"] == 0
    print(f"\n  controls {'PASS' if ok else 'FAIL -- everything below is void'}")
    if not ok:
        return 1

    print("\n" + "=" * 78)
    print("STEP 2  convergence guard")
    print("=" * 78)
    key, kr = jax.random.split(key)
    var_y, ols = task_scale(kr, args.n_ctx, args.d, 4096)
    print(f"  Var(y_q) = {var_y:.4f}   least-squares loss = {ols:.3e} (noiseless task)")
    print("  the guard is R^2 = 1 - loss/Var(y_q); least squares sits at 1, and a")
    print("  model well below it has not solved the task the claim is about\n")
    ref = var_y

    print("=" * 78)
    print("STEP 3  trained transformers, and kappa along the training trajectory")
    print("=" * 78)
    results = {}
    for tag, softmax in (("linear attention", False), ("softmax attention", True)):
        key, kt, ke = jax.random.split(key, 3)
        params, curve, snaps = train(
            kt, args.layers, args.d, args.n_ctx, softmax, args.steps, args.batch, args.lr
        )
        x, y, xq, _ = sample_batch(ke, args.n_ctx, args.d, 1)
        e0 = tokens_of(x, y, xq)[0]
        base = float(np.mean(curve[:20]))
        final = float(np.mean(curve[-100:]))
        r2 = 1.0 - final / ref
        solved = r2 > 0.95
        print(f"\n{tag}:  loss {base:.4f} -> {final:.4f}   R^2 = {r2:.4f}"
              f"  -- {'SOLVED' if solved else 'NOT SOLVED, read with care'}")

        traj = []
        for t in sorted(snaps):
            recs = layer_jacobians(snaps[t], e0, softmax, args.n_ctx, args.d)
            ks = [r["ctx_y"]["kappa"] for r in recs]
            cs = [r["ctx_y"]["n_complex_eigenvalues"] for r in recs]
            ims = [r["ctx_y"]["imag_share_of_spectrum"] for r in recs]
            lo = float(np.mean(curve[max(0, t - 50):t]))
            traj.append({"step": t, "loss": lo, "r2": 1.0 - lo / ref,
                         "kappa": ks, "complex": cs, "imag_share": ims})
            print(f"  step {t:>6}  R2 {traj[-1]['r2']:+.3f}   "
                  f"kappa " + " ".join(f"{k:.4f}" for k in ks)
                  + "   complex " + " ".join(str(c) for c in cs)
                  + "   imag " + " ".join(f"{i:.1e}" for i in ims))

        recs = layer_jacobians(params, e0, softmax, args.n_ctx, args.d)
        print("  final, all three restrictions:")
        for r in recs:
            for restriction in ("ctx_y", "all_y", "full"):
                show(f"    L{r['layer']} {restriction}", r[restriction])
        results[tag] = {
            "loss_start": base, "loss_final": final, "var_y": ref,
            "r2": r2, "solved": bool(solved),
            "trajectory": traj, "layers": recs,
        }

    print("\n" + "=" * 78)
    print("STEP 4  reading")
    print("=" * 78)
    for tag, res in results.items():
        ks = [r["ctx_y"]["kappa"] for r in res["layers"]]
        cs = [r["ctx_y"]["n_complex_eigenvalues"] for r in res["layers"]]
        ims = [r["ctx_y"]["imag_share_of_spectrum"] for r in res["layers"]]
        print(f"{tag}   (R^2 = {res['r2']:.4f}, "
              f"{'solved' if res['solved'] else 'NOT solved'})")
        print(f"  ctx-y kappa        : " + ", ".join(f"{k:.4f}" for k in ks))
        print(f"  complex eigenvalues: " + ", ".join(str(c) for c in cs))
        print(f"  imaginary share    : " + ", ".join(f"{i:.2e}" for i in ims))
        if not res["solved"]:
            v = "INCONCLUSIVE -- the model has not solved the task"
        elif all(c == 0 for c in cs) and all(k > 0.999 for k in ks):
            v = "consistent with a gradient step in the context y-variables"
        elif all(c == 0 for c in cs):
            v = "real spectrum but asymmetric: a gradient flow in some other metric is not excluded"
        else:
            v = "COMPLEX SPECTRUM -- not a gradient flow in ANY positive-definite metric"
        print(f"  -> {v}\n")
        results[tag]["verdict"] = v

    out = {
        "config": vars(args),
        "controls": {"explicit_gd": a, "linear_attention_construction": b},
        "results": results,
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
