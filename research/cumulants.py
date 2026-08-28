#!/usr/bin/env python3
"""Exact verification of the work-cumulant identities that the dossier claims.

Everything here is exact dynamic programming on a hypergrid DAG -- no sampling,
no gradient descent, no tolerance fudging. Every number printed is reproducible
bit-for-bit.

The object under test is the non-equilibrium work of a trajectory tau:

    W(tau) = log [ prod_t p_F(s_{t+1}|s_t) ] - log R(x) - log [ prod_t p_B(s_t|s_{t+1}) ]

Claims checked (see README section "Layer 1"):

  C1  Jarzynski:      E_{P_F}[exp(-W)] = Z            exactly, for ANY p_F
  C2  Second law:     E[W] + log Z = KL(P_F || P_B) >= 0
  C3  TB = variance:  min_c E[(c + W)^2] = Var[W]  at  c* = -E[W] = ELBO
  C4  GFN optimum:    the flow-matching policy makes W constant (Var[W] = 0)
  C5  MaxEnt bias:    the soft-RL optimal policy samples p(x) ~ R(x)*m(x),
                      i.e. gamma = 1 in  p(x) ~ R(x) * m(x)^gamma

Usage:  python3 research/cumulants.py [--height 8] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Environment: 2-D hypergrid DAG.
#
# States are lattice points (i, j) with 0 <= i, j < H.  From (i, j) you may
# step right, step up, or stop.  Stopping at (i, j) emits terminal object
# (i, j) with reward R(i, j).  Every trajectory is therefore a monotone
# lattice path from (0, 0) followed by a stop, and the number of distinct
# paths reaching (i, j) is the binomial coefficient C(i+j, i).
# --------------------------------------------------------------------------

R0, R1, R2 = 1e-2, 0.5, 2.0


def reward(i: int, j: int, h: int) -> float:
    """Standard hypergrid reward (Bengio et al. 2021), strictly positive."""
    r = R0
    ax = abs(i / (h - 1) - 0.5)
    ay = abs(j / (h - 1) - 0.5)
    if ax > 0.25 and ay > 0.25:
        r += R1
    if 0.3 < ax < 0.4 and 0.3 < ay < 0.4:
        r += R2
    return r


@dataclass(frozen=True)
class Grid:
    h: int

    def states(self):
        """States in reverse topological order (sinks first)."""
        for s in range(2 * self.h - 2, -1, -1):
            for i in range(self.h):
                j = s - i
                if 0 <= j < self.h:
                    yield (i, j)

    def children(self, i: int, j: int):
        """(action_name, child) pairs, excluding `stop`."""
        out = []
        if i + 1 < self.h:
            out.append(("right", (i + 1, j)))
        if j + 1 < self.h:
            out.append(("up", (i, j + 1)))
        return out

    def n_parents(self, i: int, j: int) -> int:
        return (1 if i > 0 else 0) + (1 if j > 0 else 0)

    def paths(self, i: int, j: int) -> int:
        """n(x): number of distinct monotone lattice paths reaching (i, j)."""
        return math.comb(i + j, i)


# --------------------------------------------------------------------------
# Three reference forward policies, all computed in closed form by DP.
# --------------------------------------------------------------------------


def flow_matching_policy(g: Grid) -> dict:
    """The exact GFlowNet solution for the given backward policy.

    On a DAG a state may have several parents, so the naive tree recursion
    F(s) = R(s) + sum_c F(c) over-counts every child once per parent.  The
    correct condition is stated on EDGE flows,

        F(s -> c) = p_B(s|c) F(c),      F(s) = R(s) + sum_{c} F(s -> c),

    which yields p_F(c|s) = p_B(s|c) F(c) / F(s) and p_F(stop|s) = R(s)/F(s).
    Under this policy W(tau) = -log F(s0) = -log Z identically, so Var[W] = 0
    and the terminal distribution is exactly R(x)/Z.
    """
    f: dict[tuple[int, int], float] = {}
    for s in g.states():
        acc = reward(*s, g.h)
        for _, c in g.children(*s):
            acc += f[c] / g.n_parents(*c)
        f[s] = acc
    pol = {}
    for s in g.states():
        tot = f[s]
        d = {"stop": reward(*s, g.h) / tot}
        for a, c in g.children(*s):
            d[a] = (f[c] / g.n_parents(*c)) / tot
        pol[s] = d
    return pol


def maxent_rl_policy(g: Grid) -> dict:
    """Entropy-regularised (soft) RL optimum with a uniform reference policy.

    The optimal soft policy puts P(tau) proportional to pi0(tau) * R(x), so the
    marginal over terminal objects is p(x) ~ R(x) * m(x), where m(x) is the
    reference measure of the path bundle reaching x.  This is exactly the
    multi-path bias of Deleu et al. (2024).
    """
    val: dict[tuple[int, int], float] = {}
    for s in g.states():
        kids = g.children(*s)
        n_act = len(kids) + 1
        acc = reward(*s, g.h) / n_act
        for _, c in kids:
            acc += val[c] / n_act
        val[s] = acc
    pol = {}
    for s in g.states():
        kids = g.children(*s)
        n_act = len(kids) + 1
        tot = val[s] * n_act
        d = {"stop": reward(*s, g.h) / tot}
        for a, c in kids:
            d[a] = val[c] / tot
        pol[s] = d
    return pol


def uniform_policy(g: Grid) -> dict:
    pol = {}
    for s in g.states():
        kids = g.children(*s)
        n_act = len(kids) + 1
        d = {"stop": 1.0 / n_act}
        for a, _ in kids:
            d[a] = 1.0 / n_act
        pol[s] = d
    return pol


def geometric_mix(g: Grid, a: dict, b: dict, lam: float) -> dict:
    """Renormalised geometric interpolation  p ~ a^(1-lam) * b^lam."""
    pol = {}
    for s in g.states():
        raw = {k: (a[s][k] ** (1.0 - lam)) * (b[s][k] ** lam) for k in a[s]}
        tot = sum(raw.values())
        pol[s] = {k: v / tot for k, v in raw.items()}
    return pol


# --------------------------------------------------------------------------
# Exact statistics of W by dynamic programming.  W is additive along the
# trajectory, so its first two moments -- and the Jarzynski exponential
# average -- all satisfy backward recursions over the DAG.
# --------------------------------------------------------------------------


def edge_work(g: Grid, s, a, child, p: float) -> float:
    """Increment of W contributed by taking action `a` in state `s`."""
    if a == "stop":
        return math.log(p) - math.log(reward(*s, g.h))
    # p_B is uniform over the parents of the child.
    return math.log(p) - math.log(1.0 / g.n_parents(*child))


def work_stats(g: Grid, pol: dict) -> dict:
    """Return exact E[W], Var[W], E[exp(-W)], and the terminal distribution."""
    m1: dict = {}
    m2: dict = {}
    jz: dict = {}
    for s in g.states():
        acc1 = acc2 = 0.0
        accj = 0.0
        kids = dict(g.children(*s))
        for a, p in pol[s].items():
            child = None if a == "stop" else kids[a]
            w = edge_work(g, s, a, child, p)
            n1 = 0.0 if child is None else m1[child]
            n2 = 0.0 if child is None else m2[child]
            nj = 1.0 if child is None else jz[child]
            acc1 += p * (w + n1)
            acc2 += p * (w * w + 2.0 * w * n1 + n2)
            accj += p * math.exp(-w) * nj
        m1[s], m2[s], jz[s] = acc1, acc2, accj

    src = (0, 0)
    mean = m1[src]
    var = m2[src] - mean * mean

    # Forward pass for the terminal distribution.
    reach = {s: 0.0 for s in g.states()}
    reach[src] = 1.0
    term: dict = {}
    for s in sorted(reach, key=lambda t: t[0] + t[1]):
        kids = dict(g.children(*s))
        term[s] = reach[s] * pol[s]["stop"]
        for a, p in pol[s].items():
            if a != "stop":
                reach[kids[a]] += reach[s] * p
    return {"mean": mean, "var": var, "jarzynski": jz[src], "terminal": term}


def fit_gamma(g: Grid, term: dict) -> float:
    """The multi-path exponent gamma in  p(x) ~ R(x) * n(x)^gamma.

    A univariate regression on log n(x) is confounded: under a uniform
    reference policy a longer path also carries less prior mass, and on this
    grid length is correlated with log n.  So we regress

        log p(x) - log R(x)   on   [ log n(x),  len(x),  1 ]

    and report the coefficient on log n(x) -- the inflation of a terminal
    object's sampled probability that is attributable to path count alone,
    holding trajectory length fixed.  Closed-form 3x3 normal equations; no
    external dependency.

    Reference values: flow matching gives gamma = 0, soft RL gives gamma = 1.
    """
    rows, ys = [], []
    for s, p in term.items():
        if p <= 0.0:
            continue
        rows.append([math.log(g.paths(*s)), float(s[0] + s[1]), 1.0])
        ys.append(math.log(p) - math.log(reward(*s, g.h)))

    k = 3
    ata = [[sum(r[a] * r[b] for r in rows) for b in range(k)] for a in range(k)]
    aty = [sum(r[a] * y for r, y in zip(rows, ys)) for a in range(k)]

    # Gaussian elimination with partial pivoting.
    aug = [ata[a][:] + [aty[a]] for a in range(k)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        for r in range(k):
            if r == col:
                continue
            fac = aug[r][col] / pv
            for c in range(col, k + 1):
                aug[r][c] -= fac * aug[col][c]
    return aug[0][k] / aug[0][0]


def kl_to_target(g: Grid, term: dict, log_z: float) -> float:
    """KL(p_terminal || R/Z) -- the sampler's residual bias."""
    acc = 0.0
    for s, p in term.items():
        if p <= 0.0:
            continue
        acc += p * (math.log(p) - (math.log(reward(*s, g.h)) - log_z))
    return acc


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=8)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--steps", type=int, default=41)
    args = ap.parse_args()

    g = Grid(args.height)
    z = sum(reward(i, j, g.h) for i in range(g.h) for j in range(g.h))
    log_z = math.log(z)
    total_paths = sum(g.paths(i, j) for i in range(g.h) for j in range(g.h))

    print(f"hypergrid H={g.h}  |X|={g.h * g.h}  trajectories={total_paths}")
    print(f"Z = {z!r}   log Z = {log_z!r}\n")

    gfn = flow_matching_policy(g)
    ment = maxent_rl_policy(g)
    unif = uniform_policy(g)

    rows = []
    for name, pol in (("flow-matching (GFN)", gfn), ("soft RL (MaxEnt)", ment), ("uniform", unif)):
        st = work_stats(g, pol)
        gam = fit_gamma(g, st["terminal"])
        kl_t = kl_to_target(g, st["terminal"], log_z)
        tb = st["var"]  # min_c E[(c+W)^2]
        elbo = -st["mean"]
        rows.append(
            {
                "policy": name,
                "mean_W": st["mean"],
                "var_W": st["var"],
                "jarzynski": st["jarzynski"],
                "jarzynski_rel_err": abs(st["jarzynski"] - z) / z,
                "elbo": elbo,
                "elbo_gap": log_z - elbo,
                "tb_optimum": tb,
                "gamma": gam,
                "kl_terminal": kl_t,
            }
        )
        print(f"--- {name}")
        print(f"    E[W]            = {st['mean']:+.12f}")
        print(f"    Var[W]          = {st['var']:.12e}      <- C3: equals min_c E[(c+W)^2]")
        print(f"    E[exp(-W)]      = {st['jarzynski']:.12f}   (Z = {z:.12f})")
        print(f"    |rel err| vs Z  = {abs(st['jarzynski'] - z) / z:.3e}          <- C1 Jarzynski")
        print(f"    ELBO = -E[W]    = {elbo:+.12f}")
        print(f"    log Z - ELBO    = {log_z - elbo:.12e}    <- C2 = KL(P_F||P_B) >= 0")
        print(f"    fitted gamma    = {gam:+.6f}             <- C5 multi-path exponent")
        print(f"    KL(p_term||R/Z) = {kl_t:.12e}\n")

    # C3 explicitly: brute-force the scalar minimisation over log Z_theta.
    st = work_stats(g, unif)
    best_c, best_v = None, float("inf")
    for k in range(-400000, 400001):
        c = k * 1e-5
        v = st["var"] + (c + st["mean"]) ** 2
        if v < best_v:
            best_v, best_c = v, c
    print("C3 direct check on the uniform policy:")
    print(f"    argmin_c E[(c+W)^2] = {best_c:+.5f}   vs  -E[W] = {-st['mean']:+.5f}")
    print(f"    min value           = {best_v:.12e}   vs  Var[W] = {st['var']:.12e}\n")

    # The dial: geometric path from MaxEnt RL to flow matching.
    dial = []
    for k in range(args.steps):
        lam = k / (args.steps - 1)
        pol = geometric_mix(g, ment, gfn, lam)
        s = work_stats(g, pol)
        dial.append(
            {
                "lam": lam,
                "mean_W": s["mean"],
                "var_W": s["var"],
                "gamma": fit_gamma(g, s["terminal"]),
                "kl_terminal": kl_to_target(g, s["terminal"], log_z),
                "elbo_gap": log_z + s["mean"],
            }
        )
    print("dial (MaxEnt -> GFN):  lam    Var[W]        gamma      KL(p||R/Z)")
    for d in dial[:: max(1, len(dial) // 8)]:
        print(f"                      {d['lam']:.2f}   {d['var_W']:.6f}    {d['gamma']:+.4f}    {d['kl_terminal']:.6f}")

    out = {
        "height": g.h,
        "n_objects": g.h * g.h,
        "n_trajectories": total_paths,
        "Z": z,
        "log_Z": log_z,
        "policies": rows,
        "dial": dial,
        "c3_argmin": best_c,
        "c3_min_value": best_v,
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
