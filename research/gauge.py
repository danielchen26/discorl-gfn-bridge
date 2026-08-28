#!/usr/bin/env python3
"""The backward policy is a gauge choice, and its dimension is the cycle rank.

The dossier's Onsager argument says a relaxation dynamics splits into a
potential, which fixes where you end up, and an operator, which fixes how you
get there, and that the antisymmetric part of the operator contributes exactly
zero to dPhi/dt.  That is what licenses "pin the fixed point by hand, meta-learn
the operator".  On a GFlowNet the claim becomes finite-dimensional and checkable.

Hodge on the state graph splits an edge flow into a gradient part and a cycle
part, F = F_grad + F_cyc with boundary(F_cyc) = 0.  The terminal marginal depends
only on the boundary, so the cycle component is INVISIBLE to the sampling
distribution.  Keeping the balance condition while changing F forces a matching
change in p_B, which suggests

    dim(backward-policy freedom)  ==  dim(cycle space)  ==  |E| - |V| + 1

and if that holds, the well-known dependence of GFlowNets on the choice of p_B is
exactly a gauge freedom: invisible to the answer, visible to the difficulty.

Four things are checked, all exactly, no sampling:

  G1  cycle rank of the state graph equals the free parameters in p_B
  G2  cycle rank equals the rank deficiency measured from the incidence matrix
  G3  every p_B gives the SAME terminal distribution, p(x) = R(x)/Z
  G4  but the conditioning of the flow system (I - P_B^T) F = R varies a lot,
      and so does the dynamic range of log F -- the gauge is free in the answer
      and expensive in the numerics

Usage:  python3 research/gauge.py [--height 8] [--samples 200] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from cumulants import Grid, reward


def graph(h: int):
    """Vertices, directed edges, and the parent lists of the hypergrid DAG."""
    n = h * h
    edges = []
    parents: list[list[int]] = [[] for _ in range(n)]
    for i in range(h):
        for j in range(h):
            s = i * h + j
            if i + 1 < h:
                t = (i + 1) * h + j
                edges.append((s, t))
                parents[t].append(s)
            if j + 1 < h:
                t = i * h + (j + 1)
                edges.append((s, t))
                parents[t].append(s)
    return n, edges, parents


def cycle_rank(n: int, edges: list[tuple[int, int]]) -> int:
    """|E| - |V| + 1 for a connected graph, computed from the incidence matrix.

    Taken from the rank rather than the formula so that the formula itself is
    what gets tested.
    """
    B = np.zeros((n, len(edges)))
    for k, (s, t) in enumerate(edges):
        B[s, k] = -1.0
        B[t, k] = +1.0
    return len(edges) - np.linalg.matrix_rank(B)


def pb_dof(parents: list[list[int]]) -> int:
    """Free parameters in a backward policy: a simplex over parents per state."""
    return sum(max(0, len(p) - 1) for p in parents)


def random_pb(parents: list[list[int]], rng: np.random.Generator, alpha: float):
    """A backward policy drawn from a Dirichlet over each state's parents."""
    pb: list[dict[int, float]] = []
    for ps in parents:
        if not ps:
            pb.append({})
            continue
        w = rng.dirichlet([alpha] * len(ps))
        pb.append({p: float(w[k]) for k, p in enumerate(ps)})
    return pb


def flow_and_terminal(g: Grid, parents, pb):
    """Exact flow for a given backward policy, and the terminal distribution.

    The flow condition on a DAG is stated on EDGES: F(s->c) = p_B(s|c) F(c), so
    F(s) = R(s) + sum_c p_B(s|c) F(c).  Solving that and reading off
    p_F(stop|s) = R(s)/F(s) gives the sampler.
    """
    h = g.h
    n = h * h
    F = np.zeros(n)
    for t in range(n):
        k = int(g.revTopo[t]) if hasattr(g, "revTopo") else t
    # reverse topological order by i+j descending
    order = sorted(range(n), key=lambda k: -((k // h) + (k % h)))
    for k in order:
        i, j = k // h, k % h
        acc = reward(i, j, h)
        for a, c in (("r", (i + 1) * h + j if i + 1 < h else -1),
                     ("u", i * h + (j + 1) if j + 1 < h else -1)):
            del a
            if c >= 0:
                acc += pb[c].get(k, 0.0) * F[c]
        F[k] = acc

    reach = np.zeros(n)
    reach[0] = 1.0
    term = np.zeros(n)
    for k in sorted(range(n), key=lambda k: (k // h) + (k % h)):
        i, j = k // h, k % h
        term[k] = reach[k] * reward(i, j, h) / F[k]
        for c in ((i + 1) * h + j if i + 1 < h else -1,
                  i * h + (j + 1) if j + 1 < h else -1):
            if c >= 0:
                reach[c] += reach[k] * (pb[c].get(k, 0.0) * F[c]) / F[k]
    return F, term


def flow_condition_number(g: Grid, parents, pb) -> float:
    """cond(I - P_B^T), the linear system the flow must satisfy.

    The flow recursion is LINEAR -- that is the Doob h-transform showing through
    -- so the difficulty of solving or learning it is governed by the condition
    number of exactly this matrix.
    """
    h = g.h
    n = h * h
    M = np.zeros((n, n))
    for c in range(n):
        for p, w in pb[c].items():
            M[p, c] = w
    return float(np.linalg.cond(np.eye(n) - M))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=8)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    h = args.height
    g = Grid(h)
    n, edges, parents = graph(h)
    Z = sum(reward(i, j, h) for i in range(h) for j in range(h))
    target = np.array([reward(k // h, k % h, h) / Z for k in range(n)])

    cr_formula = len(edges) - n + 1
    cr_rank = cycle_rank(n, edges)
    dof = pb_dof(parents)

    print(f"hypergrid H={h}:  |V| = {n}   |E| = {len(edges)}")
    print(f"G1  free parameters in p_B          : {dof}")
    print(f"G2  cycle rank |E|-|V|+1 (formula)  : {cr_formula}")
    print(f"    cycle rank from incidence rank  : {cr_rank}")
    ok_dim = dof == cr_formula == cr_rank
    print(f"    -> {'MATCH: the backward policy IS the cycle space' if ok_dim else 'MISMATCH'}\n")

    rng = np.random.default_rng(0)
    worst = 0.0
    conds, spans = [], []
    for s in range(args.samples):
        alpha = float(10 ** rng.uniform(-1, 1))
        pb = random_pb(parents, rng, alpha)
        F, term = flow_and_terminal(g, parents, pb)
        worst = max(worst, float(np.max(np.abs(term - target))))
        conds.append(flow_condition_number(g, parents, pb))
        lf = np.log(F)
        spans.append(float(lf.max() - lf.min()))

    uni = [{p: 1.0 / len(ps) for p in ps} for ps in parents]
    F_u, term_u = flow_and_terminal(g, parents, uni)
    err_u = float(np.max(np.abs(term_u - target)))
    cond_u = flow_condition_number(g, parents, uni)
    span_u = float(np.log(F_u).max() - np.log(F_u).min())

    print(f"G3  terminal distribution over {args.samples} random backward policies")
    print(f"    max |p(x) - R(x)/Z|             : {worst:.3e}")
    print(f"    uniform-parents backward policy : {err_u:.3e}")
    print(f"    -> {'INVARIANT: the gauge is invisible to the answer' if worst < 1e-12 else 'NOT INVARIANT'}\n")

    conds = np.array(conds)
    spans = np.array(spans)
    print("G4  but the numerics are not invariant")
    print(f"    cond(I - P_B^T)   min {conds.min():9.2f}   max {conds.max():9.2f}   "
          f"ratio {conds.max()/conds.min():6.2f}x   (uniform {cond_u:.2f})")
    print(f"    span of log F     min {spans.min():9.4f}   max {spans.max():9.4f}   "
          f"ratio {spans.max()/spans.min():6.2f}x   (uniform {span_u:.4f})")
    print("    -> the backward policy is free in the answer and expensive in the difficulty")

    out = {
        "height": h,
        "n_vertices": n,
        "n_edges": len(edges),
        "pb_dof": dof,
        "cycle_rank_formula": cr_formula,
        "cycle_rank_from_incidence": int(cr_rank),
        "dimension_match": bool(ok_dim),
        "samples": args.samples,
        "max_terminal_deviation": worst,
        "terminal_invariant": bool(worst < 1e-12),
        "cond_min": float(conds.min()),
        "cond_max": float(conds.max()),
        "cond_uniform": cond_u,
        "logF_span_min": float(spans.min()),
        "logF_span_max": float(spans.max()),
        "logF_span_uniform": span_u,
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0 if (ok_dim and worst < 1e-12) else 1


if __name__ == "__main__":
    raise SystemExit(main())
