#!/usr/bin/env python3
"""The backward policy of a GFlowNet is a gauge freedom of dimension |E|-|V|+1.

ATTRIBUTION FIRST.  This is NOT a new theorem.  Brunswic, Li, Xu, Jui and Ma,
"A Theory of Non-Acyclic Generative Flow Networks" (arXiv:2312.15246, AAAI 2024),
Proposition 4 and Theorem 5, already prove that the set of R-flows is a non-empty
affine subspace F_R = F + H^1(G) directed by H^1(G), the span of cycle indicators
-- the cycle space.  They attribute a stronger form to Kalpazidou (2007),
Theorem 3.3.1.  That every backward policy induces the same terminal
distribution is likewise standard, stated in at least six papers and most
crisply in Malkin, Jain, Bengio, Sun and Bengio, "Trajectory balance"
(arXiv:2201.13259, NeurIPS 2022), Section 3.1, which already names cycles in the
underlying undirected graph as the source of the multiplicity.

What is left here is bookkeeping on top of that: the explicit dimension
|E| - |V| + 1, its identification with the parameter count of p_B, and a
numerical check across a family of graphs.  The dimension is the first Betti
number, so it follows immediately from the published result rather than adding
to it.  The file is kept because its numbers are used elsewhere and because G4 --
tracking cond(I - P_B^T) as a function of the gauge -- is the one part the
literature search did not turn up.

STATEMENT.  Let G = (V, E) be a connected DAG with a unique source s0, every other
state reachable from it, and a strictly positive terminal reward R.  Consider
positive edge flows f : E -> R_{>0} satisfying, for every s != s0,

    sum_{e in in(s)} f(e)  =  R(s) + sum_{e in out(s)} f(e).                (1)

Then the solution set is a relatively open convex set of dimension |E|-|V|+1,
the cycle rank of G, and it is in bijection with the backward policies.

PROOF (elementary, for the dimension count only).  (1) is |V|-1 affine equations in |E| unknowns whose coefficient matrix is
the incidence matrix B of G with the row of s0 deleted.  For a connected graph
rank(B) = |V|-1, and the rows of B sum to zero, which is the only dependency, so
deleting one row leaves rank |V|-1.  The solution set is therefore affine of
dimension |E| - (|V|-1) = |E| - |V| + 1.  The uniform-over-parents policy induces
a strictly positive solution, so the intersection with the open positive orthant
is nonempty and relatively open, hence of the same dimension.

Bijection: given a valid f put F(s) = sum_{in(s)} f for s != s0 and F(s0) = Z;
then p_B(s|c) = f(s->c)/F(c) is a distribution over the parents of c.  Given p_B,
the recursion F(s) = R(s) + sum_{c in Ch(s)} p_B(s|c) F(c) has a unique solution
by topological order, and f(s->c) := p_B(s|c) F(c) satisfies (1).  The two maps
are mutually inverse.  Counting the free parameters of p_B,

    sum_{s != s0} (|Pa(s)| - 1)  =  |E| - (|V| - 1)  =  |E| - |V| + 1,

since every non-source state has at least one parent and sum_s |Pa(s)| = |E|.  QED

COROLLARY (gauge invariance).  reach(s) = F(s)/Z by induction along the DAG, so

    p(x) = reach(x) * p_F(stop|x) = (F(x)/Z) * (R(x)/F(x)) = R(x)/Z

for EVERY valid flow.  The terminal distribution does not see the gauge.  QED

What the corollary does not say is that the gauge is free in practice.  The flow
recursion is linear, F = R + P_B^T F, so the difficulty of solving or learning it
is governed by the conditioning of (I - P_B^T), which the gauge does move.  That
is the whole opening: correctness is structural, speed is not, so speed is what
is left to optimise.

This script tests all of it -- on the hypergrid, and on random DAGs so that the
dimension count is not an artefact of one graph.

Usage:  python3 research/gauge.py [--height 8] [--samples 200] [--json out.json]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from cumulants import reward


def hypergrid_graph(h: int):
    n = h * h
    edges, parents = [], [[] for _ in range(n)]
    for i in range(h):
        for j in range(h):
            s = i * h + j
            if i + 1 < h:
                edges.append((s, (i + 1) * h + j))
                parents[(i + 1) * h + j].append(s)
            if j + 1 < h:
                edges.append((s, i * h + (j + 1)))
                parents[i * h + (j + 1)].append(s)
    rew = np.array([reward(k // h, k % h, h) for k in range(n)])
    return n, edges, parents, rew


def random_dag(n: int, max_parents: int, rng: np.random.Generator):
    """Random connected DAG in topological order; node 0 is the unique source."""
    edges, parents = [], [[] for _ in range(n)]
    for j in range(1, n):
        k = int(rng.integers(1, min(max_parents, j) + 1))
        ps = rng.choice(j, size=k, replace=False)
        for p in ps:
            edges.append((int(p), j))
            parents[j].append(int(p))
    rew = np.exp(rng.normal(size=n))  # strictly positive
    return n, edges, parents, rew


def cycle_rank_from_incidence(n: int, edges) -> int:
    B = np.zeros((n, len(edges)))
    for k, (s, t) in enumerate(edges):
        B[s, k] = -1.0
        B[t, k] = +1.0
    return len(edges) - np.linalg.matrix_rank(B)


def topo_order(n: int, parents) -> list[int]:
    """Nodes sorted so that every parent precedes its child."""
    depth = [0] * n
    for j in range(n):
        for p in parents[j]:
            depth[j] = max(depth[j], depth[p] + 1)
    return sorted(range(n), key=lambda k: depth[k])


def flow_and_terminal(n, parents, children, rew, pb):
    """Exact flow for a backward policy, and the terminal distribution."""
    order = topo_order(n, parents)
    F = np.zeros(n)
    for k in reversed(order):
        F[k] = rew[k] + sum(pb[c].get(k, 0.0) * F[c] for c in children[k])
    reach = np.zeros(n)
    reach[order[0]] = 1.0
    term = np.zeros(n)
    for k in order:
        term[k] = reach[k] * rew[k] / F[k]
        for c in children[k]:
            reach[c] += reach[k] * pb[c].get(k, 0.0) * F[c] / F[k]
    return F, term


def uniform_pb(parents):
    return [{p: 1.0 / len(ps) for p in ps} for ps in parents]


def random_pb(parents, rng, alpha):
    out = []
    for ps in parents:
        if not ps:
            out.append({})
            continue
        w = rng.dirichlet([alpha] * len(ps))
        out.append({p: float(w[k]) for k, p in enumerate(ps)})
    return out


def cond_flow_operator(n, pb) -> float:
    """cond(I - P_B^T): the linear system the flow must satisfy."""
    M = np.zeros((n, n))
    for c in range(n):
        for p, w in pb[c].items():
            M[p, c] = w
    return float(np.linalg.cond(np.eye(n) - M))


def audit(name, n, edges, parents, rew, samples, rng) -> dict:
    children = [[] for _ in range(n)]
    for s, t in edges:
        children[s].append(t)
    Z = float(rew.sum())
    target = rew / Z

    dof = sum(max(0, len(p) - 1) for p in parents)
    cr_formula = len(edges) - n + 1
    cr_rank = int(cycle_rank_from_incidence(n, edges))
    dim_ok = dof == cr_formula == cr_rank

    worst, conds, spans = 0.0, [], []
    for _ in range(samples):
        pb = random_pb(parents, rng, float(10 ** rng.uniform(-1, 1)))
        F, term = flow_and_terminal(n, parents, children, rew, pb)
        worst = max(worst, float(np.max(np.abs(term - target))))
        conds.append(cond_flow_operator(n, pb))
        spans.append(float(np.log(F).max() - np.log(F).min()))

    F_u, term_u = flow_and_terminal(n, parents, children, rew, uniform_pb(parents))
    res = {
        "graph": name,
        "n_vertices": n,
        "n_edges": len(edges),
        "pb_dof": dof,
        "cycle_rank_formula": cr_formula,
        "cycle_rank_incidence": cr_rank,
        "dimension_match": bool(dim_ok),
        "max_terminal_deviation": worst,
        "terminal_invariant": bool(worst < 1e-11),
        "cond_min": float(np.min(conds)),
        "cond_max": float(np.max(conds)),
        "cond_uniform": cond_flow_operator(n, uniform_pb(parents)),
        "logF_span_min": float(np.min(spans)),
        "logF_span_max": float(np.max(spans)),
        "logF_span_uniform": float(np.log(F_u).max() - np.log(F_u).min()),
        "uniform_terminal_deviation": float(np.max(np.abs(term_u - target))),
    }
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=8)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--random-dags", type=int, default=12)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    rows = []

    for h in (4, 6, args.height, 12):
        n, e, p, r = hypergrid_graph(h)
        rows.append(audit(f"hypergrid H={h}", n, e, p, r, args.samples, rng))

    for t in range(args.random_dags):
        n = int(rng.integers(12, 60))
        mp = int(rng.integers(2, 6))
        nn, e, p, r = random_dag(n, mp, rng)
        rows.append(audit(f"random DAG n={n} maxPa={mp}", nn, e, p, r, 40, rng))

    hdr = (f"{'graph':<26}{'|V|':>5}{'|E|':>6}{'dof':>6}{'|E|-|V|+1':>11}"
           f"{'rank':>6}{'dim':>5}{'max|p-R/Z|':>13}{'cond spread':>13}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['graph']:<26}{r['n_vertices']:>5}{r['n_edges']:>6}{r['pb_dof']:>6}"
            f"{r['cycle_rank_formula']:>11}{r['cycle_rank_incidence']:>6}"
            f"{'ok' if r['dimension_match'] else 'BAD':>5}"
            f"{r['max_terminal_deviation']:>13.2e}"
            f"{r['cond_max']/r['cond_min']:>12.2f}x"
        )

    all_dim = all(r["dimension_match"] for r in rows)
    all_inv = all(r["terminal_invariant"] for r in rows)
    print(f"\ndimension count holds on all {len(rows)} graphs : {all_dim}")
    print(f"terminal distribution invariant everywhere  : {all_inv} "
          f"(worst {max(r['max_terminal_deviation'] for r in rows):.2e})")
    print(f"conditioning spread, max over graphs        : "
          f"{max(r['cond_max']/r['cond_min'] for r in rows):.2f}x")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"graphs": rows, "dimension_all": all_dim, "invariant_all": all_inv}, fh, indent=2)
        print(f"wrote {args.json}")
    return 0 if (all_dim and all_inv) else 1


if __name__ == "__main__":
    raise SystemExit(main())
