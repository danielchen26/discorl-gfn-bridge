#!/usr/bin/env python3
"""Verify the GFlowNet-to-RL reduction exactly, before building anything on it.

Tiapkin, Morozov, Naumov and Vetrov (AISTATS 2024, arXiv:2310.12934), Theorem 1:
for a DAG G with terminal states X, a fixed backward policy P_B and a GFlowNet
reward R, build the MDP with gamma = 1 and rewards

    r(s, s') = log P_B(s|s')   for s not terminal
    r(s, s') = log R(s)        for s terminal, going to the absorbing state
    r        = 0               at the absorbing state

Then the optimal policy of the ENTROPY-REGULARIZED problem AT COEFFICIENT
lambda = 1 equals the GFlowNet forward policy, and the optimal soft value equals
log F. Proposition 1 adds

    V_1^pi(s_0) = log Z - KL(q^pi || P_B),   and
    V_1^*(s_0) - V_1^pi(s_0) >= KL(d^pi || R/Z),

so maximising that value bounds the error of the terminal distribution. This is
what makes "meta-learn a GFlowNet" well posed: the meta-objective can stay a
return, because in this MDP the return IS distributional accuracy.

Everything above is quoted from someone else's paper, so it gets checked here
before anything is built on it. Soft value iteration on the constructed MDP is
exact on a hypergrid, so the check is to machine precision and involves no
training and no sampling:

  R1  V*(s) = log F(s) for every state
  R2  V*(s_0) = log Z
  R3  the induced policy's terminal distribution is exactly R/Z
  R4  the gauge does not matter: R1-R3 hold for arbitrary backward policies

And the warning that comes with it. Remark 3 of the same paper: any lambda != 1
biases the policy, because rescaling the terminal reward to R^{1/lambda} also
rescales the intermediate log P_B terms. Remark 4 reports exactly this failure
for SAC with an adaptive coefficient. A discovered update rule has no explicit
entropy coefficient at all, so it falls in that class. R5 measures the size of
the resulting bias as a function of lambda, which is the thing any attempt to
drop DiscoRL into this MDP has to defeat.

Usage:  python3 research/tiapkin.py [--height 8] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from cumulants import Grid, reward


def build(h: int):
    n = h * h
    children, parents = [[] for _ in range(n)], [[] for _ in range(n)]
    for i in range(h):
        for j in range(h):
            s = i * h + j
            for di, dj in ((1, 0), (0, 1)):
                if i + di < h and j + dj < h:
                    c = (i + di) * h + (j + dj)
                    children[s].append(c)
                    parents[c].append(s)
    rew = np.array([reward(k // h, k % h, h) for k in range(n)])
    order = sorted(range(n), key=lambda k: (k // h) + (k % h))
    return n, children, parents, rew, order


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


def soft_value_iteration(n, children, rew, pb, order, lam: float = 1.0):
    """Exact soft value iteration on the constructed MDP.

    At state s the actions are: stop, which pays log R(s) and absorbs, or move
    to a child c, which pays log p_B(s|c). With gamma = 1 and coefficient lam,

        V(s) = lam * log sum_a exp( (r(s,a) + V(next)) / lam ).

    At lam = 1 this is exactly the flow recursion log F(s) = log(R(s) + sum_c
    p_B(s|c) F(c)), which is the content of Theorem 1.
    """
    V = np.zeros(n)
    pol = [None] * n
    for s in reversed(order):
        qs = [math.log(rew[s])]  # stop
        acts = [None]
        for c in children[s]:
            w = pb[c].get(s, 0.0)
            if w <= 0:
                continue
            qs.append(math.log(w) + V[c])
            acts.append(c)
        q = np.array(qs) / lam
        m = q.max()
        V[s] = lam * (m + math.log(np.exp(q - m).sum()))
        p = np.exp(q - m)
        pol[s] = (acts, p / p.sum())
    return V, pol


def terminal_distribution(n, pol, order):
    reach = np.zeros(n)
    reach[0] = 1.0
    term = np.zeros(n)
    for s in order:
        acts, probs = pol[s]
        for a, p in zip(acts, probs):
            if a is None:
                term[s] += reach[s] * p
            else:
                reach[a] += reach[s] * p
    return term


def exact_log_flow(n, children, rew, pb, order):
    F = np.zeros(n)
    for s in reversed(order):
        F[s] = rew[s] + sum(pb[c].get(s, 0.0) * F[c] for c in children[s])
    return np.log(F)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=8)
    ap.add_argument("--gauges", type=int, default=50)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    h = args.height
    n, children, parents, rew, order = build(h)
    Z = float(rew.sum())
    target = rew / Z
    print(f"hypergrid H={h}   |S| = {n}   log Z = {math.log(Z)!r}\n")

    print("R1-R3  the reduction, at the uniform gauge")
    pb = uniform_pb(parents)
    V, pol = soft_value_iteration(n, children, rew, pb, order)
    logF = exact_log_flow(n, children, rew, pb, order)
    term = terminal_distribution(n, pol, order)
    e1 = float(np.max(np.abs(V - logF)))
    e2 = abs(V[0] - math.log(Z))
    e3 = float(np.max(np.abs(term - target)))
    print(f"  R1  max |V*(s) - log F(s)|        = {e1:.3e}")
    print(f"  R2  |V*(s0) - log Z|              = {e2:.3e}")
    print(f"  R3  max |p(x) - R(x)/Z|           = {e3:.3e}")

    print(f"\nR4  the same, over {args.gauges} random backward policies")
    rng = np.random.default_rng(0)
    w1 = w2 = w3 = 0.0
    for _ in range(args.gauges):
        g = random_pb(parents, rng, float(10 ** rng.uniform(-1, 1)))
        Vg, polg = soft_value_iteration(n, children, rew, g, order)
        lf = exact_log_flow(n, children, rew, g, order)
        tg = terminal_distribution(n, polg, order)
        w1 = max(w1, float(np.max(np.abs(Vg - lf))))
        w2 = max(w2, abs(Vg[0] - math.log(Z)))
        w3 = max(w3, float(np.max(np.abs(tg - target))))
    print(f"  worst |V* - log F|                = {w1:.3e}")
    print(f"  worst |V*(s0) - log Z|            = {w2:.3e}")
    print(f"  worst |p(x) - R(x)/Z|             = {w3:.3e}")

    ok = max(e1, e2, e3, w1, w2, w3) < 1e-10
    print(f"\n  reduction verified: {ok}")

    print("\nR5  Remark 3: the bias when the entropy coefficient is not 1")
    print(f"  {'lambda':>8}{'max |p - R/Z|':>16}{'KL(p || R/Z)':>16}")
    lam_rows = []
    for lam in (0.25, 0.5, 0.8, 0.9, 1.0, 1.1, 1.25, 2.0, 4.0):
        Vl, poll = soft_value_iteration(n, children, rew, pb, order, lam)
        tl = terminal_distribution(n, poll, order)
        mask = tl > 0
        kl = float(np.sum(tl[mask] * (np.log(tl[mask]) - np.log(target[mask]))))
        mx = float(np.max(np.abs(tl - target)))
        lam_rows.append({"lam": lam, "max_dev": mx, "kl": kl})
        print(f"  {lam:>8.2f}{mx:>16.3e}{kl:>16.3e}")
    print("\n  lambda = 1 is not a convention. Any other value biases the sampler,")
    print("  and a discovered rule has no explicit coefficient to pin at 1 --")
    print("  which is exactly the failure Remark 4 reports for adaptive-lambda SAC.")

    out = {
        "height": h, "log_Z": math.log(Z),
        "uniform": {"V_vs_logF": e1, "V0_vs_logZ": e2, "terminal": e3},
        "gauges": {"n": args.gauges, "V_vs_logF": w1, "V0_vs_logZ": w2, "terminal": w3},
        "reduction_verified": bool(ok),
        "lambda_sweep": lam_rows,
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
