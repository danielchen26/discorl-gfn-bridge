#!/usr/bin/env python3
"""Meta-learn a GFlowNet update rule, and show which meta-objective matters.

research/temperature.py established two things about pointing a discovered RL
rule at the MDP whose soft-optimal policy is a GFlowNet. The rule collapses
there unless the geometric potential is shaped in, and even once it optimises,
the sampler it produces is not a member of the soft family at any coefficient:
best fit 0.36 +/- 0.14 with a residual of 0.65 against a threshold of 0.4.

The diagnosis was that a return-maximising meta-objective has no reason to land
on the one coefficient that samples proportionally to reward. Tiapkin,
Morozov, Naumov and Vetrov (AISTATS 2024, arXiv:2310.12934) Proposition 1 says
what to use instead, and it is not expensive:

    V_1^pi(s0) = log Z - KL(q^pi || P_B),      sup_pi V_1^pi(s0) = log Z

so the entropy-regularised return in that MDP IS distribution accuracy. It is a
scalar computed from trajectories, needs no partition function and no mode
counting, and is exactly as cheap as the return.

This script meta-learns an update rule twice, changing only that scalar, and
measures what each learned rule converges to.

WHAT IS AND IS NOT GIVEN AWAY.

The rule is local. For each transition it sees the immediate reward, its own
auxiliary prediction at both ends, the current logit and log-probability, and
whether the action terminates. It never sees R/Z, log Z, the visit
distribution, or any global quantity. Learning to sample proportionally to
reward is not free from that information: the target requires accounting for
how many paths reach each terminal, which no local reward reading supplies.

Everything downstream of the policy is exact rather than sampled. The grid is a
layered DAG, so the visit distribution is e_0^T (I - T)^-1 in closed form and
differentiable, which removes sampling noise from the meta-gradient entirely.
This is the infinite-sample limit of the same rule, and is stated as such.

Generalisation is tested, not assumed. Rules are meta-trained on one set of
random reward landscapes and read out on a disjoint set, so a rule that merely
memorised a target scores nothing.

Usage:
  .venv/bin/python research/metagfn.py --json research/metagfn.json
"""

from __future__ import annotations

import argparse
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax

# The claims here are exact identities on a 36 state graph, where double
# precision costs nothing and single precision leaves 1e-6 residues that cannot
# be told apart from real error.
jax.config.update("jax_enable_x64", True)

RIGHT, UP, STOP = 0, 1, 2
NEG = -1e30


# --------------------------------------------------------------------------
# The DAG, and everything about a policy on it, in closed form
# --------------------------------------------------------------------------


def grid_tables(h: int):
    """Child indices, action validity and backward-policy payments."""
    s_of = lambda i, j: i * h + j
    n = h * h
    rc = np.full(n, 0, dtype=np.int32)
    uc = np.full(n, 0, dtype=np.int32)
    ok_r = np.zeros(n, dtype=bool)
    ok_u = np.zeros(n, dtype=bool)
    for i in range(h):
        for j in range(h):
            s = s_of(i, j)
            if i + 1 < h:
                rc[s], ok_r[s] = s_of(i + 1, j), True
            if j + 1 < h:
                uc[s], ok_u[s] = s_of(i, j + 1), True
    n_par = np.array([float((k // h > 0) + (k % h > 0)) for k in range(n)])
    log_pb = -np.log(np.maximum(n_par, 1.0))  # zero at the source
    return rc, uc, ok_r, ok_u, log_pb


def geometric_potential(h: int, rc, uc, ok_r, ok_u, log_pb) -> np.ndarray:
    """Soft value at coefficient one of this DAG with the reward flattened.

    Used as a shaping potential it cancels the term that makes travel look bad
    while leaving the soft-optimal policy unchanged at every coefficient, which
    research/temperature.py verified pointwise to 2e-15. It never reads R.
    """
    n = h * h
    V = np.zeros(n)
    for s in sorted(range(n), key=lambda k: (k // h) + (k % h), reverse=True):
        qs = [0.0]
        if ok_r[s]:
            qs.append(log_pb[rc[s]] + V[rc[s]])
        if ok_u[s]:
            qs.append(log_pb[uc[s]] + V[uc[s]])
        m = max(qs)
        V[s] = m + math.log(sum(math.exp(q - m) for q in qs))
    return V


class Grid:
    """Everything that does not depend on the reward landscape."""

    def __init__(self, h: int, shaped: bool = True):
        self.h, self.n = h, h * h
        rc, uc, ok_r, ok_u, log_pb = grid_tables(h)
        self.rc, self.uc = jnp.asarray(rc), jnp.asarray(uc)
        self.ok_r, self.ok_u = jnp.asarray(ok_r), jnp.asarray(ok_u)
        self.np_rc, self.np_uc, self.np_ok_r, self.np_ok_u = rc, uc, ok_r, ok_u
        phi = geometric_potential(h, rc, uc, ok_r, ok_u, log_pb) if shaped \
            else np.zeros(self.n)
        self.phi = jnp.asarray(phi)
        # Payment for arriving at a child, shaping folded in.
        self.r_r = jnp.asarray(log_pb[rc] + phi[rc]) - self.phi
        self.r_u = jnp.asarray(log_pb[uc] + phi[uc]) - self.phi
        self.valid = jnp.stack(
            [self.ok_r, self.ok_u, jnp.ones(self.n, dtype=bool)], axis=-1
        )
        # Rows of the reachability system, assembled once.
        idx = np.arange(self.n)
        self.rows_r = (idx[ok_r], rc[ok_r])
        self.rows_u = (idx[ok_u], uc[ok_u])

    def stop_pay(self, log_r):
        """What terminating pays, in the same gauge as the moves.

        Shaping subtracts Phi from the terminal payment because the absorbing
        state has no potential. Shaping the moves and not this leaves a gauge
        that is inconsistent by an amount depending on the landscape, which is
        what the identity check caught.
        """
        return log_r - self.phi

    def policy(self, theta):
        return jax.nn.softmax(jnp.where(self.valid, theta, NEG), axis=-1)

    def occupancy(self, pi):
        """Expected visits per state, exactly.

        The grid is layered by i + j so the move matrix is nilpotent and the
        Neumann series terminates; solving is equivalent and differentiable.
        """
        T = jnp.zeros((self.n, self.n))
        T = T.at[self.rows_r].add(pi[self.rows_r[0], RIGHT])
        T = T.at[self.rows_u].add(pi[self.rows_u[0], UP])
        e0 = jnp.zeros(self.n).at[0].set(1.0)
        return jax.scipy.linalg.solve(jnp.eye(self.n) - T.T, e0)

    def readout(self, theta, log_r):
        """Terminal distribution, return, trajectory entropy."""
        pi = self.policy(theta)
        d = self.occupancy(pi)
        term = d * pi[:, STOP]
        step_r = (pi[:, STOP] * self.stop_pay(log_r)
                  + pi[:, RIGHT] * self.r_r + pi[:, UP] * self.r_u)
        ret = jnp.sum(d * step_r)
        safe = jnp.where(self.valid, pi, 1.0)
        ent = jnp.sum(d * (-jnp.sum(jnp.where(self.valid, pi * jnp.log(safe), 0.0), -1)))
        return term, ret, ent, pi, d


def kl_to_target(term, log_r):
    tgt = jax.nn.softmax(log_r)
    m = term > 1e-12
    return jnp.sum(jnp.where(m, term * (jnp.log(jnp.where(m, term, 1.0)) - jnp.log(tgt)), 0.0))


# --------------------------------------------------------------------------
# Reward landscapes
# --------------------------------------------------------------------------


def landscape(key, h: int, n_modes: int = 3):
    """Random multimodal log-reward on the grid."""
    k1, k2, k3 = jax.random.split(key, 3)
    amp = jax.random.uniform(k1, (n_modes,), minval=3.0, maxval=6.5)
    cx = jax.random.uniform(k2, (n_modes,), minval=0.1, maxval=0.9)
    cy = jax.random.uniform(k3, (n_modes,), minval=0.1, maxval=0.9)
    i, j = jnp.meshgrid(jnp.arange(h), jnp.arange(h), indexing="ij")
    x, y = i / (h - 1), j / (h - 1)
    lr = -2.0 + jnp.sum(
        amp[:, None, None] * jnp.exp(
            -((x[None] - cx[:, None, None]) ** 2 + (y[None] - cy[:, None, None]) ** 2) / 0.05
        ), axis=0)
    return lr.reshape(-1)


# --------------------------------------------------------------------------
# The update rule
# --------------------------------------------------------------------------

FEATS = 6
HID = 32


def init_rule(key):
    ks = jax.random.split(key, 6)
    def lyr(k, a, b, scale):
        return (jax.random.normal(k, (a, b)) * scale, jnp.zeros(b))
    return {
        "w1": lyr(ks[0], FEATS, HID, 1 / math.sqrt(FEATS)),
        "w2": lyr(ks[1], HID, HID, 1 / math.sqrt(HID)),
        "w3": lyr(ks[2], HID, 2, 1e-2 / math.sqrt(HID)),
    }


def rule_out(eta, feats):
    z = jnp.tanh(feats @ eta["w1"][0] + eta["w1"][1])
    z = jnp.tanh(z @ eta["w2"][0] + eta["w2"][1])
    return z @ eta["w3"][0] + eta["w3"][1]


def visit_weight(d, n, on_policy_share=0.5):
    """Where an update gets applied.

    Half the on-policy occupancy and half uniform, which is what training with
    an exploratory behaviour policy looks like. Full support is not a detail:
    detailed balance weighted purely on-policy drives its own residual to 1e-4
    and still lands at KL 0.92, because states the policy stops visiting never
    get corrected. The learned rule and the baseline are given the same
    weighting so that the comparison is about the update, not exploration.
    """
    return on_policy_share * d + (1.0 - on_policy_share) * jnp.ones(n) / n


def inner_step(eta, grid: Grid, log_r, theta, y, lr_in):
    """One application of the learned rule.

    Per transition the rule reads the immediate reward, its own prediction at
    this state and at the successor, the current logit, the log-probability,
    and whether the action ends the episode. It reads nothing global.
    """
    pi = grid.policy(theta)
    d = grid.occupancy(pi)
    w = visit_weight(d, grid.n)
    logp = jnp.log(jnp.where(grid.valid, pi, 1.0))

    r = jnp.stack([grid.r_r, grid.r_u, grid.stop_pay(log_r)], axis=-1)
    y_next = jnp.stack([y[grid.rc], y[grid.uc], jnp.zeros_like(y)], axis=-1)
    y_next = jnp.where(grid.valid, y_next, 0.0)
    y_here = jnp.repeat(y[:, None], 3, axis=1)
    is_stop = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (grid.n, 1))

    feats = jnp.stack([r, y_here, y_next, theta, logp, is_stop], axis=-1)
    out = rule_out(eta, feats.reshape(-1, FEATS)).reshape(grid.n, 3, 2)
    out = jnp.where(grid.valid[..., None], out, 0.0)

    dtheta = w[:, None] * out[..., 0]
    dy = w * jnp.sum(pi * out[..., 1], axis=-1)
    return theta + lr_in * dtheta, y + lr_in * dy


def unroll(eta, grid, log_r, steps, lr_in):
    theta0 = jnp.zeros((grid.n, 3))
    y0 = jnp.zeros(grid.n)

    def body(carry, _):
        th, y = carry
        return inner_step(eta, grid, log_r, th, y, lr_in), None

    (th, y), _ = jax.lax.scan(body, (theta0, y0), None, length=steps)
    return th, y


# --------------------------------------------------------------------------
# Meta-objectives
# --------------------------------------------------------------------------


def meta_score(theta, grid, log_r, objective):
    term, ret, ent, _, _ = grid.readout(theta, log_r)
    if objective == "return":
        return ret
    if objective == "soft":
        # V_1(s0) = return + trajectory entropy, which Proposition 1 identifies
        # with log Z - KL(q || P_B) and whose supremum is log Z.
        return ret + ent
    raise ValueError(objective)


def meta_loss(eta, grid, log_rs, steps, lr_in, objective):
    def one(log_r):
        th, _ = unroll(eta, grid, log_r, steps, lr_in)
        return meta_score(th, grid, log_r, objective)
    return -jnp.mean(jax.vmap(one)(log_rs))


# --------------------------------------------------------------------------
# The hand-designed baseline, also in closed form
# --------------------------------------------------------------------------


def db_loss(params, grid, log_r, on_policy_share=0.5):
    """Expected squared detailed-balance residual.

    Per transition, matching the locality of the learned rule:
      log F(s) + log p_F(c|s) - log F(c) - log p_B(s|c),
    with log F pinned to log R at terminals through the stop action, all in the
    shaped gauge where log F carries -Phi and the payments carry +Phi, which
    cancels identically.

    Weighted like the learned rule. The weighting is not a detail: on this
    problem the same loss lands anywhere between near-zero and KL 2.7 as the
    on-policy share goes from zero to one, because states the policy abandons
    are never corrected. All three settings are measured and reported rather
    than one being chosen.
    """
    theta, logF = params["theta"], params["logF"]
    pi = grid.policy(theta)
    d = grid.occupancy(pi)
    logp = jnp.log(jnp.where(grid.valid, pi, 1e-300))
    res_stop = logF + logp[:, STOP] - grid.stop_pay(log_r)
    res_r = logF + logp[:, RIGHT] - logF[grid.rc] - grid.r_r
    res_u = logF + logp[:, UP] - logF[grid.uc] - grid.r_u
    w = visit_weight(d, grid.n, on_policy_share)[:, None]
    sq = jnp.stack([res_r ** 2, res_u ** 2, res_stop ** 2], axis=-1)
    return jnp.sum(jnp.where(grid.valid, w * sq, 0.0))


def run_db(grid, log_r, steps, lr, on_policy_share=0.5):
    params = {"theta": jnp.zeros((grid.n, 3)), "logF": jnp.zeros(grid.n)}
    opt = optax.adam(lr)
    st = opt.init(params)

    def body(carry, _):
        p, st = carry
        g = jax.grad(db_loss)(p, grid, log_r, on_policy_share)
        upd, st = opt.update(g, st)
        return (optax.apply_updates(p, upd), st), None

    (params, _), _ = jax.lax.scan(body, (params, st), None, length=steps)
    return params["theta"]


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=6)
    ap.add_argument("--inner", type=int, default=30)
    ap.add_argument("--outer", type=int, default=1500)
    ap.add_argument("--select-outer", type=int, default=400,
                    help="outer steps used when picking step sizes per arm")
    ap.add_argument("--train-tasks", type=int, default=16)
    ap.add_argument("--test-tasks", type=int, default=16)
    ap.add_argument("--eval-mult", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    grid = Grid(args.height, shaped=True)
    key = jax.random.PRNGKey(0)
    k_tr, k_te = jax.random.split(key)
    train = jax.vmap(lambda k: landscape(k, args.height))(
        jax.random.split(k_tr, args.train_tasks))
    test = jax.vmap(lambda k: landscape(k, args.height))(
        jax.random.split(k_te, args.test_tasks))

    print("=" * 82)
    print("STEP 1  the identity the corrected meta-objective rests on")
    print("=" * 82)
    print("  Proposition 1: V_1(s0) = log Z - KL(q || P_B), with supremum log Z.")
    print("  If that holds, maximising the entropy-regularised return in this MDP")
    print("  IS minimising divergence to the target, at the price of a scalar.")
    phi0 = float(np.asarray(grid.phi)[0])
    print(f"\n  shaping shifts every return by -Phi(source) = {-phi0:.5f}, a constant,")
    print("  so the identity to check is V_1 + Phi(source) = log Z")
    print(f"\n  {'task':>6}{'log Z':>10}{'V_1 + Phi(s0)':>16}{'gap':>12}"
          f"{'KL of that policy':>20}")
    gaps = []
    for t in range(4):
        log_r = test[t]
        # Soft-optimal policy by exact value iteration on the shaped rewards.
        h, n = args.height, grid.n
        V = np.zeros(n)
        th = np.full((n, 3), NEG)
        r_r, r_u = np.asarray(grid.r_r), np.asarray(grid.r_u)
        lr_np = np.asarray(log_r)
        phi_np = np.asarray(grid.phi)
        for s in sorted(range(n), key=lambda k: (k // h) + (k % h), reverse=True):
            q = [NEG, NEG, lr_np[s] - phi_np[s]]
            if grid.np_ok_r[s]:
                q[RIGHT] = r_r[s] + V[grid.np_rc[s]]
            if grid.np_ok_u[s]:
                q[UP] = r_u[s] + V[grid.np_uc[s]]
            qa = np.array(q)
            m = qa.max()
            V[s] = m + math.log(np.exp(qa - m).sum())
            th[s] = qa
        term, ret, ent, _, _ = grid.readout(jnp.asarray(th), log_r)
        logZ = float(jax.scipy.special.logsumexp(log_r))
        v1 = float(ret + ent) + phi0
        kl = float(kl_to_target(term, log_r))
        gaps.append(max(abs(v1 - logZ), abs(kl)))
        print(f"  {t:>6}{logZ:>10.5f}{v1:>16.5f}{v1 - logZ:>12.2e}{kl:>20.2e}")
    ok_id = max(gaps) < 1e-6
    print(f"\n  the supremum of V_1 is log Z, and the policy attaining it has zero "
          f"divergence: {'PASS' if ok_id else 'FAIL'}")
    if not ok_id:
        return 1

    print("\n" + "=" * 82)
    print(f"STEP 2  meta-learn the same rule twice, changing only that scalar")
    print("=" * 82)
    print(f"  {args.outer} outer steps, {args.inner} inner steps, "
          f"{args.train_tasks} training landscapes, {args.seeds} seeds")
    print("  the rule reads only local quantities and never sees R/Z or log Z")

    def make_loss(objective, inner_lr):
        @jax.jit
        def loss(eta, log_rs):
            return meta_loss(eta, grid, log_rs, args.inner, inner_lr, objective)
        return jax.value_and_grad(loss)

    def train_rule(objective, seed, inner_lr, outer_lr, outer):
        eta = init_rule(jax.random.PRNGKey(seed))
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(outer_lr))
        st = opt.init(eta)
        loss_fn = make_loss(objective, inner_lr)
        curve = []
        for it in range(outer):
            v, g = loss_fn(eta, train)
            upd, st = opt.update(g, st)
            eta = optax.apply_updates(eta, upd)
            if it % max(1, outer // 6) == 0 or it == outer - 1:
                curve.append((it, float(v)))
        return eta, curve

    long = args.inner * args.eval_mult

    def evaluator(inner_lr, steps):
        @jax.jit
        def evaluate(eta, log_r):
            th, _ = unroll(eta, grid, log_r, steps, inner_lr)
            term, ret, ent, _, _ = grid.readout(th, log_r)
            return kl_to_target(term, log_r), ret, ent
        return evaluate

    # Each arm gets its own step sizes, chosen by its OWN meta-objective on the
    # training landscapes and never by divergence to the target. Without this
    # the return arm saturates its inner softmax within a few steps, its
    # meta-gradient dies, and the comparison becomes trained against untrained
    # rather than one scalar against another.
    print("\n  selecting step sizes per arm, scored by that arm's own training loss")
    grids_hp = [(il, ol) for il in (0.1, 0.3, 0.5) for ol in (3e-3, 1e-2)]
    chosen = {}
    for objective in ("return", "soft"):
        best = None
        for il, ol in grids_hp:
            _, curve = train_rule(objective, 0, il, ol, args.select_outer)
            score = curve[-1][1]
            if best is None or score < best[0]:
                best = (score, il, ol)
        chosen[objective] = (best[1], best[2])
        print(f"    {objective:<7} inner_lr {best[1]}  outer_lr {best[2]}   "
              f"training meta-loss {best[0]:+.3f}")

    results = {}
    for objective in ("return", "soft"):
        il, ol = chosen[objective]
        # Read at the horizon each arm was optimised for, and again four times
        # further out. The first is where the control belongs; the second is
        # where a rule has to keep working to be worth anything.
        ev = {"trained": evaluator(il, args.inner), "long": evaluator(il, long)}
        per_seed = []
        for seed in range(args.seeds):
            eta, curve = train_rule(objective, seed, il, ol, args.outer)
            rec = {"seed": seed, "meta_curve": curve, "inner_lr": il, "outer_lr": ol}
            for tag, fn in ev.items():
                kls, rets, v1s = [], [], []
                for t in range(args.test_tasks):
                    kl, ret, ent = fn(eta, test[t])
                    kls.append(float(kl))
                    rets.append(float(ret))
                    v1s.append(float(ret + ent))
                rec[tag] = {"kl_mean": float(np.mean(kls)),
                            "kl_max": float(np.max(kls)),
                            "return_mean": float(np.mean(rets)),
                            "v1_mean": float(np.mean(v1s))}
            per_seed.append(rec)
            print(f"  {objective:<7} seed {seed}   "
                  f"at {args.inner:>3} steps KL {rec['trained']['kl_mean']:.4f}   "
                  f"at {long:>3} steps KL {rec['long']['kl_mean']:.4f} "
                  f"(worst {rec['long']['kl_max']:.4f})   "
                  f"meta-loss {curve[0][1]:.3f} -> {curve[-1][1]:.3f}")
        results[objective] = per_seed

    # The control that decides whether this is a comparison at all. Each rule
    # has to be the better one at the scalar it was trained on, at the horizon
    # it was trained for. If the return-trained rule is not better at return,
    # the difference in divergence is a difference between a trained rule and
    # an undertrained one and says nothing about meta-objectives.
    def mean_of(o, tag, k):
        return float(np.mean([s[tag][k] for s in results[o]]))

    control = {}
    for tag, label in (("trained", f"at the trained horizon, {args.inner} steps"),
                       ("long", f"extrapolated to {long} steps")):
        rr = mean_of("return", tag, "return_mean")
        rs = mean_of("soft", tag, "return_mean")
        vr = mean_of("return", tag, "v1_mean")
        vs = mean_of("soft", tag, "v1_mean")
        ok = rr > rs and vs > vr
        control[tag] = {"return_rule": {"return": rr, "v1": vr},
                        "soft_rule": {"return": rs, "v1": vs}, "ok": bool(ok)}
        print(f"\n  each rule against both scalars, {label}")
        print(f"      {'rule trained on':<20}{'return':>10}{'V_1':>10}")
        print(f"      {'return':<20}{rr:>10.3f}{vr:>10.3f}")
        print(f"      {'V_1':<20}{rs:>10.3f}{vs:>10.3f}")
        print(f"      each rule wins on the scalar it was trained on: "
              f"{'PASS' if ok else 'FAIL'}")
    control_ok = control["trained"]["ok"]

    print("\n" + "=" * 84)
    print("STEP 3  the hand-designed reference, and the reading")
    print("=" * 84)
    db = {}
    for steps in (args.inner, 400):
        for share, label in ((0.0, "full support"), (0.5, "matched weighting"),
                             (1.0, "purely on-policy")):
            kls = []
            for t in range(args.test_tasks):
                th = run_db(grid, test[t], steps, 0.05, share)
                term, _, _, _, _ = grid.readout(th, test[t])
                kls.append(float(kl_to_target(term, test[t])))
            db[f"{steps} steps, {label}"] = float(np.mean(kls))
            print(f"  detailed balance, {steps:>3} Adam steps, {label:<18} "
                  f"held-out KL = {np.mean(kls):.4f}")
    db_matched = min(v for k, v in db.items() if k.startswith(f"{args.inner} "))
    db_best = min(db.values())
    print(f"\n  the baseline is taken at its best setting throughout, which is the")
    print(f"  conservative choice; at a budget matched to the rule it reaches "
          f"{db_matched:.4f},")
    print(f"  and given 400 steps it reaches {db_best:.4f}")

    def agg(o, tag):
        return (float(np.mean([s[tag]["kl_mean"] for s in results[o]])),
                float(np.std([s[tag]["kl_mean"] for s in results[o]])))

    print(f"\n  {'':<34}{'at ' + str(args.inner) + ' steps':>18}"
          f"{'at ' + str(long) + ' steps':>18}")
    for o, label in (("return", "meta-trained on the return"),
                     ("soft", "meta-trained on V_1 = log Z - KL")):
        a, sa = agg(o, "trained")
        b, sb = agg(o, "long")
        print(f"  {label:<34}{a:>10.4f} +/-{sa:>5.3f}{b:>10.4f} +/-{sb:>5.3f}")
    r_mu, r_sd = agg("return", "trained")
    s_mu, s_sd = agg("soft", "trained")
    rl_mu, _ = agg("return", "long")
    sl_mu, _ = agg("soft", "long")
    print(f"  the two rules are identical in architecture, inputs, inner horizon and")
    print(f"  seeds and differ only in that scalar, on landscapes never seen")

    print()
    if not control_ok:
        verdict = ("VOID -- at the horizon they were trained for, the two rules do "
                   "not each win on the scalar they were trained on, so the "
                   "difference in divergence is not attributable to the objective")
    elif r_mu <= s_mu:
        verdict = (f"NO SEPARATION -- the return-trained rule reaches KL {r_mu:.3f} "
                   f"against {s_mu:.3f} for the corrected objective")
    else:
        rel = r_mu / max(s_mu, 1e-12)
        vs = ("beats" if s_mu < db_matched else "does not beat")
        verdict = (f"THE META-OBJECTIVE IS THE WHOLE DIFFERENCE -- one scalar, changed "
                   f"from the return to V_1 = log Z - KL, moves the learned rule from "
                   f"KL {r_mu:.4f} to {s_mu:.4f} at the horizon both were trained for, "
                   f"a factor of {rel:.0f}; it {vs} detailed balance at the same budget "
                   f"({db_matched:.4f}) and the soft rule degrades to {sl_mu:.4f} when "
                   f"unrolled {args.eval_mult} times further")
    print(f"Verdict: {verdict}")

    out = {"config": vars(args), "identity_gap": max(gaps), "chosen": chosen,
           "results": results, "db": db, "control": control,
           "control_ok": bool(control_ok), "verdict": verdict}
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
