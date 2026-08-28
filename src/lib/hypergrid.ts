/* ============================================================================
   Exact hypergrid DAG engine.

   A line-for-line mirror of research/cumulants.py.  Every quantity the app
   displays is computed here by exact dynamic programming -- no sampling, no
   training loop -- which is what lets the dials respond in real time and lets
   research/parity.ts assert bit-level agreement with the Python oracle.

   States are lattice points (i, j) with 0 <= i, j < h.  Actions are
   RIGHT / UP / STOP.  Every trajectory is a monotone lattice path from (0,0)
   followed by a stop, so the number of distinct paths reaching (i, j) is the
   binomial coefficient C(i+j, i).
   ========================================================================= */

export const RIGHT = 0;
export const UP = 1;
export const STOP = 2;

export const R0 = 1e-2;
export const R1 = 0.5;
export const R2 = 2.0;

/** Standard hypergrid reward (Bengio et al. 2021), strictly positive. */
export function reward(i: number, j: number, h: number): number {
  let r = R0;
  const ax = Math.abs(i / (h - 1) - 0.5);
  const ay = Math.abs(j / (h - 1) - 0.5);
  if (ax > 0.25 && ay > 0.25) r += R1;
  if (ax > 0.3 && ax < 0.4 && ay > 0.3 && ay < 0.4) r += R2;
  return r;
}

export class Grid {
  readonly h: number;
  readonly n: number;
  /** State indices sorted by i+j descending (sinks first). */
  readonly revTopo: Int32Array;
  /** State indices sorted by i+j ascending (source first). */
  readonly fwdTopo: Int32Array;
  readonly rewards: Float64Array;
  readonly logPaths: Float64Array;
  readonly len: Float64Array;
  readonly nParents: Int32Array;
  readonly Z: number;
  readonly logZ: number;

  constructor(h: number) {
    this.h = h;
    this.n = h * h;
    this.rewards = new Float64Array(this.n);
    this.logPaths = new Float64Array(this.n);
    this.len = new Float64Array(this.n);
    this.nParents = new Int32Array(this.n);

    // log C(i+j, i) via log-gamma-free exact summation: the grid is small, so
    // an integer binomial in doubles is exact well past H = 12.
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < h; j++) {
        const k = i * h + j;
        this.rewards[k] = reward(i, j, h);
        this.logPaths[k] = Math.log(binom(i + j, i));
        this.len[k] = i + j;
        this.nParents[k] = (i > 0 ? 1 : 0) + (j > 0 ? 1 : 0);
      }
    }

    const order = Array.from({ length: this.n }, (_, k) => k);
    order.sort((a, b) => this.len[a] - this.len[b]);
    this.fwdTopo = Int32Array.from(order);
    this.revTopo = Int32Array.from([...order].reverse());

    let z = 0;
    for (let k = 0; k < this.n; k++) z += this.rewards[k];
    this.Z = z;
    this.logZ = Math.log(z);
  }

  idx(i: number, j: number): number {
    return i * this.h + j;
  }
  row(k: number): number {
    return Math.floor(k / this.h);
  }
  col(k: number): number {
    return k % this.h;
  }
  /** Child state index for an action, or -1 when unavailable. */
  child(k: number, a: number): number {
    const i = this.row(k);
    const j = this.col(k);
    if (a === RIGHT) return i + 1 < this.h ? this.idx(i + 1, j) : -1;
    if (a === UP) return j + 1 < this.h ? this.idx(i, j + 1) : -1;
    return -1;
  }
}

function binom(n: number, k: number): number {
  let r = 1;
  const m = Math.min(k, n - k);
  for (let t = 0; t < m; t++) r = (r * (n - t)) / (t + 1);
  return Math.round(r);
}

/** Policy: flat [state * 3 + action] probabilities; unavailable actions are 0. */
export type Policy = Float64Array;

/**
 * The exact GFlowNet solution for a backward policy that is uniform over
 * parents.  On a DAG the flow condition must be stated on EDGE flows,
 * F(s -> c) = p_B(s|c) F(c), otherwise every child is counted once per parent.
 * Under this policy W(tau) = -log Z identically, so Var[W] = 0.
 */
export function flowMatchingPolicy(g: Grid): Policy {
  const f = new Float64Array(g.n);
  for (let t = 0; t < g.n; t++) {
    const k = g.revTopo[t];
    let acc = g.rewards[k];
    for (const a of [RIGHT, UP]) {
      const c = g.child(k, a);
      if (c >= 0) acc += f[c] / g.nParents[c];
    }
    f[k] = acc;
  }
  const pol = new Float64Array(g.n * 3);
  for (let k = 0; k < g.n; k++) {
    pol[k * 3 + STOP] = g.rewards[k] / f[k];
    for (const a of [RIGHT, UP]) {
      const c = g.child(k, a);
      if (c >= 0) pol[k * 3 + a] = f[c] / g.nParents[c] / f[k];
    }
  }
  return pol;
}

/**
 * Entropy-regularised (soft) RL optimum against a uniform reference policy.
 * It puts P(tau) proportional to pi0(tau) R(x), so the terminal marginal is
 * inflated by path multiplicity -- the multi-path bias of Deleu et al. (2024).
 */
export function maxentPolicy(g: Grid): Policy {
  const val = new Float64Array(g.n);
  for (let t = 0; t < g.n; t++) {
    const k = g.revTopo[t];
    let nAct = 1;
    for (const a of [RIGHT, UP]) if (g.child(k, a) >= 0) nAct++;
    let acc = g.rewards[k] / nAct;
    for (const a of [RIGHT, UP]) {
      const c = g.child(k, a);
      if (c >= 0) acc += val[c] / nAct;
    }
    val[k] = acc;
  }
  const pol = new Float64Array(g.n * 3);
  for (let k = 0; k < g.n; k++) {
    let nAct = 1;
    for (const a of [RIGHT, UP]) if (g.child(k, a) >= 0) nAct++;
    const tot = val[k] * nAct;
    pol[k * 3 + STOP] = g.rewards[k] / tot;
    for (const a of [RIGHT, UP]) {
      const c = g.child(k, a);
      if (c >= 0) pol[k * 3 + a] = val[c] / tot;
    }
  }
  return pol;
}

export function uniformPolicy(g: Grid): Policy {
  const pol = new Float64Array(g.n * 3);
  for (let k = 0; k < g.n; k++) {
    let nAct = 1;
    for (const a of [RIGHT, UP]) if (g.child(k, a) >= 0) nAct++;
    pol[k * 3 + STOP] = 1 / nAct;
    for (const a of [RIGHT, UP]) if (g.child(k, a) >= 0) pol[k * 3 + a] = 1 / nAct;
  }
  return pol;
}

/** Renormalised geometric interpolation p ~ a^(1-lam) b^lam. */
export function geometricMix(g: Grid, a: Policy, b: Policy, lam: number): Policy {
  const pol = new Float64Array(g.n * 3);
  for (let k = 0; k < g.n; k++) {
    let tot = 0;
    for (let x = 0; x < 3; x++) {
      const pa = a[k * 3 + x];
      const pb = b[k * 3 + x];
      const v = pa === 0 && pb === 0 ? 0 : Math.pow(pa, 1 - lam) * Math.pow(pb, lam);
      pol[k * 3 + x] = v;
      tot += v;
    }
    for (let x = 0; x < 3; x++) pol[k * 3 + x] /= tot;
  }
  return pol;
}

export type Stats = {
  /** E[W] under P_F. */
  meanW: number;
  /** Var[W] -- equals min_c E[(c+W)^2], i.e. the trajectory-balance optimum. */
  varW: number;
  /** E[exp(-W)]; the Jarzynski identity says this is Z for ANY forward policy. */
  jarzynski: number;
  /** Terminal distribution over objects. */
  terminal: Float64Array;
};

/**
 * W is additive along a trajectory, so its first two moments and its
 * exponential average all satisfy backward recursions over the DAG.
 *
 *   W(tau) = sum_t [ log p_F(a_t|s_t) - log p_B(s_t|s_t+1) ] - log R(x)
 */
export function workStats(g: Grid, pol: Policy): Stats {
  const m1 = new Float64Array(g.n);
  const m2 = new Float64Array(g.n);
  const jz = new Float64Array(g.n);

  for (let t = 0; t < g.n; t++) {
    const k = g.revTopo[t];
    let a1 = 0;
    let a2 = 0;
    let aj = 0;
    for (let a = 0; a < 3; a++) {
      const p = pol[k * 3 + a];
      if (p <= 0) continue;
      const c = a === STOP ? -1 : g.child(k, a);
      if (a !== STOP && c < 0) continue;
      // p_B is uniform over the parents of the child.
      const w = a === STOP ? Math.log(p) - Math.log(g.rewards[k]) : Math.log(p) + Math.log(g.nParents[c]);
      const n1 = c < 0 ? 0 : m1[c];
      const n2 = c < 0 ? 0 : m2[c];
      const nj = c < 0 ? 1 : jz[c];
      a1 += p * (w + n1);
      a2 += p * (w * w + 2 * w * n1 + n2);
      aj += p * Math.exp(-w) * nj;
    }
    m1[k] = a1;
    m2[k] = a2;
    jz[k] = aj;
  }

  const src = 0;
  const meanW = m1[src];
  const varW = m2[src] - meanW * meanW;

  const reach = new Float64Array(g.n);
  const terminal = new Float64Array(g.n);
  reach[src] = 1;
  for (let t = 0; t < g.n; t++) {
    const k = g.fwdTopo[t];
    terminal[k] = reach[k] * pol[k * 3 + STOP];
    for (const a of [RIGHT, UP]) {
      const c = g.child(k, a);
      if (c >= 0) reach[c] += reach[k] * pol[k * 3 + a];
    }
  }

  return { meanW, varW, jarzynski: jz[src], terminal };
}

/**
 * The multi-path exponent gamma in p(x) ~ R(x) n(x)^gamma.
 *
 * Regressing on log n(x) alone is confounded, because under a uniform
 * reference a longer path also carries less prior mass and length correlates
 * with log n on this grid.  So we regress
 *
 *     log p(x) - log R(x)   on   [ log n(x), len(x), 1 ]
 *
 * and report the coefficient on log n(x): the inflation attributable to path
 * count alone, holding trajectory length fixed.  Flow matching gives 0, soft
 * RL gives ~1.
 */
export function fitGamma(g: Grid, terminal: Float64Array): number {
  const K = 3;
  const ata = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const aty = [0, 0, 0];
  for (let k = 0; k < g.n; k++) {
    const p = terminal[k];
    if (p <= 0) continue;
    const r = [g.logPaths[k], g.len[k], 1];
    const y = Math.log(p) - Math.log(g.rewards[k]);
    for (let a = 0; a < K; a++) {
      aty[a] += r[a] * y;
      for (let b = 0; b < K; b++) ata[a][b] += r[a] * r[b];
    }
  }
  const aug = ata.map((row, a) => [...row, aty[a]]);
  for (let col = 0; col < K; col++) {
    let piv = col;
    for (let r = col; r < K; r++) if (Math.abs(aug[r][col]) > Math.abs(aug[piv][col])) piv = r;
    const tmp = aug[col];
    aug[col] = aug[piv];
    aug[piv] = tmp;
    const pv = aug[col][col];
    for (let r = 0; r < K; r++) {
      if (r === col) continue;
      const fac = aug[r][col] / pv;
      for (let c = col; c <= K; c++) aug[r][c] -= fac * aug[col][c];
    }
  }
  return aug[0][K] / aug[0][0];
}

/** KL(p_terminal || R/Z) -- the sampler's residual bias. */
export function klToTarget(g: Grid, terminal: Float64Array): number {
  let acc = 0;
  for (let k = 0; k < g.n; k++) {
    const p = terminal[k];
    if (p <= 0) continue;
    acc += p * (Math.log(p) - (Math.log(g.rewards[k]) - g.logZ));
  }
  return acc;
}

export type DialPoint = {
  lam: number;
  meanW: number;
  varW: number;
  gamma: number;
  klTerminal: number;
  elboGap: number;
  jarzynski: number;
};

/** Everything the dial needs at one setting of lambda. */
export function evaluate(g: Grid, a: Policy, b: Policy, lam: number): DialPoint & { terminal: Float64Array } {
  const pol = lam <= 0 ? a : lam >= 1 ? b : geometricMix(g, a, b, lam);
  const s = workStats(g, pol);
  return {
    lam,
    meanW: s.meanW,
    varW: s.varW,
    gamma: fitGamma(g, s.terminal),
    klTerminal: klToTarget(g, s.terminal),
    elboGap: g.logZ + s.meanW,
    jarzynski: s.jarzynski,
    terminal: s.terminal,
  };
}
