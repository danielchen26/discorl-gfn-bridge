/**
 * Parity: the browser engine must agree with the Python oracle.
 *
 * src/lib/hypergrid.ts is a hand-written mirror of research/cumulants.py, and a
 * mirror that silently drifts is worse than no mirror at all -- every number the
 * app renders would become unfalsifiable. So this compares the two
 * implementations at every policy and every dial position, and fails loudly.
 *
 * Run:  node research/parity.ts       (Node >= 22.6 strips the types natively)
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import {
  Grid,
  evaluate,
  fitGamma,
  flowMatchingPolicy,
  klToTarget,
  maxentPolicy,
  uniformPolicy,
  workStats,
} from "../src/lib/hypergrid.ts";

const here = dirname(fileURLToPath(import.meta.url));
const oracle = JSON.parse(readFileSync(join(here, "cumulants.json"), "utf8"));

/** Absolute tolerance. The two implementations do the same operations in the
 *  same order, so anything above float noise is a real divergence. */
const TOL = 1e-12;

let failures = 0;
let checked = 0;

function cmp(label: string, ts: number, py: number, tol = TOL): void {
  checked++;
  const d = Math.abs(ts - py);
  if (!(d <= tol)) {
    failures++;
    console.log(`  MISMATCH ${label}\n           ts=${ts}\n           py=${py}\n           |d|=${d}`);
  }
}

const g = new Grid(oracle.height);

cmp("Z", g.Z, oracle.Z);
cmp("log Z", g.logZ, oracle.log_Z);

const byName: Record<string, () => Float64Array> = {
  "flow-matching (GFN)": () => flowMatchingPolicy(g),
  "soft RL (MaxEnt)": () => maxentPolicy(g),
  uniform: () => uniformPolicy(g),
};

console.log(`parity: hypergrid H=${oracle.height}, ${oracle.n_trajectories} trajectories\n`);

for (const row of oracle.policies) {
  const build = byName[row.policy];
  if (!build) throw new Error(`oracle has an unknown policy: ${row.policy}`);
  const pol = build();
  const s = workStats(g, pol);
  const tag = row.policy.padEnd(20);
  cmp(`${tag} E[W]`, s.meanW, row.mean_W);
  cmp(`${tag} Var[W]`, s.varW, row.var_W);
  cmp(`${tag} E[exp(-W)]`, s.jarzynski, row.jarzynski);
  cmp(`${tag} gamma`, fitGamma(g, s.terminal), row.gamma);
  cmp(`${tag} KL(p||R/Z)`, klToTarget(g, s.terminal), row.kl_terminal);

  // The identities themselves, restated against the TypeScript numbers so the
  // app cannot ship a build that quietly violates them.
  cmp(`${tag} Jarzynski == Z`, s.jarzynski, g.Z, 1e-9);
  cmp(`${tag} second law >= 0`, Math.max(0, g.logZ + s.meanW), g.logZ + s.meanW, 1e-9);
}

const a = maxentPolicy(g);
const b = flowMatchingPolicy(g);
for (const d of oracle.dial) {
  const e = evaluate(g, a, b, d.lam);
  const tag = `dial lam=${d.lam.toFixed(3)}`;
  cmp(`${tag} E[W]`, e.meanW, d.mean_W);
  cmp(`${tag} Var[W]`, e.varW, d.var_W);
  cmp(`${tag} gamma`, e.gamma, d.gamma);
  cmp(`${tag} KL`, e.klTerminal, d.kl_terminal);
}

console.log(`${checked} comparisons, ${failures} mismatches`);
if (failures > 0) {
  console.log("\nFAIL: the TypeScript engine and the Python oracle disagree.");
  process.exit(1);
}
console.log("PASS: browser engine is bit-compatible with the Python oracle.");
