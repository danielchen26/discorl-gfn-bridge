import { useMemo, useState } from "react";

import { Grid, evaluate, flowMatchingPolicy, maxentPolicy } from "../lib/hypergrid";
import { M, Readout, Rig } from "./ui";
import { useLang } from "../i18n";

const H = 8;
const STEPS = 101;

/**
 * The signature rig.
 *
 * One dial moves the forward policy geometrically from the soft-RL optimum to
 * the flow-matching optimum. Everything on screen is recomputed exactly, by
 * dynamic programming over the DAG, at every frame -- there is no fitting and
 * no sampling anywhere in this component.
 *
 * The point it exists to make: Var[W], the sampler's bias, and the ELBO gap all
 * collapse together, while E[exp(-W)] never moves off Z. The first three are
 * properties of a particular policy; the last is an identity.
 */
export default function WorkDial() {
  const lang = useLang();
  const [i, setI] = useState(0);

  const { g, sweep } = useMemo(() => {
    const grid = new Grid(H);
    const a = maxentPolicy(grid);
    const b = flowMatchingPolicy(grid);
    const pts = Array.from({ length: STEPS }, (_, k) => evaluate(grid, a, b, k / (STEPS - 1)));
    return { g: grid, sweep: pts };
  }, []);

  const cur = sweep[i];
  const maxVar = Math.max(...sweep.map((p) => p.varW));
  const maxKl = Math.max(...sweep.map((p) => p.klTerminal));

  const W = 560;
  const Hh = 210;
  const PAD = { l: 40, r: 14, t: 14, b: 26 };
  const iw = W - PAD.l - PAD.r;
  const ih = Hh - PAD.t - PAD.b;

  const path = (sel: (p: (typeof sweep)[number]) => number, scale: number) =>
    sweep
      .map((p, k) => {
        const x = PAD.l + (k / (STEPS - 1)) * iw;
        const y = PAD.t + ih - (sel(p) / scale) * ih;
        return `${k === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");

  const cx = PAD.l + (i / (STEPS - 1)) * iw;

  // Free-energy ledger: log Z is fixed; the ELBO closes the gap as lambda -> 1.
  const barW = 560;
  const elboFrac = Math.max(0, -cur.meanW) / g.logZ;

  return (
    <Rig
      title={{
        zh: "同一个 W(τ),两种取法",
        en: "One W(τ), two ways to take it",
      }}
      note={{
        zh: "把前向策略从「软 RL 最优」几何插值到「流匹配最优」。所有数字都是 DAG 上的精确动态规划,不含采样、不含训练。",
        en: "Interpolate the forward policy from the soft-RL optimum to the flow-matching optimum. Every number is exact dynamic programming on the DAG — no sampling, no training.",
      }}
    >
      <div className="dial-row">
        <label htmlFor="lam">λ</label>
        <input
          id="lam"
          type="range"
          min={0}
          max={STEPS - 1}
          step={1}
          value={i}
          onChange={(e) => setI(Number(e.target.value))}
        />
        <span style={{ fontFamily: "var(--mono)", fontSize: 13, minWidth: 52 }}>{cur.lam.toFixed(2)}</span>
      </div>
      <div className="ends">
        <span className="a">
          λ = 0 · {lang === "zh" ? "软 RL 最优(最小化 ⟨W⟩)" : "soft-RL optimum (minimises ⟨W⟩)"}
        </span>
        <span className="b">
          {lang === "zh" ? "流匹配最优(最小化 Var[W])" : "flow matching (minimises Var[W])"} · λ = 1
        </span>
      </div>

      <div className="readouts">
        <Readout
          k={{ zh: "Var[W] — TB 目标", en: "Var[W] — the TB objective" }}
          v={Math.abs(cur.varW) < 1e-9 ? `|${Math.abs(cur.varW).toExponential(1)}|` : cur.varW.toFixed(4)}
          tone="gfn"
          note={{ zh: "= min_c E[(c+W)²]", en: "= min_c E[(c+W)²]" }}
        />
        <Readout
          k={{ zh: "γ — 多路径偏差", en: "γ — multi-path bias" }}
          v={cur.gamma.toFixed(4)}
          tone="rl"
          note={{ zh: "p(x) ∝ R(x)·n(x)^γ", en: "p(x) ∝ R(x)·n(x)^γ" }}
        />
        <Readout
          k={{ zh: "KL(p ‖ R/Z)", en: "KL(p ‖ R/Z)" }}
          v={cur.klTerminal < 1e-9 ? "0" : cur.klTerminal.toFixed(4)}
          tone="conj"
          note={{ zh: "采样器实际偏差", en: "the sampler's actual error" }}
        />
        <Readout
          k={{ zh: "E[e^−W] — Jarzynski", en: "E[e^−W] — Jarzynski" }}
          v={cur.jarzynski.toFixed(10)}
          locked
          note={{
            zh: `恒等于 Z = ${g.Z.toFixed(2)}，与 λ 无关`,
            en: `identically Z = ${g.Z.toFixed(2)}, for every λ`,
          }}
        />
      </div>

      <div className="grid2" style={{ marginTop: 26 }}>
        <div>
          <svg className="plot" viewBox={`0 0 ${W} ${Hh}`} role="img" aria-label="dial traces">
            {[0, 0.25, 0.5, 0.75, 1].map((f) => (
              <line
                key={f}
                className="grid"
                x1={PAD.l}
                x2={W - PAD.r}
                y1={PAD.t + ih - f * ih}
                y2={PAD.t + ih - f * ih}
              />
            ))}
            <line className="axis" x1={PAD.l} x2={W - PAD.r} y1={PAD.t + ih} y2={PAD.t + ih} />
            <line className="axis" x1={PAD.l} x2={PAD.l} y1={PAD.t} y2={PAD.t + ih} />

            <path d={path((p) => p.varW, maxVar)} fill="none" stroke="var(--gfn)" strokeWidth={2} />
            <path d={path((p) => p.klTerminal, maxKl)} fill="none" stroke="var(--conj)" strokeWidth={2} />
            <path d={path((p) => Math.max(0, p.gamma), 1)} fill="none" stroke="var(--rl)" strokeWidth={2} />

            <line className="axis" x1={cx} x2={cx} y1={PAD.t} y2={PAD.t + ih} stroke="var(--text-3)" strokeDasharray="3 3" />
            <circle cx={cx} cy={PAD.t + ih - (cur.varW / maxVar) * ih} r={4} fill="var(--gfn)" />
            <circle cx={cx} cy={PAD.t + ih - (cur.klTerminal / maxKl) * ih} r={4} fill="var(--conj)" />
            <circle cx={cx} cy={PAD.t + ih - Math.max(0, cur.gamma) * ih} r={4} fill="var(--rl)" />

            <text x={PAD.l} y={Hh - 8}>
              λ = 0
            </text>
            <text x={W - PAD.r} y={Hh - 8} textAnchor="end">
              λ = 1
            </text>
            <text x={PAD.l - 8} y={PAD.t + 4} textAnchor="end">
              1
            </text>
            <text x={PAD.l - 8} y={PAD.t + ih + 3} textAnchor="end">
              0
            </text>
          </svg>
          <div className="legend">
            <span>
              <i style={{ background: "var(--gfn)" }} />
              Var[W]
            </span>
            <span>
              <i style={{ background: "var(--rl)" }} />γ
            </span>
            <span>
              <i style={{ background: "var(--conj)" }} />
              KL(p ‖ R/Z)
            </span>
          </div>
          <p className="hint">
            {lang === "zh" ? (
              <>
                三条线<b>同时归零</b>。这不是巧合:Var[W] = 0 ⟺ W 恒定 ⟺ 零耗散 ⟺ 采样器精确。
                <br />
                纵轴:每条曲线各自归一化到自身在 λ ∈ [0,1] 上的最大值(γ 除外,它本身就以 1 为标度)。
              </>
            ) : (
              <>
                All three collapse <b>together</b>. Not a coincidence: Var[W] = 0 ⟺ W is constant ⟺ zero
                dissipation ⟺ the sampler is exact.
                <br />
                Vertical axis: each curve is scaled to its own maximum over λ ∈ [0,1] — except γ, which is
                already on a unit scale.
              </>
            )}
          </p>
        </div>

        <div>
          <svg className="plot" viewBox={`0 0 ${barW} ${Hh}`} role="img" aria-label="free energy ledger">
            <text x={0} y={16} className="lbl">
              {lang === "zh" ? "自由能账本" : "free-energy ledger"}
            </text>
            <text x={barW} y={16} textAnchor="end" className="lbl">
              log Z = {g.logZ.toFixed(4)}
            </text>

            <rect x={0} y={38} width={barW} height={30} rx={4} fill="var(--surface-2)" stroke="var(--line)" />
            <rect x={0} y={38} width={barW * Math.min(1, elboFrac)} height={30} rx={4} fill="var(--gfn)" opacity={0.75} />
            <text x={10} y={58} fill="#0b0e17" style={{ fontWeight: 600 }}>
              ELBO = −E[W] = {(-cur.meanW).toFixed(4)}
            </text>

            <rect
              x={barW * Math.min(1, elboFrac)}
              y={78}
              width={barW * Math.max(0, 1 - elboFrac)}
              height={18}
              rx={3}
              fill="var(--conj)"
              opacity={0.55}
            />
            <text x={4} y={92} className="lbl">
              {lang === "zh" ? "缺口 = ⟨W_diss⟩ = KL(P_F ‖ P_B)" : "gap = ⟨W_diss⟩ = KL(P_F ‖ P_B)"} ={" "}
              {Math.max(0, cur.elboGap).toFixed(4)}
            </text>

            <text x={0} y={132} className="lbl">
              {lang === "zh" ? "累积量展开" : "cumulant expansion"}
            </text>
            <foreignObject x={0} y={140} width={barW} height={66}>
              <div style={{ fontSize: 12, color: "var(--text-2)", lineHeight: 1.5 }}>
                <M tex={String.raw`\log Z=\log\mathbb E\big[e^{-W}\big]=-\langle W\rangle+\tfrac12\mathrm{Var}(W)-\tfrac16\kappa_3+\cdots`} />
              </div>
            </foreignObject>
          </svg>
          <p className="hint">
            {lang === "zh" ? (
              <>
                VI 只截到<b>一阶</b>,所以永远差一个 KL 缺口。TB 把<b>二阶</b>压到零,截断因此变成精确 ——
                而它的自由参数 log Z<sub>θ</sub> 自动落在 ELBO 上。
              </>
            ) : (
              <>
                VI truncates at the <b>first</b> cumulant, so a KL gap always remains. TB drives the{" "}
                <b>second</b> to zero, which makes the truncation exact — and its free parameter log Z
                <sub>θ</sub> lands on the ELBO by itself.
              </>
            )}
          </p>
        </div>
      </div>
    </Rig>
  );
}
