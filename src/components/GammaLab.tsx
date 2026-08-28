import { useMemo, useState } from "react";

import { Grid, evaluate, flowMatchingPolicy, maxentPolicy } from "../lib/hypergrid";
import { Chip, Readout, Rig } from "./ui";
import { useLang } from "../i18n";

const H = 8;
const STEPS = 101;
const CELL = 26;

/**
 * The measurement rig for the falsifiable claim.
 *
 * gamma is the exponent in p(x) ∝ R(x)·n(x)^γ, estimated by regressing
 * log p − log R on [log n, length, 1]. Flow matching pins it at 0; soft RL puts
 * it at ~1. The conjecture is that a Disco103-trained agent lands strictly
 * inside that interval, and this rig is what the real experiment would report.
 */
export default function GammaLab() {
  const lang = useLang();
  const [i, setI] = useState(0);

  const { g, sweep, maxLogN } = useMemo(() => {
    const grid = new Grid(H);
    const a = maxentPolicy(grid);
    const b = flowMatchingPolicy(grid);
    const pts = Array.from({ length: STEPS }, (_, k) => evaluate(grid, a, b, k / (STEPS - 1)));
    let mx = 0;
    for (let k = 0; k < grid.n; k++) mx = Math.max(mx, grid.logPaths[k]);
    return { g: grid, sweep: pts, maxLogN: mx };
  }, []);

  const cur = sweep[i];

  const target = useMemo(() => {
    const t = new Float64Array(g.n);
    for (let k = 0; k < g.n; k++) t[k] = g.rewards[k] / g.Z;
    return t;
  }, [g]);

  const maxP = Math.max(...Array.from(cur.terminal), ...Array.from(target));

  const heat = (v: Float64Array, colour: string, label: string) => (
    <div>
      <div style={{ fontFamily: "var(--mono)", fontSize: 10.5, color: "var(--text-3)", marginBottom: 8 }}>{label}</div>
      <svg
        className="plot"
        viewBox={`0 0 ${H * CELL + 1} ${H * CELL + 1}`}
        style={{ maxWidth: H * CELL + 1 }}
        role="img"
        aria-label={label}
      >
        {Array.from({ length: g.n }, (_, k) => {
          const i0 = g.row(k);
          const j0 = g.col(k);
          return (
            <rect
              key={k}
              x={i0 * CELL + 0.5}
              y={(H - 1 - j0) * CELL + 0.5}
              width={CELL - 1}
              height={CELL - 1}
              rx={2}
              fill={colour}
              fillOpacity={Math.pow(v[k] / maxP, 0.55)}
              stroke="var(--line-2)"
            />
          );
        })}
      </svg>
    </div>
  );

  // Scatter: log p − log R against log n(x).
  const SW = 480;
  const SH = 250;
  const PAD = { l: 46, r: 16, t: 16, b: 34 };
  const iw = SW - PAD.l - PAD.r;
  const ih = SH - PAD.t - PAD.b;

  const pts = useMemo(() => {
    const out: { x: number; y: number; len: number }[] = [];
    for (let k = 0; k < g.n; k++) {
      const p = cur.terminal[k];
      if (p <= 0) continue;
      out.push({ x: g.logPaths[k], y: Math.log(p) - Math.log(g.rewards[k]), len: g.len[k] });
    }
    return out;
  }, [g, cur]);

  const yMin = Math.min(...pts.map((p) => p.y));
  const yMax = Math.max(...pts.map((p) => p.y));
  const ySpan = yMax - yMin || 1;

  return (
    <Rig
      title={{ zh: "多路径偏差:γ 怎么量", en: "Multi-path bias: how γ is measured" }}
      note={{
        zh: "同一个终点可以有很多条生成路径。区分 GFlowNet 和 MaxEnt RL 的唯一决定性诊断,就是采样概率是否被路径数抬高。",
        en: "The same object can be built along many paths. Whether sampling probability gets inflated by path count is the one decisive diagnostic separating GFlowNets from MaxEnt RL.",
      }}
      right={<Chip p="verified" />}
    >
      <div className="dial-row">
        <label htmlFor="lam2">λ</label>
        <input
          id="lam2"
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
        <span className="a">{lang === "zh" ? "软 RL · γ ≈ 1" : "soft RL · γ ≈ 1"}</span>
        <span className="b">{lang === "zh" ? "流匹配 · γ = 0" : "flow matching · γ = 0"}</span>
      </div>

      <div className="readouts" style={{ marginBottom: 24 }}>
        <Readout
          k={{ zh: "拟合 γ", en: "fitted γ" }}
          v={cur.gamma.toFixed(4)}
          tone={cur.gamma > 0.5 ? "rl" : "gfn"}
          note={{ zh: "控制轨迹长度后的 log n 系数", en: "coefficient on log n, length held fixed" }}
        />
        <Readout
          k={{ zh: "最大路径数 n(x)", en: "largest n(x)" }}
          v={Math.round(Math.exp(maxLogN)).toLocaleString()}
          note={{ zh: "对角终点的单调格路数", en: "monotone lattice paths to the far corner" }}
        />
        <Readout
          k={{ zh: "KL(p ‖ R/Z)", en: "KL(p ‖ R/Z)" }}
          v={cur.klTerminal < 1e-9 ? "0" : cur.klTerminal.toFixed(4)}
          tone="conj"
        />
      </div>

      <div className="grid2">
        <div>
          <div style={{ display: "flex", gap: 22, flexWrap: "wrap" }}>
            {heat(target, "var(--gfn)", lang === "zh" ? "目标 R(x)/Z" : "target R(x)/Z")}
            {heat(cur.terminal, "var(--rl)", lang === "zh" ? "实际采样 p(x)" : "sampled p(x)")}
          </div>
          <p className="hint">
            {lang === "zh" ? (
              <>
                λ = 0 时右图的质量被拉向<b>原点</b> —— 均匀参考策略下,每多走一步就多乘一个 1/3,所以短路径的参考测度更大。
                路径数的影响被长度效应盖住了,这正是 γ 必须<b>控制轨迹长度</b>才能读出来的原因。
              </>
            ) : (
              <>
                At λ = 0 the right panel's mass is pulled toward the <b>origin</b>: under a uniform reference
                every extra step costs another factor of ⅓, so short paths carry more reference mass. The path
                count effect is buried under the length effect — which is exactly why γ has to{" "}
                <b>hold length fixed</b>.
              </>
            )}
          </p>
        </div>

        <div>
          <svg className="plot" viewBox={`0 0 ${SW} ${SH}`} role="img" aria-label="gamma scatter">
            <line className="axis" x1={PAD.l} x2={SW - PAD.r} y1={PAD.t + ih} y2={PAD.t + ih} />
            <line className="axis" x1={PAD.l} x2={PAD.l} y1={PAD.t} y2={PAD.t + ih} />
            {pts.map((p, k) => (
              <circle
                key={k}
                cx={PAD.l + (p.x / maxLogN) * iw}
                cy={PAD.t + ih - ((p.y - yMin) / ySpan) * ih}
                r={3.2}
                fill={`color-mix(in oklab, var(--gfn) ${100 - (p.len / (2 * H - 2)) * 100}%, var(--rl))`}
                fillOpacity={0.8}
              />
            ))}
            <text x={SW / 2} y={SH - 6} textAnchor="middle" className="lbl">
              log n(x)
            </text>
            <text x={12} y={PAD.t + ih / 2} className="lbl" transform={`rotate(-90 12 ${PAD.t + ih / 2})`}>
              log p(x) − log R(x)
            </text>
          </svg>
          <div className="legend">
            <span>
              <i style={{ background: "var(--gfn)" }} />
              {lang === "zh" ? "短轨迹" : "short trajectory"}
            </span>
            <span>
              <i style={{ background: "var(--rl)" }} />
              {lang === "zh" ? "长轨迹" : "long trajectory"}
            </span>
          </div>
          <p className="hint">
            {lang === "zh" ? (
              <>
                γ 是<b>控制轨迹长度后</b> log n(x) 的回归系数。不控制长度会得到假的负值 —— 这个坑我第一版就踩了。
              </>
            ) : (
              <>
                γ is the regression coefficient on log n(x) <b>with trajectory length held fixed</b>. Skip
                the control and you get a spurious negative — the first version of this script did.
              </>
            )}
          </p>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 0 }}>
        <h4>
          {lang === "zh" ? "真实验要读的那一个标量" : "The one scalar the real experiment reports"}{" "}
          <Chip p="conjecture" />
        </h4>
        <svg className="plot" viewBox="0 0 560 78" role="img" aria-label="gamma axis">
          <line className="axis" x1={40} x2={520} y1={44} y2={44} />
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <g key={f}>
              <line className="axis" x1={40 + f * 480} x2={40 + f * 480} y1={40} y2={48} />
              <text x={40 + f * 480} y={64} textAnchor="middle">
                {f}
              </text>
            </g>
          ))}
          <rect x={40 + 0.05 * 480} y={30} width={0.9 * 480} height={28} rx={4} fill="var(--conj)" fillOpacity={0.14} />
          <circle cx={40} cy={44} r={6} fill="var(--gfn)" />
          <text x={40} y={22} textAnchor="middle" fill="var(--gfn)">
            GFN 0.000
          </text>
          <circle cx={40 + 0.9788 * 480} cy={44} r={6} fill="var(--rl)" />
          <text x={40 + 0.9788 * 480} y={22} textAnchor="middle" fill="var(--rl)">
            soft RL 0.979
          </text>
          <text x={40 + 0.5 * 480} y={44 + 4} textAnchor="middle" fill="var(--conj)">
            Disco103 ?
          </text>
        </svg>
        <p className="hint" style={{ marginTop: 4 }}>
          {lang === "zh" ? (
            <>
              两个端点是<b>本机算出来的</b>(见 <code>research/cumulants.py</code>)。中间那条琥珀带是<b>猜想</b>:
              严格小于 1,因为 y/z 的 KL 结构装得下 detailed balance;严格大于 0,因为元网络看不到父节点集合。
              若测得 γ 与 soft RL 在误差棒内无差异,这个说法即被证伪。
            </>
          ) : (
            <>
              Both endpoints are <b>computed here</b> (see <code>research/cumulants.py</code>). The amber band
              is the <b>conjecture</b>: strictly below 1 because the KL structure on y/z can express detailed
              balance, strictly above 0 because the meta-network never sees a parent set. If the measured γ is
              indistinguishable from soft RL, the claim is dead.
            </>
          )}
        </p>
      </div>
    </Rig>
  );
}
