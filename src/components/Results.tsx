import probe from "../data/disco_probe.json";
import extd from "../data/logf_head.json";
import calib from "../data/calibrate.json";
import kap from "../data/kappa.json";
import { Chip, M } from "./ui";
import { useLang } from "../i18n";

const DISCO = probe.arms[0];
const KARM = kap.arms[0];

const FIT = (readout: string, target: string) =>
  calib.fits.find((f) => f.readout === readout && f.target === target)!;
const NULLS = probe.arms.slice(1);
const nullMean = (k: keyof typeof DISCO) =>
  NULLS.reduce((s, a) => s + (a[k] as number), 0) / NULLS.length;

/**
 * What the two follow-up experiments actually returned.
 *
 * Both came back partial. The point of rendering them at all is that the
 * dossier's credibility rests on reporting the ones that did not land as
 * loudly as the ones that did.
 */
export function ProbeResults() {
  const lang = useLang();
  const zh = lang === "zh";

  return (
    <>
      <div className="scroller">
        <table className="tbl">
          <thead>
            <tr>
              <th>{zh ? "元参数" : "Meta-parameters"}</th>
              <th className="num">|β|</th>
              <th className="num">|α|</th>
              <th className="num">|β/α|</th>
              <th className="num">{zh ? "安慰剂" : "placebo"}</th>
              <th className="num">{zh ? "局部性" : "locality"}</th>
              <th className="num">{zh ? "φ 幅度" : "φ spread"}</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style={{ color: "var(--rl)" }}>Disco103</td>
              <td className="num">{DISCO.beta_absmean.toExponential(2)}</td>
              <td className="num">{DISCO.alpha_absmean.toExponential(2)}</td>
              <td className="num ok">{DISCO.beta_over_alpha.toFixed(3)}</td>
              <td className="num">{DISCO.placebo_absmean.toExponential(2)}</td>
              <td className="num ok">{DISCO.locality.toFixed(1)}×</td>
              <td className="num">{DISCO.phi_spread.toExponential(2)}</td>
            </tr>
            {NULLS.map((a) => (
              <tr key={a.arm}>
                <td style={{ color: "var(--text-3)" }}>{a.arm}</td>
                <td className="num">{a.beta_absmean.toExponential(2)}</td>
                <td className="num">{a.alpha_absmean.toExponential(2)}</td>
                <td className="num">{a.beta_over_alpha.toFixed(3)}</td>
                <td className="num">{a.placebo_absmean.toExponential(2)}</td>
                <td className="num">{a.locality.toFixed(1)}×</td>
                <td className="num">{a.phi_spread.toExponential(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="readouts" style={{ marginBottom: 22 }}>
        <div className="ro">
          <span className="k">{zh ? "β 相对随机初始化" : "β against random init"}</span>
          <span className="v" style={{ color: "var(--ver)" }}>{probe.ratio_to_null.toFixed(0)}×</span>
          <span className="n">
            {zh ? "响应是真的,不是数值噪声" : "the response is real, not numerical noise"}
          </span>
        </div>
        <div className="ro">
          <span className="k">{zh ? "局部性 vs null" : "locality against null"}</span>
          <span className="v" style={{ color: "var(--ver)" }}>
            {(DISCO.locality / nullMean("locality")).toFixed(1)}×
          </span>
          <span className="n">
            {zh ? "响应集中在当前这一步,不是后缀弥散" : "concentrated on the current step, not smeared over the suffix"}
          </span>
        </div>
        <div className="ro">
          <span className="k">|β/α|</span>
          <span className="v" style={{ color: "var(--conj)" }}>{DISCO.beta_over_alpha.toFixed(2)}</span>
          <span className="n">
            {zh ? "DB 要求 ≈1,纯 value 要求 0" : "detailed balance wants ≈1; a pure value rule wants 0"}
          </span>
        </div>
      </div>

      <div className="callout">
        <strong>{zh ? "判定:部分。" : "Verdict: partial."}</strong>{" "}
        {zh ? (
          <>
            <strong>「y 只是个 value function」被推翻</strong> —— β 是随机初始化的 {probe.ratio_to_null.toFixed(0)}{" "}
            倍,而且响应高度局部(把三步之后的策略扰动放进去,响应掉到 1/{DISCO.locality.toFixed(0)})。纯 value
            bootstrap 对所走动作的概率应当完全不敏感。
            <br />
            <br />
            <strong>但「y 实现 detailed balance」也不成立</strong> —— |β/α| ≈ {DISCO.beta_over_alpha.toFixed(2)},
            离 DB 要求的 1 差三倍。跨探针配置这个数在 0.26–0.41 之间浮动,所以它是个区间,不是常数。
          </>
        ) : (
          <>
            <strong>"y is just a value function" is refuted</strong> — β is {probe.ratio_to_null.toFixed(0)}× the
            random-init null, and the response is sharply local: move the policy three steps later instead and
            it drops by a factor of {DISCO.locality.toFixed(0)}. A pure value bootstrap must be exactly
            insensitive to the probability of the action taken.
            <br />
            <br />
            <strong>But "y implements detailed balance" does not hold either</strong> — |β/α| ≈{" "}
            {DISCO.beta_over_alpha.toFixed(2)}, a factor of three short of the 1 that DB requires. Across probe
            configurations it moved between 0.26 and 0.41, so it is a range, not a constant.
          </>
        )}
      </div>

      <p className="hint">
        {zh ? (
          <>
            <b>必须声明的边界</b>:Disco103 是在 Atari / ProcGen / DMLab 上元学习出来的,这里被喂了一个
            8×8 玩具格子,属于分布外查询。φ 用的是它自带的 <code>y_net</code>,单位是任意的 —— 所以判据取
            <b>无量纲比值 β/α</b> 而不是 β 本身。批元素通过 EMA 归一化有 O(1/B) 的串扰。
          </>
        ) : (
          <>
            <b>Boundaries that have to be stated</b>: Disco103 was meta-learned on Atari / ProcGen / DMLab and
            is being asked about an 8×8 toy grid — an off-distribution query. φ is its own{" "}
            <code>y_net</code>, whose units are arbitrary, which is why the discriminant is the{" "}
            <b>dimensionless ratio β/α</b> and not β itself. Batch elements cross-talk at O(1/B) through the
            EMA normalisers.
          </>
        )}
      </p>
    </>
  );
}

export function ExtDResults() {
  const lang = useLang();
  const zh = lang === "zh";
  const scalar = extd.arms.find((a) => a.arm === "scalar")!;
  const cat = extd.arms.find((a) => a.arm === "categorical")!;

  const W = 560;
  const H = 220;
  const PAD = { l: 52, r: 14, t: 14, b: 30 };
  const iw = W - PAD.l - PAD.r;
  const ih = H - PAD.t - PAD.b;

  const all = [...scalar.curve, ...cat.curve].filter((p) => p.kl_mean > 0);
  const lo = Math.log10(Math.min(...all.map((p) => p.kl_mean)));
  const hi = Math.log10(Math.max(...all.map((p) => p.kl_mean)));
  const maxStep = Math.max(...scalar.curve.map((p) => p.step));

  const path = (curve: typeof scalar.curve) =>
    curve
      .filter((p) => p.kl_mean > 0)
      .map((p, k) => {
        const x = PAD.l + (p.step / maxStep) * iw;
        const y = PAD.t + ih - ((Math.log10(p.kl_mean) - lo) / (hi - lo)) * ih;
        return `${k === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

  const floorY = PAD.t + ih - ((Math.log10(extd.noise_floor_kl) - lo) / (hi - lo)) * ih;

  return (
    <>
      <svg className="plot" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="log F head convergence">
        <line className="axis" x1={PAD.l} x2={W - PAD.r} y1={PAD.t + ih} y2={PAD.t + ih} />
        <line className="axis" x1={PAD.l} x2={PAD.l} y1={PAD.t} y2={PAD.t + ih} />
        <line
          className="grid"
          x1={PAD.l}
          x2={W - PAD.r}
          y1={floorY}
          y2={floorY}
          stroke="var(--conj)"
          strokeDasharray="4 4"
        />
        <text x={W - PAD.r} y={floorY - 5} textAnchor="end" fill="var(--conj)">
          {zh ? "噪声底" : "noise floor"}
        </text>
        <path d={path(scalar.curve)} fill="none" stroke="var(--rl)" strokeWidth={2} />
        <path d={path(cat.curve)} fill="none" stroke="var(--gfn)" strokeWidth={2} />
        <text x={PAD.l} y={H - 8}>0</text>
        <text x={W - PAD.r} y={H - 8} textAnchor="end">
          {maxStep.toLocaleString("en-US")}
        </text>
        <text x={PAD.l - 8} y={PAD.t + 4} textAnchor="end">
          {`1e${Math.round(hi)}`}
        </text>
        <text x={PAD.l - 8} y={PAD.t + ih} textAnchor="end">
          {`1e${Math.round(lo)}`}
        </text>
        <text x={14} y={PAD.t + ih / 2} className="lbl" transform={`rotate(-90 14 ${PAD.t + ih / 2})`}>
          KL(p ‖ R/Z)
        </text>
      </svg>
      <div className="legend">
        <span>
          <i style={{ background: "var(--rl)" }} />
          {zh ? "标量 log F 回归" : "scalar log F regression"}
        </span>
        <span>
          <i style={{ background: "var(--gfn)" }} />
          {zh ? "categorical two-hot 头" : "categorical two-hot head"}
        </span>
      </div>

      <div className="readouts" style={{ marginTop: 20 }}>
        <div className="ro">
          <span className="k">{zh ? "领先检查点" : "checkpoints led"}</span>
          <span className="v" style={{ color: "var(--gfn)" }}>
            {extd.checkpoints_won_categorical}/{extd.checkpoints_total}
          </span>
          <span className="n">{zh ? "categorical 全程领先" : "categorical, over the whole run"}</span>
        </div>
        <div className="ro">
          <span className="k">{zh ? "KL 几何平均比" : "geometric-mean KL ratio"}</span>
          <span className="v" style={{ color: "var(--gfn)" }}>{extd.geomean_kl_ratio.toFixed(2)}×</span>
          <span className="n">
            {zh
              ? `仅噪声底之上的 ${extd.geomean_n_checkpoints} 个点`
              : `over the ${extd.geomean_n_checkpoints} checkpoints above the floor only`}
          </span>
        </div>
        <div className="ro">
          <span className="k">{zh ? "梯度范数 p99" : "gradient norm p99"}</span>
          <span className="v" style={{ color: "var(--gfn)" }}>
            {cat.grad_norm_p99.toFixed(2)}
          </span>
          <span className="n">
            {zh ? `标量头 ${scalar.grad_norm_p99.toFixed(2)}` : `scalar head ${scalar.grad_norm_p99.toFixed(2)}`}
          </span>
        </div>
        <div className="ro">
          <span className="k">{zh ? "末点 KL" : "final KL"}</span>
          <span className="v" style={{ color: "var(--conj)" }}>
            {zh ? "噪声底内" : "in the floor"}
          </span>
          <span className="n">
            {zh
              ? `标量 ${scalar.final_kl_mean.toExponential(1)} vs categorical ${cat.final_kl_mean.toExponential(1)} — 不构成判定`
              : `scalar ${scalar.final_kl_mean.toExponential(1)} vs categorical ${cat.final_kl_mean.toExponential(1)} — not a verdict`}
          </span>
        </div>
      </div>

      <div className="callout">
        <strong>{zh ? "判定:部分支持。" : "Verdict: partially supported."}</strong>{" "}
        {zh ? (
          <>
            categorical 头<strong>收敛更快、梯度尾更紧</strong>(p99 {cat.grad_norm_p99.toFixed(2)} vs{" "}
            {scalar.grad_norm_p99.toFixed(2)},这正是 conditioning 的论断),但<strong>渐近值并不更好</strong>:
            两条臂最后都掉进 on-policy 噪声底,末点顺序是抛硬币。
            <br />
            <br />
            而且这不是强版本的决定性检验 —— 这个格子的 log F 只跨 <strong>5.5 nats</strong>,不是拓展 D
            所设想的"几十个数量级"。
          </>
        ) : (
          <>
            The categorical head <strong>converges faster with a tighter gradient tail</strong> (p99{" "}
            {cat.grad_norm_p99.toFixed(2)} vs {scalar.grad_norm_p99.toFixed(2)}, which is precisely the
            conditioning claim), but it does <strong>not reach a better asymptote</strong>: both arms bottom out
            in the on-policy noise floor, where the final ordering is a coin flip.
            <br />
            <br />
            Nor is this a decisive test of the strong version — log F on this grid spans only{" "}
            <strong>5.5 nats</strong>, not the tens of orders of magnitude that motivate extension D.
          </>
        )}
      </div>
    </>
  );
}

export function CorrectionNote() {
  const lang = useLang();
  const zh = lang === "zh";
  return (
    <div className="card" style={{ borderColor: "#ff7a5c44" }}>
      <h4>
        {zh ? "更正:第一版提的 γ 实验是错的" : "Correction: the γ experiment as first proposed was wrong"}{" "}
        <Chip p="mine" />
      </h4>
      <div style={{ color: "var(--text-2)", fontSize: 15.5 }}>
        {zh ? (
          <>
            <p style={{ marginTop: 0 }}>
              原方案是训练一个 Disco103 agent,再拟合 <M tex={String.raw`p(x)\propto R(x)n(x)^\gamma`} />。
              动手写 harness 时才发现它不成立:<strong>DiscoRL 最大化回报</strong>,在 hypergrid 上最优策略是
              确定性的 argmax R,<M tex="p(x)" /> 塌成点质量,γ 根本没有定义。在收敛前测,量到的是
              「还没收敛」而不是「结构上无偏」。
            </p>
            <p style={{ marginBottom: 0 }}>
              γ 仍然有效 —— 但只对<strong>本来就是采样器</strong>的两者(GFlowNet 与 MaxEnt RL),那部分已经
              精确算完。对 DiscoRL,正确的问法是问它的<strong>更新规则</strong>,不是问它收敛到的分布。这就是
              上面的 β 探针,而且完全不需要训练它到收敛。
            </p>
          </>
        ) : (
          <>
            <p style={{ marginTop: 0 }}>
              The original plan was to train a Disco103 agent and fit{" "}
              <M tex={String.raw`p(x)\propto R(x)n(x)^\gamma`} />. Writing the harness is what exposed the flaw:{" "}
              <strong>DiscoRL maximises return</strong>, so on a hypergrid its optimum is a deterministic argmax
              R, <M tex="p(x)" /> degenerates to a point mass, and γ is undefined. Measuring before convergence
              measures "has not converged", not "structurally unbiased".
            </p>
            <p style={{ marginBottom: 0 }}>
              γ survives — but only for the two things that <strong>are</strong> samplers, GFlowNets and MaxEnt
              RL, and that part is already computed exactly. For DiscoRL the well-posed question is about its{" "}
              <strong>update rule</strong>, not the distribution it converges to. That is the β probe above, and
              it needs no convergence at all.
            </p>
          </>
        )}
      </div>
    </div>
  );
}

/**
 * The calibration that decides whether the beta ratio meant anything.
 *
 * beta/alpha is only a "fraction of detailed balance" if phi(y) tracks a flow.
 * On this environment it does not track anything, while the positive control --
 * the categorical q head, a value head by construction -- tracks the on-policy
 * value cleanly. So the machinery works and the y channel genuinely carries
 * neither quantity, which retires the ratio's interpretation.
 */
export function CalibrationResults() {
  const lang = useLang();
  const zh = lang === "zh";
  const rows: [string, string, string][] = [
    ["q head", "V_pi", zh ? "阳性对照" : "positive control"],
    ["phi(z)", "V_pi", ""],
    ["phi(y)", "V_pi", ""],
    ["phi(y)", "log_F", zh ? "流假设" : "the flow hypothesis"],
  ];
  return (
    <>
      <div className="scroller">
        <table className="tbl">
          <thead>
            <tr>
              <th>{zh ? "读出 ~ 目标" : "readout ~ target"}</th>
              <th className="num">R²</th>
              <th className="num">Spearman</th>
              <th>{zh ? "角色" : "role"}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([r, t, note]) => {
              const f = FIT(r, t);
              const strong = Math.abs(f.spearman) > 0.6;
              return (
                <tr key={`${r}~${t}`}>
                  <td style={{ color: t === "log_F" && r === "phi(y)" ? "var(--rl)" : undefined }}>
                    {r} ~ {t}
                  </td>
                  <td className={`num ${strong ? "ok" : ""}`}>{f.r2.toFixed(3)}</td>
                  <td className={`num ${strong ? "ok" : ""}`}>{f.spearman.toFixed(3)}</td>
                  <td style={{ color: "var(--text-3)", fontSize: 13 }}>{note}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="callout">
        <strong>{zh ? "撤回。" : "Withdrawn."}</strong>{" "}
        {zh ? (
          <>
            阳性对照通过 —— categorical q 头按构造就是 value 头,它以 R² = {FIT("q head", "V_pi").r2.toFixed(2)}、
            ρ = {FIT("q head", "V_pi").spearman.toFixed(2)} 追踪 on-policy value,所以分桶、训练和 DP 都正常。
            在这个前提下,<strong>φ(y) 对 log F 的 R² 只有 {FIT("phi(y)", "log_F").r2.toFixed(3)},秩相关{" "}
            {FIT("phi(y)", "log_F").spearman.toFixed(2)} —— 是零</strong>。
            <br />
            <br />
            所以 |β/α| 不能读作「离 detailed balance 还差多少」:那个读法预设了 φ(y) 是个 log-flow 量,
            而它不是。β 本身仍然是关于更新映射的事实(139× null、23× 局部性),但它的 DB 解释<strong>已撤回</strong>。
            <br />
            <br />
            附带发现:value 语义在 <strong>z</strong> 通道里,不在 y —— φ(z) 与 V<sub>π</sub> 的秩相关是{" "}
            {FIT("phi(z)", "V_pi").spearman.toFixed(2)}。
          </>
        ) : (
          <>
            The positive control passes: the categorical q head is a value head by construction and tracks the
            on-policy value at R² = {FIT("q head", "V_pi").r2.toFixed(2)}, ρ ={" "}
            {FIT("q head", "V_pi").spearman.toFixed(2)}, so the bucketing, the training and the DP all work.
            Against that, <strong>φ(y) explains R² = {FIT("phi(y)", "log_F").r2.toFixed(3)} of log F with rank
            correlation {FIT("phi(y)", "log_F").spearman.toFixed(2)} — nothing</strong>.
            <br />
            <br />
            So |β/α| cannot be read as "how far from detailed balance": that reading presumed φ(y) is a
            log-flow quantity, and it is not. β itself remains a fact about the update map (139× the null, 23×
            locality), but its detailed-balance interpretation is <strong>withdrawn</strong>.
            <br />
            <br />
            Incidental finding: the value semantics live in the <strong>z</strong> channel, not y — φ(z)
            against V<sub>π</sub> has rank correlation {FIT("phi(z)", "V_pi").spearman.toFixed(2)}.
          </>
        )}
      </div>
      <p className="hint">
        {zh
          ? "一个环境、400 步训练、分布外查询。这不排除 φ(y) 在 Disco103 真正元训练过的域里追踪某个流量;它排除的是「在可解析算出 log F 的地方,y 追踪它」。"
          : "One environment, 400 training steps, an off-distribution query. This does not rule out φ(y) tracking a flow in the domains Disco103 was actually meta-trained on. It rules out y tracking log F where log F can be computed exactly."}
      </p>
    </>
  );
}


/**
 * The structural answer, and the end of the mapping question.
 *
 * kappa = ||sym B|| / ||B|| for the bootstrap operator B. A true gradient gives
 * 1; any causal bootstrap gives exactly 1/sqrt(2), because a target that reads
 * only the future is strictly triangular, so tr(B^2) = 0 and the symmetric and
 * antisymmetric parts have equal norm. Disco103 sits on that floor.
 */
export function KappaResults() {
  const lang = useLang();
  const zh = lang === "zh";
  const floor = kap.causal_floor;
  return (
    <>
      <div className="readouts" style={{ marginBottom: 22 }}>
        <div className="ro">
          <span className="k">κ(bootstrap)</span>
          <span className="v" style={{ color: "var(--rl)" }}>{KARM.kappa_bootstrap.toFixed(4)}</span>
          <span className="n">± {KARM.se.toFixed(4)} · dim {KARM.dim.toLocaleString("en-US")}</span>
        </div>
        <div className="ro">
          <span className="k">{zh ? "因果地板 1/√2" : "causal floor 1/√2"}</span>
          <span className="v" style={{ color: "var(--gfn)" }}>{floor.toFixed(4)}</span>
          <span className="n">
            {zh ? `偏离 ${(KARM.kappa_bootstrap - floor >= 0 ? "+" : "")}${(KARM.kappa_bootstrap - floor).toFixed(4)}` : `off by ${(KARM.kappa_bootstrap - floor).toFixed(4)}`}
          </span>
        </div>
        <div className="ro locked">
          <span className="k">{zh ? "可梯度化份额 κ²" : "gradient share κ²"}</span>
          <span className="v">{(KARM.kappa_bootstrap ** 2).toFixed(3)}</span>
          <span className="n">{zh ? "算子层面，非行为层面" : "operator-level, not behavioural"}</span>
        </div>
      </div>

      <div className="scroller">
        <table className="tbl">
          <thead>
            <tr>
              <th>{zh ? "估计量校准" : "estimator calibration"}</th>
              <th className="num">{zh ? "解析值" : "exact"}</th>
              <th className="num">{zh ? "测得" : "measured"}</th>
              <th className="num">|err|</th>
            </tr>
          </thead>
          <tbody>
            {kap.synthetic.map((r) => (
              <tr key={`s${r.c}`}>
                <td>{zh ? "合成场 c = " : "synthetic c = "}{r.c}</td>
                <td className="num">{r.exact.toFixed(4)}</td>
                <td className="num">{r.est.toFixed(4)}</td>
                <td className="num">{r.err.toFixed(4)}</td>
              </tr>
            ))}
            {kap.causal_calibration.map((r) => (
              <tr key={`c${r.diag_scale}`}>
                <td style={{ color: "var(--gfn)" }}>
                  {zh ? "因果自举 + 对角 " : "causal bootstrap + diag "}
                  {r.diag_scale}
                </td>
                <td className="num">{r.exact.toFixed(4)}</td>
                <td className="num">{r.est.toFixed(4)}</td>
                <td className="num">{r.err.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="callout ok">
        <strong>{zh ? "结论。" : "The answer."}</strong>{" "}
        {zh ? (
          <>
            <strong>κ = {KARM.kappa_bootstrap.toFixed(4)} ≠ 1,所以 DiscoRL 的更新不是任何泛函的梯度。</strong>{" "}
            「它极小化 W 的哪个累积量」这个问题是<strong>病态的</strong> —— 没有那个泛函。
            <br />
            <br />
            但它也<strong>没有</strong>比因果性所强制的更不保守。任何「目标只读未来」的自举算子严格三角,
            tr(B²) = 0,于是 κ = 1/√2 恰好。测得偏离 +{(KARM.kappa_bootstrap - floor).toFixed(4)},
            <strong>小于估计量在该邻域已证实的系统偏差(0.0043)</strong>。所以非保守性不是元学习发现的东西,
            是自举的通用代价。
            <br />
            <br />
            <strong>部分 mapping 因此被量化了:Hodge 投影 Π_sym 保住自举<strong>算子</strong>的 Frobenius
            质量的 κ² = {(KARM.kappa_bootstrap ** 2).toFixed(3)}。</strong>
            <br />
            <br />
            这是关于<strong>算子</strong>的陈述,不是关于<strong>算法行为</strong>的。Frobenius 质量的一半可写成
            梯度,并不等于「DiscoRL 有一半是 GFlowNet」—— 后者需要把算子层面的劈分连到样本效率或解的质量上,
            而我们没有做这一步。
          </>
        ) : (
          <>
            <strong>κ = {KARM.kappa_bootstrap.toFixed(4)} ≠ 1, so DiscoRL's update is not the gradient of
            any functional.</strong> "Which cumulant of W does it minimise" is <strong>malformed</strong> —
            there is no such functional.
            <br />
            <br />
            But it is <strong>no less</strong> conservative than causality forces. Any bootstrap whose target
            reads only the future is strictly triangular, so tr(B²) = 0 and κ = 1/√2 exactly. The measured
            excess of +{(KARM.kappa_bootstrap - floor).toFixed(4)} is{" "}
            <strong>smaller than the estimator's demonstrated bias in that band (0.0043)</strong>. The
            non-conservativity is not something meta-learning discovered; it is the generic price of
            bootstrapping.
            <br />
            <br />
            <strong>So the partial mapping is quantified: the Hodge projection Π_sym retains κ² ={" "}
            {(KARM.kappa_bootstrap ** 2).toFixed(3)} of the bootstrap <strong>operator's</strong> Frobenius
            mass.</strong>
            <br />
            <br />
            That is a statement about the <strong>operator</strong>, not about <strong>algorithmic
            behaviour</strong>. Half the Frobenius mass being expressible as a gradient does not mean "half of
            DiscoRL is a GFlowNet"; that would require connecting the operator-level split to sample
            efficiency or solution quality, which we have not done.
          </>
        )}
      </div>
    </>
  );
}
