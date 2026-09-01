import { useEffect, useState } from "react";

import GammaLab from "./components/GammaLab";
import SourceMatrix from "./components/SourceMatrix";
import {
  CalibrationResults,
  CorrectionNote,
  ExtDResults,
  KappaResults,
  ProbeResults,
} from "./components/Results";
import WorkDial from "./components/WorkDial";
import { Chip, M, Section, T } from "./components/ui";
import {
  ARXIV_AUDIT,
  CUMULANTS,
  DB_COMPARE,
  DISCO_COMMIT,
  DISCO_REPO,
  EXTS,
  FAMILY_MEANS,
  GAP_SEEDS,
  GFN_ROW,
  HOSTILE_ARMS,
  LEDGER,
  LIMITS,
  METAOBJ_KL,
  NAV,
  NEW_SCRIPTS,
  REDUCTION_CHECKS,
  REFS,
  REMARK3,
  RESIDUAL_SCALE,
  RL_ROW,
} from "./data/facts";
import calibJson from "./data/calibrate.json";
import type { Lang } from "./i18n";

const CALIB_FLOW = calibJson.phi_y_logF;

const REPO_URL = "https://github.com/danielchen26/discorl-gfn-bridge";

export default function App({ lang, setLang }: { lang: Lang; setLang: (l: Lang) => void }) {
  const [active, setActive] = useState(NAV[0].id);

  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => {
        for (const e of entries) if (e.isIntersecting) setActive(e.target.id);
      },
      { rootMargin: "-20% 0px -70% 0px" },
    );
    for (const n of NAV) {
      const el = document.getElementById(n.id);
      if (el) obs.observe(el);
    }
    return () => obs.disconnect();
  }, []);

  const zh = lang === "zh";

  return (
    <div className="shell">
      <aside className="rail">
        <div className="rail-brand">
          W(τ)
          <span>DiscoRL ↔ GFlowNet</span>
        </div>
        <div className="langtog">
          <button className={zh ? "on" : ""} onClick={() => setLang("zh")}>
            中文
          </button>
          <button className={!zh ? "on" : ""} onClick={() => setLang("en")}>
            EN
          </button>
        </div>
        <nav>
          {NAV.map((n) => (
            <a key={n.id} href={`#${n.id}`} className={active === n.id ? "on" : ""}>
              <b>{n.num}</b>
              <span>{n.label[lang]}</span>
            </a>
          ))}
        </nav>
        <div className="rail-foot">
          {zh ? "全部数字由 research/ 下的脚本产生" : "every number comes from a script in research/"}
          <br />
          <a href={`${import.meta.env.BASE_URL}kappa.pdf`} target="_blank" rel="noreferrer">
            paper (pdf) ↗
          </a>
          <br />
          <a href={REPO_URL} target="_blank" rel="noreferrer">
            source ↗
          </a>
        </div>
      </aside>

      <main>
        {/* ------------------------------------------------------------ hero */}
        <header className="hero">
          <p className="eyebrow">{zh ? "一份可验证的推导笔记" : "A verifiable derivation notebook"}</p>
          <h1>
            {zh ? (
              <>
                它们最小化的是<em> 同一个 </em>随机变量的不同累积量
              </>
            ) : (
              <>
                They minimise different cumulants of <em>the same</em> random variable
              </>
            )}
          </h1>
          <p className="hero-lede">
            {zh ? (
              <>
                DiscoRL 是元学习出来的更新规则,GFlowNet 是手工设计的采样器。表面上毫无关系。但把两者都写到
                「轨迹的非平衡功 <strong>W(τ)</strong>」这一层,变分推断、GFlowNet 和 DiscoRL
                就落在同一条累积量阶梯的不同台阶上。
              </>
            ) : (
              <>
                DiscoRL is a meta-learned update rule; a GFlowNet is a hand-designed sampler. They look
                unrelated. Written in terms of the non-equilibrium work of a trajectory,{" "}
                <strong>W(τ)</strong>, variational inference, GFlowNets and DiscoRL turn out to sit on
                different rungs of one cumulant ladder.
              </>
            )}
          </p>

          <div className="hero-eq">
            <M
              block
              tex={String.raw`W(\tau)\;=\;\log\frac{\prod_t p_F(s_{t+1}\mid s_t)}{R(x)\,\prod_t p_B(s_t\mid s_{t+1})}\;=\;\log\frac{P_F(\tau)}{Z\,P_B(\tau)}`}
            />
          </div>

          <div className="kpis">
            <div className="kpi ver">
              <span className="v">{GFN_ROW.jarzynski_rel_err.toExponential(1)}</span>
              <span className="k">
                {zh ? "E[e^−W] 与 Z 的相对误差,任意策略" : "relative error of E[e^−W] against Z, any policy"}
              </span>
            </div>
            <div className="kpi gfn">
              <span className="v">|{Math.abs(GFN_ROW.var_W).toExponential(1)}|</span>
              <span className="k">
                {zh
                  ? "流匹配下的 Var[W] — 理论值恰为 0,这是浮点噪声"
                  : "Var[W] under flow matching — exactly 0 in theory; this is float noise"}
              </span>
            </div>
            <div className="kpi rl">
              <span className="v">{RL_ROW.gamma.toFixed(3)}</span>
              <span className="k">{zh ? "软 RL 的多路径指数 γ" : "the multi-path exponent γ of soft RL"}</span>
            </div>
            <div className="kpi conj">
              <span className="v">{CALIB_FLOW.r2.toFixed(3)}</span>
              <span className="k">
                {zh
                  ? "φ(y) 对精确 log F 的 R² — 流假设在这里是零"
                  : "R² of φ(y) against the exact log F — the flow hypothesis reads as nothing here"}
              </span>
            </div>
          </div>
        </header>

        {/* ------------------------------------------------------------- 01 */}
        <Section
          id="why"
          num="01"
          title={{ zh: "为什么会有人问这个问题", en: "Why anyone would ask" }}
          kicker={{
            zh: "两条线各自成熟,但它们对「什么是好的学习信号」给出了同一个答案 —— 而其中一条完全不知道另一条存在。",
            en: "Two mature lines of work give the same answer to 'what is a good learning signal' — and one of them had never heard of the other.",
          }}
        >
          <div className="body">
            <h3>{zh ? "DiscoRL:让损失函数自己被学出来" : "DiscoRL: let the loss function be learned"}</h3>
            <p>
              {zh ? (
                <>
                  传统 RL 的更新规则是人写的方程。DiscoRL 把它换成一个<strong>元网络</strong>:元网络读一段轨迹,
                  输出策略目标 π̂ 和两个<strong>没有预定义语义</strong>的预测目标 ŷ、ẑ,agent 再去匹配它们。
                  元网络本身由上百个并行 agent 的表现,通过元梯度训练出来。
                </>
              ) : (
                <>
                  A classical RL update rule is an equation a human wrote. DiscoRL replaces it with a{" "}
                  <strong>meta-network</strong> that reads a trajectory and emits a policy target π̂ plus two
                  prediction targets ŷ, ẑ with <strong>no predefined semantics</strong>; the agent then matches
                  them. The meta-network is trained by meta-gradient from the performance of hundreds of parallel
                  agents.
                </>
              )}
            </p>
            <M
              block
              tex={String.raw`\mathcal L_\theta=c_\pi\,\mathrm{KL}(\hat\pi\Vert\pi_\theta)+c_y\,\mathrm{KL}(\hat y\Vert y_\theta)+c_z\,\mathrm{KL}(\hat z\Vert z_\theta(\cdot,a))+c_{\mathrm{aux}}\,\mathrm{KL}(\cdot)`}
            />
            <p>
              {zh ? (
                <>
                  官方分析报告了一件耐人寻味的事:学到的预测会在<strong>大奖励事件之前</strong>激活,并且编码
                  <strong>未来策略熵</strong>。一个纯粹以回报为目标的搜索过程,自发漂到了「值 + 未来熵」的混合语义上。
                </>
              ) : (
                <>
                  The published analysis reports something suggestive: the discovered predictions fire{" "}
                  <strong>ahead of large-reward events</strong> and encode <strong>future policy entropy</strong>.
                  A search driven purely by return drifted into a "value plus future entropy" semantics on its
                  own.
                </>
              )}
            </p>

            <h3>{zh ? "GFlowNet:按奖励比例采样,而不是最大化奖励" : "GFlowNets: sample in proportion to reward, don't maximise it"}</h3>
            <p>
              {zh ? (
                <>
                  GFlowNet 要的不是最优解,而是让终态 x 以 p(x) ∝ R(x) 被抽到。做法是在 DAG
                  上施加<strong>守恒律</strong>。而 soft-MDP 视角下,GFN 的 state flow 正是
                </>
              ) : (
                <>
                  A GFlowNet does not want the optimum; it wants terminal objects drawn with p(x) ∝ R(x), and it
                  gets there by imposing a <strong>conservation law</strong> on a DAG. Seen as a soft MDP, its
                  state flow is
                </>
              )}
            </p>
            <M
              block
              tex={String.raw`\log F(s)=\text{soft-}V(s)=\log\!\!\sum_{\tau:\,s\to x}\!\exp\big(R(x)\big)\prod p_B`}
            />
            <p>
              {zh ? (
                <>
                  <strong>一个量同时携带未来奖励量级和未来路径熵。</strong>{" "}
                  这正是 DiscoRL 自发学到的语义。所以问题不是「它们像不像」,而是:
                  <strong>它们是不是同一个东西的两个投影?</strong>
                </>
              ) : (
                <>
                  <strong>One quantity carrying both future reward magnitude and future path entropy.</strong>{" "}
                  Which is precisely the semantics DiscoRL discovered by itself. So the question is not whether
                  they resemble each other, but whether{" "}
                  <strong>they are two projections of the same object.</strong>
                </>
              )}
            </p>
          </div>
        </Section>

        {/* ------------------------------------------------------------- 02 */}
        <Section
          id="work"
          num="02"
          title={{ zh: "同一个数学对象:非平衡功 W(τ)", en: "One mathematical object: the work W(τ)" }}
          kicker={{
            zh: "这一节没有类比。三条都是精确等式,并且已经在本机跑过。",
            en: "No analogies in this section. Three exact identities, all executed on this machine.",
          }}
        >
          <div className="body">
            <p>
              {zh ? (
                <>
                  设 DAG 上源点 s₀、终态 x、奖励 R(x) &gt; 0、Z = Σ R(x)。前向策略给出路径测度 P_F,反向策略 p_B
                  给出参考测度。功定义为顶部那个式子。于是:
                </>
              ) : (
                <>
                  Take a DAG with source s₀, terminal objects x, reward R(x) &gt; 0 and Z = Σ R(x). The forward
                  policy gives a path measure P_F; the backward policy p_B gives the reference measure. With W
                  defined as at the top of this page:
                </>
              )}
            </p>

            <div className="card">
              <h4>
                {zh ? "① Jarzynski 等式 —— 对任意前向策略成立" : "① The Jarzynski identity — for any forward policy"}{" "}
                <Chip p="verified" />
              </h4>
              <M
                block
                tex={String.raw`\mathbb E_{P_F}\big[e^{-W}\big]=\sum_\tau P_F(\tau)\frac{R(x)\prod p_B}{P_F(\tau)}=\sum_x R(x)=Z`}
              />
              <p style={{ margin: 0, fontSize: 14.5, color: "var(--text-2)" }}>
                {zh ? (
                  <>
                    注意它<strong>不依赖策略好坏</strong>。本机在三个截然不同的策略上都得到相对误差 ≤{" "}
                    {GFN_ROW.jarzynski_rel_err.toExponential(1)} —— 这是恒等式,不是拟合。
                  </>
                ) : (
                  <>
                    Note it <strong>does not depend on the policy being any good</strong>. Three very different
                    policies all give relative error ≤ {GFN_ROW.jarzynski_rel_err.toExponential(1)} here — an
                    identity, not a fit.
                  </>
                )}
              </p>
            </div>

            <div className="card">
              <h4>
                {zh ? "② 第二定律 = ELBO 缺口" : "② The second law = the ELBO gap"} <Chip p="published" />
              </h4>
              <M block tex={String.raw`\langle W_{\rm diss}\rangle=\mathbb E[W]+\log Z=D_{\rm KL}\big(P_F\,\Vert\,P_B\big)\;\ge\;0`} />
              <p style={{ margin: 0, fontSize: 14.5, color: "var(--text-2)" }}>
                {zh
                  ? "这正是 Kawai–Parrondo–Van den Broeck 的「耗散 = 相对熵」。变分推断做的 max ELBO,就是 min ⟨W⟩ —— 一阶累积量。"
                  : "This is Kawai–Parrondo–Van den Broeck's dissipation-equals-relative-entropy. Maximising the ELBO is minimising ⟨W⟩ — the first cumulant."}
              </p>
              <div className="callout" style={{ margin: "16px 0 0" }}>
                <strong>{zh ? "别把它读成平衡态。" : "Do not read this as equilibrium."}</strong>{" "}
                {zh ? (
                  <>
                    GFlowNet 是<strong>非平衡稳态</strong>:flow matching 就是 Kirchhoff 电流律,有源有汇的
                    DAG 上每条边的净电流恒非零,平衡态要求处处净流为零。GFN 的「detailed balance」说的不是净流为零,
                    而是 p_B 恰好是 p_F 关于测度 F 的<strong>对偶过程</strong>。
                    <br />
                    <br />
                    所以 W <strong>具有 excess(非绝热)熵产的结构</strong>,而不是平衡功;
                    <M tex={String.raw`\mathrm{Var}[W]=0`} /> 是<strong>零超额耗散</strong>,而不是零耗散 ——
                    housekeeping 电流(Z 从源流到汇)按构造恒非零,根本不进入平衡残差。
                    <br />
                    <br />
                    <strong>但这个身份只在最优处成立。</strong>Hatano–Sasa 的 excess/housekeeping 劈分要求
                    p_B 是稳态流的<strong>真</strong>对偶;GFN 的 p_B 是<strong>选</strong>的。两者只在平衡条件
                    被满足时重合。离开最优,W 只是平衡残差,热力学读法不适用 —— 我们没有在这个设定下证明
                    Hatano–Sasa 等式,只指出了结构同形。
                    <br />
                    <br />
                    这顺带解释了「GFN 结果依赖 p_B 的选取」:excess 与 housekeeping 的劈分本来就依赖于对偶动力学的选择。
                  </>
                ) : (
                  <>
                    A GFlowNet is a <strong>non-equilibrium steady state</strong>. Flow matching is Kirchhoff's
                    current law, and every edge of a DAG with a source and sinks carries net current, whereas
                    equilibrium demands that all net currents vanish. GFN's "detailed balance" does not say the
                    current is zero; it says p_B is exactly the <strong>dual</strong> of p_F with respect to F.
                    <br />
                    <br />
                    So W <strong>has the structure of an excess, non-adiabatic entropy production</strong>
                    rather than equilibrium work, and <M tex={String.raw`\mathrm{Var}[W]=0`} /> means{" "}
                    <strong>zero excess dissipation</strong> — the housekeeping current carrying Z from source
                    to sinks is nonzero by construction and never enters the balance residual.
                    <br />
                    <br />
                    <strong>That identification holds only at the optimum.</strong> The Hatano–Sasa split needs
                    p_B to be the <strong>true</strong> dual of the stationary flow; a GFlowNet's p_B is{" "}
                    <strong>chosen</strong>. The two coincide exactly when the balance condition holds. Away
                    from it W is just a balance residual and the thermodynamic reading does not apply — we have
                    not proved a Hatano–Sasa equality in this setting, only pointed at the shared structure.
                    <br />
                    <br />
                    That also explains why GFN results depend on the choice of p_B: the excess/housekeeping
                    split is defined only relative to a choice of dual dynamics.
                  </>
                )}
              </div>
            </div>

            <div className="card">
              <h4>
                {zh ? "③ Trajectory Balance = 二阶累积量" : "③ Trajectory balance = the second cumulant"}{" "}
                <Chip p="verified" />
              </h4>
              <M
                block
                tex={String.raw`\mathcal L_{\rm TB}(\tau)=\Big(\log\tfrac{Z_\theta\prod p_F}{R(x)\prod p_B}\Big)^2=\big(\log Z_\theta+W(\tau)\big)^2`}
              />
              <p style={{ fontSize: 14.5, color: "var(--text-2)" }}>
                {zh ? (
                  <>
                    对自由参数 log Z<sub>θ</sub> 取极小,最优点 <strong>= −E[W] = ELBO</strong>,残差{" "}
                    <strong>= Var[W]</strong>。本机直接暴力搜出 argmin ={" "}
                    {CUMULANTS.c3_argmin.toFixed(5)},与 −E[W] 一致。
                  </>
                ) : (
                  <>
                    Minimising over the free parameter log Z<sub>θ</sub> puts the optimum at{" "}
                    <strong>−E[W] = the ELBO</strong> and the residual at <strong>Var[W]</strong>. Brute-forcing
                    it here gives argmin = {CUMULANTS.c3_argmin.toFixed(5)}, matching −E[W].
                  </>
                )}
              </p>
              <M
                block
                tex={String.raw`\log Z=\log\mathbb E\big[e^{-W}\big]=-\langle W\rangle+\tfrac12\mathrm{Var}(W)-\tfrac16\kappa_3+\cdots`}
              />
              <p style={{ margin: 0, fontSize: 14.5, color: "var(--text-2)" }}>
                {zh
                  ? "VI 截到一阶,所以永远差一个 KL。TB 把二阶压到零,截断因此变成精确 —— 零涨落即可逆,可逆即精确采样器。"
                  : "VI truncates at first order, so a KL always remains. TB drives the second to zero, which makes the truncation exact — zero fluctuation is reversibility, and reversibility is an exact sampler."}
              </p>
            </div>

            <div className="scroller">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>{zh ? "方法" : "Method"}</th>
                    <th>{zh ? "对 W 做什么" : "What it does to W"}</th>
                    <th>{zh ? "物理" : "Physics"}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>VI / MaxEnt RL</td>
                    <td>
                      min <M tex={String.raw`\langle W\rangle`} /> — {zh ? "一阶累积量" : "first cumulant"}
                    </td>
                    <td>{zh ? "最小平均耗散" : "least mean dissipation"}</td>
                  </tr>
                  <tr>
                    <td style={{ color: "var(--gfn)" }}>GFlowNet (TB)</td>
                    <td>
                      min <M tex={String.raw`\mathrm{Var}(W)`} /> —{" "}
                      {zh ? "二阶,一阶自动等于 ELBO" : "second; the first lands on the ELBO by itself"}
                    </td>
                    <td>{zh ? "逼近准静态可逆" : "approach quasi-static reversibility"}</td>
                  </tr>
                  <tr>
                    <td style={{ color: "var(--rl)" }}>DiscoRL</td>
                    <td>{zh ? "元学习这个泛函本身" : "meta-learns the functional itself"}</td>
                    <td>{zh ? "协议由回报选出来" : "the protocol is chosen by return"}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <h3>{zh ? "这个框架有人写过吗" : "Has this framing been written down?"}</h3>
            <p>
              {zh
                ? "三块拼图都已发表,但拼在一起的说法我没找到。arXiv 检索(2026-08-28,all: 覆盖标题/摘要/comments,不含全文):"
                : "All three pieces are published; the assembled statement I could not find. arXiv counts (2026-08-28; all: covers title/abstract/comments, not full text):"}
            </p>
            <div className="scroller">
              <table className="tbl">
                <thead>
                  <tr>
                    <th style={{ width: "70%" }}>{zh ? "检索式" : "Query"}</th>
                    <th>{zh ? "命中" : "Records"}</th>
                  </tr>
                </thead>
                <tbody>
                  {ARXIV_AUDIT.map((a) => (
                    <tr key={a.q}>
                      <td className="num">{a.q}</td>
                      <td className={`num ${a.n === 0 ? "no" : ""}`}>{a.n}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="hint">
              {zh
                ? "第一行是对照组,证明检索式本身有效。但这个方法本身的可靠性,后来没通过检验 —— 见下。"
                : "The first row is the control, proving the query works at all. The reliability of the method itself later failed a test — see below."}
            </p>
            <div className="callout" style={{ borderLeftColor: "var(--rl)" }}>
              <strong>{zh ? "这个新颖性检查方法不可靠。" : "This novelty check is not reliable."}</strong>{" "}
              {zh ? (
                <>
                  我们后来用同样的方法判定「GFlowNet 的流空间由圈空间张成」未被发表,并据此推了一遍。
                  <strong>它是发表过的</strong> —— Brunswic et al. AAAI 2024 的 Prop. 4 / Thm. 5,写成{" "}
                  <M tex={String.raw`\mathcal F_R=F+H^1(G)`} />,更强形式可回溯到 Kalpazidou 2007。
                  摘要检索找不到它,因为那篇没有用这些字面。
                  <br />
                  <br />
                  所以上表只说明<strong>这些字面组合未被索引</strong>,不说明想法是新的。真正的新颖性判断必须读正文。
                </>
              ) : (
                <>
                  We later used the same method to conclude that "a GFlowNet's flow space is spanned by the
                  cycle space" was unpublished, and derived it. <strong>It is published</strong> — Brunswic et
                  al., AAAI 2024, Prop. 4 / Thm. 5, as{" "}
                  <M tex={String.raw`\mathcal F_R=F+H^1(G)`} />, with a stronger form going back to Kalpazidou
                  2007. Abstract search misses it because that paper never uses those words.
                  <br />
                  <br />
                  So the table says only that <strong>these literal strings are unindexed</strong>, not that an
                  idea is new. A real novelty judgement has to read full text.
                </>
              )}
            </div>
          </div>
        </Section>

        {/* ------------------------------------------------------------- 03 */}
        <Section
          id="dial"
          num="03"
          title={{ zh: "动手:三条线一起归零", en: "Rig: three curves collapsing together" }}
          kicker={{
            zh: "8×8 hypergrid,12,869 条轨迹,全枚举精确解。拖动滑块。",
            en: `An 8×8 hypergrid, ${CUMULANTS.n_trajectories.toLocaleString("en-US")} trajectories, solved exactly. Drag the slider.`,
          }}
        >
          <WorkDial />
        </Section>

        {/* ------------------------------------------------------------- 04 */}
        <Section
          id="where"
          num="04"
          title={{ zh: "DiscoRL 落在这个空间的哪个点", en: "Where DiscoRL lands in that space" }}
          kicker={{
            zh: "关键问题只有一个:它的元网络的假设空间,到底装不装得下 detailed balance?",
            en: "One question decides it: can the meta-network's hypothesis class express detailed balance at all?",
          }}
        >
          <div className="body">
            <p>
              {zh ? (
                <>
                  这不能靠读论文回答,得读代码。下面每一行都对着{" "}
                  <a href={`https://github.com/${DISCO_REPO}/tree/${DISCO_COMMIT}`} target="_blank" rel="noreferrer">
                    {DISCO_REPO}
                  </a>{" "}
                  的 pinned commit。
                </>
              ) : (
                <>
                  Reading the paper cannot settle this; reading the code can. Every row below points at a pinned
                  commit of{" "}
                  <a href={`https://github.com/${DISCO_REPO}/tree/${DISCO_COMMIT}`} target="_blank" rel="noreferrer">
                    {DISCO_REPO}
                  </a>
                  .
                </>
              )}
            </p>
          </div>
          <SourceMatrix />

          <div className="body">
            <h3>{zh ? "两个时间尺度,分工正确得可疑" : "Two time scales, suspiciously well cast"}</h3>
            <p>
              {zh
                ? "Jarzynski / Crooks 要算功,恰好需要两样东西:一个正向协议 λ(t),和一个反向路径测度。DiscoRL 的架构里两样都在。"
                : "To compute work in the Jarzynski/Crooks sense you need exactly two things: a forward protocol λ(t) and a reverse path measure. DiscoRL's architecture has both."}
            </p>
          </div>
          <div className="scales">
            <div className="scale bwd">
              <h5>{zh ? "轨迹内 · reverse = True" : "within a trajectory · reverse = True"}</h5>
              <p className="arrow">s_T ← ⋯ ← s_t ← ⋯ ← s₀</p>
              <p>
                {zh
                  ? "每条轨迹上的 LSTM 反向展开。源码注释写明用途是 bootstrapping —— 而 bootstrapping 的传播方向,与 p_B 的计算方向是同一件事。"
                  : "The per-trajectory LSTM is unrolled backwards. The source comment says the purpose is bootstrapping — and bootstrapping propagates in the same direction p_B is computed."}
              </p>
              <span className="src">meta_nets.py:109, 116</span>
            </div>
            <div className="scale fwd">
              <h5>{zh ? "生命周期 · MetaLSTM 正向" : "across the lifetime · MetaLSTM forward"}</h5>
              <p className="arrow">λ(1) → λ(k) → λ(K)</p>
              <p>
                {zh
                  ? "跨越 agent 整个生命周期的第二个 LSTM,与轨迹内表示做乘性交互。它扮演的就是退火协议 λ(t)。"
                  : "A second LSTM spanning the agent's whole lifetime, combined multiplicatively with the per-trajectory representation. It plays the annealing protocol λ(t)."}
              </p>
              <span className="src">meta_nets.py:120, 152–155, 161</span>
            </div>
          </div>
        </Section>

        {/* ------------------------------------------------------------- 05 */}
        <Section
          id="gamma"
          num="05"
          title={{ zh: "动手:γ 是怎么量出来的", en: "Rig: how γ gets measured" }}
          kicker={{
            zh: "一个标量,一张图。这就是真实验要交付的东西。",
            en: "One scalar, one plot. That is what the real experiment delivers.",
          }}
        >
          <GammaLab />
        </Section>

        {/* ------------------------------------------------------------- 06 */}
        <Section
          id="test"
          num="06"
          title={{ zh: "把猜想真的测了 —— 它没成立", en: "The claim, actually tested — and it did not hold" }}
          kicker={{
            zh: "Disco103 的权重是公开的,所以猜想不必停在猜想。跑完之后,连「部分成立」都得撤回。",
            en: "The Disco103 weights are public, so the conjecture did not have to stay one. Once run, even the partial reading had to be withdrawn.",
          }}
        >
          <div className="body">
            <CorrectionNote />

            <h3>{zh ? "判据:一个无量纲比值" : "The discriminant: one dimensionless ratio"}</h3>
            <p>
              {zh ? (
                <>
                  detailed balance 说状态量由后继状态和局部转移概率重建,<M tex={String.raw`\log F(s)=\log F(s')+\log p_B-\log p_F`} />
                  ;value bootstrap 说 <M tex={String.raw`V(s)=r+\gamma V(s')`} />,
                  <strong>完全不依赖你走这一步的概率</strong>。差别就在这一条上。
                </>
              ) : (
                <>
                  Detailed balance rebuilds a state quantity from its successor and the local transition
                  probabilities, <M tex={String.raw`\log F(s)=\log F(s')+\log p_B-\log p_F`} />. A value
                  bootstrap says <M tex={String.raw`V(s)=r+\gamma V(s')`} />, with{" "}
                  <strong>no dependence on the probability of the action taken</strong>. That is the whole
                  difference.
                </>
              )}
            </p>
            <M
              block
              tex={String.raw`\alpha=\frac{\partial\varphi(\hat y_t)}{\partial\varphi(y_{t+1})},\qquad \beta=\frac{\partial\varphi(\hat y_t)}{\partial\log\pi(a_t\mid s_t)},\qquad \rho=\frac{\partial\varphi(\hat y_t)}{\partial r_t}`}
            />
            <p>
              {zh ? (
                <>
                  φ 不是我拟合的探针 —— 它是 <strong>Disco103 自带的 <code>y_net</code></strong>(600→16→1),
                  权重就在 <code>disco_103.npz</code> 里,所以这个测量有<strong>零个自由参数</strong>。
                  DB 要求 |β/α| ≈ 1,纯 value 要求 0。
                </>
              ) : (
                <>
                  φ is not a probe I fitted — it is <strong>Disco103's own <code>y_net</code></strong>{" "}
                  (600→16→1), shipped inside <code>disco_103.npz</code>, so the measurement has{" "}
                  <strong>zero free parameters</strong>. Detailed balance wants |β/α| ≈ 1; a pure value rule
                  wants 0.
                </>
              )}
            </p>

            <h3>{zh ? "两个方法论陷阱" : "Two methodological traps"}</h3>
            <ol>
              <li>
                {zh ? (
                  <>
                    <strong>autodiff 在这里是假的。</strong> 策略输入带 <code>stop_grad</code>
                    (<code>disco.py:338</code>),反向模式导数恒为 0 —— 第一版探针给出 β = 0.0000,那是测量失效
                    不是结果。前向值不受影响,所以必须用<strong>中心差分</strong>。
                  </>
                ) : (
                  <>
                    <strong>Autodiff lies here.</strong> The policy input carries a <code>stop_grad</code>{" "}
                    (<code>disco.py:338</code>), so the reverse-mode derivative is identically zero — the first
                    probe returned β = 0.0000, which was instrument failure, not a result. Forward values are
                    untouched, so <strong>central differences</strong> are mandatory.
                  </>
                )}
              </li>
              <li>
                {zh ? (
                  <>
                    <strong>随机初始化的 agent 测不出任何东西。</strong> Disco103 是对着有能力的 agent 元学出来的;
                    喂它近似均匀的策略和预测,输出会塌成常数,一切灵敏度都读成 0。必须先用 Disco103
                    自己把 agent 训一段。
                  </>
                ) : (
                  <>
                    <strong>A freshly initialised agent measures nothing.</strong> Disco103 was meta-learned
                    against competent agents; feed it near-uniform policies and predictions and its outputs
                    collapse to a constant, reading every sensitivity as zero. The agent has to be trained by
                    Disco103 first.
                  </>
                )}
              </li>
            </ol>

            <h3>{zh ? "结果" : "What came back"}</h3>
          </div>
          <ProbeResults />
          <div className="body">
            <h3>{zh ? "但 β/α 到底测了什么?" : "But what was β/α measuring?"}</h3>
            <p>
              {zh ? (
                <>
                  上面那个比值只有在 <strong>φ(y) 确实追踪某个 log-flow 量</strong>时才有意义。这个前提我们
                  从没验证过。在 hypergrid 上 <M tex={String.raw`\log F(s)`} /> 可以 DP 精确算出,agent 的
                  on-policy 值 <M tex={String.raw`V_\pi(s)`} /> 也能算,所以直接回归就能判。
                  阳性对照用 categorical <code>q</code> 头 —— 它按构造就是 value 头,必须追踪 value。
                </>
              ) : (
                <>
                  That ratio means something only if <strong>φ(y) really tracks a log-flow quantity</strong>, a
                  premise never checked. On the hypergrid both <M tex={String.raw`\log F(s)`} /> and the agent's
                  on-policy <M tex={String.raw`V_\pi(s)`} /> are computable exactly, so a regression settles it.
                  The positive control is the categorical <code>q</code> head, a value head by construction.
                </>
              )}
            </p>
          </div>
          <CalibrationResults />
        </Section>
        <Section
          id="kappa"
          num="07"
          title={{ zh: "障碍:它在极小化任何东西吗", en: "The obstruction: is it minimising anything?" }}
          kicker={{
            zh: "前面三次找势都失败了。这一节说明为什么——在一个不保证有势的场里找势。而且这次的量不依赖任何读出「有意义」。",
            en: "Three hunts for a potential failed. This is why: they were hunts inside a field that need not have one. And this measurement depends on no readout meaning anything.",
          }}
        >
          <div className="body">
            <p>
              {zh ? (
                <>
                  GFlowNet 由一个损失定义,更新是 <M tex={String.raw`-\nabla_\theta\mathcal L_{\rm TB}`} />,
                  <strong>真梯度</strong>,势按构造存在。DiscoRL 产生目标再回归,是 <strong>semi-gradient</strong> ——
                  和 TD 同类,而 TD 不是任何函数的梯度。
                </>
              ) : (
                <>
                  A GFlowNet is defined by a loss, so its update is{" "}
                  <M tex={String.raw`-\nabla_\theta\mathcal L_{\rm TB}`} /> — a <strong>true gradient</strong>,
                  and a potential exists by construction. DiscoRL produces a target and regresses onto it: a{" "}
                  <strong>semi-gradient</strong>, the same structure that makes TD the gradient of nothing.
                </>
              )}
            </p>
            <M
              block
              tex={String.raw`v(u)=\hat p(u)-p(u),\qquad J=\underbrace{B}_{\partial\hat p/\partial u}-\underbrace{D}_{\text{对称}},\qquad \kappa=\frac{\lVert\mathrm{sym}\,B\rVert_F}{\lVert B\rVert_F}`}
            />
            <p>
              {zh ? (
                <>
                  一切非保守性都住在 <strong>B</strong> —— 正是 <code>stop_gradient</code> 丢掉的那一项。
                  Poincaré 引理:有势 ⟺ Jacobian 对称。所以问题只剩一个标量。
                </>
              ) : (
                <>
                  All non-conservativity lives in <strong>B</strong> — precisely the term{" "}
                  <code>stop_gradient</code> discards. By the Poincaré lemma a field is a gradient iff its
                  Jacobian is symmetric, so the question reduces to one scalar.
                </>
              )}
            </p>

            <h3>{zh ? "刻度是解析的" : "The scale is analytic"}</h3>
            <p>
              {zh ? (
                <>
                  对任意算子 <M tex={String.raw`\lVert B\pm B^\top\rVert^2=2\lVert B\rVert^2\pm2\,\mathrm{tr}(B^2)`} />。
                  「目标只读未来」的<strong>因果</strong>自举严格三角,<M tex={String.raw`\mathrm{tr}(B^2)=0`} />,
                  于是 <M tex={String.raw`\kappa=1/\sqrt2`} /> <strong>恰好</strong>,与维度、与细节无关。
                  这不是拟合出来的参照,是结构地板。
                </>
              ) : (
                <>
                  For any operator{" "}
                  <M tex={String.raw`\lVert B\pm B^\top\rVert^2=2\lVert B\rVert^2\pm2\,\mathrm{tr}(B^2)`} />. A{" "}
                  <strong>causal</strong> bootstrap, whose target reads only the future, is strictly triangular,
                  so <M tex={String.raw`\mathrm{tr}(B^2)=0`} /> and <M tex={String.raw`\kappa=1/\sqrt2`} />{" "}
                  <strong>exactly</strong>, independent of dimension and of every other detail. Not a fitted
                  reference — a structural floor.
                </>
              )}
            </p>
            <h3>{zh ? "结果" : "Result"}</h3>
          </div>
          <KappaResults />
        </Section>

        {/* ------------------------------------------------------------- 08 */}
        <Section
          id="reduction"
          num="08"
          title={{ zh: "归约是精确的", en: "The reduction is exact" }}
          kicker={{
            zh: "「把 GFlowNet 的更新规则元学习出来」算不算一个良定义的计划,取决于一个已发表的定理。这一节把那个定理在本机验到 1e-16。",
            en: "Whether 'meta-learn a GFlowNet update rule' is a well-posed plan turns on one published theorem. This section checks that theorem here, to 1e-16.",
          }}
        >
          <div className="body">
            <p>{zh ? "三步:" : "Three steps:"}</p>
            <ol>
              <li>{zh ? "GFlowNet 学出来的是一个策略。" : "A GFlowNet learns a policy."}</li>
              <li>
                {zh
                  ? "GFlowNet 的训练精确地等于某一个特定 MDP 上的软 RL。"
                  : "GFlowNet training is exactly soft RL on one specific MDP."}
              </li>
              <li>
                {zh ? "一条被发现的更新规则就是一个 RL 算法。" : "A discovered update rule is an RL algorithm."}
              </li>
            </ol>
            <p>
              {zh ? (
                <>
                  所以「元学习一条 GFlowNet 更新规则」不是类比,是一件可以直接去做的事:把规则指向第 2
                  步造出来的那个 MDP。第 2 步是别人的定理,所以先验它。
                </>
              ) : (
                <>
                  So "meta-learn a GFlowNet update rule" is not an analogy but something you can go and do: point
                  the rule at the MDP that step 2 builds. Step 2 is someone else's theorem, so it gets checked
                  first.
                </>
              )}
            </p>

            <div className="card">
              <h4>
                {zh ? "定理 1 —— 归约" : "Theorem 1 — the reduction"} <Chip p="published" />
              </h4>
              <p style={{ margin: 0, fontSize: 14.5, color: "var(--text-2)" }}>
                {zh ? (
                  <>
                    给定 DAG、终态集 X、固定的反向策略 P_B 和奖励 R,造这样一个 MDP:内部转移的奖励取{" "}
                    <M tex={String.raw`\log P_B(s\mid s')`} />,终止时的奖励取 <M tex={String.raw`\log R(x)`} />。
                    在熵系数<strong>恰好为 1</strong> 时,软最优策略就是 GFlowNet 的前向策略。
                  </>
                ) : (
                  <>
                    Given a DAG with terminal set X, a fixed backward policy P_B and a reward R, build the MDP
                    whose interior transitions pay <M tex={String.raw`\log P_B(s\mid s')`} /> and whose
                    termination pays <M tex={String.raw`\log R(x)`} />. At entropy coefficient{" "}
                    <strong>exactly 1</strong> the soft-optimal policy is the GFlowNet forward policy.
                  </>
                )}
              </p>
              <M
                block
                tex={String.raw`r(s\to s')=\log P_B(s\mid s'),\quad r(x\to\top)=\log R(x),\quad \lambda=1\;\Longrightarrow\;V^*(s)=\log F(s)`}
              />
            </div>

            <div className="card">
              <h4>
                {zh ? "命题 1 —— 那个回报的上确界" : "Proposition 1 — where that return tops out"}{" "}
                <Chip p="published" />
              </h4>
              <M block tex={String.raw`V_1^{\pi}(s_0)=\log Z-\mathrm{KL}\big(q^{\pi}\,\Vert\,P_B\big),\qquad \sup_\pi V_1^{\pi}(s_0)=\log Z`} />
              <p style={{ margin: 0, fontSize: 14.5, color: "var(--text-2)" }}>
                {zh
                  ? "这一行是第 12 节的全部前提:在这个 MDP 里,回报差多少,分布就差多少。"
                  : "This one line is the whole premise of section 12: in this MDP, however much return you give up is how much distribution accuracy you give up."}
              </p>
            </div>

            <h3>{zh ? "本机验了四条" : "Four checks, run here"}</h3>
            <div className="scroller">
              <table className="tbl">
                <thead>
                  <tr>
                    <th style={{ width: "18%" }}>{zh ? "检查" : "Check"}</th>
                    <th>{zh ? "量的是什么" : "What it measures"}</th>
                    <th className="num">{zh ? "结果" : "Result"}</th>
                  </tr>
                </thead>
                <tbody>
                  {REDUCTION_CHECKS.map((c) => (
                    <tr key={c.id}>
                      <td className="num">{c.id}</td>
                      <td>{c.what[lang]}</td>
                      <td className="num ok">
                        {c.result}
                        {c.note ? (
                          <span style={{ color: "var(--text-3)" }}> · {c.note[lang]}</span>
                        ) : null}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="hint">
              {zh
                ? "R4 换了 50 个随机反向策略,最差一个也只有 1.665e-16。归约不依赖 P_B 挑得好不好。"
                : "R4 swaps in 50 random backward policies; the worst of them still reads 1.665e-16. The reduction does not depend on P_B being chosen well."}
            </p>

            <h3>{zh ? "系数偏离 1 要付多少" : "What leaving coefficient 1 costs"}</h3>
            <p>
              {zh
                ? "定理里的 1 不是一个可调的旋钮。给该论文的 Remark 3 定个价 —— 终态分布到 R/Z 的 KL:"
                : "The 1 in the theorem is not a knob. Pricing that paper's Remark 3 — the KL from the terminal distribution to R/Z:"}
            </p>
            <div className="scroller">
              <table className="tbl">
                <thead>
                  <tr>
                    <th style={{ width: "50%" }}>{zh ? "熵系数 λ" : "entropy coefficient λ"}</th>
                    <th className="num">{zh ? "到 R/Z 的 KL" : "KL to R/Z"}</th>
                  </tr>
                </thead>
                <tbody>
                  {REMARK3.map((r) => (
                    <tr key={r.lam}>
                      <td className="num" style={r.lam === "1.0" ? { color: "var(--ver)" } : undefined}>
                        {r.lam}
                      </td>
                      <td className={`num ${r.lam === "1.0" ? "ok" : ""}`}>{r.kl}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="callout">
              <strong>{zh ? "这就是麻烦的来源。" : "This is where the trouble starts."}</strong>{" "}
              {zh ? (
                <>
                  该论文的 Remark 4 报告过:带自适应系数的 SAC 收敛到一个<strong>有偏</strong>的最终策略。
                  下一节把同样的失败模式在一条被发现的规则上重放一遍 —— 而那条规则连熵系数都没有。
                </>
              ) : (
                <>
                  Remark 4 of the same paper reports SAC with an adaptive coefficient converging to a{" "}
                  <strong>biased</strong> final policy. The next section reproduces that failure mode for a
                  discovered rule — one that has no entropy coefficient at all.
                </>
              )}
            </div>
          </div>
        </Section>

        {/* ------------------------------------------------------------- 09 */}
        <Section
          id="hostile"
          num="09"
          title={{ zh: "然后这个 MDP 打败了那条规则", en: "Then that MDP defeats the rule" }}
          kicker={{
            zh: "同一条规则、同一个网络、同一份预算:去掉每步的 log P_B 支付,它解开了这张格子;加上,它停在源点不动。差 4.285。",
            en: "Same rule, same network, same budget. Without the per-step log P_B payments it solves the grid; with them it stops at the source and stays there. A gap of 4.285.",
          }}
        >
          <div className="body">
            <p>
              {zh ? (
                <>
                  6×6 格子,三个光滑的奖励峰,<M tex={String.raw`\log R`} /> 跨度 5.43。带 log P_B
                  支付时能拿到的最好无折扣回报是 <strong>2.285</strong>,不带时是 <strong>3.431</strong>。
                  站在源点不动付 <strong>-2.000</strong>。
                </>
              ) : (
                <>
                  A 6x6 grid, three smooth reward modes, <M tex={String.raw`\log R`} /> spanning 5.43. The best
                  undiscounted return available is <strong>2.285</strong> with the log P_B payments and{" "}
                  <strong>3.431</strong> without. Stopping at the source pays <strong>-2.000</strong>.
                </>
              )}
            </p>
          </div>

          <div className="readouts" style={{ margin: "26px 0 22px" }}>
            <div className="ro">
              <span className="k">{zh ? "未整形归约的回报" : "return, unshaped reduction"}</span>
              <span className="v" style={{ color: "var(--rl)" }}>-2.000</span>
              <span className="n">{zh ? "就是源点的价钱" : "the price of the source"}</span>
            </div>
            <div className="ro">
              <span className="k">{zh ? "距最好回报" : "gap to the best available"}</span>
              <span className="v" style={{ color: "var(--rl)" }}>+4.285</span>
              <span className="n">{zh ? "三个种子三位小数一致" : "identical to three decimals across three seeds"}</span>
            </div>
            <div className="ro locked">
              <span className="k">{zh ? "去掉支付后的间隙" : "gap once the payments go"}</span>
              <span className="v">+0.552</span>
              <span className="n">{zh ? "同一条规则,同一张格子" : "same rule, same grid"}</span>
            </div>
          </div>

          <div className="scroller">
            <table className="tbl">
              <thead>
                <tr>
                  <th style={{ width: "44%" }}>{zh ? "臂" : "Arm"}</th>
                  <th className="num">{zh ? "回报" : "return"}</th>
                  <th className="num">{zh ? "距最好" : "gap to best"}</th>
                  <th>{zh ? "残差" : "residual"}</th>
                </tr>
              </thead>
              <tbody>
                {HOSTILE_ARMS.map((a) => (
                  <tr key={a.ret}>
                    <td>{a.arm[lang]}</td>
                    <td className={`num ${a.mark ?? ""}`}>{a.ret}</td>
                    <td className={`num ${a.mark ?? ""}`}>{a.gap}</td>
                    <td className="num" style={{ whiteSpace: "normal", color: "var(--text-3)" }}>
                      {a.residual[lang]}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="body">
            <h3>{zh ? "机制" : "The mechanism"}</h3>
            <p>
              {zh ? (
                <>
                  每走一步内部转移大约付 -0.69。所以<strong>立刻停下</strong>(-2.000)相对初始策略(-2.281)
                  是一次真正的改进,规则拿到了它,然后就不动了。要逃出去需要一条特定的五步路径。
                </>
              ) : (
                <>
                  Each interior step pays about -0.69. So <strong>stopping immediately</strong> (-2.000) is a
                  genuine improvement over the initial policy (-2.281); the rule takes it and stays. Escaping
                  needs one specific five-step path.
                </>
              )}
            </p>
            <div className="callout" style={{ borderLeftColor: "var(--rl)" }}>
              <strong>{zh ? "失败的是 MDP,不是接线。" : "The failure is the MDP, not the wiring."}</strong>{" "}
              {zh ? (
                <>
                  中间那一行是对照组:<strong>同一条规则、同一个网络、同一份观测、同一份预算</strong>,只把每步的
                  log P_B 支付去掉,它就把同一张格子解到 +0.552。所以这不是「DiscoRL 接错了」,是归约交出来的那个
                  MDP 对它是敌意的。
                </>
              ) : (
                <>
                  The middle row is the control: <strong>same rule, same network, same observation, same
                  budget</strong>, with only the per-step log P_B payments removed — and it solves the same grid
                  to within +0.552. So this is not DiscoRL wired up wrong. The MDP the reduction hands over is
                  hostile to it.
                </>
              )}
            </div>
            <div className="card">
              <h4>
                {zh ? "它连一个可以调的系数都没有" : "It has no coefficient to turn"} <Chip p="source" />
              </h4>
              <p style={{ margin: 0, fontSize: 14.5, color: "var(--text-2)" }}>
                {zh ? (
                  <>
                    Disco103 的 <code>hyper_params</code> 是 pi_cost、y_cost、z_cost、value_cost、
                    aux_policy_cost、target_params_coeff、value_fn_td_lambda、discount —— <strong>没有熵系数</strong>。
                    同一个库里的 actor-critic 带着 <code>entropy_cost</code> 0.2。定理 1 要求的那个「恰好为 1」,
                    在这条规则上没有对应物。
                  </>
                ) : (
                  <>
                    Disco103's <code>hyper_params</code> are pi_cost, y_cost, z_cost, value_cost,
                    aux_policy_cost, target_params_coeff, value_fn_td_lambda and discount —{" "}
                    <strong>no entropy coefficient</strong>. The actor-critic in the same library carries{" "}
                    <code>entropy_cost</code> 0.2. The "exactly 1" that Theorem 1 asks for has no counterpart in
                    this rule.
                  </>
                )}
              </p>
            </div>
          </div>
        </Section>

        {/* ------------------------------------------------------------- 10 */}
        <Section
          id="shaping"
          num="10"
          title={{ zh: "势能整形修好优化,并且证明不动目标", en: "Shaping fixes the optimisation, provably not the target" }}
          kicker={{
            zh: "Φ 是把奖励抹平之后、同一张 DAG 在系数 1 下的软值。它从头到尾没读过 R,所以不是把答案偷偷塞回去。",
            en: "Φ is the soft value of the same DAG at coefficient 1 with the reward flattened away. It never reads R, so it is not the answer smuggled back in.",
          }}
        >
          <div className="body">
            <p>
              {zh ? (
                <>
                  上一节的病是优化上的,不是目标上的:每步 -0.69 的支付把一条贪心的规则钉在源点。
                  势能整形正是为这种病准备的 —— 它改变每一步的账,不改变最优解。用{" "}
                  <M tex={String.raw`r'=r+\Phi(s')-\Phi(s)`} />:
                </>
              ) : (
                <>
                  The disease in the previous section is in the optimisation, not in the target: a per-step
                  payment of -0.69 pins a greedy rule to the source. Potential shaping is the standard treatment
                  — it changes the bookkeeping on every step and leaves the optimum alone. With{" "}
                  <M tex={String.raw`r'=r+\Phi(s')-\Phi(s)`} />:
                </>
              )}
            </p>
            <M
              block
              tex={String.raw`\lambda\log\!\sum_a\exp\!\Big(\tfrac{r'+V'(s')}{\lambda}\Big)=V(s)-\Phi(s),\qquad \pi'\propto\exp\!\Big(\tfrac{r+V(s')}{\lambda}\Big)=\pi`}
            />
            <p>
              {zh ? (
                <>
                  第二个等式就是要点:软最优<strong>策略</strong>一个字都没变,变的只是值函数的基准。
                  这一条没有停在推导上 —— 它是逐点验的。
                </>
              ) : (
                <>
                  The second identity is the point: the soft-optimal <strong>policy</strong> does not move at
                  all; only the baseline of the value function does. This was not left as an argument — it was
                  checked pointwise.
                </>
              )}
            </p>
          </div>

          <div className="readouts" style={{ margin: "26px 0 22px" }}>
            <div className="ro locked">
              <span className="k">{zh ? "软最优策略的最大变化" : "largest move in the soft-optimal policy"}</span>
              <span className="v">1.776e-15</span>
              <span className="n">λ ∈ {"{0.1, 0.5, 1, 2, 5}"}</span>
            </div>
            <div className="ro">
              <span className="k">{zh ? "整形后距最好回报" : "gap after shaping"}</span>
              <span className="v" style={{ color: "var(--gfn)" }}>+0.323</span>
              <span className="n">{zh ? "整形前是 +4.285" : "it was +4.285 before"}</span>
            </div>
            <div className="ro">
              <span className="k">{zh ? "整形臂的残差" : "residual, shaped arm"}</span>
              <span className="v" style={{ color: "var(--conj)" }}>0.649</span>
              <span className="n">{zh ? "第 11 节要处理的就是这个数" : "the number section 11 has to explain"}</span>
            </div>
          </div>

          <div className="body">
            <div className="callout ok">
              <strong>{zh ? "Φ 不可能是答案的伪装。" : "Φ cannot be the answer in disguise."}</strong>{" "}
              {zh ? (
                <>
                  它是把奖励抹平之后同一张 DAG 在系数 1 下的软值 —— <strong>纯粹的图几何</strong>,
                  构造上从不接触 R。所以「整形之后它就学会了」不能被读成「我们把 R 提前告诉了它」。
                  间隙从 +4.285 掉到 +0.323,而目标在 1.776e-15 的意义上没动过。
                </>
              ) : (
                <>
                  It is the soft value of the same DAG at coefficient 1 with the reward flattened away —{" "}
                  <strong>pure graph geometry</strong>, never touching R by construction. So "it learns once
                  shaped" cannot be read as "we told it the answer early". The gap falls from +4.285 to +0.323
                  while the target moves by 1.776e-15.
                </>
              )}
            </div>
          </div>
        </Section>

        {/* ------------------------------------------------------------- 11 */}
        <Section
          id="gapfit"
          num="11"
          title={{ zh: "那个失配量的是优化间隙", en: "The misfit is the optimisation gap" }}
          kicker={{
            zh: "先量出「大」到底是多大:0.0861。然后这一节撤回它自己上一版的判定 —— 残差看起来像族外,其实由智能体优化得多差预测,R² 0.993。",
            en: "First measure what 'large' means: 0.0861. Then this section withdraws its own earlier verdict — the residual that looked like family misfit is predicted by how badly the agent optimised, at R² 0.993.",
          }}
        >
          <div className="body">
            <p>
              {zh ? (
                <>
                  读出仪器的做法是:用值迭代精确算出软族,把智能体分桶成一个状态条件策略,再用黄金分割去拟合系数,
                  返回一个<strong>系数</strong>和一个<strong>任何系数都消不掉的残差</strong>。
                  仪器本身先被检验过:把精确的软最优策略灌进智能体走的同一条环境通路,读回 λ_eff 0.995、
                  残差 0.0062。所以下面这些残差是各臂自己的。
                </>
              ) : (
                <>
                  The readout works like this: compute the soft family exactly by value iteration, bucket the
                  agent into a state-conditional policy, fit the coefficient by golden section, and return both a{" "}
                  <strong>coefficient</strong> and <strong>the residual divergence no coefficient removes</strong>.
                  The instrument was checked first: drive the exact soft-optimal policy down the same environment
                  path the agents take and it reads λ_eff 0.995, residual 0.0062. So the residuals below belong
                  to the arms.
                </>
              )}
            </p>

            <h3>{zh ? "多大的残差算大,量出来" : "How large is large, measured"}</h3>
            <div className="scroller">
              <table className="tbl">
                <thead>
                  <tr>
                    <th style={{ width: "60%" }}>{zh ? "温度误差" : "temperature error"}</th>
                    <th className="num">nats</th>
                  </tr>
                </thead>
                <tbody>
                  {RESIDUAL_SCALE.map((r) => (
                    <tr key={r.nats}>
                      <td style={r.ceiling ? { color: "var(--ver)" } : undefined}>{r.err[lang]}</td>
                      <td className={`num ${r.ceiling ? "ok" : ""}`}>{r.nats}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="callout">
              <strong>{zh ? "撤回。" : "Withdrawn."}</strong>{" "}
              {zh ? (
                <>
                  <code>temperature.py</code> 原来在残差超过 <strong>0.4</strong> 时判一个智能体在族外。那个
                  0.4 是拍出来的,不是量出来的。它已经被 25% 温度误差对应的 <strong>0.0861</strong> 替换,
                  而 0.0861 是量出来的。
                </>
              ) : (
                <>
                  <code>temperature.py</code> used to call an agent outside the family when its residual passed{" "}
                  <strong>0.4</strong>. That 0.4 was picked, not measured. It has been replaced by the figure for
                  a 25 percent temperature error, <strong>0.0861</strong>, which is measured.
                </>
              )}
            </div>

            <h3>{zh ? "那把梯子只排序,不定标" : "The ladder ranks; it never calibrated"}</h3>
            <p>
              {zh ? (
                <>
                  读出仪器原来是对着 <code>entropy_cost</code> c ∈ {"{0, 0.3, 1, 3}"} 的 actor-critic
                  臂校准的,而且文件里一度把 c = 1 那条臂称作「按构造系数恰为一的臂」。<strong>那是错的</strong>,
                  而且不需要训练就能看出来:那条更新规则的精确不动点是{" "}
                  <M tex={String.raw`\pi\propto\exp(Q_\pi/c)`} />,用的是<strong>普通的</strong> Q_π,
                  那是另一个单参数族。它在 c 约 0.5 以下与软族重合,以上就离开,到 c = 3 时残差 3.02。
                </>
              ) : (
                <>
                  The readout used to be calibrated against actor-critic arms at <code>entropy_cost</code> c ∈{" "}
                  {"{0, 0.3, 1, 3}"}, and one version of the file called the arm at c = 1 "an arm whose
                  coefficient is exactly one by construction". <strong>That is false</strong>, and it needs no
                  training to see: that update rule's exact fixed point is{" "}
                  <M tex={String.raw`\pi\propto\exp(Q_\pi/c)`} /> with the <strong>ordinary</strong> Q_π, which
                  is a different one-parameter family. It coincides with the soft family below c about 0.5,
                  leaves it above, and reaches residual 3.02 at c = 3.
                </>
              )}
            </p>
            <p>
              {zh ? (
                <>
                  这个不动点族预测了它从没被拟合过的臂:c = 3 处残差 3.021 对实测 3.009,λ_eff 1.636
                  对实测 1.639。而在 c = 1 处,精确不动点坐在 <strong>λ_eff 0.836、残差 0.685</strong>,
                  是天花板的八倍。<strong>所以没有任何一级台阶钉住过绝对刻度。</strong>
                </>
              ) : (
                <>
                  That family predicts arms it was never fitted to: at c = 3, residual 3.021 against the measured
                  3.009, and λ_eff 1.636 against the measured 1.639. At c = 1 the exact fixed point sits at{" "}
                  <strong>λ_eff 0.836 with residual 0.685</strong>, eight times the ceiling.{" "}
                  <strong>So no rung ever pinned the absolute scale.</strong>
                </>
              )}
            </p>
            <p>
              {zh ? (
                <>
                  用动态规划把 c 到 λ 的映射算出来就修好了:它单调可逆。经过它,disco 整形臂的 0.362
                  对应的等效熵系数是 <strong>0.418</strong>,而把一条臂放到 λ = 1 需要的是 <strong>1.181</strong>。
                </>
              ) : (
                <>
                  Computing the c-to-λ map by dynamic programming fixes it: the map is monotone and invertible.
                  Through it, disco shaped's 0.362 is an equivalent entropy coefficient of <strong>0.418</strong>,
                  against the <strong>1.181</strong> that would place an arm at λ = 1.
                </>
              )}
            </p>

            <h3>{zh ? "两个族的比较,以及它的撤回" : "The family comparison, and its withdrawal"}</h3>
            <p>
              {zh ? (
                <>
                  软族回答「它是不是一个 GFlowNet」,上面那个不动点族回答「它是不是带熵奖励的 actor-critic」。
                  下面是当初那个判定所依据的聚合数字 —— 对最后一步可用行取的均值:
                </>
              ) : (
                <>
                  The soft family answers "is it a GFlowNet"; the fixed-point family above answers "is it an
                  entropy-bonus actor-critic". These are the aggregates the earlier verdict rested on — means
                  over final-step admissible rows:
                </>
              )}
            </p>
            <div className="scroller">
              <table className="tbl">
                <thead>
                  <tr>
                    <th style={{ width: "40%" }}>{zh ? "聚合" : "aggregate"}</th>
                    <th className="num">{zh ? "软族" : "soft family"}</th>
                    <th className="num">{zh ? "actor-critic 不动点族" : "actor-critic fixed point"}</th>
                  </tr>
                </thead>
                <tbody>
                  {FAMILY_MEANS.map((f) => (
                    <tr key={f.soft}>
                      <td>{f.seeds[lang]}</td>
                      <td className="num">{f.soft}</td>
                      <td className="num">{f.rule}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p>
              {zh
                ? "全都在 0.0861 天花板之上,于是判定写成了 NEITHER FAMILY。均值底下的逐种子行从 0.041 一路排到 1.423,而那个跨度不是噪声。这个 MDP 里能拿到的最好回报是 2.2847:"
                : "All of them sit above the 0.0861 ceiling, and the verdict read NEITHER FAMILY. The per-seed rows behind those means run from 0.041 to 1.423, and that range is not noise. Best achievable return in this MDP is 2.2847:"}
            </p>
          </div>

          <div className="scroller">
            <table className="tbl">
              <thead>
                <tr>
                  <th className="num">{zh ? "种子" : "seed"}</th>
                  <th>{zh ? "可用" : "admissible"}</th>
                  <th className="num">{zh ? "回报" : "return"}</th>
                  <th className="num">{zh ? "间隙" : "gap"}</th>
                  <th className="num">{zh ? "软族 λ" : "soft λ"}</th>
                  <th className="num">{zh ? "软族残差" : "soft residual"}</th>
                  <th className="num">{zh ? "不动点残差" : "rule residual"}</th>
                  <th>{zh ? "在天花板内" : "inside the ceiling"}</th>
                </tr>
              </thead>
              <tbody>
                {GAP_SEEDS.map((s) => (
                  <tr key={s.seed}>
                    <td className="num" style={s.best ? { color: "var(--ver)" } : undefined}>
                      {s.seed}
                    </td>
                    <td style={{ color: s.admissible ? "var(--text-2)" : "var(--text-3)" }}>
                      {s.admissible ? (zh ? "是" : "yes") : zh ? "否" : "no"}
                    </td>
                    <td className="num">{s.ret}</td>
                    <td className="num">{s.gap}</td>
                    <td className="num">{s.lam}</td>
                    <td className={`num ${s.best ? "ok" : ""}`}>{s.soft}</td>
                    <td className={`num ${s.best ? "ok" : ""}`}>{s.rule}</td>
                    <td className={s.inside.en === "both" ? "ok" : ""}>{s.inside[lang]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="body">
            <p>
              {zh ? "把残差对间隙做回归,在五个可用种子上:" : "Regressing residual on gap over the five admissible seeds:"}
            </p>
            <div className="scroller">
              <table className="tbl">
                <thead>
                  <tr>
                    <th style={{ width: "34%" }}>{zh ? "拟合" : "fit"}</th>
                    <th className="num">{zh ? "残差 = 斜率 × 间隙 + 截距" : "residual = slope × gap + intercept"}</th>
                    <th className="num">pearson</th>
                    <th className="num">R²</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>{zh ? "软族" : "soft family"}</td>
                    <td className="num">1.457 × gap + 0.0410</td>
                    <td className="num">+0.996</td>
                    <td className="num ok">0.993</td>
                  </tr>
                  <tr>
                    <td>{zh ? "actor-critic 不动点族" : "actor-critic fixed point"}</td>
                    <td className="num">0.680 × gap + 0.0958</td>
                    <td className="num">+0.982</td>
                    <td className="num">0.963</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p>
              {zh
                ? "对软族那条做留一法,pearson 保持在 0.977 到 0.999,截距保持在 0.0089 到 0.0599 nat,六次拟合全部落在 0.0861 天花板之内。所以它不是被某一个点拖出来的。"
                : "Leave-one-out on the soft fit keeps pearson in 0.977 to 0.999 and the intercept in 0.0089 to 0.0599 nats, inside the 0.0861 ceiling in all six fits. No single seed carries it."}
            </p>

            <div className="callout" style={{ borderLeftColor: "var(--rl)" }}>
              <strong>{zh ? "NEITHER FAMILY 判定撤回。" : "The NEITHER FAMILY verdict is withdrawn."}</strong>{" "}
              {zh ? (
                <>
                  一个真正的族失配<strong>不会在乎智能体优化得好不好</strong>。这一个被优化得多好预测,
                  R² 0.993,而且随着间隙趋于零,它消进天花板里。所以那个残差从来不是关于族归属的证据,
                  它是关于优化的。
                </>
              ) : (
                <>
                  A real family misfit <strong>would not care how well the agent optimised</strong>. This one is
                  predicted by it at R² 0.993, and it vanishes into the ceiling as the gap goes to zero. So that
                  residual was never evidence about family membership. It was evidence about optimisation.
                </>
              )}
            </div>

            <h3>{zh ? "同一批行支持的是这两条" : "What the same rows do support"}</h3>
            <ol>
              <li>
                {zh ? (
                  <>
                    这条被发现的规则一旦<strong>真的优化了</strong>这个 MDP,它的终态分布就落在软族的 25%
                    温度误差之内,系数在 0.19 到 0.22 附近 —— 也就是比「使它成为 GFlowNet」的那个系数
                    <strong>大约冷五倍</strong>。写成「大约冷五倍」而不是一个精确数字:可比间隙下跨种子的散布是
                    0.186 到 0.254。
                  </>
                ) : (
                  <>
                    Once the discovered rule <strong>actually optimises</strong> this MDP, its terminal
                    distribution sits within a 25 percent temperature error of the soft family, at a coefficient
                    near 0.19 to 0.22 — <strong>about five times colder</strong> than the coefficient that would
                    make it a GFlowNet. That is written as "about five times colder" rather than as a number: the
                    spread across seeds at comparable gaps is 0.186 to 0.254.
                  </>
                )}
              </li>
              <li>
                {zh ? (
                  <>
                    <strong>这个 MDP 分不开那两个族。</strong>在优化得最好的可用种子上,两个残差只差 0.0023
                    nat(0.0649 对 0.0626)。所以正确的说法是这个测量分不开它们,而不是其中哪一个赢了。
                  </>
                ) : (
                  <>
                    <strong>This MDP does not separate the two families.</strong> At the best-optimising
                    admissible seed the two residuals differ by 0.0023 nats (0.0649 against 0.0626). The correct
                    statement is that the measurement cannot tell them apart, not that either family wins.
                  </>
                )}
              </li>
            </ol>
          </div>
        </Section>

        {/* ------------------------------------------------------------- 12 */}
        <Section
          id="metaobj"
          num="12"
          title={{ zh: "元目标里的一个标量", en: "One scalar in the meta-objective" }}
          kicker={{
            zh: "架构、输入、内层步数、种子全部不动。只把给规则打分的那个标量从回报换成 V₁ = log Z − KL。留出的地形上 KL 从 1.7035 掉到 0.0666。",
            en: "Architecture, inputs, inner horizon and seeds all held fixed. Only the scalar that scores the rule changes, from the return to V₁ = log Z − KL. On held-out landscapes KL falls from 1.7035 to 0.0666.",
          }}
        >
          <div className="body">
            <p>
              {zh ? (
                <>
                  先验身份,双精度:在整形后的奖励上 <M tex={String.raw`V_1+\Phi(s_0)=\log Z`} /> 到{" "}
                  <strong>0.00e+00</strong>,而取到它的那个策略散度是 <strong>5e-16</strong>。
                  所以在这个 MDP 里,<strong>熵正则化回报就是分布精度本身</strong> ——
                  代价是一个必须从轨迹上算出来的标量。
                </>
              ) : (
                <>
                  The identity gets checked first, in double precision: on the shaped rewards{" "}
                  <M tex={String.raw`V_1+\Phi(s_0)=\log Z`} /> to <strong>0.00e+00</strong>, and the policy that
                  attains it has divergence <strong>5e-16</strong>. So in this MDP{" "}
                  <strong>the entropy-regularised return is distribution accuracy</strong>, at the price of a
                  scalar computed from trajectories.
                </>
              )}
            </p>
            <p>
              {zh ? (
                <>
                  学出来的规则是<strong>局部</strong>的。每条转移上它只读:即时奖励、它自己在两端的辅助预测、
                  当前的 logit 和 log 概率、以及这个动作是否终止。它从来看不到 R/Z、log Z 或访问分布。
                  策略下游的一切都是闭式的 —— 格子是分层 DAG,占用就是{" "}
                  <M tex={String.raw`e_0^\top(I-T)^{-1}`} /> —— 这把采样噪声从元梯度里拿掉了。
                  所以这是同一条规则的<strong>无穷样本极限</strong>。在 16 个随机地形上元训练,在另外
                  16 个不相交的地形上读出。
                </>
              ) : (
                <>
                  The learned rule is <strong>local</strong>. Per transition it reads the immediate reward, its
                  own auxiliary prediction at both ends, the current logit and log-probability, and whether the
                  action terminates. It never sees R/Z, log Z or the visit distribution. Everything downstream of
                  the policy is closed form — the grid is a layered DAG, so occupancy is{" "}
                  <M tex={String.raw`e_0^\top(I-T)^{-1}`} /> — which removes sampling noise from the
                  meta-gradient. This is the <strong>infinite-sample limit</strong> of that rule. Meta-trained on
                  16 random landscapes, read out on 16 disjoint ones.
                </>
              )}
            </p>

            <h3>{zh ? "两个对照" : "Two controls"}</h3>
            <ol>
              <li>
                {zh
                  ? "每条臂自己挑步长,评分用的是它自己的元目标在训练地形上的值,从不用到目标分布的散度。"
                  : "Each arm chooses its own step sizes, scored by its own meta-objective on training landscapes and never by divergence to the target."}
              </li>
              <li>
                {zh ? (
                  <>
                    每条规则必须在它<strong>被训练的那个标量</strong>上赢。它确实赢了,在训练时的步数上,
                    以及四倍步数之外:回报规则 -0.341 对 -1.327(比回报);V₁ 规则 +2.231 对 -0.298(比 V₁)。
                  </>
                ) : (
                  <>
                    Each rule has to win on <strong>the scalar it was trained on</strong>. It does, at the trained
                    horizon and four times beyond: the return rule reads -0.341 against -1.327 on return; the V₁
                    rule reads +2.231 against -0.298 on V₁.
                  </>
                )}
              </li>
            </ol>
            <div className="callout">
              <strong>{zh ? "撤回。" : "Withdrawn."}</strong>{" "}
              {zh ? (
                <>
                  更早的一次运行显示两个元目标相差 <strong>88 倍</strong>。第 2 个对照作废了它:
                  以回报训练的那条规则在<strong>回报本身</strong>上就输了,所以那次比的是一条训练过的规则和一条
                  欠训练的规则。
                </>
              ) : (
                <>
                  An earlier run showed an <strong>88x</strong> separation. Control 2 voided it: the
                  return-trained rule was losing on <strong>return itself</strong>, so that run compared a trained
                  rule against an undertrained one.
                </>
              )}
            </div>

            <h3>{zh ? "结果,留出地形" : "The result, on held-out landscapes"}</h3>
            <div className="scroller">
              <table className="tbl">
                <thead>
                  <tr>
                    <th style={{ width: "40%" }}>{zh ? "元目标" : "meta-objective"}</th>
                    <th className="num">{zh ? "30 步的 KL" : "KL at 30 steps"}</th>
                    <th className="num">{zh ? "120 步的 KL" : "KL at 120 steps"}</th>
                  </tr>
                </thead>
                <tbody>
                  {METAOBJ_KL.map((m) => (
                    <tr key={m.kl30}>
                      <td style={m.win ? { color: "var(--ver)" } : undefined}>{m.obj[lang]}</td>
                      <td className={`num ${m.win ? "ok" : "no"}`}>{m.kl30}</td>
                      <td className={`num ${m.win ? "ok" : "no"}`}>{m.kl120}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p>
              {zh ? (
                <>
                  <strong>一个标量换来 26 倍</strong>,架构、输入、内层步数和种子完全相同。
                  两条规则读的是同一批转移,唯一的差别是外层拿什么给它们打分。
                </>
              ) : (
                <>
                  <strong>A factor of 26 from one scalar</strong>, with architecture, inputs, inner horizon and
                  seeds identical. Both rules read the same transitions; the only difference is what the outer
                  loop scores them with.
                </>
              )}
            </p>

            <h3>{zh ? "对上手工设计的损失" : "Against the hand-designed loss"}</h3>
            <div className="scroller">
              <table className="tbl">
                <thead>
                  <tr>
                    <th style={{ width: "70%" }}>{zh ? "detailed balance,最好的加权" : "detailed balance at its best weighting"}</th>
                    <th className="num">KL</th>
                  </tr>
                </thead>
                <tbody>
                  {DB_COMPARE.map((d) => (
                    <tr key={d.kl}>
                      <td style={d.win ? { color: "var(--gfn)" } : undefined}>{d.row[lang]}</td>
                      <td className="num">{d.kl}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="callout">
              <strong>{zh ? "这是速度,不是渐近精度。" : "This is speed, not asymptotic accuracy."}</strong>{" "}
              {zh ? (
                <>
                  在<strong>同一份预算</strong>下,学出来的规则大约近 19 倍。<strong>但给 detailed balance
                  13 倍的步数,它反过来大约近 6 倍。</strong>所以这条结果只能被读成「更快」,不能被读成
                  「更准」;学出来的规则在渐近上是输的。种子散布也不小:均值 0.067 上带着 0.091。
                </>
              ) : (
                <>
                  At <strong>matched budget</strong> the learned rule is about 19 times closer.{" "}
                  <strong>Given 13 times more steps, detailed balance is about 6 times closer than the
                  rule.</strong> So this result reads as faster, never as more accurate; asymptotically the
                  learned rule loses. The seed spread is not small either: 0.091 on a mean of 0.067.
                </>
              )}
            </div>
            <p>
              {zh ? (
                <>
                  路上还量到了基线的一个性质:同一个 detailed-balance 损失,随着加权从全支撑走到纯 on-policy,
                  落点从 0.010 一路散到 2.535;纯 on-policy 加权时它把<strong>自己的</strong>残差压到 1e-4,
                  却停在 KL 0.92 —— 因为策略放弃掉的那些状态永远不会被纠正。规则和基线用的是同一套加权,
                  所以这个比较比的是更新,不是探索。
                </>
              ) : (
                <>
                  One property of the baseline turned up on the way: the same detailed-balance loss lands anywhere
                  from 0.010 to 2.535 as its weighting goes from full support to purely on-policy. Weighted purely
                  on-policy it drives <strong>its own</strong> residual to 1e-4 while sitting at KL 0.92, because
                  states the policy abandons are never corrected. Rule and baseline are given the same weighting,
                  so the comparison is about the update and not about exploration.
                </>
              )}
            </p>

            <div className="card">
              <h4>
                {zh ? "必须一起说的边界" : "Boundaries that have to be stated with it"} <Chip p="mine" />
              </h4>
              <ul style={{ margin: 0, color: "var(--text-2)", fontSize: 15 }}>
                {LIMITS.map((l) => (
                  <li key={l.en}>{l[lang]}</li>
                ))}
              </ul>
            </div>
          </div>
        </Section>

        <Section
          id="next"
          num="13"
          title={{ zh: "四条拓展", en: "Four extensions" }}
          kicker={{
            zh: "按「能不能写成论文」排序。D 最便宜,今晚就能试。",
            en: "Ordered by how readily each becomes a paper. D is the cheap one — tonight.",
          }}
        >
          {EXTS.map((e) => (
            <div className="card" key={e.tag}>
              <h4>
                <span style={{ fontFamily: "var(--mono)", color: "var(--text-3)" }}>{e.tag}</span>
                {e.title[lang]}
                <Chip p={e.prov} />
              </h4>
              <div style={{ color: "var(--text-2)", fontSize: 15.5 }}>{e.body[lang]}</div>
              <p style={{ margin: "14px 0 0", fontSize: 12.5, color: "var(--text-3)", fontFamily: "var(--mono)" }}>
                {e.cost[lang]}
              </p>
            </div>
          ))}
          <div className="body">
            <h3>{zh ? "拓展 A 也已经跑了" : "Extension A has been run too"}</h3>
            <p>
              {zh ? (
                <>
                  第 12 节就是拓展 A 被执行的样子:元学习机器原样搬过来,只把打分的标量换掉 ——
                  换成的是 V₁ = log Z − KL,不是这里写的模式数。
                </>
              ) : (
                <>
                  Section 12 is extension A carried out: the machinery ported as-is, with only the scalar that
                  scores the rule swapped — for V₁ = log Z − KL rather than for the mode count written here.
                </>
              )}
            </p>

            <h3>{zh ? "拓展 D 已经跑了" : "Extension D has been run"}</h3>
            <p>
              {zh ? (
                <>
                  两条臂共用同一个策略网络、同一个优化器、同一批种子、同一套评估点,<strong>只有 flow
                  读出头不同</strong>:标量线性输出,对固定 support 上 51 个 bin 的 softmax 期望。
                  KL 由 DAG 上的动态规划<strong>精确</strong>算出,不是采样估计。
                </>
              ) : (
                <>
                  Both arms share the policy network, the optimiser, the seeds and the evaluation schedule;{" "}
                  <strong>only the flow readout differs</strong> — a scalar linear output against a softmax
                  expectation over 51 bins on a fixed support. KL is computed <strong>exactly</strong> by
                  dynamic programming over the DAG, never sampled.
                </>
              )}
            </p>
          </div>
          <ExtDResults />
        </Section>

        {/* ------------------------------------------------------------- 14 */}
        <Section
          id="ledger"
          num="14"
          title={{ zh: "证据分层", en: "Evidence ledger" }}
          kicker={{
            zh: "把已发表结果、源码事实、我的综合和纯猜想分开摆。混在一起才是问题。",
            en: "Published results, source facts, my synthesis and pure conjecture kept apart. Blending them is the failure mode.",
          }}
        >
          <div className="scroller">
            <table className="tbl">
              <thead>
                <tr>
                  <th style={{ width: "13%" }}>{zh ? "层级" : "Tier"}</th>
                  <th style={{ width: "46%" }}>{zh ? "断言" : "Claim"}</th>
                  <th style={{ width: "41%" }}>{zh ? "凭什么" : "On what basis"}</th>
                </tr>
              </thead>
              <tbody>
                {LEDGER.map((r, k) => (
                  <tr key={k}>
                    <td>
                      <Chip p={r.prov} />
                    </td>
                    <td>{r.claim[lang]}</td>
                    <td className="num" style={{ whiteSpace: "normal" }}>
                      {r.how[lang]}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        {/* ------------------------------------------------------------- 15 */}
        <Section
          id="repro"
          num="15"
          title={{ zh: "复现", en: "Reproduce" }}
          kicker={{
            zh: "先三条命令,第三条保证浏览器里的引擎和 Python oracle 逐位一致。第 08 到 12 节的脚本跟在后面。",
            en: "Three commands first; the third guarantees the browser engine agrees with the Python oracle bit for bit. The scripts behind sections 08 to 12 follow.",
          }}
        >
          <div className="body">
            <pre>
              <code>{`git clone ${REPO_URL}
cd discorl-gfn-bridge

python3 research/cumulants.py --json research/cumulants.json
python3 research/verify_disco_source.py
node research/parity.ts`}</code>
            </pre>
            <p>
              {zh ? (
                <>
                  <code>cumulants.py</code> 在 8×8 hypergrid 上枚举 {CUMULANTS.n_trajectories.toLocaleString("en-US")}{" "}
                  条轨迹,用动态规划精确算出 W 的前两阶矩与指数平均。
                  <code>verify_disco_source.py</code> 把关于别人代码的断言写成可执行检查,钉在 commit{" "}
                  <code>{DISCO_COMMIT.slice(0, 12)}</code>,CI 每周重跑。
                  <code>parity.ts</code> 逐点比较两套实现 —— 目前 187 项比较 0 偏差。
                </>
              ) : (
                <>
                  <code>cumulants.py</code> enumerates {CUMULANTS.n_trajectories.toLocaleString("en-US")} trajectories
                  on an 8×8 hypergrid and computes the first two moments of W, plus its exponential average,
                  exactly by dynamic programming. <code>verify_disco_source.py</code> turns claims about someone
                  else's code into executable checks pinned to commit{" "}
                  <code>{DISCO_COMMIT.slice(0, 12)}</code>, re-run weekly in CI. <code>parity.ts</code> compares
                  the two implementations point by point — currently 187 comparisons, 0 mismatches.
                </>
              )}
            </p>

            <h3>{zh ? "这条弧线靠的脚本" : "The scripts this arc rests on"}</h3>
            <p>
              {zh ? (
                <>
                  第 08 到 12 节的每一个数字都出自下面某一个脚本,每个都是{" "}
                  <code>python3 research/…</code> 直接跑。它们要几个小时,不要在会话里等。
                </>
              ) : (
                <>
                  Every number in sections 08 to 12 comes from one of these; each runs as{" "}
                  <code>python3 research/…</code>. They take hours, so do not wait on them in a session.
                </>
              )}
            </p>
            <div className="scroller">
              <table className="tbl">
                <thead>
                  <tr>
                    <th style={{ width: "34%" }}>{zh ? "脚本" : "Script"}</th>
                    <th>{zh ? "它交付什么" : "What it delivers"}</th>
                    <th className="num">{zh ? "提交" : "commit"}</th>
                  </tr>
                </thead>
                <tbody>
                  {NEW_SCRIPTS.map((s) => (
                    <tr key={s.file}>
                      <td className="num">{s.file}</td>
                      <td style={{ color: "var(--text-2)" }}>{s.what[lang]}</td>
                      <td className="num" style={{ color: "var(--text-3)" }}>
                        {s.swept ? "e6db278" : ""}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="hint">
              {zh ? (
                <>
                  <b>关于那五个标了 e6db278 的</b>:它们是在另一个并行会话里写的,被一次{" "}
                  <code>git add -A</code> 扫进了那个 commit,而那条 commit message 并没有描述它们。
                  该 commit 上有一条 <code>git note</code> 记着这件事。第 11 节整节压在它们上面。
                </>
              ) : (
                <>
                  <b>About the five marked e6db278</b>: they were written in a concurrent session and swept into
                  that commit by a <code>git add -A</code> whose message does not describe them. A{" "}
                  <code>git note</code> on the commit records this. Section 11 rests on them entirely.
                </>
              )}
            </p>

            <h3>{zh ? "参考文献" : "References"}</h3>
            <ol className="refs">
              {REFS.map((r) => (
                <li key={r.id}>
                  <a href={r.url} target="_blank" rel="noreferrer">
                    {r.cite}
                  </a>
                  <br />
                  <span style={{ color: "var(--text-3)", fontSize: 13.5 }}>
                    <T v={r.what} />
                  </span>
                </li>
              ))}
            </ol>
          </div>

          <footer>
            <span>W(τ) · DiscoRL ↔ GFlowNet</span>
            <span>
              {zh ? "本机" : "machine"}: Apple M1 Pro · {zh ? "无 GPU" : "no GPU"}
            </span>
          </footer>
        </Section>
      </main>
    </div>
  );
}
