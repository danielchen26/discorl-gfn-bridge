import { useEffect, useState } from "react";

import GammaLab from "./components/GammaLab";
import SourceMatrix from "./components/SourceMatrix";
import { CorrectionNote, ExtDResults, ProbeResults } from "./components/Results";
import WorkDial from "./components/WorkDial";
import { Chip, M, Section, T } from "./components/ui";
import {
  ARXIV_AUDIT,
  CUMULANTS,
  DISCO_COMMIT,
  DISCO_REPO,
  EXTS,
  GFN_ROW,
  LEDGER,
  NAV,
  REFS,
  RL_ROW,
} from "./data/facts";
import probeJson from "./data/disco_probe.json";
import type { Lang } from "./i18n";

const PROBE = probeJson.arms[0];

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
              <span className="v">{PROBE.beta_over_alpha.toFixed(2)}</span>
              <span className="k">
                {zh
                  ? "Disco103 的 |β/α| — DB 要求 1，纯 value 要求 0"
                  : "Disco103's |β/α| — detailed balance wants 1, a pure value rule wants 0"}
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
                ? "第一行是对照组,证明检索式本身有效。后三行为零,说明这个命名下的框架至少未被索引。"
                : "The first row is the control, proving the query works at all. The remaining zeros say the framing is at least unindexed under these names."}
            </p>
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
          title={{ zh: "把猜想真的测了", en: "The claim, actually tested" }}
          kicker={{
            zh: "Disco103 的权重是公开的,所以猜想不必停在猜想。结果是「部分」——两个方向的强命题都不成立。",
            en: "The Disco103 weights are public, so the conjecture did not have to stay one. The answer came back partial: neither strong reading survives.",
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
        </Section>
        <Section
          id="next"
          num="07"
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

        {/* ------------------------------------------------------------- 08 */}
        <Section
          id="ledger"
          num="08"
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

        {/* ------------------------------------------------------------- 09 */}
        <Section
          id="repro"
          num="09"
          title={{ zh: "复现", en: "Reproduce" }}
          kicker={{
            zh: "三条命令。第三条保证浏览器里的引擎和 Python oracle 逐位一致。",
            en: "Three commands. The third guarantees the browser engine agrees with the Python oracle bit for bit.",
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
