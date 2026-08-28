/* ============================================================================
   Single source of truth.

   Every claim in the app carries a provenance tag:
     verified   — a script in research/ produces this number on this machine
     published  — a peer-reviewed result, cited below
     source     — read directly out of google-deepmind/disco_rl at a pinned SHA
     mine       — my synthesis; mathematically checkable, not published as such
     conjecture — not yet tested; the experiment that would kill it is stated
   ========================================================================= */

import type { Prov } from "../components/ui";
import type { L, LN } from "../i18n";

import cumulants from "./cumulants.json";
import calibJson from "./calibrate.json";
import probeJson from "./disco_probe.json";

const PROBE_RATIO = probeJson.ratio_to_null;
const PROBE_LOCALITY = probeJson.arms[0].locality;
const CALIB_FLOW_R2 = calibJson.phi_y_logF.r2;
const CALIB_FLOW_SP = calibJson.phi_y_logF.spearman;
const CALIB_Q_R2 = calibJson.control.q_value_r2;

export const DISCO_REPO = "google-deepmind/disco_rl";
export const DISCO_COMMIT = "9059a29f7121d60948f25ef165e08e050e9399c8";

/** Numbers lifted straight out of the verification run. */
const byName = (n: string) => cumulants.policies.find((p) => p.policy === n)!;
export const GFN_ROW = byName("flow-matching (GFN)");
export const RL_ROW = byName("soft RL (MaxEnt)");
export const UNIF_ROW = byName("uniform");
export const CUMULANTS = cumulants;

export const NAV: { id: string; num: string; label: L }[] = [
  { id: "why", num: "01", label: { zh: "问题", en: "The question" } },
  { id: "work", num: "02", label: { zh: "同一个 W(τ)", en: "One W(τ)" } },
  { id: "dial", num: "03", label: { zh: "动手:累积量", en: "Rig: cumulants" } },
  { id: "where", num: "04", label: { zh: "DiscoRL 的落点", en: "Where DiscoRL lands" } },
  { id: "gamma", num: "05", label: { zh: "动手:γ", en: "Rig: γ" } },
  { id: "test", num: "06", label: { zh: "测了:β 探针", en: "Tested: the β probe" } },
  { id: "kappa", num: "07", label: { zh: "障碍:κ", en: "The obstruction: κ" } },
  { id: "next", num: "08", label: { zh: "四条拓展 + D 的结果", en: "Extensions, and D" } },
  { id: "ledger", num: "09", label: { zh: "证据分层", en: "Evidence ledger" } },
  { id: "repro", num: "10", label: { zh: "复现", en: "Reproduce" } },
];

/* ------------------------------------------------------------- references */

export type Ref = { id: string; cite: string; what: L; url: string };

export const REFS: Ref[] = [
  {
    id: "oh2025",
    cite: "Oh, Farquhar, Kemaev, Calian, Hessel, Zintgraf, Singh, van Hasselt, Silver — Nature (2025)",
    what: {
      zh: "DiscoRL:用元网络定义损失函数,在上百个并行 agent 上元学习出的更新规则。代码与 Disco103 权重以 Apache-2.0 公开。",
      en: "DiscoRL: the loss function is itself a neural network, meta-learned across hundreds of parallel agents. Code and the Disco103 meta-parameters are public under Apache-2.0.",
    },
    url: "https://doi.org/10.1038/s41586-025-09761-x",
  },
  {
    id: "tiapkin2024",
    cite: "Tiapkin, Morozov, Naumov, Vetrov — AISTATS 2024 (Oral)",
    what: {
      zh: "GFlowNet 训练可精确重写为带特定 reward 与正则结构的熵正则化 RL。",
      en: "GFlowNet training can be rewritten exactly as entropy-regularised RL with a specific reward and regulariser.",
    },
    url: "https://arxiv.org/abs/2310.12934",
  },
  {
    id: "mohammadpour2024",
    cite: "Mohammadpour, Bengio, Frejinger, Bacon — AISTATS 2024",
    what: {
      zh: "构造出合适的 reward,给出 GFN 与最大熵 RL 的精确关系;并指出结果依赖 p_B 的选取。",
      en: "An exact relationship between GFNs and MaxEnt RL via a constructed reward — and it depends on the choice of p_B.",
    },
    url: "https://arxiv.org/abs/2312.14331",
  },
  {
    id: "deleu2024",
    cite: "Deleu, Nouri, Malkin, Precup, Bengio — 2024",
    what: {
      zh: "当同一个对象存在多条生成路径时,MaxEnt RL 诱导的分布有偏。这是本文 γ 实验的直接依据。",
      en: "When an object has several generating paths, the distribution induced by MaxEnt RL is biased. This is what the γ experiment measures.",
    },
    url: "https://arxiv.org/abs/2402.10309",
  },
  {
    id: "malkin2023",
    cite: "Malkin, Lahlou, Deleu, Ji, Hu, Everett, Zhang, Bengio — ICLR 2023",
    what: {
      zh: "GFlowNet 与变分推断的桥;并观察到 TB 比 VI 更适合 off-policy 训练。",
      en: "The bridge between GFlowNets and variational inference — and the observation that TB tolerates off-policy training better than VI.",
    },
    url: "https://arxiv.org/abs/2210.00580",
  },
  {
    id: "chertkov2025",
    cite: "Chertkov, Behjoo, Ahn — arXiv:2503.14549 (2025)",
    what: {
      zh: "同一个对象等价于 Doob h-变换、KL-最优控制器、单边 Schrödinger 输运,以及 GFlowNet 式生成的理想 flow 函数。desirability 的后向递归是线性的 —— 这是 GFN 能用回归训练而不需要 max 算子的结构性原因。",
      en: "One object with several faces: a Doob h-transform, a KL-optimal controller, a one-sided Schrödinger transport, and the ideal flow function for GFlowNet-type generation. The desirability recursion is linear — the structural reason GFNs train by regression rather than by a max operator.",
    },
    url: "https://arxiv.org/abs/2503.14549",
  },
  {
    id: "albergo2024",
    cite: "Albergo, Vanden-Eijnden — arXiv:2410.02711 (2024)",
    what: {
      zh: "NETS:显式基于 Jarzynski 等式的非平衡输运采样器,退火重要性采样加一个学出来的 drift。非平衡采样这条线是活的。",
      en: "NETS: a non-equilibrium transport sampler built explicitly on Jarzynski's equality — annealed importance sampling plus a learned drift. The non-equilibrium sampling line is very much alive.",
    },
    url: "https://arxiv.org/abs/2410.02711",
  },
  {
    id: "brunswic2024",
    cite: "Brunswic, Li, Xu, Jui, Ma — AAAI 2024",
    what: {
      zh: "证明 R-流的集合是由圈空间 H¹(G) 方向的仿射子空间。这**预先占有**了本仓库 research/gauge.py 里的结构结果 —— 我们只多算了它的维数（第一 Betti 数）。更强形式他们归于 Kalpazidou (2007)。",
      en: "Proves the set of R-flows is an affine subspace directed by the cycle space H¹(G). This **pre-empts** the structural result in research/gauge.py; all we added was its dimension, the first Betti number. They attribute a stronger form to Kalpazidou (2007).",
    },
    url: "https://arxiv.org/abs/2312.15246",
  },
  {
    id: "malkin2022tb",
    cite: "Malkin, Jain, Bengio, Sun, Bengio — NeurIPS 2022",
    what: {
      zh: "Trajectory Balance。§3.1 已经陈述任意 p_B 给出同一个 p(x) ∝ R(x)，并且**已经点名**底层无向图的圈是这个多重性的来源。",
      en: "Trajectory balance. Section 3.1 already states that every backward policy yields the same p(x) ∝ R(x), and already names cycles in the underlying undirected graph as the source of that multiplicity.",
    },
    url: "https://arxiv.org/abs/2201.13259",
  },
  {
    id: "kawai2007",
    cite: "Kawai, Parrondo, Van den Broeck — Phys. Rev. Lett. 98, 080602 (2007)",
    what: {
      zh: "耗散功的平均等于前向与反向路径测度之间的相对熵。本文第 2 层的物理来源。",
      en: "Mean dissipated work equals the relative entropy between forward and reverse path measures. The physics behind layer 2.",
    },
    url: "https://doi.org/10.1103/PhysRevLett.98.080602",
  },
  {
    id: "jarzynski1997",
    cite: "Jarzynski — Phys. Rev. Lett. 78, 2690 (1997)",
    what: {
      zh: "非平衡功的指数平均给出平衡自由能差,与所走协议无关。",
      en: "The exponential average of non-equilibrium work gives the equilibrium free-energy difference, whatever protocol you take.",
    },
    url: "https://doi.org/10.1103/PhysRevLett.78.2690",
  },
];

/* ------------------------------------------------------------- extensions */

export type Ext = {
  tag: string;
  title: L;
  body: LN;
  cost: L;
  prov: Prov;
};

export const EXTS: Ext[] = [
  {
    tag: "A",
    title: { zh: "把平衡条件本身元学习掉", en: "Meta-learn the balance condition itself" },
    body: {
      zh: (
        <>
          GFlowNet 圈现在靠人手在 DB / TB / SubTB(λ) / FL-DB 之间挑一个 ——{" "}
          <strong>这正是 DiscoRL 废掉的那种「人类手搓更新规则」</strong>。把 DiscoRL 的元学习机器原样搬过来,
          只补上父节点输入、把元目标从回报换成<strong>发现的模式数</strong>或采样器 TV 误差。
        </>
      ),
      en: (
        <>
          The GFlowNet literature still picks by hand among DB / TB / SubTB(λ) / FL-DB —{" "}
          <strong>exactly the hand-crafted update rule DiscoRL abolished</strong>. Port the machinery as-is,
          add the parent-set input, and swap the meta-objective from return to <strong>modes discovered</strong>{" "}
          or sampler TV error.
        </>
      ),
    },
    cost: { zh: "改动小,机器现成", en: "Small diff; the machinery already exists" },
    prov: "mine",
  },
  {
    tag: "B",
    title: { zh: "让 α 变成协议,而不是一次选择", en: "Make α a protocol, not a one-off choice" },
    body: {
      zh: (
        <>
          用累积量族参数化损失,在 ELBO(一阶)和 TB(二阶)之间连续插值,并让<strong>生命周期 MetaLSTM 输出 α</strong>:
          早期一阶(梯度稳),后期二阶(off-policy 稳)。物理读法是在「最小平均耗散」与「最小耗散涨落」之间退火。
        </>
      ),
      en: (
        <>
          Parametrise the loss by cumulant order, interpolating ELBO (first) to TB (second), and let the{" "}
          <strong>lifetime MetaLSTM emit α</strong>: first-order early for gradient stability, second-order late
          for off-policy stability. Physically, anneal between minimising mean dissipation and minimising its
          fluctuations.
        </>
      ),
    },
    cost: { zh: "把两选一变成一条曲线", en: "Turns a binary choice into a schedule" },
    prov: "mine",
  },
  {
    tag: "C",
    title: { zh: "学习型 p_B = 最优反向协议", en: "A learned p_B is an optimal reverse protocol" },
    body: {
      zh: (
        <>
          最大熵 GFN 依赖 p_B 的选取;非平衡物理里「最优协议最小化耗散」是个成熟问题。把 p_B 也交给元网络、
          元目标直接取 <strong>Var[W]</strong>,一步把这两件事缝成同一个优化问题。
        </>
      ),
      en: (
        <>
          Maximum-entropy GFNs depend on the choice of p_B; "optimal protocols minimise dissipation" is a mature
          problem in non-equilibrium physics. Hand p_B to the meta-network with <strong>Var[W]</strong> as the
          meta-objective and the two become one optimisation.
        </>
      ),
    },
    cost: { zh: "需要父节点枚举", en: "Needs parent enumeration" },
    prov: "mine",
  },
  {
    tag: "D",
    title: { zh: "反向搬运:log F 头改成 categorical", en: "Port back: make the log F head categorical" },
    body: {
      zh: (
        <>
          GFN 的 log F 现在是<strong>标量回归</strong>,尺度跨几十个数量级,数值条件出了名的差。DiscoRL
          规模化成功用的是 categorical + KL(<code>disco.py:245–247</code>)。换成固定 support 上的 two-hot 分布头,
          是现代 value learning 里反复验证过的修法。
        </>
      ),
      en: (
        <>
          GFN's log F is a <strong>scalar regression</strong> spanning tens of orders of magnitude — notoriously
          ill-conditioned. What scaled for DiscoRL was categorical + KL (<code>disco.py:245–247</code>). A
          two-hot head on a fixed support is a fix modern value learning has validated many times over.
        </>
      ),
    },
    cost: { zh: "一个下午;不需要新理论", en: "An afternoon; no new theory" },
    prov: "mine",
  },
];

/* --------------------------------------------------------- evidence ledger */

export type Row = { prov: Prov; claim: L; how: L };

export const LEDGER: Row[] = [
  {
    prov: "published",
    claim: { zh: "GFlowNet 训练 ≡ 熵正则化 RL", en: "GFlowNet training ≡ entropy-regularised RL" },
    how: { zh: "Tiapkin 2024 · Mohammadpour 2024", en: "Tiapkin 2024 · Mohammadpour 2024" },
  },
  {
    prov: "published",
    claim: { zh: "多路径时 MaxEnt RL 分布有偏", en: "MaxEnt RL is biased under multiple paths" },
    how: { zh: "Deleu 2024", en: "Deleu 2024" },
  },
  {
    prov: "published",
    claim: { zh: "平均耗散功 = 前向/反向路径测度的相对熵", en: "Mean dissipated work = relative entropy of the path measures" },
    how: { zh: "Kawai–Parrondo–Van den Broeck 2007", en: "Kawai–Parrondo–Van den Broeck 2007" },
  },
  {
    prov: "published",
    claim: {
      zh: "流空间是圈空间方向的仿射子空间 —— 我们以为是新结果，其实已发表",
      en: "The flow space is an affine subspace directed by the cycle space — we thought this was new; it is published",
    },
    how: {
      zh: "Brunswic et al. AAAI 2024 Prop.4/Thm.5；更强形式见 Kalpazidou 2007。我们只多算了维数 |E|−|V|+1。",
      en: "Brunswic et al. AAAI 2024, Prop. 4 / Thm. 5; stronger form in Kalpazidou 2007. We only added the dimension |E|−|V|+1.",
    },
  },
  {
    prov: "mine",
    claim: {
      zh: "GFlowNet 是非平衡稳态,不是平衡态:flow matching 就是 Kirchhoff 电流律",
      en: "A GFlowNet is a non-equilibrium steady state, not an equilibrium: flow matching is Kirchhoff's current law",
    },
    how: {
      zh: "有源有汇的 DAG 上每条边净电流非零;W 因此是 excess（Hatano–Sasa）熵产,不是平衡功",
      en: "Every edge of a DAG with a source and sinks carries net current; W is therefore the excess (Hatano–Sasa) entropy production, not equilibrium work",
    },
  },
  {
    prov: "verified",
    claim: { zh: "E[e^−W] = Z,对任意前向策略成立", en: "E[e^−W] = Z for any forward policy" },
    how: {
      zh: `research/cumulants.py — 三个策略的相对误差 ≤ ${GFN_ROW.jarzynski_rel_err.toExponential(1)}`,
      en: `research/cumulants.py — relative error ≤ ${GFN_ROW.jarzynski_rel_err.toExponential(1)} across all three policies`,
    },
  },
  {
    prov: "verified",
    claim: { zh: "TB 的最优值 = Var[W],其自由参数 = ELBO", en: "The TB optimum is Var[W]; its free parameter is the ELBO" },
    how: {
      zh: `research/cumulants.py — argmin_c = ${cumulants.c3_argmin.toFixed(5)} vs −E[W] = ${(-UNIF_ROW.mean_W).toFixed(5)}`,
      en: `research/cumulants.py — argmin_c = ${cumulants.c3_argmin.toFixed(5)} vs −E[W] = ${(-UNIF_ROW.mean_W).toFixed(5)}`,
    },
  },
  {
    prov: "verified",
    claim: { zh: "流匹配策略下 W 恒定,Var[W] = 0", en: "Under flow matching W is constant, Var[W] = 0" },
    how: {
      zh: `|Var[W]| = ${Math.abs(GFN_ROW.var_W).toExponential(1)}（浮点噪声）,E[W] = −log Z`,
      en: `|Var[W]| = ${Math.abs(GFN_ROW.var_W).toExponential(1)} (float noise), E[W] = −log Z`,
    },
  },
  {
    prov: "source",
    claim: { zh: "轨迹内递归反向展开,注释写明用于 bootstrapping", en: "The per-trajectory recurrence runs backwards, for bootstrapping" },
    how: { zh: "meta_nets.py:109,116", en: "meta_nets.py:109,116" },
  },
  {
    prov: "source",
    claim: { zh: "π、y、z 三项损失全是 categorical KL", en: "π, y and z are all trained by categorical KL" },
    how: { zh: "disco.py:245–247", en: "disco.py:245–247" },
  },
  {
    prov: "source",
    claim: { zh: "元网络从不接收 p_B 或父节点集合", en: "The meta-network never receives p_B or a parent set" },
    how: { zh: "disco.py — 0 匹配 (S4)", en: "disco.py — zero matches (S4)" },
  },
  {
    prov: "mine",
    claim: { zh: "VI = 一阶累积量,TB = 二阶累积量,DiscoRL = 元学习这个泛函", en: "VI = first cumulant, TB = second, DiscoRL meta-learns the functional" },
    how: {
      zh: "数学可验(见第 02 节);arXiv 摘要中 GFlowNet 与 Jarzynski / fluctuation theorem 共现数为 0",
      en: "Checkable (section 02); zero arXiv records pair GFlowNet with Jarzynski or fluctuation theorem",
    },
  },
  {
    prov: "mine",
    claim: { zh: "DB 的四个充分统计量已在元网络输入总线上", en: "Four of DB's five sufficient statistics are already on the input bus" },
    how: { zh: "disco.py:336–393 的逐项对照", en: "Item-by-item against disco.py:336–393" },
  },
  {
    prov: "mine",
    claim: {
      zh: "原 γ 实验设计有缺陷:DiscoRL 最大化回报,p(x) 塌成点质量,γ 无定义",
      en: "The original γ experiment was ill-posed: DiscoRL maximises return, p(x) degenerates, γ is undefined",
    },
    how: { zh: "写 harness 时发现,已由 β 探针替换", en: "Found while writing the harness; replaced by the β probe" },
  },
  {
    prov: "verified",
    claim: {
      zh: "Disco103 的 y 目标对所走动作的 log 概率有真实且局部的响应",
      en: "Disco103's y-target responds to the log-probability of the action taken, and does so locally",
    },
    how: {
      zh: `research/disco_probe.py — |β| 是随机初始化的 ${Math.round(PROBE_RATIO)} 倍，局部性 ${PROBE_LOCALITY.toFixed(0)}×`,
      en: `research/disco_probe.py — |β| is ${Math.round(PROBE_RATIO)}× the random-init null, locality ${PROBE_LOCALITY.toFixed(0)}×`,
    },
  },
  {
    prov: "verified",
    claim: {
      zh: "φ(y) 不追踪精确 log F，也不追踪 value —— y 通道什么都不追踪",
      en: "φ(y) tracks neither the exact log F nor a value function — the y channel tracks nothing",
    },
    how: {
      zh: `research/calibrate.py — R² = ${CALIB_FLOW_R2.toFixed(3)}，秩相关 ${CALIB_FLOW_SP.toFixed(2)}；阳性对照 q 头对 V_π 是 R² = ${CALIB_Q_R2.toFixed(2)}`,
      en: `research/calibrate.py — R² = ${CALIB_FLOW_R2.toFixed(3)}, rank ${CALIB_FLOW_SP.toFixed(2)}; the q-head control reaches R² = ${CALIB_Q_R2.toFixed(2)} against V_π`,
    },
  },
  {
    prov: "verified",
    claim: {
      zh: "value 语义在 z 通道，不在 y",
      en: "The value semantics live in the z channel, not in y",
    },
    how: { zh: "φ(z) 对 V_π 的秩相关 −0.72", en: "φ(z) against V_π, rank correlation −0.72" },
  },
  {
    prov: "verified",
    claim: {
      zh: "DiscoRL 的更新不是任何泛函的梯度:κ = 0.709 ≠ 1",
      en: "DiscoRL's update is not the gradient of any functional: κ = 0.709 ≠ 1",
    },
    how: {
      zh: "research/kappa.py — 估计量对真梯度精确返回 1.0000,合成场全域误差 ≤ 0.0034",
      en: "research/kappa.py — the estimator returns exactly 1.0000 for true gradients; worst synthetic error 0.0034",
    },
  },
  {
    prov: "verified",
    claim: {
      zh: "但也没有比因果性强制的更不保守:κ 落在 1/√2 的结构地板上",
      en: "Nor is it less conservative than causality forces: κ sits on the 1/√2 structural floor",
    },
    how: {
      zh: "偏离 +0.0017,小于估计量在该邻域已证实的系统偏差 0.0043",
      en: "Off by +0.0017, smaller than the estimator's demonstrated 0.0043 bias in that band",
    },
  },
  {
    prov: "mine",
    claim: {
      zh: "部分 mapping = Hodge 投影,份额 κ² = 0.50 —— 恰好一半",
      en: "The partial mapping is the Hodge projection, and its share is κ² = 0.50 — exactly one half",
    },
    how: {
      zh: "自举算子 Frobenius 质量的一半可写成梯度,另一半结构上不能",
      en: "Half the bootstrap operator's Frobenius mass is expressible as a gradient; the other half structurally is not",
    },
  },
  {
    prov: "mine",
    claim: {
      zh: "因此 |β/α| 的 detailed-balance 读法已撤回",
      en: "The detailed-balance reading of |β/α| is therefore withdrawn",
    },
    how: {
      zh: "比值预设 φ(y) 是 log-flow 量；校准证伪了该前提。β 本身仍成立。",
      en: "The ratio presumed φ(y) is a log-flow quantity; calibration refuted that. β itself stands.",
    },
  },
  {
    prov: "verified",
    claim: {
      zh: "拓展 D 部分成立:categorical flow 头收敛更快、梯度尾更紧,但渐近值不更好",
      en: "Extension D partly holds: the categorical flow head converges faster with a tighter gradient tail, but no better asymptote",
    },
    how: {
      zh: "research/logf_head.py — 18/20 检查点领先，噪声底之上 11 点几何均值 1.41×，梯度 p99 1.13 vs 1.56",
      en: "research/logf_head.py — leads 18/20 checkpoints; 1.41x geometric mean over the 11 above the floor; grad p99 1.13 vs 1.56",
    },
  },
];

/* ------------------------------------------------------------ arXiv audit */

export const ARXIV_AUDIT: { q: string; n: number }[] = [
  { q: 'abs:"GFlowNet"', n: 201 },
  { q: 'abs:"GFlowNet" AND abs:"Jarzynski"', n: 0 },
  { q: 'all:"GFlowNet" AND all:"Jarzynski"', n: 0 },
  { q: 'all:"GFlowNet" AND all:"fluctuation theorem"', n: 0 },
];
