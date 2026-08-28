# W(τ) · DiscoRL ↔ GFlowNet

**它们最小化的是同一个随机变量的不同累积量。**

DiscoRL 是元学习出来的更新规则([Oh et al., *Nature* 2025](https://doi.org/10.1038/s41586-025-09761-x)),GFlowNet 是手工设计的采样器。表面上毫无关系。但把两者都写到「轨迹的非平衡功 $W(\tau)$」这一层,变分推断、GFlowNet 和 DiscoRL 就落在同一条累积量阶梯的不同台阶上。

交互版:**https://danielchen26.github.io/discorl-gfn-bridge/**

这个 repo 里的每一个数字都由 `research/` 下的脚本产生,并且在 CI 里重跑。没有"据我所知"。

---

## 目录

- [0. 这份文档的认识论边界](#0-这份文档的认识论边界)
- [1. 第一层:同一个数学对象 W(τ)](#1-第一层同一个数学对象-wτ)
- [2. 第二层:DiscoRL 落在这个空间的哪个点](#2-第二层discorl-落在这个空间的哪个点)
- [3. 第三层:一个可证伪的猜想](#3-第三层一个可证伪的猜想)
- [4. 第四层:四条拓展](#4-第四层四条拓展)
- [5. 证据分层](#5-证据分层)
- [6. 复现](#6-复现)
- [7. 参考文献](#7-参考文献)

---

## 0. 这份文档的认识论边界

四个层级,严格分开摆:

| 标记 | 含义 |
|---|---|
| **已发表** | 同行评议过的结果,附引文 |
| **源码核实** | 直接读 `google-deepmind/disco_rl` 在 pinned commit `9059a29f` 上的代码 |
| **本机跑过** | `research/` 下的脚本在这台机器上产生的数字(Apple M1 Pro,无 GPU) |
| **我的综合** | 数学上可验证,但我没有找到以这个名字发表过的说法 |
| **猜想** | 没测过。证伪条件已写明 |

混在一起才是问题。第 5 节是完整的对照表。

---

## 1. 第一层:同一个数学对象 W(τ)

这一节没有类比。三条都是**精确等式**,并且已经跑过。

### 设定

DAG 上源点 $s_0$、终态 $x$、奖励 $R(x)>0$、$Z=\sum_x R(x)$。前向策略给出路径测度 $P_F$,反向策略 $p_B$ 给出参考测度。定义**功**:

$$W(\tau)\;\triangleq\;\log\frac{\prod_t p_F(s_{t+1}\mid s_t)}{R(x)\prod_t p_B(s_t\mid s_{t+1})}\;=\;\log\frac{P_F(\tau)}{Z\,P_B(\tau)}$$

### ① Jarzynski 等式 —— 对任意前向策略成立

$$\mathbb E_{P_F}\!\left[e^{-W}\right]=\sum_\tau P_F(\tau)\frac{R(x)\prod p_B}{P_F(\tau)}=\sum_x R(x)\underbrace{\sum_{\tau\to x}\textstyle\prod p_B}_{=1}=Z$$

注意它**不依赖策略好坏**。`research/cumulants.py` 在三个截然不同的策略(流匹配、软 RL、均匀)上都得到相对误差 $\le 4.3\times10^{-16}$ —— 这是恒等式,不是拟合。

### ② 第二定律 = ELBO 缺口

$$\langle W_{\rm diss}\rangle=\mathbb E[W]+\log Z=D_{\rm KL}\big(P_F\,\|\,P_B\big)\;\ge\;0$$

这正是 Kawai–Parrondo–Van den Broeck 的「耗散 = 相对熵」([PRL 98, 080602, 2007](https://doi.org/10.1103/PhysRevLett.98.080602))。变分推断做的 $\max$ ELBO,就是 $\min\mathbb E[W]$ —— **一阶累积量**。

### ③ Trajectory Balance = 二阶累积量

TB 损失展开后字面上就是

$$\mathcal L_{\rm TB}(\tau)=\left(\log\frac{Z_\theta\prod p_F}{R(x)\prod p_B}\right)^2=\big(\log Z_\theta+W(\tau)\big)^2$$

对自由参数 $\log Z_\theta$ 取极小:最优点 $\log Z_\theta^\star=-\mathbb E[W]=\text{ELBO}$,残差 $=\operatorname{Var}[W]$。

于是整件事塌缩成一行 —— **累积量展开**:

$$\log Z=\log\mathbb E\!\left[e^{-W}\right]=-\langle W\rangle+\tfrac12\operatorname{Var}(W)-\tfrac16\kappa_3+\cdots$$

| | 对 $W$ 做什么 | 物理 |
|---|---|---|
| VI / MaxEnt RL | $\min$ 一阶累积量 $\langle W\rangle$ | 最小平均耗散 |
| **GFlowNet (TB)** | $\min$ **二阶**累积量 $\operatorname{Var}(W)$,一阶自动等于 ELBO | 逼近准静态可逆 |
| DiscoRL | **元学习**这个泛函本身 | 协议由回报选出来 |

$\mathcal L_{\rm TB}=0\iff\operatorname{Var}W=0\iff W\equiv-\log Z\iff$ 零耗散 $\iff$ 精确采样器。
**GFN 训练的本质是把一个不可逆抽样过程压成可逆过程。**

顺带解释了 Malkin et al. 观察到的「TB 比 VI 更适合 off-policy」:方差目标不依赖采样分布的一阶矩,自然对 behaviour policy 不敏感。

### 本机验证结果

`python3 research/cumulants.py --height 8`,8×8 hypergrid,12,869 条轨迹,全枚举精确动态规划:

| 策略 | $\mathbb E[W]$ | $\operatorname{Var}[W]$ | $\mathbb E[e^{-W}]$ 相对误差 | 拟合 $\gamma$ | KL$(p\|R/Z)$ |
|---|---|---|---|---|---|
| 流匹配 (GFN) | −2.811809435393 | −1.8e−15 | 4.3e−16 | −0.0000 | 2.4e−16 |
| 软 RL (MaxEnt) | −1.339140446662 | 0.907485 | 4.3e−16 | +0.9788 | 1.460200 |
| 均匀 | −0.610885003200 | 1.979472 | 2.1e−16 | +1.5129 | 2.152027 |

- $\log Z = 2.8118094353930627$。流匹配下 $\mathbb E[W]=-\log Z$ **恰好**,$\operatorname{Var}[W]$ 落在浮点噪声上。
- ③ 的直接检验:暴力搜 $\arg\min_c\mathbb E[(c+W)^2]=+0.61089$,与 $-\mathbb E[W]=+0.61089$ 一致;最小值 $1.979471588446$ 对 $\operatorname{Var}[W]=1.979471588421$。

### 这个框架有人写过吗

三块拼图都已发表,但拼在一起的说法我没找到。arXiv 检索(2026-08-28,`all:` 覆盖标题/摘要/comments,不含全文):

| 检索式 | 命中 |
|---|---|
| `abs:"GFlowNet"` | **201**(对照组,证明检索式有效) |
| `abs:"GFlowNet" AND abs:"Jarzynski"` | 0 |
| `all:"GFlowNet" AND all:"Jarzynski"` | 0 |
| `all:"GFlowNet" AND all:"fluctuation theorem"` | 0 |

---

## 2. 第二层:DiscoRL 落在这个空间的哪个点

关键问题只有一个:**它的元网络的假设空间,到底装不装得下 detailed balance?**

这不能靠读论文回答,得读代码。以下全部对着 `google-deepmind/disco_rl` 的 commit `9059a29f7121d60948f25ef165e08e050e9399c8`。

### DiscoRL 实际在做什么(源码核实)

```python
# disco_rl/update_rules/disco.py:245-247
pi_loss_per_step = rlax.categorical_kl_divergence(pi_hat, logits)
y_loss_per_step  = rlax.categorical_kl_divergence(y_hat,  y)
z_loss_per_step  = rlax.categorical_kl_divergence(z_hat,  z_a)
```

$$\mathcal L_\theta=c_\pi\mathrm{KL}(\hat\pi\|\pi_\theta)+c_y\mathrm{KL}(\hat y\|y_\theta)+c_z\mathrm{KL}(\hat z\|z_\theta(\cdot,a))+c_{\rm aux}\mathrm{KL}(\cdot)$$

三项**全是 categorical KL** —— $y,z$ 是**分布**不是标量。这一点后面的拓展 D 会用到。

### 充分统计量对照

DB 残差需要四个量:$\log F(s),\log F(s'),\log p_F(s'|s),\log p_B(s|s')$。元网络实际收到:

| DB 残差需要 | 元网络实际收到 | 位置 | |
|---|---|---|---|
| $F(s), F(s')$ | `agent_out/y` + **`td_pair`** → $(y_t, y_{t+1})$ | `disco.py:363–369` | ✓ |
| $F(s\to s')$ | `agent_out/z` + `select_a` → $z(s_t,a_t)$ | `disco.py:370–373` | ✓ |
| $\sum_{s'}F(s\to s')$ | `agent_out/z` + **`pi_weighted_avg`** + `td_pair` | `disco.py:374–381` | ✓ |
| $p_F(s'\mid s)$ | `agent_out/logits` + `softmax` + `select_a` | `disco.py:336–343` | ✓ |
| $p_B(s\mid s')$ | **没有。父节点集合从未进入元网络。** | `disco.py` — 0 匹配 | ✗ |

**前四行全部命中。** `td_pair` 这个变换的存在本身就说明设计者要的就是「相邻两步预测的配对」—— 那正是局部平衡条件的充分统计量。所以:

> **GFN 的 detailed balance 目标,是 DiscoRL 搜索空间里一个可达点。** 只要 $\hat y$ 去承担 $\log F$ 的角色,$\mathrm{KL}(\hat y\|y_\theta)$ 项就退化成 DB 残差。

第五行缺失是**真限制**,而且是后面猜想的边界条件:元网络看不到 $s_{t+1}$ 的**其它父节点**,只看到实际走过的一条轨迹。反向 LSTM 的隐状态编码的是**后缀** $(s_t,\dots,s_T)$,不是**兄弟父节点集合**。所以它表示不了一般 DAG 上的学习型 $p_B$。

### 两个时间尺度,分工正确得可疑

```python
# disco_rl/networks/meta_nets.py:109-117
# Unroll the per-trajectory RNN core in reverse direction for bootstrapping.
x, _ = hk.dynamic_unroll(per_trajectory_rnn_core, (x, should_reset_bwd),
                         per_trajectory_rnn_core.initial_state(batch_size=batch_size),
                         reverse=True)
```

- **轨迹内 · `reverse=True`**:$s_T \leftarrow \cdots \leftarrow s_0$。注释写明用途是 bootstrapping —— 而 bootstrapping 的传播方向,与 $p_B$ 的计算方向是同一件事。
- **生命周期 · `MetaLSTM` 正向**:$\lambda(1)\to\lambda(k)\to\lambda(K)$,与轨迹内表示做**乘性交互**(`meta_nets.py:120`)。

Jarzynski / Crooks 要算功,恰好需要两样东西:**一个正向协议 $\lambda(t)$** 和**一个反向路径测度**。DiscoRL 的架构里两样都在,而且分工正确。

### 实证旁证

官方页面报告:discovered predictions "identify important features about upcoming events on moderate time-scales, such as **future policy entropies and large-reward events**"。

而 GFN 的 state flow 在 soft-MDP 视角下正是

$$\log F(s)=\text{soft-}V(s)=\log\!\!\sum_{\tau:s\to x}\!e^{R(x)}\prod p_B$$

**一个量同时携带未来奖励量级和未来路径熵。** 一个纯粹以回报为元目标、完全不知道 GFlowNet 存在的元学习过程,自发漂到了这个语义上。

---

## 3. 第三层:一个可证伪的猜想

前提:`disco_rl/update_rules/weights/disco_103.npz`(2.8 MB,Apache-2.0)**已公开**,`colabs/eval.ipynb` 现成。所以这不是空头支票。

### 为什么选多路径偏差做判别

区分 GFlowNet 与 MaxEnt RL 的**唯一**决定性诊断,就是多路径偏差:当同一个对象有多条生成路径时,MaxEnt RL 诱导的分布有偏([Deleu et al. 2024](https://arxiv.org/abs/2402.10309))。

熵正则化 RL 的最优软策略把 $P(\tau)\propto\pi_0(\tau)R(x)$,所以终态边缘分布被路径丛的参考测度抬高。GFN 用守恒律把它修正回 $p(x)\propto R(x)$。

### 猜想

在路径数解析可算的 DAG 上,用 Disco103 训练出的 agent,其多路径指数满足

$$\boxed{0<\gamma_{\rm Disco}<1}\qquad\text{其中}\quad p(x)\propto R(x)\,n(x)^{\gamma}$$

- **严格小于 1**:$y,z$ 的 KL 结构装得下 detailed balance(第 2 节前四行)。
- **严格大于 0**:元网络看不到父节点集合(第 2 节第五行)。

### 怎么量 γ

**这里有个坑,我第一版就踩了。** 直接对 $\log n(x)$ 做一元回归会得到**假的负值**:均匀参考策略下每多走一步就多乘一个 $1/3$,长度效应盖过路径数效应,而在这个格子上长度与 $\log n$ 强相关。

正确做法是**双变量最小二乘**:

$$\log p(x)-\log R(x)\;\sim\;\big[\log n(x),\;\operatorname{len}(x),\;1\big]$$

读 $\log n(x)$ 的系数,即「控制轨迹长度后,路径数对采样概率的抬高」。这样才有:

- 流匹配 → $\gamma = -0.0000$(本机算出)
- 软 RL → $\gamma = +0.9788$(本机算出)

### 实验设计

1. hypergrid(路径数 = 二项式系数,解析可算),三组 agent:**Disco103**、**actor-critic 基线**、**GFN-TB**。
2. 对每组按上式拟合 γ。
3. 参照点已在本机算好(见上)。

**证伪条件**:若 $\gamma_{\rm Disco}$ 与 actor-critic 基线在误差棒内无差异,「Disco103 的预测带 GFN 式语义」这个说法就死了。没有回旋余地。

### 更便宜的第二探针

在 $y_\theta(s)$ 的 logits 上拟合线性读出 $g(\cdot)$,测 DB 残差 $g(s)+\log p_F(s'|s)-g(s')-\log p_B(s|s')$ 的方差;对照组是在 categorical value head `q` 上拟合同样的探针。

**预言**:$y$ 的最优探针残差显著低于 `q`,且差距随 $n(x)$ 的方差增大。

---

## 4. 第四层:四条拓展

按「能不能写成论文」排序。

### A. 把平衡条件本身元学习掉

GFlowNet 圈现在靠人手在 DB / TB / SubTB(λ) / FL-DB 之间挑一个 —— **这正是 DiscoRL 废掉的那种「人类手搓更新规则」**。把 DiscoRL 的元学习机器原样搬过来,只补上父节点输入、把元目标从回报换成**发现的模式数**或采样器 TV 误差。

据我所查,没人用**采样**目标跑过 DiscoRL 机器。改动量小,机器现成。

### B. 让 α 变成协议,而不是一次选择

用累积量族参数化损失,在 ELBO(一阶)和 TB(二阶)之间连续插值,并让**生命周期 MetaLSTM 输出 α**:训练早期用一阶(梯度稳),后期切二阶(off-policy 稳)。

物理读法:在「最小平均耗散」与「最小耗散涨落」之间退火。这把 Malkin 2023 的「TB vs VI 二选一」变成一条自动退火曲线。

### C. 学习型 p_B = 最优反向协议

最大熵 GFN 依赖 $p_B$ 的选取([Mohammadpour et al. 2024](https://arxiv.org/abs/2312.14331));非平衡物理里「最优协议最小化耗散」是个成熟问题。把 $p_B$ 也交给元网络、元目标直接取 $\operatorname{Var}[W]$,一步把这两件事缝成同一个优化问题。

### D. 反向搬运:log F 头改成 categorical

**这条最便宜、最可能立刻见效。** GFN 的 $\log F$ 现在是**标量回归**,尺度跨几十个数量级,数值条件出了名的差。DiscoRL 规模化成功用的是 categorical + KL(`disco.py:245–247`)。换成固定 support 上的 two-hot / HL-Gauss 分布头,是现代 value learning 里反复验证过的修法。

不需要任何新理论,一个下午的事。

---

## 5. 证据分层

| 层级 | 断言 | 凭什么 |
|---|---|---|
| 已发表 | GFlowNet 训练 ≡ 熵正则化 RL | Tiapkin 2024 · Mohammadpour 2024 |
| 已发表 | 多路径时 MaxEnt RL 分布有偏 | Deleu 2024 |
| 已发表 | 平均耗散功 = 前向/反向路径测度的相对熵 | Kawai–Parrondo–Van den Broeck 2007 |
| **本机跑过** | $\mathbb E[e^{-W}]=Z$,对任意前向策略成立 | `cumulants.py` — 三策略相对误差 ≤ 4.3e−16 |
| **本机跑过** | TB 的最优值 = Var[W],其自由参数 = ELBO | `cumulants.py` — argmin 0.61089 vs −E[W] 0.61089 |
| **本机跑过** | 流匹配下 W 恒定,Var[W] = 0 | \|Var[W]\| = 1.8e−15,E[W] = −log Z |
| **源码核实** | 轨迹内递归反向展开,注释写明用于 bootstrapping | `meta_nets.py:109,116` |
| **源码核实** | π、y、z 三项损失全是 categorical KL | `disco.py:245–247` |
| **源码核实** | 元网络从不接收 p_B 或父节点集合 | `disco.py` — 0 匹配 (S4) |
| **我的综合** | VI = 一阶累积量,TB = 二阶,DiscoRL 元学习这个泛函 | 数学可验(第 1 节);arXiv 共现数 0 |
| **我的综合** | DB 的四个充分统计量已在元网络输入总线上 | `disco.py:336–393` 逐项对照 |
| **猜想 · 待验** | Disco103 训练的 agent 落在 0 < γ < 1 | 未测。证伪条件见第 3 节 |

---

## 6. 复现

```bash
git clone https://github.com/danielchen26/discorl-gfn-bridge
cd discorl-gfn-bridge

# ① 功的前两阶矩与指数平均，8×8 hypergrid 全枚举精确解
python3 research/cumulants.py --json research/cumulants.json

# ② 关于 disco_rl 源码的五条断言，钉在 pinned commit 上
python3 research/verify_disco_source.py

# ③ 浏览器引擎与 Python oracle 逐点比对
node research/parity.ts

# 交互版
npm install && npm run dev
```

| 脚本 | 做什么 | 依赖 |
|---|---|---|
| `research/cumulants.py` | 8×8 hypergrid,12,869 条轨迹,DP 精确算 $\mathbb E[W]$、$\operatorname{Var}[W]$、$\mathbb E[e^{-W}]$、终态分布、γ | 仅标准库 |
| `research/verify_disco_source.py` | 把关于别人代码的断言写成可执行检查,失败即非零退出 | 仅标准库 + 网络 |
| `research/parity.ts` | 187 项比较,TS 引擎 vs Python oracle,容差 1e−12 | Node ≥ 22.6 |

`.github/workflows/verify.yml` 每周一重跑全部三条 —— 因为第 2 节的断言是关于**别人的仓库**的,会悄悄腐烂。

### 目录

```
research/            三个验证脚本 + 它们产生的 JSON
src/lib/hypergrid.ts 精确 DAG 引擎（cumulants.py 的逐行镜像）
src/data/facts.tsx   全部事实的唯一来源，每条带溯源标记
src/components/      WorkDial（签名交互件）· GammaLab · SourceMatrix
```

---

## 7. 参考文献

1. Oh, Farquhar, Kemaev, Calian, Hessel, Zintgraf, Singh, van Hasselt, Silver.
   *Discovering state-of-the-art reinforcement learning algorithms.* **Nature** (2025).
   [doi:10.1038/s41586-025-09761-x](https://doi.org/10.1038/s41586-025-09761-x) ·
   [代码与 Disco103 权重](https://github.com/google-deepmind/disco_rl)
2. Tiapkin, Morozov, Naumov, Vetrov. *Generative Flow Networks as Entropy-Regularized RL.*
   **AISTATS 2024 (Oral)**. [arXiv:2310.12934](https://arxiv.org/abs/2310.12934)
3. Mohammadpour, Bengio, Frejinger, Bacon. *Maximum entropy GFlowNets with soft Q-learning.*
   **AISTATS 2024**. [arXiv:2312.14331](https://arxiv.org/abs/2312.14331)
4. Deleu, Nouri, Malkin, Precup, Bengio. *Discrete Probabilistic Inference as Control in
   Multi-path Environments.* (2024). [arXiv:2402.10309](https://arxiv.org/abs/2402.10309)
5. Malkin, Lahlou, Deleu, Ji, Hu, Everett, Zhang, Bengio. *GFlowNets and variational inference.*
   **ICLR 2023**. [arXiv:2210.00580](https://arxiv.org/abs/2210.00580)
6. Kawai, Parrondo, Van den Broeck. *Dissipation: The phase-space perspective.*
   **PRL 98, 080602** (2007). [doi:10.1103/PhysRevLett.98.080602](https://doi.org/10.1103/PhysRevLett.98.080602)
7. Jarzynski. *Nonequilibrium equality for free energy differences.*
   **PRL 78, 2690** (1997). [doi:10.1103/PhysRevLett.78.2690](https://doi.org/10.1103/PhysRevLett.78.2690)

---

机器:Apple M1 Pro,无 GPU。全部数值可在数秒内复现。
