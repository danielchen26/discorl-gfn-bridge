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

## 3. 第三层:把猜想真的测了

**结论先说:部分。两个方向的强命题都不成立。**

### 3.1 更正:第一版提的 γ 实验是错的

原方案是训练一个 Disco103 agent,再拟合 $p(x)\propto R(x)n(x)^\gamma$。**动手写 harness 时才发现它不成立**:

DiscoRL **最大化回报**。在 hypergrid 上给终态奖励 $R(x)$,它的最优策略是确定性的 $\arg\max R$ —— $p(x)$ 塌成点质量,γ 根本没有定义。在收敛前测,量到的是「还没收敛」而不是「结构上无偏」。

γ 仍然有效,但只对**本来就是采样器**的两者(GFlowNet 与 MaxEnt RL),那部分第 1 节已经精确算完。对 DiscoRL,正确的问法是问它的**更新规则**,不是问它收敛到的分布。

### 3.2 判据:一个无量纲比值

detailed balance 说状态量由后继和局部转移概率重建;value bootstrap 说 $V(s)=r+\gamma V(s')$,**完全不依赖你走这一步的概率**。差别只在这一条。定义

$$\alpha=\frac{\partial\varphi(\hat y_t)}{\partial\varphi(y_{t+1})},\qquad \beta=\frac{\partial\varphi(\hat y_t)}{\partial\log\pi(a_t\mid s_t)},\qquad \rho=\frac{\partial\varphi(\hat y_t)}{\partial r_t}$$

其中 $\varphi$ **不是我拟合的探针** —— 它是 Disco103 **自带**的 `y_net`(600→16→1,权重就在 `disco_103.npz` 里),所以这个测量有**零个自由参数**。φ 的单位是任意的,所以判据取无量纲比 $|\beta/\alpha|$:

- detailed balance → $|\beta/\alpha|\approx 1$
- 纯 value bootstrap → $|\beta/\alpha| = 0$

再加一个**局部性对照**:扰动 $t_0+3$ 处的策略,读 $t_0$ 处的目标。DB 是局部的,这个应当几乎无响应;反向 LSTM 会把后缀信息糊开,没有这个对照就分不清「平衡结构」和「弥散」。

### 3.3 两个方法论陷阱

1. **autodiff 在这里是假的。** 策略输入带 `stop_grad`(`disco.py:338`),反向模式导数**恒为 0**。第一版探针给出 β = 0.0000 —— 那是**测量失效**,不是结果。前向值不受影响,所以必须用中心差分。
2. **随机初始化的 agent 测不出任何东西。** Disco103 是对着有能力的 agent 元学出来的;喂它近似均匀的策略和 600 维预测,输出塌成常数,一切灵敏度都读成 0。必须先用 Disco103 自己把 agent 训一段(`--train-steps`)。

第三个坑,记在这里免得别人再踩:JAX 的 `.at[i]` 对**越界索引静默丢弃**,所以安慰剂 tap 取到轨迹末尾时 hi 与 lo 变成同一个张量,0/0 读出 `nan` 而不是报错。

### 3.4 结果

`.venv/bin/python research/disco_probe.py`,8×8 hypergrid,batch 48,4 个 tap,Disco103 训练 400 步:

| 元参数 | \|β\| | \|α\| | \|β/α\| | 安慰剂 | 局部性 | φ 幅度 |
|---|---|---|---|---|---|---|
| **Disco103** | 6.85e-3 | 2.12e-2 | **0.323** | 2.94e-4 | **23.3×** | 2.30e-2 |
| random-init 0 | 5.14e-5 | 1.60e-4 | 0.322 | 5.45e-6 | 9.4× | 3.96e-4 |
| random-init 1 | 5.44e-5 | 4.09e-4 | 0.133 | 1.21e-5 | 4.5× | 2.66e-4 |
| random-init 2 | 4.21e-5 | 8.80e-5 | 0.479 | 2.36e-5 | 1.8× | 5.02e-4 |

**「y 只是个 value function」被推翻。** β 是随机初始化 null 的 **139×**,而且响应高度**局部**(把扰动挪到三步之后,响应掉到 1/23)。纯 value bootstrap 对所走动作的概率应当**完全**不敏感。

**但「y 实现 detailed balance」也不成立。** $|\beta/\alpha| \approx 0.32$,离 DB 要求的 1 差三倍。跨探针配置这个数在 **0.26–0.41** 之间浮动,所以它是个区间,不是常数。

### 3.5 必须声明的边界

- Disco103 是在 Atari / ProcGen / DMLab 上元学习出来的,这里被喂了一个 8×8 玩具格子 —— **分布外查询**。
- φ 的单位任意,所以判据只能是比值,不能是 β 本身。
- 批元素通过 advantage/TD 的 EMA 归一化有 **O(1/B)** 串扰,B=48 时约 2%。
- 这不是对「DiscoRL 是不是 GFlowNet」的判决,而是对「它的 y 通道离平衡条件有多远」的一次定量读数。

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

### D. 反向搬运:log F 头改成 categorical —— **已跑,部分成立**

GFN 的 $\log F$ 现在是**标量回归**,尺度跨几十个数量级,数值条件出了名的差。DiscoRL 规模化成功用的是 categorical + KL(`disco.py:245-247`)。

`research/logf_head.py` 把这条真跑了。两条臂共用同一个策略网络、优化器、种子、评估点,**只有 flow 读出头不同**:标量线性输出 vs 固定 support 上 51 个 bin 的 softmax 期望。KL 由 DAG 上的动态规划**精确**算出,不是采样估计。

| | 标量 log F | categorical two-hot |
|---|---|---|
| 领先检查点 | 2/20 | **18/20** |
| KL 几何均值比(噪声底之上 11 点) | — | **1.41×** 更优 |
| 梯度范数 p99 | 1.561 | **1.130** |
| 梯度范数 mean | 0.100 | **0.081** |
| 末点 KL | 2.75e-5 | 6.90e-5 |

**categorical 头收敛更快、梯度尾更紧**(p99 1.13 vs 1.56 —— 这正是 conditioning 的论断)。**但渐近值并不更好**:两条臂最后都掉进 on-policy 噪声底(1.37e-4 以下),末点顺序是抛硬币,而末点恰好是标量领先。**只看 final KL 会得出与曲线完全相反的结论。**

而且这不是强版本的决定性检验 —— 这个格子的 $\log F$ 只跨 **5.5 nats**,不是拓展 D 所设想的「几十个数量级」。

> 措辞注意:18/20 是**全程**所有 step>0 的检查点,对应几何均值比 1.64×;1.41× 只对应噪声底**之上**的 11 个点。两者不能交叉引用。

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
| **我的综合** | 原 γ 实验设计有缺陷,已由 β 探针替换 | 写 harness 时发现,见 3.1 |
| **本机跑过** | Disco103 的 y 目标对所走动作的 log 概率有真实且**局部**的响应 | `disco_probe.py` — β 是 null 的 139×,局部性 23.3× |
| **本机跑过** | 但它**没有**实现 detailed balance | \|β/α\| = 0.32(跨配置 0.26–0.41),DB 要求 ≈1 |
| **本机跑过** | 拓展 D 部分成立:收敛更快、梯度尾更紧,渐近值不更好 | `logf_head.py` — 18/20 检查点领先,梯度 p99 1.13 vs 1.56 |

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

# ④ β 探针：Disco103 的 y 目标离 detailed balance 有多远（需要 JAX，见下）
python3 -m venv .venv && .venv/bin/pip install "jax[cpu]" dm-haiku rlax distrax chex optax ml_collections
.venv/bin/pip install git+https://github.com/google-deepmind/disco_rl.git
cd research && ../.venv/bin/python disco_probe.py --json disco_probe.json

# ⑤ 拓展 D：标量 vs categorical log F 头
.venv/bin/python research/logf_head.py --json research/logf_head.json

# 交互版
npm install && npm run dev
```

| 脚本 | 做什么 | 依赖 |
|---|---|---|
| `research/cumulants.py` | 8×8 hypergrid,12,869 条轨迹,DP 精确算 $\mathbb E[W]$、$\operatorname{Var}[W]$、$\mathbb E[e^{-W}]$、终态分布、γ | 仅标准库 |
| `research/verify_disco_source.py` | 把关于别人代码的断言写成可执行检查,失败即非零退出 | 仅标准库 + 网络 |
| `research/parity.ts` | 187 项比较,TS 引擎 vs Python oracle,容差 1e−12 | Node ≥ 22.6 |
| `research/hypergrid_env.py` | hypergrid 作为 disco_rl jittable 环境 | JAX + disco_rl |
| `research/disco_probe.py` | β / α / ρ 灵敏度 + 局部性安慰剂,零自由参数 | JAX + disco_rl |
| `research/logf_head.py` | 标量 vs categorical flow 头,5 种子,精确 KL | JAX |

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
