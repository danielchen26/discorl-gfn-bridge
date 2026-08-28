import discoSource from "../data/disco_source.json";
import { Chip } from "./ui";
import { useLang, type L } from "../i18n";

const REPO = discoSource.repo;
const COMMIT = discoSource.commit;

const blobUrl = (file: string, lines: number[]) =>
  `https://github.com/${REPO}/blob/${COMMIT}/${file}${lines.length ? `#L${lines[0]}` : ""}`;

/** What a detailed-balance residual needs, against what the meta-network is fed. */
type Need = { need: L; got: L; where: string; has: boolean };

const NEEDS: Need[] = [
  {
    need: { zh: "F(s), F(s′) — 相邻两个状态量", en: "F(s), F(s′) — two consecutive state quantities" },
    got: { zh: "agent_out/y + td_pair → (yₜ, yₜ₊₁)", en: "agent_out/y + td_pair → (yₜ, yₜ₊₁)" },
    where: "disco.py:363–369",
    has: true,
  },
  {
    need: { zh: "F(s→s′) — 边流", en: "F(s→s′) — the edge flow" },
    got: { zh: "agent_out/z + select_a → z(sₜ, aₜ)", en: "agent_out/z + select_a → z(sₜ, aₜ)" },
    where: "disco.py:370–373",
    has: true,
  },
  {
    need: { zh: "Σ_{s′} F(s→s′) — 对子节点求和", en: "Σ_{s′} F(s→s′) — the sum over children" },
    got: { zh: "agent_out/z + pi_weighted_avg + td_pair", en: "agent_out/z + pi_weighted_avg + td_pair" },
    where: "disco.py:374–381",
    has: true,
  },
  {
    need: { zh: "p_F(s′|s) — 前向策略", en: "p_F(s′|s) — the forward policy" },
    got: { zh: "agent_out/logits + softmax + select_a", en: "agent_out/logits + softmax + select_a" },
    where: "disco.py:336–343",
    has: true,
  },
  {
    need: { zh: "p_B(s|s′) — 反向策略", en: "p_B(s|s′) — the backward policy" },
    got: { zh: "没有。父节点集合从未进入元网络。", en: "Absent. No parent set ever reaches the meta-network." },
    where: "disco.py — 0 matches",
    has: false,
  },
];

/** The verifier emits English; these are the same claims for the zh reading. */
const CLAIM_ZH: Record<string, string> = {
  S1: "轨迹内的递归是沿轨迹反向展开的,源码注释写明用途就是 bootstrapping。",
  S2: "π、y、z 三项全部用 categorical KL 训练:预测是分布,不是标量。",
  S3: "元网络收到 (yₜ, yₜ₊₁) 配对,以及 z 在 t 与 t+1 上的策略加权平均、按动作取最大 —— 局部平衡条件的充分统计量。",
  S4: "元网络从未收到反向策略或父节点集合;唯一的非前向信号是对已实现轨迹后缀的反向递归。",
  S5: "存在两个递归 —— 轨迹内反向、生命周期内正向 —— 并以乘性方式结合。",
};

/**
 * The load-bearing table: four of the five sufficient statistics of a detailed
 * balance residual are already on the meta-network's input bus, and the fifth
 * is provably missing. That asymmetry is what makes the conjecture a bounded
 * one rather than a vibe.
 */
export default function SourceMatrix() {
  const lang = useLang();

  return (
    <>
      <div className="scroller">
        <table className="tbl">
          <thead>
            <tr>
              <th style={{ width: "34%" }}>{lang === "zh" ? "DB 残差需要" : "A DB residual needs"}</th>
              <th style={{ width: "40%" }}>{lang === "zh" ? "元网络实际收到" : "What the meta-network is fed"}</th>
              <th style={{ width: "18%" }}>{lang === "zh" ? "位置" : "Where"}</th>
              <th style={{ width: "8%" }} />
            </tr>
          </thead>
          <tbody>
            {NEEDS.map((n) => (
              <tr key={n.where}>
                <td>{n.need[lang]}</td>
                <td>{n.got[lang]}</td>
                <td className="num">{n.where}</td>
                <td className={n.has ? "ok num" : "no num"}>{n.has ? "✓" : "✗"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="callout ok">
        {lang === "zh" ? (
          <>
            <strong>结论。</strong> 前四行全部命中 —— GFlowNet 的 detailed balance 目标是 DiscoRL
            搜索空间里的一个<strong>可达点</strong>。第五行缺失是真限制:反向 LSTM 编码的是轨迹<strong>后缀</strong>,
            不是 s′ 的<strong>兄弟父节点</strong>,所以它表示不了一般 DAG 上的学习型 p_B。
          </>
        ) : (
          <>
            <strong>Reading.</strong> The first four all land — GFlowNet's detailed-balance objective is a{" "}
            <strong>reachable point</strong> in DiscoRL's search space. The fifth is a real limit: the reverse
            LSTM encodes the trajectory <strong>suffix</strong>, not s′'s <strong>sibling parents</strong>, so
            it cannot express a general learned p_B on a multi-parent DAG.
          </>
        )}
      </div>

      <h3>{lang === "zh" ? "断言的可执行版本" : "The claims, as executable assertions"}</h3>
      <p>
        {lang === "zh" ? (
          <>
            关于别人代码的断言会悄悄腐烂,所以它们被写成 <code>research/verify_disco_source.py</code>,钉在
            commit <code>{COMMIT.slice(0, 12)}</code> 上,并且每周在 CI 里重跑一次。下面是最近一次运行的结果。
          </>
        ) : (
          <>
            Claims about someone else's code rot silently, so they live in{" "}
            <code>research/verify_disco_source.py</code>, pinned to commit <code>{COMMIT.slice(0, 12)}</code>{" "}
            and re-run weekly in CI. This is the latest run.
          </>
        )}
      </p>

      <div className="scroller">
        <table className="tbl">
          <thead>
            <tr>
              <th style={{ width: "7%" }}>ID</th>
              <th style={{ width: "56%" }}>{lang === "zh" ? "断言" : "Claim"}</th>
              <th style={{ width: "27%" }}>{lang === "zh" ? "证据" : "Evidence"}</th>
              <th style={{ width: "10%" }} />
            </tr>
          </thead>
          <tbody>
            {discoSource.checks.map((c) => (
              <tr key={c.id}>
                <td className="num">{c.id}</td>
                <td>{lang === "zh" ? (CLAIM_ZH[c.id] ?? c.claim) : c.claim}</td>
                <td className="num">
                  <a href={blobUrl(c.file, c.lines)} target="_blank" rel="noreferrer">
                    {c.file.split("/").pop()}
                    {c.lines.length ? `:${c.lines.join(",")}` : ""}
                  </a>
                </td>
                <td>
                  <Chip p={c.ok ? "verified" : "conjecture"}>{c.ok ? "PASS" : "FAIL"}</Chip>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
