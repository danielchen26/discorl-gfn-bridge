import katex from "katex";
import type { ReactNode } from "react";

import { useLang, type L, type LN } from "../i18n";

/** Provenance of a statement. The whole dossier is sorted by this. */
export type Prov = "verified" | "published" | "source" | "mine" | "conjecture";

const PROV_LABEL: Record<Prov, L> = {
  verified: { zh: "本机跑过", en: "We ran it" },
  published: { zh: "已发表", en: "Published" },
  source: { zh: "源码核实", en: "Read the source" },
  mine: { zh: "我的综合", en: "My synthesis" },
  conjecture: { zh: "猜想 · 待验", en: "Conjecture" },
};

export function M({ tex, block }: { tex: string; block?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: !!block, throwOnError: false, strict: false });
  return block ? (
    <div className="mblock" dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <span dangerouslySetInnerHTML={{ __html: html }} />
  );
}

export function Chip({ p, children }: { p: Prov; children?: ReactNode }) {
  const lang = useLang();
  return <span className={`chip ${p}`}>{children ?? PROV_LABEL[p][lang]}</span>;
}

export function Section({
  id,
  num,
  title,
  kicker,
  children,
}: {
  id: string;
  num: string;
  title: LN;
  kicker?: LN;
  children: ReactNode;
}) {
  const lang = useLang();
  return (
    <section id={id}>
      <div className="sec-head">
        <span className="sec-num">{num}</span>
        <div>
          <h2>{title[lang]}</h2>
          {kicker ? <p>{kicker[lang]}</p> : null}
        </div>
      </div>
      {children}
    </section>
  );
}

export function Rig({
  title,
  note,
  right,
  children,
}: {
  title: LN;
  note?: LN;
  right?: ReactNode;
  children: ReactNode;
}) {
  const lang = useLang();
  return (
    <div className="rig">
      <div className="rig-head">
        <div>
          <h4>{title[lang]}</h4>
          {note ? <p>{note[lang]}</p> : null}
        </div>
        {right}
      </div>
      <div className="rig-body">{children}</div>
    </div>
  );
}

export function Readout({
  k,
  v,
  note,
  tone,
  locked,
}: {
  k: LN;
  v: string;
  note?: LN;
  tone?: "gfn" | "rl" | "ver" | "conj";
  locked?: boolean;
}) {
  const lang = useLang();
  return (
    <div className={`ro${locked ? " locked" : ""}`}>
      <span className="k">{k[lang]}</span>
      <span className="v" style={tone && !locked ? { color: `var(--${tone})` } : undefined}>
        {v}
      </span>
      {note ? <span className="n">{note[lang]}</span> : null}
    </div>
  );
}

export function T({ v }: { v: LN }) {
  return <>{v[useLang()]}</>;
}
