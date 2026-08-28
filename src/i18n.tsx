import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

export type Lang = "zh" | "en";

/** A localisable string. */
export type L = { zh: string; en: string };
/** Localisable rich text (may contain maths, code, emphasis). */
export type LN = { zh: ReactNode; en: ReactNode };

const LangCtx = createContext<Lang>("zh");

const STORAGE_KEY = "wtau-lang";

export function LangProvider({ children }: { children: (lang: Lang, set: (l: Lang) => void) => ReactNode }) {
  const [lang, setLang] = useState<Lang>(() => {
    const saved = typeof localStorage !== "undefined" ? localStorage.getItem(STORAGE_KEY) : null;
    if (saved === "zh" || saved === "en") return saved;
    return typeof navigator !== "undefined" && navigator.language.startsWith("zh") ? "zh" : "en";
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, lang);
    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
    document.title =
      lang === "zh"
        ? "W(τ) · DiscoRL 与 GFlowNet 的同一个数学对象"
        : "W(τ) · The object DiscoRL and GFlowNets share";
  }, [lang]);

  return <LangCtx.Provider value={lang}>{children(lang, setLang)}</LangCtx.Provider>;
}

export function useLang(): Lang {
  return useContext(LangCtx);
}
