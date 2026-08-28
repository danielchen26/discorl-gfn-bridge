import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import { LangProvider } from "./i18n";
import "katex/dist/katex.min.css";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <LangProvider>{(lang, setLang) => <App lang={lang} setLang={setLang} />}</LangProvider>
  </StrictMode>,
);
