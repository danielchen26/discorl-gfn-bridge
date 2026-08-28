import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// GitHub Pages serves this under /<repo>/; local dev serves it at the root.
export default defineConfig({
  base: process.env.GITHUB_ACTIONS ? "/discorl-gfn-bridge/" : "/",
  plugins: [react()],
});
