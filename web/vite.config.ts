/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// VITE_BASE is "/frostguard-scab/" for GitHub Pages and "/" everywhere else.
export default defineConfig({
  base: process.env.VITE_BASE ?? "/",
  plugins: [react()],
  server: {
    port: 5173,
    // Sample leaf photos are imported from ../data/samples (single copy shared with the API tests).
    fs: { allow: [".."] },
  },
  build: {
    target: "es2022",
    sourcemap: true,
    chunkSizeWarningLimit: 700,
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
