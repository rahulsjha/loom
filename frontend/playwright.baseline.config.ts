// Local only: the repo config, with the full Chromium build (no headless shell installed here).
import base from "./playwright.config";
import { defineConfig } from "@playwright/test";

export default defineConfig({
  ...base,
  retries: 0,
  timeout: 180_000,
  use: { ...base.use, channel: "chromium" },
  projects: [{ name: "desktop", testMatch: /perf-baseline\.spec\.ts/ }],
});
