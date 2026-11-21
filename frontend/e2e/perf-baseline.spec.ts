// Local baseline for #1023 (not a budget): load, navigation, filter and scroll costs on the
// data-heavy routes, small and large fixtures, cold and warm, at 4x CPU. Writes one JSON line
// per measurement to test-results/perf-baseline.jsonl.
import { appendFileSync, mkdirSync } from "node:fs";
import { expect, test, type Page } from "@playwright/test";
import { installApi, LOADED_ROW, warmUp } from "./fixtures";

const OUT = "test-results/perf-baseline.jsonl";
const RATE = Number(process.env.PERF_CPU ?? 4);

const SMALL = { memoryRows: 12, knowledgeRows: 12, runRows: 12, threadRows: 12, auditRows: 12, browseRows: 12, taskRows: 12 };
// The largest page each view asks for: Memories' 100-row page size, Spaces' `limit: 200`
// browse read, the Activity log's 100-row page (the fixtures answer every row they hold,
// so a larger count would draw more than the server ever sends).
const LARGE = { memoryRows: 100, knowledgeRows: 1000, runRows: 1000, threadRows: 300, auditRows: 100, browseRows: 200, taskRows: 300 };

const ROUTES = ["/memories", "/library", "/runs", "/conversations", "/tasks", "/admin/activity", "/spaces", "/repositories", "/recall", "/graph", "/admin/usage", "/settings"];

test.beforeAll(({ browser }, testInfo) => warmUp(browser, testInfo));

test("environment", async ({ browser }) => {
  record({
    phase: "env",
    browser: browser.version(),
    node: process.version,
    platform: process.platform,
    cpus: (await import("node:os")).cpus().length,
    commit: process.env.GITHUB_SHA ?? "local",
    runner: process.env.RUNNER_NAME ?? "local",
  });
});

function record(row: Record<string, unknown>) {
  mkdirSync("test-results", { recursive: true });
  appendFileSync(OUT, JSON.stringify(row) + "\n");
}

async function observe(page: Page) {
  await page.addInitScript(() => {
    const w = window as unknown as Record<string, unknown>;
    w.__lcp = 0;
    w.__lt = [] as number[];
    w.__ev = [] as Array<{ name: string; d: number }>;
    try {
      new PerformanceObserver((l) => {
        for (const e of l.getEntries()) w.__lcp = e.startTime;
      }).observe({ type: "largest-contentful-paint", buffered: true });
      new PerformanceObserver((l) => {
        for (const e of l.getEntries()) (w.__lt as number[]).push(e.duration);
      }).observe({ type: "longtask", buffered: true });
      new PerformanceObserver((l) => {
        for (const e of l.getEntries()) (w.__ev as Array<{ name: string; d: number }>).push({ name: e.name, d: e.duration });
      }).observe({ type: "event", buffered: true, durationThreshold: 16 } as PerformanceObserverInit);
    } catch {
      /* not supported */
    }
  });
}

async function pageStats(page: Page) {
  return page.evaluate(() => {
    const w = window as unknown as { __lcp: number; __lt: number[] };
    const res = performance.getEntriesByType("resource") as PerformanceResourceTiming[];
    const js = res.filter((r) => r.initiatorType === "script" || /\.js(\?|$)/.test(r.name));
    return {
      lcp_ms: Math.round(w.__lcp),
      long_task_ms: Math.round(w.__lt.reduce((a, b) => a + b, 0)),
      long_task_max_ms: Math.round(Math.max(0, ...w.__lt)),
      long_tasks: w.__lt.length,
      js_files: js.length,
      js_transfer_kb: Math.round(js.reduce((a, r) => a + r.transferSize, 0) / 1024),
      js_decoded_kb: Math.round(js.reduce((a, r) => a + r.decodedBodySize, 0) / 1024),
      dom_nodes: document.querySelectorAll("*").length,
    };
  });
}

async function load(page: Page, path: string) {
  const calls: string[] = [];
  const onReq = (r: { url(): string }) => {
    const u = new URL(r.url());
    if (u.pathname.startsWith("/v1/") && u.pathname !== "/v1/ui.vitals") calls.push(u.pathname.slice(4));
  };
  page.on("request", onReq);
  const t0 = Date.now();
  await page.goto(path, { waitUntil: "domcontentloaded" });
  await expect(page.locator("#main :is(h1, h2)").first()).toBeVisible();
  const heading_ms = Date.now() - t0;
  let rows_ms: number | null = null;
  try {
    await page.locator(LOADED_ROW).first().waitFor({ timeout: 8000 });
    rows_ms = Date.now() - t0;
  } catch {
    /* route draws no list rows */
  }
  await page.waitForTimeout(1500);
  page.off("request", onReq);
  const rows = await page.locator(LOADED_ROW).count();
  return { heading_ms, rows_ms, rows, api_calls: calls.length, api_distinct: new Set(calls).size, calls: [...new Set(calls)].join(",") };
}

for (const [size, opts] of [["small", SMALL], ["large", LARGE]] as const) {
  for (const path of ROUTES) {
    test(`baseline ${size} ${path}`, async ({ browser }, testInfo) => {
      const context = await browser.newContext({ baseURL: testInfo.project.use.baseURL, viewport: { width: 1400, height: 900 } });
      const page = await context.newPage();
      if (size === "large" && path === "/memories") await page.addInitScript(() => localStorage.setItem("loom_page_size_memories_v1", "100"));
      await observe(page);
      await installApi(page, opts);
      const cdp = await context.newCDPSession(page);
      await cdp.send("Emulation.setCPUThrottlingRate", { rate: RATE });
      for (const phase of ["cold", "warm"]) {
        const l = await load(page, path);
        const s = await pageStats(page);
        record({ size, path, phase, cpu: RATE, ...l, ...s });
      }
      await cdp.detach();
      await context.close();
    });
  }
}

// Interactions on the large fixture: typing in a list's search box, scrolling the list, and
// moving between routes from the sidebar.
const SEARCH: Array<[string, string, string]> = [
  ["/memories", "Search memories", "lane"],
  ["/runs", "Search runs", "RUN-00"],
  ["/tasks", "Search tasks", "triage"],
];

for (const [path, label, text] of SEARCH) {
  test(`interaction large ${path}: search and scroll`, async ({ browser }, testInfo) => {
    const context = await browser.newContext({ baseURL: testInfo.project.use.baseURL, viewport: { width: 1400, height: 900 } });
    const page = await context.newPage();
    if (path === "/memories") await page.addInitScript(() => localStorage.setItem("loom_page_size_memories_v1", "100"));
    await observe(page);
    await installApi(page, LARGE);
    const cdp = await context.newCDPSession(page);
    await cdp.send("Emulation.setCPUThrottlingRate", { rate: RATE });
    await load(page, path);
    const box = page.getByLabel(label).first();
    const searchable = await box.count();
    let search: Record<string, unknown> = { search: "no box" };
    if (searchable) {
      await page.evaluate(() => {
        const w = window as unknown as { __lt: number[]; __ev: unknown[] };
        w.__lt.length = 0;
        w.__ev.length = 0;
      });
      const t0 = Date.now();
      await box.pressSequentially(text, { delay: 60 });
      await page.waitForTimeout(1500);
      search = await page.evaluate(() => {
        const w = window as unknown as { __lt: number[]; __ev: Array<{ name: string; d: number }> };
        return {
          max_event_ms: Math.round(Math.max(0, ...w.__ev.map((e) => e.d))),
          slow_events: w.__ev.length,
          long_task_ms: Math.round(w.__lt.reduce((a, b) => a + b, 0)),
        };
      });
      search.typed_ms = Date.now() - t0;
      search.rows_after = await page.locator(LOADED_ROW).count();
    }
    await page.evaluate(() => {
      const w = window as unknown as { __lt: number[] };
      w.__lt.length = 0;
    });
    const s0 = Date.now();
    for (let i = 0; i < 20; i++) {
      await page.mouse.move(700, 500);
      await page.mouse.wheel(0, 600);
      await page.waitForTimeout(50);
    }
    const scroll = await page.evaluate(() => {
      const w = window as unknown as { __lt: number[] };
      return { scroll_long_task_ms: Math.round(w.__lt.reduce((a, b) => a + b, 0)), scroll_long_tasks: w.__lt.length };
    });
    record({ size: "large", path, phase: "interaction", cpu: RATE, ...search, ...scroll, scroll_ms: Date.now() - s0 });
    await cdp.detach();
    await context.close();
  });
}

test("navigation large: sidebar route changes", async ({ browser }, testInfo) => {
  const context = await browser.newContext({ baseURL: testInfo.project.use.baseURL, viewport: { width: 1400, height: 900 } });
  const page = await context.newPage();
  await observe(page);
  await installApi(page, LARGE);
  const cdp = await context.newCDPSession(page);
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: RATE });
  await load(page, "/memories");
  for (const name of ["Runs", "Library", "Tasks", "Memories"]) {
    const link = page.getByRole("link", { name, exact: true }).first();
    if (!(await link.count())) {
      record({ size: "large", path: name, phase: "nav", cpu: RATE, error: "no sidebar link" });
      continue;
    }
    await page.evaluate(() => {
      (window as unknown as { __lt: number[] }).__lt.length = 0;
    });
    const t0 = Date.now();
    await link.click();
    await expect(page.locator("#main :is(h1, h2)").first()).toBeVisible();
    const heading_ms = Date.now() - t0;
    let rows_ms: number | null = null;
    try {
      await page.locator(LOADED_ROW).first().waitFor({ timeout: 8000 });
      rows_ms = Date.now() - t0;
    } catch {
      /* no rows */
    }
    await page.waitForTimeout(800);
    const lt = await page.evaluate(() => Math.round((window as unknown as { __lt: number[] }).__lt.reduce((a, b) => a + b, 0)));
    record({ size: "large", path: name, phase: "nav", cpu: RATE, heading_ms, rows_ms, long_task_ms: lt, url: new URL(page.url()).pathname });
  }
  await cdp.detach();
  await context.close();
});
