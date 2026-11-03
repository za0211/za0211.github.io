/**
 * 姿勢チェックの動作確認テスト
 *
 *   cd posture/tests
 *   npm install
 *   npx playwright install chromium   # 初回のみ（ブラウザが入っていない場合）
 *   npm test                          # 任意の写真で試す: npm test -- 正面.jpg 側面.jpg
 *   （既存のChromiumを使う場合: CHROMIUM_PATH=/path/to/chrome npm test）
 *
 * やること：
 *   1. posture/ を簡易サーバーで配信
 *   2. ブラウザで開き、正面・側面に写真を読み込ませる
 *   3. 10項目の数値と総合スコアが表示されるか確認し、画面を out/ に保存
 *
 * CDN（jsdelivr）が使えない環境でも動くよう、MediaPipe は npm で入れたものを使います。
 * AIモデルとサンプル写真は .cache/ に一度だけダウンロードします。
 */
const { chromium } = require("playwright");
const http = require("http");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const CACHE = path.join(__dirname, ".cache");
const OUT = path.join(__dirname, "out");
const NM = path.join(__dirname, "node_modules/@mediapipe/tasks-vision");
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";
const SAMPLE_URL = "https://storage.googleapis.com/mediapipe-assets/pose.jpg";

async function cached(url, name) {
  const f = path.join(CACHE, name);
  if (!fs.existsSync(f)) {
    fs.mkdirSync(CACHE, { recursive: true });
    const res = await fetch(url);
    if (!res.ok) throw new Error(`download failed ${res.status}: ${url}`);
    fs.writeFileSync(f, Buffer.from(await res.arrayBuffer()));
  }
  return f;
}

function serve() {
  const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".png": "image/png", ".jpg": "image/jpeg" };
  const server = http.createServer((req, res) => {
    let p = decodeURIComponent(new URL(req.url, "http://x").pathname);
    if (p.endsWith("/")) p += "index.html";
    const f = path.join(ROOT, p);
    if (!f.startsWith(ROOT) || !fs.existsSync(f)) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { "content-type": types[path.extname(f)] || "application/octet-stream" });
    fs.createReadStream(f).pipe(res);
  });
  return new Promise((r) => server.listen(0, () => r(server)));
}

(async () => {
  const [frontArg, sideArg] = process.argv.slice(2);
  const model = await cached(MODEL_URL, "pose_landmarker_full.task");
  const sample = frontArg && sideArg ? null : await cached(SAMPLE_URL, "sample.jpg");
  const front = frontArg || sample, side = sideArg || sample;
  fs.mkdirSync(OUT, { recursive: true });

  const server = await serve();
  const base = `http://localhost:${server.address().port}/`;
  // CHROMIUM_PATH を指定すると、そのブラウザを使います（Playwright のブラウザを別に入れている環境向け）
  const browser = await chromium.launch(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {});
  let failed = false;

  for (const vp of [{ width: 1280, height: 900, name: "desktop" }, { width: 390, height: 844, name: "mobile" }]) {
    const page = await browser.newPage({ viewport: vp });
    page.on("pageerror", (e) => { failed = true; console.error("pageerror:", e.message); });
    await page.route("https://cdn.jsdelivr.net/**", (r) => {
      const rel = new URL(r.request().url()).pathname.replace(/^\/npm\/@mediapipe\/tasks-vision@[^/]+\//, "");
      const f = path.join(NM, rel);
      r.fulfill({ status: 200, body: fs.readFileSync(f), contentType: f.endsWith(".wasm") ? "application/wasm" : "text/javascript" });
    });
    await page.route("https://storage.googleapis.com/**", (r) => r.fulfill({ status: 200, body: fs.readFileSync(model) }));
    await page.route("https://fonts.**", (r) => r.abort());

    await page.goto(base);
    await page.setInputFiles("#view-front .file", front);
    await page.setInputFiles("#view-side .file", side);
    await page.waitForFunction(() => document.querySelectorAll(".metric").length >= 10, null, { timeout: 90000 });
    await page.waitForSelector("#result:not([hidden])");

    const score = await page.$eval("#scoreNum", (e) => e.textContent);
    const rows = await page.$$eval(".metric", (es) => es.map((e) => e.innerText.replace(/\s*\n\s*/g, " | ")));
    console.log(`\n[${vp.name}] 総合スコア: ${score}`);
    rows.forEach((r) => console.log("  " + r));
    if (!/^\d+$/.test(score)) { failed = true; console.error("スコアが表示されていません"); }
    await page.screenshot({ path: path.join(OUT, `${vp.name}.png`), fullPage: true });
    await page.close();
  }

  await browser.close();
  server.close();
  console.log(`\n画面の保存先: ${OUT}`);
  console.log(failed ? "NG" : "OK");
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
