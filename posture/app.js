import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task";
const MAX_SIDE = 1600;
const HISTORY_KEY = "posture-check-history-v1";

const $ = (s, r = document) => r.querySelector(s);
const toDeg = (r) => (r * 180) / Math.PI;
const fmt = (v, d = 1) => (Math.round(v * 10 ** d) / 10 ** d).toFixed(d);

/* ------------------------------------------------------------------ */
/* Model                                                               */
/* ------------------------------------------------------------------ */
let landmarkerPromise = null;
let detectQueue = Promise.resolve();

function setStatus(msg, kind = "") {
  const el = $("#modelStatus");
  if (!msg) { el.hidden = true; return; }
  el.hidden = false;
  el.className = "status " + kind;
  el.innerHTML = (kind === "loading" ? '<span class="spinner"></span>' : "") + msg;
}

function getLandmarker() {
  if (!landmarkerPromise) {
    setStatus("AIモデルを読み込んでいます…（初回のみ数秒〜数十秒かかります）", "loading");
    landmarkerPromise = (async () => {
      const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
      const make = (delegate) => PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate },
        runningMode: "IMAGE",
        numPoses: 1,
        minPoseDetectionConfidence: 0.4,
        minPosePresenceConfidence: 0.4,
      });
      try { return await make("GPU"); } catch { return await make("CPU"); }
    })();
    landmarkerPromise.then(() => setStatus(""), (err) => {
      console.error(err);
      landmarkerPromise = null;
      setStatus("AIモデルの読み込みに失敗しました。通信環境を確認して、もう一度写真を選択してください。", "error");
    });
  }
  return landmarkerPromise;
}

function detect(source) {
  const job = detectQueue.then(async () => {
    const lm = await getLandmarker();
    return lm.detect(source);
  });
  detectQueue = job.catch(() => {});
  return job;
}

/* ------------------------------------------------------------------ */
/* Scoring helpers                                                     */
/* ------------------------------------------------------------------ */
const LEVEL_LABEL = { good: "良好", mild: "軽度", poor: "要注意" };

function levelOf(a, g, m) { return a <= g ? "good" : a <= m ? "mild" : "poor"; }
function scoreOf(a, g, m) {
  if (a <= g) return 100 - (10 * a) / g;
  if (a <= m) return 90 - (30 * (a - g)) / (m - g);
  return Math.max(15, 60 - (40 * (a - m)) / m);
}
function metric(o) {
  const a = Math.abs(o.dev ?? o.value);
  return { ...o, level: levelOf(a, o.g, o.m), score: scoreOf(a, o.g, o.m) };
}

/* ------------------------------------------------------------------ */
/* Geometry                                                            */
/* ------------------------------------------------------------------ */
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const byX = (a, b) => (a.x <= b.x ? [a, b] : [b, a]);
/** angle of the vector a->b measured from straight up, sign = +x direction */
const fromVertical = (a, b) => toDeg(Math.atan2(b.x - a.x, a.y - b.y));

// MediaPipe indices
const IDX = { nose: 0, earL: 7, earR: 8, shL: 11, shR: 12, hipL: 23, hipR: 24, kneeL: 25, kneeR: 26, ankL: 27, ankR: 28, heelL: 29, heelR: 30, toeL: 31, toeR: 32 };

/* ---------------- Front / back ---------------- */
const FRONT_KEYS = ["earL", "earR", "shL", "shR", "hipL", "hipR", "kneeL", "kneeR", "ankL", "ankR"];
const FRONT_BONES = [["earL", "earR"], ["shL", "shR"], ["hipL", "hipR"], ["shL", "hipL"], ["shR", "hipR"], ["hipL", "kneeL"], ["kneeL", "ankL"], ["hipR", "kneeR"], ["kneeR", "ankR"]];

function frontPoints(lm, w, h) {
  const p = {};
  for (const k of [...FRONT_KEYS, "nose"]) {
    const l = lm[IDX[k]];
    p[k] = { x: l.x * w, y: l.y * h, v: l.visibility ?? 1 };
  }
  return p;
}

function analyzeFront(p, facing) {
  // image-left landmark is the person's RIGHT when facing the camera, LEFT when seen from behind
  const sideName = (imgLeft) => ((facing === "front") === imgLeft ? "右" : "左");
  const [shA, shB] = byX(p.shL, p.shR);
  const [hipA, hipB] = byX(p.hipL, p.hipR);
  const [earA, earB] = byX(p.earL, p.earR);
  const [ankA, ankB] = byX(p.ankL, p.ankR);
  const [knA, knB] = byX(p.kneeL, p.kneeR);
  const shC = mid(shA, shB), hipC = mid(hipA, hipB), ankC = mid(ankA, ankB);
  const topY = Math.min(p.nose.y, earA.y, earB.y);
  const H = Math.max(1, ankC.y - topY);

  const tilt = (a, b) => toDeg(Math.atan2(a.y - b.y, b.x - a.x)); // >0: image-right point is higher
  const out = [];

  const sh = tilt(shA, shB);
  out.push(metric({
    id: "shoulder", name: "肩の高さ（左右差）", value: Math.abs(sh), unit: "°", g: 1.5, m: 3,
    desc: Math.abs(sh) < 0.3 ? "左右ほぼ水平です" : `${sideName(sh < 0)}肩が高い`, high: sideName(sh < 0),
  }));

  const hp = tilt(hipA, hipB);
  out.push(metric({
    id: "pelvis", name: "骨盤の傾き（左右差）", value: Math.abs(hp), unit: "°", g: 1.5, m: 3,
    desc: Math.abs(hp) < 0.3 ? "左右ほぼ水平です" : `${sideName(hp < 0)}側の骨盤が高い`, high: sideName(hp < 0),
  }));

  const er = tilt(earA, earB);
  out.push(metric({
    id: "head", name: "頭の傾き", value: Math.abs(er), unit: "°", g: 2, m: 4,
    desc: Math.abs(er) < 0.3 ? "まっすぐです" : `頭が${sideName(er > 0)}側に傾いている`,
  }));

  const sx = ((shC.x - ankC.x) / H) * 100;
  out.push(metric({
    id: "shift", name: "上半身の左右シフト", value: Math.abs(sx), unit: "%", g: 1.5, m: 3,
    desc: Math.abs(sx) < 0.3 ? "重心線上にあります" : `上半身が${sideName(sx < 0)}側へ寄っている（身長比）`,
  }));

  const hx = ((hipC.x - ankC.x) / H) * 100;
  out.push(metric({
    id: "pshift", name: "骨盤の左右シフト", value: Math.abs(hx), unit: "%", g: 1.5, m: 3,
    desc: Math.abs(hx) < 0.3 ? "重心線上にあります" : `骨盤が${sideName(hx < 0)}側へ寄っている（片足重心の傾向）`,
  }));

  // knee alignment: knee offset from the hip→ankle line, + = outward (bow-leg), - = inward (knock-knee)
  const legDev = (hip, knee, ank) => {
    const t = (knee.y - hip.y) / Math.max(1, ank.y - hip.y);
    const lineX = hip.x + (ank.x - hip.x) * t;
    const outward = Math.sign(hip.x - hipC.x) || 1;
    const len = Math.hypot(ank.x - hip.x, ank.y - hip.y) || 1;
    return (((knee.x - lineX) * outward) / len) * 100;
  };
  const hipFor = (knee) => (Math.abs(knee.x - hipA.x) < Math.abs(knee.x - hipB.x) ? hipA : hipB);
  const ankFor = (knee) => (Math.abs(knee.x - ankA.x) < Math.abs(knee.x - ankB.x) ? ankA : ankB);
  const kd = (legDev(hipFor(knA), knA, ankFor(knA)) + legDev(hipFor(knB), knB, ankFor(knB))) / 2;
  out.push(metric({
    id: "knee", name: "脚のライン（O脚・X脚）", value: kd, dev: kd, unit: "%", g: 2.5, m: 5, signed: true,
    desc: Math.abs(kd) < 1 ? "まっすぐです" : kd > 0 ? "O脚傾向（膝が外側に開く）" : "X脚傾向（膝が内側に入る）",
  }));

  const warn = [];
  if (Math.min(p.ankL.v, p.ankR.v) < 0.5) warn.push("足首がはっきり写っていない可能性があります。全身が写った写真を使うと精度が上がります。");
  if (Math.abs(p.shL.x - p.shR.x) < H * 0.08) warn.push("体が横を向いているようです。正面（または背面）からの写真を使ってください。");
  return { metrics: out, warn, refs: { shA, shB, hipA, hipB, earA, earB, shC, hipC, ankC, topY, H, sh, hp, er } };
}

/* ---------------- Side ---------------- */
const SIDE_KEYS = ["ear", "sh", "hip", "knee", "ank"];
const SIDE_BONES = [["ear", "sh"], ["sh", "hip"], ["hip", "knee"], ["knee", "ank"]];

function sidePoints(lm, w, h) {
  const vis = (ks) => ks.reduce((s, k) => s + (lm[IDX[k]].visibility ?? 0), 0);
  const L = vis(["earL", "shL", "hipL", "kneeL", "ankL"]) >= vis(["earR", "shR", "hipR", "kneeR", "ankR"]);
  const s = L ? "L" : "R";
  const pick = (k) => { const l = lm[IDX[k + s]]; return { x: l.x * w, y: l.y * h, v: l.visibility ?? 1 }; };
  const p = { ear: pick("ear"), sh: pick("sh"), hip: pick("hip"), knee: pick("knee"), ank: pick("ank") };
  const nose = lm[IDX.nose], heel = lm[IDX["heel" + s]], toe = lm[IDX["toe" + s]];
  p.nose = { x: nose.x * w, y: nose.y * h, v: nose.visibility ?? 1 };
  // facing direction: nose ahead of ear; fall back to toe ahead of heel
  let dir = Math.sign(nose.x * w - p.ear.x);
  if (Math.abs(nose.x * w - p.ear.x) < w * 0.01) dir = Math.sign((toe.x - heel.x) * w) || 1;
  return { p, dir: dir || 1 };
}

function analyzeSide(p, dir) {
  const f = (a, b) => fromVertical(a, b) * dir; // + = forward
  const H = Math.max(1, p.ank.y - Math.min(p.ear.y, p.nose.y));
  const out = [];

  const head = f(p.sh, p.ear);
  out.push(metric({
    id: "fhp", name: "頭の位置（ストレートネック傾向）", value: head, dev: Math.max(0, head - 3), unit: "°", g: 7, m: 15, signed: true,
    desc: head > 10 ? "頭が肩より前に出ている" : head < -4 ? "頭がやや後ろ寄り" : "耳が肩のほぼ真上にあります",
  }));

  const trunk = f(p.hip, p.sh);
  out.push(metric({
    id: "trunk", name: "上体の傾き（猫背・反り）", value: trunk, unit: "°", g: 4, m: 8, signed: true,
    desc: trunk > 4 ? "上体が前に倒れている（猫背傾向）" : trunk < -4 ? "上体が後ろに反っている" : "腰の上に肩が乗っています",
  }));

  const hip = f(p.ank, p.hip);
  out.push(metric({
    id: "hipshift", name: "骨盤の前後位置", value: hip, dev: hip - 1.5, unit: "°", g: 3, m: 6, signed: true,
    desc: hip > 4.5 ? "骨盤が前に突き出ている（スウェイバック傾向）" : hip < -1.5 ? "骨盤が後ろに引けている（お尻が後ろ）" : "くるぶしの上に骨盤があります",
  }));

  // knee: + = flexed (knee in front of hip–ankle line), - = hyperextended
  const t = (p.knee.y - p.hip.y) / Math.max(1, p.ank.y - p.hip.y);
  const lineX = p.hip.x + (p.ank.x - p.hip.x) * t;
  const legLen = Math.hypot(p.ank.x - p.hip.x, p.ank.y - p.hip.y) || 1;
  const knee = toDeg(Math.atan2(((p.knee.x - lineX) * dir) * 2, legLen));
  out.push(metric({
    id: "kneeSide", name: "膝の伸び具合", value: knee, unit: "°", g: 4, m: 9, signed: true,
    desc: knee > 4 ? "膝が曲がっている" : knee < -4 ? "膝が反りすぎている（反張膝傾向）" : "自然に伸びています",
  }));

  const ear = ((p.ear.x - p.ank.x) * dir / H) * 100;
  const warn = [];
  if (p.ank.v < 0.5) warn.push("足首がはっきり写っていない可能性があります。全身が写った写真を使うと精度が上がります。");
  return { metrics: out, warn, refs: { H, head, trunk, hip, knee, ear } };
}

function postureType(side) {
  if (!side) return null;
  const v = Object.fromEntries(side.metrics.map((m) => [m.id, m.value]));
  const allGood = side.metrics.every((m) => m.level === "good");
  if (allGood) return { name: "理想的なアライメントに近い", text: "耳・肩・骨盤・膝・くるぶしがほぼ一直線に並んでいます。" };
  if (v.hipshift > 4.5 && v.trunk < -2) return { name: "スウェイバック傾向", text: "骨盤が前に出て上体が後ろに反り、骨盤が後傾しやすいタイプです。腰や太もも前の張りにつながりやすい姿勢です。" };
  if (v.trunk < -4) return { name: "反り腰傾向", text: "上体が後ろへ反りやすく、腰椎のカーブが強くなりやすいタイプです（骨盤前傾の可能性）。" };
  if (v.fhp > 10 && v.trunk > 4) return { name: "猫背（円背）傾向", text: "背中が丸まり頭が前に出やすいタイプです。首・肩こりの原因になりやすい姿勢です。" };
  if (v.kneeSide > 4 && v.fhp > 10) return { name: "フラットバック傾向", text: "骨盤が後傾して膝が曲がり、頭が前に出やすいタイプです。" };
  if (v.fhp > 10) return { name: "ストレートネック傾向", text: "頭が肩より前に出ています。首の自然なカーブが減りやすい姿勢です。" };
  if (v.kneeSide < -4) return { name: "反張膝傾向", text: "膝を伸ばしきって立つ癖があります。" };
  return { name: "軽度のアライメントの乱れ", text: "大きな崩れはありませんが、一部に気になるポイントがあります。" };
}

/* ------------------------------------------------------------------ */
/* Advice                                                              */
/* ------------------------------------------------------------------ */
function adviceFor(m) {
  switch (m.id) {
    case "shoulder": return { t: `肩の高さの左右差（${m.high}肩が高い）`, tips: [
      `${m.high}側の首〜肩（僧帽筋上部・肩甲挙筋）のストレッチ：頭を反対側の斜め前に倒して20〜30秒`,
      "バッグをいつも同じ肩に掛ける・頬杖をつくなどの癖を見直す",
      "肩をすくめて5秒→ストンと脱力、を10回" ] };
    case "pelvis": return { t: `骨盤の左右の傾き（${m.high}側が高い）`, tips: [
      `${m.high}側の腰（腰方形筋）のストレッチ：立って${m.high === "右" ? "左" : "右"}側へ体を倒し、脇腹〜腰を伸ばす 20秒×2`,
      "脚を組む・片足に体重をかけて立つ癖を控える",
      "お尻の横（中殿筋）の強化：横向きに寝て上の脚を上げ下げ 10回×2（左右）" ] };
    case "head": return { t: "頭の傾き", tips: [
      "首の横（胸鎖乳突筋・斜角筋）のストレッチ：傾いている側と反対へゆっくり倒して20秒",
      "スマホ・PC画面を体の正面、目の高さに置く" ] };
    case "shift": case "pshift": return { t: "左右の重心の偏り", tips: [
      "鏡の前で両足に均等に体重を乗せて立つ練習",
      "片脚立ちバランス 左右30秒ずつ（ふらつく側を多めに）",
      "立ち仕事では左右の足を交互に休めるより、両足で立つ時間を増やす" ] };
    case "knee": return m.value > 0
      ? { t: "O脚傾向", tips: [
        "内ももの強化：膝の間にクッションを挟んで潰す 10秒×10",
        "つま先と膝を正面にそろえたスクワット 10回×2",
        "お尻の奥（外旋筋）のストレッチ：あぐらで前屈 30秒" ] }
      : { t: "X脚傾向", tips: [
        "お尻（中殿筋）の強化：クラムシェル（横向きで膝を開閉）15回×2",
        "スクワットやしゃがむ時に膝が内側に入らないよう意識",
        "内もものストレッチ：開脚して上体を倒す 30秒" ] };
    case "fhp": return { t: "頭が前に出ている（ストレートネック傾向）", tips: [
      "チンタック（あご引き運動）：あごを水平に後ろへ引いて5秒キープ×10回",
      "胸椎の伸展：丸めたバスタオルを肩甲骨の下に置いて仰向けで1〜2分",
      "画面の高さを目線に合わせ、30分に一度は姿勢をリセット" ] };
    case "trunk": return m.value > 0
      ? { t: "上体が前傾している（猫背傾向）", tips: [
        "胸（大胸筋）のストレッチ：壁に手をついて体を反対へひねる 30秒",
        "肩甲骨寄せ：ひじを後ろに引いて肩甲骨を寄せ5秒×10回",
        "座るときは骨盤を立て、坐骨で座る意識" ] }
      : { t: "上体が後ろに反っている", tips: [
        "ドローイン：息を吐きながらお腹を薄くへこませ10秒×10回",
        "股関節の前（腸腰筋）のストレッチ：片膝立ちで前へ体重移動 30秒",
        "みぞおちを少し下げ、肋骨を締める意識で立つ" ] };
    case "hipshift": return m.value > 1.5
      ? { t: "骨盤が前に出ている（スウェイバック傾向）", tips: [
        "腹筋の強化：ドローインやデッドバグ 10回×2",
        "太もも裏（ハムストリングス）のストレッチ 30秒",
        "骨盤を前に突き出して“ぶら下がるように”立つ癖を見直す" ] }
      : { t: "骨盤が後ろに引けている", tips: [
        "股関節の前（腸腰筋）のストレッチ：片膝立ちで前へ体重移動 30秒",
        "お尻（大殿筋）の強化：ヒップリフト 10回×2",
        "かかとだけでなく足裏全体に体重を乗せる意識" ] };
    case "kneeSide": return m.value > 0
      ? { t: "膝が曲がっている", tips: [
        "太もも裏・ふくらはぎのストレッチ 各30秒",
        "股関節の前（腸腰筋）のストレッチで骨盤を起こしやすくする" ] }
      : { t: "膝の反りすぎ（反張膝傾向）", tips: [
        "膝を伸ばしきらず“少しゆるめて”立つ意識",
        "太もも裏の強化：ヒップリフト 10回×2" ] };
  }
  return null;
}

/* ------------------------------------------------------------------ */
/* Drawing                                                             */
/* ------------------------------------------------------------------ */
const COLOR = { good: "#22c55e", mild: "#f59e0b", poor: "#ef4444", ref: "rgba(255,255,255,.85)", plumb: "#22d3ee", bone: "rgba(255,255,255,.9)" };

function dashed(ctx, x1, y1, x2, y2, color, w, dash) {
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = w; ctx.setLineDash(dash);
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  ctx.restore();
}
function line(ctx, a, b, color, w) {
  ctx.save();
  ctx.lineCap = "round";
  ctx.strokeStyle = "rgba(0,0,0,.45)"; ctx.lineWidth = w + 2.5;
  ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  ctx.strokeStyle = color; ctx.lineWidth = w;
  ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  ctx.restore();
}
function extend(a, b, k) {
  const dx = b.x - a.x, dy = b.y - a.y;
  return [{ x: a.x - dx * k, y: a.y - dy * k }, { x: b.x + dx * k, y: b.y + dy * k }];
}
function label(ctx, text, x, y, color, s, align = "left") {
  ctx.save();
  ctx.font = `700 ${Math.round(15 * s)}px system-ui, sans-serif`;
  const w = ctx.measureText(text).width + 12 * s, h = 22 * s;
  let bx = align === "right" ? x - w : align === "center" ? x - w / 2 : x;
  bx = Math.max(4, Math.min(ctx.canvas.width - w - 4, bx));
  const by = Math.max(4, Math.min(ctx.canvas.height - h - 4, y - h / 2));
  ctx.fillStyle = "rgba(10,20,20,.78)";
  ctx.beginPath(); ctx.roundRect(bx, by, w, h, 6 * s); ctx.fill();
  ctx.fillStyle = color; ctx.textBaseline = "middle";
  ctx.fillText(text, bx + 6 * s, by + h / 2 + 0.5);
  ctx.restore();
}
function dot(ctx, p, s, editing, active) {
  ctx.save();
  const r = (editing ? 8 : 5.5) * s;
  ctx.fillStyle = active ? "#22d3ee" : "#fff";
  ctx.strokeStyle = "#0f766e"; ctx.lineWidth = 2.5 * s;
  ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  if (editing) { ctx.strokeStyle = "rgba(255,255,255,.6)"; ctx.lineWidth = 1.5 * s; ctx.beginPath(); ctx.arc(p.x, p.y, r * 2.2, 0, Math.PI * 2); ctx.stroke(); }
  ctx.restore();
}

function drawFront(v) {
  const { ctx, canvas, base, pts, analysis } = v;
  const s = Math.max(canvas.width, canvas.height) / 700;
  ctx.drawImage(base, 0, 0);
  ctx.fillStyle = "rgba(0,0,0,.12)"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  const r = analysis.refs, lv = Object.fromEntries(analysis.metrics.map((m) => [m.id, m.level]));

  // plumb line through ankle centre
  dashed(ctx, r.ankC.x, 0, r.ankC.x, canvas.height, COLOR.plumb, 2 * s, [10 * s, 8 * s]);
  // bones
  for (const [a, b] of FRONT_BONES.slice(3)) line(ctx, pts[a], pts[b], COLOR.bone, 3 * s);
  // horizontal refs + measured lines
  const rows = [
    ["ear", r.earA, r.earB, lv.head, r.er, "頭"],
    ["sh", r.shA, r.shB, lv.shoulder, r.sh, "肩"],
    ["hip", r.hipA, r.hipB, lv.pelvis, r.hp, "骨盤"],
  ];
  for (const [, a, b, level, ang, name] of rows) {
    const c = mid(a, b), half = Math.abs(b.x - a.x) * 0.95 + 20 * s;
    dashed(ctx, c.x - half, c.y, c.x + half, c.y, COLOR.ref, 1.5 * s, [6 * s, 6 * s]);
    const [e1, e2] = extend(a, b, 0.25);
    line(ctx, e1, e2, COLOR[level], 4 * s);
    label(ctx, `${name} ${fmt(Math.abs(ang))}°`, e2.x + 8 * s, e2.y, COLOR[level], s);
  }
  // centre markers
  for (const c of [r.shC, r.hipC]) {
    ctx.save(); ctx.fillStyle = COLOR.plumb; ctx.beginPath(); ctx.arc(c.x, c.y, 5 * s, 0, Math.PI * 2); ctx.fill(); ctx.restore();
  }
  line(ctx, r.shC, r.hipC, "rgba(34,211,238,.7)", 2 * s);
  for (const k of FRONT_KEYS) dot(ctx, pts[k], s, v.editing, v.drag === k);
  const sideL = v.facing === "front" ? "右" : "左", sideR = v.facing === "front" ? "左" : "右";
  label(ctx, sideL, 10 * s, 22 * s, "#fff", s);
  label(ctx, sideR, canvas.width - 10 * s, 22 * s, "#fff", s, "right");
}

function drawSide(v) {
  const { ctx, canvas, base, pts, analysis, dir } = v;
  const s = Math.max(canvas.width, canvas.height) / 700;
  ctx.drawImage(base, 0, 0);
  ctx.fillStyle = "rgba(0,0,0,.12)"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  const lv = Object.fromEntries(analysis.metrics.map((m) => [m.id, m.level]));
  const val = Object.fromEntries(analysis.metrics.map((m) => [m.id, m.value]));

  dashed(ctx, pts.ank.x, 0, pts.ank.x, canvas.height, COLOR.plumb, 2 * s, [10 * s, 8 * s]);
  label(ctx, "重心線", pts.ank.x + 8 * s * dir, 22 * s, COLOR.plumb, s, dir > 0 ? "left" : "right");
  const segLevel = { "ear-sh": lv.fhp, "sh-hip": lv.trunk, "hip-knee": lv.kneeSide, "knee-ank": lv.kneeSide };
  for (const [a, b] of SIDE_BONES) line(ctx, pts[a], pts[b], COLOR[segLevel[`${a}-${b}`]], 4 * s);
  // projection ticks from each point to the plumb line
  for (const k of ["ear", "sh", "hip", "knee"]) dashed(ctx, pts[k].x, pts[k].y, pts.ank.x, pts[k].y, "rgba(255,255,255,.55)", 1.2 * s, [4 * s, 4 * s]);
  for (const k of SIDE_KEYS) dot(ctx, pts[k], s, v.editing, v.drag === k);

  const side = dir > 0 ? "right" : "left";
  const off = 18 * s * dir;
  const sign = (x) => (x > 0 ? "+" : "") + fmt(x);
  label(ctx, `頭 ${sign(val.fhp)}°`, pts.ear.x + off * 1.6, pts.ear.y, COLOR[lv.fhp], s, side === "right" ? "left" : "right");
  label(ctx, `上体 ${sign(val.trunk)}°`, pts.sh.x + off * 1.6, pts.sh.y, COLOR[lv.trunk], s, side === "right" ? "left" : "right");
  label(ctx, `骨盤 ${sign(val.hipshift)}°`, pts.hip.x + off * 1.6, pts.hip.y, COLOR[lv.hipshift], s, side === "right" ? "left" : "right");
  label(ctx, `膝 ${sign(val.kneeSide)}°`, pts.knee.x + off * 1.6, pts.knee.y, COLOR[lv.kneeSide], s, side === "right" ? "left" : "right");
  label(ctx, dir > 0 ? "前 →" : "← 前", dir > 0 ? canvas.width - 10 * s : 10 * s, canvas.height - 22 * s, "#fff", s, dir > 0 ? "right" : "left");
}

/* ------------------------------------------------------------------ */
/* View controller                                                     */
/* ------------------------------------------------------------------ */
const views = {};

function createView(kind) {
  const root = $(`#view-${kind}`);
  const canvas = $("canvas", root);
  const v = {
    kind, root, canvas, ctx: canvas.getContext("2d"),
    facing: "front", editing: false, drag: null,
    base: null, pts: null, origPts: null, dir: 1, analysis: null, thumb: null,
  };
  views[kind] = v;

  const input = $(".file", root);
  const drop = $(".drop", root);
  input.addEventListener("change", () => { if (input.files[0]) loadFile(v, input.files[0]); input.value = ""; });
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("is-over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) loadFile(v, f); });

  $(".change-btn", root).addEventListener("click", () => input.click());
  $(".edit-btn", root).addEventListener("click", () => setEditing(v, !v.editing));
  $(".reset-btn", root).addEventListener("click", () => {
    if (!v.origPts) return;
    v.pts = structuredClone(v.origPts);
    update(v);
  });

  root.querySelectorAll(".seg-btn").forEach((b) => b.addEventListener("click", () => {
    root.querySelectorAll(".seg-btn").forEach((x) => x.classList.toggle("is-on", x === b));
    v.facing = b.dataset.facing;
    if (v.pts) update(v);
  }));

  // dragging
  const toCanvas = (e) => {
    const r = canvas.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * canvas.width, y: ((e.clientY - r.top) / r.height) * canvas.height, k: canvas.width / r.width };
  };
  const keys = () => (kind === "front" ? FRONT_KEYS : SIDE_KEYS);
  canvas.addEventListener("pointerdown", (e) => {
    if (!v.editing || !v.pts) return;
    const p = toCanvas(e);
    let best = null, bd = 28 * p.k;
    for (const k of keys()) { const d = Math.hypot(v.pts[k].x - p.x, v.pts[k].y - p.y); if (d < bd) { bd = d; best = k; } }
    if (!best) return;
    v.drag = best; canvas.setPointerCapture(e.pointerId); e.preventDefault();
    render(v);
  });
  let raf = 0;
  canvas.addEventListener("pointermove", (e) => {
    if (!v.drag) return;
    const p = toCanvas(e);
    v.pts[v.drag].x = Math.max(0, Math.min(canvas.width, p.x));
    v.pts[v.drag].y = Math.max(0, Math.min(canvas.height, p.y));
    if (!raf) raf = requestAnimationFrame(() => { raf = 0; update(v, true); });
  });
  const end = () => { if (v.drag) { v.drag = null; update(v); } };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  return v;
}

function setEditing(v, on) {
  v.editing = on;
  v.root.classList.toggle("editing", on);
  $(".edit-btn", v.root).classList.toggle("is-on", on);
  $(".edit-btn", v.root).textContent = on ? "修正を完了" : "ポイント修正";
  $(".edit-hint", v.root).hidden = !on;
  if (v.pts) render(v);
}

function loadImage(file) {
  return new Promise((res, rej) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); res(img); };
    img.onerror = () => { URL.revokeObjectURL(url); rej(new Error("画像を読み込めませんでした")); };
    img.src = url;
  });
}

async function loadFile(v, file) {
  if (!file.type.startsWith("image/")) { alert("画像ファイルを選択してください。"); return; }
  let img;
  try { img = await loadImage(file); } catch (e) { alert(e.message + "（HEIC形式の場合はJPEGで保存し直してください）"); return; }
  const k = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
  const base = document.createElement("canvas");
  base.width = Math.round(img.naturalWidth * k);
  base.height = Math.round(img.naturalHeight * k);
  base.getContext("2d").drawImage(img, 0, 0, base.width, base.height);

  v.base = base; v.pts = null; v.analysis = null;
  v.canvas.width = base.width; v.canvas.height = base.height;
  v.ctx.drawImage(base, 0, 0);
  v.root.classList.add("has-image");
  $(".stage", v.root).hidden = false;
  $(".metrics", v.root).innerHTML = "";
  setEditing(v, false);
  const busy = $(".busy", v.root);
  busy.hidden = false;

  try {
    const res = await detect(base);
    const lm = res.landmarks?.[0];
    if (!lm) throw new Error("noperson");
    if (v.kind === "front") v.pts = frontPoints(lm, base.width, base.height);
    else { const r = sidePoints(lm, base.width, base.height); v.pts = r.p; v.dir = r.dir; }
    v.origPts = structuredClone(v.pts);
    v.thumb = makeThumb(base);
    update(v);
  } catch (e) {
    console.error(e);
    v.analysis = null;
    $(".metrics", v.root).innerHTML = `<li class="metric-warn">${e.message === "noperson"
      ? "人物を検出できませんでした。全身がはっきり写った写真でお試しください。"
      : "分析に失敗しました。もう一度お試しください。"}</li>`;
    updateResult();
  } finally {
    busy.hidden = true;
  }
}

function makeThumb(base) {
  const h = 220, w = Math.round((base.width / base.height) * h);
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  c.getContext("2d").drawImage(base, 0, 0, w, h);
  return c.toDataURL("image/jpeg", 0.6);
}

function update(v, live = false) {
  v.analysis = v.kind === "front" ? analyzeFront(v.pts, v.facing) : analyzeSide(v.pts, v.dir);
  render(v);
  renderMetrics(v);
  if (!live) updateResult();
}

function render(v) { if (v.analysis) (v.kind === "front" ? drawFront : drawSide)(v); }

function renderMetrics(v) {
  const ul = $(".metrics", v.root);
  const items = v.analysis.metrics.map((m) => {
    const val = m.signed ? (m.value > 0 ? "+" : "") + fmt(m.value) : fmt(m.value);
    return `<li class="metric ${m.level}">
      <span class="metric-name">${m.name}<span class="badge ${m.level}">${LEVEL_LABEL[m.level]}</span></span>
      <span class="metric-val">${val}${m.unit}</span>
      <span class="metric-desc">${m.desc}</span></li>`;
  });
  const warns = v.analysis.warn.map((w) => `<li class="metric-warn">⚠ ${w}</li>`);
  ul.innerHTML = warns.join("") + items.join("");
}

/* ------------------------------------------------------------------ */
/* Overall result                                                      */
/* ------------------------------------------------------------------ */
const WEIGHT = { shoulder: 1.2, pelvis: 1.5, head: 0.8, shift: 0.8, pshift: 1, knee: 0.8, fhp: 1.4, trunk: 1.2, hipshift: 1.2, kneeSide: 0.7 };
let current = null;
let lastSavedId = null;

function computeOverall() {
  const all = [views.front.analysis, views.side.analysis].filter(Boolean).flatMap((a) => a.metrics);
  if (!all.length) return null;
  let sw = 0, ss = 0;
  for (const m of all) { const w = WEIGHT[m.id] ?? 1; sw += w; ss += m.score * w; }
  return { score: Math.round(ss / sw), metrics: all, type: postureType(views.side.analysis) };
}

function scoreLabel(s) {
  if (s >= 90) return "とても良い姿勢です";
  if (s >= 75) return "おおむね良い姿勢です";
  if (s >= 60) return "やや姿勢の崩れがあります";
  return "姿勢の崩れが目立ちます";
}
const scoreColor = (s) => (s >= 75 ? "var(--good)" : s >= 60 ? "var(--mild)" : "var(--poor)");

function updateResult() {
  lastSavedId = null;
  const box = $("#result");
  current = computeOverall();
  if (!current) { box.hidden = true; return; }
  box.hidden = false;
  const { score, metrics, type } = current;
  $("#scoreNum").textContent = score;
  $("#scoreLabel").textContent = scoreLabel(score);
  const fg = $("#ringFg");
  fg.style.strokeDashoffset = String(326.7 * (1 - score / 100));
  fg.style.stroke = scoreColor(score);

  const parts = [];
  if (type) parts.push(`側面の推定タイプ：<b>${type.name}</b> — ${type.text}`);
  if (!views.front.analysis) parts.push("正面の写真を追加すると、肩・骨盤の左右差も分析できます。");
  if (!views.side.analysis) parts.push("側面の写真を追加すると、猫背・反り腰・ストレートネックの傾向も分析できます。");
  $("#postureType").innerHTML = parts.join("<br>");

  const bad = metrics.filter((m) => m.level !== "good").sort((a, b) => a.score - b.score);
  $("#findings").innerHTML = bad.length
    ? bad.map((m) => `<li><span class="badge ${m.level}">${LEVEL_LABEL[m.level]}</span> <b>${m.name}</b>：${m.desc}</li>`).join("")
    : '<li class="ok">✓ 大きな崩れは見られません。今の姿勢を保ちましょう。</li>';

  const seen = new Set();
  const adv = bad.map(adviceFor).filter((a) => a && !seen.has(a.t) && seen.add(a.t)).slice(0, 4);
  $("#advice").innerHTML = adv.length
    ? adv.map((a) => `<div class="advice-item"><h4>${a.t}</h4><ul>${a.tips.map((t) => `<li>${t}</li>`).join("")}</ul></div>`).join("")
    : '<div class="advice-item"><h4>姿勢キープのために</h4><ul><li>1時間に1回は立ち上がって体を動かす</li><li>深呼吸をしながら胸を開くストレッチ</li><li>定期的に写真を撮って変化をチェック</li></ul></div>';

  renderCompare();
}

function renderCompare() {
  const el = $("#compare");
  const name = $("#clientName").value.trim();
  const hist = loadHistory();
  const prev = name ? hist.find((h) => h.name === name && h.id !== lastSavedId) : null;
  if (!prev || !current) { el.hidden = true; return; }
  const d = current.score - prev.score;
  el.hidden = false;
  el.innerHTML = `前回（${new Date(prev.date).toLocaleDateString("ja-JP")}）：${prev.score}点 → 今回：${current.score}点 ` +
    (d === 0 ? "（変化なし）" : `<span class="${d > 0 ? "up" : "down"}">${d > 0 ? "▲" : "▼"}${Math.abs(d)}</span>`);
}

/* ------------------------------------------------------------------ */
/* History                                                             */
/* ------------------------------------------------------------------ */
function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch { return []; }
}
function saveHistory(list) {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(list)); return true; } catch { return false; }
}

function renderHistory() {
  const list = loadHistory();
  const ul = $("#historyList");
  $("#clearHistory").hidden = !list.length;
  if (!list.length) { ul.innerHTML = '<li class="empty">まだ保存された結果はありません。</li>'; return; }
  ul.innerHTML = list.map((h) => `
    <li class="h-item">
      <div class="h-thumbs">${[h.thumbFront, h.thumbSide].filter(Boolean).map((t) => `<img src="${t}" alt="">`).join("")}</div>
      <div class="h-meta"><span class="h-name">${esc(h.name || "（名前なし）")}</span><span class="h-score" style="color:${scoreColor(h.score)}">${h.score}<small>点</small></span></div>
      <div class="h-date">${new Date(h.date).toLocaleString("ja-JP", { dateStyle: "medium", timeStyle: "short" })}${h.type ? "・" + esc(h.type) : ""}</div>
      ${h.memo ? `<div class="h-memo">${esc(h.memo)}</div>` : ""}
      <button type="button" class="btn ghost small h-del" data-id="${h.id}">削除</button>
    </li>`).join("");
}
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

/* ------------------------------------------------------------------ */
/* Report image                                                        */
/* ------------------------------------------------------------------ */
function downloadReport() {
  if (!current) return;
  const cans = [views.front, views.side].filter((v) => v.analysis).map((v) => v.canvas);
  const H = 900, head = 150, pad = 24;
  const ws = cans.map((c) => Math.round((c.width / c.height) * H));
  const W = Math.max(900, ws.reduce((a, b) => a + b, 0) + pad * (cans.length + 1));
  const out = document.createElement("canvas");
  out.width = W; out.height = head + H + pad * 2 + 40;
  const g = out.getContext("2d");
  g.fillStyle = "#f4f7f6"; g.fillRect(0, 0, out.width, out.height);
  g.fillStyle = "#0f766e"; g.fillRect(0, 0, W, 6);
  g.fillStyle = "#17312e"; g.font = "700 34px system-ui, sans-serif";
  g.fillText("AI姿勢チェック 分析結果", pad, 58);
  g.font = "400 20px system-ui, sans-serif"; g.fillStyle = "#5d7270";
  const name = $("#clientName").value.trim(), memo = $("#clientMemo").value.trim();
  g.fillText([new Date().toLocaleString("ja-JP"), name, memo].filter(Boolean).join("  /  "), pad, 94);
  if (current.type) g.fillText("推定タイプ：" + current.type.name, pad, 126);
  g.textAlign = "right"; g.fillStyle = "#17312e"; g.font = "800 64px system-ui, sans-serif";
  g.fillText(String(current.score), W - pad - 70, 100);
  g.font = "400 22px system-ui, sans-serif"; g.fillStyle = "#5d7270"; g.fillText("/100点", W - pad, 100);
  g.textAlign = "left";
  let x = pad;
  cans.forEach((c, i) => { g.drawImage(c, x, head + pad, ws[i], H); x += ws[i] + pad; });
  g.font = "400 15px system-ui, sans-serif"; g.fillStyle = "#5d7270";
  g.fillText("※ 写真から推定した参考値であり、医学的な診断ではありません。", pad, out.height - 18);
  out.toBlob((blob) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `posture-${new Date().toISOString().slice(0, 10)}${name ? "-" + name.replace(/[\\/:*?"<>|\s]+/g, "_") : ""}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }, "image/png");
}

/* ------------------------------------------------------------------ */
/* Init                                                                */
/* ------------------------------------------------------------------ */
createView("front");
createView("side");
renderHistory();

$("#clientName").addEventListener("input", renderCompare);
$("#printBtn").addEventListener("click", () => window.print());
$("#reportBtn").addEventListener("click", downloadReport);
$("#saveBtn").addEventListener("click", () => {
  if (!current) return;
  const list = loadHistory();
  list.unshift({
    id: Date.now().toString(36),
    date: new Date().toISOString(),
    name: $("#clientName").value.trim(),
    memo: $("#clientMemo").value.trim(),
    score: current.score,
    type: current.type?.name || "",
    metrics: current.metrics.map((m) => ({ id: m.id, value: +fmt(m.value, 2), level: m.level })),
    thumbFront: views.front.analysis ? views.front.thumb : null,
    thumbSide: views.side.analysis ? views.side.thumb : null,
  });
  lastSavedId = list[0].id;
  if (!saveHistory(list.slice(0, 60))) { alert("保存できませんでした（ブラウザの保存容量が不足しているか、プライベートモードの可能性があります）。"); return; }
  renderHistory();
  const b = $("#saveBtn"); b.textContent = "保存しました ✓"; b.disabled = true;
  setTimeout(() => { b.textContent = "履歴に保存"; b.disabled = false; }, 1800);
});
$("#historyList").addEventListener("click", (e) => {
  const id = e.target.closest(".h-del")?.dataset.id;
  if (!id || !confirm("この履歴を削除しますか？")) return;
  saveHistory(loadHistory().filter((h) => h.id !== id));
  renderHistory(); renderCompare();
});
$("#clearHistory").addEventListener("click", () => {
  if (!confirm("すべての履歴を削除しますか？この操作は元に戻せません。")) return;
  saveHistory([]); renderHistory(); renderCompare();
});

// warm up the model as soon as the user shows intent
["pointerdown", "focusin"].forEach((t) => $("#analyze").addEventListener(t, () => getLandmarker(), { once: true }));
