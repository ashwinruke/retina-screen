/* In-browser DR grading.
 *
 * The browser does the first preprocessing steps exactly like
 * src/retina/preprocess.py (crop the black border, pad to a square, resize
 * with area averaging like cv2.INTER_AREA). Everything after that (Ben Graham
 * normalisation, the network, flip TTA and Grad-CAM) is inside model.onnx,
 * exported by scripts/export_onnx.py.
 */
"use strict";

const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
const ADVICE = [
  "Routine rescreening in 12 months.",
  "Rescreen in 6 to 12 months.",
  "Refer to an ophthalmologist.",
  "Urgent referral to an ophthalmologist.",
  "Urgent referral to an ophthalmologist.",
];
const VIEW_NOTES = {
  photo: "The photo cropped to the retina, as the model receives it before colour normalisation.",
  heatmap: "Warmer areas raised the severity score more (Grad-CAM).",
  seen: "After colour normalisation, which evens out lighting so small lesions stand out.",
};

const $ = (id) => document.getElementById(id);
const state = { session: null, meta: null, result: null, view: "photo" };

// ---------------------------------------------------------------- preprocessing

/** cv2.cvtColor(RGB2GRAY) for 8-bit images: fixed-point weights, rounded. */
function gray(r, g, b) {
  return (r * 4899 + g * 9617 + b * 1868 + 8192) >> 14;
}

/** Bounding box of pixels brighter than the threshold (preprocess.crop_to_fundus). */
function fundusBox(rgba, w, h, threshold) {
  const rowAny = new Uint8Array(h);
  const colAny = new Uint8Array(w);
  let count = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      if (gray(rgba[i], rgba[i + 1], rgba[i + 2]) > threshold) {
        rowAny[y] = 1; colAny[x] = 1; count++;
      }
    }
  }
  if (count < 0.01 * w * h) return { x0: 0, y0: 0, x1: w, y1: h, found: false };
  const first = (a) => a.indexOf(1);
  const last = (a) => a.length - 1 - [...a].reverse().indexOf(1);
  return { x0: first(colAny), y0: first(rowAny), x1: last(colAny) + 1, y1: last(rowAny) + 1, found: true };
}

/** Per-destination-index source ranges and weights for cv2.INTER_AREA (downscaling). */
function areaWeights(src, dst) {
  const scale = src / dst;
  const out = [];
  for (let d = 0; d < dst; d++) {
    const f1 = d * scale, f2 = f1 + scale;
    const s1 = Math.ceil(f1), s2 = Math.floor(f2);
    const cell = Math.min(scale, src - f1);
    const taps = [];
    if (s1 - f1 > 1e-3) taps.push([s1 - 1, (s1 - f1) / cell]);
    for (let s = s1; s < s2; s++) taps.push([s, 1 / cell]);
    if (f2 - s2 > 1e-3 && s2 < src) taps.push([s2, Math.min(Math.min(f2 - s2, 1), cell) / cell]);
    out.push(taps);
  }
  return out;
}

/** Bilinear weights with OpenCV's half-pixel convention (used when enlarging). */
function linearWeights(src, dst) {
  const scale = src / dst;
  const out = [];
  for (let d = 0; d < dst; d++) {
    let f = (d + 0.5) * scale - 0.5;
    let s = Math.floor(f);
    let t = f - s;
    if (s < 0) { s = 0; t = 0; }
    if (s >= src - 1) { s = src - 1; t = 0; }
    out.push(t === 0 ? [[s, 1]] : [[s, 1 - t], [s + 1, t]]);
  }
  return out;
}

/**
 * Crop to the retina, pad to a square (centred, black) and resize to size x size.
 * Returns RGB bytes (size * size * 3) and whether a retina was found.
 */
function cropPadResize(rgba, w, h, size, threshold) {
  const box = fundusBox(rgba, w, h, threshold);
  const cw = box.x1 - box.x0, ch = box.y1 - box.y0;
  const side = Math.max(cw, ch);
  const ox = Math.floor((side - cw) / 2), oy = Math.floor((side - ch) / 2);
  const weights = side >= size ? areaWeights(side, size) : linearWeights(side, size);

  // Horizontal pass over the padded square: rows (side) x size x 3, as floats.
  const tmp = new Float32Array(side * size * 3);
  for (let y = 0; y < side; y++) {
    const sy = y - oy;
    if (sy < 0 || sy >= ch) continue; // padding row: stays 0
    const rowBase = ((box.y0 + sy) * w + box.x0) * 4;
    for (let dx = 0; dx < size; dx++) {
      let r = 0, g = 0, b = 0;
      for (const [sx, wt] of weights[dx]) {
        const cx = sx - ox;
        if (cx < 0 || cx >= cw) continue; // padding column
        const i = rowBase + cx * 4;
        r += rgba[i] * wt; g += rgba[i + 1] * wt; b += rgba[i + 2] * wt;
      }
      const o = (y * size + dx) * 3;
      tmp[o] = r; tmp[o + 1] = g; tmp[o + 2] = b;
    }
  }
  // Vertical pass, then round like cv2's saturate_cast.
  const out = new Uint8ClampedArray(size * size * 3);
  for (let dy = 0; dy < size; dy++) {
    for (let x = 0; x < size; x++) {
      let r = 0, g = 0, b = 0;
      for (const [sy, wt] of weights[dy]) {
        const o = (sy * size + x) * 3;
        r += tmp[o] * wt; g += tmp[o + 1] * wt; b += tmp[o + 2] * wt;
      }
      const o = (dy * size + x) * 3;
      out[o] = Math.round(r); out[o + 1] = Math.round(g); out[o + 2] = Math.round(b);
    }
  }
  return { rgb: out, found: box.found };
}

function toTensor(rgb, size) {
  const n = size * size;
  const data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    data[i] = rgb[i * 3];
    data[n + i] = rgb[i * 3 + 1];
    data[2 * n + i] = rgb[i * 3 + 2];
  }
  return data;
}

// ---------------------------------------------------------------- drawing

/** OpenCV's COLORMAP_JET, as RGB. */
function jet(v) {
  const c = (x) => Math.round(255 * Math.min(1, Math.max(0, x)));
  return [c(1.5 - Math.abs(4 * v - 3)), c(1.5 - Math.abs(4 * v - 2)), c(1.5 - Math.abs(4 * v - 1))];
}

function drawRGB(rgb, size) {
  const canvas = $("canvas");
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    img.data[i * 4] = rgb[i * 3];
    img.data[i * 4 + 1] = rgb[i * 3 + 1];
    img.data[i * 4 + 2] = rgb[i * 3 + 2];
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function heatmapOverlay(rgb, cam, alpha = 0.45) {
  const out = new Uint8ClampedArray(rgb.length);
  for (let i = 0; i < cam.length; i++) {
    const [hr, hg, hb] = jet(cam[i]);
    out[i * 3] = Math.round(rgb[i * 3] * (1 - alpha) + hr * alpha);
    out[i * 3 + 1] = Math.round(rgb[i * 3 + 1] * (1 - alpha) + hg * alpha);
    out[i * 3 + 2] = Math.round(rgb[i * 3 + 2] * (1 - alpha) + hb * alpha);
  }
  return out;
}

function planarToRGB(planar, size) {
  const n = size * size;
  const out = new Uint8ClampedArray(3 * n);
  for (let i = 0; i < n; i++) {
    out[i * 3] = planar[i]; out[i * 3 + 1] = planar[n + i]; out[i * 3 + 2] = planar[2 * n + i];
  }
  return out;
}

function showView(view) {
  const r = state.result;
  if (!r) return;
  state.view = view;
  const images = { photo: r.photo, heatmap: r.heatmap, seen: r.seen };
  drawRGB(images[view], r.size);
  $("canvas").setAttribute("aria-label", { photo: "Fundus photo", heatmap: "Grad-CAM heatmap over the photo", seen: "Colour-normalised model input" }[view]);
  $("view-note").textContent = VIEW_NOTES[view];
}

const svg = (tag, attrs, text) => {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text !== undefined) el.textContent = text;
  return el;
};

/** 0-4 track split at the tuned cut points, with the score as a needle. */
function drawScale(score, thresholds) {
  // Draw at the real on-screen width so text stays the same size on phones.
  const W = Math.max(260, Math.round($("scale").clientWidth) || 400), L = 8, R = W - 8;
  const x = (v) => L + (Math.min(4, Math.max(0, v)) / 4) * (R - L);
  const root = svg("svg", { viewBox: `0 0 ${W} 74`, role: "img",
    "aria-label": `Severity score ${score.toFixed(2)} on a scale from 0 to 4` });
  const edges = [0, ...thresholds, 4];
  for (let g = 0; g < 5; g++) {
    const x0 = x(edges[g]), x1 = x(edges[g + 1]);
    const rect = svg("rect", { x: x0, y: 30, width: Math.max(0, x1 - x0), height: 18, class: "seg" });
    rect.style.fill = `var(--g${g})`;
    root.append(rect);
    if (x1 - x0 > 16) {
      const label = svg("text", { x: (x0 + x1) / 2, y: 43, "text-anchor": "middle", class: "seg-label" }, String(g));
      label.style.fill = g >= 3 ? "var(--seg-dark-text, #ffffff)" : "var(--ink)";
      root.append(label);
    }
  }
  edges.forEach((v, i) => {
    const anchor = i === 0 ? "start" : i === edges.length - 1 ? "end" : "middle";
    root.append(svg("text", { x: x(v), y: 68, "text-anchor": anchor, class: "tick-label" },
      i === 0 || i === edges.length - 1 ? String(v) : v.toFixed(2)));
  });
  // Needle: a marker above the band and a tick below it, so it never covers a grade number.
  const needle = svg("g", { class: "needle" });
  needle.append(svg("path", { d: "M -6 18 L 6 18 L 0 27 Z" }));
  needle.append(svg("line", { x1: 0, y1: 48, x2: 0, y2: 55 }));
  const sx = x(score);
  const anchor = sx < 40 ? "start" : sx > W - 40 ? "end" : "middle";
  needle.append(svg("text", { x: 0, y: 12, "text-anchor": anchor }, score.toFixed(2)));
  root.append(needle);
  $("scale").replaceChildren(root);
  needle.style.transform = `translateX(${x(0)}px)`;
  requestAnimationFrame(() => requestAnimationFrame(() => { needle.style.transform = `translateX(${sx}px)`; }));
}

const ICON_REFER = '<svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true"><path d="M10 2.5 18.5 17h-17L10 2.5Z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M10 8v4.2M10 14.4v.1" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>';
const ICON_CLEAR = '<svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="8" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="m6.3 10.3 2.5 2.5 5-5.3" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>';

function showResult(r) {
  const { meta } = state;
  const grade = meta.thresholds.filter((t) => r.score >= t).length;
  const referable = grade >= meta.referable_grade;
  $("grade-name").textContent = `Grade ${grade}: ${meta.grade_names[grade]}`;
  $("grade-sub").textContent = `Severity score ${r.score.toFixed(2)} out of 4`;
  drawScale(r.score, meta.thresholds);
  const verdict = $("verdict");
  verdict.className = `verdict ${referable ? "refer" : "clear"}`;
  verdict.innerHTML = `${referable ? ICON_REFER : ICON_CLEAR}<p><strong>${referable ? "Referable diabetic retinopathy" : "No referral needed"}</strong><span>${ADVICE[grade]}</span></p>`;
  const warn = $("warn");
  warn.hidden = r.found;
  warn.textContent = r.found ? "" : "No retina was found in this photo, so this result is unreliable. Use a colour fundus photograph.";
  $("state-status").hidden = true;
  $("state-result").hidden = false;
  state.result = { ...r, grade };
  document.querySelectorAll('input[name="view"]').forEach((el) => { el.disabled = false; });
  showView(state.view);
  if (window.matchMedia("(max-width: 820px)").matches) {
    const smooth = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    $("state-result").scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "start" });
  }
}

// ---------------------------------------------------------------- flow

function setStatus(text, isError = false) {
  $("state-status").hidden = false;
  $("state-result").hidden = true;
  const s = $("status");
  s.textContent = text;
  s.classList.toggle("error", isError);
}

async function fetchWithProgress(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} returned ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const buf = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf.set(value, got);
    got += value.length;
    $("status").textContent = `Loading the model (${(got / 2 ** 20).toFixed(0)} of ${(total / 2 ** 20).toFixed(0)} MB)…`;
    $("meter").firstElementChild.style.width = `${(100 * got) / total}%`;
  }
  return buf;
}

async function loadModel() {
  try {
    ort.env.wasm.wasmPaths = window.ORT_WASM_PATH || ORT_DIST;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    state.meta = await (await fetch("model.json")).json();
    const m = state.meta.metrics;
    if (m) {
      $("model-facts").textContent =
        `${state.meta.model_name} trained on ${m.train_data}. ` +
        `On ${m.n_test} held-out test photos it scored a quadratic weighted kappa of ${m.qwk.toFixed(3)}, ` +
        `and found referable retinopathy with ${Math.round(m.sensitivity * 100)}% sensitivity and ` +
        `${Math.round(m.specificity * 100)}% specificity (ROC-AUC ${m.auc.toFixed(3)}).`;
    }
    const bytes = await fetchWithProgress("model.onnx");
    $("status").textContent = "Preparing the model…";
    $("meter").firstElementChild.style.width = "100%";
    state.session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    $("meter").hidden = true;
    setStatus("Ready. Add a fundus photo to grade it.");
  } catch (err) {
    console.error(err);
    setStatus(`The model didn't load (${err.message}). Check your connection and reload the page.`, true);
  }
}

async function grade(file) {
  if (!file) return;
  if (!state.session) { setStatus("The model is still loading. Try again in a moment."); return; }
  let bitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    setStatus("This file couldn't be opened as an image. Use a JPG or PNG photo.", true);
    return;
  }
  setStatus("Grading…");
  $("lens").classList.add("has-image");
  await new Promise((r) => setTimeout(r, 30)); // let the status paint before the heavy work

  const work = document.createElement("canvas");
  work.width = bitmap.width; work.height = bitmap.height;
  const wctx = work.getContext("2d", { willReadFrequently: true });
  wctx.drawImage(bitmap, 0, 0);
  const { data } = wctx.getImageData(0, 0, bitmap.width, bitmap.height);
  const size = state.meta.input_size;
  const t0 = performance.now();
  const { rgb, found } = cropPadResize(data, bitmap.width, bitmap.height, size, state.meta.crop_threshold);
  const input = new ort.Tensor("float32", toTensor(rgb, size), [1, 3, size, size]);
  try {
    const out = await state.session.run({ image: input });
    const result = {
      size, found,
      score: out.score.data[0],
      photo: rgb,
      heatmap: heatmapOverlay(rgb, out.cam.data),
      seen: planarToRGB(out.seen.data, size),
      ms: performance.now() - t0,
    };
    window.retinaDebug = { input: rgb, score: result.score, cam: out.cam.data, ms: result.ms };
    showResult(result);
  } catch (err) {
    console.error(err);
    setStatus(`Grading failed (${err.message}).`, true);
  }
}

function init() {
  for (const id of ["file", "file2"]) {
    $(id).addEventListener("change", (e) => { grade(e.target.files[0]); e.target.value = ""; });
  }
  document.querySelectorAll('input[name="view"]').forEach((el) =>
    el.addEventListener("change", () => showView(el.value)));
  const lens = $("lens");
  lens.addEventListener("dragover", (e) => { e.preventDefault(); lens.classList.add("dragging"); });
  lens.addEventListener("dragleave", () => lens.classList.remove("dragging"));
  lens.addEventListener("drop", (e) => {
    e.preventDefault();
    lens.classList.remove("dragging");
    grade(e.dataTransfer.files[0]);
  });
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (state.result) drawScale(state.result.score, state.meta.thresholds);
    }, 150);
  });
  loadModel();
}

init();
