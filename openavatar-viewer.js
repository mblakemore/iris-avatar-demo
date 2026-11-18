/*
 * OpenAvatar viewer: a dependency-free ES module (no imports, no build step) that renders a Gaussian-splat
 * avatar with WebGL 2 and drives it from audio.
 *
 * The renderer, the head-turn rig, the camera and the loudness / frequency-band mouth driver are ported
 * from the project owner's own prototype viewer. Changes from the prototype: page globals and URL
 * parameters became options, several viewers can live on one page, the audio-lookup-table path was
 * dropped, and idle motion, presets, a second ("timed") driver and a format-version check were added.
 *
 *   const v = await OpenAvatarViewer.create(canvas, { modelUrl, rig?, background?, pointer?, on? });
 *
 * Model file layout (little endian): u32 header length, JSON header, then the arrays listed in
 * header.fields. The header may carry `version` (number or "major.minor"); a missing field means 1. A file
 * whose major version is higher than FORMAT_MAJOR is refused (error code "version").
 */

export const FORMAT_MAJOR = 1;

// ------------------------------------------------------------------ small helpers
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const ss = x => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };
const rnd = (a, b) => a + (b - a) * Math.random();
const rndRange = r => rnd(r[0], r[1]);
const DEG = Math.PI / 180;
const isObj = o => o && typeof o === "object" && !Array.isArray(o);
function mergeDeep(base, add) {                                  // arrays and scalars are replaced, objects merged; returns a new object
  const out = Array.isArray(base) ? base.slice() : { ...base };
  for (const k of Object.keys(add || {})) out[k] = isObj(out[k]) && isObj(add[k]) ? mergeDeep(out[k], add[k]) : (isObj(add[k]) ? mergeDeep({}, add[k]) : Array.isArray(add[k]) ? add[k].slice() : add[k]);
  return out;
}
function fail(code, message) { const e = new Error(message); e.code = code; return e; }

// ------------------------------------------------------------------ preset (plain JSON)
export const DEFAULT_PRESET = {
  version: 1,
  driver: {
    name: "loudness",                      // "loudness" | "timed" | any registered driver
    gain: 1,                               // loudness driver: overall mouth opening (prototype "gain")
    smoothing: 0.045,                      // seconds, one-pole smoothing of the mouth weights
    maxPerControl: {},                     // e.g. {"ah": 0.7}; missing = 1
    clampToTrained: true,                  // keep speech weights inside the range seen in training (prototype behaviour)
    timed: { attack: 0.05, release: 0.10, timeOffset: 0, minVisible: 0.07, tongueMin: 0.07 }   // seconds
  },
  idle: {
    enabled: true,
    gaze: { enabled: true, mode: "saccade",          // "saccade" (design default) or "sine" (the prototype's slow drift)
            amount: 0.06,                            // control units around the rest value (prototype: 0.06)
            intervalS: [0.6, 2], shiftS: [0.08, 0.15],
            aversion: { enabled: true, amount: 0.15, everySpeakingS: [4, 8], everyListeningS: [8, 15], durationS: [1, 3] } },
    sway: { enabled: true, yawDeg: 3, tiltDeg: 1.2, periodS: [3, 6], limitFraction: 0.8 },   // fraction of the rig's limit that sway may use
    blink: { enabled: true, intervalIdleS: 3.5, intervalSpeakingS: 2.3, minS: 0.8, maxS: 8, closeS: 0.08, holdS: 0.04, openS: 0.16, doubleChance: 0.12 }
  },
  speech: { swayWhileSpeaking: true, emphasisDeg: 1, blinkFaster: true, aversionsWhileSpeaking: true }
};

// ------------------------------------------------------------------ model file
let H2F = null;
function halfTable() {
  if (H2F) return H2F; H2F = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    H2F[h] = e === 0 ? s * m * 5.960464477539063e-8 : e === 31 ? (m ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + m / 1024);
  }
  return H2F;
}
function parseModel(buf) {
  if (buf.length < 8) throw fail("format", "Not an OpenAvatar model file (too short).");
  const hl = new DataView(buf.buffer, buf.byteOffset, buf.byteLength).getUint32(0, true);
  if (hl < 2 || 4 + hl > buf.length) throw fail("format", "Not an OpenAvatar model file (bad header length).");
  let hdr; try { hdr = JSON.parse(new TextDecoder().decode(buf.subarray(4, 4 + hl))); } catch (e) { throw fail("format", "Not an OpenAvatar model file (header is not JSON)."); }
  const ver = hdr.version ?? hdr.format_version ?? 1, major = parseInt(String(ver), 10);
  if (!(major >= 1)) throw fail("format", "Model file has an unreadable format version: " + JSON.stringify(ver));
  if (major > FORMAT_MAJOR) throw fail("version", `This model file uses format version ${ver}; this viewer supports major version ${FORMAT_MAJOR} and older. Update the viewer to open it.`);
  if (!Array.isArray(hdr.fields) || !(hdr.n > 0)) throw fail("format", "Model header has no fields.");
  const f16 = halfTable();
  let off = 4 + hl; const d = { hdr, n: hdr.n, Km: hdr.Km, Kc: hdr.Kc, Ko: hdr.Ko, frames: hdr.frames, fps: hdr.fps, version: ver };
  for (const [name, type, shape] of hdr.fields) {
    const cnt = hdr.n * shape.reduce((a, b) => a * b, 1), bytes = cnt * (type === "f4" ? 4 : 2);
    if (off + bytes > buf.length) throw fail("format", "Model file is truncated (field " + name + ").");
    const raw = buf.slice(off, off + bytes).buffer; off += bytes;
    if (type === "f4") d[name] = new Float32Array(raw);
    else { const u = new Uint16Array(raw), f = new Float32Array(cnt); for (let i = 0; i < cnt; i++) f[i] = f16[u[i]]; d[name] = f; }
  }
  return d;
}
async function fetchModel(url, onProgress) {
  const resp = await fetch(url);
  if (!resp.ok) throw fail("http", `Could not load the model (${resp.status} ${resp.statusText}) from ${url}`);
  const total = +resp.headers.get("content-length") || 0;
  if (!resp.body || !resp.body.getReader) return new Uint8Array(await resp.arrayBuffer());
  const rd = resp.body.getReader(), parts = []; let got = 0;
  for (;;) { const { done, value } = await rd.read(); if (done) break; parts.push(value); got += value.length; if (total && onProgress) onProgress(got / total); }
  const buf = new Uint8Array(got); let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; } return buf;
}

// ------------------------------------------------------------------ GL shaders (unchanged from the prototype)
const VS = `#version 300 es
precision highp float;
layout(location=0) in vec2 aQuad; layout(location=1) in vec3 aCenter; layout(location=2) in vec3 aC0; layout(location=3) in vec3 aC1; layout(location=4) in vec4 aColor;
uniform mat4 uView; uniform vec2 uFocal; uniform vec2 uViewport; uniform vec2 uPP;
out vec4 vColor; out vec2 vPos;
void main() {
  vec4 cam = uView * vec4(aCenter, 1.0);
  if (cam.z < 0.2) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  mat3 W = mat3(uView);
  mat3 V = mat3(aC0.x, aC0.y, aC0.z, aC0.y, aC1.x, aC1.y, aC0.z, aC1.y, aC1.z);
  mat3 S = W * V * transpose(W);
  float iz = 1.0 / cam.z;
  vec3 j0 = vec3(uFocal.x * iz, 0.0, -uFocal.x * cam.x * iz * iz), j1 = vec3(0.0, uFocal.y * iz, -uFocal.y * cam.y * iz * iz);
  float a = dot(j0, S * j0) + 0.3, b = dot(j0, S * j1), c = dot(j1, S * j1) + 0.3;
  float mid = 0.5 * (a + c), rad = length(vec2(0.5 * (a - c), b));
  float l1 = mid + rad, l2 = max(mid - rad, 0.1);
  vec2 dg = rad < 1e-6 ? vec2(1.0, 0.0) : normalize(vec2(b, l1 - a));
  vec2 maj = min(sqrt(2.0 * l1), 1024.0) * dg, mnr = min(sqrt(2.0 * l2), 1024.0) * vec2(dg.y, -dg.x);
  vec2 ctr = vec2(2.0 * (uFocal.x * cam.x * iz + uPP.x) / uViewport.x - 1.0, 1.0 - 2.0 * (uFocal.y * cam.y * iz + uPP.y) / uViewport.y);
  vec2 px = aQuad.x * maj + aQuad.y * mnr;
  gl_Position = vec4(ctr + vec2(px.x * 2.0 / uViewport.x, -px.y * 2.0 / uViewport.y), 0.0, 1.0);
  vColor = aColor; vPos = aQuad;
}`;
const FS = `#version 300 es
precision highp float; in vec4 vColor; in vec2 vPos; out vec4 o;
void main() { float A = -dot(vPos, vPos); if (A < -4.0) discard; o = vec4(vColor.rgb, vColor.a * exp(A)); }`;

// ------------------------------------------------------------------ 3x3 maths (row-major), as in the prototype
const C0 = 0.28209479177387814, C1 = 0.4886025119029199, S2a = 1.0925484305920792, S2c = 0.31539156525252005, S2e = 0.5462742152960396;
function rodrigues(ax, ang) {
  const l = Math.hypot(...ax), x = ax[0] / l, y = ax[1] / l, z = ax[2] / l, c = Math.cos(ang), s = Math.sin(ang), t = 1 - c;
  return [t * x * x + c, t * x * y - s * z, t * x * z + s * y, t * x * y + s * z, t * y * y + c, t * y * z - s * x, t * x * z - s * y, t * y * z + s * x, t * z * z + c];
}
const mul3 = (A, B) => { const R = []; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) R.push(A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j]); return R; };
const mv3 = (A, v) => [A[0] * v[0] + A[1] * v[1] + A[2] * v[2], A[3] * v[0] + A[4] * v[1] + A[5] * v[2], A[6] * v[0] + A[7] * v[1] + A[8] * v[2]];

// ------------------------------------------------------------------ audio analysis (ported: FFT, mel bands, spectral shape, viseme heuristic)
const NFFT = 2048, NB = 20, WIN_S = 0.040, VSIL = -7.5;
const fRe = new Float32Array(NFFT), fIm = new Float32Array(NFFT);
function fft(re, im) {
  const n = re.length; for (let i = 1, j = 0; i < n; i++) { let b = n >> 1; for (; j & b; b >>= 1) j ^= b; j ^= b; if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (let len = 2; len <= n; len <<= 1) { const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) { let cr = 1, ci = 0; for (let j = 0; j < len / 2; j++) { const a = i + j, b = a + len / 2, xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr; re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi; const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t; } } }
}
const bankCache = new Map();
function bankFor(sr) {                                           // mel band edges in FFT bins, and band centres in Hz (the prototype fixed sr = 48000)
  let b = bankCache.get(sr); if (b) return b;
  const mel = f => 2595 * Math.log10(1 + f / 700), inv = m => 700 * (Math.pow(10, m / 2595) - 1), lo = mel(150), hi = mel(6000), pts = [];
  for (let i = 0; i < NB + 2; i++) pts.push(inv(lo + (hi - lo) * i / (NB + 1)) * NFFT / sr);
  b = { pts, hz: Array.from({ length: NB }, (_, k) => pts[k + 1] * sr / NFFT) }; bankCache.set(sr, b); return b;
}
function spectralShape(buf, centre, out, sr) {                   // out[0..NB-1] = log mel band power minus its mean; returns the mean (loudness)
  const bank = bankFor(sr), mb = bank.pts, w = Math.min(NFFT, Math.round(WIN_S * sr)), s0 = Math.round(centre - w / 2); fRe.fill(0); fIm.fill(0);
  for (let i = 0; i < w; i++) { const j = s0 + i; if (j >= 0 && j < buf.length) fRe[i] = buf[j] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (w - 1))); }
  fft(fRe, fIm); let mean = 0;
  for (let b = 0; b < NB; b++) { const a = mb[b], c = mb[b + 1], e = mb[b + 2]; let p = 0; for (let k = Math.ceil(a); k < e; k++) { const wgt = k < c ? (k - a) / (c - a) : (e - k) / (e - c); p += wgt * (fRe[k] * fRe[k] + fIm[k] * fIm[k]); } out[b] = Math.log(p + 1e-10); mean += out[b] / NB; }
  for (let b = 0; b < NB; b++) out[b] -= mean; return mean;
}
// Loudness opens the mouth; where the energy sits picks the shape (first-formant region -> "ah", strong 1.7-3.2 kHz -> "ee",
// energy held below ~550 Hz -> "oo", hiss above 3.5 kHz -> teeth together). A heuristic, not a phoneme recogniser.
function viseme(feat, erel, out, set, bandHz, gain) {
  out.fill(0); if (erel < VSIL) return;
  let tot = 0, lo = 0, a = 0, e = 0, hs = 0;
  for (let b = 0; b < NB; b++) { const hz = bandHz[b]; if (hz < 300) continue; const p = Math.exp(feat[b]); tot += p; if (hz < 550) lo += p; else if (hz >= 600 && hz < 1300) a += p; else if (hz >= 1700 && hz < 3200) e += p; else if (hz >= 3500) hs += p; }   // below 300 Hz is the voice's pitch, not the mouth shape
  lo /= tot; a /= tot; e /= tot; hs /= tot;
  const g = gain * ss((erel - VSIL) / 5.5);
  const hiss = ss((hs - 0.3) / 0.3), vee = ss((e - 0.08) / 0.2) * (1 - hiss), vah = ss((a - 0.16) / 0.3) * (1 - hiss) * (1 - 0.6 * vee), voo = ss((lo - 0.55) / 0.3) * (1 - vee) * (1 - vah) * (1 - hiss);
  set("ah", g * 0.8 * vah); set("wide", g * 0.3 * vah * ss((erel + 2) / 2)); set("ee", g * (0.75 * vee + 0.4 * hiss)); set("oo", g * 0.7 * voo);
  set("mid", g * (0.3 + 0.4 * Math.max(0, 1 - vah - vee - voo - hiss)));                  // any voiced sound parts the lips a little
}

// ------------------------------------------------------------------ audio sources (shared by the drivers)
const elementTaps = new WeakMap();                               // an element can be wired into a context only once, ever
function audioContextFor() { const AC = globalThis.AudioContext || globalThis.webkitAudioContext; if (!AC) throw fail("audio", "This browser has no Web Audio."); return new AC(); }
function makeSource(src, opts) {
  let ctx, analyser, el = null, node = null, tapped;
  if (typeof HTMLMediaElement !== "undefined" && src instanceof HTMLMediaElement) {
    el = src; tapped = elementTaps.get(el);
    if (!tapped) { const c = audioContextFor(), s = c.createMediaElementSource(el); s.connect(c.destination); tapped = { ctx: c, node: s }; elementTaps.set(el, tapped); }
    ctx = tapped.ctx; node = tapped.node;
    const resume = () => { if (ctx.state === "suspended") ctx.resume().catch(() => {}); }; el.addEventListener("play", resume); resume();
  } else if (src && src.context && typeof src.connect === "function") { node = src; ctx = src.context; }
  else throw fail("audio", "attachAudio needs an HTMLAudioElement / HTMLVideoElement or a Web Audio node.");
  analyser = ctx.createAnalyser(); analyser.fftSize = 2048; analyser.smoothingTimeConstant = 0; node.connect(analyser);   // a tap: not connected to the output
  const startTime = opts && typeof opts.startTime === "number" ? opts.startTime : null;
  const latency = () => (ctx.outputLatency || ctx.baseLatency || 0);
  return {
    el, node, ctx, analyser, buf: new Float32Array(analyser.fftSize), sr: ctx.sampleRate,
    playing: () => (el ? !el.paused && !el.ended : ctx.state === "running"),
    // time of the sound that is audible now, in the clip's own time: the audio clock minus the output latency
    mediaTime() { return el ? el.currentTime - latency() : ctx.currentTime - latency() - (startTime || 0); },
    hasClock: !!el || startTime !== null,
    detach() { try { node.disconnect(analyser); } catch (e) { /* already gone */ } }
  };
}

// ------------------------------------------------------------------ mouth drivers: a small interface
// A driver factory is called as factory(env) and returns { attach(source), detach(), update(dt, out), speaking, level, setData?(d) }.
//   env.controls  names of all controls; env.Ks number of speech controls; env.index(name) -> speech index or -1
//   env.settings() -> the live preset.driver object (gain, smoothing, timed settings)
// update(dt, out) writes the speech weights (length Ks, 0..1) for this frame. Register more with registerDriver(name, factory).
function loudnessDriver(env) {
  let src = null, peak = -20; const tmp = new Float32Array(NB), cur = new Float32Array(env.Ks), st = new Float32Array(env.Ks); let speaking = false, level = 0;
  const set = (n, v) => { const i = env.index(n); if (i >= 0) cur[i] = clamp(v, 0, 1); };
  return {
    name: "loudness",
    attach(s) { src = s; peak = -20; st.fill(0); }, detach() { src = null; st.fill(0); speaking = false; },
    get speaking() { return speaking; }, get level() { return level; },
    update(dt, out) {
      const S = env.settings();
      if (!src) { out.fill(0); speaking = false; return; }
      src.analyser.getFloatTimeDomainData(src.buf); const w = Math.round(WIN_S * src.sr), tail = src.buf.length - w;
      let ss2 = 0; for (let i = Math.max(0, tail); i < src.buf.length; i++) ss2 += src.buf[i] * src.buf[i];
      const rms = Math.sqrt(ss2 / Math.min(w, src.buf.length));
      let erel;
      if (rms < 2e-4) erel = -99;                                // digital silence: no running-peak reference to compare with
      else { const e = spectralShape(src.buf, src.buf.length - w / 2, tmp, src.sr); peak = Math.max(e, peak - dt * 0.3); erel = e - peak; }
      viseme(tmp, erel, cur, set, bankFor(src.sr).hz, S.gain);
      const a = dt > 0 ? 1 - Math.exp(-dt / Math.max(1e-3, S.smoothing)) : 1; for (let j = 0; j < st.length; j++) st[j] += a * (cur[j] - st[j]);
      out.set(st); speaking = erel > VSIL; level = speaking ? ss((erel - VSIL) / 5.5) : 0;
    }
  };
}

// ---- timed driver ------------------------------------------------------------------------------------------
// FIRST CUT, UNTUNED, UNTESTED against real speech by eye. The mechanism is complete (a track of events played against the
// audio clock, attack / release smoothing, the p/b/m rule, pauses) but every number is the proposal from
// RESEARCH_LIPSYNC_DESIGN.md and the viseme weights below are a guess to be tuned by eye. Known gaps: no diphthong splitting,
// no loudness scaling of the vowel opening, rounded shapes do not yet override neighbouring tongue-only consonants, Kokoro
// `phoneme` timings are not mapped (only characters are).
//
// Track item: { viseme: "pbm" | "aa" | ... , start, end }  or  { weights: {ah: 0.8, ...}, start, end }   (seconds, audio time)
const VISEME_WEIGHTS = {                                          // UNTUNED: controls named as in the rig; unknown controls are ignored
  sil: {},
  pbm: { press: 1.0 },                                            // p b m: closes fully
  fv: { press: 0.35, mid: 0.3, ee: 0.15 },                        // poor fit: no lip-under-teeth shape
  th: { mid: 0.6, ah: 0.1 },                                      // poor fit: no tongue
  td: { mid: 0.6, ee: 0.15 },
  kg: { mid: 0.5, ah: 0.2 },
  sh: { oo: 0.45, mid: 0.4 },                                     // ch j sh
  sz: { ee: 0.5, mid: 0.3 },
  n: { mid: 0.5 },
  l: { ah: 0.35, mid: 0.3 },                                      // poor fit: tongue only
  r: { oo: 0.35, mid: 0.35 },
  aa: { wide: 0.6, ah: 0.4 },                                     // father, cat (stressed)
  ah: { ah: 0.8 },                                                // cut (unstressed)
  eh: { ah: 0.5, ee: 0.3 },
  schwa: { mid: 0.6, ah: 0.2 },
  ih: { ee: 0.8, mid: 0.2 },                                      // i, y
  oh: { oo: 0.55, ah: 0.45 },
  uw: { oo: 0.9, mid: 0.1 }                                       // u, w
};
const VISEME_KIND = { pbm: "plosive", th: "tongue", td: "tongue", kg: "tongue", n: "tongue", l: "tongue", oh: "round", uw: "round", sh: "round", r: "round", sil: "none" };
const kindOf = v => VISEME_KIND[v] || (["aa", "ah", "eh", "schwa", "ih"].includes(v) ? "vowel" : "cons");

// APPROXIMATE letter -> viseme mapping for ElevenLabs character timings. English spelling is not phonetic; this is a crude
// rule set (digraphs, a few vowel pairs, silent final e) and no substitute for a grapheme-to-phoneme step.
const LETTER = { a: "aa", e: "eh", i: "ih", o: "oh", u: "uw", y: "ih", p: "pbm", b: "pbm", m: "pbm", f: "fv", v: "fv", t: "td", d: "td", k: "kg", g: "kg", c: "kg", q: "kg", x: "kg", j: "sh", s: "sz", z: "sz", n: "n", l: "l", r: "r", w: "uw" };
function lettersToVisemes(timings) {
  const T = (timings || []).map(t => ({ c: String(t.char ?? "").toLowerCase(), start: +t.start, end: +t.end })), out = [], isL = c => c.length === 1 && c >= "a" && c <= "z";
  for (let i = 0; i < T.length; i++) {
    const c = T[i].c; if (!isL(c)) continue;
    const nx = i + 1 < T.length ? T[i + 1].c : "", pair = c + nx; let v = null, span = 1;
    if (pair === "th") { v = "th"; span = 2; } else if (pair === "sh" || pair === "ch") { v = "sh"; span = 2; } else if (pair === "ph") { v = "fv"; span = 2; }
    else if (pair === "ng") { v = "kg"; span = 2; } else if (pair === "oo" || pair === "ou" || pair === "ow") { v = "uw"; span = 2; } else if (pair === "ee" || pair === "ea") { v = "ih"; span = 2; }
    else if (c === "h") v = null;                                 // no shape of its own
    else if (c === "e" && !isL(nx) && i >= 2 && isL(T[i - 1].c) && isL(T[i - 2].c)) v = null;   // silent final e of a word of 3+ letters
    else v = LETTER[c] || null;
    if (v && Number.isFinite(T[i].start) && Number.isFinite(T[i + span - 1].end)) out.push({ viseme: v, start: T[i].start, end: T[i + span - 1].end });
    i += span - 1;
  }
  return out;
}

function timedDriver(env) {
  let src = null, events = [], cur = new Float32Array(env.Ks), target = new Float32Array(env.Ks), lastEnd = 0, speaking = false, fallback = null, hasTrack = false;
  const fb = () => { if (!fallback) { fallback = loudnessDriver(env); if (src) fallback.attach(src); } return fallback; };   // no track (or no audio clock): the loudness driver
  const wOf = (item) => { const w = item.weights || VISEME_WEIGHTS[item.viseme] || {}, arr = []; for (const k of Object.keys(w)) { const i = env.index(k); if (i >= 0) arr.push([i, w[k]]); } return arr; };
  function compile(items) {
    const T = env.settings().timed, L = (items || []).filter(it => Number.isFinite(it.start) && Number.isFinite(it.end)).map(it => ({ ...it })).sort((a, b) => a.start - b.start);
    const M = [];                                                 // merge identical neighbours: first duration + 0.7 * second
    for (const it of L) { const p = M[M.length - 1]; if (p && it.viseme && p.viseme === it.viseme && it.start <= p.end + 0.02) p.end = p.start + (p.end - p.start) + 0.7 * (it.end - it.start); else M.push(it); }
    const ev = [];
    M.forEach((it, i) => {
      let kind = it.weights ? "vowel" : kindOf(it.viseme), dur = it.end - it.start;
      if (kind === "none" || (!it.weights && it.viseme === "sil")) return;
      if (kind === "tongue" && dur < T.tongueMin) return;         // tongue-only consonants this short get no shape of their own
      if (kind !== "plosive" && dur < T.minVisible) dur = T.minVisible;
      const prevDur = i > 0 ? M[i - 1].end - M[i - 1].start : 0.1;
      const e = { w: wOf(it), kind, start: it.start, dur, atk: 0.05, hold: it.start + dur, rel: 0.08 };
      if (kind === "vowel") { e.atk = Math.min(0.12, dur * 2 / 3); e.hold = it.start + 0.75 * dur; e.rel = 0.12; }
      else if (kind === "round") { e.atk = 0.15; e.rel = 0.15; e.hold = it.start + 0.75 * dur; }
      else if (kind === "plosive") { e.atk = clamp(prevDur / 2, 0.04, 0.12); e.hold = it.start + Math.max(dur, 0.04); e.rel = 0.06; }
      ev.push(e);
    });
    ev.sort((a, b) => a.start - b.start);
    const pauses = [];                                            // gaps: under 120 ms hold the shape, to 350 ms relax to mid 0.2, longer rest
    for (let i = 0; i + 1 < ev.length; i++) {
      const a = ev[i], b = ev[i + 1];
      const aEnd = a.hold, bStart = b.start - b.atk, g = bStart - aEnd;
      if (g > 0 && g < 0.12) a.hold = bStart;
      else if (g >= 0.12 && g <= 0.35) { const mi = env.index("mid"); if (mi >= 0) pauses.push({ w: [[mi, 0.2]], kind: "pause", start: aEnd, dur: g, atk: 0.08, hold: bStart, rel: 0.08 }); }
    }
    events = ev.concat(pauses); lastEnd = events.reduce((m, e) => Math.max(m, e.hold + e.rel), 0);
  }
  const envAt = (e, t) => { const s = e.start - e.atk; if (t < s) return 0; if (t < e.start) return ss((t - s) / e.atk); if (t <= e.hold) return 1; if (t < e.hold + e.rel) return 1 - ss((t - e.hold) / e.rel); return 0; };
  const pressIdx = () => env.index("press"), ahIdx = () => env.index("ah"), wideIdx = () => env.index("wide");
  return {
    name: "timed",
    attach(s) { src = s; if (fallback) fallback.attach(s); cur.fill(0); }, detach() { src = null; if (fallback) fallback.detach(); cur.fill(0); speaking = false; },
    get speaking() { return hasTrack ? speaking : (fallback ? fallback.speaking : false); }, get level() { return fallback ? fallback.level : 0; },
    // data: { track } or { timings: [{char,start,end}] } or null (falls back to the loudness driver)
    setData(d) {
      hasTrack = false; events = [];
      if (d && Array.isArray(d.track)) { compile(d.track); hasTrack = true; }
      else if (d && Array.isArray(d.timings) && d.timings.length) { compile(lettersToVisemes(d.timings)); hasTrack = true; }
      if (!hasTrack && src) fb();
    },
    update(dt, out) {
      if (!hasTrack || !src || !src.hasClock) { fb().update(dt, out); return; }
      const T = env.settings().timed;
      if (!src.playing()) { target.fill(0); speaking = false; }
      else {
        const t = src.mediaTime() + (T.timeOffset || 0); target.fill(0); speaking = false;
        for (const e of events) { const v = envAt(e, t); if (v <= 0) continue; if (e.kind !== "pause") speaking = true; for (const [i, w] of e.w) target[i] = Math.max(target[i], v * w); }
        let P = 0; const pi = pressIdx(); if (pi >= 0) P = target[pi];
        if (P > 0) { const a = ahIdx(), w = wideIdx(); if (a >= 0) target[a] *= 1 - P; if (w >= 0) target[w] *= 1 - P; }   // closed lips: no opening while p/b/m is on
      }
      const pi = pressIdx();
      for (let j = 0; j < cur.length; j++) {
        const rising = target[j] > cur[j], tau = rising ? T.attack : T.release;
        if (j === pi && rising) { cur[j] = target[j]; continue; }                         // the closure is never blended away
        const a = dt > 0 ? 1 - Math.exp(-dt / Math.max(1e-3, tau)) : 1; cur[j] += a * (target[j] - cur[j]);
      }
      out.set(cur);
    },
    debug: () => ({ events: events.length, lastEnd, hasTrack })
  };
}
const driverRegistry = { loudness: loudnessDriver, timed: timedDriver };

// ------------------------------------------------------------------ the viewer
async function create(canvas, options = {}) {
  const handlers = { frame: [], loaded: [], error: [], progress: [] };
  const on = (ev, fn) => { (handlers[ev] || (handlers[ev] = [])).push(fn); return () => { const a = handlers[ev], i = a.indexOf(fn); if (i >= 0) a.splice(i, 1); }; };
  const emit = (ev, arg) => { for (const fn of (handlers[ev] || []).slice()) { try { fn(arg); } catch (e) { console.error(e); } } };
  for (const k of Object.keys(options.on || {})) on(k, options.on[k]);
  const failWith = e => { emit("error", e); return e; };

  const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, preserveDrawingBuffer: true });
  if (!gl) throw failWith(fail("webgl2", "This browser has no WebGL 2."));

  // ---- load
  let D;
  try {
    const src = options.model instanceof ArrayBuffer ? new Uint8Array(options.model) : options.model instanceof Uint8Array ? options.model : await fetchModel(options.modelUrl, f => emit("progress", f));
    D = parseModel(src);
  } catch (e) { throw failWith(e); }
  const hasRig = !!D.hdr.rig, rig = Object.assign({}, D.hdr.rig || {}, options.rig || {});
  const controlNames = Array.isArray(rig.controls) ? rig.controls.slice() : [], named = hasRig && controlNames.length > 0;
  const Km = D.Km, Ks = rig.Ks || 0, Kh = rig.Kh || 0;

  // ---- GL setup (ported)
  const mk = (t, s) => { const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x); if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x)); return x; };
  const P = gl.createProgram(); gl.attachShader(P, mk(gl.VERTEX_SHADER, VS)); gl.attachShader(P, mk(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(P);
  if (!gl.getProgramParameter(P, gl.LINK_STATUS)) throw failWith(fail("webgl2", gl.getProgramInfoLog(P))); gl.useProgram(P);
  const uView = gl.getUniformLocation(P, "uView"), uFocal = gl.getUniformLocation(P, "uFocal"), uViewport = gl.getUniformLocation(P, "uViewport"), uPP = gl.getUniformLocation(P, "uPP");
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const qb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, qb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, -2, 2, 2, 2]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  const ib = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, ib);
  const STRIDE = 40;
  for (const [loc, size, type, norm, offs] of [[1, 3, gl.FLOAT, false, 0], [2, 3, gl.FLOAT, false, 12], [3, 3, gl.FLOAT, false, 24], [4, 4, gl.UNSIGNED_BYTE, true, 36]]) {
    gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, type, norm, STRIDE, offs); gl.vertexAttribDivisor(loc, 1);
  }
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  const nG = D.n, inst = new ArrayBuffer(nG * STRIDE), inst32 = new Float32Array(inst), inst8 = new Uint8Array(inst);
  gl.bufferData(gl.ARRAY_BUFFER, nG * STRIDE, gl.DYNAMIC_DRAW);
  const vx = new Float32Array(nG), vy = new Float32Array(nG), vz = new Float32Array(nG), vcol = new Uint8Array(nG * 4), vidx = new Int32Array(nG), vdep = new Float32Array(nG), counts = new Uint32Array(65536);
  const bm = new Float32Array(Km), bc = new Float32Array(D.Kc), bo = new Float32Array(D.Ko);

  // ---- pose and limits
  let limits = null, tele = 1;
  const lo = options.limits || {};
  { const yawDeg = lo.yawDeg ?? rig.yaw_limit ?? (named ? 35 : null); if (yawDeg !== null && +yawDeg > 0) limits = { yaw: +yawDeg * DEG, pitch: (lo.pitchDeg ?? rig.pitch_limit ?? 12) * DEG, negFade: rig.pitch_neg_fade || null }; }
  tele = +(options.tele ?? rig.tele ?? 1);
  // ---- ORBIT (new): the whole bust is one rigid body and the camera goes round it. rig.orbit = { min, max (deg; omitted = full circle), axisZ (m, model frame), followHead, followMaxDeg, exprFadeDeg: [from, to] }
  const OR = rig.orbit || null, orbitOn = !!OR, orbitFull = orbitOn && (OR.min === undefined || OR.max === undefined || OR.max - OR.min >= 359.9);
  const orb = orbitOn ? { min: orbitFull ? -Infinity : OR.min * DEG, max: orbitFull ? Infinity : OR.max * DEG, axisZ: OR.axisZ ?? -0.07, follow: options.followHead ?? OR.followHead ?? true, followMax: (OR.followMaxDeg ?? 30) * DEG, fade: (OR.exprFadeDeg || [40, 70]).map(x => x * DEG), followFade: (OR.followFadeDeg || [60, 100]).map(x => x * DEG) } : null;
  const base = { yaw: 0, pitch: 0, zoom: 1, orbit: 0 }, eff = { yaw: 0, pitch: 0, zoom: 1, orbit: 0, face: 0, exprGain: 1 };
  const clampAngles = (yaw, pitch) => {                          // the prototype's clampView
    if (!limits) return [yaw, pitch];
    yaw = clamp(yaw, -limits.yaw, limits.yaw); const nf = limits.negFade, lw = nf ? limits.pitch * clamp((nf[1] - yaw) / (nf[1] - nf[0]), 0, 1) : limits.pitch;
    return [yaw, clamp(pitch, -lw, limits.pitch)];
  };

  // ---- head-turn mode (ported)
  let head = null;
  function headSetup() {
    const hw = new Float32Array(nG), H = rig.head || {}, y0 = H.y0 ?? 1.325, dy = H.dy ?? 0.055;
    if (D.hw) hw.set(D.hw); else for (let i = 0; i < nG; i++) { const t = Math.min(1, Math.max(0, (D.mean[i * 3 + 1] - y0) / dy)); hw[i] = t * t * (3 - 2 * t); }
    head = { hw, covV: new Float32Array(D.cov), piv: H.pivot || [-0.0145, 1.36, 0.085], zfix: !!H.zfix, tilt: H.tilt || null, Q: [1, 0, 0, 0, 1, 0, 0, 0, 1], key: "0,0" };
  }
  if (named && options.head !== false) headSetup();
  function headCov() {
    const key = eff.yaw.toFixed(4) + "," + eff.pitch.toFixed(4); if (key === head.key) return; head.key = key;
    const Q = head.Q, c = D.cov, o = head.covV, hw = head.hw, m00 = Q[0], m01 = Q[3], m02 = Q[6], m10 = Q[1], m11 = Q[4], m12 = Q[7], m20 = Q[2], m21 = Q[5], m22 = Q[8];
    for (let i = 0; i < nG; i++) { if (hw[i] < 0.5) continue; const j = i * 6, a = c[j], b = c[j + 1], cc_ = c[j + 2], d = c[j + 3], e = c[j + 4], f = c[j + 5];
      const r0x = m00 * a + m01 * b + m02 * cc_, r0y = m00 * b + m01 * d + m02 * e, r0z = m00 * cc_ + m01 * e + m02 * f, r1x = m10 * a + m11 * b + m12 * cc_, r1y = m10 * b + m11 * d + m12 * e, r1z = m10 * cc_ + m11 * e + m12 * f, r2x = m20 * a + m21 * b + m22 * cc_, r2y = m20 * b + m21 * d + m22 * e, r2z = m20 * cc_ + m21 * e + m22 * f;
      o[j] = r0x * m00 + r0y * m01 + r0z * m02; o[j + 1] = r0x * m10 + r0y * m11 + r0z * m12; o[j + 2] = r0x * m20 + r0y * m21 + r0z * m22; o[j + 3] = r1x * m10 + r1y * m11 + r1z * m12; o[j + 4] = r1x * m20 + r1y * m21 + r1z * m22; o[j + 5] = r2x * m20 + r2y * m21 + r2z * m22; }
  }

  // ---- camera (ported): orbit around the head pivot in OpenCV convention; in head mode the camera stays and the head turns
  function cameraMatrices() {
    const c = (hasRig && rig.cam_c2w) || D.hdr.cam_c2w, Rc = [c[0][0], c[0][1], c[0][2], c[1][0], c[1][1], c[1][2], c[2][0], c[2][1], c[2][2]], tc = [c[0][3], c[1][3], c[2][3]], Pv = D.hdr.pivot;
    const Ry = rodrigues([0, 1, 0], eff.yaw), right = mv3(Ry, [Rc[0], Rc[3], Rc[6]]); let Q = mul3(rodrigues(right, eff.pitch), Ry);
    if (head) {
      if (head.tilt) {                                           // follow the trained pose path: the few degrees of pitch/roll the reference stills have at this yaw
        const T = head.tilt, u = Math.min(T.e.length - 1, Math.max(0, (eff.yaw - T.y0) / T.dy)), i0 = Math.min(T.e.length - 2, Math.floor(u)), f = u - i0, a = T.e[i0], b = T.e[i0 + 1];
        const e = [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])], ang = Math.hypot(e[0], e[1], e[2]);
        if (ang > 1e-6) Q = mul3(Q, rodrigues([e[0] / ang, e[1] / ang, e[2] / ang], -ang));
      }
      head.Q = Q; Q = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    }
    const Qo = orbitOn ? rodrigues([0, 1, 0], eff.orbit) : null;            // ORBIT: the camera goes round the vertical axis through the bust
    const R = orbitOn ? mul3(mul3(Qo, Q), Rc) : mul3(Q, Rc), e = [Pv[0] - tc[0], Pv[1] - tc[1], (orbitOn ? orb.axisZ : Pv[2]) - tc[2]], z = eff.zoom;
    const pc = [(Rc[0] * e[0] + Rc[3] * e[1] + Rc[6] * e[2]) * z, (Rc[1] * e[0] + Rc[4] * e[1] + Rc[7] * e[2]) * z, (Rc[2] * e[0] + Rc[5] * e[1] + Rc[8] * e[2]) * z * tele], d = mv3(R, pc), t = [Pv[0] - d[0], Pv[1] - d[1], (orbitOn ? orb.axisZ : Pv[2]) - d[2]];
    const V = new Float32Array(16); for (let i = 0; i < 3; i++) { for (let j = 0; j < 3; j++) V[j * 4 + i] = R[j * 3 + i]; V[12 + i] = -(R[i] * t[0] + R[3 + i] * t[1] + R[6 + i] * t[2]); } V[15] = 1;
    return { V, pos: t };
  }

  // ---- per-frame splat evaluation + depth sort on the CPU (ported)
  let rigW = hasRig ? new Float32Array(Km) : null, phase = 0;
  function evaluate(tau, cam, V) {
    const { n, Km: Kmm, Kc, Ko, mean, dc, op, rest, mot, cc, oc } = D, cov = head ? head.covV : D.cov, hQ = head && head.Q, hW = head && head.hw, hP = head && head.piv; if (head) headCov();
    if (hasRig) { bm.set(rigW); bc.set(rigW); bo.set(rigW); }   // live rig: the same linear model, weights set at run time
    else { for (let k = 0; k < Kmm; k++) bm[k] = Math.cos(Math.PI * (k + 1) * tau); for (let k = 0; k < Kc; k++) bc[k] = Math.cos(Math.PI * (k + 1) * tau); for (let k = 0; k < Ko; k++) bo[k] = Math.cos(Math.PI * (k + 1) * tau); }
    const r2x = V[2], r2y = V[6], r2z = V[10], t2 = V[14]; let m = 0, dmin = 1e9, dmax = -1e9;
    // refB models were trained with the head turning about the model origin plus an in-image shift; shifting the head along the view direction by the difference restores the trained front-to-back order at the collar
    let zfx = 0, zfy = 0, zfz = 0; if (head && head.zfix) { const qx = hQ[0] * hP[0] + hQ[3] * hP[1] + hQ[6] * hP[2], qy = hQ[1] * hP[0] + hQ[4] * hP[1] + hQ[7] * hP[2], qz = hQ[2] * hP[0] + hQ[5] * hP[1] + hQ[8] * hP[2], sft = r2x * (hP[0] - qx) + r2y * (hP[1] - qy) + r2z * (hP[2] - qz); zfx = -r2x * sft; zfy = -r2y * sft; zfz = -r2z * sft; }
    for (let i = 0; i < n; i++) {
      const i3 = i * 3; let x = mean[i3], y = mean[i3 + 1], z = mean[i3 + 2];
      let o = i * Kmm * 3; for (let k = 0; k < Kmm; k++) { const b = bm[k]; x += b * mot[o++]; y += b * mot[o++]; z += b * mot[o++]; }
      let ov = op[i]; o = i * Ko; for (let k = 0; k < Ko; k++) ov += bo[k] * oc[o + k];
      const a = 1 / (1 + Math.exp(-ov)); if (a < 0.004) continue;
      const hwi = head ? hW[i] : 0; if (hwi > 0) { const px = x - hP[0], py = y - hP[1], pz = z - hP[2]; x += hwi * (hP[0] + hQ[0] * px + hQ[3] * py + hQ[6] * pz - x + zfx); y += hwi * (hP[1] + hQ[1] * px + hQ[4] * py + hQ[7] * pz - y + zfy); z += hwi * (hP[2] + hQ[2] * px + hQ[5] * py + hQ[8] * pz - z + zfz); }
      const dep = r2x * x + r2y * y + r2z * z + t2; if (dep < 0.2) continue;
      let cr = dc[i3], cg = dc[i3 + 1], cb = dc[i3 + 2]; o = i * Kc * 3; for (let k = 0; k < Kc; k++) { const b = bc[k]; cr += b * cc[o++]; cg += b * cc[o++]; cb += b * cc[o++]; }
      let dx = x - cam[0], dy = y - cam[1], dz = z - cam[2]; const il = 1 / (Math.hypot(dx, dy, dz) + 1e-9); dx *= il; dy *= il; dz *= il;
      if (hwi > 0) { let ex = hQ[0] * dx + hQ[1] * dy + hQ[2] * dz, ey = hQ[3] * dx + hQ[4] * dy + hQ[5] * dz, ez = hQ[6] * dx + hQ[7] * dy + hQ[8] * dz;   // view direction in the head's own frame
        if (D.hw) { if (hwi < 1) { ex = dx + hwi * (ex - dx); ey = dy + hwi * (ey - dy); ez = dz + hwi * (ez - dz); const l = 1 / (Math.hypot(ex, ey, ez) + 1e-9); ex *= l; ey *= l; ez *= l; } dx = ex; dy = ey; dz = ez; }
        else if (hwi >= 0.5) { dx = ex; dy = ey; dz = ez; } }
      const xx = dx * dx, yy = dy * dy, zz = dz * dz;
      const B = [-C1 * dy, C1 * dz, -C1 * dx, S2a * dx * dy, -S2a * dy * dz, S2c * (2 * zz - xx - yy), -S2a * dx * dz, S2e * (xx - yy)];
      let r = C0 * cr + 0.5, g = C0 * cg + 0.5, bl = C0 * cb + 0.5; const ro = i * 24;
      for (let j = 0; j < 8; j++) { const w = B[j]; r += w * rest[ro + j * 3]; g += w * rest[ro + j * 3 + 1]; bl += w * rest[ro + j * 3 + 2]; }
      vx[m] = x; vy[m] = y; vz[m] = z; vidx[m] = i; vdep[m] = dep; const c4 = m * 4;
      vcol[c4] = r < 0 ? 0 : r > 1 ? 255 : r * 255; vcol[c4 + 1] = g < 0 ? 0 : g > 1 ? 255 : g * 255; vcol[c4 + 2] = bl < 0 ? 0 : bl > 1 ? 255 : bl * 255; vcol[c4 + 3] = a * 255;
      if (dep < dmin) dmin = dep; if (dep > dmax) dmax = dep; m++;
    }
    counts.fill(0); const sc = 65535 / Math.max(dmax - dmin, 1e-6);   // counting sort, far -> near
    for (let j = 0; j < m; j++) { const k = 65535 - ((vdep[j] - dmin) * sc | 0); vdep[j] = k; counts[k]++; }
    let acc = 0; for (let k = 0; k < 65536; k++) { const c = counts[k]; counts[k] = acc; acc += c; }
    for (let j = 0; j < m; j++) {
      const s = counts[vdep[j]]++, f = s * 10, ci = vidx[j] * 6;
      inst32[f] = vx[j]; inst32[f + 1] = vy[j]; inst32[f + 2] = vz[j];
      inst32[f + 3] = cov[ci]; inst32[f + 4] = cov[ci + 1]; inst32[f + 5] = cov[ci + 2]; inst32[f + 6] = cov[ci + 3]; inst32[f + 7] = cov[ci + 4]; inst32[f + 8] = cov[ci + 5];
      const b = s * STRIDE + 36, c4 = j * 4; inst8[b] = vcol[c4]; inst8[b + 1] = vcol[c4 + 1]; inst8[b + 2] = vcol[c4 + 2]; inst8[b + 3] = vcol[c4 + 3];
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, ib); gl.bufferSubData(gl.ARRAY_BUFFER, 0, inst8, 0, m * STRIDE); return m;
  }

  // ---- controls
  const stats = (() => {                                         // range of each rig component seen in training
    if (!named || !rig.track) return null; const tr = rig.track, K = tr[0].length, lo_ = [], hi_ = [];
    for (let j = 0; j < K; j++) { let a = Infinity, b = -Infinity; for (const r of tr) { if (r[j] < a) a = r[j]; if (r[j] > b) b = r[j]; } lo_.push(a); hi_.push(b); } return { lo: lo_, hi: hi_ };
  })();
  const ctlInfo = controlNames.map((name, index) => {
    const speech = index < Ks, rest = rig.rest ? rig.rest[index] : 0;
    return { name, index, kind: speech ? "speech" : "free", min: speech ? 0 : (stats ? stats.lo[index] : -1), max: speech ? 1 : (stats ? stats.hi[index] : 1), rest, trained: stats ? [stats.lo[index], stats.hi[index]] : null };
  });
  const ctlIndex = n => controlNames.indexOf(n);
  const manual = {};                                             // explicit values from setControls; they override the driver and idle motion
  let bg = options.background || "#000000";

  // ---- preset, drivers
  let preset = mergeDeep(DEFAULT_PRESET, {});
  const env = { controls: controlNames, Ks, index: n => { const i = ctlIndex(n); return i >= 0 && i < Ks ? i : -1; }, settings: () => preset.driver };
  const drivers = {}; let driver = null, source = null, driverData = null;
  const getDriver = name => { if (drivers[name]) return drivers[name]; const f = driverRegistry[name]; if (!f) throw new Error("unknown driver " + name); return (drivers[name] = f(env)); };
  function selectDriver() {
    if (driver) driver.detach(); driver = null;
    const name = preset.driver.name; if (!name || name === "none" || !named) return;
    driver = getDriver(name); if (driver.setData) driver.setData(driverData); if (source) driver.attach(source);
  }

  // ---- idle motion (gaze, head sway, blink)
  const idle = { t: 0, gaze: { from: [0, 0], to: [0, 0], t0: 0, dur: 0.1, next: 0.5, aversion: null, nextAversion: rnd(8, 15) }, sway: { from: [0, 0], to: [0, 0], t0: 0, dur: 4 }, blink: { next: 2 + Math.random() * 2, t0: -9, doubleAt: -1 }, off: [0, 0], gazeVal: [0, 0] };
  const speakingNow = () => !!(driver && driver.speaking && (!source || source.playing()));
  const gi = { x: ctlIndex("gazeX"), y: ctlIndex("gazeY") };
  function stepIdle(dt) {
    const I = preset.idle, S = preset.speech, id = idle, speaking = speakingNow(); id.t += dt; const t = id.t;
    if (!I.enabled) { id.off[0] = id.off[1] = 0; return; }
    // head sway: a slow random walk inside the rig's limits, eased between targets
    const sw = I.sway; let oy = 0, op_ = 0;
    if (sw.enabled && limits && (!speaking || S.swayWhileSpeaking)) {
      const q = id.sway; if (t - q.t0 >= q.dur) { q.from = q.to.slice(); q.t0 = t; q.dur = rndRange(sw.periodS);
        const ya = (sw.yawDeg + (speaking ? S.emphasisDeg : 0)) * DEG, ta = sw.tiltDeg * DEG; q.to = [rnd(-ya, ya), rnd(-ta, ta)]; }
      const u = ss((t - q.t0) / q.dur); oy = q.from[0] + (q.to[0] - q.from[0]) * u; op_ = q.from[1] + (q.to[1] - q.from[1]) * u;
      const ym = limits.yaw * sw.limitFraction, pm = limits.pitch * sw.limitFraction; oy = clamp(oy, -ym, ym); op_ = clamp(op_, -pm, pm);
    }
    id.off[0] = oy; id.off[1] = op_;
    // gaze
    const g = I.gaze;
    if (g.enabled && gi.x >= 0 && gi.y >= 0) {
      const rx = rig.rest ? rig.rest[gi.x] : 0, ry = rig.rest ? rig.rest[gi.y] : 0, q = id.gaze;
      if (g.mode === "sine") { id.gazeVal[0] = rx + g.amount * Math.sin(t * (0.37 + 0.21 * gi.x) + gi.x); id.gazeVal[1] = ry + g.amount * Math.sin(t * (0.37 + 0.21 * gi.y) + gi.y); }
      else {
        const av = g.aversion || {}, avOn = av.enabled && (!speaking || S.aversionsWhileSpeaking);
        const retarget = (tx, ty, dur) => { const k = ss((t - q.t0) / q.dur); q.from = [q.from[0] + (q.to[0] - q.from[0]) * k, q.from[1] + (q.to[1] - q.from[1]) * k]; q.to = [tx, ty]; q.t0 = t; q.dur = dur; };
        if (q.aversion && t >= q.aversion.until) { q.aversion = null; retarget(0, 0, rndRange(g.shiftS)); q.next = t + rndRange(g.intervalS); q.nextAversion = t + rndRange(speaking ? av.everySpeakingS || [4, 8] : av.everyListeningS || [8, 15]); }
        else if (!q.aversion && avOn && t >= q.nextAversion) { const sx = Math.random() < 0.5 ? -1 : 1; q.aversion = { until: t + rndRange(av.durationS || [1, 3]) }; retarget(sx * av.amount, -Math.abs(av.amount) * 0.5, rndRange(g.shiftS)); }
        else if (!q.aversion && t >= q.next) { retarget(rnd(-1, 1) * g.amount, rnd(-1, 1) * g.amount * 0.6, rndRange(g.shiftS)); q.next = t + rndRange(g.intervalS); }
        const k = ss((t - q.t0) / q.dur); id.gazeVal[0] = rx + q.from[0] + (q.to[0] - q.from[0]) * k; id.gazeVal[1] = ry + q.from[1] + (q.to[1] - q.from[1]) * k;
      }
    } else if (gi.x >= 0 && gi.y >= 0) { id.gazeVal[0] = rig.rest ? rig.rest[gi.x] : 0; id.gazeVal[1] = rig.rest ? rig.rest[gi.y] : 0; }
  }
  function blinkValue(dt) {                                      // a no-op unless the model has a "blink" control
    const bi = ctlIndex("blink"), B = preset.idle.blink; if (bi < 0 || !preset.idle.enabled || !B.enabled) return null;
    const b = idle.blink, t = idle.t, speaking = speakingNow();
    if (t >= b.next && t - b.t0 > B.closeS + B.holdS + B.openS) {
      b.t0 = t; const mean = speaking && preset.speech.blinkFaster ? B.intervalSpeakingS : B.intervalIdleS;
      const wait = Math.min(B.maxS, B.minS + (-Math.log(1 - Math.random())) * Math.max(0.1, mean - B.minS)); b.next = t + (Math.random() < B.doubleChance ? B.closeS + B.holdS + B.openS + 0.1 : wait);
    }
    const u = t - b.t0, c = B.closeS, h = B.holdS, o = B.openS; if (u < 0 || u > c + h + o) return 0;
    return u < c ? ss(u / c) : u < c + h ? 1 : 1 - ss((u - c - h) / o);
  }
  function trackAt(fp, j0, j1) { const tr = rig.track, F = D.frames, f0 = Math.max(0, Math.min(F - 1, Math.floor(fp))), f1 = Math.min(F - 1, f0 + 1), u = Math.max(0, Math.min(1, fp - f0)); for (let j = j0; j < j1; j++) rigW[j] = tr[f0][j] * (1 - u) + tr[f1][j] * u; }
  const speechBuf = new Float32Array(Math.max(Ks, 1));
  function computeWeights(dt) {
    if (!named) { if (rig.track) trackAt(Math.min(phase * D.fps, D.frames - 1), 0, Km); return; }
    const t = idle.t; stepIdle(dt);
    // components after the speech block: head components play the clip's own track back and forth; free ones sit at rest (plus idle gaze)
    if (Kh > 0 && rig.track) { let p = (t * D.fps) % (2 * (D.frames - 1)); if (p > D.frames - 1) p = 2 * (D.frames - 1) - p; trackAt(p, Ks, Ks + Kh); }
    for (let j = Ks + Kh; j < Km; j++) rigW[j] = rig.rest ? rig.rest[j] : 0;
    if (gi.x >= 0) { rigW[gi.x] = idle.gazeVal[0]; rigW[gi.y] = idle.gazeVal[1]; }
    // speech
    speechBuf.fill(0); if (driver) driver.update(dt, speechBuf);
    const Dv = preset.driver;
    for (let j = 0; j < Ks; j++) {
      let v = speechBuf[j]; if (Dv.clampToTrained && stats) v = clamp(v, stats.lo[j], stats.hi[j]);
      const mx = Dv.maxPerControl && Dv.maxPerControl[controlNames[j]]; if (typeof mx === "number") v = Math.min(v, mx);
      rigW[j] = v;
    }
    const bv = blinkValue(dt); if (bv !== null) rigW[ctlIndex("blink")] = bv;
    if (orbitOn && eff.exprGain < 1) { for (let j = 0; j < Ks; j++) rigW[j] *= eff.exprGain; for (const q of [gi.x, gi.y]) if (q >= 0) { const r0 = rig.rest ? rig.rest[q] : 0; rigW[q] = r0 + (rigW[q] - r0) * eff.exprGain; } }   // manual values fade too (set below)
    for (const k in manual) { const q = ctlIndex(k); rigW[q] = orbitOn && q < Ks ? manual[k] * eff.exprGain : manual[k]; }
  }

  // ---- rendering
  let dirty = true, playing = true, lastNow = 0, raf = 0, destroyed = false, fps = 0, frameCount = 0, fpsT = 0, count = 0, lastFrame = null;
  const fixed = options.size || null, pixelRatio = options.pixelRatio;
  function resize() {
    const dpr = pixelRatio || Math.min(globalThis.devicePixelRatio || 1, 2), w = fixed ? fixed.width : Math.round(canvas.clientWidth * dpr), h = fixed ? fixed.height : Math.round(canvas.clientHeight * dpr);
    if (w < 1 || h < 1) return false;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; gl.viewport(0, 0, w, h); dirty = true; } return true;
  }
  const hexRgb = s => { s = String(s).trim(); if (/^#[0-9a-f]{3}$/i.test(s)) s = "#" + [...s.slice(1)].map(c => c + c).join(""); const m = /^#([0-9a-f]{6})$/i.exec(s) || [0, "000000"]; return [0, 2, 4].map(i => parseInt(m[1].slice(i, i + 2), 16) / 255); };
  function applyPose() {
    [base.yaw, base.pitch] = clampAngles(base.yaw, base.pitch);
    if (orbitOn) { base.orbit = orbitFull ? ((base.orbit + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI : clamp(base.orbit, orb.min, orb.max); eff.orbit = base.orbit; }
    // head follow: with the camera at orbit angle o the head turns on the neck by the same angle (up to followMax), so the face keeps looking at the camera; beyond that the face turns away with the body
    const fol = orbitOn && orb.follow && limits ? -clamp(eff.orbit, -Math.min(orb.followMax, limits.yaw), Math.min(orb.followMax, limits.yaw)) * (1 - ss((Math.abs(eff.orbit) - orb.followFade[0]) / (orb.followFade[1] - orb.followFade[0]))) : 0;   // fades to 0 beyond followFadeDeg: the head was only seen twisted on the front arc, and a clamp that stays on at the back would flip sign (a 60 deg pop) when auto-rotate crosses 180
    const [y, p] = clampAngles(base.yaw + idle.off[0] + fol, base.pitch + idle.off[1]); eff.yaw = y; eff.pitch = p; eff.zoom = base.zoom;
    // face angle relative to the camera (0 = looking into it); expression controls were trained only on the front arc and fade out with it
    eff.face = orbitOn ? eff.yaw + eff.orbit : eff.yaw; eff.face = Math.atan2(Math.sin(eff.face), Math.cos(eff.face));
    eff.exprGain = orbitOn ? 1 - ss((Math.abs(eff.face) - orb.fade[0]) / (orb.fade[1] - orb.fade[0])) : 1;
  }
  function renderFrame(dt, now) {
    if (hasRig) computeWeights(dt);
    applyPose();
    const tau = Math.min(phase * D.fps / Math.max(1, D.frames - 1), 1), { V, pos } = cameraMatrices();
    count = evaluate(tau, pos, V);
    const [r, g, b] = hexRgb(bg); gl.clearColor(r, g, b, 1); gl.clear(gl.COLOR_BUFFER_BIT);
    const sc = canvas.height / D.hdr.img_h, f = D.hdr.K[1][1] * sc * tele;   // the training crop fills the canvas height
    gl.uniformMatrix4fv(uView, false, V); gl.uniform2f(uFocal, f, f); gl.uniform2f(uViewport, canvas.width, canvas.height);
    gl.uniform2f(uPP, canvas.width / 2 + (D.hdr.K[0][2] - D.hdr.img_w / 2) * sc, D.hdr.K[1][2] * sc);   // principal point: row 640 of the ring frame, not the crop centre
    gl.bindVertexArray(vao); gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
    frameCount++; if (now - fpsT > 1000) { fps = frameCount * 1000 / (now - fpsT); frameCount = 0; fpsT = now; }
    emit("frame", { fps, visible: count, n: nG, yaw: eff.yaw, pitch: eff.pitch, orbit: eff.orbit, speaking: speakingNow() });
  }
  function loop(now) {
    if (destroyed) return; raf = requestAnimationFrame(loop);
    if (!playing || !resize()) { lastNow = now; return; }
    const dt = lastNow ? Math.min(0.25, (now - lastNow) / 1000) : 0; lastNow = now;
    if (!named) { phase += dt; const per = D.frames / D.fps; if (phase >= per) phase -= per; dirty = true; }   // plain clip models loop their stored motion
    if (named && (preset.idle.enabled || driver)) dirty = true;
    if (!dirty) return; dirty = false; renderFrame(dt, now);
  }

  // ---- public API
  const api = {
    on,
    get controls() { return ctlInfo.map(c => ({ ...c, value: rigW ? rigW[c.index] : 0 })); },
    get limits() { return { yaw: limits ? [-limits.yaw, limits.yaw] : null, pitch: limits ? [-limits.pitch, limits.pitch] : null, pitchNegFade: limits && limits.negFade ? limits.negFade.slice() : null, zoom: [0.25, 2.5] }; },
    get pose() { return { yaw: eff.yaw, pitch: eff.pitch, zoom: eff.zoom, orbit: eff.orbit, face: eff.face, exprGain: eff.exprGain }; },
    get basePose() { return { yaw: base.yaw, pitch: base.pitch, zoom: base.zoom, orbit: base.orbit }; },
    get orbitLimits() { return orbitOn ? { min: orbitFull ? null : orb.min, max: orbitFull ? null : orb.max, full: orbitFull, followHead: orb.follow } : null; },   // what setPose / dragging set, without idle sway
    get info() { return { aspect: D.hdr.img_w / D.hdr.img_h, n: nG, visible: count, frames: D.frames, fps: D.fps, version: D.version, named, hasRig, head: !!head, tele }; },
    get stats() { return { fps, visible: count, n: nG }; },
    get speaking() { return speakingNow(); },
    get audioLatency() { return source ? (source.ctx.outputLatency || source.ctx.baseLatency || 0) : 0; },   // seconds the timed driver subtracts from the audio clock
    setControls(obj) {
      if (!named) throw new Error("this model has no named controls");
      for (const [k, v] of Object.entries(obj || {})) { const i = ctlIndex(k); if (i < 0) throw new Error("unknown control " + k); if (v === null || v === undefined) delete manual[k]; else manual[k] = clamp(+v, ctlInfo[i].min, ctlInfo[i].max); }
      dirty = true;
    },
    clearControls() { for (const k of Object.keys(manual)) delete manual[k]; dirty = true; },
    setFollowHead(on) { if (orb) { orb.follow = !!on; applyPose(); dirty = true; } return !!orb && orb.follow; },   // head turns with the camera near the front (orbit rigs only)
    setPose(p) { if (p.yaw !== undefined) base.yaw = +p.yaw; if (p.pitch !== undefined) base.pitch = +p.pitch; if (p.zoom !== undefined) base.zoom = clamp(+p.zoom, 0.25, 2.5); if (p.orbit !== undefined && orbitOn) base.orbit = +p.orbit; applyPose(); dirty = true; return api.pose; },
    getPreset() { return JSON.parse(JSON.stringify(preset)); },
    setPreset(p) { const next = mergeDeep(DEFAULT_PRESET, p || {}); const was = preset.driver.name; preset = next; if (was !== next.driver.name || !driver) selectDriver(); dirty = true; return api.getPreset(); },
    getPresetName() { return preset.driver.name; },
    setDriver(name, o = {}) {                                    // settings go into the preset; { track } / { timings } are runtime data for the timed driver
      const { track, timings, ...settings } = o; if (name && name !== "none" && !driverRegistry[name]) throw new Error("unknown driver " + name);
      const d = mergeDeep(preset.driver, settings); d.name = name; preset.driver = d;
      if (track !== undefined || timings !== undefined) driverData = { track, timings };
      selectDriver(); dirty = true; return api.getPreset();
    },
    setTrack(d) { driverData = d; if (driver && driver.setData) driver.setData(d); },
    attachAudio(src, o = {}) {
      api.detachAudio(); source = makeSource(src, o);
      const set = {}; if (o.gain !== undefined) set.gain = o.gain; if (o.driver) { api.setDriver(o.driver, set); } else if (o.gain !== undefined) preset.driver.gain = o.gain;
      if (!driver) selectDriver(); if (driver) driver.attach(source); dirty = true; return api;
    },
    detachAudio() { if (source) { source.detach(); source = null; } if (driver) driver.detach(); },
    setBackground(c) { bg = c; dirty = true; },
    pause() { playing = false; }, resume() { playing = true; lastNow = 0; dirty = true; },
    renderOnce(dt = 0) { resize(); renderFrame(dt, performance.now()); return count; },   // draw now, whatever the loop is doing (used by tests)
    getWeights() { return rigW ? Array.from(rigW) : null; },
    readPixels() { return canvas; },
    enablePointer() {                                            // drag to turn the head (or orbit), wheel / pinch to zoom, as in the prototype
      canvas.style.touchAction = "none"; const ptr = new Map(); let pinch = 0;
      const down = e => { canvas.setPointerCapture(e.pointerId); ptr.set(e.pointerId, [e.clientX, e.clientY]); if (ptr.size === 2) { const [a, b] = [...ptr.values()]; pinch = Math.hypot(a[0] - b[0], a[1] - b[1]); } };
      const move = e => {
        if (!ptr.has(e.pointerId)) return; const p = ptr.get(e.pointerId), dx = e.clientX - p[0], dy = e.clientY - p[1]; ptr.set(e.pointerId, [e.clientX, e.clientY]);
        if (ptr.size === 1) { if (orbitOn) api.setPose({ orbit: base.orbit - dx * 0.008, pitch: clamp(base.pitch + dy * 0.004, -0.5, 0.5) }); else api.setPose({ yaw: base.yaw - dx * 0.006, pitch: clamp(base.pitch + dy * 0.004, -1, 1) }); }
        else if (ptr.size === 2) { const [a, b] = [...ptr.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]); if (pinch) api.setPose({ zoom: base.zoom * pinch / d }); pinch = d; }
      };
      const up = e => { ptr.delete(e.pointerId); pinch = 0; }, wheel = e => { e.preventDefault(); api.setPose({ zoom: base.zoom * Math.exp(e.deltaY * 0.001) }); };
      canvas.addEventListener("pointerdown", down); canvas.addEventListener("pointermove", move); canvas.addEventListener("pointerup", up); canvas.addEventListener("pointercancel", up); canvas.addEventListener("wheel", wheel, { passive: false });
      const off = () => { canvas.removeEventListener("pointerdown", down); canvas.removeEventListener("pointermove", move); canvas.removeEventListener("pointerup", up); canvas.removeEventListener("pointercancel", up); canvas.removeEventListener("wheel", wheel); };
      disposers.push(off); return off;
    },
    destroy() {
      if (destroyed) return; destroyed = true; cancelAnimationFrame(raf); api.detachAudio(); while (disposers.length) disposers.pop()();
      const ext = gl.getExtension("WEBGL_lose_context"); if (ext) ext.loseContext(); for (const k of Object.keys(handlers)) handlers[k].length = 0;
    }
  };
  const disposers = [];
  if (options.preset) preset = mergeDeep(DEFAULT_PRESET, options.preset);
  if (options.pointer) api.enablePointer();
  if (named) { if (hasRig && rig.track) trackAt(0, 0, Km); selectDriver(); }
  if (options.autostart !== false) raf = requestAnimationFrame(loop);
  emit("loaded", { n: nG, version: D.version, controls: api.controls, limits: api.limits, info: api.info });
  return api;
}

function registerDriver(name, factory) { driverRegistry[name] = factory; }
export const OpenAvatarViewer = { create, registerDriver, FORMAT_MAJOR, DEFAULT_PRESET, VISEME_WEIGHTS, lettersToVisemes };
export default OpenAvatarViewer;
