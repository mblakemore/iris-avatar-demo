// OpenAvatar listener: what the USER is doing with their voice, for an avatar that listens (nods at pauses, knows when the user has finished).
// Three tiers, cheapest first:
//   1. own level + adaptive noise floor + pause detector: no files, runs everywhere (always on);
//   2. optional Silero VAD v6 (MIT, ~1.3 MB ONNX) decides voice / no voice instead of the level rule;
//   3. optional Smart Turn v3.2 (BSD-2, ~8.7 MB int8 ONNX, Whisper-tiny encoder) answers "has the user finished?" at each pause.
// Tiers 2 and 3 run through onnxruntime-web (MIT, WASM, one thread), loaded only when asked for, from /vendor/ by default.
//
// Assumes an ECHO-CANCELLED microphone (getUserMedia({audio: {echoCancellation: true}})): without it the avatar's own voice coming out of the
// speakers is heard as the user talking. Hosts without AEC should ignore listener speech while the avatar speaks, or use headphones.
//
// Everything runs on audio resampled to 16 kHz in frames of 512 samples (32 ms), the unit Silero needs. Nothing leaves the page.
// API: see working-docs/spikes/lf_attend/API.md. createListener(MediaStream | AudioNode | null, opts) -> {start, stop, push, on, off, state, reset}.

export const SR = 16000, FRAME = 512, FRAME_S = FRAME / SR;

export const DEFAULTS = {
  // tier 1: level rule (dB relative to full scale of the 16 kHz signal, RMS over a 32 ms frame)
  onDb: 12,            // voice when the level is this far above the noise floor ...
  offDb: 7,            // ... and stays voice until it drops below floor + offDb (hysteresis)
  absMinDb: -52,       // never voice below this level (an echo-cancelled mic in a quiet room sits near -70..-60)
  hpHz: 150,           // the level is measured after a high-pass (room rumble, hum and handling noise sit below it; voice does not)
  floorWinS: 3,        // noise floor = the lowest (0.1 s smoothed) level of the last floorWinS seconds (minimum statistics)
  floorSmoothS: 0.1,
  minSpeechS: 0.08,    // speechstart after this much voice
  hangoverS: 0.12,     // speechend after this much silence
  pauseMinS: 0.25,     // a silence inside a turn becomes a "pause" at this length
  turnEndS: 0.9,       // tier 1/2 (no turn model): the turn ends after this much silence
  minTurnS: 0.2,       // turns with less voice than this (a cough, a click) end without a turnend
  levelEvery: 1,       // emit "level" every n frames (1 = 31.25 Hz)
  // tier 2 / 3 (false = off). true = defaults below; or an object overriding them
  vad: false,
  turn: false,
  ortUrl: "/vendor/onnxruntime-web/ort.wasm.min.mjs",
};
export const VAD_DEFAULTS = { modelUrl: "/vendor/silero-vad/silero_vad_16k_op15.onnx", threshold: 0.5, negDelta: 0.15, gateDb: -62 };
export const TURN_DEFAULTS = { modelUrl: "/vendor/smart-turn/smart-turn-v3.2-cpu.onnx", threshold: 0.5, checkAfterS: 0.2, minSilenceS: 0.6, maxWaitS: 2.0, maxAudioS: 8 };

const dbOf = rms => 20 * Math.log10(rms + 1e-10);

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Streaming resampler to 16 kHz (windowed sinc, cutoff at 0.45 of the lower rate, 64 sub-phases). Pure; used by push().
export function makeResampler(inRate, outRate = SR) {
  const ratio = inRate / outRate, half = 8, scale = Math.min(1, 1 / ratio) * 0.9;   // taps either side (in output-rate units), cutoff
  const width = Math.ceil(half / scale), PH = 64, taps = 2 * width + 1;
  const table = new Float32Array(PH * taps);                      // table[ph][j]: weight of input sample (i0 - width + j) for fractional phase ph/PH
  for (let ph = 0; ph < PH; ph++) {
    let s = 0;
    for (let j = 0; j < taps; j++) {
      const x = j - width - ph / PH, a = x * scale, w = 0.5 + 0.5 * Math.cos(Math.PI * Math.max(-1, Math.min(1, x / (width + 1))));
      const v = (Math.abs(a) < 1e-9 ? 1 : Math.sin(Math.PI * a) / (Math.PI * a)) * w; table[ph * taps + j] = v; s += v;
    }
    for (let j = 0; j < taps; j++) table[ph * taps + j] /= s;
  }
  let hist = new Float32Array(0), base = 0, pos = 0;              // hist[k] = input sample number base + k; pos = next output position (input units)
  return {
    push(x) {
      if (inRate === outRate) return Float32Array.from(x);
      const h = new Float32Array(hist.length + x.length); h.set(hist); h.set(x, hist.length); hist = h;
      const out = [];
      while (true) {
        const i0 = Math.floor(pos), k0 = i0 - width - base;
        if (i0 + width - base >= hist.length) break;
        const ph = Math.min(PH - 1, Math.round((pos - i0) * PH)), off = ph * taps; let v = 0;
        for (let j = 0; j < taps; j++) { const k = k0 + j; if (k >= 0) v += hist[k] * table[off + j]; }
        out.push(v); pos += ratio;
      }
      const drop = Math.max(0, Math.floor(pos) - width - 1 - base);   // keep what later outputs still need
      if (drop > 0) { hist = hist.slice(drop); base += drop; }
      return Float32Array.from(out);
    },
  };
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Whisper log-mel features (80 x 800 for 8 s at 16 kHz), as transformers' WhisperFeatureExtractor(chunk_length=8, do_normalize=True):
// zero-mean / unit-variance waveform, centred STFT (reflect pad, periodic Hann 400, hop 160), power, Slaney mel 0-8 kHz, log10, floor at
// max - 8, (x + 4) / 4, last frame dropped. Pure; checked against the Python extractor in tests (tests/viewer/listen_features.test.mjs).
const NFFT = 400, HOP = 160, NMEL = 80, NBIN = NFFT / 2 + 1;
let melBank = null, hann = null, fftTw = null;
function hzToMel(f) { return f < 1000 ? 3 * f / 200 : 15 + Math.log(f / 1000) * 27 / Math.log(6.4); }
function melToHz(m) { return m < 15 ? 200 * m / 3 : 1000 * Math.exp((m - 15) * Math.log(6.4) / 27); }
function initMel() {
  if (melBank) return;
  const mMin = hzToMel(0), mMax = hzToMel(8000), ff = new Float64Array(NMEL + 2);
  for (let i = 0; i < NMEL + 2; i++) ff[i] = melToHz(mMin + (mMax - mMin) * i / (NMEL + 1));
  melBank = new Float32Array(NMEL * NBIN);
  for (let m = 0; m < NMEL; m++) {
    const lo = ff[m], c = ff[m + 1], hi = ff[m + 2], en = 2 / (hi - lo);
    for (let b = 0; b < NBIN; b++) { const f = b * 8000 / (NBIN - 1); const v = Math.max(0, Math.min((f - lo) / (c - lo), (hi - f) / (hi - c))); melBank[m * NBIN + b] = v * en; }
  }
  hann = new Float32Array(NFFT); for (let i = 0; i < NFFT; i++) hann[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / NFFT);
  fftTw = { c: new Float64Array(NFFT), s: new Float64Array(NFFT) }; for (let i = 0; i < NFFT; i++) { fftTw.c[i] = Math.cos(-2 * Math.PI * i / NFFT); fftTw.s[i] = Math.sin(-2 * Math.PI * i / NFFT); }
}
// mixed-radix DFT of a length that factors into small primes (400 = 2^4 5^2); recursive decimation in time
function dft(inRe, inIm, inOff, stride, n, outRe, outIm, outOff, N) {
  if (n === 1) { outRe[outOff] = inRe[inOff]; outIm[outOff] = inIm[inOff]; return; }
  let p = 2; while (n % p) p++;
  const m = n / p;
  for (let j = 0; j < p; j++) dft(inRe, inIm, inOff + j * stride, stride * p, m, outRe, outIm, outOff + j * m, N);
  const tr = new Float64Array(p), ti = new Float64Array(p), step = N / n;
  for (let k = 0; k < m; k++) {
    for (let j = 0; j < p; j++) { tr[j] = outRe[outOff + j * m + k]; ti[j] = outIm[outOff + j * m + k]; }
    for (let q = 0; q < p; q++) {
      const kk = k + q * m; let sr = 0, si = 0;
      for (let j = 0; j < p; j++) { const w = (j * kk * step) % N, c = fftTw.c[w], s = fftTw.s[w]; sr += tr[j] * c - ti[j] * s; si += tr[j] * s + ti[j] * c; }
      outRe[outOff + kk] = sr; outIm[outOff + kk] = si;
    }
  }
}
export function whisperLogMel(audio, nSamples = 8 * SR) {
  initMel();
  // pad (at the START) / truncate (keep the end) to nSamples, as Smart Turn's inference does; then normalise
  const x = new Float64Array(nSamples), n = Math.min(audio.length, nSamples); for (let i = 0; i < n; i++) x[nSamples - n + i] = audio[audio.length - n + i];
  let mean = 0; for (let i = 0; i < nSamples; i++) mean += x[i]; mean /= nSamples;
  let v = 0; for (let i = 0; i < nSamples; i++) v += (x[i] - mean) ** 2; v /= nSamples; const sd = Math.sqrt(v + 1e-7);
  for (let i = 0; i < nSamples; i++) x[i] = (x[i] - mean) / sd;
  const frames = Math.floor(nSamples / HOP), pad = NFFT / 2, out = new Float32Array(NMEL * frames);
  const re = new Float64Array(NFFT), im = new Float64Array(NFFT), oR = new Float64Array(NFFT), oI = new Float64Array(NFFT), pw = new Float64Array(NBIN);
  let mx = -Infinity;
  for (let f = 0; f < frames; f++) {
    for (let i = 0; i < NFFT; i++) { let k = f * HOP + i - pad; if (k < 0) k = -k; else if (k >= nSamples) k = 2 * (nSamples - 1) - k; re[i] = x[k] * hann[i]; im[i] = 0; }
    dft(re, im, 0, 1, NFFT, oR, oI, 0, NFFT);
    for (let b = 0; b < NBIN; b++) pw[b] = oR[b] * oR[b] + oI[b] * oI[b];
    for (let m = 0; m < NMEL; m++) {
      let s = 0; const o = m * NBIN; for (let b = 0; b < NBIN; b++) s += melBank[o + b] * pw[b];
      const l = Math.log10(Math.max(1e-10, s)); out[m * frames + f] = l; if (l > mx) mx = l;
    }
  }
  for (let i = 0; i < out.length; i++) out[i] = (Math.max(out[i], mx - 8) + 4) / 4;
  return { data: out, frames };
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// ONNX models (tier 2 / 3). Loaded lazily; `ort` may be passed in (tests in node), else imported from opts.ortUrl.
async function loadOrt(opts) {
  if (opts.ort) return opts.ort;
  const ort = await import(/* webpackIgnore: true */ opts.ortUrl);
  const o = ort.default && ort.default.InferenceSession ? ort.default : ort;
  o.env.wasm.numThreads = 1;                                      // no SharedArrayBuffer / cross-origin isolation needed
  if (!o.env.wasm.wasmPaths && typeof location !== "undefined") o.env.wasm.wasmPaths = new URL(".", new URL(opts.ortUrl, location.href)).href;
  return o;
}
export async function createSileroVad(ort, cfg) {
  const session = await ort.InferenceSession.create(cfg.modelBytes || cfg.modelUrl, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  let state = new Float32Array(2 * 128), ctx = new Float32Array(64);
  const sr = new ort.Tensor("int64", BigInt64Array.from([16000n]), []);
  return {
    async prob(frame) {                                           // 512 samples at 16 kHz -> speech probability
      const x = new Float32Array(64 + FRAME); x.set(ctx); x.set(frame, 64); ctx = frame.slice(FRAME - 64);
      const r = await session.run({ input: new ort.Tensor("float32", x, [1, 64 + FRAME]), state: new ort.Tensor("float32", state, [2, 1, 128]), sr });
      state = Float32Array.from(r.stateN.data); return r.output.data[0];
    },
    reset() { state = new Float32Array(2 * 128); ctx = new Float32Array(64); },
  };
}
export async function createSmartTurn(ort, cfg) {
  const session = await ort.InferenceSession.create(cfg.modelBytes || cfg.modelUrl, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  return {
    async prob(audio16k) {                                        // the user's turn so far (16 kHz); returns P(turn complete)
      const { data, frames } = whisperLogMel(audio16k, (cfg.maxAudioS || 8) * SR);
      const r = await session.run({ input_features: new ort.Tensor("float32", data, [1, NMEL, frames]) });
      const v = r[session.outputNames[0]].data[0]; return v;      // the exported model ends in a sigmoid
    },
  };
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// The detector: frames in, events out. Pure apart from the optional model calls (async). `vad` / `turn` are objects with prob() or null.
export function createDetector(opts = {}, models = {}) {
  const o = { ...DEFAULTS, ...opts }, V = models.vad ? { ...VAD_DEFAULTS, ...(typeof o.vad === "object" ? o.vad : {}) } : null;
  const T = models.turn ? { ...TURN_DEFAULTS, ...(typeof o.turn === "object" ? o.turn : {}) } : null;
  const handlers = {}, emit = (n, e) => { const hs = handlers[n]; if (hs) for (const h of [...hs]) { try { h(e); } catch (err) { console.error(err); } } };
  let HPF = null;
  if (o.hpHz > 0) { const w = 2 * Math.PI * o.hpHz / SR, al = Math.sin(w) / (2 * Math.SQRT1_2), c = Math.cos(w), a0 = 1 + al;
    HPF = { b0: (1 + c) / 2 / a0, b1: -(1 + c) / a0, b2: (1 + c) / 2 / a0, a1: -2 * c / a0, a2: (1 - al) / a0 }; }
  const ring = new Float32Array(Math.ceil((T ? T.maxAudioS : 8) * SR / FRAME + 2) * FRAME); let ringN = 0;   // last ~8 s of audio
  let s;
  function reset() {
    s = { frame: 0, floor: -100, hist: new Float32Array(Math.max(1, Math.round(o.floorWinS / FRAME_S))).fill(Infinity), sm: null, hp: [0, 0, 0, 0], voiced: false, run: 0, silent: 0, speaking: false, onset: 0, lastVoice: 0, inTurn: false, turnStart: 0,
      turnVoice: 0, pause: null, checked: false, vad: null, turnProb: null, stats: { vadMs: 0, vadN: 0, turnMs: 0, turnN: 0, frameMs: 0 } };
    ringN = 0; if (models.vad && models.vad.reset) models.vad.reset();
  }
  reset();
  const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
  function endTurn(t, reason, prob) {
    const sil = t - s.lastVoice;
    if (s.pause) { emit("pause", { t, start: s.pause.start, duration: sil, final: true, inTurn: true }); s.pause = null; }
    const voice = s.turnVoice; s.inTurn = false; s.checked = false; s.complete = false;
    if (voice >= o.minTurnS) emit("turnend", { t, turnStart: s.turnStart, turnDuration: s.lastVoice - s.turnStart, silence: sil, reason, prob: prob ?? null });
  }
  function turnAudio() {                                          // the current turn's audio (up to maxAudioS), oldest first
    const need = Math.min(ringN, Math.round((s.frame * FRAME_S - s.turnStart + 0.3) * SR), ring.length);
    const out = new Float32Array(need), L = ring.length, end = ringN % L;
    for (let i = 0; i < need; i++) out[i] = ring[(end - need + i + L * 2) % L];
    return out;
  }
  async function frame(x) {                                       // x: 512 samples at 16 kHz
    const t0 = now();
    let e = 0, eh = 0; const h = s.hp;
    for (let i = 0; i < FRAME; i++) {
      const v = x[i]; e += v * v;
      if (HPF) { const y = HPF.b0 * v + HPF.b1 * h[0] + HPF.b2 * h[1] - HPF.a1 * h[2] - HPF.a2 * h[3]; h[1] = h[0]; h[0] = v; h[3] = h[2]; h[2] = y; eh += y * y; } else eh += v * v;
    }
    const rms = Math.sqrt(e / FRAME), db = dbOf(Math.sqrt(eh / FRAME));
    for (let i = 0; i < FRAME; i++) ring[(ringN + i) % ring.length] = x[i]; ringN += FRAME;
    const t = (s.frame + 1) * FRAME_S;                            // end of this frame on the listener clock
    // noise floor
    s.sm = s.sm === null ? db : s.sm + (db - s.sm) * (1 - Math.exp(-FRAME_S / o.floorSmoothS));
    s.hist[s.frame % s.hist.length] = Math.min(s.sm, db + 3); let fl = Infinity; for (const v of s.hist) if (v < fl) fl = v; s.floor = fl;
    // voice decision
    let voiced;
    if (V) {
      const a = now(); const p = await models.vad.prob(x); s.stats.vadMs += now() - a; s.stats.vadN++; s.vad = p;
      voiced = db > V.gateDb && (p >= V.threshold || (s.voiced && p >= V.threshold - V.negDelta));
    } else {
      const on = Math.max(s.floor + o.onDb, o.absMinDb), off = Math.max(s.floor + o.offDb, o.absMinDb - 6);
      voiced = db > on || (s.voiced && db > off);
    }
    s.voiced = voiced; s.frame++;
    if (o.levelEvery > 0 && s.frame % o.levelEvery === 0) emit("level", { t, rms, db, floorDb: s.floor, speech: s.speaking, vad: V ? s.vad : null });
    // speech start / end
    if (voiced) { s.run++; s.silent = 0; } else { s.silent++; if (!s.speaking) s.run = 0; }
    if (!s.speaking && s.run * FRAME_S >= o.minSpeechS - 1e-9) {
      s.speaking = true; s.onset = t - s.run * FRAME_S;
      if (!s.inTurn) { s.inTurn = true; s.turnStart = s.onset; s.turnVoice = 0; s.checked = false; }
      else if (s.pause) { emit("pause", { t: s.onset, start: s.pause.start, duration: s.onset - s.pause.start, final: true, inTurn: true }); s.pause = null; }
      s.checked = false; s.complete = false;
      emit("speechstart", { t: s.onset });
    }
    if (s.speaking) {
      if (voiced) { s.lastVoice = t; s.turnVoice += FRAME_S; }
      if (s.silent * FRAME_S >= o.hangoverS - 1e-9) { s.speaking = false; s.run = 0; emit("speechend", { t: s.lastVoice, duration: s.lastVoice - s.onset }); }
    }
    // pauses and turn end
    if (s.inTurn && !s.speaking) {
      const sil = t - s.lastVoice;
      if (!s.pause && sil >= o.pauseMinS - 1e-9 && s.turnVoice >= o.minTurnS) {   // (a lone click is no turn, so no pause)
        s.pause = { start: s.lastVoice }; emit("pause", { t, start: s.lastVoice, duration: sil, final: false, inTurn: true }); }
      if (T) {
        if (!s.checked && sil >= T.checkAfterS - 1e-9 && s.turnVoice >= o.minTurnS) {
          s.checked = true; const a = now(); const p = await models.turn.prob(turnAudio()); const ms = now() - a;
          s.stats.turnMs += ms; s.stats.turnN++; s.turnProb = p; emit("turncheck", { t, prob: p, silence: sil, ms });
          s.complete = p >= T.threshold;
        }
        if (s.inTurn && s.complete && sil >= T.minSilenceS - 1e-9) endTurn(t, "model", s.turnProb);
        if (s.inTurn && sil >= T.maxWaitS - 1e-9) endTurn(t, "timeout", s.turnProb);
      } else if (sil >= o.turnEndS - 1e-9) endTurn(t, "silence", null);
    }
    s.stats.frameMs += now() - t0;
  }
  return {
    on(n, f) { (handlers[n] ||= new Set()).add(f); return () => handlers[n].delete(f); },
    off(n, f) { if (handlers[n]) handlers[n].delete(f); },
    frame, reset,
    state: () => ({ t: s.frame * FRAME_S, speaking: s.speaking, inTurn: s.inTurn, floorDb: s.floor, vadProb: s.vad, turnProb: s.turnProb, stats: { ...s.stats, frames: s.frame } }),
    options: o,
  };
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// AudioWorklet that hands raw blocks to the main thread (inlined so the module stays one file).
const WORKLET = `class OAListenTap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(2048); this.n = 0; }
  process(inputs) { const ch = inputs[0] && inputs[0][0]; if (ch) { if (this.n + ch.length > this.buf.length) { this.port.postMessage(this.buf.slice(0, this.n)); this.n = 0; } this.buf.set(ch, this.n); this.n += ch.length; } return true; }
}
registerProcessor("oa-listen-tap", OAListenTap);`;

export function createListener(input = null, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const handlers = {}; const models = {}; let det = createDetector(o, models), unsub = [];
  const st = { status: "off", vad: o.vad ? "pending" : "off", turn: o.turn ? "pending" : "off", sampleRate: null, error: null };
  let rs = null, rsRate = 0, pend = new Float32Array(0), chain = Promise.resolve(), graph = null;
  const fwd = n => det.on(n, e => { const hs = handlers[n]; if (hs) for (const h of [...hs]) h(e); });
  const NAMES = ["level", "pause", "speechstart", "speechend", "turnend", "turncheck"];
  function rebuild() { unsub.forEach(u => u()); det = createDetector(o, models); unsub = NAMES.map(fwd); }
  unsub = NAMES.map(fwd);
  async function loadModels() {
    if (!o.vad && !o.turn) return;
    let ort;
    try { ort = await loadOrt(o); } catch (err) { st.vad = o.vad ? "failed" : "off"; st.turn = o.turn ? "failed" : "off"; st.error = String(err); return; }
    if (o.vad) { st.vad = "loading"; try { models.vad = await createSileroVad(ort, { ...VAD_DEFAULTS, ...(typeof o.vad === "object" ? o.vad : {}) }); st.vad = "on"; } catch (err) { st.vad = "failed"; st.error = String(err); } }
    if (o.turn) { st.turn = "loading"; try { models.turn = await createSmartTurn(ort, { ...TURN_DEFAULTS, ...(typeof o.turn === "object" ? o.turn : {}) }); st.turn = "on"; } catch (err) { st.turn = "failed"; st.error = String(err); } }
    rebuild();
  }
  // feed samples at any rate; resolves when every whole frame in them has been processed
  function push(x, rate = SR) {
    if (!rs || rsRate !== rate) { rs = makeResampler(rate, SR); rsRate = rate; }
    const y = rs.push(x), all = new Float32Array(pend.length + y.length); all.set(pend); all.set(y, pend.length);
    const n = Math.floor(all.length / FRAME); pend = all.slice(n * FRAME);
    for (let i = 0; i < n; i++) { const f = all.subarray(i * FRAME, (i + 1) * FRAME); chain = chain.then(() => det.frame(f)); }
    return chain;
  }
  return {
    on(n, f) { (handlers[n] ||= new Set()).add(f); return () => handlers[n].delete(f); },
    off(n, f) { if (handlers[n]) handlers[n].delete(f); },
    push,
    async start() {
      if (st.status === "on" || st.status === "starting") return;
      st.status = "starting"; await loadModels();
      if (input) {
        const isNode = typeof input.connect === "function" && input.context;
        const ctx = isNode ? input.context : (o.context || new AudioContext());
        if (ctx.state === "suspended") { try { await ctx.resume(); } catch {} }
        const src = isNode ? input : ctx.createMediaStreamSource(input);
        st.sampleRate = ctx.sampleRate; let tap;
        if (ctx.audioWorklet) {
          const url = URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" }));
          try { await ctx.audioWorklet.addModule(url); } finally { URL.revokeObjectURL(url); }
          tap = new AudioWorkletNode(ctx, "oa-listen-tap", { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1, channelCountMode: "explicit" });
          tap.port.onmessage = ev => push(ev.data, ctx.sampleRate);
        } else {
          tap = ctx.createScriptProcessor(2048, 1, 1); tap.onaudioprocess = ev => push(ev.inputBuffer.getChannelData(0).slice(), ctx.sampleRate);
        }
        const mute = ctx.createGain(); mute.gain.value = 0; src.connect(tap); tap.connect(mute); mute.connect(ctx.destination);   // pulled, silent
        graph = { ctx, src, tap, mute, ownCtx: !isNode && !o.context };
      }
      st.status = "on";
    },
    stop() {
      if (graph) { try { graph.src.disconnect(graph.tap); } catch {} try { graph.tap.disconnect(); graph.mute.disconnect(); } catch {} if (graph.tap.port) graph.tap.port.onmessage = null;
        if (graph.ownCtx) graph.ctx.close().catch(() => {}); graph = null; }
      st.status = "off";
    },
    reset() { det.reset(); pend = new Float32Array(0); rs = null; },
    state: () => ({ ...st, tier: models.turn ? 3 : models.vad ? 2 : 1, ...det.state() }),
    flush: () => chain,
  };
}
export default { createListener, createDetector, whisperLogMel, makeResampler, createSileroVad, createSmartTurn, DEFAULTS, VAD_DEFAULTS, TURN_DEFAULTS, SR, FRAME };
