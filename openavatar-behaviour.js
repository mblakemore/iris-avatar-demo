// OpenAvatar behaviour: an OPTIONAL, dependency-free module that makes the avatar behave like a person in a conversation.
// It drives the viewer only through its motion-layer API (v.addLayer, see docs/CONTRACT.md "motion layers"); a viewer without that API gets a
// reduced stand-in (pose + held controls from the frame event). Nothing here runs unless a page creates it: the viewer alone behaves as before.
//
//   import { createBehaviour, parseTags, anchorCues } from "./openavatar-behaviour.js";
//   const b = createBehaviour(viewer, { persona: { seed: 7 } });
//   b.attachListener(analyserNodeOrAttendListener);     // the user's microphone level: listening, pauses, turn ends
//   b.think();                                           // the LLM is working (or automatic at the user's turn end with autoThink)
//   b.speak({ pcm, sampleRate, timings, cues });         // a reply clip is about to play: prosody, breaths, turn-end gaze are planned from it
//   b.on("turnyield", e => openMic());                   // her gaze returned to the user for the end of the reply
//   b.yield(); b.holdFloor(2); b.cue("smile"); b.setMood({ valence: 0.5 });
//
// States: idle / listening / thinking / speaking / yielding, automatic from the signals (clip playing, user level, LLM wait) or set by hand.
// Rules and numbers: working-docs/spikes/lifelike/ROADMAP.md section 3.1 (gaze shares, turn-start aversion, turn-end gaze return, blink rates,
// inbreaths, backchannel nods), 3.5 (energy-scaled vowels, breaths and sighs) and 3.6 (persona, LLM tags, reactions, presence, memory).
// Angles inside are DEGREES; the layer output converts to the viewer's radians. Signs as the viewer's (spikes/lf_core/API.md): gaze x+ = toward the
// picture's right, y+ = up, (0, 0) = the camera; +yaw turns the face toward the picture's LEFT (so a look to x+ goes with a negative yaw);
// +pitch lifts the chin; +roll tilts the top of the head toward the picture's left; a smaller zoom is closer (lean-in = negative zoom delta).

export const BEHAVIOUR_VERSION = 1;
export const STATES = ["idle", "listening", "thinking", "speaking", "yielding"];
const DEG = Math.PI / 180;
const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const ss = x => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };
const bump = u => (u <= 0 || u >= 1 ? 0 : Math.sin(Math.PI * u) ** 2);         // raised cosine 0 -> 1 -> 0, zero velocity at both ends
const envelope = (t, dur, rise, fall) => (t <= 0 || t >= dur ? 0 : Math.min(ss(t / rise), ss((dur - t) / fall)));
const median = a => { if (!a.length) return 0; const s = Float64Array.from(a).sort(); const m = s.length >> 1; return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]); };
const quant = (a, q) => { if (!a.length) return 0; const s = Float64Array.from(a).sort(); return s[clamp(Math.round(q * (s.length - 1)), 0, s.length - 1)]; };

// ------------------------------------------------------------------ random, noise, smoothing
export function seededRandom(seed = 1) {                         // mulberry32
  let s = (seed >>> 0) || 1;
  return () => { s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
// band-limited noise: a few octaves of Catmull-Rom interpolated value noise. Never "arrives" at a target the way eased random retargets do.
export function makeNoise(rand, octaves) {
  const oct = octaves.map(o => ({ hz: o.hz, amp: o.amp, off: rand() * 100, k: [0, 0, 0, 0].map(() => rand() * 2 - 1), n: null }));
  const cr = (k, u) => { const [a, b, c, d] = k; return 0.5 * ((2 * b) + (-a + c) * u + (2 * a - 5 * b + 4 * c - d) * u * u + (-a + 3 * b - 3 * c + d) * u * u * u); };
  return t => {
    let s = 0;
    for (const o of oct) {
      const x = t * o.hz + o.off, i = Math.floor(x);
      if (o.n === null) o.n = i;
      while (o.n < i) { o.k.shift(); o.k.push(rand() * 2 - 1); o.n++; }
      s += o.amp * cr(o.k, x - i) / 1.15;                         // Catmull-Rom of uniform knots stays within about +-1.15
    }
    return s;
  };
}
// a follower with a bounded acceleration (deg/s^2) and no overshoot: what keeps the head's per-frame velocity change under the motion bound
export function makeFollower(amax, tau = 0.05) {
  let x = 0, v = 0;
  return {
    step(y, dt) {
      if (!(dt > 0)) return x;
      const e = y - x, ae = Math.abs(e), vd = Math.sign(e) * Math.min(ae / tau, Math.sqrt(2 * amax * ae));
      const a = clamp((vd - v) / dt, -amax, amax); v += a * dt; x += v * dt; return x;
    },
    get value() { return x; }, get velocity() { return v; }, reset(x0 = 0) { x = x0; v = 0; }
  };
}

// ------------------------------------------------------------------ pitch (F0): McLeod normalised square difference, own code
// x: a frame of samples (any length >= 2 periods of fmin), sr: its rate. Returns { f0 (Hz, 0 = none), clarity (0..1) }.
export function pitchMcLeod(x, sr, fmin = 70, fmax = 420, k = 0.9) {
  const n = x.length, tmin = Math.max(2, Math.floor(sr / fmax)), tmax = Math.min(Math.ceil(sr / fmin), n - 2);
  if (tmax <= tmin) return { f0: 0, clarity: 0 };
  const nsdf = new Float64Array(tmax + 2);
  let m = 0; for (let i = 0; i < n; i++) m += 2 * x[i] * x[i];
  if (m < 1e-10) return { f0: 0, clarity: 0 };
  for (let tau = 0; tau <= tmax + 1; tau++) {
    let r = 0; for (let i = 0; i < n - tau; i++) r += x[i] * x[i + tau];
    nsdf[tau] = m > 0 ? 2 * r / m : 0;
    m -= x[n - 1 - tau] * x[n - 1 - tau] + x[tau] * x[tau];      // m(tau + 1)
  }
  // key maxima: the highest point of each positive lobe after the first negative-going zero crossing
  const keys = []; let t = 1;
  while (t < tmax && nsdf[t] > 0) t++;                            // leave the lobe at lag 0
  while (t <= tmax) {
    while (t <= tmax && nsdf[t] <= 0) t++;
    let best = -1, bi = -1;
    while (t <= tmax && nsdf[t] > 0) { if (nsdf[t] > best) { best = nsdf[t]; bi = t; } t++; }
    if (bi >= tmin) keys.push(bi);
  }
  if (!keys.length) return { f0: 0, clarity: 0 };
  let top = 0; for (const i of keys) top = Math.max(top, nsdf[i]);
  const pick = keys.find(i => nsdf[i] >= k * top);
  const a = nsdf[pick - 1], b = nsdf[pick], c = nsdf[pick + 1], den = a - 2 * b + c;
  const off = Math.abs(den) > 1e-12 ? clamp(0.5 * (a - c) / den, -0.5, 0.5) : 0;
  return { f0: sr / (pick + off), clarity: clamp(b - 0.25 * (a - c) * off, 0, 1) };
}

// ------------------------------------------------------------------ whole-clip analysis (timed clips: done before / as the clip starts)
// pcm: Float32Array mono; sr: rate; opts.timings: character timings [{char, start, end}] of the clip (optional); opts.nonSpeech: the aligner's
// [{start, end}] (optional). Returns per-frame arrays at `hop` s plus events: accents (pitch / energy peaks), phrase ends, pauses, non-speech.
const PUNCT = /[.,!?;:…—-]/;
export function analyseClip(pcm, sr, opts = {}) {
  const hop = opts.hop || 0.01, dsr = 8000, dec = Math.max(1, Math.round(sr / dsr)), rate = sr / dec;
  const ds = new Float32Array(Math.floor(pcm.length / dec));       // box-filtered downsample: enough for F0 up to 420 Hz
  for (let i = 0; i < ds.length; i++) { let s = 0; for (let j = 0; j < dec; j++) s += pcm[i * dec + j]; ds[i] = s / dec; }
  const N = Math.max(1, Math.floor(pcm.length / sr / hop)), db = new Float32Array(N), f0 = new Float32Array(N), clar = new Float32Array(N);
  const rw = Math.round(0.03 * sr), fw = Math.round(0.04 * rate), frame = new Float32Array(fw);
  for (let f = 0; f < N; f++) {
    const c = Math.round((f + 0.5) * hop * sr); let e = 0, k = 0;
    for (let i = Math.max(0, c - (rw >> 1)); i < Math.min(pcm.length, c + (rw >> 1)); i++) { e += pcm[i] * pcm[i]; k++; }
    db[f] = 10 * Math.log10(k ? e / k + 1e-12 : 1e-12);
    const cd = Math.round((f + 0.5) * hop * rate) - (fw >> 1);
    for (let i = 0; i < fw; i++) { const j = cd + i; frame[i] = j >= 0 && j < ds.length ? ds[j] : 0; }
    const p = pitchMcLeod(frame, rate); f0[f] = p.f0; clar[f] = p.clarity;
  }
  const maxDb = Math.max(-120, ...db), floorDb = Math.max(quant(Array.from(db), 0.05), maxDb - 60);
  const voiced = new Uint8Array(N); for (let f = 0; f < N; f++) voiced[f] = clar[f] > 0.7 && f0[f] > 0 && db[f] > maxDb - 32 ? 1 : 0;
  // octave-jump guard: a voiced frame more than 7 semitones from the median of its voiced neighbours (+-60 ms) is dropped
  const st = new Float32Array(N); for (let f = 0; f < N; f++) st[f] = voiced[f] ? 12 * Math.log2(f0[f] / 100) : NaN;
  for (let f = 0; f < N; f++) if (voiced[f]) { const nb = []; for (let g = Math.max(0, f - 6); g <= Math.min(N - 1, f + 6); g++) if (voiced[g]) nb.push(st[g]); if (Math.abs(st[f] - median(nb)) > 7) { voiced[f] = 0; st[f] = NaN; } }
  const vst = [], vdb = []; for (let f = 0; f < N; f++) if (voiced[f]) { vst.push(st[f]); vdb.push(db[f]); }
  const stMed = median(vst), stSpread = Math.max(1.5, (quant(vst, 0.84) - quant(vst, 0.16)) / 2), dbMed = vst.length ? median(vdb) : maxDb - 10, dbSpread = Math.max(3, (quant(vdb, 0.84) - quant(vdb, 0.16)) / 2);
  // smoothed z-scores: pitch held through unvoiced gaps (decaying to 0), energy over all frames
  const f0z = new Float32Array(N), ez = new Float32Array(N); let last = 0;
  for (let f = 0; f < N; f++) { if (voiced[f]) last = (st[f] - stMed) / stSpread; else last *= 0.92; f0z[f] = last; ez[f] = (db[f] - dbMed) / dbSpread; }
  const smooth = (a, w) => { const o = new Float32Array(a.length); for (let f = 0; f < a.length; f++) { let s = 0, k = 0; for (let g = Math.max(0, f - w); g <= Math.min(a.length - 1, f + w); g++) { s += a[g]; k++; } o[f] = s / k; } return o; };
  const f0zS = smooth(f0z, 4), ezS = smooth(ez, 3);
  // speech span
  let sStart = 0, sEnd = N * hop;
  const loud = f => db[f] > maxDb - 30;
  { let a = 0; while (a < N && !loud(a)) a++; let b = N - 1; while (b > 0 && !loud(b)) b--; sStart = a * hop; sEnd = (b + 1) * hop; }
  const T = Array.isArray(opts.timings) ? opts.timings.filter(c => c && Number.isFinite(c.start)) : null;
  const letters = T ? T.filter(c => /[\p{L}\p{N}]/u.test(c.char || "")) : [];
  if (letters.length) { sStart = letters[0].start; sEnd = letters[letters.length - 1].end; }
  // pauses inside the speech: from the timings (gaps between letters >= 0.18 s) or from the level (>= 0.18 s under maxDb - 30)
  const pauses = [];
  if (letters.length) { for (let i = 1; i < letters.length; i++) { const g0 = letters[i - 1].end, g1 = letters[i].start; if (g1 - g0 >= 0.18) pauses.push({ t0: g0, t1: g1 }); } }
  else { let r0 = -1; for (let f = Math.round(sStart / hop); f <= Math.round(sEnd / hop) && f < N; f++) { const q = !loud(f); if (q && r0 < 0) r0 = f; if (!q && r0 >= 0) { if ((f - r0) * hop >= 0.18) pauses.push({ t0: r0 * hop, t1: f * hop }); r0 = -1; } } }
  // phrase ends: punctuation in the timings, else the pauses
  const phraseEnds = [];
  if (T) { for (let i = 0; i < T.length; i++) if (PUNCT.test(T[i].char || "")) { let j = i - 1; while (j >= 0 && !/[\p{L}\p{N}]/u.test(T[j].char || "")) j--; if (j >= 0) { const t = T[j].end; if (!phraseEnds.length || t - phraseEnds[phraseEnds.length - 1] > 0.3) phraseEnds.push(t); } } }
  for (const p of pauses) if (!phraseEnds.some(t => Math.abs(t - p.t0) < 0.25)) phraseEnds.push(p.t0);
  phraseEnds.sort((a, b) => a - b);
  // accents: local maxima of 0.6 pitch + 0.4 energy inside voiced speech, >= 0.35 s apart
  const score = new Float32Array(N); for (let f = 0; f < N; f++) score[f] = voiced[f] || (f > 0 && voiced[f - 1]) ? 0.6 * f0zS[f] + 0.4 * ezS[f] : -9;
  const accents = [];
  for (let f = 3; f < N - 3; f++) {
    const s = score[f]; if (s < 0.45) continue; let isMax = true;
    for (let g = f - 15; g <= f + 15; g++) if (g !== f && g >= 0 && g < N && score[g] > s) { isMax = false; break; }
    if (!isMax) continue;
    const t = f * hop, prev = accents[accents.length - 1];
    if (prev && t - prev.t < 0.35) { if (s > prev.s) accents[accents.length - 1] = { t, s, f0z: f0zS[f], ez: ezS[f] }; continue; }
    accents.push({ t, s, f0z: f0zS[f], ez: ezS[f] });
  }
  const sTop = quant(accents.map(a => a.s), 0.7);
  for (const a of accents) a.big = a.s >= sTop && a.f0z > 0.8;      // the larger pitch peaks (a minority): brow candidates
  // non-speech (breath / sigh): the aligner's list, or louder-than-silence, unvoiced stretches inside pauses and before the first sound
  const nsIn = Array.isArray(opts.nonSpeech) ? opts.nonSpeech.map(x => ({ start: x.start, end: x.end })) : [];
  if (!nsIn.length) {
    const spans = [{ t0: 0, t1: sStart }, ...pauses, { t0: sEnd, t1: N * hop }];
    for (const p of spans) { let r0 = -1; const f0_ = Math.round(p.t0 / hop), f1_ = Math.min(N, Math.round(p.t1 / hop)); for (let f = f0_; f <= f1_; f++) { const on = f < f1_ && db[f] > floorDb + 14 && db[f] > maxDb - 45; if (on && r0 < 0) r0 = f; if (!on && r0 >= 0) { if ((f - r0) * hop >= 0.12) nsIn.push({ start: r0 * hop, end: f * hop }); r0 = -1; } } }
  }
  const nonSpeech = nsIn.map(x => {
    const a = Math.max(0, Math.round(x.start / hop)), b = Math.min(N, Math.round(x.end / hop)); let vo = 0, pk = -120; for (let f = a; f < b; f++) { vo += voiced[f]; pk = Math.max(pk, db[f]); }
    const dur = x.end - x.start, kind = dur >= 0.45 && (vo >= 3 || pk > maxDb - 18) ? "sigh" : "breath";
    return { start: x.start, end: x.end, kind };
  });
  const at = (arr, t) => { const f = clamp(t / hop, 0, N - 1), i = Math.floor(f), u = f - i; return i + 1 < N ? arr[i] * (1 - u) + arr[i + 1] * u : arr[i]; };
  return { hop, n: N, duration: pcm.length / sr, db, f0, clarity: clar, voiced, f0z: f0zS, ez: ezS, stMedian: stMed, stSpread, dbMedian: dbMed, dbSpread, maxDb, floorDb,
    speechStart: sStart, speechEnd: sEnd, pauses, phraseEnds, accents, nonSpeech,
    f0zAt: t => at(f0zS, t), ezAt: t => at(ezS, t), voicedAt: t => voiced[clamp(Math.round(t / hop), 0, N - 1)] === 1 };
}

// ------------------------------------------------------------------ LLM expression tags (section 3.6.2)
export const TAGS = ["smile", "thinks", "brow", "frown", "glance", "nod", "shake", "grin", "wide", "wink", "laugh"];
// "That's a good question. [thinks] I'd start with... [smile] the boring answer." -> { text (what goes to TTS), tags: [{ name, index }] }
// index = position in the CLEAN text the tag stood before. Unknown [words] are left in the text.
export function parseTags(text) {
  const tags = []; let out = "", i = 0; const s = String(text || ""), re = /\[([a-zA-Z]+)\]/g; let m;
  while ((m = re.exec(s))) {
    const name = m[1].toLowerCase(); if (!TAGS.includes(name)) continue;
    out += s.slice(i, m.index); i = m.index + m[0].length;
    if (out.endsWith(" ") && s[i] === " ") i++;                    // "a [smile] b" -> "a b"
    tags.push({ name, index: out.length });
  }
  out += s.slice(i);
  const lead = out.length - out.replace(/^\s+/, "").length; out = out.trim();
  for (const t of tags) t.index = clamp(t.index - lead, 0, out.length);
  return { text: out, tags };
}
// tags -> cues in CLIP time: each tag lands on the start of the next word (the "neighbouring word"); a tag at the end lands on the last letter.
// timings: character timings of the clean text, as the TTS service returns them ({char, start, end}).
export function anchorCues(tags, timings) {
  const T = Array.isArray(timings) ? timings : [], out = [];
  for (const tg of tags || []) {
    let k = tg.index; while (k < T.length && !/[\p{L}\p{N}]/u.test(T[k].char || "")) k++;
    let t;
    if (k < T.length) t = T[k].start; else { let j = Math.min(T.length, tg.index) - 1; while (j >= 0 && !/[\p{L}\p{N}]/u.test(T[j].char || "")) j--; t = j >= 0 ? T[j].end : 0; }
    // the word's end and the end of its phrase (next punctuation) for cues that last through a phrase
    let we = k; while (we < T.length && /[\p{L}\p{N}']/u.test(T[we].char || "")) we++;
    let pe = k; while (pe < T.length && !/[.!?;:,]/.test(T[pe].char || "")) pe++;
    const wordEnd = we > k && T[we - 1] ? T[we - 1].end : t + 0.3, phraseEnd = T[Math.min(pe, T.length) - 1] ? T[Math.min(pe, T.length) - 1].end : wordEnd;
    out.push({ name: tg.name, t, wordEnd, phraseEnd, index: tg.index });
  }
  return out;
}

// ------------------------------------------------------------------ persona (section 3.6.1): a seed plus a dozen numbers
export const DEFAULT_PERSONA = {
  seed: 1,
  thinkDir: "up-left",          // where she looks when thinking: up-left | up-right | left | right (always the same: a recognisable habit)
  tiltSide: "right",            // listening head tilt side (image side), changes rarely
  tiltDeg: 4,                   // listening tilt (roll); probes clean to +-8, kept <= 6
  nodStyle: "single",           // single | double backchannel nods
  nodDeg: 2.6,                  // backchannel nod depth (chin down), cap 3 (throat flecks below -3)
  backchannel: 0.75,            // chance of a nod at a qualifying user pause
  blinkScale: 1,                // x on blink rates
  smileFadeS: 1.4,              // how quickly a smile fades
  smileBase: 0.05,              // resting smile
  leanIn: 0.03,                 // zoom when the user talks for a while
  aversionDeg: 9,               // size of a speaking aversion
  expressiveness: 1,            // x on prosody head motion and cue sizes
  signature: "slowBlinkHalfSmile",   // her tell: a slow blink + half smile (before a joke lands, after "got it")
  gazeShare: { listening: 0.7, speaking: 0.42 }
};
export function makePersona(seed = 1, over = {}) {
  const r = seededRandom(seed * 7919 + 13), pick = a => a[Math.floor(r() * a.length)];
  const p = { ...DEFAULT_PERSONA, seed,
    thinkDir: pick(["up-left", "up-right", "up-left", "up-right", "left", "right"]), tiltSide: pick(["left", "right"]),
    nodStyle: r() < 0.35 ? "double" : "single", nodDeg: 2.2 + r() * 0.8, backchannel: 0.6 + r() * 0.3, blinkScale: 0.85 + r() * 0.3,
    smileFadeS: 1 + r() * 0.9, smileBase: 0.03 + r() * 0.05, leanIn: 0.02 + r() * 0.015, aversionDeg: 7 + r() * 4, tiltDeg: 3.2 + r() * 1.6,
    expressiveness: 0.85 + r() * 0.3 };
  if (seed === DEFAULT_PERSONA.seed && !Object.keys(over).length) return { ...DEFAULT_PERSONA };
  return { ...p, ...over, gazeShare: { ...DEFAULT_PERSONA.gazeShare, ...(over.gazeShare || {}) } };
}

// ------------------------------------------------------------------ the user's voice without ATTEND: level, pauses, turn ends from an AnalyserNode
// feed(dbfs, dt, t) per frame -> events [{type: "speechstart"|"pause"|"speechend"|"turnend", t, t0}] ; level 0..1
export function createVoiceDetector(o = {}) {
  const onDb = o.onDb ?? 10, minDb = o.minDb ?? -58, pauseS = o.pauseS ?? 0.25, turnEndS = o.turnEndS ?? 0.7, startS = o.startS ?? 0.08;
  let floor = -70, voice = false, run = 0, silentSince = -1, turnStart = -1, inTurn = false, pauseSent = false, ended = true, lastVoice = -9, level = 0;
  return {
    get level() { return level; }, get speaking() { return voice; }, get inTurn() { return inTurn; }, get turnStart() { return turnStart; }, get floor() { return floor; },
    feed(dbv, dt, t) {
      const ev = [];
      floor = dbv < floor ? dbv : Math.min(floor + 3 * dt, dbv);     // follows the quiet parts down at once, up at 3 dB/s
      floor = Math.max(floor, -90);
      const above = dbv > floor + onDb && dbv > minDb;
      level = clamp((dbv - floor - 6) / 30, 0, 1) * (dbv > minDb ? 1 : 0);
      if (above) { run += dt; if (!voice && run >= startS) { voice = true; const t0 = t - run; if (!inTurn) { inTurn = true; turnStart = t0; } ev.push({ type: "speechstart", t, t0, resumed: !ended }); ended = false; pauseSent = false; silentSince = -1; } if (voice) lastVoice = t; }
      else { run = 0; if (voice) { voice = false; silentSince = t; ev.push({ type: "speechend", t, t0: t }); } }
      if (!voice && inTurn && silentSince >= 0) {
        const q = t - silentSince;
        if (!pauseSent && q >= pauseS) { pauseSent = true; ev.push({ type: "pause", t, t0: silentSince, turnLen: silentSince - turnStart }); }
        if (q >= turnEndS) { inTurn = false; ended = true; ev.push({ type: "turnend", t, t0: silentSince, turnLen: silentSince - turnStart }); }
      }
      return ev;
    }
  };
}

// ------------------------------------------------------------------ defaults
export const DEFAULT_BEHAVIOUR = {
  autoThink: true,              // the user's turn end starts "thinking" by itself (a host with an LLM may call think() instead)
  autoYield: false,             // barge-in: yield() by itself when the user talks over her (the host usually also stops the audio)
  thinkTimeoutS: 9,             // thinking without a reply falls back to listening
  idleAfterS: 8,                // no user voice this long after a reply / turn: idle
  gaze: {
    listen: { onS: [1.9, 3.6], awayS: [0.9, 1.6] },             // ~70 % on the user (Argyle)
    speak: { onS: [0.9, 1.7], awayS: [1.6, 3.0] },              // ~40 %
    idle: { onS: [1.5, 4], awayS: [1, 3] },
    turnStartAwayS: [1.0, 1.8],                                  // look away at the turn start, back 1-2 s in (Kendon)
    yieldLeadS: [0.6, 0.9],                                      // back on the user for the last 0.5-1 s, then hold
    thinkHoldS: [0.12, 0.25],                                    // a moment on the user after the turn end, then the thinking aversion
    thinkAwayS: [2.8, 4.2], thinkBackS: [0.5, 0.9]               // long waits: thinking aversions about 3.5 s with short glances back
  },
  blinkPerMin: { idle: 17, listening: 17, thinking: 20, speaking: 26, yielding: 17 },   // Bentivoglio; the viewer's blinker draws the intervals
  nod: { afterPauseS: 0.4, minTurnS: 2, delayS: [0.42, 0.5], refractoryS: 2.4, durS: 0.75 },   // backchannel nods: pause >= 0.4 s in a turn > 2 s
  maxNodDeg: 4, headAccel: 90,  // deg/s^2: the per-frame velocity change bound at 60 fps (1.5 deg/s per frame; motion spike: 1.5-1.95 after)
  prosody: { pitchDegPerZ: 0.9, pitchUpMax: 1.6, pitchDownMax: 1.0, nodDeg: [1.0, 2.6], browUp: 0.55, vowelGain: 0.1, vowelMin: 0.9, vowelMax: 1.1 },
  breath: { periodS: [3.8, 5], amp: 0.55, deep: 1, standInZoom: 0.0015, standInPitchDeg: 0.06 },
  smileShare: 0.3,              // smiling speech: this share of oo moves to ee per unit of smile (section 3.1.8)
  // trained expression controls (spikes/lf_train x*, models v23+; spikes/lf_expr): used when the model has them, else the synthetic / stand-in ones.
  // Weight 1 = the generated still's full expression (strong), so the cues stay well below it; caps hold every sum.
  expr: { brow: 0.6, browMax: 0.6, frown: 0.5, frownMax: 0.6, eyeSmile: 0.35, eyeSmileMax: 0.5, grin: 0.5, laugh: 0.6, openMax: 0.65, wide: 0.5, wideMax: 0.6,
          sadPerValence: 0.35, sadMax: 0.3, smileAh: 0.5, smileAhMax: 0.4, laughSmile: 0.3, joyValence: 0.6, joyOpen: 0.25, joyRefractoryS: 12 }
};

// ------------------------------------------------------------------ the engine: pure, stepped once per frame (tests drive it directly)
// o.controls: the model's control names; o.ranges: {name: [min, max]}; o.random; o.persona; o.clock: () => clip time of the playing reply (s)
export function createEngine(o = {}) {
  const cfg = mergeDeep(DEFAULT_BEHAVIOUR, o.config || {});
  const persona = o.persona && o.persona.thinkDir ? { ...DEFAULT_PERSONA, ...o.persona } : makePersona((o.persona && o.persona.seed) || o.seed || 1, o.persona || {});
  const rand = o.random || seededRandom((persona.seed || 1) * 31 + 5), rr = (a, b) => a + (b - a) * rand(), rrA = a => rr(a[0], a[1]);
  const names = (o.controls || []).slice(), has = n => names.includes(n), ranges = o.ranges || {};
  const handlers = {}; const emit = (ev, arg) => { for (const fn of (handlers[ev] || []).slice()) { try { fn(arg); } catch (e) { console.error(e); } } };
  let T = 0;                                                       // engine time (s)
  const st = { name: "idle", since: 0, manual: false, auto: o.auto !== false };
  const user = { voice: false, level: 0, turnStart: -1, inTurn: false, lastVoiceT: -99, lastStop: -99, pausesInTurn: 0, levelAvg: 0, riseRun: 0, pos: { x: 0, y: 0 }, present: null, smiling: 0,
                 voicedS: 0, turnS: 0, rate: 0, onsets: 0, f0: [], lastSeen: -99 };
  const reply = { active: false, t0: 0, clock: null, ana: null, cues: [], cueI: 0, end: null, dur: null, yielded: false, holdUntil: -1, speechSeen: false, lastSpeak: -99, accI: 0, phI: 0, nsI: 0, pauseI: 0 };
  const mood = { valence: 0, arousal: 0.4, engagement: 0.6 }, moodT = { ...mood };
  const pulses = [];                                               // { ch, t0, dur, amp, shape(u, t), tag }
  const last = {};                                                  // refractory memory: gesture -> last start time
  const hist = { avSide: [] };
  const gaze = { onUser: true, x: 0, y: 0, until: 0, kind: "user", override: null, lockOn: false };
  let inStep = false, pendingBlink = false, yieldHold = false, headF0 = 0, blinkNow = false, wasSpeaking = false, notSpeakingFor = 0, speakingFor = 0, idleQuiet = 0, settle = 0, lipPart = 0, browHold = 0, tilt = 0, tiltTarget = 0, lean = 0, leanTarget = 0, settled = false;
  const fol = { yaw: makeFollower(cfg.headAccel), pitch: makeFollower(cfg.headAccel), roll: makeFollower(cfg.headAccel * 0.6) };
  const noise = { yaw: makeNoise(rand, [{ hz: 0.06, amp: 1.6 }, { hz: 0.15, amp: 0.7 }, { hz: 0.37, amp: 0.2 }]), pitch: makeNoise(rand, [{ hz: 0.07, amp: 0.6 }, { hz: 0.19, amp: 0.3 }, { hz: 0.43, amp: 0.1 }]),
                  roll: makeNoise(rand, [{ hz: 0.05, amp: 0.5 }, { hz: 0.13, amp: 0.25 }]) };
  const breath = { v: 0.3, o1: 0.3, out: 0.3, phase: rand(), period: rrA(cfg.breath.periodS), inhale: null, deep: 0 };
  let blinkRateState = null;
  const memo = { visits: 0, greeted: false };

  function on(ev, fn) { (handlers[ev] || (handlers[ev] = [])).push(fn); return () => { const a = handlers[ev], i = a.indexOf(fn); if (i >= 0) a.splice(i, 1); }; }
  function rollRoom(side) { const L = o.limits ? o.limits() : null; if (!L || !L.roll) return 6; return Math.max(0, (side > 0 ? L.roll[1] : -L.roll[0]) / DEG); }   // degrees the viewer allows that way
  function refractory(name, s) { if (T - (last[name] ?? -99) < s) return false; last[name] = T; return true; }
  function pulse(ch, dur, amp, shape, t0 = T, tag) { pulses.push({ ch, t0, dur, amp, shape, tag }); }
  function ctl(name) { return "ctl:" + name; }

  // ---- state
  function setState(name, opts = {}) {
    if (!STATES.includes(name)) throw new Error("unknown state " + name);
    if (opts.manual) st.manual = true;
    if (st.name === name) return;
    const prev = st.name; st.name = name; st.since = T;
    enter(name, prev);
    emit("state", { state: name, prev, t: T });
  }
  function enter(name, prev) {
    if (name === "thinking") {
      // a moment on the user, then the thinking aversion (persona direction), an inbreath, lips part slightly
      gaze.override = null; setGaze(true, rrA(cfg.gaze.thinkHoldS), "user");
      inhale(0.7, 1);
      if (rand() < 0.5 && refractory("blinkThink", 2)) blinkLater(gaze.until - T);   // people blink on large gaze shifts
    } else if (name === "speaking") {
      reply.speechSeen = true; gaze.lockOn = false;
      // back 1-2 s in; a short reply comes back later in that range so that the turn-end hold is not most of it
      const R = reply.active && reply.end !== null ? reply.end - reply.yieldLead - clipTime() : 9, away = R > 3.5 ? rrA(cfg.gaze.turnStartAwayS) : clamp(R - 0.9, 1.0, 2.0);
      if (gaze.onUser || gaze.kind !== "think") setGaze(false, away, "turnstart", speakAway(true));
      else gaze.until = T + away;                                     // already looking away (thinking): keep it, come back 1-2 s in
      leanTarget = 0; tiltTarget = 0;
    } else if (name === "yielding") {
      setGaze(true, 1.5, "yield"); gaze.lockOn = true;
      if (has("browUp")) pulse(ctl("browUp"), 1.1, 0.25 * persona.expressiveness, u => envelope(u * 1.1, 1.1, 0.25, 0.6));
      if (mood.valence >= -0.2) pulse(ctl("smile"), 1.6, 0.08, u => envelope(u * 1.6, 1.6, 0.3, 0.9));
    } else if (name === "listening") {
      gaze.lockOn = false; if (!gaze.onUser) setGaze(true, rrA(cfg.gaze.listen.onS), "user");
    } else if (name === "idle") { gaze.lockOn = false; idleQuiet = 0; }
    applyBlinkRate();
  }
  // the viewer's own blinker keeps its interval after a triggered blink, so the blinks this module triggers (pauses, phrase ends, cues) come out of the
  // state's budget: the viewer is given the rest (at least 6 / min)
  const evBlinks = [];
  function applyBlinkRate() {
    while (evBlinks.length && T - evBlinks[0] > 20) evBlinks.shift();
    const evRate = Math.round(evBlinks.length * 60 / Math.max(10, Math.min(20, T)) / 2) * 2;   // whole steps: the viewer's preset is re-set only on a change
    const per = Math.max(6, (cfg.blinkPerMin[st.name] || 17) - evRate), k = persona.blinkScale * (0.85 + 0.35 * mood.arousal), mean = 60 / (per * k);
    const iv = [+(mean * 0.65).toFixed(2), +(mean * 1.35).toFixed(2)];
    if (!blinkRateState || blinkRateState[0] !== iv[0] || blinkRateState[1] !== iv[1]) { blinkRateState = iv; emit("blinkrate", { state: st.name, intervalS: iv, perMin: +(60 / mean).toFixed(1) }); }
  }
  // ---- gaze
  function speakAway(first) {
    // side aversion (alternating more often than not: refractory memory), a little up or down
    const lastSide = hist.avSide[hist.avSide.length - 1] || 0, side = lastSide && rand() < 0.65 ? -lastSide : (rand() < 0.5 ? -1 : 1);
    hist.avSide.push(side); if (hist.avSide.length > 6) hist.avSide.shift();
    const a = persona.aversionDeg * rr(0.75, 1.2) * (first ? 1.1 : 1);
    return { x: side * a, y: rr(-0.35, 0.25) * a, head: -side * rr(1.5, 2.8) };   // the head turns a little the same way as the eyes
  }
  function thinkAway() {
    const d = persona.thinkDir, sx = /left/.test(d) ? -1 : 1, up = /up/.test(d) ? 1 : 0.2, a = persona.aversionDeg * rr(1.1, 1.4);
    return { x: sx * a, y: up * a * rr(0.6, 0.85), head: -sx * rr(2, 3.2), headPitch: up * rr(0.8, 1.4) };
  }
  // a hesitant user (slow, quiet speech): softer, smaller aversions
  function listenAway() { const hes = user.turnS > 2 && user.voicedS / Math.max(0.1, user.turnS) < 0.45, side = rand() < 0.5 ? -1 : 1, a = persona.aversionDeg * rr(0.4, 0.65) * (hes ? 0.6 : 1); return { x: side * a, y: -rr(0.3, 0.6) * a, head: -side * rr(0.5, 1.2) }; }
  function setGaze(onUser, dur, kind, away) {
    gaze.onUser = onUser; gaze.kind = kind; gaze.until = T + dur; gaze.t0 = T;
    if (!onUser) { gaze.x = away.x; gaze.y = away.y; gaze.head = away.head || 0; gaze.headPitch = away.headPitch || 0; } else { gaze.head = 0; gaze.headPitch = 0; }
    emit("gaze", { onUser, kind, t: T, until: gaze.until });
  }
  function planGaze() {
    const s = st.name, G = cfg.gaze;
    if (gaze.override && T >= gaze.override.until) gaze.override = null;
    if (s === "speaking" && reply.active && !reply.yielded && reply.end !== null) {
      const tc = clipTime(), rem = reply.end - tc;
      if (rem <= reply.yieldLead && T >= reply.holdUntil) { reply.yielded = true; gaze.override = null; setGaze(true, rem + 2, "yield"); gaze.lockOn = true; emit("turnyield", { t: T, clipTime: tc, remaining: rem }); return; }
    }
    if (gaze.lockOn) return;
    if (T < gaze.until) return;
    if (s === "thinking") {
      const thinkS = T - st.since;
      if (gaze.onUser) setGaze(false, rrA(G.thinkAwayS), "think", thinkAway());
      else if (thinkS > 3) setGaze(true, rrA(G.thinkBackS), "user");          // long wait: a short glance back
      else gaze.until = T + 0.5;
      return;
    }
    if (s === "speaking") {
      let on = !gaze.onUser, dur = on ? rrA(G.speak.onS) : rrA(G.speak.awayS);
      if (reply.active && reply.end !== null && !reply.yielded) {     // plan backwards from the turn-end hold: the last aversion ends at it
        const room = reply.end - reply.yieldLead - clipTime();
        if (on) { if (room - dur < 0.9) dur = room >= 1.5 ? Math.max(0.6, room - rr(0.9, 1.4)) : room + 0.5; }
        else if (room < 0.6) { on = true; dur = room + 0.5; }
        else dur = Math.min(dur, room);
      }
      if (on) setGaze(true, dur, "user"); else setGaze(false, dur, "speak", speakAway(false));
      return;
    }
    if (s === "listening" || s === "yielding") {
      const on = !gaze.onUser;
      if (on || user.voice === false && T - user.lastStop < 0.6) setGaze(true, rrA(G.listen.onS), "user");     // keep the eyes on the user at their pauses
      else setGaze(false, rrA(G.listen.awayS), "listen", listenAway());
      return;
    }
    // idle / presence: the user (if any) half the time; after a long quiet the attention drifts (settle)
    const on = !gaze.onUser && !settled;
    if (on) setGaze(true, rrA(G.idle.onS), "user");
    else { const a = listenAway(); if (settled) { a.y = -persona.aversionDeg * rr(0.7, 1); a.x *= 0.6; } setGaze(false, settled ? rr(4, 8) : rrA(G.idle.awayS), settled ? "settle" : "idle", a); }
  }
  // ---- breath: rest cycle 12-16 / min; inhales before replies and at her pauses; exhale across a phrase
  function inhale(dur, depth, at = T) { breath.inhale = { t0: at, dur, depth }; }
  function stepBreath(dt) {
    const sp = st.name === "speaking";
    if (breath.inhale && T >= breath.inhale.t0) {
      const b = breath.inhale, u = (T - b.t0) / b.dur;
      if (T >= b.t0 + b.dur + 0.3) breath.inhale = null; else breath.v = Math.max(breath.v, b.depth * ss(u));   // a short hold at the top
    }
    if (breath.inhale && T >= breath.inhale.t0) { /* inhaling */ }
    else if (sp) breath.v = Math.max(0, breath.v - dt * 0.12);           // a slow exhale across the phrase
    else {
      breath.phase += dt / breath.period; if (breath.phase >= 1) { breath.phase -= 1; breath.period = rrA(cfg.breath.periodS); }
      const p = breath.phase, target = p < 0.4 ? ss(p / 0.4) : 1 - ss((p - 0.4) / 0.6);   // inhale 40 %, exhale 60 %
      breath.v += (target * cfg.breath.amp - breath.v) * Math.min(1, dt * 3);
    }
    const k = Math.min(1, dt / 0.09); breath.o1 += (breath.v - breath.o1) * k; breath.out += (breath.o1 - breath.out) * k;   // two lags: no step in what the chest does
  }
  // ---- gestures
  function nod(depth, opts = {}) {
    depth = Math.min(depth, cfg.maxNodDeg, 3); const dur = opts.dur || cfg.nod.durS, dbl = opts.double ?? persona.nodStyle === "double";
    if (dbl) { pulse("pitch", dur * 0.75, -depth * 0.75, bump); pulse("pitch", dur * 0.7, -depth * 0.45, bump, T + dur * 0.62); }
    else pulse("pitch", dur, -depth, u => bump(u) - 0.12 * bump(clamp(u * 3 - 2, 0, 1)));   // down, a hint of rebound
    emit("cue", { name: "nod", t: T, depth });
  }
  function blinkLater(dt) { pulses.push({ ch: "blinkNow", t0: T + Math.max(0, dt), dur: 0.02, amp: 1, shape: () => 1 }); }
  const BLINK_RANK = { false: 0, half: 1, true: 2, long: 3 };
  function doBlink(kind = true) { if (kind === false) return; if (evBlinks.length && T - evBlinks[evBlinks.length - 1] < (kind === "long" ? 0.6 : 1.2)) return;   // not two of ours within 1.2 s (0.6 before a long one)
    evBlinks.push(T); const cur = inStep ? blinkNow : pendingBlink; const k = BLINK_RANK[String(kind)] > BLINK_RANK[String(cur)] ? kind : cur; if (inStep) blinkNow = k; else pendingBlink = k; }
  function longBlink() { doBlink("long"); emit("cue", { name: "longblink", t: T }); }   // the viewer's long blink (~0.6 s: close 0.13, shut 0.27, open 0.2)

  function smile(amount, dur, rise = 0.3) { pulse(ctl("smile"), dur, amount, u => envelope(u * dur, dur, rise, Math.min(persona.smileFadeS, dur * 0.6))); }
  function brow(amount, dur = 0.8) {
    if (has("xBrowUp")) pulse(ctl("xBrowUp"), dur, amount * cfg.expr.brow * persona.expressiveness, u => envelope(u * dur, dur, 0.18, 0.45));   // trained brow raise (forehead lines)
    else if (has("browUp")) pulse(ctl("browUp"), dur, amount * persona.expressiveness, u => envelope(u * dur, dur, 0.18, 0.45)); else pulse("pitch", dur, 0.6 * amount, bump);
  }
  const env = (dur, rise, fall) => u => envelope(u * dur, dur, rise, fall);
  // an open, toothy smile (grin / laugh / joy): xSmileOpen with smile (its still has smile 1); faded out while she talks (stepped below)
  function openSmile(amount, dur, rise = 0.2) { smile(Math.min(1, amount * 1.2), dur, rise); pulse(ctl("xSmileOpen"), dur, amount, env(dur, rise, Math.min(0.6, dur * 0.4))); }
  function cue(name, opts = {}) {
    const k = (opts.strength ?? 1) * persona.expressiveness, at = opts.at ?? T, untilT = opts.until;
    if (at > T + 1e-6) { pulses.push({ ch: "cue", t0: at, dur: 0.02, amp: 0, shape: () => 0, cue: { name, opts: { ...opts, at: undefined } } }); return; }
    const dur = opts.dur || (untilT ? Math.max(0.6, untilT - T) : null);
    switch (name) {
      case "smile": smile(0.35 * k, dur || 2.2); if (has("xEyeSmile")) pulse(ctl("xEyeSmile"), dur || 2.2, cfg.expr.eyeSmile * k, env(dur || 2.2, 0.35, Math.min(persona.smileFadeS, (dur || 2.2) * 0.6))); break;   // a Duchenne smile: the eyes smile too
      case "grin":
        if (has("xSmileOpen")) { openSmile(cfg.expr.grin * k, dur || 1.4); if (has("xEyeSmile")) pulse(ctl("xEyeSmile"), dur || 1.4, 0.5 * cfg.expr.eyeSmile * k, env(dur || 1.4, 0.2, 0.5)); break; }
        smile(0.75 * k, dur || 1.4, 0.2); if (has("blink")) pulse(ctl("blink"), dur || 1.4, 0.12, u => bump(u)); break;   // a grin narrows the eyes a touch
      case "laugh":
        if (has("xSmileOpen")) { openSmile(cfg.expr.laugh * k, dur || 1.8, 0.15); if (has("xEyeSmile")) pulse(ctl("xEyeSmile"), dur || 1.8, cfg.expr.eyeSmile * k, env(dur || 1.8, 0.15, 0.6)); }
        else { smile(0.8 * k, dur || 1.8, 0.15); if (has("blink")) pulse(ctl("blink"), dur || 1.8, 0.15, u => bump(u)); }
        break;
      case "brow": brow(0.7 * k, dur || 0.8); break;
      case "frown": if (has("xFrown")) pulse(ctl("xFrown"), dur || 1.5, cfg.expr.frown * k, env(dur || 1.5, 0.25, 0.5)); else if (has("furrow")) pulse(ctl("furrow"), dur || 1.5, 0.6 * k, env(dur || 1.5, 0.25, 0.5)); break;
      case "thinks": gaze.override = { until: T + (dur || 1.3) }; setGaze(false, dur || 1.3, "think", thinkAway()); break;
      case "glance": gaze.override = { until: T + 0.55 }; setGaze(false, 0.55, "glance", { x: (rand() < 0.5 ? -1 : 1) * persona.aversionDeg * 1.3, y: rr(-2, 2) }); break;
      case "nod": nod(persona.nodDeg * k, { double: false }); break;
      case "shake": {
        if (!o.canShake || !o.canShake()) { nod(1.2 * k); break; }     // far side must be clean at the current yaw; else a small nod
        const dd = dur || 1.1; pulse("yaw", dd, 3 * k, u => Math.sin(2 * Math.PI * 2 * u) * bump(u)); break;
      }
      case "wide":
        if (has("xEyesWide")) { pulse(ctl("xEyesWide"), dur || 1, cfg.expr.wide * k, env(dur || 1, 0.15, 0.4)); brow(0.3 * k, dur || 1); break; }
        if (has("blink")) pulse(ctl("blink"), dur || 1, -0.22 * k, u => envelope(u * (dur || 1), dur || 1, 0.15, 0.4)); brow(0.5 * k, dur || 1); break;
      case "wink": {
        const w = names.find(n => /^(wink|blink)(L|R|Left|Right)$/.test(n));
        if (w && o.canWink && o.canWink()) pulse(ctl(w), 0.4, 1, u => bump(u));
        else signature(); break;                                       // no per-eye lid control: her slow blink + half smile instead
      }
      default: return;
    }
    emit("cue", { name, t: T, opts });
  }
  function signature() { longBlink(0.55); smile(0.22, 1.8); }

  // ---- user side
  function userEvent(e) {
    const tt = e.at ?? T;
    if (e.type === "level") { user.level = e.level; return; }
    if (e.type === "speechstart") {
      user.voice = true; user.lastVoiceT = tt; user.lastSeen = T; if (!user.inTurn) { user.inTurn = true; user.turnStart = e.t0 ?? tt; user.pausesInTurn = 0; user.onsets = 0; }
      user.onsets++; pendingNod = null; settled = false;
      // laughter-like bursts (3.6.3): three or more short voice onsets within 1.5 s -> she smiles with it
      user.bursts = (user.bursts || []).filter(x => T - x < 1.5); user.bursts.push(T);
      if (user.bursts.length >= 3 && st.name !== "speaking" && refractory("laughSmile", 6)) {
        if (has("xSmileOpen")) openSmile(cfg.expr.laughSmile * persona.expressiveness, 2.6, 0.35); else smile(0.4 * persona.expressiveness, 2.6, 0.35);
        emit("cue", { name: "laughsmile", t: T });
      }
      if (st.auto && (st.name === "idle" || st.name === "yielding" || st.name === "thinking")) setState("listening");
      if (st.name === "idle" && idleQuiet > 15) notice();
      if (st.name === "speaking") { emit("bargein", { t: T }); if (cfg.autoYield) yieldTurn(); }
      return;
    }
    if (e.type === "speechend") { user.voice = false; user.lastStop = e.t0 ?? tt; return; }
    if (e.type === "pause") {
      user.voice = false; user.lastStop = e.t0 ?? user.lastStop; user.pausesInTurn++;
      if (st.name !== "listening") return;
      if (rand() < 0.5 && refractory("blinkPause", 1.5)) doBlink(true);      // listeners blink at the speaker's pauses
      const turnLen = user.lastStop - user.turnStart, pauseLen = T - user.lastStop, N = cfg.nod;
      // backchannel: only after a pause >= 0.4 s inside a turn > 2 s; decided at 0.4 s, played 0.42-0.62 s after the pause began
      if (turnLen >= N.minTurnS) pendingNod = { stop: user.lastStop, at: user.lastStop + Math.max(N.afterPauseS + 0.02, rrA(N.delayS)), decideAt: user.lastStop + N.afterPauseS, pauseLen };
      return;
    }
    if (e.type === "turnend") {
      user.inTurn = false; user.voice = false; user.lastStop = e.t0 ?? user.lastStop;
      const turnLen = e.turnLen ?? (user.lastStop - user.turnStart);
      userTurnEnded(turnLen, e);
    }
  }
  let pendingNod = null;
  function userTurnEnded(turnLen, e) {
    // context reactions (3.6.3): a question (rising pitch at the end) -> brow; laughter (energy bursts) -> smile; long turn -> "got it"
    if (e && e.question) brow(0.6, 0.9);
    if (st.name === "listening" && turnLen > 2.5 && !pendingNod && T - (last.nod ?? -99) > 1.5 && rand() < 0.45 && refractory("gotit", 6)) { last.nod = T; longBlink(0.6); nod(persona.nodDeg * 0.8, { double: false }); }
    leanTarget = 0; tiltTarget = 0;
    if (st.auto && cfg.autoThink && st.name === "listening") setState("thinking");
  }
  function stepUser(dt) {
    // pending backchannel nod: cancelled when the user goes on before it, never during speech
    if (pendingNod) {
      if (user.voice || user.level > 0.2 || st.name !== "listening") pendingNod = null;   // the user goes on: no nod (level reacts before the voice detector)
      else if (T >= pendingNod.at) {
        const p = pendingNod; pendingNod = null;
        const hes = user.turnS > 2 && user.voicedS / Math.max(0.1, user.turnS) < 0.45;   // hesitant user: fewer nods
        const chance = persona.backchannel * (hes ? 0.5 : 1);
        const gap = cfg.nod.refractoryS * (user.rate > 0 ? clamp(4 / user.rate, 0.7, 1.4) : 1);   // mirrors the user's pace
        if (rand() < chance && refractory("nod", gap)) { nod(persona.nodDeg); if (rand() < 0.2) longBlink(0.5); }
      }
    }
    // listening: tilt after 1.2 s of the user talking, lean in after 2 s, small smile when the user's voice rises in energy
    if (st.name === "listening" && user.inTurn) {
      const tl = T - user.turnStart; user.turnS = tl; if (user.voice) user.voicedS += dt;
      if (tl > 1.2) {                                                // toward the persona's side if the model has room there (v21b: roll -1..+4), else the other
        let side = persona.tiltSide === "left" ? 1 : -1; const want = persona.tiltDeg * (0.8 + 0.4 * mood.engagement);
        if (rollRoom(side) < 2.5 && rollRoom(-side) > rollRoom(side)) side = -side;
        tiltTarget = side * Math.min(want, 0.85 * rollRoom(side));
      }
      if (tl > 2) leanTarget = persona.leanIn * (0.6 + 0.6 * mood.engagement);
      user.levelAvg += (user.level - user.levelAvg) * Math.min(1, dt / 2.5);
      if (user.voice && user.level > user.levelAvg + 0.18) user.riseRun += dt; else user.riseRun = 0;
      if (user.riseRun > 0.3 && refractory("listenSmile", 7)) { smile(0.16, 2.2, 0.4); user.riseRun = 0; }
      if (tl > 2) user.rate = user.onsets / tl;                     // voice onsets per second: a rough pace
    } else if (!user.inTurn) { user.voicedS = 0; user.turnS = 0; }
    if (st.name !== "listening") { if (st.name !== "yielding") tiltTarget = 0; leanTarget = 0; }
  }
  function notice() {                                               // the "noticing" beat after a long quiet: head turns to the user, brow, gaze, smile
    settled = false; idleQuiet = 0; gaze.override = null; setGaze(true, rr(2, 3.5), "user");
    pulse("yaw", 0.9, (rand() < 0.5 ? -1 : 1) * 1.6, bump); brow(0.5, 0.9); smile(0.12 + (memo.visits > 1 ? 0.1 : 0), 2);
    emit("cue", { name: "notice", t: T });
  }

  // ---- reply (her clip)
  function speak(clip = {}) {
    const ana = clip.analysis || (clip.pcm && clip.sampleRate ? analyseClip(clip.pcm, clip.sampleRate, { timings: clip.timings, nonSpeech: clip.nonSpeech }) : null);
    reply.active = true; reply.t0 = T; reply.clock = clip.clock || null; reply.ana = ana; reply.yielded = false; reply.speechSeen = false;
    reply.cues = (clip.cues || []).slice().sort((a, b) => a.t - b.t); reply.cueI = 0; reply.accI = 0; reply.phI = 0; reply.nsI = 0; reply.pauseI = 0;
    reply.dur = clip.duration ?? (ana ? ana.duration : null);
    const lastLetter = Array.isArray(clip.timings) ? [...clip.timings].reverse().find(c => /[\p{L}\p{N}]/u.test(c.char || "")) : null;
    reply.end = lastLetter ? lastLetter.end : ana ? ana.speechEnd : reply.dur;
    reply.start = Array.isArray(clip.timings) && clip.timings.length ? (clip.timings.find(c => /[\p{L}\p{N}]/u.test(c.char || "")) || {}).start ?? 0 : ana ? ana.speechStart : 0;
    reply.yieldLead = rrA(cfg.gaze.yieldLeadS); reply.holdUntil = -1;
    if (ana && ana.speechStart > 0.35) inhale(Math.min(0.6, ana.speechStart), cfg.breath.deep);   // an inhale before the first word
    return ana;
  }
  function clipTime() { if (reply.clock) { const c = reply.clock(); if (Number.isFinite(c)) return c; } return T - reply.t0; }
  function stepReply(dt, f) {
    if (!reply.active) return;
    const tc = clipTime(), A = reply.ana, P = cfg.prosody;
    // clip-anchored cues (LLM tags): fire 0.12 s ahead of their word
    while (reply.cueI < reply.cues.length && reply.cues[reply.cueI].t - 0.12 <= tc) {
      const c = reply.cues[reply.cueI++], k = c.strength ?? 1;
      const thru = c.name === "smile" ? Math.max(0.8, (c.phraseEnd ?? c.t + 1.5) - tc + 0.3) : undefined;
      cue(c.name, { strength: k, dur: thru });
    }
    if (!A) return;
    // accents: a nod scaled by pitch and energy (cap 4 deg incl. the rest of the head motion), a brow on the bigger pitch peaks
    while (reply.accI < A.accents.length && A.accents[reply.accI].t - 0.3 <= tc) {
      const a = A.accents[reply.accI++]; if (a.t < tc - 0.1) continue;
      const depth = clamp(P.nodDeg[0] + (P.nodDeg[1] - P.nodDeg[0]) * clamp(0.5 * (a.f0z + a.ez) / 2, 0, 1), P.nodDeg[0], P.nodDeg[1]) * persona.expressiveness;
      // a beat: the head rides up with the pitch rise into the accent, then drops past rest after it (peak down ~0.2 s after the vowel)
      if (refractory("beat", 0.9)) pulse("pitch", 0.9, Math.min(depth, cfg.maxNodDeg - 0.5), u => 0.35 * bump(clamp(u / 0.45, 0, 1)) - bump(clamp((u - 0.25) / 0.75, 0, 1)), T, "beat");
      if (a.big && refractory("browBeat", 1.8)) brow(P.browUp, 0.75);
    }
    // blinks at phrase boundaries (not inside a stressed word): at the phrase end, 60 %
    while (reply.phI < A.phraseEnds.length && A.phraseEnds[reply.phI] <= tc) { reply.phI++; if (rand() < 0.45 && refractory("blinkPhrase", 2.2)) doBlink(true); }
    // her own pauses >= 0.4 s: a quick inhale
    while (reply.pauseI < A.pauses.length && A.pauses[reply.pauseI].t0 <= tc) { const p = A.pauses[reply.pauseI++]; if (p.t1 - p.t0 >= 0.4) inhale(Math.min(0.45, p.t1 - p.t0 - 0.05), 0.85); }
    // non-speech in the clip: a breath -> the breath control; a sigh -> a long exhale + gaze down
    while (reply.nsI < A.nonSpeech.length && A.nonSpeech[reply.nsI].start <= tc) {
      const x = A.nonSpeech[reply.nsI++];
      if (x.kind === "breath") inhale(Math.max(0.25, x.end - x.start), 0.9);
      else { breath.v = Math.max(breath.v, 0.9); pulse("breathOut", x.end - x.start + 0.6, 1, u => u); gaze.override = { until: T + (x.end - x.start) + 0.5 }; setGaze(false, x.end - x.start + 0.5, "sigh", { x: rr(-3, 3), y: -persona.aversionDeg * 0.9, head: 0, headPitch: -0.6 }); emit("cue", { name: "sigh", t: T }); }
    }
  }
  function endReply() { reply.active = false; reply.clock = null; }
  function yieldTurn() {                                            // barge-in: brows up a touch, eyes on the user, give up the floor
    brow(0.45, 0.9); gaze.override = null; setGaze(true, 2.5, "user"); gaze.lockOn = false; reply.yielded = true;
    endReply(); yieldHold = true; setState("listening"); emit("yield", { t: T });
  }
  function holdFloor(s = 2) {                                       // "I am not done": gaze away, a slight inhale, no turn yield until it passes
    reply.holdUntil = T + s; gaze.override = { until: T + Math.min(s, 2.5) }; setGaze(false, Math.min(s, 2.5), "hold", speakAway(false)); inhale(0.45, 0.7); gaze.lockOn = false; reply.yielded = false;
    emit("holdfloor", { t: T, until: T + s });
  }

  // ---- the frame
  const ctlOut = {};
  function step(f = {}) {
    const dt = clamp(f.dt ?? 1 / 60, 0, 0.25); T += dt; blinkNow = pendingBlink; pendingBlink = false; inStep = true;
    for (const k of Object.keys(ctlOut)) delete ctlOut[k];
    // mood eases toward its target (seconds)
    for (const k of ["valence", "arousal", "engagement"]) mood[k] += (moodT[k] - mood[k]) * Math.min(1, dt / 1.5);
    // speaking from the viewer (clip playing + driver), with a hangover across her own pauses
    const vs = !!f.speaking;
    if (vs) { speakingFor += dt; notSpeakingFor = 0; reply.lastSpeak = T; } else { notSpeakingFor += dt; speakingFor = 0; }
    wasSpeaking = vs;
    if (yieldHold && notSpeakingFor > 0.3) yieldHold = false;      // after a yield the clip may run on for a moment until the host stops it
    if (st.auto && vs && !yieldHold && st.name !== "speaking" && (speakingFor > 0.05 || reply.active)) setState("speaking");
    if (st.auto && st.name === "speaking" && T >= reply.holdUntil) {   // holdFloor: the floor is kept (gaze away) until it passes, even if the clip ended
      const tc = reply.active ? clipTime() : 0;
      const over = reply.active && reply.end !== null ? tc >= reply.end + 0.1 && notSpeakingFor > 0.05 : notSpeakingFor > 0.7;
      if (over || (!reply.active && notSpeakingFor > 0.7) || (reply.active && notSpeakingFor > 1.5 && (reply.dur === null || tc > reply.dur - 0.05))) {
        if (!reply.yielded) emit("turnyield", { t: T, late: true });
        endReply(); setState("yielding");
      }
    }
    if (st.auto && st.name === "yielding" && T - st.since > 1.6) setState(user.inTurn ? "listening" : "listening");
    if (st.auto && st.name === "thinking" && T - st.since > cfg.thinkTimeoutS) setState("listening");
    if (st.auto && st.name === "listening" && !user.inTurn && T - Math.max(user.lastStop, st.since) > cfg.idleAfterS) setState("idle");
    if (st.name === "idle") { idleQuiet += dt; if (idleQuiet > (o.settleAfterS ?? 30)) settled = true; } else { idleQuiet = 0; settled = false; }
    stepUser(dt); stepReply(dt, f); planGaze(); stepBreath(dt); applyBlinkRate();

    // ---- head: band-limited noise (scaled by arousal), gaze-led head turn, tilt, prosody pitch, pulses; then the acceleration bound
    const ar = 0.6 + 0.8 * mood.arousal, sp = st.name === "speaking";
    let yaw = noise.yaw(T) * ar * (sp ? 1.2 : 1), pitch = noise.pitch(T) * ar, roll = noise.roll(T) * ar, zoom = 0;
    if (!gaze.onUser) { yaw += gaze.head || 0; pitch += gaze.headPitch || 0; }
    else if (user.pos) yaw -= clamp(user.pos.x * 0.3, -6, 6);        // part of the way toward where the user actually is (camera attention)
    if (settled) pitch -= 1.2;
    tilt += (tiltTarget - tilt) * Math.min(1, dt / 0.9); roll += tilt;
    lean += (leanTarget - lean) * Math.min(1, dt / 0.6); zoom -= lean;   // closer
    const fz = sp && reply.active && reply.ana ? reply.ana.f0zAt(clipTime()) : 0; headF0 += (fz - headF0) * Math.min(1, dt / 0.15);
    { const P = cfg.prosody; pitch += clamp(headF0 * P.pitchDegPerZ * persona.expressiveness, -P.pitchDownMax, P.pitchUpMax); }   // the head follows the pitch (Munhall)
        for (let i = pulses.length - 1; i >= 0; i--) {
      const p = pulses[i], u = (T - p.t0) / p.dur;
      if (u < 0) continue;
      if (p.ch === "blinkNow") { doBlink(true); pulses.splice(i, 1); continue; }
      if (p.ch === "cue") { pulses.splice(i, 1); cue(p.cue.name, p.cue.opts); continue; }
      if (u >= 1) { pulses.splice(i, 1); continue; }
      const v = p.amp * p.shape(u, T - p.t0);
      if (p.ch === "pitch") pitch += v; else if (p.ch === "yaw") yaw += v; else if (p.ch === "roll") roll += v; else if (p.ch === "zoom") zoom += v;
      else if (p.ch === "breathOut") breath.v = Math.max(0, breath.v - 0.02 * v);
      else if (p.ch.startsWith("ctl:")) { const n = p.ch.slice(4); ctlOut[n] = (ctlOut[n] || 0) + v; }
    }
    // chin down: throat flecks below -3. Stay inside the viewer's limits short of its soft knee (it eases from 75 % of the room), so its knee adds no jerk
    { const L = o.limits ? o.limits() : null, dn = L && L.pitch ? -L.pitch[0] / DEG : 3.5, up = L && L.pitch ? L.pitch[1] / DEG : 4; pitch = clamp(pitch, -Math.min(cfg.maxNodDeg - 0.5, 0.82 * dn), 0.72 * up); }
    yaw = fol.yaw.step(yaw, dt); pitch = fol.pitch.step(pitch, dt); roll = fol.roll.step(clamp(roll, -6, 6), dt);

    // ---- face: mood baseline, breath, lips during the wait, energy-scaled vowels, smiling speech
    const dw = f.driverWeights || {};
    const smileBase = persona.smileBase + clamp(0.22 * mood.valence, -0.05, 0.22);
    if (has("smile")) ctlOut.smile = (ctlOut.smile || 0) + Math.max(0, smileBase);
    if (has("blink")) {                                              // lids: relaxed when calm or settled, a touch wider when engaged
      let lid = (1 - mood.arousal) * 0.08 + (settled ? 0.08 : 0) - Math.max(0, mood.engagement - 0.7) * 0.15;
      ctlOut.blink = (ctlOut.blink || 0) + lid;
    }
    if (has("browUp")) ctlOut.browUp = (ctlOut.browUp || 0) + Math.max(0, mood.engagement - 0.6) * 0.3;
    if (has("xSad") && mood.valence < 0) ctlOut.xSad = (ctlOut.xSad || 0) + Math.min(cfg.expr.sadMax, -mood.valence * cfg.expr.sadPerValence);   // a sad mood: inner brows up, corners down
    if (has("xSmileOpen") && mood.valence >= cfg.expr.joyValence && st.name !== "speaking" && !user.voice && refractory("joy", cfg.expr.joyRefractoryS) && rand() < 0.5) {
      openSmile(cfg.expr.joyOpen * persona.expressiveness, 2.2, 0.4); emit("cue", { name: "joy", t: T });   // joy: now and then an open smile between turns
    }
    const bv = breath.out;
    if (has("breath")) ctlOut.breath = (ctlOut.breath || 0) + bv;
    else { zoom -= cfg.breath.standInZoom * bv; pitch += cfg.breath.standInPitchDeg * bv; }   // stand-in: the bust swells toward the camera a hair
    // the reply gap: lips part slightly (mid 0.15) while thinking; gone before the first sound
    const anySpeech = ["ah", "wide", "oo", "ee", "mid", "press"].some(n => (dw[n] || 0) > 0.02);
    const wantLip = st.name === "thinking" && T - st.since > 0.5 && !anySpeech && !vs ? 0.15 : 0;
    lipPart += (wantLip - lipPart) * Math.min(1, dt / (wantLip > lipPart ? 0.35 : 0.06)); if (anySpeech || vs) lipPart = Math.min(lipPart, 0.15 * Math.max(0, 1 - 20 * (dw.press || 0)));
    if (has("mid") && lipPart > 0.002) ctlOut.mid = (ctlOut.mid || 0) + lipPart;
    if (sp && reply.active) {
      const P = cfg.prosody, open = ["ah", "wide", "oo", "ee", "mid"].filter(has), sum = open.reduce((s, n) => s + (dw[n] || 0), 0);
      // energy-scaled vowel opening (3.5.1): only on open vowels (sum > 0.2), never on a closure (press), and never past an opening sum of 1
      if (reply.ana && sum > 0.2 && (dw.press || 0) < 0.05) {
        const z = reply.ana.ezAt(clipTime()), g = clamp(1 + P.vowelGain * z, P.vowelMin, P.vowelMax), room = Math.max(sum, 1);
        let k = g; if (sum * k > room) k = room / sum;
        const fade = ss((sum - 0.2) / 0.15);
        for (const n of open) ctlOut[n] = (ctlOut[n] || 0) + (dw[n] || 0) * (k - 1) * fade;
      }
      // smiling speech (3.1.8): a held smile spreads the lips: part of oo moves to ee
      const sm = (ctlOut.smile || 0) + (dw.smile || 0);
      if (sm > 0.15 && has("oo") && has("ee") && (dw.oo || 0) > 0) { const d = cfg.smileShare * clamp(sm, 0, 1) * dw.oo; ctlOut.oo = (ctlOut.oo || 0) - d; ctlOut.ee = (ctlOut.ee || 0) + 0.8 * d; }
      // smiling speech with the trained corrective (xSmileAh: its still = ah 1 + smile 1): a share of the ah opening laughs, only while ah is open
      if (has("xSmileAh") && sm > 0.2 && (dw.press || 0) < 0.05) ctlOut.xSmileAh = (ctlOut.xSmileAh || 0) + Math.min(cfg.expr.smileAhMax, cfg.expr.smileAh * ss((sm - 0.2) / 0.5) * (dw.ah || 0));
    }
    // the open smile never fights the speech driver: gone on a closure, a fifth of it while she talks (the corrective above carries the laugh)
    if (ctlOut.xSmileOpen) {
      const closing = (dw.press || 0) > 0.02 || ((dw.mid || 0) > 0.02 && !["ah", "wide", "oo", "ee"].some(n => (dw[n] || 0) > 0.05));
      ctlOut.xSmileOpen *= closing ? 0 : (vs || anySpeech ? 0.2 : 1);
    }
    // caps: trained expressions stay modest; the mouth ones together under openMax
    const capX = { xBrowUp: cfg.expr.browMax, xFrown: cfg.expr.frownMax, xEyeSmile: cfg.expr.eyeSmileMax, xEyesWide: cfg.expr.wideMax, xSad: cfg.expr.sadMax, xSmileOpen: cfg.expr.openMax, xSmileAh: cfg.expr.smileAhMax };
    for (const n of Object.keys(capX)) if (ctlOut[n] !== undefined) ctlOut[n] = clamp(ctlOut[n], 0, capX[n]);
    if ((ctlOut.xSmileOpen || 0) + (ctlOut.xSmileAh || 0) > cfg.expr.openMax) { const f2 = cfg.expr.openMax / ((ctlOut.xSmileOpen || 0) + (ctlOut.xSmileAh || 0)); if (ctlOut.xSmileOpen) ctlOut.xSmileOpen *= f2; if (ctlOut.xSmileAh) ctlOut.xSmileAh *= f2; }
    // clamp every control to its range (current + delta)
    for (const n of Object.keys(ctlOut)) {
      if (!has(n)) { delete ctlOut[n]; continue; }
      const r0 = ranges[n] || [-1, 1.4], r = n === "blink" ? [-0.25, Math.max(1, r0[1])] : r0, cur = dw[n] || 0, lo = n === "smile" ? 0 : r[0], hi = n === "smile" ? Math.min(r[1], 1) : r[1];
      ctlOut[n] = clamp(cur + ctlOut[n], lo, hi) - cur;
    }
    const g = gaze.onUser ? { x: user.pos ? user.pos.x : 0, y: user.pos ? user.pos.y : 0 } : { x: gaze.x, y: gaze.y };
    inStep = false;
    return { yaw: yaw * DEG, pitch: pitch * DEG, roll: roll * DEG, zoom, controls: { ...ctlOut }, gaze: g, blinkNow,
             debug: { state: st.name, onUser: gaze.onUser, gazeKind: gaze.kind, yawDeg: yaw, pitchDeg: pitch, rollDeg: roll, breath: bv } };
  }

  const api = {
    on, step, setState: (s, opts) => setState(s, { manual: true, ...opts }), get state() { return st.name; }, get time() { return T; },
    set auto(v) { st.auto = !!v; }, get auto() { return st.auto; },
    think() { setState("thinking"); }, speak, yield: yieldTurn, holdFloor, cue, endReply,
    user: userEvent,                                                // {type: speechstart|pause|speechend|turnend|level, ...}
    setUser(p) { if (p.pos) user.pos = { x: +p.pos.x || 0, y: +p.pos.y || 0 }; if ("present" in p) { const was = user.present; user.present = p.present; if (p.present && was === false) { memo.visits++; notice(); } } if ("smile" in p) { if (p.smile > 0.5 && refractory("mirrorSmile", 6)) smile(0.12 + 0.12 * persona.expressiveness, 2.4, 0.6); } },
    setMood(m) { for (const k of ["valence", "arousal", "engagement"]) if (k in m) moodT[k] = clamp(+m[k], k === "valence" ? -1 : 0, 1); },
    get mood() { return { ...mood }; }, persona,
    glanceAt(x, y, dur = 0.8) { if (st.name === "speaking" || st.name === "listening" && user.voice) return false; if (!refractory("pointer", 4)) return false; gaze.override = { until: T + dur }; setGaze(false, dur, "pointer", { x, y, head: -clamp(x * 0.2, -3, 3) }); return true; },
    noticeUser: notice, signature,
    remember(m) { Object.assign(memo, m || {}); if (memo.visits > 1 && !memo.greeted) { memo.greeted = true; smile(0.3, 2.6, 0.5); brow(0.4, 0.9); } },
    get memory() { return { ...memo }; },
    get debug() { return { state: st.name, gaze: { ...gaze }, breath: breath.v, pulses: pulses.length, user: { ...user }, reply: { active: reply.active, end: reply.end, yielded: reply.yielded } }; }
  };
  return api;
}

function mergeDeep(a, b) {
  if (Array.isArray(b)) return b.slice();
  if (!b || typeof b !== "object") return b === undefined ? (a && typeof a === "object" ? mergeDeep({}, a) : a) : b;
  const o = Array.isArray(a) ? a.slice() : { ...(a || {}) };
  for (const k of Object.keys(b)) o[k] = a && typeof a[k] === "object" && a[k] !== null && !Array.isArray(a[k]) ? mergeDeep(a[k], b[k]) : (Array.isArray(b[k]) ? b[k].slice() : b[k]);
  for (const k of Object.keys(o)) if (o[k] && typeof o[k] === "object" && !Array.isArray(o[k]) && !(b && k in b)) o[k] = mergeDeep({}, o[k]);
  return o;
}

// ------------------------------------------------------------------ the viewer adapter
// createBehaviour(viewer, { persona, seed, random, config, clock, enabled }) -> the engine's API plus:
//   enable(on) / enabled, attachListener(src) (an ATTEND listener with on(), an AnalyserNode, or null), attachAttention(att), destroy(), mode
// With the layer API (viewer.addLayer) the engine's output is added each frame. Without it (an older viewer): a reduced stand-in that moves the
// pose and holds gaze / lid / brow controls from the frame event (no mouth deltas: they would override the speech driver).
export function createBehaviour(viewer, o = {}) {
  const ctlList = typeof viewer.controlNames === "function" ? viewer.controlNames() : (viewer.controls || []).map(c => c.name);
  const ranges = {}; for (const c of viewer.controls || []) ranges[c.name] = [c.min, c.max];
  const yawLim = () => { const L = viewer.limits; return L && L.yaw ? L.yaw[1] / DEG : 35; };
  const canShake = () => Math.abs((viewer.basePose ? viewer.basePose.yaw : 0) / DEG) + 4 <= Math.min(20, yawLim());
  const canWink = () => Math.abs((viewer.basePose ? viewer.basePose.yaw : 0) / DEG) <= 20;
  const eng = createEngine({ ...o, controls: ctlList, ranges, canShake, canWink, limits: () => viewer.limits || null });
  let enabled = o.enabled !== false, listener = null, offs = [], attention = null, savedPreset = null, last = null;
  const mode = typeof viewer.addLayer === "function" ? "layer" : "stub";
  const AC = { buf: null };
  function pollListener(dt) {
    if (!listener || listener.kind !== "analyser") return;
    const an = listener.node; if (!AC.buf || AC.buf.length !== an.fftSize) AC.buf = new Float32Array(an.fftSize || 2048);
    an.getFloatTimeDomainData(AC.buf); let e = 0; for (let i = 0; i < AC.buf.length; i++) e += AC.buf[i] * AC.buf[i];
    const dbv = 10 * Math.log10(e / AC.buf.length + 1e-12), t = eng.time;
    const sr = (an.context && an.context.sampleRate) || 48000;
    // the end-of-turn pitch (question?) from the last voiced frames
    if (listener.vad.speaking && dbv > -50) { const p = pitchMcLeod(AC.buf.subarray(AC.buf.length - Math.min(AC.buf.length, Math.round(sr * 0.04))), sr); if (p.clarity > 0.75) { listener.f0.push({ t, st: 12 * Math.log2(p.f0 / 100) }); if (listener.f0.length > 400) listener.f0.shift(); } }
    eng.user({ type: "level", level: listener.vad.level });
    for (const ev of listener.vad.feed(dbv, dt, t)) {
      if (ev.type === "turnend") ev.question = isQuestion(listener.f0, ev.t0);
      eng.user(ev);
    }
  }
  function isQuestion(f0, tEnd) {                                   // the last 0.4 s of voice rose by > 2 semitones over the turn median
    const tail = f0.filter(p => p.t > tEnd - 0.45 && p.t <= tEnd), all = f0.map(p => p.st); if (tail.length < 4 || all.length < 10) return false;
    const a = tail.slice(0, tail.length >> 1).map(p => p.st), b = tail.slice(tail.length >> 1).map(p => p.st);
    return median(b) - median(a) > 1.5 && median(b) > median(all) + 1.5;
  }
  let stubPrev = { yaw: 0, pitch: 0, zoom: 0 }, stubHeld = new Set();
  function layer(fr) {
    if (!enabled) return null;
    pollListener(fr.dt ?? 1 / 60);
    const dw = fr.driverWeights && !Array.isArray(fr.driverWeights) ? fr.driverWeights : arrToObj(fr.driverWeights);
    last = eng.step({ ...fr, driverWeights: dw });
    const out = { ...last }; delete out.debug; return out;
  }
  function arrToObj(a) { const o2 = {}; if (a) ctlList.forEach((n, i) => { o2[n] = a[i]; }); return o2; }
  // stub: no layer API. Pose offsets ride on the base pose; gaze -> gazeX / gazeY units (about 40 deg per unit; replaced by the layer API)
  let stubT = null;
  function stubFrame(info) {
    if (!enabled) return;
    const now = performance.now(), dt = stubT ? Math.min(0.25, (now - stubT) / 1000) : 1 / 60; stubT = now;
    const w = viewer.getWeights ? viewer.getWeights() : null;
    const outp = layer({ dt, speaking: !!info.speaking, driverWeights: w ? arrToObj(w) : {} });
    if (!outp) return;
    const bp = viewer.basePose; viewer.setPose({ yaw: bp.yaw - stubPrev.yaw + outp.yaw, pitch: bp.pitch - stubPrev.pitch + outp.pitch, zoom: bp.zoom - stubPrev.zoom + outp.zoom });
    stubPrev = { yaw: outp.yaw, pitch: outp.pitch, zoom: outp.zoom };
    const set = {}, rest = n => { const c = (viewer.controls || []).find(c2 => c2.name === n); return c ? c.rest || 0 : 0; };
    if (ctlList.includes("gazeX") && outp.gaze) { set.gazeX = rest("gazeX") + outp.gaze.x / 40; set.gazeY = rest("gazeY") + outp.gaze.y / 40; }
    for (const n of ["browUp", "furrow", "breath", "xBrowUp", "xFrown", "xEyesWide", "xSad", "xEyeSmile"]) if (outp.controls[n] !== undefined) set[n] = outp.controls[n];
    try { if (Object.keys(set).length) { viewer.setControls(set); for (const k of Object.keys(set)) stubHeld.add(k); } } catch (e) { /* unknown control */ }
  }
  function tunePreset(on) {
    if (!viewer.getPreset || !viewer.setPreset) return;
    if (on) {
      if (!savedPreset) savedPreset = viewer.getPreset();
      const p = viewer.getPreset();
      p.idle.sway.enabled = false;                                   // replaced by band-limited noise (3.1.9)
      p.idle.gaze.aversion.enabled = false;                          // aversions are scheduled by state
      if (p.speech && p.speech.beat) { p.speech.beat.nodDeg = 0; p.speech.beat.enabled = false; }   // beats and the phrase-end settle: replaced by prosody nods (3.1.4)
      if (p.idle.blink && "halfChance" in p.idle.blink) { p.idle.blink.halfChance = 0.12; p.idle.blink.gazeDipDeg = 1.5; }   // the viewer's blink quality (3.1.10), off in its defaults
      // the viewer's eye model would turn the head after large saccades (leadHead) and blink on them: this layer does both itself (bounded head
      // acceleration, gaze-shift blinks counted in the blink budget), so the viewer's are off while enabled (doubled, they gave a 4 deg head step)
      if (p.idle.eyes || "eyes" in p.idle) { p.idle.eyes = { ...(p.idle.eyes || {}), leadHead: { ...((p.idle.eyes || {}).leadHead || {}), enabled: false }, blinkOnSaccadeChance: 0 }; }
      viewer.setPreset(p);
    } else if (savedPreset) {
      const p = viewer.getPreset(), s = savedPreset;                 // restore only what was changed here; driver settings stay as the page has them now
      p.idle.sway.enabled = s.idle.sway.enabled; p.idle.gaze.aversion.enabled = s.idle.gaze.aversion.enabled; p.idle.blink.intervalS = s.idle.blink.intervalS; p.idle.blink.intervalSpeakingS = s.idle.blink.intervalSpeakingS;
      if (p.speech && p.speech.beat && s.speech && s.speech.beat) { p.speech.beat.nodDeg = s.speech.beat.nodDeg; p.speech.beat.enabled = s.speech.beat.enabled; }
      if (p.idle.blink && s.idle.blink && "halfChance" in s.idle.blink) { p.idle.blink.halfChance = s.idle.blink.halfChance; p.idle.blink.gazeDipDeg = s.idle.blink.gazeDipDeg; }
      if ("eyes" in s.idle || p.idle.eyes) p.idle.eyes = s.idle.eyes ? JSON.parse(JSON.stringify(s.idle.eyes)) : p.idle.eyes;
      viewer.setPreset(p); savedPreset = null;
    }
  }
  eng.on("blinkrate", e => {
    if (!enabled || !viewer.getPreset) return; const p = viewer.getPreset();
    if (e.state === "speaking") p.idle.blink.intervalSpeakingS = e.intervalS; else p.idle.blink.intervalS = e.intervalS;
    viewer.setPreset(p);
  });
  function attach() {
    if (mode === "layer") viewer.addLayer(layer);
    else offs.push(viewer.on("frame", stubFrame));
    for (const ev of ["speechstart", "speechend"]) if (viewer.on) { const off = viewer.on(ev, () => {}); if (typeof off === "function") offs.push(off); }
  }
  function detach() {
    if (mode === "layer" && viewer.removeLayer) viewer.removeLayer(layer);
    for (const f of offs.splice(0)) f();
    if (mode === "stub") { const bp = viewer.basePose; viewer.setPose({ yaw: bp.yaw - stubPrev.yaw, pitch: bp.pitch - stubPrev.pitch, zoom: bp.zoom - stubPrev.zoom }); stubPrev = { yaw: 0, pitch: 0, zoom: 0 }; const rel = {}; for (const k of stubHeld) rel[k] = null; try { viewer.setControls(rel); } catch (e) { /* */ } stubHeld.clear(); }
  }
  if (enabled) { tunePreset(true); attach(); }
  const api = Object.create(eng);
  Object.defineProperties(api, Object.getOwnPropertyDescriptors({
    mode, engine: eng,
    get enabled() { return enabled; },
    enable(on) { on = !!on; if (on === enabled) return enabled; enabled = on; if (on) { tunePreset(true); attach(); } else { detach(); tunePreset(false); } return enabled; },
    // the user's voice: an ATTEND listener ({on(ev, fn)}), an AnalyserNode (polled each frame by the own detector), or null
    attachListener(src) {
      if (listener && listener.offs) for (const f of listener.offs) try { f(); } catch (e) { /* */ }
      listener = null; if (!src) return;
      if (typeof src.getFloatTimeDomainData === "function") { listener = { kind: "analyser", node: src, vad: createVoiceDetector(o.vad || {}), f0: [] }; return; }
      if (typeof src.on === "function") {
        const offs2 = [], hook = (ev, fn) => { const r = src.on(ev, fn); if (typeof r === "function") offs2.push(r); };
        // ATTEND's times are its own audio clock: re-anchor them on the engine's clock (now minus how long ago)
        hook("level", x => eng.user({ type: "level", level: typeof x === "number" ? x : x && Number.isFinite(x.db) ? clamp((x.db - (x.floorDb ?? -70) - 6) / 30, 0, 1) : (x && x.level) || 0 }));
        hook("speechstart", () => eng.user({ type: "speechstart", t0: eng.time }));
        hook("speechend", () => eng.user({ type: "speechend", t0: eng.time - 0.12 }));
        hook("pause", x => { if (x && x.final) return; eng.user({ type: "pause", t0: eng.time - ((x && x.duration) || 0.25) }); });
        hook("turnend", x => eng.user({ type: "turnend", t0: eng.time - ((x && x.silence) || 0.9), turnLen: x && x.turnDuration }));
        listener = { kind: "attend", src, offs: offs2 };
      }
    },
    // ATTEND camera attention: face position (deg), smile, presence
    attachAttention(att) {
      if (attention && attention.offs) for (const f of attention.offs) try { f(); } catch (e) { /* */ }
      attention = null; if (!att) { eng.setUser({ pos: { x: 0, y: 0 } }); return; }
      const offs2 = [], hook = (ev, fn) => { const r = att.on(ev, fn); if (typeof r === "function") offs2.push(r); };
      hook("face", x => { if (x) eng.setUser({ pos: { x: x.x ?? (x.pos && x.pos.x) ?? 0, y: x.y ?? (x.pos && x.pos.y) ?? 0 } }); });
      hook("smile", x => eng.setUser({ smile: typeof x === "number" ? x : (x && (x.smile ?? x.value)) || 0 }));
      hook("present", () => eng.setUser({ present: true })); hook("absent", () => eng.setUser({ present: false }));
      attention = { att, offs: offs2 };
    },
    get last() { return last; },
    destroy() { api.attachListener(null); api.attachAttention(null); if (enabled) { detach(); tunePreset(false); } enabled = false; }
  }));
  return api;
}

export default { createBehaviour, createEngine, analyseClip, pitchMcLeod, parseTags, anchorCues, makePersona, createVoiceDetector, seededRandom, makeNoise, makeFollower, DEFAULT_PERSONA, DEFAULT_BEHAVIOUR, TAGS, STATES, BEHAVIOUR_VERSION };
