/*
 * OpenAvatar viewer: a dependency-free ES module (no imports, no build step) that renders a Gaussian-splat
 * avatar with WebGL 2 and drives it from audio.
 *
 * The renderer, the head-turn rig, the camera and the loudness / frequency-band mouth driver are ported
 * from the project owner's own prototype viewer. Changes from the prototype: page globals and URL
 * parameters became options, several viewers can live on one page, the audio-lookup-table path was
 * dropped, and idle motion, presets, a second ("timed") driver and a format-version check were added.
 * The mouth drivers were rebuilt on 2026-11-20 (lookahead, shape blends, letters -> phones); the first ones remain as "loudness-v1" / "timed-v1".
 *
 *   const v = await OpenAvatarViewer.create(canvas, { modelUrl, rig?, background?, pointer?, on?, renderer?: "cpu", sortWorker?: false, gpuPrepass?: false });
 *
 * Rendering (2026-11-20): by default the model is evaluated on the GPU (VS_GPU / FS_EVAL) and the CPU only sorts by depth, in a worker while the render loop runs;
 * the prototype's all-CPU renderer is kept (renderer: "cpu", and the fallback). Both draw the same picture: see docs/CONTRACT.md, "Rendering".
 *
 * Model file layout (little endian): u32 header length, JSON header, then the arrays listed in
 * header.fields. The header may carry `version` (number or "major.minor"); a missing field means 1. A file
 * whose major version is higher than FORMAT_MAJOR is refused (error code "version").
 */

export const FORMAT_MAJOR = 2;                                   // 2 = the compact format (spikes/lf_finish/FORMAT.md B); format-1 files read as before

// ------------------------------------------------------------------ small helpers
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const ss = x => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };
const rnd = (a, b) => a + (b - a) * Math.random();
const rndRange = r => rnd(r[0], r[1]);
const DEG = Math.PI / 180;
const PITCH_DOWN = -1;                                            // sign of a pitch that tilts the head down (chin towards the chest)
const isObj = o => o && typeof o === "object" && !Array.isArray(o);
function mergeDeep(base, add) {                                  // arrays and scalars are replaced, objects merged; returns a new object
  const out = Array.isArray(base) ? base.slice() : { ...base };
  for (const k of Object.keys(add || {})) out[k] = isObj(out[k]) && isObj(add[k]) ? mergeDeep(out[k], add[k]) : (isObj(add[k]) ? mergeDeep({}, add[k]) : Array.isArray(add[k]) ? add[k].slice() : add[k]);
  return out;
}
function fail(code, message) { const e = new Error(message); e.code = code; return e; }

// Head-follow curve (orbit rigs): how far the head turns on the neck toward the camera when the camera stands at orbit angle `o` (radians, same sign as o).
//   mode "ease" (default): rate * o * (1 - (o/end)^2)^2 inside +-end, 0 outside. The head always turns the same way as the body, only slower near the front:
//     the face's angle to the camera (o minus this) changes at 1 - rate .. 1 + 0.8 * rate times the orbit speed, continuously, and the curve meets 0 at `end`
//     with zero slope. The rate is lowered when the peak (0.2862 * rate * end, at 0.447 * end) would pass `max`.
//   mode "lock" (the first version): min(|o|, max), faded out between fade[0] and fade[1]: the face stays locked on the camera while the body turns, then
//     turns away at up to twice the orbit speed. Kept as an option; it reads as a head that is not attached.
export function followCurve(o, c) {
  const max = c.max, end = c.fade[1];
  if (c.mode === "lock") return clamp(o, -max, max) * (1 - ss((Math.abs(o) - c.fade[0]) / (end - c.fade[0])));
  const u = o / end; if (!(Math.abs(u) < 1)) return 0;
  const w = 1 - u * u; return Math.min(c.rate, max / (0.28622 * end)) * o * w * w;
}

// ------------------------------------------------------------------ preset (plain JSON)
export const DEFAULT_PRESET = {
  version: 1,
  driver: {
    name: "loudness",                      // "loudness" | "timed" | "loudness-v1" | "timed-v1" | any registered driver
    gain: 1,                               // loudness driver: overall mouth opening (prototype "gain")
    smoothing: 0.05,                       // loudness driver: seconds; the centred smoothing window is 1.6 times this to each side (0.045 until lipsync4: the mouth flapped; closures keep their own narrow window. More than ~0.05 opens a phrase > 100 ms before its sound)
    expand: 0.7,                           // loudness driver: opening curve (exponent below 1 lifts small openings past the range where the lips do not part)
    lead: 0.02,                            // seconds the mouth runs ahead of the sound (both drivers)
    lookahead: 0.12,                       // seconds a media element's sound is delayed so that the loudness driver can see it coming (0 = no delay line)
    maxPerControl: {},                     // e.g. {"ah": 0.7}; missing = 1
    clampToTrained: true,                  // keep speech weights inside the range seen in training (prototype behaviour)
    dipBridge: 0.035,                      // loudness driver: seconds; a dip of the level shorter than this (a short stop inside a vowel: "I", "avatar") is bridged before the narrow window, so the mouth does not snap shut for one frame. Pauses and the lip closures (driver.lips) are not affected. 0 = off (spikes/snaps)
    openGain: 1,                           // both drivers, OPT-IN (spikes/mouthgain): the five opening shapes x this, on top of articulation, AND the opening-sum limit raised to match (1 = as before). A number, or per control {wide, ah, ee, oo, mid} (missing = 1). Closures are not affected. On v20 1.2 opens wide vowels ~15 % further and the interior smear grows with it; 1.35 and up add dark specks at the corners
    openSum: 0,                            // the opening-sum limit with openGain; 0 = auto (max(1, the largest gain)); never below 1
    articulation: 1.1,                     // both drivers: how far the mouth opens on vowels (opening shapes x this; above a sum of 1 the jaw takes the room). 1.1 = default, 0.8 about the earlier look, 1.3 emphatic. Closures are not affected
    timed: { attack: 0.05, release: 0.10, timeOffset: 0, minVisible: 0.07, tongueMin: 0.05,   // attack / release scale the events' own ramps (0.05 / 0.10 = as designed); seconds
             stressOpen: 1.06, unstressedOpen: 0.75 },  // (stressOpen 1.12 until lipsync4) vowels with a known lexical stress (pronouncing dictionary, or ARPAbet phones with stress digits): primary stress opens this much more, unstressed this much less
    lips: { enabled: true, hum: true, stop: true,  // loudness driver: lip closures from the sound itself (m hum, p / b stop; spikes/loudlips, see LIP CLOSURES FROM THE SOUND). Seconds and dB
            attack: 0.035, release: 0.03, mergeS: 0.06, mergeDb: 10, lead: 0,   // ramps outside each closure; two closures closer than mergeS, with no vowel (0.5-3 kHz within mergeDb of its max) between, stay shut between; lead: how far the closures run ahead of the sound (not driver.lead)
            humDb: 10, humMidDrop: 6, humLoDrop: 7, humDrop: 8, humMin: 0.025, humF2Rise: 1.15,   // hum (humF2Rise: not when the 0.9-2.8 kHz centre rose by this factor over the 35 ms before it, as into an ng): < 400 Hz above 0.5-3 / > 3 kHz by humDb; 0.5-3 kHz humMidDrop under its 0.3 s max and humDrop under the 60 ms before; < 400 Hz within humLoDrop of its max
            releaseS: 0.035, releaseDb: 4, releaseToVowel: 16, hissDb: 3,   // ...and it must end in a release: 0.5-3 kHz up releaseDb, to within releaseToVowel of the vowel, within releaseS, before a silence or a hiss (> 3 kHz above 0.5-3 kHz by hissDb)
            humNoLook: true, humNoLookDb: 4, humMinNoLook: 0.04, humMaxNoLook: 0.08,   // no lookahead (a microphone): a hum humNoLookDb clearer than humDb, humMinNoLook long, closes without its release, for at most humMaxNoLook
            place: true, placeDb: 2,   // (spikes/nvsm) n is not m: no hum closure when the murmur's median n-zero index (1.65-2.1 kHz minus 1.2-1.5 kHz, dB) is >= placeDb (n ~ +4..+17, m ~ -11..+0.5)
            humLoFall: 3,   // (spikes/snaps) not a lip hum when the < 400 Hz band's median over the hum is more than humLoFall dB under its max in the 60 ms before (a v or w: "every", "what"; m -1.5 dB at worst, v / w -4.1 and -5.4 on the test clips). 0 = off
            hissEntry: true, hissEntryS: 0.03, hissHumDb: 10, hissLoDrop: 14,   // (nvsm) an m right after a hiss ("sends me") counts as entered even without a mid fall: a hiss within hissEntryS before it; its build-up (< 400 Hz above the rest by humDb + hissHumDb, within hissLoDrop of its max) belongs to the closure
            minOpen: 0.05, humPend: 0.005, burstOpen: 0.015,   // (nvsm) after a closure opens, the vowel's opening does not drop back for minOpen s; no lookahead: while an m-like hum is under way (>= humPend s) the opening does not grow; with lookahead a p / b's closure starts to open burstOpen s before its burst (a microphone's comes late: it keeps its whole closure)
            silDb: 25, stopMin: 0.02, stopMax: 0.2, burstDb: 20, burstHiMid: -4, burstJump: 9, burstLoMid: -10 }   // stop: a silence silDb under the 0.3 s max, stopMin..stopMax long, then a burst burstDb up, not hissy (> 3 kHz minus 0.5-3 kHz <= burstHiMid), sharp (> 3 kHz up burstJump in 5 ms), weak in 0.5-3 kHz (< 400 Hz minus 0.5-3 kHz >= burstLoMid)
  },
  idle: {
    enabled: true,
    gaze: { enabled: true, mode: "saccade",          // "saccade" (design default) or "sine" (the prototype's slow drift)
            amount: 0.06,                            // control units around the rest value (prototype: 0.06)
            intervalS: [0.6, 2], shiftS: [0.08, 0.15],
            aversion: { enabled: true, amount: 0.15, everySpeakingS: [4, 8], everyListeningS: [8, 15], durationS: [1, 3] } },
    sway: { enabled: true, yawDeg: 3, tiltDeg: 1.2, periodS: [3, 6], limitFraction: 0.8 },   // fraction of the rig's limit that sway may use
    blink: { enabled: true,                          // only for a model with a "blink" control (spikes/blink). Seconds; [lo, hi] = drawn uniformly per blink
             intervalS: [2, 6], intervalSpeakingS: [2, 4.5],   // start to start; the speaking one with speech.blinkFaster
             closeS: [0.1, 0.15], holdS: 0.02, openS: [0.15, 0.25],
             doubleChance: 0.12, doubleGapS: [0.06, 0.12],    // now and then a second blink right after the first
             gazeGuardS: 0.05,                               // no blink starts during a gaze jump (nor this long after it), no gaze jump starts during a blink
             halfChance: 0, halfDepth: [0.5, 0.65],          // (lifelike, off) share of automatic blinks that close only to halfDepth
             gazeDipDeg: 0, browDip: 0 },                    // (lifelike, off) the eyes dip this many degrees at a full blink (try 1.5); browUp drops by browDip control units (a model with browUp)
    eyes: { enabled: false,                          // (lifelike) the eye model below for the idle gaze too; a gaze target (setGazeTarget, a layer) always uses it. Degrees via GAZE_CAL
            hold: 1, holdBase: 0,                    // share of the head's motion (sway, nods, layers) / of the base turn (setPose, drag) the eyes cancel to stay on their target
            saccadeMs: [25, 2.4], pursuitS: 0.12, saccadeMinDeg: 0.6,   // saccade = 25 ms + 2.4 ms per degree (minimum jerk); smaller target moves are followed smoothly
            micro: { enabled: true, perS: [1, 2], maxDeg: 0.4 },        // microsaccades while fixating
            leadHead: { enabled: true, minDeg: 15, share: 0.4, delayS: 0.03, timeS: 0.35 },   // a shift of minDeg or more: the head follows share of it, delayS later
            blinkOnSaccadeDeg: 10, blinkOnSaccadeChance: 0.5 }   // a saccade this large blinks with this chance
  },
  speech: { swayWhileSpeaking: true, emphasisDeg: 1, blinkFaster: true, aversionsWhileSpeaking: true,
            beat: { enabled: true, nodDeg: 0.9, settleDeg: 0.6 } },   // head co-motion from the speech: a small nod on phrase starts and stressed vowels, a slight settle at phrase ends (needs idle motion on)
  // (spikes/lf_motion) the temporal layer: no steps or kinks in what is shown. enabled: false = the motion before 2026-11-26, frame for frame.
  // Seconds are time constants T of critically damped followers (x'' = (u - x) / T^2 - 2 x' / T): no overshoot, a ramp lags by 2 T, a step in the
  // input's velocity or acceleration comes out continuous.
  motion: { enabled: true,
            poseS: 0.035,                          // layers' head offsets (yaw, pitch, roll, zoom): a bounded-acceleration or exponential input gets a continuous acceleration
            exprS: 0.045,                          // layers' expression deltas (smile, x controls, brows, lids, breath): no one-frame steps, rounded corners at caps
            mouthExprS: [0.05, 0.012],             // ...the expression controls that open the mouth (mouthExpr), rising / falling: they leave a closure within ~25 ms
            mouthExpr: ["xSmileOpen", "xSmileAh"],
            closureGate: true,                     // the mouth-opening expressions fade with the lips' press (1 - smoothstep(press / 0.5)): no teeth through a closure, whatever a layer sends
            blinkGate: ["xEyesWide"],              // layer deltas that fight a closing lid fade with the blink (1 - smoothstep(blink / 0.8)): wide eyes + a blink smeared the shut lid (sheets/combos_eyes.jpg)
            raw: [],                               // more control names a layer sets unfiltered (the speech shapes but smile, gazeX / gazeY and press are always unfiltered)
            softLimit: true,                       // a layer's control delta eases into the control's range (as the pose does) instead of a flat-topped clamp
            idleSaccade: true,                     // classic idle gaze: saccades on the main sequence (25 ms + 2.4 ms / deg, minimum jerk) instead of 80-150 ms smoothsteps
            swaySpline: true,                      // classic idle sway: a C1 spline through the random targets (no stop at each one)
            beatEaseIn: true,                      // the speech beat eases in when a clip starts mid-beat (it stepped by up to 0.5 deg)
            pursuitOrder2: true,                   // eye model: smooth pursuit as a critically damped follower (pursuitS / 2) instead of a first-order lag (velocity steps)
            pointerS: 0.03 }                       // enablePointer: drags ease the pose (setPose(p, { easeS })); 0 = set at once
};

// ------------------------------------------------------------------ render options (spikes/lf_render; ROADMAP 3.4). All off by default except `detail` (lf_postfx2):
// with { detail: false } a frame is drawn exactly as before (same shaders' arithmetic, clear + splats). create(canvas, { render: {...} }), v.setRender({...}) or setPreset({ render: {...} }) turn them on.
export const RENDER_DEFAULTS = {
  supersample: 1,          // desktop AA: the canvas gets supersample x the device pixel ratio (the browser scales it down); 1 = off, 1.5 suggested on desktop GPUs
  maxDpr: 3,               // cap on device pixel ratio x supersample
  aaCompensation: "auto",  // gsplat "antialiased" mode: opacity x sqrt(det(S) / det(S + 0.3 I)). "auto" = only when the model header says antialiased: true (TRAIN 3.3.5); true / false force
  sort: "z",               // depth order key: "z" (view depth, as before) | "radial" (distance to the camera)
  sortEvery: 1,            // GPU renderer, loop frames: sort at most every Nth frame and draw the others at once with the last order (phones: 2); 1 = every frame
  sortEveryExact: false,   // tests: apply sortEvery to renderOnce frames too
  shClampDeg: 0,           // SH colour seen from at most +-this head yaw (the stills' arc, 35): beyond it each Gaussian keeps the colour it has at the arc's end. 0 = off
  shDegree: 2,             // view-dependent colour terms: 2 = all (as trained), 1 = phones (skip the 5 degree-2 terms), 0 = base colour only
  quadExtent: 4,           // CREATE-TIME: splat cut-off, |q|^2 in the shader (4 = as before; 3 = smaller quads, less fill: mobile)
  splatScale: 1,           // splats drawn into a target this fraction of the canvas and scaled up (0.5 = half resolution: mobile fill rate); 1 = off
  catchlights: null,       // eye glints: true / {} = on with CATCHLIGHT_DEFAULTS, or an object overriding them
  relight: null,           // reserved (ROADMAP 4.5: not adopted, see spikes/lf_render/LOG.md)
  post: null,              // portrait finish: "portrait" (= PORTRAIT_POST) or an object { vignette, grain, plate, shadow }
  detail: true             // screen-space detail pass (spikes/lf_postfx2), ON by default: true / {} = DETAIL_DEFAULTS, or an object overriding them; null / false = off (the old look)
};
// The detail pass: the splats are drawn into a texture, then ONE pass resolves it to the canvas with FidelityFX CAS sharpening (MIT) and a fine mid-tone luma grain.
// With supersample > 1 the texture is the supersampled one and the canvas stays at the device pixel ratio, so the sharpening is applied after the downscale
// (the browser's own downscale would blur it away). With splatScale < 1 (phones) the half-size texture is first scaled up with FSR1 EASU (MIT) instead of bilinear.
export const DETAIL_DEFAULTS = {
  sharpen: 0.35,           // CAS sharpness 0..1 (0 = no sharpening; 1 = strongest). It crisps lid lines, iris rims, brows and lip edges; it cannot create lashes
  grain: 0.015,            // fine luma grain, strongest in the mid-tones (4 L (1 - L)), animated per drawn frame; hides the splats' brush-stroke texture. 0 = none
  upscale: "fsr1"          // splatScale < 1: "fsr1" (EASU edge-adaptive) | "bilinear" (as before)
};
export const CATCHLIGHT_DEFAULTS = {
  radius: 0.008,           // m: the glint sits on a sphere this big (the cornea's curvature, 7.8 mm; 0.012 = the eyeball: glints run off the iris at +-35) at centre + radius * normalize(to camera + to light)
  eyes: null,              // [[x, y, z], [x, y, z]] sphere centres (model metres) or null = from the header's blinkSweep.eyes + the cornea surface in front of them
  // lights, camera-relative: az (deg, + = viewer's right), el (deg, + = up), size (m, sigma on the cornea), intensity (added light, 0..1), color
  lights: [{ az: -12, el: 14, size: 0.0006, intensity: 0.55, color: [1, 0.98, 0.95] },
           { az: 16, el: -6, size: 0.00035, intensity: 0.16, color: [0.93, 0.96, 1] }],
  minPx: 0.6,              // smallest glint sigma in canvas pixels
  halo: 0.22,              // a faint wide halo (x 3 the size) around each glint: reads as wet rather than a dot
  fadeDeg: [38, 50]        // the glint fades out between these angles from the eye's forward axis (it leaves the cornea; turned away from the light it goes)
};
export const PORTRAIT_POST = {
  vignette: 0.28,          // darkening in the corners (0..1)
  grain: 0.03,             // fine animated grain amplitude (relative)
  plate: { lift: 0.10, falloff: 0.9, tint: [1.0, 0.97, 0.93], centre: [0.5, 0.38] },   // soft lighter area behind the head instead of a flat colour (lift = brightness added on dark, darkening on light backgrounds)
  shadow: { amount: 0.35, width: 1.25, height: 0.22 }                                // soft contact shadow behind the torso fade (sizes x the bust's width)
};
export const RENDER_PROFILES = {                                // starting points for a page; nothing applies them by itself
  desktop: { supersample: 1.5 },
  phone: { sortEvery: 2, shDegree: 1, quadExtent: 3, splatScale: 0.5 },
  portrait: { catchlights: true, post: "portrait" }
};
const NO_XO = { shc: null, deg: 2, radial: false };
function readRender(base, add) {                                 // merge, then expand the shorthands (true / "portrait")
  const a = add ? { ...add } : {};
  if (a.catchlights === true) a.catchlights = {}; if (a.detail === true) a.detail = {}; if (a.detail === false) a.detail = null; if (a.post === "portrait") a.post = PORTRAIT_POST; if (a.post === true) a.post = PORTRAIT_POST;
  const r = mergeDeep(base, a);
  if (r.catchlights) r.catchlights = mergeDeep(CATCHLIGHT_DEFAULTS, r.catchlights);
  if (r.post) r.post = mergeDeep({ vignette: 0, grain: 0, plate: null, shadow: null }, r.post);
  r.detail = readDetail(r.detail);
  return r;
}
export function readDetail(d) {                                   // render.detail -> null (off) or the full options (pure, tested): true / {} = DETAIL_DEFAULTS
  if (d === true) d = {}; if (!d || typeof d !== "object") return null; const r = mergeDeep(DETAIL_DEFAULTS, d);
  r.sharpen = Math.min(1, Math.max(0, +r.sharpen || 0)); r.grain = Math.max(0, +r.grain || 0); if (r.upscale !== "bilinear") r.upscale = "fsr1"; return r;
}
// The pupils (catchlights): per blinkSweep eye, the opacity-weighted centre of the dark (base colour luminance < 0.18), opaque Gaussians within 8 mm that move
// with gazeX; null for an eye with too few. A few ms over the model once at load (front models with a blink sweep only).
function findPupils(D, BS, alpha0, gk, Km) {
  if (gk < 0 || !D.dc || !D.mot) return null; const m = D.mean, mo = D.mot, dc = D.dc, n = D.n;
  return BS.eyes.map(([cx0, top, bot]) => { const cy0 = 0.5 * (top + bot); let sx = 0, sy = 0, sw = 0;
    for (let i = 0; i < n; i++) { const x = m[i * 3], y = m[i * 3 + 1]; if (Math.abs(x - cx0) > 0.008 || Math.abs(y - cy0) > 0.008 || alpha0[i] < 0.3) continue;
      const o = (i * Km + gk) * 3; if (Math.hypot(mo[o], mo[o + 1], mo[o + 2]) < 0.0005) continue;
      const l = 0.299 * (C0 * dc[i * 3] + 0.5) + 0.587 * (C0 * dc[i * 3 + 1] + 0.5) + 0.114 * (C0 * dc[i * 3 + 2] + 0.5); if (l >= 0.18) continue;
      sx += alpha0[i] * x; sy += alpha0[i] * y; sw += alpha0[i]; }
    return sw > 3 ? [sx / sw, sy / sw] : null; });
}
// Where a catchlight sits (pure, tested): eye sphere centre c, camera position cam, unit direction to the light l (all world), radius r -> c + r * normalize(normalize(cam - c) + l)
function catchlightPoint(c, cam, l, r) {
  let vx = cam[0] - c[0], vy = cam[1] - c[1], vz = cam[2] - c[2]; const iv = 1 / (Math.hypot(vx, vy, vz) || 1); vx = vx * iv + l[0]; vy = vy * iv + l[1]; vz = vz * iv + l[2];
  const ih = 1 / (Math.hypot(vx, vy, vz) || 1); return [c[0] + r * vx * ih, c[1] + r * vy * ih, c[2] + r * vz * ih, vx * ih, vy * ih, vz * ih];   // point, then the unit half vector
}

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
  if (major === 2) { try { return decodeCompact(buf, hdr, hl, ver); } catch (e) { throw e.code ? e : fail("format", "Compact model file could not be read: " + e.message); } }
  const f16 = halfTable();
  let off = 4 + hl; const d = { hdr, n: hdr.n, Km: hdr.Km, Kc: hdr.Kc, Ko: hdr.Ko, frames: hdr.frames, fps: hdr.fps, version: ver, h16: {} };
  for (const [name, type, shape] of hdr.fields) {
    const cnt = hdr.n * shape.reduce((a, b) => a * b, 1), bytes = cnt * (type === "f4" ? 4 : 2);
    if (off + bytes > buf.length) throw fail("format", "Model file is truncated (field " + name + ").");
    const raw = buf.slice(off, off + bytes).buffer; off += bytes;
    if (type === "f4") d[name] = new Float32Array(raw);
    else { const u = new Uint16Array(raw), f = new Float32Array(cnt); for (let i = 0; i < cnt; i++) f[i] = f16[u[i]]; d[name] = f; d.h16[name] = u; }
  }
  return d;
}
// Compact format 2.0 (spikes/lf_finish/FORMAT.md section B; FINISH's reference decoder): fields in order, each blob padded to 4 bytes; encodings f4, f2 (float16;
// kept as h16 for the GPU upload), f2s (float16 x 1/scale), q16 (16-bit fixed point between lo and hi per column), cells (per control a row bit mask, then
// float16 values for the set rows only). Returns exactly what parseModel returns for the format-1 file.
function decodeCompact(buf, hdr, hl, ver) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength), n = hdr.n, f16 = halfTable();
  const d = { hdr, n, Km: hdr.Km, Kc: hdr.Kc, Ko: hdr.Ko, frames: hdr.frames, fps: hdr.fps, version: ver, h16: {} };
  let off = 4 + hl; const align = () => { off += (4 - (off % 4)) % 4; }, need = b => { if (off + b > buf.length) throw fail("format", "Model file is truncated."); };
  const u16 = cnt => { need(2 * cnt); const u = new Uint16Array(cnt); for (let i = 0; i < cnt; i++) u[i] = dv.getUint16(off + 2 * i, true); off += 2 * cnt; return u; };
  for (const [name, enc, shape, p = {}] of hdr.fields) {
    const per = shape.reduce((a, b) => a * b, 1), cnt = n * per;
    if (enc === "f4") { need(4 * cnt); const a = new Float32Array(cnt); for (let i = 0; i < cnt; i++) a[i] = dv.getFloat32(off + 4 * i, true); off += 4 * cnt; d[name] = a; }
    else if (enc === "f2" || enc === "f2s") { const u = u16(cnt), a = new Float32Array(cnt), sc = enc === "f2s" ? 1 / p.scale : 1; for (let i = 0; i < cnt; i++) a[i] = f16[u[i]] * sc; d[name] = a; if (enc === "f2") d.h16[name] = u; }
    else if (enc === "q16") { const u = u16(cnt), a = new Float32Array(cnt); for (let i = 0; i < n; i++) for (let c = 0; c < per; c++) a[i * per + c] = p.lo[c] + u[i * per + c] / 65535 * (p.hi[c] - p.lo[c]); d[name] = a; }
    else if (enc === "cells") {
      const K = shape[0], C = per / K, mb = Math.ceil(n / 8), mbp = mb + ((4 - (mb % 4)) % 4), masks = [];
      for (let k = 0; k < K; k++) { need(mbp); masks.push(buf.subarray(off, off + mb)); off += mbp; }
      const u = new Uint16Array(cnt);
      for (let k = 0; k < K; k++) { const m = masks[k]; need(2 * C * (p.counts ? p.counts[k] : 0)); let c = 0;
        for (let i = 0; i < n; i++) if (m[i >> 3] & (1 << (i & 7))) { for (let j = 0; j < C; j++) u[(i * K + k) * C + j] = dv.getUint16(off + 2 * (c * C + j), true); c++; }
        if (!p.counts || c !== p.counts[k]) throw fail("format", `Compact model: field ${name} control ${k} row count does not match the header.`); off += 2 * c * C; }
      const a = new Float32Array(cnt); for (let i = 0; i < cnt; i++) a[i] = f16[u[i]]; d[name] = a; d.h16[name] = u;
    } else throw fail("format", "Compact model: unknown field encoding " + enc);
    align();
  }
  if (off !== buf.length) throw fail("format", "Compact model: " + (buf.length - off) + " bytes after the last field.");
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

// ------------------------------------------------------------------ GL shaders (VS / FS unchanged from the prototype; VS is the CPU renderer's)
const VS = `#version 300 es
precision highp float;
layout(location=0) in vec2 aQuad; layout(location=1) in vec3 aCenter; layout(location=2) in vec3 aC0; layout(location=3) in vec3 aC1; layout(location=4) in vec4 aColor;
uniform mat4 uView; uniform vec2 uFocal; uniform vec2 uViewport; uniform vec2 uPP; uniform int uAA;
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
  if (uAA == 1) vColor.a *= sqrt(max((a - 0.3) * (c - 0.3) - b * b, 0.0) / max(a * c - b * b, 1e-12));   // render: gsplat antialiased mode (models trained that way)
}`;
const FS = `#version 300 es
precision highp float; in vec4 vColor; in vec2 vPos; out vec4 o;
void main() { float A = -dot(vPos, vPos); if (A < -4.0) discard; o = vec4(vColor.rgb, vColor.a * exp(A)); }`;

// ------------------------------------------------------------------ GPU evaluation (default renderer since 2026-11-20)
// The same per-Gaussian model as evaluate() below, run in the vertex shader from textures that are uploaded once: position = mean + sum of
// weight x motion coefficient, the head turn, opacity, the colour's control terms and its view-dependent (SH degree 2) part, the head-turned
// covariance. Colour and opacity are cut to 8 bits exactly as the CPU path stores them. Per frame the CPU only works out the depth order
// (sortSplats: positions and depths, no colour) and uploads 4 bytes per Gaussian; in the render loop that runs in a worker.
// By default this runs as a pre-pass that evaluates each Gaussian once into two float textures (EVAL), and the splats read those (SPLAT); without
// float render targets every splat vertex evaluates its Gaussian (VS_GPU). Textures, 2048 texels wide; texel j of item i sits at IX(j, i): planar
// (j * count + i) for the pre-pass, which reads the Gaussians in index order (3x faster there), interleaved (i * J + j) for VS_GPU, which reads them in depth order: tGeo 4 planes [mean, head weight] [cov 0-3] [cov 4-5, opacity, look row] [dc, motion row]; tSH 6
// planes (the 24 SH coefficients); tMot Km planes [motion coefficient] over the Gaussians that have any; tLook max(Kc, Ko) planes [colour coefficient,
// opacity coefficient] over the Gaussians whose colour or opacity depends on a control.
const GPU_EVAL = `
uniform mat4 uView; uniform vec2 uFocal; uniform vec2 uViewport; uniform vec2 uPP;
uniform sampler2D tGeo, tSH, tMot, tLook;
uniform int uKm, uKl, uHead, uHwField, uNg, uNm, uNl; uniform vec4 uWm[16], uWc[16], uWo[16];
uniform mat3 uHQ; uniform vec3 uPiv, uZf, uCam;
uniform int uBk; uniform float uBw, uBs, uBm; uniform vec3 uBe[2];
uniform int uAA, uShDeg; uniform vec4 uShC;   // render options (RENDER_DEFAULTS): antialiased opacity, SH degree, SH clamp (xyz = camera in the head frame at the clamped yaw, w = 1 on)   // blink sweep (header blinkSweep): control index (-1 = none), its weight, soft, motion share, eyes [cx, top, bottom]
ivec2 tx(int t) { return ivec2(t & 2047, t >> 11); }
// Gaussian i on screen: centre (clip units), major and minor axis (pixels), 8-bit colour. false = not drawn (faint or behind the camera)
bool evalSplat(int i, out vec2 ctr, out vec2 maj, out vec2 mnr, out vec4 col) {
  vec4 g0 = texelFetch(tGeo, tx(IX(0, i, uNg, 4)), 0), g1 = texelFetch(tGeo, tx(IX(1, i, uNg, 4)), 0), g2 = texelFetch(tGeo, tx(IX(2, i, uNg, 4)), 0), g3 = texelFetch(tGeo, tx(IX(3, i, uNg, 4)), 0);
  vec3 p = g0.xyz, p0 = g0.xyz; float hw = g0.w; int rl = int(g2.w), rm = int(g3.w);
  if (rm >= 0) { for (int j = 0; j < 16; j++) { if (j * 4 >= uKm) break; vec4 w = uWm[j]; int k = j * 4;   // weights past Km are 0
    if (w.x != 0.0) p += w.x * texelFetch(tMot, tx(IX(k, rm, uNm, uKm)), 0).xyz; if (w.y != 0.0) p += w.y * texelFetch(tMot, tx(IX(k + 1, rm, uNm, uKm)), 0).xyz;
    if (w.z != 0.0) p += w.z * texelFetch(tMot, tx(IX(k + 2, rm, uNm, uKm)), 0).xyz; if (w.w != 0.0) p += w.w * texelFetch(tMot, tx(IX(k + 3, rm, uNm, uKm)), 0).xyz; } }
  vec3 cl = g3.xyz; float ov = g2.z;
  if (rl >= 0) { for (int j = 0; j < 16; j++) { if (j * 4 >= uKl) break; vec4 wc = uWc[j], wo = uWo[j]; int k = j * 4; vec4 c;
    if (wc.x != 0.0 || wo.x != 0.0) { c = texelFetch(tLook, tx(IX(k, rl, uNl, uKl)), 0); cl += wc.x * c.rgb; ov += wo.x * c.a; }
    if (wc.y != 0.0 || wo.y != 0.0) { c = texelFetch(tLook, tx(IX(k + 1, rl, uNl, uKl)), 0); cl += wc.y * c.rgb; ov += wo.y * c.a; }
    if (wc.z != 0.0 || wo.z != 0.0) { c = texelFetch(tLook, tx(IX(k + 2, rl, uNl, uKl)), 0); cl += wc.z * c.rgb; ov += wo.z * c.a; }
    if (wc.w != 0.0 || wo.w != 0.0) { c = texelFetch(tLook, tx(IX(k + 3, rl, uNl, uKl)), 0); cl += wc.w * c.rgb; ov += wo.w * c.a; } } }
  if (uBk >= 0 && uBw > 0.0 && uBw < 1.0) {   // blink sweep: this Gaussian's own blink weight from its height in the eye (see blinkSweepWeight); the linear terms above used uBw
    vec3 mb = rm >= 0 ? texelFetch(tMot, tx(IX(uBk, rm, uNm, uKm)), 0).xyz : vec3(0.0); int e = abs(p0.x - uBe[0].x) <= abs(p0.x - uBe[1].x) ? 0 : 1;
    float h = clamp((uBe[e].y - (p0.y + uBm * mb.y)) / (uBe[e].y - uBe[e].z), 0.0, 1.0), u = clamp(((1.0 + uBs) * uBw - h) / uBs, 0.0, 1.0), dl = u * u * (3.0 - 2.0 * u) - uBw;
    p += dl * mb; if (rl >= 0 && uBk < uKl) { vec4 c = texelFetch(tLook, tx(IX(uBk, rl, uNl, uKl)), 0); cl += dl * c.rgb; ov += dl * c.a; } }
  float a = 1.0 / (1.0 + exp(-ov));
  bool hd = uHead == 1 && hw > 0.0; vec3 ph = p;
  if (hd) p += hw * (uPiv + uHQ * (p - uPiv) - p + uZf);
  vec4 cam = uView * vec4(p, 1.0);
  ctr = vec2(0.0); maj = vec2(0.0); mnr = vec2(0.0); col = vec4(0.0);
  if (a < 0.004 || cam.z < 0.2) return false;
  mat3 V = mat3(g1.x, g1.y, g1.z, g1.y, g1.w, g2.x, g1.z, g2.x, g2.y);
  if (uHead == 1 && hw >= 0.5) V = uHQ * V * transpose(uHQ);
  vec3 d = p - uCam; d *= 1.0 / (length(d) + 1e-9);
  if (hd) { vec3 e = uShC.w > 0.5 ? normalize(ph - uShC.xyz) : transpose(uHQ) * d;
    if (uHwField == 1) { if (hw < 1.0) { e = d + hw * (e - d); e *= 1.0 / (length(e) + 1e-9); } d = e; } else if (hw >= 0.5) d = e; }
  vec4 s0 = texelFetch(tSH, tx(IX(0, i, uNg, 6)), 0), s1 = texelFetch(tSH, tx(IX(1, i, uNg, 6)), 0), s2 = texelFetch(tSH, tx(IX(2, i, uNg, 6)), 0), s3 = texelFetch(tSH, tx(IX(3, i, uNg, 6)), 0), s4 = texelFetch(tSH, tx(IX(4, i, uNg, 6)), 0), s5 = texelFetch(tSH, tx(IX(5, i, uNg, 6)), 0);
  float xx = d.x * d.x, yy = d.y * d.y, zz = d.z * d.z;
  vec3 c = 0.28209479177387814 * cl + 0.5;
  if (uShDeg >= 1) { c += -0.4886025119029199 * d.y * s0.xyz; c += 0.4886025119029199 * d.z * vec3(s0.w, s1.xy); c += -0.4886025119029199 * d.x * vec3(s1.zw, s2.x); }
  if (uShDeg >= 2) { c += 1.0925484305920792 * d.x * d.y * s2.yzw; c += -1.0925484305920792 * d.y * d.z * s3.xyz; c += 0.31539156525252005 * (2.0 * zz - xx - yy) * vec3(s3.w, s4.xy);
  c += -1.0925484305920792 * d.x * d.z * vec3(s4.zw, s5.x); c += 0.5462742152960396 * (xx - yy) * s5.yzw; }
  col = vec4(clamp(floor(c * 255.0), 0.0, 255.0), floor(a * 255.0));   // 0..255, cut as the CPU renderer stores them
  mat3 W = mat3(uView);
  mat3 S = W * V * transpose(W);
  float iz = 1.0 / cam.z;
  vec3 j0 = vec3(uFocal.x * iz, 0.0, -uFocal.x * cam.x * iz * iz), j1 = vec3(0.0, uFocal.y * iz, -uFocal.y * cam.y * iz * iz);
  float A = dot(j0, S * j0) + 0.3, B = dot(j0, S * j1), C = dot(j1, S * j1) + 0.3;
  if (uAA == 1) col.a = floor(col.a * sqrt(max((A - 0.3) * (C - 0.3) - B * B, 0.0) / max(A * C - B * B, 1e-12)));   // gsplat antialiased mode (only for models trained that way)
  float mid = 0.5 * (A + C), rad = length(vec2(0.5 * (A - C), B));
  float l1 = mid + rad, l2 = max(mid - rad, 0.1);
  vec2 dg = rad < 1e-6 ? vec2(1.0, 0.0) : normalize(vec2(B, l1 - A));
  maj = min(sqrt(2.0 * l1), 1024.0) * dg; mnr = min(sqrt(2.0 * l2), 1024.0) * vec2(dg.y, -dg.x);
  ctr = vec2(2.0 * (uFocal.x * cam.x * iz + uPP.x) / uViewport.x - 1.0, 1.0 - 2.0 * (uFocal.y * cam.y * iz + uPP.y) / uViewport.y);
  return true;
}`;
const GLSL_HEAD = "#version 300 es\nprecision highp float; precision highp int; precision highp sampler2D;\n";
const SPLAT_OUT = `void splat(vec2 ctr, vec2 maj, vec2 mnr, vec4 col) {
  vec2 px = aQuad.x * maj + aQuad.y * mnr;
  gl_Position = vec4(ctr + vec2(px.x * 2.0 / uVp.x, -px.y * 2.0 / uVp.y), 0.0, 1.0); vColor = col / 255.0; vPos = aQuad;
}`;
// one pass: every vertex of a splat evaluates its Gaussian (4 times per Gaussian). Used when float render targets are missing.
const VS_GPU = GLSL_HEAD + "#define IX(j, i, N, J) ((i) * (J) + (j))\n" + `layout(location=0) in vec2 aQuad; layout(location=1) in uint aIdx; uniform vec2 uVp; out vec4 vColor; out vec2 vPos;` + GPU_EVAL + SPLAT_OUT + `
void main() { vec2 c, M, N; vec4 col; if (!evalSplat(int(aIdx), c, M, N, col)) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; } splat(c, M, N, col); }`;
// two passes (default): EVAL writes one texel pair per Gaussian (centre + major axis | minor axis + packed colour, colour < 0 = not drawn), SPLAT reads them
const VS_EVAL = GLSL_HEAD + `void main() { gl_Position = vec4(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0, 0.0, 1.0); }`;
const FS_EVAL = GLSL_HEAD + "#define IX(j, i, N, J) ((j) * (N) + (i))\n" + `uniform int uN; layout(location=0) out vec4 o0; layout(location=1) out vec4 o1;` + GPU_EVAL + `
void main() { ivec2 q = ivec2(gl_FragCoord.xy); int i = q.y * 2048 + q.x; if (i >= uN) discard;
  vec2 c, M, N; vec4 col; bool ok = evalSplat(i, c, M, N, col);
  o0 = vec4(c, M); o1 = vec4(N, ok ? col.r * 256.0 + col.g : -1.0, col.b * 256.0 + col.a); }`;
const VS_SPLAT = GLSL_HEAD + `layout(location=0) in vec2 aQuad; layout(location=1) in uint aIdx; uniform vec2 uVp; uniform sampler2D tP0, tP1; out vec4 vColor; out vec2 vPos;
ivec2 tx(int t) { return ivec2(t & 2047, t >> 11); }` + SPLAT_OUT + `
void main() { int i = int(aIdx); vec4 p0 = texelFetch(tP0, tx(i), 0), p1 = texelFetch(tP1, tx(i), 0);
  if (p1.z < 0.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  float r = floor(p1.z / 256.0), b = floor(p1.w / 256.0); splat(p0.xy, p0.zw, p1.xy, vec4(r, p1.z - 256.0 * r, b, p1.w - 256.0 * b)); }`;

const fsFor = ext => (ext === 4 ? FS : FS.replace("A < -4.0", "A < -" + ext.toFixed(4)));   // render.quadExtent
// render extras: a full-screen triangle (background plate, post) and a glint quad (catchlights)
const VS_FULL = GLSL_HEAD + `out vec2 vUv; void main() { vec2 p = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0); vUv = 0.5 * p + 0.5; gl_Position = vec4(p, 0.0, 1.0); }`;
const FS_PLATE = GLSL_HEAD + `in vec2 vUv; out vec4 o; uniform vec3 uBg, uTint; uniform vec2 uCtr, uAsp; uniform float uLift, uFall; uniform vec4 uSh; uniform vec2 uShS; uniform float uShA;
void main() { vec2 q = (vUv - uCtr) * uAsp; float g = exp(-dot(q, q) / (uFall * uFall)); float lum = dot(uBg, vec3(0.299, 0.587, 0.114));
  vec3 c = lum < 0.5 ? uBg + uLift * g * uTint : uBg * (1.0 - uLift * 0.6 * (1.0 - g));   // dark: a soft lit area behind the head; light: the edges fall off instead
  if (uShA > 0.0) { vec2 s = (vUv - uSh.xy) / uShS; float k = exp(-dot(s, s) * 2.0); c *= 1.0 - uShA * k; }
  o = vec4(c, 1.0); }`;
// Detail pass (DETAIL_DEFAULTS): CAS from ffx_cas.h (FidelityFX, MIT, Copyright (c) 2020 Advanced Micro Devices, Inc.): 3x3 min/max -> adaptive negative-lobe
// weight. Taps are bilinear at output-pixel offsets, so the same pass resolves a larger (supersampled) or equal texture to the canvas. + mid-tone luma grain.
const FS_CAS = GLSL_HEAD + `out vec4 o; uniform sampler2D uSrc; uniform vec2 uOut; uniform float uSharp, uGrain, uSeed;
vec3 T(vec2 d) { return texture(uSrc, (gl_FragCoord.xy + d) / uOut).rgb; }
float hash(vec2 p) { p = fract(p * vec2(443.897, 441.423) + uSeed); p += dot(p, p.yx + 19.19); return fract((p.x + p.y) * p.x); }
void main() { vec3 a = T(vec2(-1, 1)), b = T(vec2(0, 1)), c = T(vec2(1, 1)), d = T(vec2(-1, 0)), e = T(vec2(0)), f = T(vec2(1, 0)), g = T(vec2(-1, -1)), h = T(vec2(0, -1)), i = T(vec2(1, -1));
  vec3 col = e;
  if (uSharp > 0.0) { vec3 mn = min(min(min(d, e), min(f, b)), h); mn += min(mn, min(min(a, c), min(g, i)));
    vec3 mx = max(max(max(d, e), max(f, b)), h); mx += max(mx, max(max(a, c), max(g, i)));
    vec3 amp = sqrt(clamp(min(mn, 2.0 - mx) / max(mx, vec3(1e-4)), 0.0, 1.0)); vec3 w = -amp / (8.0 - 3.0 * uSharp);
    col = clamp((e + w * (b + d + f + h)) / (1.0 + 4.0 * w), 0.0, 1.0); }
  if (uGrain > 0.0) { float L = dot(col, vec3(0.299, 0.587, 0.114)); col = clamp(col + (hash(gl_FragCoord.xy) + hash(gl_FragCoord.xy + 17.3) - 1.0) * uGrain * 4.0 * L * (1.0 - L), 0.0, 1.0); }
  o = vec4(col, 1.0); }`;
// FSR1 EASU (ffx_fsr1.h FsrEasuF, FidelityFX, MIT, Copyright (c) 2021 Advanced Micro Devices, Inc.), GLSL ES port: 12-tap edge-adaptive Lanczos-2 approximation,
// clamped to the 2x2 neighbourhood. Input uSrc (texel size uIn), output canvas uOut.
const FS_EASU = GLSL_HEAD + `out vec4 o; uniform sampler2D uSrc; uniform vec2 uIn, uOut;
vec3 F(vec2 p) { return texelFetch(uSrc, ivec2(clamp(p, vec2(0), uIn - 1.0)), 0).rgb; }
float Lm(vec3 c) { return c.b * 0.5 + (c.r * 0.5 + c.g); }
void setF(inout vec2 dir, inout float len, float w, float lA, float lB, float lC, float lD, float lE) {
  float dc = lD - lC, cb = lC - lB, lenX = max(abs(dc), abs(cb)); lenX = lenX > 0.0 ? 1.0 / lenX : 0.0; float dirX = lD - lB; dir.x += dirX * w; lenX = clamp(abs(dirX) * lenX, 0.0, 1.0); len += lenX * lenX * w;
  float ec = lE - lC, ca = lC - lA, lenY = max(abs(ec), abs(ca)); lenY = lenY > 0.0 ? 1.0 / lenY : 0.0; float dirY = lE - lA; dir.y += dirY * w; lenY = clamp(abs(dirY) * lenY, 0.0, 1.0); len += lenY * lenY * w; }
void tap(inout vec3 aC, inout float aW, vec2 off, vec2 dir, vec2 len2, float lob, float clp, vec3 c) {
  vec2 v = vec2(off.x * dir.x + off.y * dir.y, off.x * -dir.y + off.y * dir.x) * len2; float d2 = min(dot(v, v), clp);
  float wB = 0.4 * d2 - 1.0, wA = lob * d2 - 1.0; wB *= wB; wA *= wA; wB = 1.5625 * wB - 0.5625; float w = wB * wA; aC += c * w; aW += w; }
void main() { vec2 pp = gl_FragCoord.xy * (uIn / uOut) - 0.5, fp = floor(pp); pp -= fp;   // y up in both (texel row 0 = bottom), so 'up' (b, c) is -1 in y as in FSR's y-down only by naming
  vec3 b = F(fp + vec2(0, -1)), c = F(fp + vec2(1, -1)), e = F(fp + vec2(-1, 0)), f = F(fp), g = F(fp + vec2(1, 0)), h = F(fp + vec2(2, 0));
  vec3 i = F(fp + vec2(-1, 1)), j = F(fp + vec2(0, 1)), k = F(fp + vec2(1, 1)), l = F(fp + vec2(2, 1)), n = F(fp + vec2(0, 2)), q = F(fp + vec2(1, 2));
  float bL = Lm(b), cL = Lm(c), eL = Lm(e), fL = Lm(f), gL = Lm(g), hL = Lm(h), iL = Lm(i), jL = Lm(j), kL = Lm(k), lL = Lm(l), nL = Lm(n), oL = Lm(q);
  vec2 dir = vec2(0); float len = 0.0;
  setF(dir, len, (1.0 - pp.x) * (1.0 - pp.y), bL, eL, fL, gL, jL); setF(dir, len, pp.x * (1.0 - pp.y), cL, fL, gL, hL, kL);
  setF(dir, len, (1.0 - pp.x) * pp.y, fL, iL, jL, kL, nL); setF(dir, len, pp.x * pp.y, gL, jL, kL, lL, oL);
  float dirR = dot(dir, dir); bool zro = dirR < 1.0 / 32768.0; dirR = zro ? 1.0 : inversesqrt(dirR); dir.x = zro ? 1.0 : dir.x; dir *= dirR;
  len = len * 0.5; len *= len; float stretch = dot(dir, dir) / max(abs(dir.x), abs(dir.y)); vec2 len2 = vec2(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
  float lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len, clp = 1.0 / lob; vec3 aC = vec3(0); float aW = 0.0;
  tap(aC, aW, vec2(0, -1) - pp, dir, len2, lob, clp, b); tap(aC, aW, vec2(1, -1) - pp, dir, len2, lob, clp, c); tap(aC, aW, vec2(-1, 1) - pp, dir, len2, lob, clp, i);
  tap(aC, aW, vec2(0, 1) - pp, dir, len2, lob, clp, j); tap(aC, aW, vec2(0, 0) - pp, dir, len2, lob, clp, f); tap(aC, aW, vec2(-1, 0) - pp, dir, len2, lob, clp, e);
  tap(aC, aW, vec2(1, 1) - pp, dir, len2, lob, clp, k); tap(aC, aW, vec2(2, 1) - pp, dir, len2, lob, clp, l); tap(aC, aW, vec2(2, 0) - pp, dir, len2, lob, clp, h);
  tap(aC, aW, vec2(1, 0) - pp, dir, len2, lob, clp, g); tap(aC, aW, vec2(1, 2) - pp, dir, len2, lob, clp, q); tap(aC, aW, vec2(0, 2) - pp, dir, len2, lob, clp, n);
  vec3 mn = min(min(f, g), min(j, k)), mx = max(max(f, g), max(j, k)); o = vec4(min(mx, max(mn, aC / aW)), 1.0); }`;
const FS_POST = GLSL_HEAD + `in vec2 vUv; out vec4 o; uniform float uVig, uGrain, uSeed; uniform vec2 uAsp;
float hash(vec2 p) { p = fract(p * vec2(443.897, 441.423) + uSeed); p += dot(p, p.yx + 19.19); return fract((p.x + p.y) * p.x); }
void main() { vec2 q = (vUv - 0.5) * uAsp; float v = 1.0 - uVig * smoothstep(0.25, 0.95, dot(q, q) * 1.6);
  float n = (hash(gl_FragCoord.xy) + hash(gl_FragCoord.xy + 17.3) - 1.0) * uGrain;
  o = vec4(vec3(0.5 * v * (1.0 + n)), 1.0); }`;   // drawn with blendFunc(DST_COLOR, SRC_COLOR): result = 2 * this * the picture
const VS_GLINT = GLSL_HEAD + `uniform vec2 uC, uR; uniform float uK; out vec2 vQ; void main() { vec2 q = vec2((gl_VertexID & 1) == 1 ? 1.0 : -1.0, gl_VertexID >= 2 ? 1.0 : -1.0) * uK; vQ = q; gl_Position = vec4(uC + q * uR, 0.0, 1.0); }`;
const FS_GLINT = GLSL_HEAD + `in vec2 vQ; out vec4 o; uniform vec3 uCol; uniform float uI, uHalo; void main() { float r2 = dot(vQ, vQ); float k = exp(-0.5 * r2) + uHalo * exp(-0.5 * r2 / 9.0); o = vec4(uCol * uI * k, 1.0); }`;   // additive

// Depth order of the visible Gaussians, far to near: the position and culling half of evaluate() (same arithmetic, same 16-bit counting sort,
// so the order is identical), without colour. Self-contained: it is also the body of the sort worker (see sortWorker).
// S: { n, Km, Ko, mean, hw|null, op, fixedLook, alpha0, motH|motF, ocH|ocF, H (half -> float table), counts, dep, idx }
// F: { wm, wo (Float32Array weights), head, Q (9, row-major), piv, zf, r2 (view row 2: x, y, z, t), cam? (render.sort "radial": camera position; the key is the distance
// to it instead of the view depth) }. Writes `out`, returns the count.
function sortSplats(S, F, out) {
  const { n, Km, Ko, mean, hw, op, fixedLook, alpha0, motH, motF, ocH, ocF, H, counts, dep, idx } = S, wm = F.wm, wo = F.wo;
  const bs = S.bs, bk = bs ? bs.k : -1, bw = bk >= 0 ? wm[bk] : 0, sweep = bk >= 0 && bw > 0 && bw < 1;   // blink sweep (as evalSplat; same formula as blinkSweepWeight)
  const actM = [], actO = []; for (let k = 0; k < Km; k++) if (wm[k] !== 0) actM.push(k); for (let k = 0; k < Ko; k++) if (wo[k] !== 0) actO.push(k);
  const nM = actM.length, nO = actO.length, head = !!F.head && !!hw, Q = F.Q, P = F.piv, zfx = F.zf[0], zfy = F.zf[1], zfz = F.zf[2];
  const r2x = F.r2[0], r2y = F.r2[1], r2z = F.r2[2], t2 = F.r2[3], C = F.cam || null; let m = 0, dmin = 1e9, dmax = -1e9;
  for (let i = 0; i < n; i++) {
    const i3 = i * 3; let x = mean[i3], y = mean[i3 + 1], z = mean[i3 + 2], o = i * Km * 3;
    if (motF) for (let q = 0; q < nM; q++) { const k = actM[q], b = wm[k], p = o + k * 3; x += b * motF[p]; y += b * motF[p + 1]; z += b * motF[p + 2]; }
    else for (let q = 0; q < nM; q++) { const k = actM[q], b = wm[k], p = o + k * 3; x += b * H[motH[p]]; y += b * H[motH[p + 1]]; z += b * H[motH[p + 2]]; }
    let a, dl = 0;
    if (sweep) { const p = o + bk * 3, mx = motF ? motF[p] : H[motH[p]], my = motF ? motF[p + 1] : H[motH[p + 1]], mz = motF ? motF[p + 2] : H[motH[p + 2]], E = bs.eyes, x0 = mean[i3];
      const e = Math.abs(x0 - E[0][0]) <= Math.abs(x0 - E[1][0]) ? E[0] : E[1]; let h = (e[1] - (mean[i3 + 1] + bs.mot * my)) / (e[1] - e[2]); h = h < 0 ? 0 : h > 1 ? 1 : h;
      let u = ((1 + bs.soft) * bw - h) / bs.soft; u = u < 0 ? 0 : u > 1 ? 1 : u; dl = u * u * (3 - 2 * u) - bw; x += dl * mx; y += dl * my; z += dl * mz; }
    if (fixedLook[i]) a = alpha0[i]; else { let ov = op[i]; o = i * Ko; for (let q = 0; q < nO; q++) { const k = actO[q]; ov += wo[k] * (ocF ? ocF[o + k] : H[ocH[o + k]]); } if (dl !== 0 && bk < Ko) ov += dl * (ocF ? ocF[o + bk] : H[ocH[o + bk]]); a = 1 / (1 + Math.exp(-ov)); }
    if (a < 0.004) continue;
    const hwi = head ? hw[i] : 0; if (hwi > 0) { const px = x - P[0], py = y - P[1], pz = z - P[2]; x += hwi * (P[0] + Q[0] * px + Q[3] * py + Q[6] * pz - x + zfx); y += hwi * (P[1] + Q[1] * px + Q[4] * py + Q[7] * pz - y + zfy); z += hwi * (P[2] + Q[2] * px + Q[5] * py + Q[8] * pz - z + zfz); }
    let d = r2x * x + r2y * y + r2z * z + t2; if (d < 0.2) continue;
    if (C) d = Math.hypot(x - C[0], y - C[1], z - C[2]);
    idx[m] = i; dep[m] = d; if (d < dmin) dmin = d; if (d > dmax) dmax = d; m++;
  }
  counts.fill(0); const sc = 65535 / Math.max(dmax - dmin, 1e-6);
  for (let j = 0; j < m; j++) { const k = 65535 - ((dep[j] - dmin) * sc | 0); dep[j] = k; counts[k]++; }
  let acc = 0; for (let k = 0; k < 65536; k++) { const c = counts[k]; counts[k] = acc; acc += c; }
  for (let j = 0; j < m; j++) out[counts[dep[j]]++] = idx[j];
  return m;
}
// The sort worker: sortSplats in a worker started from a blob: URL (the module stays one file, no build step). Message in: { init: S } once,
// then { gen, F, buf }; out: { gen, buf, m, ms } with the buffer handed back (transferred both ways).
function sortWorker(S) {
  const src = "const sortSplats = " + sortSplats.toString() + ";\nlet S = null;\nonmessage = e => { const d = e.data; if (d.init) { S = d.init; S.counts = new Uint32Array(65536); S.dep = new Float32Array(S.n); S.idx = new Int32Array(S.n); return; }\n" +
    "  const t0 = performance.now(), m = sortSplats(S, d.F, d.buf); postMessage({ gen: d.gen, buf: d.buf, m, ms: performance.now() - t0 }, [d.buf.buffer]); };";
  const url = URL.createObjectURL(new Blob([src], { type: "text/javascript" }));
  try { const w = new Worker(url); w.postMessage({ init: S }); return { w, url }; } catch (e) { URL.revokeObjectURL(url); return null; }
}

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
// LOOKAHEAD. An analyser can only see sound that has already been produced, so a mouth driven from it is late by the analysis window plus the
// smoothing. For a media element the viewer therefore plays the sound through a short delay line (`lookahead` seconds, preset driver.lookahead)
// and taps the analyser BEFORE the delay: the driver sees the sound `lead()` seconds before it is heard and can place the mouth on time (or a
// little ahead, as in natural speech) whatever the output latency of the device is. For a Web Audio node the caller owns the routing:
// `attachAudio(node, { lookahead })` states how long after the node's signal the sound is heard (default: the context's output latency).
const elementTaps = new WeakMap();                               // an element can be wired into a context only once, ever
function audioContextFor() { const AC = globalThis.AudioContext || globalThis.webkitAudioContext; if (!AC) throw fail("audio", "This browser has no Web Audio."); return new AC(); }
function makeSource(src, opts) {
  let ctx, analyser, el = null, node = null, tapped, delay = 0;
  const want = opts && typeof opts.lookahead === "number" && opts.lookahead >= 0 ? Math.min(1, opts.lookahead) : null;
  if (typeof HTMLMediaElement !== "undefined" && src instanceof HTMLMediaElement) {
    el = src; tapped = elementTaps.get(el);
    if (!tapped) {
      const c = audioContextFor(), s = c.createMediaElementSource(el); let d = null;
      try { d = c.createDelay(1); d.delayTime.value = 0; s.connect(d); d.connect(c.destination); } catch (e) { d = null; s.connect(c.destination); }
      tapped = { ctx: c, node: s, delay: d }; elementTaps.set(el, tapped);
    }
    ctx = tapped.ctx; node = tapped.node;
    if (tapped.delay) { delay = want === null ? 0 : want; tapped.delay.delayTime.value = delay; }
    const resume = () => { if (ctx.state === "suspended") ctx.resume().catch(() => {}); }; el.addEventListener("play", resume); resume();
  } else if (src && src.context && typeof src.connect === "function") { node = src; ctx = src.context; }
  else throw fail("audio", "attachAudio needs an HTMLAudioElement / HTMLVideoElement or a Web Audio node.");
  analyser = ctx.createAnalyser(); analyser.fftSize = 2048; analyser.smoothingTimeConstant = 0; node.connect(analyser);   // a tap: not connected to the output
  const startTime = opts && typeof opts.startTime === "number" ? opts.startTime : null;
  const latency = () => (ctx.outputLatency || ctx.baseLatency || 0);
  const lead = () => (el ? delay + latency() : want !== null ? want : latency());   // seconds between the analyser's newest sample and the moment it is heard
  return {
    el, node, ctx, analyser, buf: new Float32Array(analyser.fftSize), sr: ctx.sampleRate, lead, fixedLookahead: !!(opts && opts.fixedLookahead),
    setLookahead(s) { if (el && tapped.delay) { delay = clamp(+s || 0, 0, 1); tapped.delay.delayTime.value = delay; } },   // media elements only: the length of the delay line
    playing: () => (el ? !el.paused && !el.ended : ctx.state === "running"),
    // time of the sound that is audible now, in the clip's own time: the audio clock minus everything between the clock and the ear
    mediaTime() { return el ? el.currentTime - lead() : ctx.currentTime - lead() - (startTime || 0); },
    hasClock: !!el || startTime !== null,
    detach() { try { node.disconnect(analyser); } catch (e) { /* already gone */ } }
  };
}

// ------------------------------------------------------------------ mouth drivers: a small interface
// A driver factory is called as factory(env) and returns { attach(source), detach(), update(dt, out), speaking, level, setData?(d) }.
//   env.controls  names of all controls; env.Ks number of speech controls; env.index(name) -> speech index or -1
//   env.settings() -> the live preset.driver object (gain, smoothing, lead, timed settings)
// update(dt, out) writes the speech weights (length Ks, 0..1) for this frame. Register more with registerDriver(name, factory).
//
// What the two built-in drivers share (measured on the front-only avatar, working-docs/spikes/lipsync):
//  * the shapes of a trained model are single stills at weight 1. Sums of weights far above 1 leave what was trained (torn lips, a black hole), so both
//    drivers output a BLEND of shapes (weights that sum to about 1 or less), never a pile of them.
//  * weights below about 0.25 do not part the lips at all (the lips overlap at rest), so a "slightly open" target is invisible: closed is 0, and a
//    visible opening needs a total of 0.4 or more. The opening curve (driver.expand) accounts for it.
//  * the mouth runs `driver.lead` seconds ahead of the sound (lips move before the sound in natural speech, and the picture needs a frame to appear).

// ---- loudness driver (audio only) -----------------------------------------------------------------------------
// Per frame: 40 ms spectrum -> (a) how open: level against a running peak; (b) which shape: first-formant region high -> open jaw, second-formant
// region high -> spread ("ee"), both low -> closed and slightly rounded, hiss -> teeth together. The frames go into a short time-stamped queue and
// the mouth is read out of it at (heard time + lead) with a centred window, so smoothing costs no delay when the source gives lookahead. A narrow
// window on the opening keeps the short closures (stop gaps, p/b/m, pauses) that a wide one would wash out.
const LOUD_SHAPES = {                                            // each sums to <= 1; named controls a model lacks are skipped
  open: { wide: 0.55, ah: 0.3 }, openSpread: { wide: 0.4, ee: 0.45 },   // lipsync4 (v8): open vowels at ~55 % of the widest still, was ~66 %
  ee: { ee: 0.75, mid: 0.1 }, low: { mid: 0.5, oo: 0.3 }, hiss: { ee: 0.45, mid: 0.3 }, mid: { mid: 0.55, ah: 0.3 }
};
const LOUD_CLASSES = Object.keys(LOUD_SHAPES);
function mouthClasses(feat, erel, bandHz, out) {                 // out[0] = opening 0..1, out[1..] = share of each LOUD_CLASSES shape (sums to 1)
  out.fill(0); if (erel < VOPEN) return;
  let tot = 0, lo = 0, a = 0, hs = 0, n1 = 0, d1 = 0, n2 = 0, d2 = 0;
  for (let b = 0; b < NB; b++) {
    const hz = bandHz[b], p = Math.exp(feat[b]);
    if (hz >= 250 && hz < 1100) { n1 += p * hz; d1 += p; } if (hz >= 900 && hz < 3200) { n2 += p * hz; d2 += p; }
    if (hz < 300) continue; tot += p; if (hz < 550) lo += p; else if (hz >= 600 && hz < 1300) a += p; else if (hz >= 3500) hs += p;   // below 300 Hz is the voice's pitch, not the mouth shape
  }
  const f1 = d1 > 0 ? n1 / d1 : 0, f2 = d2 > 0 ? n2 / d2 : 0; lo /= tot; a /= tot; hs /= tot;
  const hiss = ss((hs - 0.3) / 0.3), op = Math.max(ss((f1 - 490) / 160), ss((a - 0.16) / 0.3)) * (1 - hiss), spread = ss((f2 - 1650) / 400), closed = (1 - op) * (1 - hiss);
  out[0] = ss((erel - VOPEN) / 4.5);                               // nasals and voiced consonants sit 1.5-3 below the vowels: they stay nearly shut
  out[1] = op * (1 - 0.6 * spread); out[2] = op * 0.6 * spread; out[3] = closed * spread; out[4] = closed * (1 - spread) * ss((lo - 0.4) / 0.3); out[5] = hiss;
  out[6] = Math.max(0, 1 - out[1] - out[2] - out[3] - out[4] - out[5]);
}
const VOPEN = -6.5;                                             // log power below the running peak at which the lips start to part (the v1 driver used VSIL = -7.5 and a softer ramp)
const hann = u => (u <= -1 || u >= 1 ? 0 : 0.5 + 0.5 * Math.cos(Math.PI * u));
// head co-motion pulses: a beat rises in the 0.15 s before its vowel (raised cosine, peak 70 ms into it) and fades over ~0.3 s; a settle eases in over 0.35 s and out over 1.2 s.
// The nod's velocity is continuous everywhere (motion LOG): the fall is critically damped, (1 + y) e^-y, which leaves the peak with zero speed and about the
// curvature the rise arrives with. Until 2026-11-21 a smoothstep rise ran straight into an exponential decay: the head reversed within one frame at the bottom
// of every nod (180-295 deg/s^2, a ~1 degree jab on each stressed vowel, audit4 item 4). Same size and nearly the same timing and area as that pulse.
const BEAT_PRE = 0.15, BEAT_PEAK = 0.07, BEAT_FALL = 0.09;
const beatPulse = u => { if (u <= -BEAT_PRE || u > 1.2) return 0; if (u < BEAT_PEAK) return 0.5 - 0.5 * Math.cos(Math.PI * (u + BEAT_PRE) / (BEAT_PRE + BEAT_PEAK)); const y = (u - BEAT_PEAK) / BEAT_FALL; return (1 + y) * Math.exp(-y); };
const settlePulse = u => (u < 0 || u > 1.55 ? 0 : u < 0.35 ? ss(u / 0.35) : 1 - ss((u - 0.35) / 1.2));
// SPEECH RELEASE (spikes/ends). When the speech is cut (the clip ends or is paused, the audio is detached or replaced, a new track or driver is set, the
// clip's clock jumps) the speech-driven values (mouth weights, nod / settle pitch) do not jump to their new value (rest, or the new clip's): they ease there
// over SPEECH_RELEASE_S (head) / SPEECH_RELEASE_MOUTH_S (mouth), leaving with the speed they had (a cubic Hermite on the difference; its start speed is limited to [-3d, 2d] for a difference d (mouth [-3d, d/2]),
// so it never crosses the target and overshoots by at most ~16 % of d (mouth ~2 %)). Until 2026-11-21 a running settle or nod dropped to 0 in one frame (up to 0.6 deg of head pitch).
// A cut while nothing speech-driven is shown (from rest) changes nothing, so a clip's first sound is not delayed.
const SPEECH_RELEASE_S = 0.25, SPEECH_RELEASE_MOUTH_S = 0.18;
// AUTO-BLINK (spikes/blink). makeBlinker(rand) -> { step(t, B, ctx) -> 0..1, busy() }. t = idle time (s), B = preset.idle.blink, ctx = { speaking, faster
// (speech.blinkFaster), gazeEnd (time the current gaze jump ends, s), held (a page holds the control) }. A blink starts every B.intervalS (B.intervalSpeakingS while
// speaking and faster), start to start, drawn uniformly; it closes over closeS (smoothstep), stays shut holdS, opens over openS (each drawn per blink). With chance
// doubleChance a second one follows doubleGapS after the first has opened (never a third). A blink that is due while the gaze jumps waits until gazeEnd + gazeGuardS;
// busy() is true from a blink's start to its end (and through a double's gap), so the caller holds gaze jumps back. While held, nothing runs and the next blink
// comes a fresh interval after the release. `first` = time of the first blink.
// BLINK SWEEP (spikes/blinkedge). A model header may carry blinkSweep = { control: "blink", eyes: [[cx, top, bottom], [cx, top, bottom]] (model metres, y up),
// soft, mot }: each Gaussian then gets its own weight of that control, from its height in the nearer eye (by x), so the lid edge sweeps down instead of the
// whole lid cross-fading. h = (top - (y + mot * blink motion y)) / (top - bottom) clamped to 0..1 (y = its rest height; with the motion share it counts from
// partway along its own blink path), and its weight = smoothstep(((1 + soft) w - h) / soft): 0 at w = 0 and 1 at w = 1 for every Gaussian (rest and fully
// closed frames are exactly the linear model's), top first. Weights outside (0, 1) are used as they are. Without the header nothing changes.
function blinkSweepWeight(w, h, soft) { if (!(w > 0 && w < 1)) return w; const u = clamp(((1 + soft) * w - h) / soft, 0, 1); return u * u * (3 - 2 * u); }
function blinkSweepHeight(bs, x, y, my) { const E = bs.eyes, e = Math.abs(x - E[0][0]) <= Math.abs(x - E[1][0]) ? E[0] : E[1]; return clamp((e[1] - (y + bs.mot * my)) / (e[1] - e[2]), 0, 1); }
// the header's blinkSweep checked against the rig: { k, eyes, soft, mot } or null (missing, malformed, or no such control)
function readBlinkSweep(hdr, controlNames, Km) {
  const b = hdr && hdr.blinkSweep; if (!b || typeof b !== "object") return null;
  const k = controlNames.indexOf(typeof b.control === "string" ? b.control : "blink"), soft = +b.soft, mot = b.mot === undefined ? 0 : +b.mot;
  const eyes = Array.isArray(b.eyes) && b.eyes.length === 2 && b.eyes.every(e => Array.isArray(e) && e.length === 3 && e.every(Number.isFinite) && e[1] > e[2]) ? b.eyes.map(e => e.map(Number)) : null;
  return k >= 0 && k < Km && eyes && soft > 0 && Number.isFinite(mot) ? { k, eyes, soft, mot } : null;
}
const blinkDraw = (v, d, rand) => (Array.isArray(v) ? v[0] + (v[1] - v[0]) * rand() : typeof v === "number" ? v : d);
// Lifelike additions (spikes/lf_core; all off by default, the random stream is unchanged unless used): B.halfChance (0) of an automatic blink being a half blink
// closing to B.halfDepth ([0.5, 0.65]); trigger(kind, o) starts a blink as soon as the running one has ended, whatever the interval and the gaze guard:
// "full" (drawn as an automatic one), "half" (depth o.depth or halfDepth), "long" (~0.6 s: close 0.13, shut 0.27, open 0.2; "got it"); o.closeS / holdS / openS
// override; o.auto === false in step() = no automatic blinks (triggered ones still run). kind() = the running blink's kind.
function makeBlinker(rand = Math.random, first = 2) {
  const b = { next: first, t0: -9, c: 0.12, h: 0.02, o: 0.2, second: false, held: false, busy: false, depth: 1, trig: null, kind: "auto" };
  const len = () => b.c + b.h + b.o;
  return {
    busy: () => b.busy,
    pending: () => !!b.trig || b.busy,
    kind: () => b.kind,
    state: () => ({ ...b }),
    trigger(kind = "full", o = {}) { b.trig = { kind: ["full", "half", "long"].includes(kind) ? kind : "full", o: o || {} }; },
    step(t, B, ctx = {}) {
      const interval = () => blinkDraw(ctx.speaking && ctx.faster ? B.intervalSpeakingS : B.intervalS, 4, rand);
      if (ctx.held) { b.held = true; b.t0 = -9; b.second = false; b.busy = false; b.trig = null; return 0; }
      if (b.held) { b.held = false; b.next = t + interval(); }
      if (b.trig && t - b.t0 > len()) {
        const { kind, o } = b.trig; b.trig = null; b.t0 = t; b.kind = kind; b.second = false;
        if (kind === "long") { b.c = o.closeS ?? 0.13; b.h = o.holdS ?? 0.27; b.o = o.openS ?? 0.2; b.depth = o.depth ?? 1; }
        else { b.c = o.closeS ?? blinkDraw(B.closeS, 0.12, rand); b.h = o.holdS ?? blinkDraw(B.holdS, 0.02, rand); b.o = o.openS ?? blinkDraw(B.openS, 0.2, rand); b.depth = kind === "half" ? (o.depth ?? blinkDraw(B.halfDepth, 0.6, rand)) : (o.depth ?? 1); }
        b.next = Math.max(b.next, t + len() + 1);
      } else if (ctx.auto !== false && t >= b.next && t - b.t0 > len() && !(t < (ctx.gazeEnd ?? -Infinity) + (B.gazeGuardS ?? 0))) {
        b.t0 = t; b.c = blinkDraw(B.closeS, 0.12, rand); b.h = blinkDraw(B.holdS, 0.02, rand); b.o = blinkDraw(B.openS, 0.2, rand);
        const dbl = !b.second && rand() < (B.doubleChance ?? 0); b.second = dbl;
        b.next = dbl ? t + len() + blinkDraw(B.doubleGapS, 0.1, rand) : t + interval();
        b.kind = "auto"; b.depth = 1; if ((B.halfChance ?? 0) > 0 && rand() < B.halfChance) { b.depth = blinkDraw(B.halfDepth, 0.6, rand); b.kind = "half"; }
      }
      const u = t - b.t0, c = b.c, h = b.h, o = b.o, on = u >= 0 && u <= c + h + o;
      b.busy = on || (b.second && t < b.next);
      return (!on ? 0 : u < c ? ss(u / c) : u < c + h ? 1 : 1 - ss((u - c - h) / o)) * b.depth;
    }
  };
}
function makeRelease(n, eps, T = SPEECH_RELEASE_S, out = 2) {           // out: how far the start speed may carry on away from the target (x the difference)
  const prev = new Float64Array(n), vel = new Float64Array(n), d0 = new Float64Array(n), v0 = new Float64Array(n); let u = Infinity;
  return {
    get active() { return u < T; },
    reset() { prev.fill(0); vel.fill(0); u = Infinity; },
    // live: this frame's values without the release (overwritten with what is shown); cut: the speech was cut since the last frame
    step(dt, live, cut, fromZero = false) {                         // fromZero (spikes/lf_motion): also ease when the held value was 0 and the new one is not (a clip starting mid-beat)
      if (cut) {
        let held = 0; for (let i = 0; i < n; i++) held = Math.max(held, Math.abs(prev[i]), fromZero ? Math.abs(prev[i] - live[i]) : 0);
        if (held > eps) { u = 0; for (let i = 0; i < n; i++) { const d = prev[i] - live[i], v = vel[i] * T; d0[i] = d; v0[i] = d >= 0 ? clamp(v, -3 * d, out * d) : clamp(v, out * d, -3 * d); } }
        else u = Infinity;
      }
      if (u < T) {
        u += dt; const s = Math.min(1, u / T), h00 = 1 - s * s * (3 - 2 * s), h10 = s * (1 - s) * (1 - s);
        if (s < 1) for (let i = 0; i < n; i++) live[i] += d0[i] * h00 + v0[i] * h10;
      }
      for (let i = 0; i < n; i++) { if (dt > 0) vel[i] = (live[i] - prev[i]) / dt; prev[i] = live[i]; }
      return live;
    }
  };
}
// Articulation: the opening shapes are scaled by k; when that takes the five shapes above a sum of 1 (where the rig's shapes start to tear), the shapes other
// than the jaw ("wide") give way first, so a bigger k means more jaw, never a broken mouth. k = 1 leaves the blend as it is. `press` is not touched.
const OPEN_CTL = ["wide", "ah", "ee", "oo", "mid"];
function articulationIndex(env) { return { wide: env.index("wide"), rest: OPEN_CTL.slice(1).map(n => env.index(n)).filter(i => i >= 0), restN: OPEN_CTL.slice(1).filter(n => env.index(n) >= 0) }; }
// OPEN GAIN (spikes/mouthgain): driver.openGain multiplies the opening shapes further (per control when an object) and lifts the sum limit from 1 to
// driver.openSum (0 = the largest gain): the trained opening shapes undershoot the stills (v20 at weight 1: ah 0.60, wide 0.72 of their chin drop), so a
// little more than one whole shape in total still reads as a mouth. `ramp` (0..1) fades it in like articulation (loudness: real openings only; timed: 1 - shut).
function openGainOf(S, ix) {
  const G = S.openGain ?? 1, per = n => { const v = isObj(G) ? G[n] : G; return Number.isFinite(+v) && +v > 0 ? +v : 1; };
  const gw = per("wide"), gr = OPEN_CTL.slice(1).map(per), top = Math.max(gw, ...gr), sum = Math.max(1, +S.openSum > 0 ? +S.openSum : top);
  return gw === 1 && sum === 1 && gr.every(x => x === 1) ? null : { gw, gr: OPEN_CTL.slice(1).map((n, j) => ix.restN.includes(n) ? gr[j] : null).filter(x => x !== null), sum };
}
function articulate(out, k, ix, floorWide = 0, og = null, ramp = 1) {
  if (!(k >= 0)) k = 1;
  const w = ix.wide; let sumO = 0, cap = 1;
  if (k !== 1) { if (w >= 0) out[w] *= k; for (const i of ix.rest) out[i] *= k; }
  if (og && ramp > 0) { const f = g => 1 + (g - 1) * ramp; if (w >= 0) { out[w] *= f(og.gw); floorWide *= f(og.gw); } ix.rest.forEach((i, j) => { out[i] = Math.min(out[i] * f(og.gr[j]), 1); }); cap = f(og.sum); }   // no single shape past its trained weight
  if (w >= 0 && floorWide > out[w]) out[w] = floorWide;              // a stressed open vowel: the jaw drops at least this far
  if (w >= 0) out[w] = Math.min(out[w], 1);
  for (const i of ix.rest) sumO += out[i];
  const room = cap - (w >= 0 ? out[w] : 0);
  if (sumO > room && sumO > 0) { const f = Math.max(0, room) / sumO; for (const i of ix.rest) out[i] *= f; }
}
// LIP CLOSURES FROM THE SOUND (loudness driver, spikes/loudlips). The opening follows the level, so the lips used to shut only where the level dips: not
// for an m inside a word (its hum is nearly as loud as the vowels) nor for a short p / b. makeLipCloser() watches 10 ms band levels every 5 ms
// (lo < 400 Hz, mid 500-3000 Hz, hi > 3 kHz; two or three 512-point FFTs per frame, no model) for two events and shuts the lips over them (driver.lips):
//  * hum: mid and hi at least humDb below lo, mid humMidDrop below the vowel before, lo still voiced (a nasal murmur, or a b's voice bar), for >= humMin s,
//    ENDING in a quick rise of mid back towards the vowel (within releaseS, no silence or hiss first): the lips part into a vowel. An n / ng mostly runs into
//    a t, d, z or a pause instead (English codas) and is left as it was. Without lookahead (a microphone) the release is not seen in time: the hum alone
//    then closes the lips (humNoLook, a stricter hum and a shorter closure).
//    PLACE (spikes/nvsm): an n before a vowel ("in the", "and the") is acoustically such a hum too. Its murmur has a zero near 1.2-1.5 kHz and its second nasal
//    resonance near 1.65-2.1 kHz; an m's murmur falls smoothly there. The median over the hum's hops of level(1.65-2.1 kHz) - level(1.2-1.5 kHz) (bins by centre,
//    so the same band at 16 and 48 kHz) was n +4.3..+16.9 dB and m -10.9..+0.5 dB on the 8 test clips (22 murmurs, no overlap; 16 kHz: n >= +3.0, m <= +0.6):
//    at placeDb (2) or above it is an n and the lips stay as they were. It does not tell ng from m (ng's zero is higher); the F2 rule above handles most ng.
//    An m right after a hiss ("sends me") has no mid fall to see (the hiss had no vowel-level mid): a hiss within hissEntryS before the hum enters it instead.
//  * stop: a silence (silDb below the level of the last 0.3 s) of stopMin..stopMax s after sound, ended by a burst that is not mostly above 3 kHz (hi - mid <=
//    burstHiMid: a p / b; a t's burst is a hiss): shut from the silence to the burst. Without lookahead the burst is seen ~35 ms late: a shorter closure.
// Each event shuts the lips fully over its span, with an attack / release ramp outside it (with lookahead a stop's starts to open burstOpen s before the burst); the closure fades
// the opening shapes and presses the lips as the timed driver's p/b/m do. Events are decided once (a later frame cannot undo a closure already shown).
// Anti-chatter (nvsm, for 30 fps and the microphone): for minOpen s after a closure opens, the opening does not drop back (a level dip at the next k / t made
// open-shut-open flaps); without lookahead, while an m-like hum is under way the opening does not grow (it opened 1-2 frames on the hum before its closure came).
const LIPS_HOP = 0.005, LIPS_WIN = 0.010, LIPS_KEEP = 0.6;
function makeLipCloser() {
  let H = [], next = null, sr = 0, n = 0, nfft = 0, re = null, im = null, wnd = null, bins = null, ev = [], scanFrom = 0, cfg = {};
  const lvl = (a, b) => { let p = 0; for (let k = a; k < b; k++) p += re[k] * re[k] + im[k] * im[k]; return 10 * Math.log10(p + 1e-12); };
  function setup(rate) {
    sr = rate; n = Math.round(LIPS_WIN * sr); nfft = 1; while (nfft < n) nfft <<= 1; re = new Float32Array(nfft); im = new Float32Array(nfft);
    wnd = Float32Array.from({ length: n }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)));
    const bin = f => Math.max(1, Math.min(nfft / 2, Math.round(f * nfft / sr))); const binC = f => Math.min(nfft / 2, Math.ceil(f * nfft / sr - 1e-9)); bins = { lo: [bin(100), bin(400)], mid: [bin(500), bin(3000)], hi: [bin(3000), bin(Math.min(8000, sr / 2))], f2: [bin(900), bin(2800)], zlo: [binC(1200), binC(1500)], zhi: [binC(1650), binC(2100)] };
  }
  function hop(buf, i0, t, S) {                                   // band levels of the 10 ms from buf[i0], centred at heard time t, against the last 0.3 s
    re.fill(0); im.fill(0); for (let i = 0; i < n; i++) re[i] = buf[i0 + i] * wnd[i]; fft(re, im);
    const lo = lvl(...bins.lo), mid = lvl(...bins.mid), hi = lvl(...bins.hi), tot = 10 * Math.log10(10 ** (lo / 10) + 10 ** (mid / 10) + 10 ** (hi / 10));
    let pf = 0, pp = 1e-20; for (let k = bins.f2[0]; k < bins.f2[1]; k++) { const q = re[k] * re[k] + im[k] * im[k]; pf += q * k; pp += q; } const f2 = pf / pp * sr / nfft;   // centre of 0.9-2.8 kHz (follows the second formant)
    let ref = tot, midRef = mid, loRef = lo; for (let k = H.length - 1; k >= 0 && H[k].t > t - 0.3; k--) { const q = H[k]; if (q.tot > ref) ref = q.tot; if (q.mid > midRef) midRef = q.mid; if (q.lo > loRef) loRef = q.lo; }
    const sil = tot < ref - (S.silDb ?? 25) || tot < -95;
    const mur = !sil && lo - Math.max(mid, hi) >= (S.humDb ?? 10) && mid <= midRef - (S.humMidDrop ?? 6) && lo >= loRef - (S.humLoDrop ?? 7);
    const nz = lvl(...bins.zhi) - lvl(...bins.zlo);              // the n zero: an n murmur dips at 1.2-1.5 kHz and has its 2nd nasal resonance at 1.65-2.1 kHz (> 0); an m falls smoothly (< 0)
    const mur2 = !sil && lo - Math.max(mid, hi) >= (S.humDb ?? 10) + (S.hissHumDb ?? 10) && mid <= midRef - (S.humMidDrop ?? 6) && lo >= loRef - (S.hissLoDrop ?? 14);   // a murmur still building up after a hiss
    const hiss = !sil && hi > mid + (S.hissDb ?? 3);
    H.push({ t, lo, mid, hi, tot, ref, midRef, loRef, f2, sil, mur, nz, mur2, hiss });
  }
  function push(e, a) {                                           // bridge to the previous closure when it ended less than mergeS ago and nothing as loud as a vowel came between
    const p = ev[ev.length - 1], S = cfg;
    if (p && p.e !== Infinity && e.s - p.e < (S.mergeS ?? 0.06)) { let mx = -Infinity, ref = -Infinity; for (let k = a - 1; k >= 0 && H[k].t > p.e; k--) { mx = Math.max(mx, H[k].mid); ref = Math.max(ref, H[k].midRef); } p.bridge = mx <= ref - (S.mergeDb ?? 10); }
    ev.push(e);
  }
  const zs = [];
  function alveolar(a, b, S) {                                    // the murmur's median n-zero index says n (tongue tip), not m (lips)
    if (S.place === false) return false;
    zs.length = 0; for (let k = a; k <= b; k++) zs.push(H[k].nz); zs.sort((x, y) => x - y);
    const m = zs.length & 1 ? zs[zs.length >> 1] : 0.5 * (zs[zs.length / 2 - 1] + zs[zs.length / 2]); return m >= (S.placeDb ?? 2);
  }
  const hops = k => Math.max(1, Math.round(k / LIPS_HOP));
  function entry(a, b, S) {                                       // how the hum H[a..b] was entered: { fell, lab, a0 } (a0: where its closure starts)
    let mn = Infinity, pre = -Infinity; for (let k = a; k <= b; k++) mn = Math.min(mn, H[k].mid); for (let k = Math.max(0, a - 12); k < a; k++) pre = Math.max(pre, H[k].mid);
    let fell = mn - pre <= -(S.humDrop ?? 8);                      // the mid band fell into the hum within 60 ms (the lips closed), it was not low already
    let a0 = a, fromHiss = false;                                 // ...or the hum follows a hiss ("sends me"): the mid band was down already. Its first hops are still building up (mur2)
    if (!fell && S.hissEntry !== false) {
      while (a0 > 1 && H[a0 - 1].mur2 && !H[a0 - 1].mur) a0--;
      for (let k = a0 - 1; k >= 0 && k >= a0 - hops(S.hissEntryS ?? 0.03); k--) if (H[k].hiss) { fromHiss = fell = true; break; }
      if (!fromHiss) a0 = a;
    }
    const p0 = H[a - 2], p1 = H[a - 9];                           // the vowel before: its second formant rises into an ng (and a y), it falls or stays into an m
    let lab = fromHiss || !(p0 && p1 && !p0.sil && !p1.sil && !p0.mur && !p1.mur && p0.f2 > p1.f2 * (S.humF2Rise ?? 1.15));
    const LF = S.humLoFall ?? 3;                                  // (snaps) an m's murmur keeps the voice's < 400 Hz level; a v or w (an oral narrowing, no nasal outlet) drops it
    if (lab && !fromHiss && LF > 0) { let preLo = -Infinity; for (let k = Math.max(0, a - 12); k < a; k++) preLo = Math.max(preLo, H[k].lo); zs.length = 0; for (let k = a; k <= b; k++) zs.push(H[k].lo); zs.sort((x, y) => x - y); if (zs[zs.length >> 1] < preLo - LF) lab = false; }
    return { fell, lab, a0 };
  }
  function scan(S, look) {                                        // find hum / stop events in the history; each is decided once, in time order
    const N = H.length;
    for (let a = Math.max(1, scanFrom); a < N; a++) {
      const q = H[a];
      if (q.mur && !H[a - 1].mur && S.hum !== false) {           // a hum starts
        let b = a; while (b + 1 < N && H[b + 1].mur) b++;
        const { fell, lab, a0 } = entry(a, b, S);
        if (b + 1 >= N) {                                         // still humming at the newest sound: decide when it ends; without lookahead a long, clear hum closes now
          if (!look && (S.humNoLook ?? true) && fell && lab && b - a + 1 >= hops(S.humMinNoLook ?? 0.04) && !alveolar(a, b, S)) {
            let clear = true; for (let k = a; k <= b; k++) if (H[k].lo - Math.max(H[k].mid, H[k].hi) < (S.humDb ?? 10) + (S.humNoLookDb ?? 4)) clear = false;
            if (clear) { ev.push({ s: H[a0].t, e: Infinity, kind: "hum0" }); scanFrom = N; return; }
          }
          scanFrom = a; return;
        }
        if (b - a0 + 1 >= hops(S.humMin ?? 0.025) && fell && lab && !alveolar(a, b, S)) {
          let ok = null;                                          // the release: mid rises back towards the vowel within releaseS, before any silence or hiss
          for (let j = b + 1; j <= b + hops(S.releaseS ?? 0.035); j++) {
            if (j >= N) { ok = undefined; break; }
            const r = H[j]; if (r.sil || r.hi > r.mid + (S.hissDb ?? 3)) { ok = false; break; }
            if (r.mid >= H[b].mid + (S.releaseDb ?? 4) && r.mid >= H[a].midRef - (S.releaseToVowel ?? 16)) { ok = true; break; }
          }
          if (ok === undefined) { scanFrom = a; return; }          // the release is not in view yet
          if (ok) push({ s: H[a0].t, e: H[b].t, kind: "hum" }, a0);
        }
        a = b; continue;
      }
      if (q.sil && !H[a - 1].sil && S.stop !== false) {          // a silence starts after sound
        let b = a; while (b + 1 < N && H[b + 1].sil) b++;
        const dur = (b - a + 1) * LIPS_HOP;
        if (dur > (S.stopMax ?? 0.2)) { a = b; continue; }        // a pause: the opening already rests
        if (b + 2 >= N) { scanFrom = a; return; }                 // the burst is not in view yet
        if (dur >= (S.stopMin ?? 0.02)) {
          const j = b + 1, hm = (H[j].hi - H[j].mid + H[j + 1].hi - H[j + 1].mid) / 2; let fl = Infinity; for (let k = a; k <= b; k++) fl = Math.min(fl, H[k].tot);
          const jump = Math.max(H[j].hi - H[j - 1].hi, H[j + 1].hi - H[j].hi);   // a burst is a click: the band above 3 kHz jumps within 5 ms (a vowel, dh or v starting after a silence does not)
          const lm = H[j].lo - H[j].mid;                          // a lip burst has little 0.5-3 kHz energy of its own (no front cavity): a vowel starting after a pause, or a k, has a lot
          if (Math.max(H[j].tot, H[j + 1].tot) - fl >= (S.burstDb ?? 20) && hm <= (S.burstHiMid ?? -4) && jump >= (S.burstJump ?? 9) && lm >= (S.burstLoMid ?? -10)) push({ s: H[a].t, e: Math.max(H[a].t, H[j].t - LIPS_HOP / 2 - (look ? S.burstOpen ?? 0.015 : 0)), kind: "stop" }, a);
        }
        a = b; continue;
      }
    }
    scanFrom = N;
  }
  return {
    reset() { H = []; next = null; ev = []; scanFrom = 0; },
    get events() { return ev; },
    // a lip hum is under way at the newest sound (entered like an m, not an n so far, >= humPend s): without lookahead the opening should not grow meanwhile
    pending(S) {
      const N = H.length; if (N < 3 || !H[N - 1].mur || S.hum === false) return false;
      let a = N - 1; while (a > 1 && H[a - 1].mur) a--;
      if (N - a < hops(S.humPend ?? 0.005)) return false;
      const e = entry(a, N - 1, S); return e.fell && e.lab && !alveolar(a, N - 1, S);
    },
    // buf = the analyser's newest samples, the last one heard at tEnd (driver clock); look = the source gives lookahead
    feed(buf, rate, tEnd, S, look) {
      cfg = S;
      if (rate !== sr) { setup(rate); H = []; next = null; ev = []; scanFrom = 0; }
      if (next !== null && next > tEnd + 0.05) { H = []; ev = []; scanFrom = 0; next = null; }   // the clock went back: start over
      const first = tEnd - (buf.length - n / 2) / sr;             // earliest hop centre this buffer can give
      if (next === null || next < first - LIPS_HOP) next = Math.ceil(first / LIPS_HOP) * LIPS_HOP;
      for (; next + LIPS_WIN / 2 <= tEnd; next += LIPS_HOP) { const i0 = Math.round(buf.length - (tEnd - next) * sr - n / 2); if (i0 >= 0 && i0 + n <= buf.length) hop(buf, i0, next, S); }
      let cut = 0; while (cut < H.length && H[cut].t < tEnd - LIPS_KEEP) cut++;
      if (cut) { H.splice(0, cut); scanFrom = Math.max(0, scanFrom - cut); }
      const maxNL = S.humMaxNoLook ?? 0.08;                        // a hum closed without lookahead ends where its hum ends (at most humMaxNoLook after it started)
      for (const e of ev) if (e.e === Infinity) { const last = H[H.length - 1]; if (!last.mur || last.t - e.s >= maxNL) { let k = H.length - 1; while (k > 0 && !H[k].mur) k--; e.e = Math.min(Math.max(H[k].t, e.s), e.s + maxNL); } }
      scan(S, look);
      ev = ev.filter(e => e.e > tEnd - 1);
    },
    // how shut the lips are at heard time t (0..1): full over each event, ramps of attack / release outside it; shut between two events less than mergeS apart
    at(t, S) {
      const atk = S.attack ?? 0.035, rel = S.release ?? 0.03; let c = 0;
      for (let i = 1; i < ev.length; i++) { const p = ev[i - 1], q = ev[i]; if (p.bridge && t > p.e && t < q.s) return 1; }
      for (const e of ev) { const k = t < e.s ? ss((t - (e.s - atk)) / atk) : t <= e.e ? 1 : 1 - ss((t - e.e) / rel); if (k > c) c = k; }
      return c;
    }
  };
}
// DIP BRIDGE (spikes/snaps). The narrow window lets the opening follow a real closure closely, but a short stop inside a vowel (the t / v of "avatar", the
// break in "I") then shut the mouth for exactly one 60 fps frame: a visible snap of the teeth. bridgedNarrow() reads the per-frame level as steps (each frame
// holds to halfway to the next; 5 ms grid), closes it morphologically over B s (dilate, then erode, each B / 2 to either side: a dip narrower than B is filled to its edges, a
// wider one, such as a pause or a 90 ms stop, keeps its depth and its edges) and takes the narrow window over that. Only frames already in Q count: without
// lookahead (a microphone) there is little to the right of the read-out point and little is bridged. Lip closures (driver.lips) come after it, unchanged.
const BRIDGE_STEP = 0.005;
function bridgedNarrow(Q, td, narrow, B) {
  const h = Math.max(1, Math.round(B / 2 / BRIDGE_STEP)), nw = Math.ceil(narrow / BRIDGE_STEP), n = nw + 2 * h, raw = [], dil = [];
  let k = 0;
  for (let j = -n; j <= n; j++) {                                  // each frame's level holds to halfway to its neighbours (-1 before the first frame and after the newest)
    const u = td + j * BRIDGE_STEP; while (k + 1 < Q.length && Q[k + 1].t - u < u - Q[k].t) k++;
    const q = Q[k], sp = Q.length > 1 ? (Q[Q.length - 1].t - Q[0].t) / (Q.length - 1) : 0;
    raw.push(Math.abs(u - q.t) <= 0.5 * sp + 1e-6 ? q.f[0] : -1);
  }
  for (let i = h; i <= 2 * n - h; i++) { let d = -1; for (let j = i - h; j <= i + h; j++) if (raw[j] > d) d = raw[j]; dil[i] = d; }
  let s = 0, o = 0;
  for (let i = n - nw; i <= n + nw; i++) {
    if (raw[i] < 0) continue;
    let e = Infinity; for (let j = i - h; j <= i + h; j++) if (dil[j] >= 0 && dil[j] < e) e = dil[j];
    const w = hann((i - n) * BRIDGE_STEP / narrow); if (w > 0) { s += w; o += w * Math.max(raw[i], e); }
  }
  return s > 0 ? o / s : -1;
}
function loudnessDriver(env) {
  const NC = LOUD_CLASSES.length, tmp = new Float32Array(NB), f = new Float32Array(1 + NC), acc = new Float32Array(1 + NC);
  let src = null, peak = -20, clock = 0, speaking = false, level = 0, Q = [], quiet = 9, talked = false, beatT = -9, beatT0 = -9, settleT = -9;
  const shapes = LOUD_CLASSES.map(c => Object.entries(LOUD_SHAPES[c]).map(([k, v]) => [env.index(k), v]).filter(e => e[0] >= 0)), aix = articulationIndex(env);
  const lips = makeLipCloser(), pressI = env.index("press"); let shut = 0, relT = -9, oHold = 0, oPrev = 0;
  return {
    name: "loudness",
    attach(s) { src = s; peak = -20; Q = []; clock = 0; quiet = 9; talked = false; beatT = beatT0 = settleT = -9; lips.reset(); shut = 0; relT = -9; oHold = 0; }, detach() { src = null; Q = []; speaking = false; level = 0; lips.reset(); shut = 0; relT = -9; oHold = 0; },
    get closure() { return shut; }, get lipEvents() { return lips.events; },   // how shut the lips are from the sound's own closures (0..1), and those events (tests)
    get speaking() { return speaking; }, get level() { return level; },
    get beat() { return 1 - (1 - beatPulse(clock - beatT)) * (1 - beatPulse(clock - beatT0)); }, get settle() { return settlePulse(clock - settleT) * (beatT > settleT ? 1 - ss((clock - beatT + BEAT_PRE) / 0.3) : 1); },        // no timings: a beat at each phrase start (sound after >= 0.3 s of quiet), a settle 0.25 s into a pause
    update(dt, out) {
      const S = env.settings(); out.fill(0);
      if (!src) { speaking = false; level = 0; return; }
      clock += dt;
      src.analyser.getFloatTimeDomainData(src.buf); const w = Math.round(WIN_S * src.sr), tail = src.buf.length - w;
      let ss2 = 0; for (let i = Math.max(0, tail); i < src.buf.length; i++) ss2 += src.buf[i] * src.buf[i];
      const rms = Math.sqrt(ss2 / Math.min(w, src.buf.length));
      let erel;
      if (rms < 2e-4) erel = -99;                                // digital silence: no running-peak reference to compare with
      else { const e = spectralShape(src.buf, src.buf.length - w / 2, tmp, src.sr); peak = Math.max(e, peak - dt * 0.3); erel = e - peak; }
      mouthClasses(tmp, erel, bankFor(src.sr).hz, f);
      const lead = src.lead ? src.lead() : 0, tf = clock + lead - WIN_S / 2;                 // the heard time this frame's spectrum belongs to
      for (let c = 1; c <= NC; c++) f[c] *= f[0];                                           // shares weighted by the opening: silence does not vote on the shape
      Q.push({ t: tf, f: Float32Array.from(f) }); while (Q.length > 2 && Q[0].t < clock - 0.4) Q.shift();
      let td = clock + (S.lead ?? 0.02), wide = Math.max(0.02, (S.smoothing ?? 0.045) * 1.6); const narrow = Math.min(wide, Math.max(0.028, dt * 1.1));
      const look = !(tf - td < 0.6 * wide);
      if (!look) { wide = Math.min(wide, 0.045); td = Math.min(td, tf - 0.5 * wide); }   // too little lookahead (a microphone, a node without delay): a shorter, mostly one-sided window; the mouth is then late by about 40 ms
      const L = S.lips || {}, lipsOn = L.enabled !== false;
      if (lipsOn) lips.feed(src.buf, src.sr, clock + lead, L, look);
      let sw = 0, sn = 0, on = 0; acc.fill(0);
      for (const q of Q) { const u = q.t - td, kw = hann(u / wide), kn = hann(u / narrow); if (kw > 0) { sw += kw; for (let c = 0; c <= NC; c++) acc[c] += kw * q.f[c]; } if (kn > 0) { sn += kn; on += kn * q.f[0]; } }
      if (sw <= 0) { let q = null; for (const e of Q) if (e.t <= td) q = e; sw = 1; if (q) acc.set(q.f); }   // no frame inside the window (a long frame gap): hold the last one before the read-out point; none yet = rest
      const B = S.dipBridge ?? 0.035; if (B > 0 && sn > 0) { const b = bridgedNarrow(Q, td, narrow, B); if (b >= 0) on = b * sn; }
      const ow = acc[0] / sw, onar = sn > 0 ? on / sn : ow;
      let o = Math.min(ow, onar * 1.15 + 0.03);                                            // closures stay: the narrow window pulls the opening down, never up
      if (lipsOn && clock - relT < (L.minOpen ?? 0.05)) { oHold = Math.max(oHold, o); o = oHold; }
      else if (lipsOn && !look && lips.pending(L)) o = Math.min(o, oPrev);              // no lookahead: while a hum that may be an m is under way, the opening does not grow (it dipped into the m, then opened on the hum's own level for 1-2 frames before the closure came)
      oPrev = o;   // just after a lip closure opened: the vowel's opening does not drop back within minOpen s (a level dip there, at a k or t, made the lips flap open-shut at 30 fps)
      level = o; speaking = o > 0.06;
      if (o < 0.06) { quiet += dt; if (talked && quiet >= 0.25) { settleT = clock; talked = false; } }                 // pulses start from zero when detected: no step in the head
      else { if (o > 0.3 && quiet >= 0.3) { beatT0 = beatT; beatT = clock + BEAT_PRE; } if (o > 0.3) { quiet = 0; talked = true; } }   // the previous nod fades on under the new one
      if (acc[0] > 1e-6 && o > 0) {
        const amp = (S.gain ?? 1) * Math.pow(o, S.expand ?? 0.7) / acc[0];
        for (let c = 0; c < NC; c++) { const share = acc[c + 1] * amp; if (share > 0) for (const [i, v] of shapes[c]) out[i] += share * v; }
        for (let i = 0; i < out.length; i++) out[i] = clamp(out[i], 0, 1);
        const art = S.articulation ?? 1, ra = ss((o - 0.35) / 0.4); articulate(out, 1 + (art - 1) * ra, aix, 0, openGainOf(S, aix), ra);   // only real openings grow: narrowed closures (m, stop gaps) stay as narrow as without it
      }
      const want = lipsOn ? lips.at(Math.min(clock + (L.lead ?? 0), td), L) : 0;   // the sound's own lip closures (m hum, p / b stop), on the heard time (not run ahead by lead)
      const up = 1.5 * dt / (L.attack ?? 0.035), down = 1.5 * dt / (L.release ?? 0.03);   // never faster than the ramps: a closure decided late (no lookahead) still eases in
      const was = shut; shut = dt > 0 ? clamp(want, shut - down, shut + up) : want;
      if (was >= 0.5 && shut < 0.5) { relT = clock; oHold = 0; }
      if (shut > 0) { for (let i = 0; i < out.length; i++) if (i !== pressI) out[i] *= 1 - shut; if (pressI >= 0) out[pressI] = Math.max(out[pressI], shut); }
    }
  };
}

// ---- timed driver ------------------------------------------------------------------------------------------
// A track of mouth events played against the audio clock. Track item (seconds, audio time):
//   { viseme: "pbm" | "aa" | ..., start, end }   one of VISEME_WEIGHTS
//   { phone: "AY", start, end }                  ARPAbet without stress marks (what a pronouncing dictionary gives); diphthongs are split in two
//   { weights: { ah: 0.8, ... }, start, end }    explicit control weights
// or character timings ({ timings: [{char, start, end}] }, as ElevenLabs returns them), turned into phones by a small English rule set.
//
// How the track becomes motion (a dominance blend, after Cohen & Massaro's co-articulation model, much simplified):
//   every event has a shape, a strength ("dominance") and a ramp that starts BEFORE its sound (lips anticipate) and ends after it.
//   At any moment the mouth is the dominance-weighted AVERAGE of the active events' shapes, times how active they are. So
//   - weak events (t, d, n, l, k, g: tongue sounds with no lip shape of their own) take most of their shape from the vowels around them,
//   - strong ones (rounded vowels, f/v, s/z) hold their own; p/b/m is not blended at all: it fades everything else out, and the lips are shut before the burst is heard,
//   - the result is always a blend, never a sum (see the note at the top of the drivers).
// Gaps under 150 ms hold the last shape; longer ones return to rest (lips closed); the first sound after a pause opens early.
const VISEME_WEIGHTS = {                                          // tuned on the front-only avatar's sweep (lip gap, width, pucker per control); each sums to <= 1
  sil: {},
  pbm: { press: 1.0 },                                            // p b m: lips shut
  fv: { mid: 0.45, press: 0.3 },                                  // nearly shut (no lip-under-teeth shape in the rig)
  th: { mid: 0.6, ah: 0.2 },
  td: { mid: 0.55, ee: 0.2 },
  kg: { mid: 0.5, ah: 0.25 },
  sh: { oo: 0.55, mid: 0.35 },                                    // ch j sh zh
  sz: { ee: 0.5, mid: 0.25 },                                     // teeth together, lips apart
  n: { mid: 0.6, ee: 0.1 },
  l: { ah: 0.45, mid: 0.3 },
  r: { oo: 0.45, mid: 0.35 },
  er: { oo: 0.35, mid: 0.4, ah: 0.15 },                           // her, bird
  aa: { wide: 0.55, ah: 0.28 },                                   // father: the jaw drops ("wide" is the rig's jaw-open shape). lipsync4 (v8): was wide .68 + ah .3, which put open vowels at 64-77 % of the widest still; now ~57 % (63 % stressed)
  ae: { wide: 0.4, ee: 0.45 },                                    // cat (lipsync4: wide was .5)
  ah: { ah: 0.6, wide: 0.35 },                                    // cut
  eh: { ee: 0.4, ah: 0.3, wide: 0.28 },                           // bed
  schwa: { mid: 0.5, ah: 0.35 },
  ih: { ee: 0.65, mid: 0.2 },                                     // bit, y
  iy: { ee: 0.85 },                                               // beat
  oh: { oo: 0.6, wide: 0.3, ah: 0.08 },                           // caught, the start of "go": rounded with the jaw down
  uw: { oo: 0.95 }                                                // boot, w
};
// per viseme: [dominance, attack s (ramp before the sound), release s (ramp after it)]
const VISEME_DYN = {
  pbm: [0, 0.045, 0.045], fv: [3, 0.06, 0.05], sz: [1.0, 0.05, 0.06], sh: [1.3, 0.09, 0.08], th: [0.5, 0.05, 0.06], td: [0.4, 0.04, 0.05], n: [0.4, 0.04, 0.05], l: [0.5, 0.05, 0.06], kg: [0.4, 0.04, 0.05], r: [0.9, 0.08, 0.08], er: [0.9, 0.08, 0.09],
  aa: [1, 0.08, 0.1], ae: [1, 0.08, 0.1], ah: [1, 0.07, 0.1], eh: [1, 0.07, 0.1], schwa: [0.7, 0.06, 0.09], ih: [1, 0.07, 0.09], iy: [1.1, 0.08, 0.09], oh: [1.3, 0.1, 0.1], uw: [1.6, 0.11, 0.1]
};
const TONGUE = { th: 1, td: 1, n: 1, l: 1, kg: 1 };
// jaw floor ("wide") of the open vowels while they sound, so that a vowel squeezed between consonants still opens; full for a long (stressed) vowel,
// half for a short one (the letter timings carry no stress marks; a stressed vowel is the longer one)
const VOWEL_VIS = { aa: 1, ae: 1, ah: 1, eh: 1, schwa: 1, ih: 1, iy: 1, oh: 1, uw: 1, er: 1 }, VOWEL_FLOOR_ANY = e => !!VOWEL_VIS[e.vis];
const VOWEL_FLOOR = { aa: 0.38, ae: 0.3, ah: 0.3, eh: 0.24, oh: 0.22 }, STRESS_S = 0.075;   // aa .5 / ae .4 until lipsync4
const PAUSE_REST = 0.08, VOWEL_OPEN_MIN = 0.04, VOWEL_PULL_MAX = 0.05, CLOSE_DELAY_MAX = 0.03;   // seconds; see compile() in the timed driver (motion LOG)
// ARPAbet phone -> viseme(s). Two entries = a diphthong: the first takes 60 % of the time.
const PHONE_VISEMES = {
  AA: ["aa"], AE: ["ae"], AH: ["ah"], AX: ["schwa"], AO: ["oh"], AW: ["aa", "oh"], AY: ["aa", "ih"], EH: ["eh"], ER: ["er"], EY: ["eh", "ih"], IH: ["ih"], IY: ["iy"], OW: ["oh", "uw"], OY: ["oh", "ih"], UH: ["uw"], UW: ["uw"],
  P: ["pbm"], B: ["pbm"], M: ["pbm"], F: ["fv"], V: ["fv"], TH: ["th"], DH: ["th"], T: ["td"], D: ["td"], N: ["n"], NG: ["kg"], K: ["kg"], G: ["kg"], L: ["l"], R: ["r"], S: ["sz"], Z: ["sz"], SH: ["sh"], ZH: ["sh"], CH: ["sh"], JH: ["sh"],
  W: ["uw"], Y: ["ih"], HH: []
};
const VOWEL_PHONE = p => "AEIOU".includes(p[0]);

// ---- English letters -> phones. A RULE SET, not a dictionary: digraphs, silent e, a few endings and ~120 common irregular words. Good enough to put the
// lips in the right class most of the time (closed / spread / round / open); a server-side pronouncing dictionary would do better (see docs/CONTRACT.md).
const G2P_WORDS = (() => {
  const s = "a:AH|i:AY|the:DH AH|to:T UW|of:AH V|you:Y UW|your:Y AO R|are:AA R|was:W AH Z|were:W ER|one:W AH N|once:W AH N S|two:T UW|do:D UW|does:D AH Z|done:D AH N|said:S EH D|says:S EH Z|have:HH AE V|give:G IH V|live:L IH V|love:L AH V|" +
    "come:K AH M|some:S AH M|what:W AH T|who:HH UW|where:W EH R|there:DH EH R|their:DH EH R|they:DH EY|would:W UH D|could:K UH D|should:SH UH D|know:N OW|show:SH OW|low:L OW|slow:S L OW|own:OW N|only:OW N L IY|be:B IY|me:M IY|he:HH IY|" +
    "she:SH IY|we:W IY|people:P IY P AH L|because:B IH K AH Z|been:B IH N|any:EH N IY|many:M EH N IY|other:AH DH ER|another:AH N AH DH ER|mother:M AH DH ER|water:W AO T ER|want:W AA N T|put:P UH T|good:G UH D|look:L UH K|great:G R EY T|" +
    "again:AH G EH N|get:G EH T|girl:G ER L|here:HH IY R|hi:HH AY|yes:Y EH S|okay:OW K EY|ok:OW K EY|today:T AH D EY|into:IH N T UW|through:TH R UW|though:DH OW|thought:TH AO T|enough:IH N AH F|sure:SH UH R|eye:AY|eyes:AY Z|move:M UW V|" +
    "world:W ER L D|work:W ER K|word:W ER D|friend:F R EH N D|our:AW R|hour:AW R|from:F R AH M|is:IH Z|his:HH IH Z|as:AE Z|has:HH AE Z|us:AH S|very:V EH R IY|every:EH V R IY|about:AH B AW T|above:AH B AH V|over:OW V ER|open:OW P AH N|" +
    "even:IY V AH N|woman:W UH M AH N|women:W IH M AH N|won:W AH N|son:S AH N|month:M AH N TH|front:F R AH N T|nothing:N AH TH IH NG|something:S AH M TH IH NG|answer:AE N S ER|listen:L IH S AH N|often:AO F AH N|talk:T AO K|walk:W AO K|" +
    "half:HH AE F|idea:AY D IY AH|use:Y UW Z|few:F Y UW|new:N UW|view:V Y UW|beautiful:B Y UW T AH F AH L|busy:B IH Z IY|business:B IH Z N AH S|minute:M IH N AH T|heart:HH AA R T|hear:HH IY R|heard:HH ER D|learn:L ER N|early:ER L IY|" +
    "earth:ER TH|year:Y IY R|real:R IY L|really:R IY L IY|ready:R EH D IY|head:HH EH D|read:R IY D|dead:D EH D|bread:B R EH D|break:B R EY K|you're:Y UH R|i'm:AY M|it's:IH T S|don't:D OW N T|can't:K AE N T|that's:DH AE T S|" +
    "i'll:AY L|we're:W IY R|they're:DH EH R|isn't:IH Z AH N T|doesn't:D AH Z AH N T|won't:W OW N T|i've:AY V|let's:L EH T S|what's:W AH T S|hello:HH AH L OW|everyone:EH V R IY W AH N|everybody:EH V R IY B AA D IY|anyone:EH N IY W AH N|someone:S AH M W AH N|iris:AY R IH S|worked:W ER K T|works:W ER K S|working:W ER K IH NG|avatar:AE V AH T AA R";
  const m = {}; for (const e of s.split("|")) { const i = e.indexOf(":"); m[e.slice(0, i)] = e.slice(i + 1).split(" "); } return m;
})();
const isV = c => c !== undefined && "aeiouy".includes(c), isC = c => c !== undefined && c >= "a" && c <= "z" && !"aeiou".includes(c);
const G2P_MULTI = [                                               // [letters, phones, condition(word, i)?]; first match wins, longest first
  ["tion", ["SH", "AH", "N"], (w, i) => i > 0], ["sion", ["ZH", "AH", "N"], (w, i) => i > 0], ["ture", ["CH", "ER"], (w, i) => i > 0 && i + 4 === w.length],
  ["ough", ["AO"]], ["augh", ["AO"]], ["eigh", ["EY"]], ["igh", ["AY"]], ["tch", ["CH"]], ["sch", ["S", "K"]], ["dge", ["JH"]],
  ["ch", ["CH"]], ["sh", ["SH"]], ["th", ["TH"]], ["ph", ["F"]], ["wh", ["W"]], ["ck", ["K"]], ["qu", ["K", "W"]], ["ng", ["NG"]], ["kn", ["N"], (w, i) => i === 0], ["wr", ["R"], (w, i) => i === 0],
  ["gh", [], (w, i) => i > 0], ["mb", ["M"], (w, i) => i + 2 === w.length],
  ["oo", ["UW"]], ["ee", ["IY"]], ["ea", ["IY"]], ["ai", ["EY"]], ["ay", ["EY"]], ["oa", ["OW"]], ["oi", ["OY"]], ["oy", ["OY"]], ["ou", ["AW"]],
  ["ow", ["OW"], (w, i) => i + 2 === w.length && w.length > 4], ["ow", ["AW"]], ["au", ["AO"]], ["aw", ["AO"]], ["ew", ["UW"]], ["ue", ["UW"], (w, i) => i + 2 === w.length],
  ["ie", ["AY"], (w, i) => i + 2 === w.length && w.length <= 3], ["ie", ["IY"]], ["ey", ["IY"], (w, i) => i + 2 === w.length && w.length > 3], ["ei", ["EY"]], ["ey", ["EY"]],
  ["ar", ["AA", "R"], (w, i) => !isV(w[i + 2])], ["or", ["AO", "R"], (w, i) => !isV(w[i + 2])], ["er", ["ER"], (w, i) => !isV(w[i + 2])], ["ir", ["ER"], (w, i) => !isV(w[i + 2])], ["ur", ["ER"], (w, i) => !isV(w[i + 2])],
  ["le", ["AH", "L"], (w, i) => i + 2 === w.length && i > 0 && isC(w[i - 1])]
];
const G2P_LONG = { a: "EY", e: "IY", i: "AY", y: "AY", o: "OW", u: "UW" }, G2P_SHORT = { a: "AE", e: "EH", i: "IH", o: "AA", u: "AH" };
const G2P_CONS = { b: "B", d: "D", f: "F", h: "HH", j: "JH", k: "K", l: "L", m: "M", n: "N", p: "P", q: "K", r: "R", t: "T", v: "V", w: "W", z: "Z" };
// word (lower case letters and apostrophes) -> [{ p: phone, a: first letter index, b: index after its last letter }]
function rulesToPhones(w) {
  const dict = G2P_WORDS[w]; if (dict) { const wt = dict.map(p => (VOWEL_PHONE(p) ? (PHONE_VISEMES[p].length > 1 ? 2 : 1.6) : 1)), tot = wt.reduce((x, y) => x + y, 0); let u = 0; return dict.map((p, k) => { const a = u; u += wt[k] * w.length / tot; return { p, a, b: k + 1 === dict.length ? w.length : u }; }); }   // irregular word: its sounds share the word's time, vowels longer
  const out = [], n = w.length, push = (ps, a, b) => ps.forEach((p, k) => out.push({ p, a: a + k * (b - a) / ps.length, b: a + (k + 1) * (b - a) / ps.length }));
  const otherVowel = k => { for (let j = 0; j < n; j++) if (j !== k && isV(w[j])) return true; return false; };
  for (let i = 0; i < n;) {
    const c = w[i], nx = w[i + 1];
    if (c === "'") { i++; continue; }
    const m = G2P_MULTI.find(r => w.startsWith(r[0], i) && (!r[2] || r[2](w, i)));
    if (m) { push(m[1], i, i + m[0].length); i += m[0].length; continue; }
    if (isC(c) && c !== "y" && nx === c) { push([G2P_CONS[c] || (c === "s" ? "S" : c === "c" ? "K" : c === "g" ? "G" : "K")], i, i + 2); i += 2; continue; }   // doubled consonant: one sound
    if ("aeiou".includes(c) || (c === "y" && i > 0)) {
      const magic = isC(w[i + 1]) && !"xw".includes(w[i + 1]) && w[i + 2] === "e" && (i + 3 === n || (i + 4 === n && w[i + 3] === "s")) && !(i > 0 && "aeiou".includes(w[i - 1]));
      let p;
      if (magic) p = G2P_LONG[c];
      else if (c === "y") p = i + 1 === n ? (otherVowel(i) ? "IY" : "AY") : "IH";
      else if (c === "e" && i + 1 === n) p = otherVowel(i) ? null : "IY";                                   // silent final e
      else if (c === "e" && i + 2 === n && nx === "d" && n > 3 && !"td".includes(w[i - 1]) && isC(w[i - 1])) p = null;   // walked, named
      else if (c === "e" && i + 2 === n && nx === "s" && n > 3 && isC(w[i - 1]) && !"sxz".includes(w[i - 1]) && !(w[i - 1] === "h" && "cs".includes(w[i - 2]))) p = null;   // makes, times
      else if (c === "a" && (w.startsWith("ll", i + 1) || w.startsWith("lk", i + 1) || w.startsWith("lt", i + 1))) p = "AO";
      else if (c === "a" && i + 1 === n && n > 1) p = "AH";
      else if (c === "a" && w[i - 1] === "w") p = "AA";
      else if (c === "i" && i + 3 === n && (w.startsWith("nd", i + 1) || w.startsWith("ld", i + 1))) p = "AY";
      else if (c === "i" && i + 1 === n) p = "AY";
      else if (c === "o" && (i + 1 === n || (i + 3 === n && (w.startsWith("ld", i + 1) || w.startsWith("st", i + 1))))) p = "OW";
      else p = G2P_SHORT[c];
      if (p) push([p], i, i + 1); i++; continue;
    }
    let p;
    if (c === "y") p = "Y"; else if (c === "c") p = "eiy".includes(nx) && nx !== undefined ? "S" : "K"; else if (c === "g") p = "eiy".includes(nx) && nx !== undefined && i + 2 < n ? "JH" : "G";
    else if (c === "s") p = i > 0 && isV(w[i - 1]) && (isV(nx) && "aeiou".includes(nx)) ? "Z" : "S"; else if (c === "x") { push(["K", "S"], i, i + 1); i++; continue; } else p = G2P_CONS[c];
    if (p) push([p], i, i + 1); i++;
  }
  return out;
}
// ---- optional pronouncing dictionary: word -> ARPAbet phones WITH lexical stress (0 unstressed, 1 primary, 2 secondary on the vowels).
// The data is the CMU Pronouncing Dictionary (BSD-2-Clause, Carnegie Mellon University), installed by scripts/get_cmudict.sh as
// viewer/lexicon/cmudict.txt (one "word PH0 PH1 ..." per line, first pronunciation only). Nothing needs it: without it the letter rules above are used.
// Pages call OpenAvatarViewer.loadLexicon() once (a 404 just leaves the rules in charge); tests can hand setLexicon() a few lines of text.
const LEXICON_URL = (() => { try { return new URL("lexicon/cmudict.txt", import.meta.url).href; } catch { return "lexicon/cmudict.txt"; } })();
let LEXICON = null, LEXICON_SRC = null, lexiconLoading = null;
// grammatical words are said without stress in running speech whatever the dictionary's citation form says: their reductions (the -> DH AH0) are kept,
// their primary stress is not (no extra opening, no nod); they fall back to the length rule like words from the letter rules
const FUNCTION_WORDS = new Set(("a an the and or but nor of to in on at by for from with as into onto than then if so not is am are was were be been being it its it's i i'm i'll i've i'd " +
  "you you're you'll he he's she she's we we're we'll they they're me him her us them my your his our their this that these those do does did have has had will would can could shall should may might must there what which who").split(" "));
function setLexicon(src, label) {                                 // text ("word PH PH" lines; # or ;;; comments) | Map | {word: "PH PH"} | null
  if (src == null) { LEXICON = null; LEXICON_SRC = null; return 0; }
  const m = new Map();
  if (typeof src === "string") {
    for (const line of src.split("\n")) {
      if (!line || line[0] === "#" || line.startsWith(";;;")) continue;
      const i = line.indexOf(" "); if (i <= 0) continue;
      let w = line.slice(0, i).toLowerCase(); const p = line.slice(i + 1).split("#")[0].trim(); if (!p) continue;
      const k = w.indexOf("("); if (k > 0) { w = w.slice(0, k); if (m.has(w)) continue; }      // CMU style "word(2)": alternative pronunciations, the first one wins
      if (!m.has(w)) m.set(w, p);
    }
  } else for (const [w, p] of (src instanceof Map ? src : Object.entries(src))) m.set(String(w).toLowerCase(), Array.isArray(p) ? p.join(" ") : String(p));
  LEXICON = m.size ? m : null; LEXICON_SRC = m.size ? (label || "set") : null; return m.size;
}
async function loadLexicon(url = LEXICON_URL) {                   // resolves to the number of words (0 when the file is not installed); never rejects
  if (LEXICON && LEXICON_SRC === url) return LEXICON.size;
  if (lexiconLoading && lexiconLoading.url === url) return lexiconLoading.p;
  const p = (async () => { try { const r = await fetch(url); if (!r.ok) return 0; return setLexicon(await r.text(), url); } catch { return 0; } finally { lexiconLoading = null; } })();
  lexiconLoading = { url, p }; return p;
}
const lexiconInfo = () => ({ words: LEXICON ? LEXICON.size : 0, source: LEXICON_SRC });
// which letters a dictionary phone came from: the letter rules give every phone of their own reading a letter span; the dictionary's phones are matched to
// those (edit distance, similar phones cheap), matched phones take their span and the others share the time between their matched neighbours
const PH_GROUP = { S: "s", Z: "s", SH: "s", ZH: "s", CH: "s", JH: "s", T: "t", D: "t", TH: "t", DH: "t", K: "k", G: "k", NG: "n", N: "n", M: "m", P: "p", B: "p", F: "f", V: "f", L: "l", R: "r", ER: "r", W: "w", Y: "y", HH: "h" };
const phCost = (x, y) => (x === y ? 0 : VOWEL_PHONE(x) && VOWEL_PHONE(y) ? 0.4 : PH_GROUP[x] && PH_GROUP[x] === PH_GROUP[y] ? 0.5 : 1);
function spansFromRules(w, P) {
  const R = rulesToPhones(w), n = P.length, m = R.length, D = [], INDEL = 0.7;
  for (let i = 0; i <= n; i++) { D.push(new Float64Array(m + 1)); for (let j = 0; j <= m; j++) D[i][j] = i === 0 ? j * INDEL : j === 0 ? i * INDEL : Math.min(D[i - 1][j - 1] + phCost(P[i - 1], R[j - 1].p), D[i - 1][j] + INDEL, D[i][j - 1] + INDEL); }
  const a = new Array(n).fill(null), b = new Array(n).fill(null);
  for (let i = n, j = m; i > 0 && j > 0;) {
    if (D[i][j] === D[i - 1][j - 1] + phCost(P[i - 1], R[j - 1].p)) { a[i - 1] = R[j - 1].a; b[i - 1] = R[j - 1].b; i--; j--; }
    else if (D[i][j] === D[i - 1][j] + INDEL) i--; else j--;
  }
  for (let i = 0; i < n;) {                                       // runs without a match share the room between their neighbours (or split the previous phone's letters)
    if (a[i] !== null) { i++; continue; }
    let j = i; while (j < n && a[j] === null) j++;
    let lo = i > 0 ? (b[i - 1] > a[i - 1] ? b[i - 1] : a[i - 1]) : 0, hi = j < n ? a[j] : w.length;
    if (hi - lo < 0.2 * (j - i) && i > 0) { lo = a[i - 1] + 0.5 * (b[i - 1] - a[i - 1]); b[i - 1] = lo; }   // no room: the previous phone gives up half its letters
    if (hi <= lo) hi = lo + 0.01 * (j - i);
    for (let k = i; k < j; k++) { a[k] = lo + (hi - lo) * (k - i) / (j - i); b[k] = lo + (hi - lo) * (k - i + 1) / (j - i); }
    i = j;
  }
  for (let k = 0; k < n; k++) { if (k > 0 && a[k] < b[k - 1]) a[k] = b[k - 1]; if (b[k] <= a[k]) b[k] = a[k] + 0.01; }
  b[n - 1] = Math.max(b[n - 1], w.length);
  return P.map((p, k) => ({ p, a: a[k], b: b[k] }));
}
// word -> [{ p, a, b, s? }]: the dictionary's phones when the word is in it (s = lexical stress of a vowel), else the letter rules (no s)
function wordToPhones(w, opts) {
  const e = LEXICON && !(opts && opts.lexicon === false) ? LEXICON.get(w) : null;
  if (!e) return rulesToPhones(w);
  let raw = e.split(/\s+/).filter(Boolean), P = raw.map(x => x.replace(/[0-9]/g, "")); const fw = FUNCTION_WORDS.has(w);
  if (P.some(p => !PHONE_VISEMES[p])) return rulesToPhones(w);
  const irr = G2P_WORDS[w];                                        // a word on the hand-made list (chosen for the voices in use) keeps its phones; the dictionary adds the stress when the vowels line up
  if (irr && irr.join(" ") !== P.join(" ")) {
    const sv = raw.filter(x => /[0-2]$/.test(x)).map(x => x.slice(-1)), iv = irr.filter(VOWEL_PHONE);
    if (sv.length !== iv.length) return rulesToPhones(w);
    let k = 0; raw = irr.map(p => (VOWEL_PHONE(p) ? p + sv[k++] : p)); P = irr.slice();
  }
  return spansFromRules(w, P).map((q, k) => { const d = raw[k].match(/[0-2]$/); if (d && VOWEL_PHONE(q.p)) { const s = +d[0]; if (!fw || s === 0) q.s = s; else if (q.p === "AH") q.s = 0; } return q; });   // function words: stressed AH is reduced too (of, but, from)
}
// character timings -> [{ phone, start, end, stress? }]. A phone's time is that of its letters; the last sound of a word also takes a short space after it
// (ElevenLabs puts the tail of the word there). Digits get a neutral vowel each so that spoken numbers still move the mouth.
function timingsToPhones(timings) {
  const T = (timings || []).map(t => ({ c: String(t.char ?? "").toLowerCase(), start: +t.start, end: +t.end })).filter(t => Number.isFinite(t.start) && Number.isFinite(t.end)), out = [];
  const letter = c => c.length === 1 && ((c >= "a" && c <= "z") || c === "'" || c === "\u2019");
  for (let i = 0; i < T.length;) {
    if (T[i].c.length === 1 && T[i].c >= "0" && T[i].c <= "9") { out.push({ phone: "AH", start: T[i].start, end: T[i].end }); i++; continue; }
    if (T[i].c && ",.;:!?".includes(T[i].c)) { out.push({ phone: "SIL", start: T[i].start, end: T[i].end }); i++; continue; }
    if (!letter(T[i].c) || T[i].c === "'" || T[i].c === "\u2019") { i++; continue; }
    let j = i; while (j < T.length && letter(T[j].c)) j++;
    const w = T.slice(i, j).map(t => (t.c === "\u2019" ? "'" : t.c)).join("").replace(/'+$/, ""), n = w.length, at = x => { const k = Math.min(n - 1, Math.floor(x)), f = x - k; return T[i + k].start + f * (T[i + k].end - T[i + k].start); };
    const ph = wordToPhones(w), first = out.length;
    for (const q of ph) { const o = { phone: q.p, start: at(q.a), end: q.b >= n ? T[i + n - 1].end : at(q.b) }; if (q.s !== undefined) o.stress = q.s; out.push(o); }
    let sp = 0, k = j; while (k < T.length && /^\s$/.test(T[k].c)) { sp += T[k].end - T[k].start; k++; }
    if (out.length > first && sp > 0 && sp <= 0.15 && k < T.length && letter(T[k].c)) out[out.length - 1].end += VOWEL_PHONE(out[out.length - 1].phone) ? sp : sp / 2;
    i = j;
  }
  return out;
}
function phonesToTrack(phones) {                                  // [{ phone, start, end, stress? }] -> viseme items (diphthongs split 60 / 40); p and b close BEFORE their burst
  const out = [];                                                 // a vowel's stress comes from `stress` or a digit on the phone ("AY1"); unstressed AH is the reduced vowel (schwa)
  for (const q of phones || []) {
    const raw = String(q.phone || "").toUpperCase(), p = raw.replace(/[0-9]/g, ""); if (p === "SIL" || p === "HH") { out.push({ viseme: p === "SIL" ? "sil" : "hh", start: q.start, end: q.end }); continue; }   // hh: no shape of its own, but sound (not a pause)
    const dg = raw.match(/([0-2])$/), st = VOWEL_PHONE(p) ? (q.stress === 0 || q.stress === 1 || q.stress === 2 ? q.stress : dg ? +dg[1] : undefined) : undefined;
    const vs = p === "AH" && st === 0 ? ["schwa"] : PHONE_VISEMES[p]; if (!vs || !vs.length) continue;
    const S = st === undefined ? {} : { stress: st };
    if (vs.length === 1) out.push({ viseme: vs[0], start: q.start, end: q.end, stop: p === "P" || p === "B", ...S });
    else { const m = q.start + 0.6 * (q.end - q.start); out.push({ viseme: vs[0], start: q.start, end: m, ...S }, { viseme: vs[1], start: m, end: q.end, glide: true, ...S }); }
  }
  return out;
}
const lettersToVisemes = timings => phonesToTrack(timingsToPhones(timings)).filter(v => v.viseme !== "sil" && v.viseme !== "hh").map(({ viseme, start, end }) => ({ viseme, start, end }));

function timedDriver(env) {
  let src = null, events = [], speaking = false, level = 0, fallback = null, hasTrack = false, lastEnd = 0, beats = [], settles = [], beat = 0, settle = 0;
  const fb = () => { if (!fallback) { fallback = loudnessDriver(env); if (src) fallback.attach(src); } return fallback; };   // no track (or no audio clock): the loudness driver
  const wOf = w => { const arr = []; for (const k of Object.keys(w || {})) { const i = env.index(k); if (i >= 0) arr.push([i, +w[k]]); } return arr; };
  function compile(items) {
    const T = env.settings().timed || {}, minVis = T.minVisible ?? 0.07, tongueMin = T.tongueMin ?? 0.05;
    let L = []; for (const it of items || []) { if (!Number.isFinite(it.start) || !Number.isFinite(it.end)) continue; if (it.phone && !it.viseme && !it.weights) L.push(...phonesToTrack([it])); else L.push({ ...it }); }
    L.sort((a, b) => a.start - b.start);
    const ev = [], sil = [], hh = []; beats = []; settles = [];
    for (const it of L) {
      if (!it.weights && it.viseme === "sil") { sil.push(it); continue; }
      if (!it.weights && it.viseme === "hh") { hh.push(it); continue; }
      if (!it.weights && (!it.viseme || !VISEME_WEIGHTS[it.viseme])) continue;
      const dyn = it.weights ? [1, 0.07, 0.1] : VISEME_DYN[it.viseme] || [1, 0.07, 0.1]; let s = it.start, e = it.end;
      if (it.viseme === "pbm") { if (it.stop) { s = it.start - 0.05; e = it.start + 0.015; } else { s = it.start + 0.02; e = Math.max(s + 0.04, it.end - 0.01); } }   // p, b: shut during the silent closure before the letter's burst; m: for the hum (half-way shut already reads as shut, so the ramp ends 20 ms into it)
      else { if (!it.weights && TONGUE[it.viseme] && e - s < tongueMin) continue; if (e - s < minVis) e = s + minVis; }
      // stress: from the track (dictionary words, phones with digits) when known: primary opens more and gets a full jaw floor even when short, unstressed opens
      // less with half the floor; without it (letter rules) the longer vowel counts as the stressed one
      const ps = !it.weights && VOWEL_VIS[it.viseme] && (it.stress === 0 || it.stress === 1 || it.stress === 2) ? it.stress : undefined, long = it.end - it.start >= STRESS_S;
      const stressed = ps === undefined ? long : ps === 1 || (ps === 2 && long), sg = ps === 1 ? T.stressOpen ?? 1.12 : ps === 0 && it.viseme !== "schwa" ? T.unstressedOpen ?? 0.75 : 1;   // schwa is already the reduced shape
      const fl = !it.weights && VOWEL_FLOOR[it.viseme] ? VOWEL_FLOOR[it.viseme] * (stressed ? 1 : 0.5) * (ps === 1 ? sg : 1) : 0;
      const w0 = wOf(it.weights || VISEME_WEIGHTS[it.viseme]);
      ev.push({ fl, st: stressed, ps, glide: !!it.glide, w: sg === 1 ? w0 : w0.map(([i, v]) => [i, v * sg]), d: dyn[0], atk: dyn[1], rel: dyn[2], s, s0: s, e, end0: e, vis: it.viseme, on: it.viseme === "pbm" ? 0 : Math.min(0.04, 0.4 * (e - s)), plosive: !it.weights && it.viseme === "pbm" });   // on: the shape is full a little after the sound starts (the sound's energy peaks later still)
    }
    ev.sort((a, b) => a.s - b.s);
    for (let i = 0; i < ev.length; i++) {                         // gaps: short ones hold the shape, a pause relaxes slowly and the next sound opens early
      const a = ev[i], b = ev[i + 1], g = b ? b.s - a.e : Infinity;
      const marked = g >= 0.08 && b && sil.some(q => q.start < b.s && q.end > a.e);        // a comma or full stop in the gap: a phrase boundary for the head even when short
      if (g >= 0.15 || marked || !b) { a.pend = true; if (b) b.pstart = true; }
      // the mouth: a gap under 0.15 s holds the shape, a comma too: its letters often still carry the vowel ("Hello, I": the o sounds to 40 ms before the I) and
      // closing for it shut the lips for a frame or two (audit4 item 10); at a comma the shape is held only until the next sound starts to form, so that one takes
      // over fully (a hold to its start blunted the "I" after it). Longer gaps relax to rest; their ramps are shortened so the rest lasts at least PAUSE_REST.
      const gm = b ? g - hh.reduce((m, q) => m + Math.max(0, Math.min(q.end, b.s) - Math.max(q.start, a.e)), 0) : g;   // an h in the gap is sound: the mouth holds through it ("the house")
      if (g > 0 && gm < 0.15) { if (!a.plosive) a.e = marked ? Math.max(a.e, b.s + b.on - b.atk) : b.s; } else if (g >= 0.15 && b) {
        let rl = a.plosive ? a.rel : Math.max(a.rel, 0.12), at = b.plosive ? b.atk : Math.max(b.atk, g >= 0.25 ? 0.11 : 0.09); const over = rl + at - b.on + PAUSE_REST - g;
        if (over > 0) { const f = Math.max(0.5, 1 - over / (rl + at)); if (!a.plosive) rl *= f; if (!b.plosive) at *= f; }
        if (!a.plosive) a.rel = rl; if (!b.plosive) b.atk = at;
      } else if (!b && !a.plosive) a.rel = Math.max(a.rel, 0.12);
      if (i === 0 && !a.plosive) a.atk = Math.max(a.atk, 0.11);
    }
    // two lip closures around a short vowel (map, people, maybe at speed): their ramps overlapped and the lips stayed shut through the vowel (lipsync4).
    // Give the vowel at least 40 ms: shorten the release of the first and the attack of the second closure (to 15 ms), then their shut parts (to 30 ms).
    let lastP = null, vow = false;
    for (const e of ev) {
      if (!e.plosive) { if (VOWEL_VIS[e.vis]) vow = true; continue; }
      if (lastP && vow) {
        let room = e.s - lastP.e; const need = 0.04 + lastP.rel + e.atk;
        if (room < need) { const r = Math.max(0.015, (room - 0.04) / 2); lastP.rel = Math.min(lastP.rel, r); e.atk = Math.min(e.atk, r); }
        const short = 0.04 + lastP.rel + e.atk - room;
        if (short > 0) { const c1 = Math.min(short / 2, Math.max(0, lastP.e - lastP.s - 0.03)); lastP.e -= c1; room += c1; const c2 = Math.min(short - c1, Math.max(0, e.e - e.s - 0.03)); e.s += c2; }
      }
      lastP = e; vow = false;
    }
    // a short vowel right before a lip closure ("happy" after a pause, timings whose p starts at the closure): the closure's ramp began before the vowel had
    // opened, so it showed for one frame (audit4 item 10). Give the vowel VOWEL_OPEN_MIN fully open before the lips start to close, in this order:
    // (1) reach the full shape sooner (its `on` delay; a pause's slow attack shortened to an ordinary vowel's), (2) open earlier, into the h, the pause or the
    // consonant before it (at most VOWEL_PULL_MAX, never into a closure: the mouth only gets further ahead of the sound), (3) start the closure a little later
    // (at most CLOSE_DELAY_MAX, shut for >= 35 ms; it still ends where it did, so the lips are shut before the burst and never late).
    const leadS = env.settings().lead ?? 0.02;
    for (let i = 1; i < ev.length; i++) {
      const P = ev[i], V = ev[i - 1]; if (!P.plosive || V.plosive || !VOWEL_VIS[V.vis]) continue;
      let need = VOWEL_OPEN_MIN - ((P.s - P.atk) - (V.s + V.on - leadS)); if (need <= 0) continue;
      const x = Math.min(need, V.on); V.on -= x; V.atk = Math.max(Math.min(V.atk, 0.08), V.atk - x); need -= x; if (need <= 0) continue;   // the opening's speed stays that of an ordinary vowel (>= 80 ms)
      const Q = ev[i - 2], lo = !Q ? -Infinity : Q.plosive ? Q.e + Q.rel + leadS - V.on : V.pstart ? -Infinity : Q.s0 + 0.5 * (Q.end0 - Q.s0);
      const pull = Math.max(0, Math.min(need, VOWEL_PULL_MAX, V.s - lo)); V.s -= pull; need -= pull;
      if (need > 0) P.s += Math.min(need, CLOSE_DELAY_MAX, Math.max(0, P.e - P.s - 0.035));
    }
    events = ev; lastEnd = ev.reduce((m, e) => Math.max(m, e.e + e.rel), 0);
    // head beats: the first vowel of a phrase (1) and stressed vowels (0.55), at most one per 0.3 s; settles at phrase ends (where the sound ends, not the hold).
    // With lexical stress: the phrase's nod waits for its first stressed vowel (up to 0.4 s), every primary stress nods, other vowels do not; without it
    // (letter rules) the long open vowels nod.
    beats = []; settles = []; let first = true, t0 = 0;
    if (ev.length) ev[0].pstart = true;
    for (const e of ev) {
      if (e.pstart) { first = true; t0 = e.s0; }
      if (!e.plosive && VOWEL_FLOOR_ANY(e) && !e.glide) { if (first && e.ps === 0 && e.s0 - t0 < 0.4) continue;
        const a = first ? 1 : e.ps !== undefined ? (e.ps === 1 ? 0.55 : 0) : e.st && e.fl > 0 ? 0.55 : 0; first = false;
        if (a > 0) { const last = beats[beats.length - 1]; if (last && e.s0 - last.t < 0.3) { if (a > last.a) { last.t = e.s0; last.a = a; } } else beats.push({ t: e.s0, a }); } }   // the sound's own start (s0), not an opening moved earlier
      if (e.pend) settles.push({ t: e.end0, next: Infinity });
    }
    for (const st of settles) { const nx = ev.find(e => e.pstart && e.s0 > st.t); if (nx) st.next = nx.s0; }
  }
  const acc = new Float32Array(Math.max(1, env.Ks)), aix = articulationIndex(env);
  return {
    name: "timed",
    attach(s) { src = s; if (fallback) fallback.attach(s); }, detach() { src = null; if (fallback) fallback.detach(); speaking = false; level = 0; },
    get speaking() { return hasTrack ? speaking : (fallback ? fallback.speaking : false); }, get level() { return hasTrack ? level : (fallback ? fallback.level : 0); },
    get beat() { return hasTrack ? beat : (fallback ? fallback.beat : 0); }, get settle() { return hasTrack ? settle : (fallback ? fallback.settle : 0); },
    // data: { track } or { phones: [{phone,start,end}] } or { timings: [{char,start,end}] } or null (falls back to the loudness driver)
    setData(d) {
      hasTrack = false; events = [];
      if (d && Array.isArray(d.track)) { compile(d.track); hasTrack = true; }
      else if (d && Array.isArray(d.phones) && d.phones.length) { compile(phonesToTrack(d.phones)); hasTrack = true; }
      else if (d && Array.isArray(d.timings) && d.timings.length) { compile(phonesToTrack(timingsToPhones(d.timings))); hasTrack = true; }
      if (!hasTrack && src) fb();
    },
    update(dt, out) {
      if (!hasTrack || !src || !src.hasClock) { fb().update(dt, out); return; }
      const S = env.settings(), T = S.timed || {}; out.fill(0); speaking = false; level = 0; beat = 0; settle = 0;
      if (!src.playing()) return;
      const lead = S.lead ?? 0.02, tv = src.mediaTime() + lead + (T.timeOffset || 0), ka = (T.attack ?? 0.05) / 0.05, kr = (T.release ?? 0.1) / 0.1;
      let sum = 0, top = 0, shut = 0, floor = 0; acc.fill(0);
      for (const e of events) {
        const t = e.plosive ? tv - lead : tv;                     // closures are not run ahead: lips that shut before the vowel has ended are seen at once
        const a = e.atk * (e.plosive ? 1 : ka), r = e.rel * (e.plosive ? 1 : kr), p = e.s + e.on; if (t <= p - a) { if (e.s - t > 1) break; continue; } if (t >= e.e + r) continue;
        const v = t < p ? ss((t - (p - a)) / a) : t <= e.e ? 1 : 1 - ss((t - e.e) / r);
        if (e.plosive) { if (v > shut) shut = v; continue; }
        const dv = e.d * v; sum += dv; if (v > top) top = v; for (const [i, w] of e.w) acc[i] += dv * w; if (e.fl * v > floor) floor = e.fl * v;
      }
      const k = sum > 1e-9 ? top * (1 - shut) / sum : 0, pi = env.index("press");   // p/b/m: the blend of everything else is faded out and the lips press together
      for (let i = 0; i < out.length; i++) out[i] = clamp(acc[i] * k, 0, 1);
      const art = S.articulation ?? 1; articulate(out, art, aix, floor * art * (1 - shut), openGainOf(S, aix), 1 - shut);
      if (pi >= 0 && shut > 0) out[pi] = Math.max(out[pi], shut * (VISEME_WEIGHTS.pbm.press ?? 1));
      top = Math.max(top, shut); speaking = top > 0.05; level = top;
      let nb = 1; for (const b of beats) { const u = tv - b.t; if (u <= -BEAT_PRE) break; nb *= 1 - b.a * beatPulse(u); } beat = 1 - nb;   // overlapping nods add smoothly (a max had a kink where one overtook the other)
      for (const st of settles) { const p = settlePulse(tv - st.t) * (1 - ss((tv - st.next + 0.15) / 0.3)); if (p > settle) settle = p; }   // the next phrase takes over
    },
    debug: () => ({ events: events.length, lastEnd, hasTrack })
  };
}

// ------------------------------------------------------------------ the first drivers, kept for comparison as "loudness-v1" and "timed-v1"
// Unchanged behaviour with their original settings (one-pole smoothing 45 ms; timed attack 50 ms / release 100 ms): late by the smoothing, sums of shapes.
const V1_TIMED = { attack: 0.05, release: 0.10, timeOffset: 0, minVisible: 0.07, tongueMin: 0.07 };
function loudnessDriverV1(env) {
  let src = null, peak = -20; const tmp = new Float32Array(NB), cur = new Float32Array(env.Ks), st = new Float32Array(env.Ks); let speaking = false, level = 0;
  const set = (n, v) => { const i = env.index(n); if (i >= 0) cur[i] = clamp(v, 0, 1); };
  return {
    name: "loudness-v1",
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
      const a = dt > 0 ? 1 - Math.exp(-dt / 0.045) : 1; for (let j = 0; j < st.length; j++) st[j] += a * (cur[j] - st[j]);
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
const VISEME_WEIGHTS_V1 = {                                          // UNTUNED: controls named as in the rig; unknown controls are ignored
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
function lettersToVisemesV1(timings) {
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

function timedDriverV1(env) {
  let src = null, events = [], cur = new Float32Array(env.Ks), target = new Float32Array(env.Ks), lastEnd = 0, speaking = false, fallback = null, hasTrack = false;
  const fb = () => { if (!fallback) { fallback = loudnessDriverV1(env); if (src) fallback.attach(src); } return fallback; };   // no track (or no audio clock): the loudness driver
  const wOf = (item) => { const w = item.weights || VISEME_WEIGHTS_V1[item.viseme] || {}, arr = []; for (const k of Object.keys(w)) { const i = env.index(k); if (i >= 0) arr.push([i, w[k]]); } return arr; };
  function compile(items) {
    const T = V1_TIMED, L = (items || []).filter(it => Number.isFinite(it.start) && Number.isFinite(it.end)).map(it => ({ ...it })).sort((a, b) => a.start - b.start);
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
    name: "timed-v1",
    attach(s) { src = s; if (fallback) fallback.attach(s); cur.fill(0); }, detach() { src = null; if (fallback) fallback.detach(); cur.fill(0); speaking = false; },
    get speaking() { return hasTrack ? speaking : (fallback ? fallback.speaking : false); }, get level() { return fallback ? fallback.level : 0; },
    // data: { track } or { timings: [{char,start,end}] } or null (falls back to the loudness driver)
    setData(d) {
      hasTrack = false; events = [];
      if (d && Array.isArray(d.track)) { compile(d.track); hasTrack = true; }
      else if (d && Array.isArray(d.timings) && d.timings.length) { compile(lettersToVisemesV1(d.timings)); hasTrack = true; }
      if (!hasTrack && src) fb();
    },
    update(dt, out) {
      if (!hasTrack || !src || !src.hasClock) { fb().update(dt, out); return; }
      const T = { ...V1_TIMED, timeOffset: (env.settings().timed || {}).timeOffset || 0 };
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
const driverRegistry = { loudness: loudnessDriver, timed: timedDriver, "loudness-v1": loudnessDriverV1, "timed-v1": timedDriverV1 };

// ------------------------------------------------------------------ lifelike (spikes/lf_core, 2026-11-25): head range, eyes, motion layers, clip words
// Everything here is OFF unless a page asks for it (addLayer, setGazeTarget, nod / tilt / lean / blink, preset.idle.eyes.enabled, the blink-quality preset
// fields, limits options): with none of these a model renders and moves exactly as before (tests/viewer/lifelike.test.mjs, frame identity).
// GAZE_CAL: degrees of eye rotation per gaze control unit, measured on v21b full (CPU renders, iris template shift at gaze rest -0.4..+0.4, eye radius
// 10 mm, working-docs/spikes/lf_core/gazecal2.json): gazeX + = toward the picture's right, ~22 deg / unit (eyes 26 / 18); gazeY + = up, ~11.5 deg / unit.
// Trained range on v21b: gazeX rest -0.60..+0.60 (about +-13 deg), gazeY +-0.4 (about +-4.6 deg). A header may carry rig.gazeCal = { xDegPerUnit, yDegPerUnit }.
// Head signs (same renders): + yaw turns the face toward the picture's left, + pitch lifts the chin, + roll tilts the top of the head toward the picture's left.
export const GAZE_CAL = { xDegPerUnit: 22, yDegPerUnit: 11.5 };
// FREE_HEAD: the head range measured clean at 1:1 on v21b (CPU renders of the jaw / throat / collar, full / small / tiny, yaw 0 / +-20 / +-35 / +-40, owner pose,
// zoom 0.45 / 1 / 1.6, black and white; spikes/lf_core LOG): chin down 3 (-4 starts throat flecks), chin up 3 (+2 = today's limit already shows a faint dark tick at
// the collar on the left, +4 a dark streak), roll -1..+4 (the top of the head toward the picture's right tears a pale wisp off the left jaw from -1.5; toward the left
// the right jaw edge frays past +4). Pass as options.limits to open the head up; the header's pitch_limit / pitch_up_limit / roll_limit do the same per model.
export const FREE_HEAD = { pitchDownDeg: 3, pitchUpDeg: 3, rollDeg: [-1, 4] };
export const ROLL_DEFAULT_DEG = [-1, 4];                          // roll range when neither the header nor the options give one (nothing before 2026-11-25 sets roll)
// soft limit of an offset x inside [lo, hi] (lo <= 0 <= hi): linear up to 75 % of the room, then eases into the limit (no flat-topped clip). 0 -> 0 exactly.
function softLimit(x, lo, hi) {
  const side = (v, L) => { if (!(L > 0)) return 0; const k = 0.75 * L; return v <= k ? v : k + (L - k) * Math.tanh((v - k) / (L - k)); };
  return x > 0 ? side(x, hi) : x < 0 ? -side(-x, -lo) : 0;
}
// a nod, "up 1, down 3, settle": pitch in degrees (+ = chin up) at u = 0..1 of its duration; cosine steps between extremes (velocity continuous, 0 at each extreme)
const NOD_KEYS = [[0, 0], [0.16, 1], [0.45, -1], [0.72, 0.12], [1, 0]];
function nodCurve(u, up, down) {
  if (!(u > 0 && u < 1)) return 0; let i = 1; while (NOD_KEYS[i][0] < u) i++;
  const [t0, a0] = NOD_KEYS[i - 1], [t1, a1] = NOD_KEYS[i], val = a => (a > 0 ? a * up : a < 0 ? a * down : 0), s = 0.5 - 0.5 * Math.cos(Math.PI * (u - t0) / (t1 - t0));
  return val(a0) + (val(a1) - val(a0)) * s;
}
// minimum-jerk step 0..1 (saccade profile)
const mjerk = u => (u <= 0 ? 0 : u >= 1 ? 1 : u * u * u * (10 - 15 * u + 6 * u * u));
// one step of a critically damped follower s = { x, v } toward u held over dt, time constant T (spikes/lf_motion): exact for any dt (frame-rate independent)
function cdStep(s, u, dt, T) {
  if (!(dt > 0)) return s.x; if (!(T > 0)) { s.x = u; s.v = 0; return u; }
  const w = 1 / T, d = s.x - u, c = s.v + w * d, e = Math.exp(-w * dt); s.x = u + (d + c * dt) * e; s.v = (s.v - w * c * dt) * e;
  if (Math.abs(s.x - u) < 1e-9 && Math.abs(s.v) < 1e-8) { s.x = u; s.v = 0; } return s.x;
}
// words of a clip from character timings ({char, start, end}): [{ text, start, end, stress (1 content word, 0 function word), phrase, phraseStart }]
// a phrase starts at the first word, after , . ; : ! ? and after a gap of 0.15 s or more
function wordsFromTimings(timings) {
  const T = (timings || []).map(t => ({ c: String(t.char ?? ""), start: +t.start, end: +t.end })).filter(t => Number.isFinite(t.start) && Number.isFinite(t.end)), W = [];
  const isL = c => /^[\p{L}\p{N}'’]$/u.test(c); let mark = true, ph = -1;
  for (let i = 0; i < T.length;) {
    if (!isL(T[i].c)) { if (",.;:!?".includes(T[i].c) && T[i].c) mark = true; i++; continue; }
    let j = i; while (j < T.length && isL(T[j].c)) j++;
    const text = T.slice(i, j).map(t => t.c).join(""), w = { text, start: T[i].start, end: T[j - 1].end };
    const prev = W[W.length - 1]; if (mark || !prev || w.start - prev.end >= 0.15) { ph++; w.phraseStart = true; } else w.phraseStart = false;
    w.phrase = ph; w.stress = FUNCTION_WORDS.has(text.toLowerCase().replace(/’/g, "'")) ? 0 : 1; W.push(w); mark = false; i = j;
  }
  return W;
}
// per-hop (10 ms) RMS and F0 of a clip (normalised autocorrelation on a ~8 kHz copy, 70-400 Hz; 0 = unvoiced). pcm: Float32Array, sr: Hz
function analysePcm(pcm, sr) {
  const hop = 0.01, n = Math.floor(pcm.length / (hop * sr)), rms = new Float32Array(n), f0 = new Float32Array(n), dec = Math.max(1, Math.round(sr / 8000)), sr2 = sr / dec;
  const x = new Float32Array(Math.floor(pcm.length / dec)); for (let i = 0; i < x.length; i++) { let s = 0; for (let k = 0; k < dec; k++) s += pcm[i * dec + k]; x[i] = s / dec; }
  const hw = Math.round(0.01 * sr), win = Math.round(0.04 * sr2), lmin = Math.floor(sr2 / 400), lmax = Math.ceil(sr2 / 70); let top = 0;
  for (let h = 0; h < n; h++) { const c = Math.round((h + 0.5) * hop * sr); let s = 0, m = 0; for (let i = c - hw; i < c + hw; i++) if (i >= 0 && i < pcm.length) { s += pcm[i] * pcm[i]; m++; } rms[h] = Math.sqrt(s / Math.max(1, m)); if (rms[h] > top) top = rms[h]; }
  for (let h = 0; h < n; h++) {
    if (rms[h] < 0.03 * top) continue; const c = Math.round((h + 0.5) * hop * sr2) - (win >> 1); if (c < 0 || c + win + lmax >= x.length) continue;
    let best = 0, bl = 0; const r = new Float32Array(lmax + 2);
    for (let l = lmin; l <= lmax + 1; l++) { let a = 0, e0 = 0, e1 = 0; for (let i = 0; i < win; i++) { const p = x[c + i], q = x[c + i + l]; a += p * q; e0 += p * p; e1 += q * q; } r[l] = a / Math.sqrt(e0 * e1 + 1e-12); if (l <= lmax && r[l] > best) { best = r[l]; bl = l; } }
    if (best > 0.55 && bl > lmin) { const den = r[bl - 1] - 2 * r[bl] + r[bl + 1], fr = den < 0 ? 0.5 * (r[bl - 1] - r[bl + 1]) / den : 0; f0[h] = sr2 / (bl + fr); }
  }
  return { hop, rms, f0, sr, duration: pcm.length / sr };
}

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
  const BS = named ? readBlinkSweep(D.hdr, controlNames, Km) : null;   // blink sweep (header blinkSweep), null for every older model
  let RS = readRender(RENDER_DEFAULTS, options.render);          // render options (RENDER_DEFAULTS; all off by default except detail)
  const QEXT = clamp(+RS.quadExtent || 4, 1, 4), FSX = fsFor(QEXT), hdrAA = !!(D.hdr.antialiased || (D.hdr.render && D.hdr.render.antialiased));
  const aaOn = () => (RS.aaCompensation === "auto" ? hdrAA : !!RS.aaCompensation), shDeg = () => clamp(Math.round(+RS.shDegree), 0, 2);
  const restOf = j => (rig.rest && Number.isFinite(rig.rest[j]) ? rig.rest[j] : 0);   // a control appended by the finish (browUp, breath, ...) may have no rest entry: 0

  // ---- GL setup. Two renderers draw the same picture: "gpu" (default: the model is evaluated in the vertex shader, the CPU only sorts) and "cpu"
  // (the prototype's: everything on the CPU, 40 bytes per Gaussian uploaded every frame), used when asked for (options.renderer) or when the GPU one cannot be set up.
  const mk = (t, s) => { const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x); if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x)); return x; };
  const link = (vs, fs) => { const p = gl.createProgram(); gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p); if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw fail("webgl2", gl.getProgramInfoLog(p)); return p; };
  const qb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, qb); { const e = Math.sqrt(QEXT); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-e, -e, e, -e, -e, e, e, e]), gl.STATIC_DRAW); }   // the splat quad (+-2 = quadExtent 4)
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  const nG = D.n, STRIDE = 40;
  let R = null, P, uAAc, uView, uFocal, uViewport, uPP, vao, ib, inst32, inst8, vx, vy, vz, vcol, vidx, vdep, counts, bm, bc, bo, actM, actC, actO;   // R: the GPU renderer's state; the rest is the CPU renderer's
  const fixedLook = new Uint8Array(nG), alpha0 = new Float64Array(nG);
  // evaluation shortcuts that do not change a single bit of the result: (1) only the coefficients whose weight is not zero this frame are looped over (at rest that is the gaze pair, in speech two
  // to four of nine); (2) Gaussians whose colour and opacity do not depend on any control (9 in 10) skip those loops and use an opacity computed once.
  for (let i = 0; i < nG; i++) { let z = 1; for (let k = 0, o = i * D.Ko; k < D.Ko; k++) if (D.oc[o + k] !== 0) { z = 0; break; } if (z) for (let k = 0, o = i * D.Kc * 3, e = o + D.Kc * 3; o < e; o++) if (D.cc[o] !== 0) { z = 0; break; } fixedLook[i] = z; alpha0[i] = 1 / (1 + Math.exp(-D.op[i])); }
  const PUPILS = BS ? findPupils(D, BS, alpha0, controlNames.indexOf("gazeX"), Km) : null;
  // controls whose trained track column is flat (max - min < 1e-6: e.g. TRAIN's expression controls in a base-only finish view) but that DO move something
  // (a non-zero motion, colour or opacity column): their range falls back to [0, 1] below instead of [lo, lo] (which pinned them). Found here, before the GPU
  // setup frees the arrays; only the flat columns are scanned. Every existing model (v21b, v22: no such column) is unchanged.
  const FLAT_MOVES = (() => { const out = new Set(); if (!named || !Array.isArray(rig.track) || !rig.track.length) return out; const tr = rig.track;
    for (let j = Ks; j < Math.min(Km, tr[0].length); j++) { let a = Infinity, b = -Infinity; for (const r of tr) { if (r[j] < a) a = r[j]; if (r[j] > b) b = r[j]; } if (!(b - a < 1e-6)) continue;
      let mv = false; for (let i = 0; i < nG && !mv; i++) { const o = (i * Km + j) * 3; if (D.mot[o] !== 0 || D.mot[o + 1] !== 0 || D.mot[o + 2] !== 0) mv = true;
        if (!mv && j < D.Kc) { const q = (i * D.Kc + j) * 3; if (D.cc[q] !== 0 || D.cc[q + 1] !== 0 || D.cc[q + 2] !== 0) mv = true; } if (!mv && j < D.Ko && D.oc[i * D.Ko + j] !== 0) mv = true; }
      if (mv) out.add(j); }
    return out; })();   // catchlight eye centres (before the GPU setup frees the base colours)
  function cpuSetup() {                                          // the prototype's renderer (ported)
    P = link(VS, FSX); gl.useProgram(P); uAAc = gl.getUniformLocation(P, "uAA");
    uView = gl.getUniformLocation(P, "uView"); uFocal = gl.getUniformLocation(P, "uFocal"); uViewport = gl.getUniformLocation(P, "uViewport"); uPP = gl.getUniformLocation(P, "uPP");
    vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, qb); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    ib = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, ib);
    for (const [loc, size, type, norm, offs] of [[1, 3, gl.FLOAT, false, 0], [2, 3, gl.FLOAT, false, 12], [3, 3, gl.FLOAT, false, 24], [4, 4, gl.UNSIGNED_BYTE, true, 36]]) {
      gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, type, norm, STRIDE, offs); gl.vertexAttribDivisor(loc, 1);
    }
    const inst = new ArrayBuffer(nG * STRIDE); inst32 = new Float32Array(inst); inst8 = new Uint8Array(inst);
    gl.bufferData(gl.ARRAY_BUFFER, nG * STRIDE, gl.DYNAMIC_DRAW);
    vx = new Float32Array(nG); vy = new Float32Array(nG); vz = new Float32Array(nG); vcol = new Uint8Array(nG * 4); vidx = new Int32Array(nG); vdep = new Float32Array(nG); counts = new Uint32Array(65536);
    bm = new Float32Array(Km); bc = new Float32Array(D.Kc); bo = new Float32Array(D.Ko); actM = new Int32Array(Km); actC = new Int32Array(D.Kc); actO = new Int32Array(D.Ko);
    if (head) head.covV = new Float32Array(D.cov);
  }

  // ---- pose and limits
  let limits = null, tele = 1;
  const lo = options.limits || {};
  // pitch: `pitch` is the chin-down limit (faded by pitch_neg_fade at large + yaw), `pitchUp` the chin-up one (header pitch_up_limit, new; default = pitch_limit);
  // roll: header roll_limit (new), default ROLL_DEFAULT_DEG. options.limits: { yawDeg, pitchDeg (both ways, as before), pitchDownDeg, pitchUpDeg, rollDeg, pitchNegFade (null = off) }
  { const yawDeg = lo.yawDeg ?? rig.yaw_limit ?? (named ? 35 : null);
    if (yawDeg !== null && +yawDeg > 0) limits = { yaw: +yawDeg * DEG, pitch: +(lo.pitchDownDeg ?? lo.pitchDeg ?? rig.pitch_limit ?? 12) * DEG, pitchUp: +(lo.pitchUpDeg ?? lo.pitchDeg ?? rig.pitch_up_limit ?? rig.pitch_limit ?? 12) * DEG,
      roll: (r => (Array.isArray(r) && r.length === 2 ? [Math.min(0, +r[0]), Math.max(0, +r[1])] : [-Math.abs(+r || 0), Math.abs(+r || 0)]).map(x => x * DEG))(lo.rollDeg ?? rig.roll_limit ?? ROLL_DEFAULT_DEG), negFade: lo.pitchNegFade !== undefined ? lo.pitchNegFade : rig.pitch_neg_fade || null }; }
  tele = +(options.tele ?? rig.tele ?? 1);
  // ---- ORBIT (new): the whole bust is one rigid body and the camera goes round it. rig.orbit = { min, max (deg; omitted = full circle), axisZ (m, model frame), followHead, followMaxDeg, exprFadeDeg: [from, to] }
  const OR = rig.orbit || null, orbitOn = !!OR, orbitFull = orbitOn && (OR.min === undefined || OR.max === undefined || OR.max - OR.min >= 359.9);
  const orb = orbitOn ? { min: orbitFull ? -Infinity : OR.min * DEG, max: orbitFull ? Infinity : OR.max * DEG, axisZ: OR.axisZ ?? -0.07, follow: options.followHead ?? OR.followHead ?? true, followMax: (OR.followMaxDeg ?? 30) * DEG, fade: (OR.exprFadeDeg || [40, 70]).map(x => x * DEG), followFade: (OR.followFadeDeg || [60, 100]).map(x => x * DEG), tiltWithFollow: !!OR.tiltWithFollow,
    mode: (options.followMode ?? OR.followMode) === "lock" ? "lock" : "ease", rate: clamp(+(OR.followRate ?? 0.5), 0, 0.9), easeS: Math.max(0, +(OR.followEaseS ?? 0.6)), auto: !!options.autoOrbit, gainP: 0 } : null;
  // follow gain: 1 while the user turns the bust by hand and head-follow is on, 0 while the page turns it by itself (setAutoOrbit: the bust then turns as one rigid body); eased in time, never switched
  const followTarget = () => (orb && orb.follow && !orb.auto ? 1 : 0);
  if (orb) orb.gainP = followTarget();
  const base = { yaw: 0, pitch: 0, zoom: 1, orbit: 0, roll: 0 }, eff = { yaw: 0, pitch: 0, zoom: 1, orbit: 0, face: 0, exprGain: 1, tiltYaw: 0, roll: 0 };
  const downLimit = yaw => { const nf = limits.negFade; return nf ? limits.pitch * clamp((nf[1] - yaw) / (nf[1] - nf[0]), 0, 1) : limits.pitch; };
  const clampAngles = (yaw, pitch) => {                          // the prototype's clampView
    if (!limits) return [yaw, pitch];
    yaw = clamp(yaw, -limits.yaw, limits.yaw); const lw = downLimit(yaw);
    return [yaw, clamp(pitch, -lw, limits.pitchUp)];
  };

  // ---- head-turn mode (ported)
  let head = null;
  function headSetup() {
    const hw = new Float32Array(nG), H = rig.head || {}, y0 = H.y0 ?? 1.325, dy = H.dy ?? 0.055;
    if (D.hw) hw.set(D.hw); else for (let i = 0; i < nG; i++) { const t = Math.min(1, Math.max(0, (D.mean[i * 3 + 1] - y0) / dy)); hw[i] = t * t * (3 - 2 * t); }
    head = { hw, covV: null, piv: H.pivot || [-0.0145, 1.36, 0.085], zfix: !!H.zfix, tilt: H.tilt || null, Q: [1, 0, 0, 0, 1, 0, 0, 0, 1], key: "0,0", nt: null };   // nt: the nod pivot shift (lifelike), null = none
  }
  if (named && options.head !== false) headSetup();

  // ---- the GPU renderer (see VS_GPU and sortSplats). Returns null when this device or model cannot take it; the CPU renderer is used then.
  function gpuSetup() {
    const TW = 2048, maxRows = gl.getParameter(gl.MAX_TEXTURE_SIZE), Kc = D.Kc, Ko = D.Ko, Kl = Math.max(Kc, Ko), H16 = D.h16 || {};
    if (!(Km <= 64 && Kl <= 64) || !D.rest || D.rest.length !== nG * 24) return null;
    const rows = t => Math.max(1, Math.ceil(t / TW));
    const rowM = new Int32Array(nG).fill(-1), rowL = new Int32Array(nG).fill(-1); let nm = 0, nl = 0;
    for (let i = 0; i < nG; i++) { for (let o = i * Km * 3, e = o + Km * 3; o < e; o++) if (D.mot[o] !== 0) { rowM[i] = nm++; break; } if (!fixedLook[i]) rowL[i] = nl++; }
    if (Math.max(rows(nG * 6), rows(nG * 4), rows(nm * Km), rows(nl * Kl)) > maxRows) return null;
    const textures = [];
    let pre = null;                                              // the evaluation pre-pass: needs float render targets (EXT_color_buffer_float)
    if (options.gpuPrepass !== false && gl.getExtension("EXT_color_buffer_float")) {
      try {
        const Pe = link(VS_EVAL, FS_EVAL), Ps = link(VS_SPLAT, FSX), h = rows(nG), fbo = gl.createFramebuffer(), outs = [];
        for (const unit of [4, 5]) { const t = gl.createTexture(); gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t); textures.push(t); outs.push(t);
          for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
          gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, TW, h); }
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outs[0], 0); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, outs[1], 0);
        gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]); const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE; gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        if (ok) { gl.useProgram(Ps); gl.uniform1i(gl.getUniformLocation(Ps, "tP0"), 4); gl.uniform1i(gl.getUniformLocation(Ps, "tP1"), 5);
          pre = { Pe, Ps, fbo, h, uVp: gl.getUniformLocation(Ps, "uVp"), evao: gl.createVertexArray() }; }
      } catch (e) { pre = null; }
    }
    const IX = pre ? (jj, i, N) => jj * N + i : (jj, i, N, J) => i * J + jj;   // texel of plane jj of item i (see VS_GPU / FS_EVAL)
    const buf = (half, texels) => (half ? new Uint16Array(TW * rows(texels) * 4) : new Float32Array(TW * rows(texels) * 4));
    const upload = (unit, half, data) => {
      const t = gl.createTexture(); gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t); textures.push(t);
      for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1); gl.texImage2D(gl.TEXTURE_2D, 0, half ? gl.RGBA16F : gl.RGBA32F, TW, data.length / 4 / TW, 0, gl.RGBA, half ? gl.HALF_FLOAT : gl.FLOAT, data);
    };
    const hw = head ? head.hw : null, geo = buf(false, nG * 4), { mean, cov, op, dc } = D;
    for (let i = 0; i < nG; i++) {
      const g0 = IX(0, i, nG, 4) * 4, g1 = IX(1, i, nG, 4) * 4, g2 = IX(2, i, nG, 4) * 4, g3 = IX(3, i, nG, 4) * 4, i3 = i * 3, i6 = i * 6;
      geo[g0] = mean[i3]; geo[g0 + 1] = mean[i3 + 1]; geo[g0 + 2] = mean[i3 + 2]; geo[g0 + 3] = hw ? hw[i] : 0;
      geo[g1] = cov[i6]; geo[g1 + 1] = cov[i6 + 1]; geo[g1 + 2] = cov[i6 + 2]; geo[g1 + 3] = cov[i6 + 3]; geo[g2] = cov[i6 + 4]; geo[g2 + 1] = cov[i6 + 5]; geo[g2 + 2] = op[i]; geo[g2 + 3] = rowL[i];
      geo[g3] = dc[i3]; geo[g3 + 1] = dc[i3 + 1]; geo[g3 + 2] = dc[i3 + 2]; geo[g3 + 3] = rowM[i];
    }
    upload(0, false, geo);
    const shH = !!H16.rest, sh = buf(shH, nG * 6), rs = shH ? H16.rest : D.rest;   // half floats in the file go up as they are
    for (let i = 0; i < nG; i++) for (let j = 0; j < 6; j++) { const o = IX(j, i, nG, 6) * 4, s = i * 24 + j * 4; sh[o] = rs[s]; sh[o + 1] = rs[s + 1]; sh[o + 2] = rs[s + 2]; sh[o + 3] = rs[s + 3]; }
    upload(1, shH, sh);
    const mH = !!H16.mot, mt = buf(mH, nm * Km), ms = mH ? H16.mot : D.mot;
    for (let i = 0; i < nG; i++) { const r = rowM[i]; if (r >= 0) for (let k = 0; k < Km; k++) { const o = IX(k, r, nm, Km) * 4, p = (i * Km + k) * 3; mt[o] = ms[p]; mt[o + 1] = ms[p + 1]; mt[o + 2] = ms[p + 2]; } }
    upload(2, mH, mt);
    const lH = !!(H16.cc && H16.oc), lk = buf(lH, nl * Kl), cs = lH ? H16.cc : D.cc, os = lH ? H16.oc : D.oc;
    for (let i = 0; i < nG; i++) { const r = rowL[i]; if (r < 0) continue;
      for (let k = 0; k < Kc; k++) { const o = IX(k, r, nl, Kl) * 4, p = (i * Kc + k) * 3; lk[o] = cs[p]; lk[o + 1] = cs[p + 1]; lk[o + 2] = cs[p + 2]; }
      for (let k = 0; k < Ko; k++) lk[IX(k, r, nl, Kl) * 4 + 3] = os[i * Ko + k]; }
    upload(3, lH, lk);
    const Pg = pre ? pre.Pe : link(VS_GPU, FSX);                   // the program that evaluates the Gaussians
    gl.useProgram(Pg); const U = {};
    for (const n of ["uView", "uFocal", "uViewport", "uPP", "tGeo", "tSH", "tMot", "tLook", "uKm", "uKl", "uHead", "uHwField", "uWm", "uWc", "uWo", "uHQ", "uPiv", "uZf", "uCam", "uVp", "uN", "uNg", "uNm", "uNl", "uBk", "uBw", "uBs", "uBm", "uBe", "uAA", "uShDeg", "uShC"]) U[n] = gl.getUniformLocation(Pg, n);
    gl.uniform1i(U.uShDeg, 2); gl.uniform1i(U.uAA, 0); gl.uniform4f(U.uShC, 0, 0, 0, 0);
    gl.uniform1i(U.uBk, BS ? BS.k : -1); gl.uniform1f(U.uBw, 0); if (BS) { gl.uniform1f(U.uBs, BS.soft); gl.uniform1f(U.uBm, BS.mot); gl.uniform3fv(U.uBe, new Float32Array(BS.eyes.flat())); }
    gl.uniform1i(U.tGeo, 0); gl.uniform1i(U.tSH, 1); gl.uniform1i(U.tMot, 2); gl.uniform1i(U.tLook, 3); gl.uniform1i(U.uKm, Km); gl.uniform1i(U.uKl, Kl); gl.uniform1i(U.uN, nG); gl.uniform1i(U.uNg, nG); gl.uniform1i(U.uNm, nm); gl.uniform1i(U.uNl, nl);
    gl.uniform1i(U.uHead, head ? 1 : 0); gl.uniform1i(U.uHwField, D.hw ? 1 : 0); if (head) gl.uniform3fv(U.uPiv, head.piv);
    const va = gl.createVertexArray(); gl.bindVertexArray(va); gl.bindBuffer(gl.ARRAY_BUFFER, qb); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    const ob = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, ob); gl.bufferData(gl.ARRAY_BUFFER, nG * 4, gl.DYNAMIC_DRAW);   // the depth order: one Gaussian index per instance
    gl.enableVertexAttribArray(1); gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_INT, 4, 0); gl.vertexAttribDivisor(1, 1); gl.bindVertexArray(null);
    const SS = { n: nG, Km, Ko, mean, hw, op, fixedLook, alpha0, motH: mH ? H16.mot : null, motF: mH ? null : D.mot, ocH: H16.oc || null, ocF: H16.oc ? null : D.oc, H: halfTable(), bs: BS };
    const r = { P: Pg, pre, U, vao: va, ob, textures, S: Object.assign({ counts: new Uint32Array(65536), dep: new Float32Array(nG), idx: new Int32Array(nG) }, SS), out: new Uint32Array(nG),
      wk: null, wbuf: new Uint32Array(nG), inflight: null, queued: null, gen: 0, sortMs: 0, nMain: 0, nWorker: 0, wpad: [new Float32Array(64), new Float32Array(64), new Float32Array(64)], hq: new Float32Array(9) };
    if (options.sortWorker !== false && typeof Worker !== "undefined" && typeof Blob !== "undefined") {
      r.wk = sortWorker(SS);
      if (r.wk) {
        r.wk.w.onmessage = e => { const d = e.data; r.wbuf = d.buf; r.sortMs = d.ms; r.nWorker++; workerDone(d); };
        r.wk.w.onerror = () => { const f = r.inflight; r.inflight = null; if (r.wk) { r.wk.w.terminate(); URL.revokeObjectURL(r.wk.url); } r.wk = null; lastSig = null; if (!destroyed) dirty = true; };   // no worker: every frame sorts on the main thread
      }
    }
    D.cov = D.dc = D.rest = D.cc = null; if (mH) D.mot = null; if (H16.oc) D.oc = null; D.h16 = { mot: H16.mot, oc: H16.oc };   // now on the GPU: free the main thread's copies
    return r;
  }
  const want = options.renderer === "cpu" || options.renderer === "gpu" ? options.renderer : "auto";
  if (want !== "cpu") { try { R = gpuSetup(); } catch (e) { R = null; console.warn("OpenAvatar viewer: GPU renderer not available, using the CPU one:", e && e.message); } }
  if (!R) { try { cpuSetup(); } catch (e) { throw e.code ? failWith(e) : e; } }   // a link failure is reported as before (error event, code "webgl2")
  function headCov() {
    const key = eff.yaw.toFixed(4) + "," + eff.pitch.toFixed(4) + (eff.roll ? "," + eff.roll.toFixed(4) : ""); if (key === head.key) return; head.key = key;
    const Q = head.Q, c = D.cov, o = head.covV, hw = head.hw, m00 = Q[0], m01 = Q[3], m02 = Q[6], m10 = Q[1], m11 = Q[4], m12 = Q[7], m20 = Q[2], m21 = Q[5], m22 = Q[8];
    for (let i = 0; i < nG; i++) { if (hw[i] < 0.5) continue; const j = i * 6, a = c[j], b = c[j + 1], cc_ = c[j + 2], d = c[j + 3], e = c[j + 4], f = c[j + 5];
      const r0x = m00 * a + m01 * b + m02 * cc_, r0y = m00 * b + m01 * d + m02 * e, r0z = m00 * cc_ + m01 * e + m02 * f, r1x = m10 * a + m11 * b + m12 * cc_, r1y = m10 * b + m11 * d + m12 * e, r1z = m10 * cc_ + m11 * e + m12 * f, r2x = m20 * a + m21 * b + m22 * cc_, r2y = m20 * b + m21 * d + m22 * e, r2z = m20 * cc_ + m21 * e + m22 * f;
      o[j] = r0x * m00 + r0y * m01 + r0z * m02; o[j + 1] = r0x * m10 + r0y * m11 + r0z * m12; o[j + 2] = r0x * m20 + r0y * m21 + r0z * m22; o[j + 3] = r1x * m10 + r1y * m11 + r1z * m12; o[j + 4] = r1x * m20 + r1y * m21 + r1z * m22; o[j + 5] = r2x * m20 + r2y * m21 + r2z * m22; }
  }

  // ---- camera (ported): orbit around the head pivot in OpenCV convention; in head mode the camera stays and the head turns
  function cameraMatrices() {
    const c = (hasRig && rig.cam_c2w) || D.hdr.cam_c2w, Rc = [c[0][0], c[0][1], c[0][2], c[1][0], c[1][1], c[1][2], c[2][0], c[2][1], c[2][2]], tc = [c[0][3], c[1][3], c[2][3]], Pv = D.hdr.pivot;
    const Ry = rodrigues([0, 1, 0], eff.yaw), right = mv3(Ry, [Rc[0], Rc[3], Rc[6]]); let Q = mul3(rodrigues(right, eff.pitch), Ry);
    if (eff.roll !== 0) Q = mul3(rodrigues([Rc[2], Rc[5], Rc[8]], eff.roll), Q);   // roll about the camera's view axis (+ = top of the head toward the picture's left)
    if (head) {
      // a nod about a lower pivot (M.nodPivot m below the head pivot): the head turns about its pivot as always, plus the shift that makes the nod's own
      // rotation (about `right`) turn about the lower point instead: (I - Mn) d, Mn = the nod's rotation as the head transform applies it (transposed), d = pivot drop
      head.nt = null;
      if (M.nodPivot > 0 && M.nodPitch !== 0) { const Rn = rodrigues(right, M.nodPitch), dd = [0, -M.nodPivot, 0]; head.nt = [dd[0] - (Rn[0] * dd[0] + Rn[3] * dd[1] + Rn[6] * dd[2]), dd[1] - (Rn[1] * dd[0] + Rn[4] * dd[1] + Rn[7] * dd[2]), dd[2] - (Rn[2] * dd[0] + Rn[5] * dd[1] + Rn[8] * dd[2])]; }
      if (head.tilt) {                                           // follow the trained pose path: the few degrees of pitch/roll the reference stills have at this yaw
        const T = head.tilt, u = Math.min(T.e.length - 1, Math.max(0, (eff.tiltYaw - T.y0) / T.dy)), i0 = Math.min(T.e.length - 2, Math.floor(u)), f = u - i0, a = T.e[i0], b = T.e[i0 + 1];
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
  function evaluate(tau, cam, V, xo = NO_XO) {                  // xo (render options): { shc: camera in the head frame for the SH clamp | null, deg: SH degree, radial }
    const { n, Km: Kmm, Kc, Ko, mean, dc, op, rest, mot, cc, oc } = D, cov = head ? head.covV : D.cov, hQ = head && head.Q, hW = head && head.hw, hP = head && head.piv; if (head) headCov();
    if (hasRig) { bm.set(rigW); bc.set(rigW.subarray(0, Math.min(Km, Kc))); bo.set(rigW.subarray(0, Math.min(Km, Ko))); }   // live rig: the same linear model, weights set at run time
    else { for (let k = 0; k < Kmm; k++) bm[k] = Math.cos(Math.PI * (k + 1) * tau); for (let k = 0; k < Kc; k++) bc[k] = Math.cos(Math.PI * (k + 1) * tau); for (let k = 0; k < Ko; k++) bo[k] = Math.cos(Math.PI * (k + 1) * tau); }
    const r2x = V[2], r2y = V[6], r2z = V[10], t2 = V[14]; let m = 0, dmin = 1e9, dmax = -1e9;
    const shc = xo.shc, g1 = xo.deg >= 1, g2 = xo.deg >= 2, rad = !!xo.radial;
    const bw = BS && hasRig ? bm[BS.k] : 0, sweep = !!BS && hasRig && bw > 0 && bw < 1;   // blink sweep (see blinkSweepWeight)
    let nM = 0, nC = 0, nO = 0; for (let k = 0; k < Kmm; k++) if (bm[k] !== 0) actM[nM++] = k; for (let k = 0; k < Kc; k++) if (bc[k] !== 0) actC[nC++] = k; for (let k = 0; k < Ko; k++) if (bo[k] !== 0) actO[nO++] = k;
    // refB models were trained with the head turning about the model origin plus an in-image shift; shifting the head along the view direction by the difference restores the trained front-to-back order at the collar
    let zfx = 0, zfy = 0, zfz = 0; if (head && head.zfix) { const qx = hQ[0] * hP[0] + hQ[3] * hP[1] + hQ[6] * hP[2], qy = hQ[1] * hP[0] + hQ[4] * hP[1] + hQ[7] * hP[2], qz = hQ[2] * hP[0] + hQ[5] * hP[1] + hQ[8] * hP[2], sft = r2x * (hP[0] - qx) + r2y * (hP[1] - qy) + r2z * (hP[2] - qz); zfx = -r2x * sft; zfy = -r2y * sft; zfz = -r2z * sft; }
    if (head && head.nt) { zfx += head.nt[0]; zfy += head.nt[1]; zfz += head.nt[2]; }
    for (let i = 0; i < n; i++) {
      const i3 = i * 3; let x = mean[i3], y = mean[i3 + 1], z = mean[i3 + 2];
      let o = i * Kmm * 3; for (let q = 0; q < nM; q++) { const k = actM[q], b = bm[k], p = o + k * 3; x += b * mot[p]; y += b * mot[p + 1]; z += b * mot[p + 2]; }
      let dl = 0; if (sweep) { const p = o + BS.k * 3; dl = blinkSweepWeight(bw, blinkSweepHeight(BS, mean[i3], mean[i3 + 1], mot[p + 1]), BS.soft) - bw; x += dl * mot[p]; y += dl * mot[p + 1]; z += dl * mot[p + 2]; }
      const fx = fixedLook[i]; let a;
      if (fx) a = alpha0[i]; else { let ov = op[i]; o = i * Ko; for (let q = 0; q < nO; q++) { const k = actO[q]; ov += bo[k] * oc[o + k]; } if (dl !== 0 && BS.k < Ko) ov += dl * oc[o + BS.k]; a = 1 / (1 + Math.exp(-ov)); }
      if (a < 0.004) continue;
      const hx = x, hy = y, hz = z, hwi = head ? hW[i] : 0; if (hwi > 0) { const px = x - hP[0], py = y - hP[1], pz = z - hP[2]; x += hwi * (hP[0] + hQ[0] * px + hQ[3] * py + hQ[6] * pz - x + zfx); y += hwi * (hP[1] + hQ[1] * px + hQ[4] * py + hQ[7] * pz - y + zfy); z += hwi * (hP[2] + hQ[2] * px + hQ[5] * py + hQ[8] * pz - z + zfz); }
      let dep = r2x * x + r2y * y + r2z * z + t2; if (dep < 0.2) continue; if (rad) dep = Math.hypot(x - cam[0], y - cam[1], z - cam[2]);
      let cr = dc[i3], cg = dc[i3 + 1], cb = dc[i3 + 2]; if (!fx) { o = i * Kc * 3; for (let q = 0; q < nC; q++) { const k = actC[q], b = bc[k], p = o + k * 3; cr += b * cc[p]; cg += b * cc[p + 1]; cb += b * cc[p + 2]; }
        if (dl !== 0 && BS.k < Kc) { const p = o + BS.k * 3; cr += dl * cc[p]; cg += dl * cc[p + 1]; cb += dl * cc[p + 2]; } }
      let dx = x - cam[0], dy = y - cam[1], dz = z - cam[2]; const il = 1 / (Math.hypot(dx, dy, dz) + 1e-9); dx *= il; dy *= il; dz *= il;
      if (hwi > 0) { let ex = hQ[0] * dx + hQ[1] * dy + hQ[2] * dz, ey = hQ[3] * dx + hQ[4] * dy + hQ[5] * dz, ez = hQ[6] * dx + hQ[7] * dy + hQ[8] * dz;   // view direction in the head's own frame
        if (shc) { ex = hx - shc[0]; ey = hy - shc[1]; ez = hz - shc[2]; const l = 1 / (Math.hypot(ex, ey, ez) + 1e-9); ex *= l; ey *= l; ez *= l; }   // render.shClampDeg: as seen from the clamped yaw
        if (D.hw) { if (hwi < 1) { ex = dx + hwi * (ex - dx); ey = dy + hwi * (ey - dy); ez = dz + hwi * (ez - dz); const l = 1 / (Math.hypot(ex, ey, ez) + 1e-9); ex *= l; ey *= l; ez *= l; } dx = ex; dy = ey; dz = ez; }
        else if (hwi >= 0.5) { dx = ex; dy = ey; dz = ez; } }
      const xx = dx * dx, yy = dy * dy, zz = dz * dz;
      const B0 = g1 ? -C1 * dy : 0, B1 = g1 ? C1 * dz : 0, B2 = g1 ? -C1 * dx : 0, B3 = g2 ? S2a * dx * dy : 0, B4 = g2 ? -S2a * dy * dz : 0, B5 = g2 ? S2c * (2 * zz - xx - yy) : 0, B6 = g2 ? -S2a * dx * dz : 0, B7 = g2 ? S2e * (xx - yy) : 0;   // render.shDegree   // no array per Gaussian (it was 240,000 allocations a frame)
      let r = C0 * cr + 0.5, g = C0 * cg + 0.5, bl = C0 * cb + 0.5; const ro = i * 24;
      r += B0 * rest[ro]; g += B0 * rest[ro + 1]; bl += B0 * rest[ro + 2]; r += B1 * rest[ro + 3]; g += B1 * rest[ro + 4]; bl += B1 * rest[ro + 5]; r += B2 * rest[ro + 6]; g += B2 * rest[ro + 7]; bl += B2 * rest[ro + 8];
      r += B3 * rest[ro + 9]; g += B3 * rest[ro + 10]; bl += B3 * rest[ro + 11]; r += B4 * rest[ro + 12]; g += B4 * rest[ro + 13]; bl += B4 * rest[ro + 14]; r += B5 * rest[ro + 15]; g += B5 * rest[ro + 16]; bl += B5 * rest[ro + 17];
      r += B6 * rest[ro + 18]; g += B6 * rest[ro + 19]; bl += B6 * rest[ro + 20]; r += B7 * rest[ro + 21]; g += B7 * rest[ro + 22]; bl += B7 * rest[ro + 23];
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

  // ---- render extras (RENDER_DEFAULTS; spikes/lf_render): SH clamp camera, half-resolution splat target, background plate, catchlights, post. With the
  // defaults none of this runs and the frame is the old clear + splats.
  function headQAt(yaw, pitch, tiltYaw) {                        // the head rotation cameraMatrices() makes for this pose (front models)
    const c = (hasRig && rig.cam_c2w) || D.hdr.cam_c2w, Rc = [c[0][0], c[0][1], c[0][2], c[1][0], c[1][1], c[1][2], c[2][0], c[2][1], c[2][2]];
    const Ry = rodrigues([0, 1, 0], yaw), right = mv3(Ry, [Rc[0], Rc[3], Rc[6]]); let Q = mul3(rodrigues(right, pitch), Ry);
    if (eff.roll) Q = mul3(rodrigues([Rc[2], Rc[5], Rc[8]], eff.roll), Q);   // as cameraMatrices (roll, CORE)
    if (head.tilt) { const T = head.tilt, u = Math.min(T.e.length - 1, Math.max(0, (tiltYaw - T.y0) / T.dy)), i0 = Math.min(T.e.length - 2, Math.floor(u)), f = u - i0, a = T.e[i0], b = T.e[i0 + 1];
      const e = [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])], ang = Math.hypot(e[0], e[1], e[2]); if (ang > 1e-6) Q = mul3(Q, rodrigues([e[0] / ang, e[1] / ang, e[2] / ang], -ang)); }
    return Q;
  }
  function zfixOf(Q, V) {                                        // the zfix shift (see frameParams) for head rotation Q
    if (!head.zfix) return head.nt ? head.nt.slice() : [0, 0, 0]; const hP = head.piv, r2 = [V[2], V[6], V[10]], qx = Q[0] * hP[0] + Q[3] * hP[1] + Q[6] * hP[2], qy = Q[1] * hP[0] + Q[4] * hP[1] + Q[7] * hP[2], qz = Q[2] * hP[0] + Q[5] * hP[1] + Q[8] * hP[2];
    const sft = r2[0] * (hP[0] - qx) + r2[1] * (hP[1] - qy) + r2[2] * (hP[2] - qz), nt = head.nt || [0, 0, 0]; return [-r2[0] * sft + nt[0], -r2[1] * sft + nt[1], -r2[2] * sft + nt[2]];   // + the nod-pivot shift (CORE)
  }
  // SH clamp: past +-shClampDeg of head yaw the view direction for the colour is the one the camera had at the arc's end: the camera position in the head
  // frame for the clamped yaw (camH = piv + Q (cam - piv - zf); with the real Q this gives exactly the unclamped direction). null = not clamping this frame.
  function shClampCam(pos, V) {
    const A = +RS.shClampDeg * DEG; if (!(A > 0) || !head || orbitOn || Math.abs(eff.yaw) <= A) return null;
    const yc = clamp(eff.yaw, -A, A), Q = headQAt(yc, eff.pitch, yc), zf = zfixOf(Q, V), hP = head.piv, v = mv3(Q, [pos[0] - hP[0] - zf[0], pos[1] - hP[1] - zf[1], pos[2] - hP[2] - zf[2]]);
    return [hP[0] + v[0], hP[1] + v[1], hP[2] + v[2]];
  }
  const XT = { prog: null, fbo: null, tex: null, w: 0, h: 0, eyes: null, bust: null, seed: 0, last: false, ss: 1, up: null };   // ss: supersample kept in the texture (detail pass); up: EASU target
  const xLink = (vs, fs) => { const p = link(vs, fs), u = {}; const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); for (let i = 0; i < n; i++) { const a = gl.getActiveUniform(p, i); u[a.name] = gl.getUniformLocation(p, a.name); } return { p, u }; };
  function xProgs() { if (!XT.prog) XT.prog = { plate: xLink(VS_FULL, FS_PLATE), post: xLink(VS_FULL, FS_POST), glint: xLink(VS_GLINT, FS_GLINT), cas: xLink(VS_FULL, FS_CAS), easu: xLink(VS_FULL, FS_EASU), vao: gl.createVertexArray() }; return XT.prog; }
  function texTarget(o, w, h, filt) {                            // (re)make an RGBA8 texture + framebuffer pair of this size in o ({ tex, fbo, w, h })
    if (o.tex && o.w === w && o.h === h) return o; if (o.tex) gl.deleteTexture(o.tex); if (o.fbo) gl.deleteFramebuffer(o.fbo);
    o.tex = gl.createTexture(); gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, o.tex); gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, w, h);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filt); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filt);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    o.fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, o.fbo); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, o.tex, 0); gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    o.w = w; o.h = h; return o;
  }
  function target() {                                            // where the splats go this frame: the canvas, or (render.splatScale < 1) a smaller texture scaled up after
    if (RS.detail) { const s0 = +RS.splatScale, s = (s0 > 0 && s0 < 1 ? s0 : 1) * XT.ss, w = Math.max(1, Math.round(canvas.width * s)), h = Math.max(1, Math.round(canvas.height * s));   // detail pass: always a texture
      const D_ = XT.dt || (XT.dt = {}); texTarget(D_, w, h, gl.LINEAR); return { w, h, fbo: D_.fbo, tex: D_.tex, detail: true }; }
    const s = +RS.splatScale; if (!(s > 0 && s < 1)) return { w: canvas.width, h: canvas.height, fbo: null };
    const w = Math.max(1, Math.round(canvas.width * s)), h = Math.max(1, Math.round(canvas.height * s));
    if (XT.w !== w || XT.h !== h || !XT.fbo) {
      if (XT.tex) gl.deleteTexture(XT.tex); if (XT.fbo) gl.deleteFramebuffer(XT.fbo);
      XT.tex = gl.createTexture(); gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, XT.tex); gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, w, h);
      XT.fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, XT.fbo); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, XT.tex, 0); gl.bindFramebuffer(gl.FRAMEBUFFER, null); XT.w = w; XT.h = h;
    }
    return { w, h, fbo: XT.fbo };
  }
  function project(T, V, p) {                                    // model point -> target pixels (x right, y down) and depth, as the splat shaders do
    const sc = T.h / D.hdr.img_h, fl = D.hdr.K[1][1] * sc * tele, cx = V[0] * p[0] + V[4] * p[1] + V[8] * p[2] + V[12], cy = V[1] * p[0] + V[5] * p[1] + V[9] * p[2] + V[13], cz = V[2] * p[0] + V[6] * p[1] + V[10] * p[2] + V[14];
    return [fl * cx / cz + T.w / 2 + (D.hdr.K[0][2] - D.hdr.img_w / 2) * sc, fl * cy / cz + D.hdr.K[1][2] * sc, cz, fl];
  }
  function bustInfo() {                                          // the torso's lower edge (for the contact shadow): opaque body Gaussians' 3rd percentile height, x extent
    if (XT.bust) return XT.bust; const m = D.mean, hw = head ? head.hw : null, ys = [], xs = [];
    for (let i = 0; i < nG; i++) { if (hw && hw[i] > 0.01) continue; if (alpha0[i] < 0.5) continue; ys.push(m[i * 3 + 1]); xs.push(m[i * 3]); }
    if (ys.length < 50) return (XT.bust = { ok: false }); ys.sort((a, b) => a - b); xs.sort((a, b) => a - b); const q = (a, f) => a[Math.floor(f * (a.length - 1))];
    const y0 = q(ys, 0.03); let zs = 0, k = 0; for (let i = 0; i < nG; i++) if ((!hw || hw[i] <= 0.01) && alpha0[i] >= 0.5 && Math.abs(m[i * 3 + 1] - y0) < 0.02) { zs += m[i * 3 + 2]; k++; }
    return (XT.bust = { ok: true, y: y0, x0: q(xs, 0.02), x1: q(xs, 0.98), z: k ? zs / k : 0 });
  }
  function eyeCentres(cl) {                                      // catchlight sphere centres in the model frame (head at rest)
    if (Array.isArray(cl.eyes) && cl.eyes.length === 2) return cl.eyes.map(e => e.map(Number));
    if (XT.eyes) return XT.eyes; if (!BS) return (XT.eyes = []);
    const m = D.mean, hw = head ? head.hw : null, r = +cl.radius || 0.012;
    XT.eyes = BS.eyes.map(([cx0, top, bot], k) => { const pu = PUPILS && PUPILS[k], cx = pu ? pu[0] : cx0, cy = pu ? pu[1] : 0.5 * (top + bot), zs = [];   // pupil (load time) or the lid box's middle
      for (let i = 0; i < nG; i++) { const x = m[i * 3], y = m[i * 3 + 1]; if (Math.abs(x - cx) > 0.0025 || Math.abs(y - cy) > 0.0025 || alpha0[i] < 0.3 || (hw && hw[i] < 0.5)) continue; zs.push(m[i * 3 + 2]); }
      if (!zs.length) return null; zs.sort((a, b) => a - b); return [cx, cy, zs[Math.floor(0.95 * (zs.length - 1))] - r]; }).filter(Boolean);
    return XT.eyes;
  }
  function toWorld(p, Q, zf) {                                   // a head point (hw = 1) where the head is now
    if (!head) return p; const hP = head.piv, d = [p[0] - hP[0], p[1] - hP[1], p[2] - hP[2]];
    return [hP[0] + Q[0] * d[0] + Q[3] * d[1] + Q[6] * d[2] + zf[0], hP[1] + Q[1] * d[0] + Q[4] * d[1] + Q[7] * d[2] + zf[1], hP[2] + Q[2] * d[0] + Q[5] * d[1] + Q[8] * d[2] + zf[2]];
  }
  function glints(V, pos) {                                      // this frame's catchlights: [{ x, y (canvas px), s (sigma px), i, col }]
    const cl = RS.catchlights; if (!cl) return []; const E = eyeCentres(cl); if (!E.length) return [];
    const T = { w: canvas.width, h: canvas.height }, Q = head ? head.Q : [1, 0, 0, 0, 1, 0, 0, 0, 1], zf = head ? zfixOf(Q, V) : [0, 0, 0], r = +cl.radius || 0.012, out = [];
    const fw = head ? [Q[6], Q[7], Q[8]] : [0, 0, 1], bk = BS ? BS.k : ctlIndex("blink"), bw = rigW && bk >= 0 ? rigW[bk] : 0;
    const c0 = Math.cos(cl.fadeDeg[0] * DEG), c1 = Math.cos(cl.fadeDeg[1] * DEG);
    for (const e of E) {
      const c = toWorld(e, Q, zf);
      for (const L of cl.lights) {
        const az = (+L.az || 0) * DEG, el = (+L.el || 0) * DEG, lc = [Math.sin(az) * Math.cos(el), -Math.sin(el), -Math.cos(az) * Math.cos(el)];   // camera frame (x right, y down, z forward), toward the light
        const l = [V[0] * lc[0] + V[1] * lc[1] + V[2] * lc[2], V[4] * lc[0] + V[5] * lc[1] + V[6] * lc[2], V[8] * lc[0] + V[9] * lc[1] + V[10] * lc[2]];   // world (V's rotation transposed)
        const g = catchlightPoint(c, pos, l, r), hdot = g[3] * fw[0] + g[4] * fw[1] + g[5] * fw[2]; let vis = ss((hdot - c1) / Math.max(1e-6, c0 - c1));
        if (bw > 0) { const yh = e[1] + r * (Q[3] * g[3] + Q[4] * g[4] + Q[5] * g[5]);   // the glint's height on the resting eye (head frame) -> how far the lid has swept over it
          const be = BS ? (Math.abs(e[0] - BS.eyes[0][0]) <= Math.abs(e[0] - BS.eyes[1][0]) ? BS.eyes[0] : BS.eyes[1]) : null, sw = be ? blinkSweepWeight(bw, clamp((be[1] - yh) / (be[1] - be[2]), 0, 1), BS.soft) : bw; vis *= (1 - ss((sw - 0.12) / 0.33)) * (1 - ss((bw - 0.12) / 0.16)); }   // the visible lid runs ahead of the sweep's height model: also gone by blink 0.28
        if (vis <= 0.002) continue;
        const pr = project(T, V, g); if (!(pr[2] > 0.2)) continue;
        out.push({ x: pr[0], y: pr[1], s: Math.max(+cl.minPx || 0, (+L.size || 0.0005) * pr[3] / pr[2]), i: (+L.intensity || 0) * vis, col: L.color || [1, 1, 1] });
      }
    }
    return out;
  }
  function beginFrame(T, bgc, V) {                                  // bind the splat target, clear it, draw the background plate (render.post.plate / shadow)
    gl.bindFramebuffer(gl.FRAMEBUFFER, T.fbo); gl.viewport(0, 0, T.w, T.h);
    const [cr, cg, cb] = hexRgb(bgc); gl.clearColor(cr, cg, cb, 1); gl.clear(gl.COLOR_BUFFER_BIT);
    const po = RS.post; if (!po || !(po.plate || po.shadow)) return;
    const X = xProgs(), u = X.plate.u, pl = po.plate || {}, sh = po.shadow || null; gl.useProgram(X.plate.p); gl.disable(gl.BLEND); gl.bindVertexArray(X.vao);
    const asp = T.w / T.h; gl.uniform3f(u.uBg, cr, cg, cb); gl.uniform3fv(u.uTint, pl.tint || [1, 1, 1]); gl.uniform2fv(u.uCtr, pl.centre || [0.5, 0.4]); gl.uniform2f(u.uAsp, asp, 1);
    gl.uniform1f(u.uLift, po.plate ? +pl.lift || 0 : 0); gl.uniform1f(u.uFall, +pl.falloff || 0.9);
    const b = sh ? bustInfo() : null;
    if (b && b.ok) { const a = project(T, V, [b.x0, b.y, b.z]), c = project(T, V, [b.x1, b.y, b.z]), mid = project(T, V, [0.5 * (b.x0 + b.x1), b.y, b.z]), wpx = Math.abs(c[0] - a[0]);
      gl.uniform4f(u.uSh, mid[0] / T.w, 1 - mid[1] / T.h, 0, 0); gl.uniform2f(u.uShS, Math.max(1e-3, (sh.width || 1) * wpx * 0.5 / T.w), Math.max(1e-3, (sh.height || 0.2) * wpx / T.h)); gl.uniform1f(u.uShA, +sh.amount || 0); }
    else gl.uniform1f(u.uShA, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3); gl.enable(gl.BLEND);
  }
  function detailResolve(T) {                                    // the detail pass: [EASU up to the canvas size], then CAS + grain onto the canvas
    const X = xProgs(), dt = RS.detail; let src = T.tex, sw = T.w, sh = T.h; gl.disable(gl.BLEND); gl.bindVertexArray(X.vao);
    if (sw < canvas.width && dt.upscale === "fsr1") {
      const U_ = XT.up || (XT.up = {}); texTarget(U_, canvas.width, canvas.height, gl.LINEAR); gl.bindFramebuffer(gl.FRAMEBUFFER, U_.fbo); gl.viewport(0, 0, U_.w, U_.h);
      gl.useProgram(X.easu.p); gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, src); gl.uniform1i(X.easu.u.uSrc, 7); gl.uniform2f(X.easu.u.uIn, sw, sh); gl.uniform2f(X.easu.u.uOut, U_.w, U_.h);
      gl.drawArrays(gl.TRIANGLES, 0, 3); src = U_.tex; sw = U_.w; sh = U_.h; }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, canvas.width, canvas.height);
    const u = X.cas.u; gl.useProgram(X.cas.p); gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, src); gl.uniform1i(u.uSrc, 7); gl.uniform2f(u.uOut, canvas.width, canvas.height);
    gl.uniform1f(u.uSharp, clamp(+dt.sharpen || 0, 0, 1)); gl.uniform1f(u.uGrain, Math.max(0, +dt.grain || 0)); gl.uniform1f(u.uSeed, (XT.seed = (XT.seed + 0.6180339887) % 1));
    gl.drawArrays(gl.TRIANGLES, 0, 3); gl.enable(gl.BLEND);
  }
  function endFrame(T, V, pos) {                                 // scale the splat target up, then catchlights and post on the canvas
    if (T.detail) detailResolve(T);
    else if (T.fbo) { gl.bindFramebuffer(gl.READ_FRAMEBUFFER, T.fbo); gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null); gl.blitFramebuffer(0, 0, T.w, T.h, 0, 0, canvas.width, canvas.height, gl.COLOR_BUFFER_BIT, gl.LINEAR); gl.bindFramebuffer(gl.FRAMEBUFFER, null); }
    gl.viewport(0, 0, canvas.width, canvas.height);
    const G = RS.catchlights ? glints(V, pos) : [], po = RS.post, pv = po && (po.vignette > 0 || po.grain > 0);
    if (!G.length && !pv) return;
    const X = xProgs(); gl.bindVertexArray(X.vao);
    if (G.length) { const u = X.glint.u, cl = RS.catchlights; gl.useProgram(X.glint.p); gl.blendFunc(gl.ONE, gl.ONE); gl.uniform1f(u.uHalo, +cl.halo || 0); gl.uniform1f(u.uK, cl.halo > 0 ? 9 : 3.2);
      for (const g of G) { gl.uniform2f(u.uC, 2 * g.x / canvas.width - 1, 1 - 2 * g.y / canvas.height); gl.uniform2f(u.uR, 2 * g.s / canvas.width, 2 * g.s / canvas.height); gl.uniform3fv(u.uCol, g.col); gl.uniform1f(u.uI, g.i); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4); } }
    if (pv) { const u = X.post.u; gl.useProgram(X.post.p); gl.blendFunc(gl.DST_COLOR, gl.SRC_COLOR); gl.uniform1f(u.uVig, +po.vignette || 0); gl.uniform1f(u.uGrain, +po.grain || 0);
      gl.uniform1f(u.uSeed, (XT.seed = (XT.seed + 0.6180339887) % 1)); gl.uniform2f(u.uAsp, canvas.width / canvas.height, 1); gl.drawArrays(gl.TRIANGLES, 0, 3); }
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  }

  // ---- GPU renderer, per frame. A loop frame hands its weights and pose to the sort worker and is drawn when the worker returns that frame's own
  // depth order (a few milliseconds later, normally before the screen is next composited); while the worker is busy only the newest frame waits.
  // Drawing with an order sorted for an earlier frame was measured and rejected: half a degree of turn or one frame of mouth motion already moves
  // single pixels by 20-40 levels (working-docs/spikes/render/LOG.md). renderOnce and pages without workers sort on the main thread.
  function frameParams(tau, V, pos) {                                 // the inputs of sortSplats for this frame (copies: the worker gets them)
    let wm, wc, wo;
    if (hasRig) { wm = Float32Array.from(rigW); const cut = K => { if (K === Km) return wm; const a = new Float32Array(K); a.set(wm.subarray(0, Math.min(K, Km))); return a; }; wc = cut(D.Kc); wo = cut(D.Ko); }
    else { const cb = K => { const w = new Float32Array(K); for (let k = 0; k < K; k++) w[k] = Math.cos(Math.PI * (k + 1) * tau); return w; }; wm = cb(Km); wc = cb(D.Kc); wo = cb(D.Ko); }
    const r2 = [V[2], V[6], V[10], V[14]]; let zf = [0, 0, 0], Q = null;
    if (head) { const hQ = head.Q, hP = head.piv; Q = hQ.slice();
      if (head.zfix) { const qx = hQ[0] * hP[0] + hQ[3] * hP[1] + hQ[6] * hP[2], qy = hQ[1] * hP[0] + hQ[4] * hP[1] + hQ[7] * hP[2], qz = hQ[2] * hP[0] + hQ[5] * hP[1] + hQ[8] * hP[2], sft = r2[0] * (hP[0] - qx) + r2[1] * (hP[1] - qy) + r2[2] * (hP[2] - qz); zf = [-r2[0] * sft, -r2[1] * sft, -r2[2] * sft]; }
      if (head.nt) zf = [zf[0] + head.nt[0], zf[1] + head.nt[1], zf[2] + head.nt[2]]; }
    const F = { wm, wc, wo, head: !!head, Q, piv: head ? head.piv : null, zf, r2 }; if (RS.sort === "radial" && pos) F.cam = [pos[0], pos[1], pos[2]]; return F;
  }
  const sameSig = (a, b) => !!a && !!b && a.length === b.length && a.every((x, k) => x === b[k]);
  function gpuDraw(f, m) {                                       // f: { F, V, pos, bg, shc }; m: number of Gaussians in the order buffer
    const r = R, U = r.U, F = f.F, pre = r.pre, [pm, pc, po] = r.wpad, T = target();   // T: the canvas, or the smaller splat target (render.splatScale)
    const sc = T.h / D.hdr.img_h, fl = D.hdr.K[1][1] * sc * tele;   // the training crop fills the canvas height
    gl.useProgram(r.P);
    gl.uniformMatrix4fv(U.uView, false, f.V); gl.uniform2f(U.uFocal, fl, fl); gl.uniform2f(U.uViewport, T.w, T.h); gl.uniform2f(U.uVp, T.w, T.h);
    gl.uniform2f(U.uPP, T.w / 2 + (D.hdr.K[0][2] - D.hdr.img_w / 2) * sc, D.hdr.K[1][2] * sc);   // principal point: row 640 of the ring frame, not the crop centre
    gl.uniform1i(U.uAA, aaOn() ? 1 : 0); gl.uniform1i(U.uShDeg, shDeg()); if (f.shc) gl.uniform4f(U.uShC, f.shc[0], f.shc[1], f.shc[2], 1); else gl.uniform4f(U.uShC, 0, 0, 0, 0);
    pm.fill(0); pm.set(F.wm); pc.fill(0); pc.set(F.wc); po.fill(0); po.set(F.wo);
    gl.uniform4fv(U.uWm, pm); gl.uniform4fv(U.uWc, pc); gl.uniform4fv(U.uWo, po); gl.uniform3f(U.uCam, f.pos[0], f.pos[1], f.pos[2]);
    if (BS) gl.uniform1f(U.uBw, F.wm[BS.k]);
    if (F.Q) { r.hq.set(F.Q); gl.uniformMatrix3fv(U.uHQ, false, r.hq); gl.uniform3f(U.uZf, F.zf[0], F.zf[1], F.zf[2]); }
    if (pre) {                                                   // pre-pass: one fragment per Gaussian
      gl.bindFramebuffer(gl.FRAMEBUFFER, pre.fbo); gl.viewport(0, 0, 2048, pre.h); gl.disable(gl.BLEND); gl.bindVertexArray(pre.evao); gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, canvas.width, canvas.height); gl.enable(gl.BLEND); gl.useProgram(pre.Ps); gl.uniform2f(pre.uVp, T.w, T.h);
    }
    if (!xOn()) { const [cr, cg, cb] = hexRgb(f.bg); gl.clearColor(cr, cg, cb, 1); gl.clear(gl.COLOR_BUFFER_BIT); }
    else { beginFrame(T, f.bg, f.V); gl.useProgram(pre ? pre.Ps : r.P); }
    gl.bindVertexArray(r.vao); gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, m);
    if (xOn()) endFrame(T, f.V, f.pos);
  }
  function xOn() { const s = +RS.splatScale; return !!(RS.catchlights || RS.post || RS.detail || (s > 0 && s < 1)); }   // any render extra this frame (else: the old clear + draw)
  function gpuPost(f) { const r = R, buf = r.wbuf; r.wbuf = null; r.inflight = f; r.wk.w.postMessage({ gen: r.gen, F: f.F, buf }, [buf.buffer]); }
  function workerDone(d) {                                       // the order for the frame in flight: upload, draw, report the frame; then the newest waiting frame goes
    const r = R, f = r.inflight; r.inflight = null;
    if (destroyed) return;
    if (f && d.gen === r.gen) { gl.bindBuffer(gl.ARRAY_BUFFER, r.ob); gl.bufferSubData(gl.ARRAY_BUFFER, 0, d.buf, 0, d.m); count = d.m; gpuDraw(f, d.m); frameDone(f.now, f.info); }
    if (r.queued) { const q = r.queued; r.queued = null; if (q.gen === r.gen) gpuPost(q); }
  }

  // ---- controls
  const stats = (() => {                                         // range of each rig component seen in training
    if (!named || !rig.track) return null; const tr = rig.track, K = tr[0].length, lo_ = [], hi_ = [];
    for (let j = 0; j < K; j++) { let a = Infinity, b = -Infinity; for (const r of tr) { if (r[j] < a) a = r[j]; if (r[j] > b) b = r[j]; } lo_.push(a); hi_.push(b); } return { lo: lo_, hi: hi_ };
  })();
  // extra controls (finish-made, appended after the trained ones as kind "free": browUp, furrow, breath, ...): range from rig.ranges[name] = [lo, hi] when given,
  // else the trained track's range when the track has that column, else -1..1; rest from rig.rest or 0. Names past Km (no motion column) are not controls.
  // FINISH writes rig.synth = { version, unit, controls: { browUp: { mm, min, max }, ... } } (their track columns are 0: not a range)
  const SY = isObj(rig.synth) && isObj(rig.synth.controls) ? rig.synth.controls : {}, ranges = { ...(isObj(rig.ranges) ? rig.ranges : {}) };
  for (const k of Object.keys(SY)) if (isObj(SY[k]) && Number.isFinite(+SY[k].min) && Number.isFinite(+SY[k].max)) ranges[k] = [+SY[k].min, +SY[k].max];
  const synth = (Array.isArray(rig.synthetic) ? rig.synthetic : []).concat(Object.keys(SY));
  const flatNote = [];
  const ctlInfo = controlNames.slice(0, Number.isFinite(Km) ? Km : controlNames.length).map((name, index) => {
    const speech = index < Ks, rest = restOf(index), tw = !!stats && index < stats.lo.length && !(name in SY), rg = Array.isArray(ranges[name]) && ranges[name].length === 2 && ranges[name].every(Number.isFinite) ? ranges[name].map(Number) : null;
    const c = { name, index, kind: speech ? "speech" : "free", min: speech ? 0 : rg ? rg[0] : (tw ? stats.lo[index] : -1), max: speech ? 1 : rg ? rg[1] : (tw ? stats.hi[index] : 1), rest, trained: tw ? [stats.lo[index], stats.hi[index]] : null };
    if (!speech && !rg && tw && FLAT_MOVES.has(index)) { c.min = 0; c.max = 1; c.flatTrack = true; flatNote.push(name); }   // flat track, but the control moves the model
    if (synth.includes(name)) c.synthetic = true; return c;
  });
  if (flatNote.length) console.info("OpenAvatar viewer: controls with a flat track column but trained motion get the range [0, 1]: " + flatNote.join(", "));
  const ctlIndex = n => { const i = controlNames.indexOf(n); return i < ctlInfo.length ? i : -1; };
  const manual = {};                                             // explicit values from setControls; they override the driver and idle motion
  let bg = options.background || "#000000";

  // ---- preset, drivers
  let preset = mergeDeep(DEFAULT_PRESET, {});
  const env = { controls: controlNames, Ks, index: n => { const i = ctlIndex(n); return i >= 0 && i < Ks ? i : -1; }, settings: () => preset.driver };
  const drivers = {}; let driver = null, source = null, driverData = null;
  const getDriver = name => { if (drivers[name]) return drivers[name]; const f = driverRegistry[name]; if (!f) throw new Error("unknown driver " + name); return (drivers[name] = f(env)); };
  let speechCut = false;                                          // set by every change of driver, track or audio: the speech release starts at the next frame
  function selectDriver() {
    speechCut = true; if (driver) driver.detach(); driver = null;
    const name = preset.driver.name; if (!name || name === "none" || !named) return;
    driver = getDriver(name); if (driver.setData) driver.setData(driverData); if (source) driver.attach(source);
  }

  // ---- idle motion (gaze, head sway, blink)
  const idle = { t: 0, gaze: { from: [0, 0], to: [0, 0], t0: 0, dur: 0.1, next: 0.5, aversion: null, nextAversion: rnd(8, 15) }, sway: { from: [0, 0], to: [0, 0], t0: 0, dur: 4 }, blink: makeBlinker(Math.random, 2 + Math.random() * 2), off: [0, 0], gazeVal: [0, 0] };
  const speakingNow = () => !!(driver && driver.speaking && (!source || source.playing()));
  const gi = { x: ctlIndex("gazeX"), y: ctlIndex("gazeY") };
  const relMouth = makeRelease(Math.max(Ks, 1), 1e-3, SPEECH_RELEASE_MOUTH_S, 0.5), relPitch = makeRelease(1, 2e-5), pitchBuf = new Float64Array(1), live = { on: false, src: null, mt: NaN, cut1: false };
  function speechLive(dt) {                                       // once per frame: was the speech cut since the last frame (see SPEECH RELEASE)
    const on = !!(source && source.playing()), mt = on && source.hasClock ? source.mediaTime() : NaN;
    let cut = speechCut || on !== live.on || source !== live.src;
    if (on && live.on && Number.isFinite(mt) && Number.isFinite(live.mt) && Math.abs(mt - live.mt - dt) > 0.25) cut = true;   // the clip's clock jumped (a seek, a new clip in the same element)
    live.on = on; live.src = source; live.mt = mt; speechCut = false; return cut;
  }
  function stepIdle(dt, cut) {
    const I = preset.idle, S = preset.speech, id = idle, speaking = speakingNow(); id.t += dt; const t = id.t, MO = moOn();
    if (!I.enabled) { id.off[0] = id.off[1] = 0; relPitch.reset(); return; }
    // head sway: a slow random walk inside the rig's limits, eased between targets (motion.swaySpline: a C1 spline through them, the head never stops)
    const sw = I.sway; let oy = 0, op_ = 0;
    if (sw.enabled && limits && (!speaking || S.swayWhileSpeaking)) {
      const q = id.sway, spl = !!(MO && MO.swaySpline), ya = (sw.yawDeg + (speaking ? S.emphasisDeg : 0)) * DEG, ta = sw.tiltDeg * DEG;
      const draw = () => { const d = rndRange(sw.periodS); return { d, p: [rnd(-ya, ya), rnd(-ta, ta)] }; };   // the same three draws per target as before
      if (t - q.t0 >= q.dur) {
        const tb = q.t0 + q.dur; q.prev = q.from; q.pdur = q.dur; q.from = q.to.slice(); q.t0 = t;
        if (spl) { const a = q.next || draw(); q.dur = a.d; q.to = a.p; q.next = draw(); if (t - tb < q.dur) q.t0 = tb; }   // spline: targets drawn one ahead (the tangent at a knot needs the next one); the segment starts at the knot's time (no one-frame pause)
        else { const a = draw(); q.dur = a.d; q.to = a.p; q.next = null; }
      }
      const u = (t - q.t0) / q.dur;
      if (spl && q.next) {                                       // Hermite through prev -> from -> to -> next, Catmull-Rom tangents for uneven segment times: C1, never stops
        const P = q.prev || q.from, pd = q.pdur || q.dur, nd = q.next.d, N = q.next.p, h = clamp(u, 0, 1), h2 = h * h, h3 = h2 * h, a0 = 2 * h3 - 3 * h2 + 1, a1 = h3 - 2 * h2 + h, a2 = -2 * h3 + 3 * h2, a3 = h3 - h2;
        for (let k = 0; k < 2; k++) {
          const m0 = (q.to[k] - P[k]) / (pd + q.dur) * q.dur, m1 = (N[k] - q.from[k]) / (q.dur + nd) * q.dur, v = a0 * q.from[k] + a1 * m0 + a2 * q.to[k] + a3 * m1;
          if (k === 0) oy = v; else op_ = v;
        }
      } else { const k = ss(u); oy = q.from[0] + (q.to[0] - q.from[0]) * k; op_ = q.from[1] + (q.to[1] - q.from[1]) * k; }
      const ym = limits.yaw * sw.limitFraction, pm = limits.pitch * sw.limitFraction; oy = clamp(oy, -ym, ym); op_ = clamp(op_, -pm, pm);
    }
    const BT = S.beat; pitchBuf[0] = 0;
    if (BT && BT.enabled !== false && driver && limits && source && source.playing()) { const b = driver.beat || 0, st = driver.settle || 0; pitchBuf[0] = PITCH_DOWN * ((BT.nodDeg ?? 0.9) * b + (BT.settleDeg ?? 0.6) * st) * DEG; }   // (no source: no beat; a detached timed driver kept its last value)
    op_ += relPitch.step(dt, pitchBuf, cut || live.cut1, !!(MO && MO.beatEaseIn))[0]; live.cut1 = cut;   // eases out instead of dropping to 0 when the clip ends, pauses or is replaced. The beat read here is the
    // driver's value from the previous frame's update, so a new clip or track reaches it one frame after the cut: the release is (re)started on both frames
    id.off[0] = oy; id.off[1] = op_;
    // gaze
    const g = I.gaze;
    if (g.enabled && gi.x >= 0 && gi.y >= 0) {
      const rx = rig.rest ? rig.rest[gi.x] : 0, ry = rig.rest ? rig.rest[gi.y] : 0, q = id.gaze;
      if (g.mode === "sine") { id.gazeVal[0] = rx + g.amount * Math.sin(t * (0.37 + 0.21 * gi.x) + gi.x); id.gazeVal[1] = ry + g.amount * Math.sin(t * (0.37 + 0.21 * gi.y) + gi.y); }
      else {
        const av = g.aversion || {}, avOn = av.enabled && (!speaking || S.aversionsWhileSpeaking);
        const prof = q.prof || ss, sac = !!(MO && MO.idleSaccade);       // motion.idleSaccade: minimum jerk over the main-sequence time (the drawn shiftS is still drawn: same random stream)
        const retarget = (tx, ty, dur) => { const k = prof((t - q.t0) / q.dur); q.from = [q.from[0] + (q.to[0] - q.from[0]) * k, q.from[1] + (q.to[1] - q.from[1]) * k]; q.to = [tx, ty]; q.t0 = t; q.dur = dur;
          if (sac) { const E = eyesCfg(), amp = Math.hypot((tx - q.from[0]) * cal.x, (ty - q.from[1]) * cal.y); q.dur = Math.max(dt, (E.saccadeMs[0] + E.saccadeMs[1] * amp) / 1000); q.prof = mjerk; } else q.prof = ss; };
        const bb = idle.blink.busy();                                    // a blink is running (only a rig with a blink control): the gaze jump waits for it
        if (bb) { /* hold the gaze where it is until the blink is over */ }
        else if (q.aversion && t >= q.aversion.until) { q.aversion = null; retarget(0, 0, rndRange(g.shiftS)); q.next = t + rndRange(g.intervalS); q.nextAversion = t + rndRange(speaking ? av.everySpeakingS || [4, 8] : av.everyListeningS || [8, 15]); }
        else if (!q.aversion && avOn && t >= q.nextAversion) { const sx = Math.random() < 0.5 ? -1 : 1; q.aversion = { until: t + rndRange(av.durationS || [1, 3]) }; retarget(sx * av.amount, -Math.abs(av.amount) * 0.5, rndRange(g.shiftS)); }
        else if (!q.aversion && t >= q.next) { retarget(rnd(-1, 1) * g.amount, rnd(-1, 1) * g.amount * 0.6, rndRange(g.shiftS)); q.next = t + rndRange(g.intervalS); }
        const k = (q.prof || ss)((t - q.t0) / q.dur); id.gazeVal[0] = rx + q.from[0] + (q.to[0] - q.from[0]) * k; id.gazeVal[1] = ry + q.from[1] + (q.to[1] - q.from[1]) * k;
      }
    } else if (gi.x >= 0 && gi.y >= 0) { id.gazeVal[0] = rig.rest ? rig.rest[gi.x] : 0; id.gazeVal[1] = rig.rest ? rig.rest[gi.y] : 0; }
  }
  function blinkValue(dt) {                                      // a no-op unless the model has a "blink" control (then makeBlinker: timing, gaze guard, hand hold)
    const bi = ctlIndex("blink"), B = preset.idle.blink; if (bi < 0) return null;
    const auto = preset.idle.enabled && B.enabled; if (!auto && !idle.blink.pending()) return null;   // a triggered blink (v.blink, a layer) runs without the automatic ones
    const g = preset.idle.gaze, q = idle.gaze, jumps = g.enabled && g.mode !== "sine";
    return idle.blink.step(idle.t, B, { speaking: speakingNow(), faster: preset.speech.blinkFaster, gazeEnd: jumps ? q.t0 + q.dur : -Infinity, held: "blink" in manual, auto });
  }
  // ---- lifelike (spikes/lf_core): motion layers, primitives, eyes, speech events. Contract: working-docs/spikes/lf_core/API.md
  // Nothing here runs unless a page adds a layer, sets a gaze target, calls nod / tilt / lean / blink, or turns preset.idle.eyes on: M.on, EY.on stay false
  // and the pose, the weights and the picture are those of the viewer before (tests/viewer/lifelike.test.mjs).
  const layers = [];
  const moOn = () => (preset.motion && preset.motion.enabled !== false ? preset.motion : null);   // motion (spikes/lf_motion): null = the old motion exactly
  // the layers' summed outputs before the motion filter (LS) and the filter's followers (LF): pose 0..3 = yaw, pitch, roll, zoom; one per control
  const LS = { p: new Float64Array(4), c: new Float32Array(Math.max(ctlInfo.length, 1)), con: new Uint8Array(Math.max(ctlInfo.length, 1)) };
  const LF = { p: [0, 1, 2, 3].map(() => ({ x: 0, v: 0 })), c: ctlInfo.map(() => ({ x: 0, v: 0 })), cls: null, clsKey: "" };
  function ctlClasses(MO) {                                       // per control: 0 = unfiltered (speech shapes but smile, gaze, raw), 1 = expression, 2 = mouth-opening expression
    const key = JSON.stringify([MO.raw || [], MO.mouthExpr || []]); if (LF.cls && LF.clsKey === key) return LF.cls;
    LF.clsKey = key; LF.cls = ctlInfo.map((c, j) => ((j < Ks && c.name !== "smile") || c.name === "gazeX" || c.name === "gazeY" || (MO.raw || []).includes(c.name) ? 0 : (MO.mouthExpr || []).includes(c.name) ? 2 : 1));
    return LF.cls;
  }
  const lfRest = s => s.x === 0 && s.v === 0;
  const M = { on: false, yaw: 0, pitch: 0, roll: 0, zoom: 0, nodPitch: 0, nodPivot: 0, ctl: new Float32Array(Math.max(ctlInfo.length, 1)), ctlOn: false, gaze: null };
  const spring = T => ({ x: 0, v: 0, to: 0, T });
  function stepSpring(s, dt) {                                    // critically damped, exact for any dt; settles in about s.T
    if (!(dt > 0)) return; const w = 4.5 / Math.max(0.05, s.T);
    let goal = s.to;
    if (s.g !== undefined) { s.g += (s.to - s.g) * (1 - Math.exp(-1.5 * w * dt)); if (Math.abs(s.g - s.to) < 1e-7) s.g = s.to; goal = s.g; }   // smoothed target: no acceleration step at onset
    const d = s.x - goal, c = s.v + w * d, e = Math.exp(-w * dt);
    s.x = goal + (d + c * dt) * e; s.v = (s.v - w * c * dt) * e;
    if (goal === s.to && Math.abs(s.x - s.to) < 1e-7 && Math.abs(s.v) < 1e-6) { s.x = s.to; s.v = 0; }
  }
  const smooth = sp => (sp.g = 0, sp);                            // head-lead: target low-passed first (BEHAVE 08:34: a step in acceleration read as a jerk)
  const prim = { nods: [], tilt: spring(0.8), lean: spring(0.6), hy: smooth(spring(0.3)), hp: smooth(spring(0.3)), hlPend: null };
  let gazeTarget = null;                                          // setGazeTarget: [x, y] degrees (x + = the picture's right, y + = up; 0, 0 = the camera) or null
  const cal = { x: +(rig.gazeCal && rig.gazeCal.xDegPerUnit) || GAZE_CAL.xDegPerUnit, y: +(rig.gazeCal && rig.gazeCal.yDegPerUnit) || GAZE_CAL.yDegPerUnit };
  const EY = { off: null, on: false, fix: [0, 0], from: [0, 0], to: [0, 0], t0: -9, dur: 0, goal: null, mic: [0, 0], mFrom: [0, 0], mTo: [0, 0], mT0: -9, mDur: 0.03, nextMicro: 0, shown: [0, 0] };
  const EYES_DEFAULT = { enabled: false, hold: 1, holdBase: 0, saccadeMs: [25, 2.4], pursuitS: 0.12, saccadeMinDeg: 0.6,
    micro: { enabled: true, perS: [1, 2], maxDeg: 0.4 }, leadHead: { enabled: true, minDeg: 15, share: 0.4, delayS: 0.03, timeS: 0.35 },
    blinkOnSaccadeDeg: 10, blinkOnSaccadeChance: 0.5 };
  const eyesCfg = () => mergeDeep(EYES_DEFAULT, preset.idle.eyes || {});
  const lifeT = () => idle.t;
  // the head's turn the eyes cancel (degrees, picture axes): the motion offset (sway, nods, layers: eff minus the base pose) x hold, plus the base x holdBase
  function headDeg(E) {
    const dy = eff.yaw - base.yaw, dp = eff.pitch - base.pitch;
    return [-(E.hold * dy + E.holdBase * base.yaw) / DEG, (E.hold * dp + E.holdBase * base.pitch) / DEG];
  }
  function startSaccade(to, t, E) {
    const amp = Math.hypot(to[0] - EY.fix[0], to[1] - EY.fix[1]); EY.from = EY.fix.slice(); EY.to = to.slice(); EY.t0 = t; EY.landed = false; EY.dur = (E.saccadeMs[0] + E.saccadeMs[1] * amp) / 1000;
    if (E.blinkOnSaccadeDeg > 0 && amp >= E.blinkOnSaccadeDeg && Math.random() < (E.blinkOnSaccadeChance ?? 0) && ctlIndex("blink") >= 0) idle.blink.trigger("full");
    const L = E.leadHead || {};
    if (L.enabled && limits) {                                    // eyes lead, the head follows part of a large shift a little later (and comes back with a small target)
      if (amp >= L.minDeg) prim.hlPend = { at: t + (L.delayS ?? 0.03), x: to[0] * L.share, y: to[1] * L.share, T: L.timeS ?? 0.35 };
      else if (Math.hypot(to[0], to[1]) < 0.5 * L.minDeg && (prim.hy.to !== 0 || prim.hp.to !== 0)) prim.hlPend = { at: t + (L.delayS ?? 0.03), x: 0, y: 0, T: L.timeS ?? 0.35 };
    }
    return amp;
  }
  function motionStep(dt) {
    const t = lifeT(); let any = layers.length > 0;
    M.yaw = M.pitch = M.roll = M.zoom = 0; M.nodPitch = 0; M.nodPivot = 0; M.gaze = null; if (M.ctlOn) { M.ctl.fill(0); M.ctlOn = false; }
    for (let i = prim.nods.length - 1; i >= 0; i--) {
      const n = prim.nods[i], u = (t - n.t0) / n.dur; if (u >= 1) { prim.nods.splice(i, 1); continue; } any = true;
      const p = nodCurve(u, n.up, n.down) * DEG; M.pitch += p; if (n.pivot > 0) { M.nodPitch += p; M.nodPivot = Math.max(M.nodPivot, n.pivot); }
    }
    if (prim.hlPend && t >= prim.hlPend.at) { const h = prim.hlPend; prim.hlPend = null; prim.hy.to = -h.x * DEG; prim.hp.to = h.y * DEG; prim.hy.T = prim.hp.T = h.T; }
    for (const [sp, key] of [[prim.tilt, "roll"], [prim.lean, "zoom"], [prim.hy, "yaw"], [prim.hp, "pitch"]]) {
      if (sp.x === 0 && sp.v === 0 && sp.to === 0 && !sp.g) continue; stepSpring(sp, dt); M[key] += sp.x; any = true;
    }
    const MO = moOn(), F = MO ? LS : null;                       // motion on: the layers' sums go through the followers below first
    if (F) { F.p.fill(0); F.c.fill(0); F.con.fill(0); }
    if (layers.length) {
      const dw = {}; for (let j = 0; j < Ks; j++) dw[controlNames[j]] = rigW[j];
      const ctx = { t, dt, speaking: speakingNow(), driverWeights: dw, pose: api.pose, state: lifeState() };
      const num = v => (Number.isFinite(+v) ? +v : 0);
      for (const fn of layers.slice()) {
        let r; try { r = fn(ctx); } catch (e) { console.error("OpenAvatar layer:", e); continue; }
        if (!r || typeof r !== "object") continue;
        const ly = num(r.yaw) + num(r.yawDeg) * DEG, lp = num(r.pitch) + num(r.pitchDeg) * DEG, lr = num(r.roll) + num(r.rollDeg) * DEG, lz = num(r.zoom);
        if (F) { F.p[0] += ly; F.p[1] += lp; F.p[2] += lr; F.p[3] += lz; } else { M.yaw += ly; M.pitch += lp; M.roll += lr; M.zoom += lz; }
        if (isObj(r.controls)) for (const k of Object.keys(r.controls)) { const j = ctlIndex(k), v = +r.controls[k]; if (j >= 0 && Number.isFinite(v) && v !== 0) { if (F) { F.c[j] += v; F.con[j] = 1; } else { M.ctl[j] += v; M.ctlOn = true; } } }   // unknown names: ignored
        if (r.gaze === "camera") M.gaze = [0, 0]; else if (r.gaze && Number.isFinite(+r.gaze.x) && Number.isFinite(+r.gaze.y)) M.gaze = [+r.gaze.x, +r.gaze.y];
        if (r.blinkNow) api.blink(typeof r.blinkNow === "string" ? { kind: r.blinkNow } : isObj(r.blinkNow) ? r.blinkNow : {});
      }
    }
    if (F) {                                                     // the motion filter: critically damped followers (cdStep); a removed layer fades out instead of dropping
      const Tp = +MO.poseS || 0, Te = +MO.exprS || 0, Tm = Array.isArray(MO.mouthExprS) ? MO.mouthExprS : [Te, Te], cls = ctlClasses(MO);
      for (let k = 0; k < 4; k++) { const s = LF.p[k], u = F.p[k]; if (u === 0 && lfRest(s)) continue; const x = cdStep(s, u, dt, Tp); any = true;
        if (k === 0) M.yaw += x; else if (k === 1) M.pitch += x; else if (k === 2) M.roll += x; else M.zoom += x; }
      for (let j = 0; j < LF.c.length; j++) {
        const u = F.c[j], s = LF.c[j], c = cls[j];
        if (c === 0) { if (F.con[j]) { M.ctl[j] += u; M.ctlOn = true; } s.x = s.v = 0; continue; }
        if (u === 0 && lfRest(s)) continue;
        const x = cdStep(s, u, dt, c === 2 ? (u > s.x ? +Tm[0] || 0 : +Tm[1] || 0) : Te); any = true;
        if (x !== 0) { M.ctl[j] += x; M.ctlOn = true; }
      }
    } else { for (const s of LF.p) s.x = s.v = 0; for (const s of LF.c) s.x = s.v = 0; }
    M.on = any;
  }
  function lifeState() {
    const bi = ctlIndex("blink");
    return { blink: bi >= 0 && rigW ? rigW[bi] : 0, blinkKind: idle.blink.kind(), gaze: EY.on ? EY.shown.slice() : null, gazeTarget: gazeTarget ? gazeTarget.slice() : null,
      level: driver ? driver.level || 0 : 0, beat: driver ? driver.beat || 0 : 0, settle: driver ? driver.settle || 0 : 0, inSpeech: SE.on, wordIndex: SE.wi - 1, phrase: SE.phrase,
      nodding: prim.nods.length > 0, tilt: prim.tilt.to / DEG, lean: prim.lean.to, limits: api.limits, gazeCal: { ...cal } };
  }
  // after applyPose: the eyes (a gaze target, a layer's gaze, or preset.idle.eyes.enabled), layer control deltas, blink dips
  function lifeAfterPose(dt) {
    const t = lifeT(), E = eyesCfg(), bi = ctlIndex("blink"), bv = bi >= 0 ? rigW[bi] : 0, B = preset.idle.blink || {};
    const tgt = M.gaze || gazeTarget, eyesOn = gi.x >= 0 && gi.y >= 0 && (tgt !== null || (E.enabled === true && preset.idle.enabled));
    let ux = null, uy = null;
    if (eyesOn) {
      const H = headDeg(E), q = idle.gaze, T = tgt ? tgt.slice() : [q.to[0] * cal.x, q.to[1] * cal.y];
      if (!EY.on) {                                               // start from where the eyes are now (world = eye-in-head + head)
        EY.on = true; const ex = (rigW[gi.x] - restOf(gi.x)) * cal.x, ey = (rigW[gi.y] - restOf(gi.y)) * cal.y; EY.fix = [ex + H[0], ey + H[1]]; EY.goal = null; EY.pv = null; EY.t0 = -9; EY.mic = [0, 0]; EY.mT0 = -9; EY.nextMicro = t + 0.3;
      }
      if (!EY.goal || Math.hypot(T[0] - EY.goal[0], T[1] - EY.goal[1]) >= E.saccadeMinDeg) { EY.goal = T.slice(); startSaccade(T, t, E); }
      else { EY.goal = T.slice(); }
      const MOe = moOn(), p2 = !!(MOe && MOe.pursuitOrder2);
      if (t - EY.t0 < EY.dur) { const k = mjerk((t - EY.t0) / EY.dur); EY.fix = [EY.from[0] + (EY.to[0] - EY.from[0]) * k, EY.from[1] + (EY.to[1] - EY.from[1]) * k]; EY.pv = null; }
      else if (!EY.landed) { EY.landed = true; EY.fix = EY.to.slice(); EY.pv = null; }   // the saccade lands exactly
      else if (p2) {                                              // pursuit (motion.pursuitOrder2): critically damped, T = pursuitS / 2 (the same lag on a ramp, no velocity step when the target starts to move)
        if (!EY.pv) EY.pv = [{ x: EY.fix[0], v: 0 }, { x: EY.fix[1], v: 0 }];
        const Tq = Math.max(0.005, E.pursuitS / 2); EY.fix = [cdStep(EY.pv[0], EY.goal[0], dt, Tq), cdStep(EY.pv[1], EY.goal[1], dt, Tq)];
      }
      else { const a = dt > 0 ? 1 - Math.exp(-dt / Math.max(0.01, E.pursuitS)) : 0; EY.fix = [EY.fix[0] + (EY.goal[0] - EY.fix[0]) * a, EY.fix[1] + (EY.goal[1] - EY.fix[1]) * a]; EY.pv = null; }   // pursuit of a slowly moving target
      const mc = E.micro || {};                                   // microsaccades: 1-2 / s, under maxDeg, pulled back toward the fixation
      if (mc.enabled && t - EY.t0 > EY.dur && t >= EY.nextMicro) {
        const r = (mc.maxDeg ?? 0.4) * (0.3 + 0.7 * Math.random()), a = Math.random() * 2 * Math.PI, to = [-0.5 * EY.mic[0] + r * Math.cos(a), -0.5 * EY.mic[1] + r * Math.sin(a)], l = Math.hypot(to[0], to[1]), mx = mc.maxDeg ?? 0.4;
        if (l > mx) { to[0] *= mx / l; to[1] *= mx / l; }
        EY.mFrom = EY.mic.slice(); EY.mTo = to; EY.mT0 = t; EY.mDur = (E.saccadeMs[0] + E.saccadeMs[1] * Math.hypot(to[0] - EY.mic[0], to[1] - EY.mic[1])) / 1000;
        const per = mc.perS || [1, 2]; EY.nextMicro = t + 1 / (per[0] + (per[1] - per[0]) * Math.random());
      }
      if (EY.mT0 > -9) { const k = mjerk((t - EY.mT0) / EY.mDur); EY.mic = [EY.mFrom[0] + (EY.mTo[0] - EY.mFrom[0]) * k, EY.mFrom[1] + (EY.mTo[1] - EY.mFrom[1]) * k]; }
      const ex = EY.fix[0] + EY.mic[0] - H[0], ey = EY.fix[1] + EY.mic[1] - H[1]; EY.shown = [EY.fix[0] + EY.mic[0], EY.fix[1] + EY.mic[1]];
      ux = restOf(gi.x) + ex / cal.x; uy = restOf(gi.y) + ey / cal.y;
    } else if (EY.on) {                                           // back to the classic gaze: one saccade from where the eyes are to it, the head lead goes home
      EY.on = false; EY.off = { t0: t, u: [rigW[gi.x], rigW[gi.y]], dur: (E.saccadeMs[0] + E.saccadeMs[1] * 5) / 1000 };
      if (prim.hy.to !== 0 || prim.hp.to !== 0) prim.hlPend = { at: t, x: 0, y: 0, T: 0.5 };
    }
    if (!eyesOn && EY.off && gi.x >= 0) {
      const k = mjerk((t - EY.off.t0) / EY.off.dur); if (k >= 1) EY.off = null;
      else { if (!("gazeX" in manual)) rigW[gi.x] += (EY.off.u[0] - rigW[gi.x]) * (1 - k); if (!("gazeY" in manual)) rigW[gi.y] += (EY.off.u[1] - rigW[gi.y]) * (1 - k); }
    }
    const dip = +B.gazeDipDeg || 0;                               // blink quality: the eyes dip during a blink (classic gaze too)
    if (dip > 0 && bv > 0 && gi.y >= 0) uy = (uy ?? rigW[gi.y]) - dip * bv / cal.y;
    const clampG = (u, j) => { const c = ctlInfo[j]; return c.trained ? clamp(u, c.trained[0], c.trained[1]) : clamp(u, c.min, c.max); };
    const g = orbitOn ? eff.exprGain : 1;
    if (ux !== null && !("gazeX" in manual)) rigW[gi.x] = restOf(gi.x) + (clampG(ux, gi.x) - restOf(gi.x)) * g;
    if (uy !== null && !("gazeY" in manual)) rigW[gi.y] = restOf(gi.y) + (clampG(uy, gi.y) - restOf(gi.y)) * g;
    const bd = +B.browDip || 0, bu = ctlIndex("browUp");             // ...and the brows dip a touch (only a model with a browUp control)
    if (bd > 0 && bv > 0 && bu >= 0 && !("browUp" in manual)) rigW[bu] = Math.max(ctlInfo[bu].min, rigW[bu] - bd * bv);
    const MO = moOn(), soft = !!(MO && MO.softLimit), cls = MO ? ctlClasses(MO) : null;   // motion.softLimit: expression deltas ease into the range (softLimit, as the pose) instead of clipping flat
    const pi = ctlIndex("press"), gate = MO && MO.closureGate && pi >= 0 ? 1 - ss(rigW[pi] / 0.5) : 1;   // motion.closureGate
    const bg = MO && Array.isArray(MO.blinkGate) && MO.blinkGate.length && bv > 0 ? 1 - ss(bv / 0.8) : 1;   // motion.blinkGate (bv: the blink before layer deltas)
    if (M.ctlOn) for (let j = 0; j < ctlInfo.length; j++) { if (M.ctl[j] === 0 || controlNames[j] in manual) continue; const c = ctlInfo[j]; if (gate < 1 && cls[j] === 2 && M.ctl[j] > 0) M.ctl[j] *= gate;
      if (bg < 1 && M.ctl[j] > 0 && MO.blinkGate.includes(c.name)) M.ctl[j] *= bg; const b0 = rigW[j];
      rigW[j] = soft && cls[j] !== 0 && b0 >= c.min && b0 <= c.max ? b0 + softLimit(M.ctl[j], c.min - b0, c.max - b0) : clamp(b0 + M.ctl[j], c.min, c.max); }
  }
  // speech events: speechstart / speechend (an utterance: the first opening of a clip .. its track's end, the clip stopping, or 1 s of quiet without a track),
  // word / phrase (timed tracks: words from the character timings or data.words, at their start in the clip's time + driver.lead; loudness: phrase only)
  const SE = { on: false, quiet: 0, wi: 0, phrase: -1, lastMt: NaN, words: [], qT: 0 };
  function setWords(d) {
    let W = [];
    if (d && Array.isArray(d.words)) W = d.words.filter(w => w && Number.isFinite(+w.start)).map((w, i, a) => ({ text: String(w.text ?? w.word ?? ""), start: +w.start, end: Number.isFinite(+w.end) ? +w.end : +w.start, stress: w.stress ?? 1, phrase: w.phrase ?? 0, phraseStart: w.phraseStart ?? i === 0 }));
    else if (d && Array.isArray(d.timings)) W = wordsFromTimings(d.timings);
    SE.words = W; SE.wi = 0; SE.phrase = -1; SE.lastMt = NaN; clipA.words = W;
  }
  function trackEnd() { if (SE.words.length) return SE.words[SE.words.length - 1].end; const dbg = driver && driver.debug ? driver.debug() : null; return dbg && dbg.hasTrack ? dbg.lastEnd : NaN; }
  function speechEvents(dt) {
    if (!named) return;
    const playing = !!(source && source.playing()), spk = speakingNow(), mt = playing && source.hasClock ? source.mediaTime() + (preset.driver.lead ?? 0.02) : NaN, end = trackEnd();
    if (!SE.on) { if (spk && !(Number.isFinite(end) && Number.isFinite(mt) && mt > end)) { SE.on = true; SE.quiet = 0; SE.qT = 0; emit("speechstart", { t: lifeT(), mediaTime: mt, timed: SE.words.length > 0 || Number.isFinite(end), words: SE.words.length });
        if (!SE.words.length) { SE.phrase++; emit("phrase", { wordIndex: -1, t: Number.isFinite(mt) ? mt : lifeT(), phrase: SE.phrase }); } } }
    else {
      SE.quiet = spk ? 0 : SE.quiet + dt;
      const reason = !playing ? "stopped" : Number.isFinite(end) && Number.isFinite(mt) && mt > end + 0.05 ? "end" : !Number.isFinite(end) && SE.quiet >= 1 ? "quiet" : null;
      if (reason) { SE.on = false; emit("speechend", { t: lifeT(), mediaTime: mt, reason }); }
      else if (!SE.words.length) { if (!spk) SE.qT += dt; else { if (SE.qT >= 0.3) { SE.phrase++; emit("phrase", { wordIndex: -1, t: Number.isFinite(mt) ? mt : lifeT(), phrase: SE.phrase }); } SE.qT = 0; } }
    }
    if (playing && Number.isFinite(mt) && SE.words.length) {
      const W = SE.words;
      if (!(mt >= SE.lastMt - 0.25)) { SE.wi = 0; while (SE.wi < W.length && W[SE.wi].start < mt - 0.3) SE.wi++; }   // first frame, or the clock went back: no burst of old words
      while (SE.wi < W.length && W[SE.wi].start <= mt) {
        const w = W[SE.wi], late = mt - w.start > 0.3;
        if (!late && !SE.on) { SE.on = true; SE.quiet = 0; SE.qT = 0; emit("speechstart", { t: lifeT(), mediaTime: mt, timed: true, words: W.length }); }
        if (!late) { if (w.phraseStart) { SE.phrase = w.phrase; emit("phrase", { wordIndex: SE.wi, t: w.start, phrase: w.phrase }); } emit("word", { index: SE.wi, text: w.text, t: w.start, end: w.end, stress: w.stress, phrase: w.phrase }); }
        SE.wi++;
      }
    }
    SE.lastMt = mt;
  }
  // clip analysis: words (timed tracks), and per-10 ms RMS / F0 once the clip's samples are known (setClipAudio, or decoded from the attached media element's src)
  const clipA = { words: [], key: null, a: null, busy: false };
  function clipAnalysis() {
    if (!clipA.a && !clipA.busy && source && source.el && (source.el.currentSrc || source.el.src) && source.ctx && source.ctx.decodeAudioData) {
      const url = source.el.currentSrc || source.el.src; clipA.busy = true; clipA.key = url;
      fetch(url).then(r => r.arrayBuffer()).then(b => source.ctx.decodeAudioData(b)).then(ab => { if (clipA.key !== url) return; clipA.a = analysePcm(ab.getChannelData(0), ab.sampleRate); emit("analysis", api.getClipAnalysis()); })
        .catch(e => { console.warn("OpenAvatar: clip analysis failed:", e && e.message); }).finally(() => { if (clipA.key === url) clipA.busy = false; });
    }
    const a = clipA.a, ph = []; for (const w of clipA.words) { if (w.phraseStart || !ph.length) ph.push({ index: w.phrase, start: w.start, end: w.end, wordIndex: clipA.words.indexOf(w) }); else ph[ph.length - 1].end = w.end; }
    if (!a && !clipA.words.length) return null;
    return { words: clipA.words.map(w => ({ ...w })), phrases: ph, duration: a ? a.duration : (clipA.words.length ? clipA.words[clipA.words.length - 1].end : null), hop: a ? a.hop : null, rms: a ? a.rms : null, f0: a ? a.f0 : null, sr: a ? a.sr : null };
  }
  function trackAt(fp, j0, j1) { const tr = rig.track; j1 = Math.min(j1, tr[0].length); const F = D.frames, f0 = Math.max(0, Math.min(F - 1, Math.floor(fp))), f1 = Math.min(F - 1, f0 + 1), u = Math.max(0, Math.min(1, fp - f0)); for (let j = j0; j < j1; j++) rigW[j] = tr[f0][j] * (1 - u) + tr[f1][j] * u; }
  const speechBuf = new Float32Array(Math.max(Ks, 1));
  function computeWeights(dt) {
    if (!named) { if (rig.track) trackAt(Math.min(phase * D.fps, D.frames - 1), 0, Km); return; }
    const t = idle.t, cut = speechLive(dt); stepIdle(dt, cut);
    // components after the speech block: head components play the clip's own track back and forth; free ones sit at rest (plus idle gaze)
    if (Kh > 0 && rig.track) { let p = (t * D.fps) % (2 * (D.frames - 1)); if (p > D.frames - 1) p = 2 * (D.frames - 1) - p; trackAt(p, Ks, Ks + Kh); }
    for (let j = Ks + Kh; j < Km; j++) rigW[j] = restOf(j);
    if (gi.x >= 0) { rigW[gi.x] = idle.gazeVal[0]; rigW[gi.y] = idle.gazeVal[1]; }
    // speech
    speechBuf.fill(0); if (driver) driver.update(dt, speechBuf);
    relMouth.step(dt, speechBuf, cut); for (let j = 0; j < Ks; j++) speechBuf[j] = clamp(speechBuf[j], 0, 1);   // the mouth eases to rest (or to the new clip) after a cut
    const Dv = preset.driver;
    for (let j = 0; j < Ks; j++) {
      let v = speechBuf[j]; if (Dv.clampToTrained && stats) v = clamp(v, stats.lo[j], stats.hi[j]);
      const mx = Dv.maxPerControl && Dv.maxPerControl[controlNames[j]]; if (typeof mx === "number") v = Math.min(v, mx);
      rigW[j] = v;
    }
    motionStep(dt);                                              // layers and primitives (lifelike): pose offsets, control deltas, gaze target, blink triggers
    const bv = blinkValue(dt); if (bv !== null) rigW[ctlIndex("blink")] = bv;
    if (orbitOn && eff.exprGain < 1) { for (let j = 0; j < Ks; j++) rigW[j] *= eff.exprGain; for (const q of [gi.x, gi.y]) if (q >= 0) { const r0 = rig.rest ? rig.rest[q] : 0; rigW[q] = r0 + (rigW[q] - r0) * eff.exprGain; } }   // manual values fade too (set below)
    for (const k in manual) { const q = ctlIndex(k); rigW[q] = orbitOn && q < Ks ? manual[k] * eff.exprGain : manual[k]; }
    speechEvents(dt);
  }

  // ---- rendering
  let dirty = true, playing = true, lastNow = 0, raf = 0, destroyed = false, fps = 0, frameCount = 0, fpsT = 0, count = 0, lastFrame = null, cpuMs = 0, cpuLast = 0, skipped = 0;
  let lastSig = null, lastBg = null;                            // what the last frame depended on: a loop frame with the same inputs is not drawn again (the canvas keeps its picture)
  const fixed = options.size || null, pixelRatio = options.pixelRatio;
  function resize() {
    const d0 = pixelRatio || Math.min(globalThis.devicePixelRatio || 1, 2), ssf = +RS.supersample, dss = ssf > 1 && !pixelRatio ? Math.min(d0 * ssf, Math.max(d0, +RS.maxDpr || 3)) : d0;
    XT.ss = RS.detail ? dss / d0 : 1; const dpr = RS.detail ? d0 : dss, w = fixed ? fixed.width : Math.round(canvas.clientWidth * dpr), h = fixed ? fixed.height : Math.round(canvas.clientHeight * dpr);
    if (w < 1 || h < 1) return false;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; gl.viewport(0, 0, w, h); dirty = true; lastSig = null; } return true;
  }
  const hexRgb = s => { s = String(s).trim(); if (/^#[0-9a-f]{3}$/i.test(s)) s = "#" + [...s.slice(1)].map(c => c + c).join(""); const m = /^#([0-9a-f]{6})$/i.exec(s) || [0, "000000"]; return [0, 2, 4].map(i => parseInt(m[1].slice(i, i + 2), 16) / 255); };
  function applyPose() {
    [base.yaw, base.pitch] = clampAngles(base.yaw, base.pitch);
    if (orbitOn) { base.orbit = orbitFull ? ((base.orbit + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI : clamp(base.orbit, orb.min, orb.max); eff.orbit = base.orbit; }
    // head follow: with the camera at orbit angle o the head turns on the neck by the same angle (up to followMax), so the face keeps looking at the camera; beyond that the face turns away with the body
    const fol = orbitOn && limits && orb.gainP > 0 ? -ss(orb.gainP) * followCurve(eff.orbit, { mode: orb.mode, rate: orb.rate, max: Math.min(orb.followMax, limits.yaw), fade: orb.followFade }) : 0;   // 0 beyond followFadeDeg[1]: the head was only seen twisted on the front arc, and a turn that stays on at the back would flip sign (a pop) when the orbit crosses 180
    let [y, p] = clampAngles(base.yaw + idle.off[0] + fol, base.pitch + idle.off[1]); eff.zoom = base.zoom;
    if (M.on && limits) {                                        // layers and primitives (lifelike): added inside the room the limits leave, easing into them
      if (M.yaw !== 0) y += softLimit(M.yaw, -limits.yaw - y, limits.yaw - y);
      if (M.pitch !== 0 || p < -downLimit(y)) p = clamp(p, -downLimit(y), limits.pitchUp) + softLimit(M.pitch, -downLimit(y) - clamp(p, -downLimit(y), limits.pitchUp), limits.pitchUp - clamp(p, -downLimit(y), limits.pitchUp));   // (the yaw may have moved the chin-down limit)
      if (M.zoom !== 0) eff.zoom = clamp(base.zoom + M.zoom, 0.25, 2.5);
    }
    eff.yaw = y; eff.pitch = p;
    { const r0 = limits ? clamp(base.roll, limits.roll[0], limits.roll[1]) : 0; eff.roll = M.on && limits && M.roll !== 0 ? r0 + softLimit(M.roll, limits.roll[0] - r0, limits.roll[1] - r0) : r0; }
    // the tilt path (a few degrees of pitch and roll) was measured on stills where the head was turned against a fixed camera. When the head only turns to follow the orbit camera, the face stays square to the camera and the nod must not be applied: it moved the ears and the whole head up and down by up to 6 mm, with a step near 20 degrees. orbit.tiltWithFollow = true restores the old behaviour.
    eff.tiltYaw = orbitOn && !orb.tiltWithFollow ? clampAngles(base.yaw + idle.off[0], 0)[0] : y;
    // face angle relative to the camera (0 = looking into it); expression controls were trained only on the front arc and fade out with it
    eff.face = orbitOn ? eff.yaw + eff.orbit : eff.yaw; eff.face = Math.atan2(Math.sin(eff.face), Math.cos(eff.face));
    eff.exprGain = orbitOn ? 1 - ss((Math.abs(eff.face) - orb.fade[0]) / (orb.fade[1] - orb.fade[0])) : 1;
  }
  function frameSig(tau, V) {                                    // canvas size, clip time, view matrix, head turn, weights
    const hq = head ? head.Q : null, nw = rigW ? rigW.length : 0, sig = new Float64Array(19 + (hq ? 9 : 0) + nw);
    sig[0] = canvas.width; sig[1] = canvas.height; sig[2] = tau; sig.set(V, 3); if (hq) sig.set(hq, 19); if (nw) sig.set(rigW, 19 + (hq ? 9 : 0)); return sig;
  }
  function frameDone(now, info) {
    frameCount++; if (now - fpsT > 1000) { fps = frameCount * 1000 / (now - fpsT); frameCount = 0; fpsT = now; }
    info.fps = fps; info.visible = count; emit("frame", info);
  }
  // eased base pose (spikes/lf_motion): setPose(p, { easeS }) moves the base pose to p through a critically damped follower (drags, a reset); a plain setPose sets at once
  const BE = { on: false, T: 0.03, tgt: {}, e: {} };
  const wrapA = a => Math.atan2(Math.sin(a), Math.cos(a));
  function stepBaseEase(dt) {
    if (!BE.on) return; let live = false;
    for (const k of Object.keys(BE.tgt)) { const s = BE.e[k]; cdStep(s, 0, dt, BE.T); base[k] = BE.tgt[k] + s.x; if (s.x === 0 && s.v === 0) { delete BE.tgt[k]; delete BE.e[k]; } else live = true; }
    BE.on = live; if (!live) BE.tgt = {};
  }
  const poseTgt = k => (BE.on && k in BE.tgt ? BE.tgt[k] : base[k]);   // where the base pose is going (pointer drags add to it)
  function renderFrame(dt, now, exact) {
    const tc0 = performance.now(); stepBaseEase(dt); if (hasRig) computeWeights(dt);
    if (orb) { const g = followTarget(); if (orb.gainP !== g) orb.gainP = orb.easeS > 0 ? (g > orb.gainP ? Math.min(g, orb.gainP + dt / orb.easeS) : Math.max(g, orb.gainP - dt / orb.easeS)) : g; }   // head-follow eases in and out
    applyPose();
    if (hasRig && named) lifeAfterPose(dt);
    const tau = Math.min(phase * D.fps / Math.max(1, D.frames - 1), 1), { V, pos } = cameraMatrices();
    const info = { fps, visible: count, n: nG, yaw: eff.yaw, pitch: eff.pitch, orbit: eff.orbit, speaking: speakingNow() };
    const sig = frameSig(tau, V), same = !exact && bg === lastBg && sameSig(sig, lastSig) && !(RS.post && RS.post.grain > 0), shc = shClampCam(pos, V);   // animated grain redraws every loop frame
    const took = () => { const ms = performance.now() - tc0; cpuMs = cpuMs ? cpuMs + 0.1 * (ms - cpuMs) : ms; cpuLast = ms; };
    lastSig = sig; lastBg = bg;
    if (same) { skipped++; took(); if (!(R && (R.inflight || R.queued))) frameDone(now, info); return; }   // nothing the picture depends on changed: the canvas keeps it (a frame still waiting for the worker reports itself when drawn)
    if (R) {
      const f = { F: frameParams(tau, V, pos), V, pos, bg, now, info, gen: 0, shc };
      if (exact) R.gen++;                                          // a frame drawn now outdates any loop frame still waiting for the worker
      f.gen = R.gen;
      const se = Math.max(1, Math.round(+RS.sortEvery || 1));       // render.sortEvery: draw now with the last order on the frames in between (no wait for the worker)
      if (se > 1 && (!exact || RS.sortEveryExact) && R.order && R.skipN < se - 1 && !R.inflight && !R.queued) { R.skipN++; took(); gpuDraw(f, count); frameDone(now, info); return; }
      R.skipN = 0; R.order = true;
      if (!exact && R.wk) { if (R.inflight) R.queued = f; else gpuPost(f); took(); return; }   // drawn in workerDone
      const t0 = performance.now(); count = sortSplats(R.S, f.F, R.out); R.sortMs = performance.now() - t0; R.nMain++;
      gl.bindBuffer(gl.ARRAY_BUFFER, R.ob); gl.bufferSubData(gl.ARRAY_BUFFER, 0, R.out, 0, count); took(); gpuDraw(f, count);
    } else {
      const xo = shc || RS.shDegree !== 2 || RS.sort === "radial" ? { shc, deg: shDeg(), radial: RS.sort === "radial" } : NO_XO;
      count = evaluate(tau, pos, V, xo); took();
      const x = xOn(), T = x ? target() : { w: canvas.width, h: canvas.height, fbo: null };
      if (!x) { const [r, g, b] = hexRgb(bg); gl.clearColor(r, g, b, 1); gl.clear(gl.COLOR_BUFFER_BIT); } else beginFrame(T, bg, V);
      gl.useProgram(P); gl.uniform1i(uAAc, aaOn() ? 1 : 0);
      const sc = T.h / D.hdr.img_h, f = D.hdr.K[1][1] * sc * tele;   // the training crop fills the canvas height
      gl.uniformMatrix4fv(uView, false, V); gl.uniform2f(uFocal, f, f); gl.uniform2f(uViewport, T.w, T.h);
      gl.uniform2f(uPP, T.w / 2 + (D.hdr.K[0][2] - D.hdr.img_w / 2) * sc, D.hdr.K[1][2] * sc);   // principal point: row 640 of the ring frame, not the crop centre
      gl.bindVertexArray(vao); gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
      if (x) endFrame(T, V, pos);
    }
    frameDone(now, info);
  }
  function loop(now) {
    if (destroyed) return; raf = requestAnimationFrame(loop);
    if (!playing || !resize()) { lastNow = now; return; }
    const dt = lastNow ? Math.min(0.25, (now - lastNow) / 1000) : 0; lastNow = now;
    if (!named) { phase += dt; const per = D.frames / D.fps; if (phase >= per) phase -= per; dirty = true; }   // plain clip models loop their stored motion
    if (named && (preset.idle.enabled || driver || layers.length || M.on || EY.on || EY.off || gazeTarget || idle.blink.pending())) dirty = true;
    if (BE.on) dirty = true;
    if (orb && orb.gainP !== followTarget()) dirty = true;
    if (!dirty) return; dirty = false; renderFrame(dt, now, false);
  }

  // ---- public API
  const api = {
    on,
    get controls() { return ctlInfo.map(c => ({ ...c, value: rigW ? rigW[c.index] : 0 })); },
    get limits() { return { yaw: limits ? [-limits.yaw, limits.yaw] : null, pitch: limits ? [-limits.pitch, limits.pitchUp] : null, roll: limits ? limits.roll.slice() : null, pitchNegFade: limits && limits.negFade ? limits.negFade.slice() : null, zoom: [0.25, 2.5] }; },
    get pose() { return { yaw: eff.yaw, pitch: eff.pitch, roll: eff.roll, zoom: eff.zoom, orbit: eff.orbit, face: eff.face, exprGain: eff.exprGain, tiltYaw: eff.tiltYaw, follow: orb ? ss(orb.gainP) : 0 }; },
    get basePose() { return { yaw: base.yaw, pitch: base.pitch, zoom: base.zoom, orbit: base.orbit }; },
    get orbitLimits() { return orbitOn ? { min: orbitFull ? null : orb.min, max: orbitFull ? null : orb.max, full: orbitFull, followHead: orb.follow, followMode: orb.mode, autoOrbit: orb.auto } : null; },   // what setPose / dragging set, without idle sway
    get info() { return { aspect: D.hdr.img_w / D.hdr.img_h, n: nG, visible: count, frames: D.frames, fps: D.fps, version: D.version, named, hasRig, head: !!head, tele }; },
    get stats() { return { fps, visible: count, n: nG, cpuMs, cpuLastMs: cpuLast, renderer: R ? (R.pre ? "gpu" : "gpu-1pass") : "cpu", sortMs: R ? R.sortMs : null, sortWorker: !!(R && R.wk), sorts: R ? { main: R.nMain, worker: R.nWorker } : null, skipped }; },   // cpuMs: main-thread milliseconds of a frame (weights, then the CPU renderer's evaluation + sort + upload, or the GPU renderer's depth sort when it is not in the worker, + upload), smoothed; the draw runs on the GPU and is not in it. sortMs: the last depth sort (worker or main thread). sorts: how many depth sorts ran on the main thread / in the worker; skipped: loop frames not drawn because nothing changed
    get speaking() { return speakingNow(); },
    get audioLatency() { return source ? source.lead() : 0; },   // seconds between the audio clock and the ear (output latency + the lookahead delay); the timed driver subtracts it
    setControls(obj, o = {}) {                                   // o.ignoreUnknown: skip names this model does not have (e.g. browUp on an older model) instead of throwing
      if (!named) throw new Error("this model has no named controls");
      for (const [k, v] of Object.entries(obj || {})) { const i = ctlIndex(k); if (i < 0) { if (o && o.ignoreUnknown) continue; throw new Error("unknown control " + k); } if (v === null || v === undefined) delete manual[k]; else manual[k] = clamp(+v, ctlInfo[i].min, ctlInfo[i].max); }
      dirty = true;
    },
    clearControls() { for (const k of Object.keys(manual)) delete manual[k]; dirty = true; },
    // orbit rigs only. setFollowHead: the head turns part of the way toward the camera near the front while the user turns the bust. setAutoOrbit(true): the page
    // is turning the bust by itself, so it turns rigidly (no follow) until setAutoOrbit(false). setAutoOrbit eases over rig.orbit.followEaseS unless `now` is true;
    // setFollowHead acts at once as it always did, unless `ease` is true (a checkbox on a live page).
    setFollowHead(on, ease) { if (orb) { orb.follow = !!on; if (!ease) orb.gainP = followTarget(); applyPose(); dirty = true; } return !!orb && orb.follow; },
    setAutoOrbit(on, now) { if (orb) { orb.auto = !!on; if (now) orb.gainP = followTarget(); applyPose(); dirty = true; } return !!orb && orb.auto; },
    // o.easeS (spikes/lf_motion, with preset.motion on): go there through a critically damped follower with this time constant instead of at once (a drag, a reset button)
    setPose(p, o) {
      const ease = o && +o.easeS > 0 && moOn() ? +o.easeS : 0, set = (k, val) => {
        if (!ease) { base[k] = val; if (k in BE.tgt) { delete BE.tgt[k]; delete BE.e[k]; } return; }
        const s = BE.e[k] || (BE.e[k] = { x: 0, v: 0 }); s.x = k === "orbit" && orbitFull ? wrapA(base[k] - val) : base[k] - val; BE.tgt[k] = val; BE.T = ease; BE.on = true;   // keeps the shown pose and its speed
      };
      if (p.yaw !== undefined) set("yaw", +p.yaw); if (p.pitch !== undefined) set("pitch", +p.pitch); if (p.zoom !== undefined) set("zoom", clamp(+p.zoom, 0.25, 2.5)); if (p.orbit !== undefined && orbitOn) set("orbit", +p.orbit); if (p.roll !== undefined && Number.isFinite(+p.roll)) set("roll", +p.roll);
      applyPose(); dirty = true; return api.pose;
    },
    getPreset() { return JSON.parse(JSON.stringify(preset)); },
    // render options (RENDER_DEFAULTS): merged onto the current ones; null turns catchlights / post off; quadExtent is fixed at create
    setRender(o) { RS = readRender(RS, o || {}); lastSig = null; dirty = true; return api.getRender(); },
    getRender() { return JSON.parse(JSON.stringify({ ...RS, quadExtent: QEXT, antialiasedModel: hdrAA, aaActive: aaOn() })); },
    setPreset(p) { if (p && p.render) api.setRender(p.render); const next = mergeDeep(DEFAULT_PRESET, p || {}); const was = preset.driver.name; preset = next; if (was !== next.driver.name || !driver) selectDriver(); if (source && !source.fixedLookahead) source.setLookahead(preset.driver.lookahead); dirty = true; return api.getPreset(); },
    getPresetName() { return preset.driver.name; },
    setDriver(name, o = {}) {                                    // settings go into the preset; { track } / { timings } are runtime data for the timed driver
      const { track, timings, phones, words, ...settings } = o; if (name && name !== "none" && !driverRegistry[name]) throw new Error("unknown driver " + name);
      const d = mergeDeep(preset.driver, settings); d.name = name; preset.driver = d;
      if (track !== undefined || timings !== undefined || phones !== undefined || words !== undefined) { driverData = { track, timings, phones, words }; setWords(driverData); }
      selectDriver(); dirty = true; return api.getPreset();
    },
    setTrack(d) { driverData = d; setWords(d); speechCut = true; if (driver && driver.setData) driver.setData(d); dirty = true; },
    attachAudio(src, o = {}) {
      api.detachAudio(); clipA.a = null; clipA.key = null; clipA.busy = false; source = makeSource(src, { ...o, fixedLookahead: typeof o.lookahead === "number", lookahead: o.lookahead ?? (typeof HTMLMediaElement !== "undefined" && src instanceof HTMLMediaElement ? preset.driver.lookahead : undefined) });
      const set = {}; if (o.gain !== undefined) set.gain = o.gain; if (o.driver) { api.setDriver(o.driver, set); } else if (o.gain !== undefined) preset.driver.gain = o.gain;
      if (!driver) selectDriver(); if (driver) driver.attach(source); dirty = true; return api;
    },
    detachAudio() { speechCut = true; if (source) { source.detach(); source = null; } if (driver) driver.detach(); },
    setBackground(c) { bg = c; dirty = true; },
    // ---- lifelike (spikes/lf_core/API.md). Layers: fn(ctx) each frame -> { yaw, pitch, roll (radians; or yawDeg / pitchDeg / rollDeg), zoom, controls: {name: delta},
    // gaze: {x, y} degrees | "camera" | null, blinkNow: true | "half" | "long" | {kind, ...} } (all optional, additive; the last layer's gaze wins). ctx = { t, dt, speaking,
    // driverWeights, pose, state }. Returns a function that removes the layer.
    addLayer(fn) { if (typeof fn !== "function") throw new Error("addLayer needs a function"); if (!layers.includes(fn)) layers.push(fn); dirty = true; return () => api.removeLayer(fn); },
    removeLayer(fn) { const i = layers.indexOf(fn); if (i >= 0) layers.splice(i, 1); dirty = true; return i >= 0; },
    setGazeTarget(g) { gazeTarget = g === "camera" ? [0, 0] : g && Number.isFinite(+g.x) && Number.isFinite(+g.y) ? [+g.x, +g.y] : null; dirty = true; return gazeTarget ? { x: gazeTarget[0], y: gazeTarget[1] } : null; },
    get gazeTarget() { return gazeTarget ? { x: gazeTarget[0], y: gazeTarget[1] } : null; },
    get gazeCalibration() {                                       // degrees per control unit and the trained range in degrees around rest (null without gaze controls)
      if (gi.x < 0 || gi.y < 0) return null; const r = j => { const c = ctlInfo[j], k = j === gi.x ? cal.x : cal.y; return c.trained ? [(c.trained[0] - c.rest) * k, (c.trained[1] - c.rest) * k] : null; };
      return { xDegPerUnit: cal.x, yDegPerUnit: cal.y, xRangeDeg: r(gi.x), yRangeDeg: r(gi.y) };
    },
    // a nod "up, down, settle" (degrees; down = the chin-down part): { deg: 4 (total, split up 1 : down 3), up, down, durS: 0.8, pivot: m below the head pivot (0 = about it; ~0.08 = neck base) }
    nod(o = {}) { const deg = +(o.deg ?? 4), up = +(o.up ?? deg * 0.25), down = +(o.down ?? deg * 0.75); prim.nods.push({ t0: lifeT(), dur: Math.max(0.2, +(o.durS ?? 0.8)), up, down, pivot: Math.max(0, +(o.pivot ?? 0)) }); dirty = true; },
    // listening tilt: roll toward rollDeg (+ = the top of the head toward the picture's left), eased over o.easeS (0.8); tilt(0) to come back. Held until changed.
    tilt(rollDeg, o = {}) { prim.tilt.to = (+rollDeg || 0) * DEG; prim.tilt.T = +(o.easeS ?? 0.8); dirty = true; },
    // lean in: the camera comes `amount` closer (0.03 = 3 %), eased over o.easeS (0.6); lean(0) to come back. Held until changed.
    lean(amount, o = {}) { prim.lean.to = -(+amount || 0) * base.zoom; prim.lean.T = +(o.easeS ?? 0.6); dirty = true; },
    // a blink now (after a running one): { kind: "full" | "half" | "long" (~0.6 s), depth, closeS, holdS, openS }; only a model with a blink control; not while a page holds it
    blink(o = {}) { if (ctlIndex("blink") < 0) return false; idle.blink.trigger(o.kind || "full", o); dirty = true; return true; },
    controlNames() { return ctlInfo.map(c => c.name); },
    getClipAnalysis() { return clipAnalysis(); },
    setClipAudio(pcm, sr) { if (pcm && pcm.getChannelData) { sr = pcm.sampleRate; pcm = pcm.getChannelData(0); } clipA.key = "pcm"; clipA.busy = false; clipA.a = pcm && sr > 0 ? analysePcm(pcm, sr) : null; return api.getClipAnalysis(); },
    pause() { playing = false; }, resume() { playing = true; lastNow = 0; dirty = true; },
    renderOnce(dt = 0) { resize(); renderFrame(dt, performance.now(), true); return count; },   // draw now, whatever the loop is doing, with this frame's exact depth order (used by tests)
    getWeights() { return rigW ? Array.from(rigW) : null; },
    readPixels() { return canvas; },
    enablePointer() {                                            // drag to turn the head (or orbit), wheel / pinch to zoom, as in the prototype
      canvas.style.touchAction = "none"; const ptr = new Map(); let pinch = 0;
      const down = e => { canvas.setPointerCapture(e.pointerId); ptr.set(e.pointerId, [e.clientX, e.clientY]); if (ptr.size === 2) { const [a, b] = [...ptr.values()]; pinch = Math.hypot(a[0] - b[0], a[1] - b[1]); } };
      const move = e => {
        if (!ptr.has(e.pointerId)) return; const p = ptr.get(e.pointerId), dx = e.clientX - p[0], dy = e.clientY - p[1]; ptr.set(e.pointerId, [e.clientX, e.clientY]);
        const MO = moOn(), eo = MO && +MO.pointerS > 0 ? { easeS: +MO.pointerS } : undefined;   // motion.pointerS: the drag eases (from where it is going, so no drag distance is lost)
        if (ptr.size === 1) {
          if (orbitOn) { let o2 = poseTgt("orbit") - dx * 0.008; if (eo && !orbitFull) o2 = clamp(o2, orb.min, orb.max); api.setPose({ orbit: o2, pitch: clamp(poseTgt("pitch") + dy * 0.004, -0.5, 0.5) }, eo); }
          else { let y2 = poseTgt("yaw") - dx * 0.006, p2 = clamp(poseTgt("pitch") + dy * 0.004, -1, 1); if (eo) [y2, p2] = clampAngles(y2, p2); api.setPose({ yaw: y2, pitch: p2 }, eo); }
        }
        else if (ptr.size === 2) { const [a, b] = [...ptr.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]); if (pinch) api.setPose({ zoom: poseTgt("zoom") * pinch / d }, eo); pinch = d; }
      };
      const up = e => { ptr.delete(e.pointerId); pinch = 0; }, wheel = e => { e.preventDefault(); const MO = moOn(); api.setPose({ zoom: poseTgt("zoom") * Math.exp(e.deltaY * 0.001) }, MO && +MO.pointerS > 0 ? { easeS: 2 * MO.pointerS } : undefined); };
      canvas.addEventListener("pointerdown", down); canvas.addEventListener("pointermove", move); canvas.addEventListener("pointerup", up); canvas.addEventListener("pointercancel", up); canvas.addEventListener("wheel", wheel, { passive: false });
      const off = () => { canvas.removeEventListener("pointerdown", down); canvas.removeEventListener("pointermove", move); canvas.removeEventListener("pointerup", up); canvas.removeEventListener("pointercancel", up); canvas.removeEventListener("wheel", wheel); };
      disposers.push(off); return off;
    },
    destroy() {
      if (destroyed) return; destroyed = true; cancelAnimationFrame(raf); api.detachAudio(); while (disposers.length) disposers.pop()();
      if (R && R.wk) { R.wk.w.terminate(); URL.revokeObjectURL(R.wk.url); R.wk = null; }
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
// a driver on its own, for tests and for pages that want the weights without a viewer: makeDriver(name, { controls, Ks, index(name), settings() })
function makeDriver(name, env) { const f = driverRegistry[name]; if (!f) throw new Error("unknown driver " + name); return f(env); }
export const OpenAvatarViewer = { create, RENDER_DEFAULTS, RENDER_PROFILES, DETAIL_DEFAULTS, readDetail, CATCHLIGHT_DEFAULTS, PORTRAIT_POST, catchlightPoint, GAZE_CAL, FREE_HEAD, ROLL_DEFAULT_DEG, nodCurve, wordsFromTimings, analysePcm, softLimit, registerDriver, makeDriver, makeRelease, makeBlinker, blinkSweepWeight, blinkSweepHeight, readBlinkSweep, SPEECH_RELEASE_S, SPEECH_RELEASE_MOUTH_S, followCurve, FORMAT_MAJOR, DEFAULT_PRESET, VISEME_WEIGHTS, lettersToVisemes, timingsToPhones, wordToPhones, phonesToTrack,
  setLexicon, loadLexicon, lexiconInfo };
export default OpenAvatarViewer;
