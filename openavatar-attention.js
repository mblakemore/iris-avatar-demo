// OpenAvatar attention: where the USER is, for an avatar that looks at them. Optional, opt-in, on-device.
// MediaPipe Face Landmarker (Apache-2.0, WASM, CPU) runs on the camera frames IN THIS PAGE; frames are never copied out, stored or sent.
// Nothing runs and no camera is opened before an explicit start(). state() feeds an on-screen indicator; stop() closes what start() opened.
// The vendored MediaPipe bundle has its usage-statistics logger disabled (web/vendor/SOURCES.md), so the page makes no network request
// except loading the files below.
//
// Output per processed frame ("face"): x, y = where the avatar should look to meet the user's eyes, in DEGREES relative to looking straight
// out of the screen (x > 0: the user is to the right as the user sees the screen; y > 0: above), with the camera's offset from the avatar's
// eyes corrected using the face's distance (from the eye spacing). Also the user's head angles, whether they look at the screen, smile, blink.
// API: working-docs/spikes/lf_attend/API.md. Pure parts are exported for tests.

export const DEFAULTS = {
  fps: 10,                       // detection rate while the user moves
  stillFps: 5,                   // ... while they hold still (gaze target moved < stillDeg in the last second, no smile change); 0 = always fps
  absentFps: 4,                  // ... while nobody is there
  stillDeg: 1.5,
  inputWidth: 320,               // canvases / images wider than this are scaled down (on a CPU canvas) before detection: fewer pixels to
                                 // upload and convert; the face models work on 128 / 256 px crops anyway. 0 = full size. Video elements go
                                 // to MediaPipe as they are (the browser's video -> texture path is cheaper than a CPU copy) unless
  scaleVideo: false,             // ... this is true
  hfovDeg: 60,                   // the camera's HORIZONTAL field of view; typical built-in webcams 55-70 (wrong by 10 % -> ~2.5 deg at 25 deg)
  cameraOffsetCm: { x: 0, y: 0 },// where the camera is relative to the avatar's eyes on screen (x right as the user sees it, y up), cm
  ipdCm: 6.3,                    // assumed distance between the user's eye centres (adult mean); sets the distance estimate
  defaultDistanceCm: 60,         // used when no distance can be measured
  mirror: false,                 // true if the frames you pass are already mirrored (a mirrored canvas); camera tracks are not
  lookToleranceDeg: 12,          // "looking at the screen": head + eye direction within this angle of the avatar
  eyeRangeDeg: 30,               // eyeLookIn/Out/Up/Down = 1 corresponds to this many degrees of eye rotation (rough)
  presentAfterS: 0.4, absentAfterS: 1.5,
  smileOn: 0.45, smileOff: 0.3,
  lookOnS: 0.2, lookOffS: 0.4,   // debounce of looking / looking away
  smoothing: { minCutoff: 1.0, beta: 0.05 },   // one-euro filter on x / y (Hz, per deg/s)
  numFaces: 1,                   // > 1: the largest (nearest) face wins
  visionUrl: "/vendor/mediapipe/vision_bundle.mjs",
  wasmBase: "/vendor/mediapipe/wasm",
  modelUrl: "/vendor/mediapipe/face_landmarker.task",
  delegate: "CPU",
  video: null,                   // an HTMLVideoElement (or anything detectForVideo takes) you own; else start() opens the camera
  constraints: { video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }, audio: false },
};

const DEG = 180 / Math.PI, RAD = Math.PI / 180;
// eye corners in the 478-point face mesh: subject's right eye outer 33 / inner 133, left eye inner 362 / outer 263
const R_OUT = 33, R_IN = 133, L_IN = 362, L_OUT = 263;

// ---- pure geometry ------------------------------------------------------------------------------------------------------------------------
// landmarks: [{x, y, z}] normalised (x by width, y by height, z ~ x scale). Returns eye midpoint in pixels and the eye spacing in pixels,
// measured in 3D so that it does not shrink when the head turns.
export function faceGeometry(lm, w, h) {
  const c = (a, b) => ({ x: (lm[a].x + lm[b].x) / 2 * w, y: (lm[a].y + lm[b].y) / 2 * h, z: (lm[a].z + lm[b].z) / 2 * w });
  const r = c(R_OUT, R_IN), l = c(L_IN, L_OUT);
  return { cx: (r.x + l.x) / 2, cy: (r.y + l.y) / 2, ipdPx: Math.hypot(r.x - l.x, r.y - l.y, r.z - l.z), ipd2dPx: Math.hypot(r.x - l.x, r.y - l.y) };
}
export function focalPx(w, hfovDeg) { return w / 2 / Math.tan(hfovDeg * RAD / 2); }
// pixel position + eye spacing -> the user's eye midpoint in "screen" coordinates (cm; X right as the user sees the screen, Y up,
// Z = distance out of the screen), and the gaze target angles for the avatar
export function userTarget(g, w, h, o = DEFAULTS) {
  const f = focalPx(w, o.hfovDeg ?? DEFAULTS.hfovDeg), ipd = o.ipdCm ?? DEFAULTS.ipdCm;
  let u = (g.cx - w / 2) / f; const v = (g.cy - h / 2) / f;
  if (!o.mirror) u = -u;                                         // camera frames are unmirrored: the user's right is on the image's left
  // pinhole: an eye pair at depth Z spans f * ipd / Z pixels (the 3D spacing does not shrink when the head turns)
  const Z = g.ipdPx > 1 ? f * ipd / g.ipdPx : (o.defaultDistanceCm ?? DEFAULTS.defaultDistanceCm);
  const off = o.cameraOffsetCm || { x: 0, y: 0 };
  const X = u * Z + (off.x || 0), Y = -v * Z + (off.y || 0);
  return { x: Math.atan2(X, Z) * DEG, y: Math.atan2(Y, Z) * DEG, X, Y, Z, camX: Math.atan(u) * DEG, camY: Math.atan(-v) * DEG };
}
// Screen coordinates used below: X = the user's right as they face the screen, Y = up, Z = out of the screen toward the user (cm).
// MediaPipe camera coordinates: x = image right, y = up, the camera looks down -z (the user is at z < 0); for unmirrored frames
// X = -x, Y = y, Z = -z.
// headPose(matrix): the user's head angles from the facial transformation matrix (4x4, column-major, canonical face -> camera; the canonical
// face looks along its +z). yaw > 0 = face turned toward the user's own right, pitch > 0 = chin up, roll > 0 = head tilted toward the user's
// right shoulder. forward = where the face points (screen coordinates; facing the screen = (0, 0, -1)).
export function headPose(m, mirror = false) {
  const R = (r, c) => m[c * 4 + r], sx = mirror ? 1 : -1;
  const F = { X: sx * R(0, 2), Y: R(1, 2), Z: -R(2, 2) }, U = { X: sx * R(0, 1), Y: R(1, 1), Z: -R(2, 1) };
  const yaw = Math.atan2(F.X, -F.Z) * DEG, pitch = Math.atan2(F.Y, Math.hypot(F.X, F.Z)) * DEG, roll = Math.atan2(U.X, U.Y) * DEG;
  return { yaw, pitch, roll, forward: F };
}
// eye rotation from the blendshapes, degrees: x > 0 = toward the user's right, y > 0 = up. (Blendshape names refer to the subject's eyes:
// right eye "out" and left eye "in" both turn toward the subject's right.)
export function eyeGaze(bs, rangeDeg = 30) {
  const g = n => bs[n] || 0;
  const right = (g("eyeLookOutRight") - g("eyeLookInRight") + g("eyeLookInLeft") - g("eyeLookOutLeft")) / 2;
  const up = (g("eyeLookUpLeft") + g("eyeLookUpRight") - g("eyeLookDownLeft") - g("eyeLookDownRight")) / 2;
  return { x: right * rangeDeg, y: up * rangeDeg };
}
// angle (deg) between where the user looks (head yaw / pitch + eye rotation) and the direction from the user's eyes P (screen coords, cm)
// to the avatar's eyes at the origin
export function lookError(P, head, eye = { x: 0, y: 0 }) {
  const yaw = (head.yaw + (eye.x || 0)) * RAD, pitch = (head.pitch + (eye.y || 0)) * RAD;
  const d = { X: Math.sin(yaw) * Math.cos(pitch), Y: Math.sin(pitch), Z: -Math.cos(yaw) * Math.cos(pitch) };
  const n = Math.hypot(P.X, P.Y, P.Z) || 1;
  return Math.acos(Math.max(-1, Math.min(1, -(d.X * P.X + d.Y * P.Y + d.Z * P.Z) / n))) * DEG;
}

// ---- pure state ---------------------------------------------------------------------------------------------------------------------------
export function makeDebounce(onS, offS, init = false) {         // a boolean that changes only after the input held for onS / offS
  let v = init, since = null;
  return { step(t, x) { if (x === v) { since = null; return null; } if (since === null) since = t; if (t - since >= (x ? onS : offS) - 1e-9) { v = x; since = null; return v; } return null; }, get value() { return v; } };
}
export function makeHysteresis(on, off, init = false) {
  let v = init; return { step(x) { const nv = v ? x > off : x >= on; if (nv !== v) { v = nv; return v; } return null; }, get value() { return v; } };
}
export function makeOneEuro({ minCutoff = 1, beta = 0.05, dCutoff = 1 } = {}) {
  let x = null, dx = 0, tp = null;
  const a = (dt, fc) => { const r = 2 * Math.PI * fc * dt; return r / (r + 1); };
  return { step(t, v) { if (x === null || tp === null || t <= tp) { x = v; tp = t; dx = 0; return v; } const dt = t - tp; tp = t;
      const d = (v - x) / dt; dx += a(dt, dCutoff) * (d - dx); const fc = minCutoff + beta * Math.abs(dx); x += a(dt, fc) * (v - x); return x; },
    reset() { x = null; tp = null; dx = 0; } };
}

// one detector result (MediaPipe FaceLandmarkerResult) -> a face record, or null. Pure: tests feed recorded results.
export function analyse(res, w, h, o = DEFAULTS) {
  if (!res || !res.faceLandmarks || !res.faceLandmarks.length) return null;
  let k = 0, best = -1;
  for (let i = 0; i < res.faceLandmarks.length; i++) { const g = faceGeometry(res.faceLandmarks[i], w, h); if (g.ipdPx > best) { best = g.ipdPx; k = i; } }
  const lm = res.faceLandmarks[k], g = faceGeometry(lm, w, h), tg = userTarget(g, w, h, o);
  const bs = {}; const cats = res.faceBlendshapes && res.faceBlendshapes[k] && res.faceBlendshapes[k].categories;
  if (cats) for (const c of cats) bs[c.categoryName] = c.score;
  const mx = res.facialTransformationMatrixes && res.facialTransformationMatrixes[k];
  const head = mx ? headPose(mx.data || mx, o.mirror) : null;
  const eye = cats ? eyeGaze(bs, o.eyeRangeDeg ?? 30) : { x: 0, y: 0 };
  const lookErrDeg = head ? lookError({ X: tg.X, Y: tg.Y, Z: tg.Z }, head, eye) : null;
  return {
    x: tg.x, y: tg.y, distanceCm: tg.Z, camX: tg.camX, camY: tg.camY, px: { x: g.cx, y: g.cy, ipd: g.ipdPx },
    head: head ? { yaw: head.yaw, pitch: head.pitch, roll: head.roll } : null, eye, lookErrDeg,
    smile: ((bs.mouthSmileLeft || 0) + (bs.mouthSmileRight || 0)) / 2, blink: ((bs.eyeBlinkLeft || 0) + (bs.eyeBlinkRight || 0)) / 2,
    blendshapes: bs, faces: res.faceLandmarks.length,
  };
}

// the event logic around analyse(): smoothing, presence, smile and looking hysteresis. step(t, face | null) -> events [[name, payload]]
export function makeTracker(o = DEFAULTS) {
  const fx = makeOneEuro(o.smoothing), fy = makeOneEuro(o.smoothing);
  const present = makeDebounce(o.presentAfterS, o.absentAfterS), smile = makeHysteresis(o.smileOn, o.smileOff), look = makeDebounce(o.lookOnS, o.lookOffS);
  let last = null;
  return {
    step(t, f) {
      const ev = [];
      const p = present.step(t, !!f); if (p !== null) ev.push([p ? "present" : "absent", { t }]);
      if (!f) { if (!present.value) { fx.reset(); fy.reset(); } const l = look.step(t, false); if (l === false) ev.push(["lookaway", { t }]); return ev; }
      const rec = { t, ...f, rawX: f.x, rawY: f.y, x: fx.step(t, f.x), y: fy.step(t, f.y) };
      const lk = f.lookErrDeg !== null && f.lookErrDeg <= (o.lookToleranceDeg ?? 12);
      const l = look.step(t, lk); if (l !== null) ev.push([l ? "look" : "lookaway", { t, lookErrDeg: f.lookErrDeg }]);
      const s = smile.step(f.smile); if (s !== null) ev.push(["smile", { t, smiling: s, score: f.smile }]);
      rec.looking = look.value; rec.smiling = smile.value; rec.present = present.value;
      ev.unshift(["face", rec]); last = rec; return ev;
    },
    get present() { return present.value; }, get looking() { return look.value; }, get smiling() { return smile.value; }, get last() { return last; },
  };
}

// ---- the browser part ---------------------------------------------------------------------------------------------------------------------
export function createAttention(opts = {}) {
  const o = { ...DEFAULTS, ...opts, smoothing: { ...DEFAULTS.smoothing, ...(opts.smoothing || {}) }, cameraOffsetCm: { ...DEFAULTS.cameraOffsetCm, ...(opts.cameraOffsetCm || {}) } };
  const handlers = {}, emit = (n, e) => { const hs = handlers[n]; if (hs) for (const h of [...hs]) { try { h(e); } catch (err) { console.error(err); } } };
  const st = { status: "off", present: false, looking: false, smiling: false, smile: 0, fps: 0, msPerFrame: 0, cpuShare: 0, frames: 0, error: null, camera: false, faces: 0 };
  let recent = [], small = null, smallCtx = null, lm = null, video = o.video, ownStream = null, ownVideo = false, timer = null, tracker = makeTracker(o), lastTs = -1, ms = [], stamps = [], gen = 0;
  const setStatus = s => { if (st.status !== s) { st.status = s; emit("state", { ...st }); } };
  async function load() {
    const vision = await import(/* webpackIgnore: true */ o.visionUrl);
    const fileset = await vision.FilesetResolver.forVisionTasks(o.wasmBase);
    return vision.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: o.modelUrl, delegate: o.delegate }, runningMode: "VIDEO", numFaces: o.numFaces,
      outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
    });
  }
  // one detection on the current frame (also usable directly on canvases / images in tests: process(source, tMs))
  function process(src, tMs) {
    if (!lm) return null;
    let w = src.videoWidth || src.naturalWidth || src.width, h = src.videoHeight || src.naturalHeight || src.height;
    if (!w || !h) return null;
    const a0 = performance.now(); let k = 1;                    // k: source pixels per detector pixel (px in the result are source pixels)
    if (o.inputWidth > 0 && w > o.inputWidth && (o.scaleVideo || !src.videoWidth)) {
      const sw = o.inputWidth, sh = Math.round(h * sw / w);
      if (!small || small.width !== sw || small.height !== sh) {
        small = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(sw, sh) : Object.assign(document.createElement("canvas"), { width: sw, height: sh });
        smallCtx = small.getContext("2d", { willReadFrequently: true, alpha: false });
      }
      smallCtx.drawImage(src, 0, 0, sw, sh); src = small; k = w / sw; w = sw; h = sh;
    }
    const ts = Math.max(tMs, lastTs + 1); lastTs = ts;
    const res = lm.detectForVideo(src, ts); const d = performance.now() - a0;
    ms.push(d); if (ms.length > 30) ms.shift(); stamps.push(a0); if (stamps.length > 30) stamps.shift();
    st.frames++; st.msPerFrame = ms.reduce((x, y) => x + y, 0) / ms.length;
    st.fps = stamps.length > 1 ? (stamps.length - 1) * 1000 / (stamps[stamps.length - 1] - stamps[0]) : 0; st.cpuShare = st.msPerFrame * (st.fps || o.fps) / 1000;
    const f = analyse(res, w, h, o); st.faces = f ? f.faces : 0;
    if (f && k !== 1) f.px = { x: f.px.x * k, y: f.px.y * k, ipd: f.px.ipd * k };
    for (const [n, e] of tracker.step(ts / 1000, f)) emit(n, e);
    st.present = tracker.present; st.looking = tracker.looking; st.smiling = tracker.smiling; st.smile = f ? f.smile : 0;
    return f ? tracker.last : null;
  }
  function loop(g) {
    if (g !== gen || st.status !== "on") return;
    const t0 = performance.now();
    let f = null;
    try { if (video && (video.readyState === undefined || video.readyState >= 2)) f = process(video, t0); }
    catch (err) { st.error = String(err); emit("error", err); }
    // adaptive rate: full rate while the face moves or the smile changes, lower while still or absent
    let rate = o.fps;
    if (!f && !tracker.present && o.absentFps > 0) rate = Math.min(o.fps, o.absentFps);
    else if (f && o.stillFps > 0) {
      recent.push({ t: t0, x: f.rawX, y: f.rawY, s: f.smiling }); while (recent.length && t0 - recent[0].t > 1000) recent.shift();
      let mx = 0; for (const r of recent) mx = Math.max(mx, Math.abs(r.x - f.rawX), Math.abs(r.y - f.rawY));
      if (recent.length >= 3 && mx < o.stillDeg && recent.every(r => r.s === f.smiling)) rate = Math.min(o.fps, o.stillFps);
    }
    st.rate = rate;
    timer = setTimeout(() => loop(g), Math.max(0, 1000 / rate - (performance.now() - t0)));
  }
  return {
    on(n, f) { (handlers[n] ||= new Set()).add(f); return () => handlers[n].delete(f); },
    off(n, f) { if (handlers[n]) handlers[n].delete(f); },
    async start() {
      if (st.status === "on" || st.status === "starting") return;
      const g = ++gen; setStatus("starting"); st.error = null;
      try {
        if (!video && !o.noCamera) {
          ownStream = await navigator.mediaDevices.getUserMedia(o.stream ? undefined : o.constraints);
          const v = document.createElement("video"); v.muted = true; v.playsInline = true; v.srcObject = ownStream; await v.play(); video = v; ownVideo = true; st.camera = true;
        }
        if (!lm) lm = await load();
        if (g !== gen) return;                                    // stopped while starting
        setStatus("on"); loop(g);
      } catch (err) {
        st.error = String(err && err.message || err); setStatus(err && (err.name === "NotAllowedError" || err.name === "SecurityError") ? "denied" : "error"); emit("error", err);
        if (ownStream) { ownStream.getTracks().forEach(t => t.stop()); ownStream = null; }
      }
    },
    stop() {
      gen++; clearTimeout(timer); timer = null;
      if (ownStream) { ownStream.getTracks().forEach(t => t.stop()); ownStream = null; }
      if (ownVideo && video) { video.srcObject = null; video = null; ownVideo = false; }
      st.camera = false;
      if (tracker.present) emit("absent", { t: performance.now() / 1000 });
      tracker = makeTracker(o); st.present = st.looking = st.smiling = false;
      if (lm && o.closeOnStop !== false) { try { lm.close(); } catch {} lm = null; }
      setStatus("off");
    },
    state: () => ({ ...st }),
    process,                                                      // process(source, tMs) after a start({noCamera}) / with your own loop
    async load() { if (!lm) lm = await load(); },
    options: o,
  };
}
export default { createAttention, analyse, makeTracker, faceGeometry, userTarget, headPose, eyeGaze, lookError, makeDebounce, makeHysteresis, makeOneEuro, focalPx, DEFAULTS };
