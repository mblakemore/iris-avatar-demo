# Vendored third-party files

Loaded only after you press "Let her see you" (camera attention). Nothing here is fetched from a CDN.

## mediapipe/ - MediaPipe Tasks Vision 1.2.3 + Face Landmarker model (Apache-2.0)

| file | bytes | sha256 |
|---|---:|---|
| `vision_bundle.mjs` (MODIFIED, see below) | 156351 | `6fae20b1ac820ffe665f38d46be8e6cb646ea3f0ddcc0ea1062cb5d96d33470b` |
| `wasm/vision_wasm_internal.js` | 336624 | `2aa5fd3fdd3d8208cd7587df0bb19fb6925c4ea16a76dd025aedc1f64d6a6309` |
| `wasm/vision_wasm_internal.wasm` | 14665716 | `225bc06c0f51a9cf31fea0e9e0b0243e9e01fb5ef332cc1a7fa6656dbdfdaabf` |
| `face_landmarker.task` (float16, v1) | 3758596 | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |
| `LICENSE` (Apache License 2.0) | 12331 | `8707eef0533987efc5b155d64761eeb6e20793f50b9bd1a68dad1cf4719d0ed8` |

- Code: npm `@mediapipe/tasks-vision@1.2.3` (https://www.npmjs.com/package/@mediapipe/tasks-vision, published 2026-11-11), files from the
  package root and `wasm/`. `package.json` says `"license": "Apache-2.0"`; the package carries no LICENSE file, so `LICENSE` is the
  Apache-2.0 text from https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE.
- **Modification (Apache-2.0 section 4(b))**: upstream `vision_bundle.mjs` (sha256 `fb0c02c296f44ce0cc324ba16bd646d8773eb4a7847db4019cc4d0430db7a9b5`)
  contains a usage-statistics logger that POSTs task type, running mode, OS, library version and latency counters to
  `https://odml.pa.googleapis.com/v1/log` every 60 s (with an API key decoded from the wasm). OpenAvatar promises that the camera feature
  is on-device only, so the logger's `send()` now returns a failure at once and never calls `fetch` (marker `OPENAVATAR-PATCH`, plus a
  header comment). Nothing else is changed. To update: take the new bundle, find the logger's `send(t,e,r){` (the class right before the
  `odml.pa.googleapis.com` URL) and add the same early return; the vendor test fails until the marker is back.
- Model: `face_landmarker.task` from https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task,
  byte-identical to the copy linked from the MediaPipe docs at https://huggingface.co/litert-community/MediaPipe-FaceLandmarker
  (model card `license: apache-2.0`). The three upstream model cards in that repo (BlazeFace short range, Face Mesh V2, Blendshape V2 PDFs)
  each state "LICENSED UNDER Apache License, Version 2.0".
- Not vendored: `vision_bundle.cjs/.js`, source maps (the bundle's `sourceMappingURL` comment points at a map that is not here: harmless),
  `vision_wasm_module_internal.*` (an alternative build).

