# Iris — OpenAvatar demo

A talking head-and-shoulders avatar rendered live in the browser with WebGL 2 (3D Gaussian splatting, no libraries).

- Drag to turn her head; scroll or pinch to zoom. A few seconds after you let go, she turns back to face you.
- Press **Hear her speak**, or click her, to play a sample line. She takes a short thinking beat and a breath, then answers; lip movement follows the line's word timings (lips close on p, b and m), her head and brows follow the pitch of her voice, and she looks back at you as she finishes ("your turn").
- She is alive between lines: she breathes, blinks, shifts her gaze, glances at your pointer when it comes near her face, and notices when you come back to the tab.
- **Talk to her** (optional, microphone): she listens while you talk, with a listening tilt and nods at your pauses. Only the loudness of your voice is used, in your browser; she does not answer yet.
- **Let her see you** (optional, camera, asks first): she looks at where you actually are and reacts when you smile or look away. The image is analysed in your browser with MediaPipe; nothing is recorded or sent, and MediaPipe's usage logging is disabled in the copy here.

Files: `index.html` (the page), `openavatar-viewer.js` (the viewer), `openavatar-behaviour.js` (states, gaze, breathing, prosody, persona), `openavatar-listen.js` (microphone listening), `openavatar-attention.js` (camera attention), `iris.bin` (the avatar, about 30 MB, front-only v22: v21b plus brow, frown and breathing controls), `sample.mp3` / `sample.json` (the sample line and its character timings), `lexicon/` (CMU Pronouncing Dictionary, BSD-2-Clause, see `lexicon/LICENSE-cmudict.txt`), `vendor/mediapipe/` (MediaPipe Tasks Vision and the Face Landmarker model, Apache-2.0, see `vendor/SOURCES.md`).

This repository only hosts the demo page. All rights reserved for now; a licence will be chosen when the main project is released. Third-party files keep their own licences.
