# Iris — OpenAvatar demo

A talking head-and-shoulders avatar rendered live in the browser with WebGL 2 (3D Gaussian splatting, no libraries).

- Drag to turn her head; scroll or pinch to zoom.
- Press **Hear her speak**, or click her, to play a sample line. Lip movement follows the line's word timings (lips close on p, b and m).
- She blinks and breathes on her own.

Files: `index.html` (the page), `openavatar-viewer.js` (the viewer module), `iris.bin` (the avatar, about 16 MB, front-only v19 small), `sample.mp3` (the sample line), `sample.json` (its character timings), `lexicon/` (CMU Pronouncing Dictionary, BSD-2-Clause, see `lexicon/LICENSE-cmudict.txt`).

This repository only hosts the demo page. All rights reserved for now; a licence will be chosen when the main project is released.
