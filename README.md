# Iris — OpenAvatar demo

**Live demo: https://mblakemore.github.io/iris-avatar-demo/**

A talking head-and-shoulders avatar generated from a single portrait: a 3D Gaussian-splat model (about 124,000 Gaussians) with a learned rig of mouth shapes, eyes, brows, breathing and seven trained facial expressions (brow raise, frown, eye smile, open smile, wide eyes, sad, smiling speech), rendered live in the browser with WebGL 2 (no libraries, no video, no server).

- Drag to turn her head (drag up and she looks up); scroll or pinch to zoom. A few seconds after you let go, she turns back to face you.
- Press **Hear her speak**, or click her, to play a sample line. She takes a short thinking beat and a breath, then answers; lip movement follows the line's word timings (lips close on p, b and m), her head and brows follow the pitch of her voice, and she looks back at you as she finishes ("your turn").
- The line is written with expression tags, as an LLM would write them (`[smile]`, `[brow]`), and the caption highlights each word as she says it.
- She is alive between lines: every few seconds she looks somewhere else for a moment, lets her head drift and tilt, smiles softly, lifts her brows, has a passing thought or notices something, using the model's trained expressions at low strength. It all pauses while she talks or you drag. She also breathes, blinks, shifts her gaze, glances at your pointer when it comes near her face, and notices when you come back to the tab.

Files: `index.html` (the page), `openavatar-viewer.js` (the viewer), `openavatar-behaviour.js` (states, gaze, breathing, prosody, persona), `iris.bin` (the avatar, about 47 MB, front-only v23d: the ear area retrained on extra in-between-angle stills, plus the ear and cheek colour refit), `sample.mp3` / `sample.json` (the sample line and its character timings), `lexicon/` (CMU Pronouncing Dictionary, BSD-2-Clause, see `lexicon/LICENSE-cmudict.txt`).

This repository only hosts the demo page. All rights reserved for now; a licence will be chosen when the main project is released.
