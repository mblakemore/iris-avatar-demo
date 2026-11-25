# Iris — OpenAvatar demo

A talking head-and-shoulders avatar generated from a single portrait: a 3D Gaussian-splat model (about 124,000 Gaussians) with a learned rig of mouth shapes, eyes, brows, breathing and seven trained facial expressions (brow raise, frown, eye smile, open smile, wide eyes, sad, smiling speech), rendered live in the browser with WebGL 2 (no libraries, no video, no server).

- Drag to turn her head (drag up and she looks up); scroll or pinch to zoom. A few seconds after you let go, she turns back to face you.
- Press **Hear her speak**, or click her, to play a sample line. She takes a short thinking beat and a breath, then answers; lip movement follows the line's word timings (lips close on p, b and m), her head and brows follow the pitch of her voice, and she looks back at you as she finishes ("your turn").
- **Delivery** plays the same recording four ways (Warm, Cheerful, Curious, Concerned; keys 1-4). Each is the line written with expression tags, as an LLM would write them (`[smile]`, `[laugh]`, `[brow]`, `[frown]`, `[wide]`, `[nod]`, `[thinks]`), plus a mood. The caption shows the tags and highlights each word as she says it.
- **Expressions** plays one trained expression on demand. They blend with her blinks, breathing and gaze rather than replacing them.
- **Portrait look** turns catchlights in her eyes and a soft vignette / grade on or off (on by default). Desktop browsers also render at 1.5x for smoother edges.
- She is alive between lines: she breathes, blinks, shifts her gaze, glances at your pointer when it comes near her face, and notices when you come back to the tab.

Files: `index.html` (the page), `openavatar-viewer.js` (the viewer), `openavatar-behaviour.js` (states, gaze, breathing, prosody, persona), `iris.bin` (the avatar, about 47 MB, front-only v23: with every expression at 0 it renders exactly like v22b), `sample.mp3` / `sample.json` (the sample line and its character timings), `lexicon/` (CMU Pronouncing Dictionary, BSD-2-Clause, see `lexicon/LICENSE-cmudict.txt`).

This repository only hosts the demo page. All rights reserved for now; a licence will be chosen when the main project is released.
