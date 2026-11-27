# Iris — OpenAvatar demo

**Live demo: https://mblakemore.github.io/iris-avatar-demo/**

A talking head-and-shoulders avatar generated from a single portrait: a 3D Gaussian-splat model (about 124,000 Gaussians) with a learned rig of mouth shapes, eyes, brows, breathing and seven trained facial expressions (brow raise, frown, eye smile, open smile, wide eyes, sad, smiling speech), rendered live in the browser with WebGL 2 (no libraries, no video, no server).

- Drag to turn her head (drag up and she looks up); scroll or pinch to zoom. A few seconds after you let go, she turns back to face you.
- Press **Hear her speak**, or click her, to play a sample line. She takes a short thinking beat and a breath, then answers; lip movement follows the line's word timings (lips close on p, b and m), her head and brows follow the pitch of her voice, and she looks back at you as she finishes ("your turn").
- The line is written with expression tags, as an LLM would write them (`[smile]`, `[brow]`), and the caption highlights each word as she says it.
- Press **Show me what you can do** for a guided tour (about 34 s): she says what she can do and does it as she says it: turns her head to the side and back, looks up and down, opens her eyes wide, raises her brows, frowns, thinks ("Hmm..."), looks sad, smiles, laughs, nods yes, shakes her head no and blinks slowly. The voice carries the emotion (ElevenLabs v3 with voice tags); each move is timed on her words. Long captions show one sentence at a time.
- She is alive between lines: an idle director built into the behaviour module has her look around the room and back at you, shift her posture, smile, lift her brows, have a passing thought, swallow or sigh, with irregular timing and no back-to-back repeats, using the model's trained expressions at low strength. It pauses while she talks or you drag. She also breathes, blinks (more while thinking, and as her eyes move), keeps her eyes subtly alive while fixating, lets her eyelids follow her gaze, glances at your pointer when it comes near her face, and notices when you come back to the tab.
- Motion is smoothed throughout: head turns and expressions ease in and out at any frame rate, and the teeth never show while her lips are pressed.

Files: `index.html` (the page), `openavatar-viewer.js` (the viewer), `openavatar-behaviour.js` (states, gaze, breathing, prosody, persona), `iris.bin` (the avatar, about 47 MB, front-only v23h: the ear area and the mouth retrained on cleaned renders at in-between angles and speech shapes, plus colour clean-ups of the cheeks and hair), `sample.mp3` / `sample.json` and `sample2.mp3` / `sample2.json` (the two recorded lines and their character timings), `lexicon/` (CMU Pronouncing Dictionary, BSD-2-Clause, see `lexicon/LICENSE-cmudict.txt`).

This repository only hosts the demo page. All rights reserved for now; a licence will be chosen when the main project is released.
