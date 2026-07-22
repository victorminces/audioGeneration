# Gesture Instrument

A camera- and mouse-driven instrument: sing a short phrase while drawing a
gesture, and the instrument learns to map that 2D gesture back to the sound
you sang. Afterwards, moving through the same space — with a hand tracked by
your webcam, or a mouse — replays a continuous, gesture-controlled version of
what you taught it.

Open `http://127.0.0.1:8321` (see Running, below) and:

- **Sing to teach**: hold left-drag (mouse) or a pinch (hand) while singing,
  drawing the sound's path through the 2D space as you go.
- **Play**: move the cursor/hand through that space; the instrument
  synthesizes pitch/timbre/loudness for wherever you are, continuously.

## Architecture

- **Gesture -> voice mapping** ([mapper.py](mapper.py)): kernel smoothing
  (Nadaraya-Watson) over the demonstrated (x, y) -> (pitch, loudness, timbre)
  samples. The prediction at any point is a weighted average of nearby
  demonstrations, so it can never extrapolate wildly — and a confidence value
  (the kernel weight at the nearest sample) fades the instrument to silence
  away from anything you've taught it. Two kernel modes: isotropic (round) or
  anisotropic (narrow along the stroke direction, wide across it, so
  consecutive sounds in a stroke don't blur together).
- **Voice synthesis** ([ddsp.py](ddsp.py)): a DDSP model (Engel et al., ICLR
  2020) — harmonic-plus-noise synthesis from per-frame (f0, loudness, timbre
  latent z). Trained end-to-end through the synthesizer with a multi-scale
  spectral loss; outputs waveform directly, no phase reconstruction needed.
- **Everything real-time runs in the browser**: the gesture map evaluation
  and the DDSP decoder both execute in an `AudioWorklet`
  ([web/static/worklet.js](web/static/worklet.js)) for glitch-free audio.
  The server only serves the decoder weights once and handles the
  "sing to teach" analysis step.
- **Hand tracking** is MediaPipe's `HandLandmarker`, running client-side in
  the browser (loaded from a CDN; the model file itself is served locally by
  the backend, see `models/hand_landmarker.task`).

## Running

```bash
pip install -r requirements.txt
cd web
python server.py       # or: uvicorn server:app --port 8321
```

Then open `http://127.0.0.1:8321` in a browser with webcam/microphone
permissions available.

## Reproducing the DDSP model

The DDSP checkpoint that ships with this repo lives in `checkpoints/`
(gitignored — not part of the repo itself). To retrain it from scratch:

```bash
# 1. Get training audio (LibriSpeech by default) into data/raw/
python download_data.py --dataset librispeech --max-files 500

# 2. Train. Analyzes clips into (f0, loudness, mel) once (slow — torchcrepe
#    pitch tracking) and caches the result, then trains the DDSP model.
python train_ddsp.py
```

`train_ddsp.py` saves `checkpoints/<name>.pt` + `<name>.json`; the server
always picks up the newest checkpoint whose config has `"domain": "ddsp"`.
Key options (see `python train_ddsp.py --help`): `--steps`, `--batch-size`,
`--lr`, `--clip-seconds`. A quick smoke test: `python train_ddsp.py --steps
200 --batch-size 8`.
