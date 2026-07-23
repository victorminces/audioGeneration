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

## Running

```bash
pip install -r requirements.txt
cd web
python server.py       # or: uvicorn server:app --port 8321
```

Then open `http://127.0.0.1:8321` in a browser with webcam/microphone
permissions available. A trained DDSP checkpoint ships with the repo
(`checkpoints/ddsp_003.pt`), so this works immediately on a fresh clone —
no training required to get started.

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
  See "Training the DDSP model" below for the full input/output/loss picture.

### Front end (`web/static/`)

Everything real-time runs in the browser, not the server:

- **`index.html`** — page shell and controls (kernel mode selector, sigma
  sliders, camera/clear buttons).
- **`main.js`** — orchestrates everything outside the audio thread: mouse and
  MediaPipe `HandLandmarker` gesture input (hand tracking loaded from a CDN;
  the model file itself, `models/hand_landmarker.task`, is served locally by
  the backend via `/api/hand_model`), the "sing to teach" recording flow
  (posts audio + cursor path to `/api/take`), and drawing the on-screen path.
- **`worklet.js`** — an `AudioWorklet` that runs the gesture-map evaluation
  *and* the DDSP decoder directly in the audio callback, sample-accurate and
  glitch-free. It fetches the decoder weights once from `/api/decoder/weights`
  and never talks to the server again for audio — this is why the instrument
  stays responsive even under load. Bump its `?v=N` query string in
  `main.js` if you change `worklet.js`; Chrome aggressively caches
  `AudioWorklet` modules across hard refreshes.
- **`web/server.py`** (backend) only does four things: serves the decoder
  weights/manifest once, serves the hand-tracking model file, runs the
  "sing to teach" analysis (`/api/take`, using `ddsp.analyze` +
  `mapper.KernelMapper.add`), and serves the static files. No audio ever
  passes through it during play.

## Training the DDSP model

**Input -> output -> loss**, for anyone extending this: the encoder takes an
80-bin log-mel spectrogram per frame and produces a 16-dim timbre latent `z`;
the decoder takes `(f0, loudness, z)` — with `f0`/`loudness` extracted
directly from the audio via `torchcrepe`, not learned — and produces harmonic
amplitudes + noise-band gains, which the (non-learned, fully differentiable)
synthesizer turns into a waveform. Nothing is directly supervised — the only
loss is a multi-scale spectral distance between the synthesized waveform and
the original, and gradients flow back through the whole synthesizer into the
network. See the `TimbreEncoder`/`Decoder`/`synthesize` classes in
[ddsp.py](ddsp.py) for the exact shapes.

### Reproducing or retraining

```bash
# 1. Get training audio (LibriSpeech by default) into data/raw/
python download_data.py --dataset librispeech --max-files 500

# 2. Train. Analyzes clips into (f0, loudness, mel) once (slow — torchcrepe
#    pitch tracking) and caches the result, then trains the DDSP model.
python train_ddsp.py
```

`train_ddsp.py` saves `checkpoints/<name>.pt` + `<name>.json`, checkpointing
every ~2.5% of the run (not just at the end) so you can try the model
mid-training. The server always picks up the newest checkpoint whose config
has `"domain": "ddsp"`.

Key options (`python train_ddsp.py --help`): `--steps`, `--batch-size`,
`--lr`, `--clip-seconds`, `--out-name`. To continue training an existing
checkpoint rather than starting over: `--resume <name>` (e.g.
`python train_ddsp.py --resume ddsp_003 --steps 5000` trains 5000 more steps
on top of it, under the same checkpoint name). A quick smoke test:
`python train_ddsp.py --steps 200 --batch-size 8`.

### What to expect

`checkpoints/ddsp_003.pt` (the shipped default, 20000 steps, ~501 LibriSpeech
files / 1538 two-second clips) plateaued hard around step 8000-9000 — val
loss went from 1.44 (step 50) to 0.84 (step 9000) to 0.839 (step 20000).
The last ~11000 steps bought essentially nothing on this metric. On CPU
(no CUDA), expect roughly 4s/step, so a full 20000-step run is close to a day
of wall time; stopping around step 8000-10000 gets you nearly all the
measurable quality for a fraction of the compute. That said, this loss number
covers the whole signal at once (pitch + timbre + noise) and can plateau
while small perceptual details keep improving — listening (via "sing to
teach") is the real test, not just the number.
