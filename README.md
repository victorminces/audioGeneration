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
  synthesizes pitch/timbre/dynamics for wherever you are, continuously.

How loud you are recorded does not affect what the instrument sounds like —
see [Level is factored out](#level-is-factored-out).

## Running

```bash
pip install -r requirements.txt
cd web
python server.py       # or: uvicorn server:app --port 8321
```

Then open `http://127.0.0.1:8321` in a browser with webcam/microphone
permissions available. Trained DDSP checkpoints ship with the repo —
`checkpoints/ddsp_006.pt` and `ddsp_007.pt`, one per harmonic mode — so this
works immediately on a fresh clone, with no training required to get started.

## Architecture

- **Gesture -> voice mapping** ([mapper.py](mapper.py)): kernel smoothing
  (Nadaraya-Watson) over the demonstrated (x, y) -> (pitch, level, timbre)
  samples. The prediction at any point is a weighted average of nearby
  demonstrations, so it can never extrapolate wildly — and a confidence value
  (the kernel weight at the nearest sample) fades the instrument to silence
  away from anything you've taught it. Two kernel modes: isotropic (round) or
  anisotropic (narrow along the stroke direction, wide across it, so
  consecutive sounds in a stroke don't blur together).
- **Voice synthesis** ([ddsp.py](ddsp.py)): a DDSP model (Engel et al., ICLR
  2020) — harmonic-plus-noise synthesis from a per-frame (f0, timbre latent z),
  scaled by a level the network never sees. Trained end-to-end through the
  synthesizer with a multi-scale spectral loss; outputs waveform directly, no
  phase reconstruction needed. See "Training the DDSP model" below for the full
  input/output/loss picture.

### Two harmonic parameterisations, switchable live

The `synth:` button switches between two ways of producing harmonic
amplitudes, so they can be compared by ear on the same demonstrations:

- **`harmonics`** (`harm_mode: "amp"`) — the decoder predicts all 60 amplitudes
  directly. Indexed by harmonic *number*, so it is pitch-relative: harmonic 3
  of a low note and harmonic 3 of a high one share a weight even though they
  sit an octave and a half apart in absolute frequency.
- **`filter`** (`harm_mode: "filter"`) — the decoder predicts a spectral
  envelope on absolute frequency and the amplitudes are read off it at `k*f0`,
  the source-filter model. Formants stay where they are as the pitch moves,
  rather than the network relearning them at every `f0`. Bands are log-spaced,
  65 of them from 40 Hz to Nyquist — 8.37 per octave, where a linear grid would
  put 123 Hz between neighbours and barely resolve the first two harmonics of a
  low voice.

Both decoders are fetched at startup and both live in the worklet, so the
toggle is a pointer change: you can switch mid-phrase without the audio
dropping. `z` is encoder-specific, so **every take is analysed by both models**
and both codes are stored in the same map row — same gestures, same thinning,
same silence-dropping, only the timbre pathway differing. That is what makes
the toggle a comparison of methods rather than of two separate recordings.

Trained to the same budget on the same split, the two land within 0.6% of each
other (`ddsp_006` val 0.8939, `ddsp_007` 0.8995), so the choice is a listening
question rather than a loss question. One asymmetry to keep in mind: a
filter-mode model cannot inherit `head_harm` when warm-starting — that head is
indexed by harmonic number and this one by frequency — so it begins from a
hand-set spectral tilt while the amplitude model begins from trained weights.
The gap that opens early (5% at step 170) closes as training runs.

### Level is factored out

The instrument is invariant to recording level: step back from the microphone,
or change its gain, and the sound it makes does not change. That is structural
rather than a normalisation applied on the way in —

- the **encoder** subtracts each take's own level from its log-mel before the
  GRU sees it, so `z` cannot encode level;
- the **decoder** takes no level input and predicts no amplitude — it emits
  harmonic shape (sums to 1), noise shape (unit RMS) and their balance
  (`mix`, in [0, 1]), all dimensionless;
- **level enters once**, multiplying the mixed output in `synthesize`.

What is stored per frame is therefore *relative* level — dynamics within the
performance — not absolute log-RMS. Playing back puts it on an absolute footing
via `REF_LOUD`.

The consequence worth knowing: analysis is no longer causal. It needs the
take's overall level, so it cannot be driven straight from a live microphone
without buffering.

[test_ddsp.py](test_ddsp.py) checks these properties directly (`python
test_ddsp.py`). They hold at any weights, so it needs no checkpoint. Worth
running after touching anything in the analysis path — the invariance has been
broken twice by changes that looked unrelated, once by an absolute epsilon
added before a log and once by normalising over the wrong axis, and neither was
audible until much later.

### Front end (`web/static/`)

Everything real-time runs in the browser, not the server:

- **`index.html`** — page shell and controls (kernel mode selector, sigma
  sliders, the `synth:` A/B toggle, camera/clear buttons).
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
80-bin log-mel spectrogram per frame, subtracts the take's level, and produces
a 16-dim timbre latent `z`; the decoder takes `(f0, z)` — `f0` extracted
directly from the audio via `torchcrepe`, not learned — and produces harmonic
shape, noise shape and their balance, which the (non-learned, fully
differentiable) synthesizer scales by the frame's amplitude and turns into a
waveform. Nothing is directly supervised — the only loss is a multi-scale
spectral distance between the synthesized waveform and the original, and
gradients flow back through the whole synthesizer into the network. See the
`TimbreEncoder`/`Decoder`/`synthesize` classes in [ddsp.py](ddsp.py) for the
exact shapes.

Training runs on level-normalised clips: every clip is scaled to the same
performed level and its per-frame RMS is handed to the model directly. The
level term of the loss is then matched by construction, so what is left to
reduce is spectral shape. One consequence: **val losses are not comparable
with `ddsp_001..005`**, which were scored on absolute level as well.

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
mid-training. The server picks the highest-numbered checkpoint whose config has
`"domain": "ddsp"` **and** a matching `"arch"` — `ddsp_001..005` predate the
level factoring and are skipped by name rather than failing on a shape
mismatch.

Key options (`python train_ddsp.py --help`): `--steps`, `--batch-size`,
`--lr`, `--clip-seconds`, `--out-name`. To continue training an existing
checkpoint rather than starting over: `--resume <name>` (e.g.
`python train_ddsp.py --resume ddsp_006 --steps 5000` trains 5000 more steps
on top of it, under the same checkpoint name). To warm-start from a
pre-factoring checkpoint instead, `--adapt ddsp_004`: it folds that model's
loudness input into the biases at the reference level, which makes the
17-input decoder exactly the old 18-input one held at normal singing level,
and keeps every other tensor. `--harm-mode filter` trains the source-filter
variant instead (ignored with `--resume`, which takes the mode from the
checkpoint). A quick smoke test:
`python train_ddsp.py --steps 200 --batch-size 8`.

The app needs **one checkpoint per harmonic mode** and says so if either is
missing, rather than silently offering half an A/B. The pair that ships was
built as:

```bash
python train_ddsp.py --steps 400  --batch-size 4 --adapt ddsp_004 --out-name ddsp_006
python train_ddsp.py --resume ddsp_006 --steps 2500 --batch-size 4
python train_ddsp.py --steps 400  --batch-size 4 --harm-mode filter --adapt ddsp_004 --out-name ddsp_007
python train_ddsp.py --resume ddsp_007 --steps 2500 --batch-size 4
```

On a machine with little free memory, drop `--batch-size`. The synthesizer
holds a `batch x samples x 60` harmonics tensor, so batch 32 over 2 s clips
wants several GB and dies with a bare segfault rather than a Python error.

### What to expect

The shipped pair, `ddsp_006` (harmonics) and `ddsp_007` (filter), were trained
on ~501 LibriSpeech files plus synthetic high-pitch tones, warm-started from
`ddsp_004` and run for 2900 steps each. They finish at val 0.8939 and 0.8995
and were both still creeping down — `ddsp_004` had 9375 steps before its loss
went flat, so there is more to get here. On CPU (no CUDA) budget roughly a day
for 20000 steps.

Three notes on reading the loss. It covers the whole signal at once — pitch,
timbre and noise together — so it can plateau while perceptual details keep
improving; listening via "sing to teach" is the real test. It is not comparable
between models trained on different data: `ddsp_004` reports a worse val figure
than `ddsp_003` purely because its validation set contains the high-pitch
material. And it is not comparable across the level factoring at all — from
`ddsp_006` on, clips are level-normalised and the amplitude is handed to the
model, so the level term is matched by construction and the number measures
only spectral shape.
