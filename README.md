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
permissions available. A trained DDSP checkpoint ships with the repo
(`checkpoints/ddsp_009.pt`), so this works immediately on a fresh clone,
with no training required to get started.

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
  2020) — harmonic-plus-noise synthesis from a per-frame (f0, noisiness,
  timbre latent z), scaled by a level the network never sees. Trained
  end-to-end through the synthesizer with a multi-scale spectral loss;
  outputs waveform directly, no phase reconstruction needed. See "Training
  the DDSP model" below for the full input/output/loss picture.

### The harmonics come off a spectral envelope

The decoder predicts a spectral envelope on **absolute** frequency, and each
harmonic's amplitude is read off it at `k*f0` — the source-filter model, rather
than predicting the 60 harmonic amplitudes directly. Formants then stay where
they are as the pitch moves, instead of the network relearning them at every
`f0`. Bands are log-spaced, 65 of them from 40 Hz to Nyquist — 8.37 per octave,
where a linear grid would put 123 Hz between neighbours and barely resolve the
first two harmonics of a low voice.

Both parameterisations were built and compared by ear on the same
demonstrations (`ddsp_006` direct amplitudes, `ddsp_007` envelope); trained to
the same budget on the same split they landed within 0.6% of each other, and
the envelope was kept. The direct-amplitude path has since been removed rather
than left switchable, so there is one code path to reason about.

### Three things are measured, not learned

The network is asked only for timbre. Everything that has a right answer is
measured from the signal:

- **Pitch** — pYIN (`librosa.pyin`), searching 60–4000 Hz. The predecessor,
  CREPE, is a classifier over 360 fixed bins topping out at 2093 Hz, and a
  whistle above that ceiling gets reported an octave low with full confidence:
  a 2041 Hz whistle came back as 1939 Hz and wandered, which is audible as an
  electronic warble. pYIN's range is a search bound rather than an output
  vocabulary, so raising it costs nothing structural.
- **Noisiness** (`extract_aperiodicity`) — the fraction of energy sitting
  *between* the harmonics rather than on them. The decoder used to predict this
  balance and was demonstrably bad at it: it answered 0.078 on a whispered
  vowel, where the truth is ~1, rebuilding a whisper as a pitched voice.
  Handing it in also makes the balance interpolate as a straight line between
  demonstrations rather than stepping off a cliff. See the docstring for why
  Boersma's HNR, the thirty-year standard, was tried and rejected here.
- **Level** — see below.

Analysis high-passes at 60 Hz first, below the pitch floor. Breath on the
microphone capsule puts real energy down there — 6 to 15 dB more while a take
is sounding than between takes — and it biases both the level and the
noisiness measurements without ever being audible as pitch.

### Level is factored out

The instrument is invariant to recording level: step back from the microphone,
or change its gain, and the sound it makes does not change. That is structural
rather than a normalisation applied on the way in —

- the **encoder** subtracts each take's own level from its log-mel before the
  GRU sees it, so `z` cannot encode level;
- the **decoder** takes no level input and predicts no amplitude — it emits
  harmonic shape (sums to 1) and noise shape (unit RMS), both
  dimensionless, and their balance is measured rather than predicted;
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

One test is optional. `pip install pyworld` enables a cross-check of the
aperiodicity measurement against WORLD's D4C, an unrelated algorithm from a
different decade, on the speech-like material D4C is authoritative for. It is
**deliberately not in `requirements.txt`**: the app never calls it, it is a C
extension that will not build everywhere, and WORLD's own pitch tracker
declines to voice a whistle — see the note in `ddsp.extract_aperiodicity` for
why it is a cross-check rather than the estimator itself. The test skips
cleanly when it is absent.

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
80-bin log-mel spectrogram per frame, subtracts the take's level, and produces
a 16-dim timbre latent `z`; the decoder takes `(f0, noisiness, z)` — both `f0`
and noisiness measured directly from the audio, not learned — and produces a
spectral envelope and a noise shape, which the (non-learned, fully
differentiable) synthesizer scales by the frame's amplitude, mixes at the
measured balance, and turns into a waveform. Nothing is directly supervised —
the only loss is a multi-scale spectral distance between the synthesized
waveform and the original, and gradients flow back through the whole
synthesizer into the network. See the `TimbreEncoder`/`Decoder`/`synthesize`
classes in [ddsp.py](ddsp.py) for the exact shapes.

Training runs on level-normalised clips: every clip is scaled to the same
performed level and its per-frame RMS is handed to the model directly. The
level term of the loss is then matched by construction, so what is left to
reduce is spectral shape. One consequence: **val losses are not comparable
with `ddsp_001..005`**, which were scored on absolute level as well.

### Reproducing or retraining

```bash
# 1. Get training audio (LibriSpeech by default) into data/raw/
python download_data.py --dataset librispeech --max-files 500

# 2. Train. Analyzes clips into (f0, loudness, mel) once (slow — pYIN pitch
#    tracking) and caches the result, then trains the DDSP model.
python train_ddsp.py
```

`train_ddsp.py` saves `checkpoints/<name>.pt` + `<name>.json`, checkpointing
every ~2.5% of the run (not just at the end) so you can try the model
mid-training. The server picks the highest-numbered checkpoint whose config has
`"domain": "ddsp"` **and** an `"arch"` matching `ddsp.ARCH` — older checkpoints
are skipped by name rather than failing on a shape mismatch. The name is what
gates loading, not the shape, because `v2` and `v3` have *identical* shapes and
differ only in which pitch tracker measured the training `f0`; a model trained
against one reads the other wrongly, and no shape check would catch it.

Key options (`python train_ddsp.py --help`): `--steps`, `--batch-size`,
`--lr`, `--clip-seconds`, `--out-name`. To continue training an existing
checkpoint rather than starting over: `--resume <name>`. To warm-start from an
older architecture instead, `--adapt <name>`: it reads the source layout from
that checkpoint's own config and does the column surgery to match, inserting
the aperiodicity input as a **zero** column so the new decoder is *exactly* the
old one at initialisation, and folding a pre-factoring model's loudness input
into the biases at normal singing level. Adapting from a checkpoint that
already has the current layout copies every tensor straight across. A quick
smoke test: `python train_ddsp.py --steps 200 --batch-size 8`.

`--refresh-f0` recomputes only the pitch track in an existing cache, leaving
the cached waveforms alone. That is the cheap path when the tracker changes:
the cache stores `wav` and `f0`, and everything else is derived on load.

The shipped model was built as:

```bash
python train_ddsp.py --refresh-f0
python train_ddsp.py --steps 2900 --batch-size 4 --adapt ddsp_008 --out-name ddsp_009
```

On a machine with little free memory, drop `--batch-size`. The synthesizer
holds a `batch x samples x 60` harmonics tensor, so batch 32 over 2 s clips
wants several GB and dies with a bare segfault rather than a Python error.

### What to expect

The shipped model, `ddsp_009`, was trained on ~501 LibriSpeech files plus
synthetic high-pitch tones (2419 clips), warm-started from `ddsp_008` and run
for 2900 steps. It finishes at val 0.9327 and was still creeping down —
`ddsp_004` had 9375 steps before its loss went flat, so there is more to get
here. On CPU (no CUDA) budget roughly a day for 20000 steps.

Three notes on reading the loss. It covers the whole signal at once — pitch,
timbre and noise together — so it can plateau while perceptual details keep
improving; listening via "sing to teach" is the real test. It is not comparable
between models trained on different data: `ddsp_004` reports a worse val figure
than `ddsp_003` purely because its validation set contains the high-pitch
material. And it is not comparable across the level factoring at all — from
`ddsp_006` on, clips are level-normalised and the amplitude is handed to the
model, so the level term is matched by construction and the number measures
only spectral shape.

On the checkpoints in `checkpoints/`: `ddsp_009` is the one the app loads.
`ddsp_004` (pre-factoring) and `ddsp_006`/`ddsp_007` (the amplitude/envelope
pair) are kept only as warm-start ancestors for `--adapt`. None of the three
can be loaded by the current model, which is the point of the `arch` gate.
