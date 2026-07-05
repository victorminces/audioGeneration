import os
import io
import json
import heapq
import glob
import shutil
import random
import tarfile
import tempfile
import requests
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image, ImageDraw

from model import (Autoencoder, VAE, MLPAutoencoder, MLPVAE,
                   WaveformAutoencoder, WaveformVAE, LATENT_H, LATENT_W)
from dataset import SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, CLIP_FRAMES, CLIP_SAMPLES, waveform_to_mel

DATA_DIR = "data/raw"
CHECKPOINT_DIR = "checkpoints"
DEBUG_DIR = "debug"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── helpers ───────────────────────────────────────────────────────────────────

_inverse_mel = T.InverseMelScale(n_stft=N_FFT // 2 + 1, n_mels=N_MELS, sample_rate=SAMPLE_RATE)
_window = torch.hann_window(N_FFT)


def mel_to_waveform(mel, phase=None):
    """(1, N_MELS, frames) log-mel → (1, samples).

    If phase (complex STFT, shape (n_fft//2+1, frames)) is provided, it is used
    directly — no phase estimation needed.  Otherwise falls back to zero-phase
    ISTFT (sounds buzzy but has no iteration cost and no Griffin-Lim smearing).
    """
    # InverseMelScale inverts a power-scale mel → power STFT; take sqrt for magnitude.
    # Clamp first because the pseudo-inverse can produce small negatives.
    stft_mag = _inverse_mel(mel.exp()).clamp(min=0).sqrt()   # (1, n_fft//2+1, frames)
    n_frames = stft_mag.shape[-1]
    if phase is not None:
        unit_phase = phase / (phase.abs() + 1e-8)
        stft_c = stft_mag[0] * unit_phase
    else:
        stft_c = stft_mag[0].to(torch.complex64)   # zero imaginary → zero phase
    wav = torch.istft(stft_c, n_fft=N_FFT, hop_length=HOP_LENGTH,
                      window=_window, length=n_frames * HOP_LENGTH)
    return wav.unsqueeze(0)                     # (1, samples)


def load_waveform(source):
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    data, sr = sf.read(source, dtype="float32", always_2d=True)
    waveform = torch.tensor(data.T)  # (channels, samples)
    if sr != SAMPLE_RATE:
        waveform = T.Resample(sr, SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def load_clips():
    clips = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.lower().endswith(('.wav', '.flac', '.aif', '.aiff')):
            continue
        try:
            waveform = load_waveform(os.path.join(DATA_DIR, fname))
            mel = waveform_to_mel(waveform)             # (1, N_MELS, frames)
            total = mel.shape[2]
            for start in range(0, total - CLIP_FRAMES + 1, CLIP_FRAMES):
                clips.append(mel[:, :, start:start + CLIP_FRAMES])
        except Exception as e:
            st.warning(f"Skipping {fname}: {e}")
    return clips


def save_recording(waveform):
    existing = [f for f in os.listdir(DATA_DIR) if f.startswith("recording_")]
    path = os.path.join(DATA_DIR, f"recording_{len(existing):04d}.wav")
    sf.write(path, waveform.numpy().T, SAMPLE_RATE)
    return path


def tensor_to_bytes(waveform):
    buf = io.BytesIO()
    sf.write(buf, waveform.numpy().T, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return buf.read()


def list_saved_models():
    configs = glob.glob(os.path.join(CHECKPOINT_DIR, "*.json"))
    models = []
    for cfg_path in sorted(configs):
        with open(cfg_path) as f:
            cfg = json.load(f)
        pt_path = cfg_path.replace(".json", ".pt")
        if os.path.exists(pt_path):
            models.append(cfg)
    return models


def load_model(cfg):
    model_type = cfg.get("model_type", "ae")
    arch       = cfg.get("arch", "conv")
    is_mel     = cfg.get("domain", "waveform") == "mel"
    is_vae     = model_type == "vae"
    if arch == "mlp":
        cls = MLPVAE if is_vae else MLPAutoencoder
    elif is_mel:
        cls = VAE if is_vae else Autoencoder
    else:
        cls = WaveformVAE if is_vae else WaveformAutoencoder
    m = cls(latent_ch=cfg["latent_ch"]).to(device)
    pt_path = os.path.join(CHECKPOINT_DIR, cfg["filename"])
    m.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
    m.eval()
    return m


_XFADE = 80  # crossfade length in samples (~5 ms at 16 kHz)

def _stitch(chunks):
    """Concatenate decoded waveform chunks with a short linear crossfade at each boundary."""
    if len(chunks) == 1:
        return chunks[0]
    fade_out = torch.linspace(1.0, 0.0, _XFADE)
    fade_in  = torch.linspace(0.0, 1.0, _XFADE)
    result = chunks[0].clone()
    for chunk in chunks[1:]:
        result[:, -_XFADE:] = result[:, -_XFADE:] * fade_out + chunk[:, :_XFADE] * fade_in
        result = torch.cat([result, chunk[:, _XFADE:]], dim=1)
    return result


def reconstruct(model, source):
    waveform  = load_waveform(source)
    stft_orig = torch.stft(waveform[0], n_fft=N_FFT, hop_length=HOP_LENGTH,
                           window=_window, return_complex=True)   # (bins, total_frames)
    mel          = waveform_to_mel(waveform)                       # (1, N_MELS, total_frames)
    total_frames = mel.shape[2]
    pad = (CLIP_FRAMES - total_frames % CLIP_FRAMES) % CLIP_FRAMES
    if pad:
        mel       = F.pad(mel, (0, pad))
        stft_orig = F.pad(stft_orig, (0, pad))

    # Decode every chunk, keeping the original STFT phases throughout.
    # Collect all frames then do one single ISTFT — no chunk boundaries.
    all_frames = []
    with torch.no_grad():
        for start in range(0, mel.shape[2], CLIP_FRAMES):
            mel_chunk = mel[:, :, start:start + CLIP_FRAMES].unsqueeze(0).to(device)
            mel_out   = model.decode(model.encode(mel_chunk)).squeeze(0).cpu()
            stft_mag  = _inverse_mel(mel_out[0].exp()).clamp(min=0).sqrt()
            orig_ph   = stft_orig[:, start:start + CLIP_FRAMES]
            unit_ph   = orig_ph / (orig_ph.abs() + 1e-8)
            all_frames.append(stft_mag * unit_ph)

    full_stft = torch.cat(all_frames, dim=1)[:, :total_frames]
    wav = torch.istft(full_stft, n_fft=N_FFT, hop_length=HOP_LENGTH,
                      window=_window, length=total_frames * HOP_LENGTH).unsqueeze(0)
    in_rms  = waveform.pow(2).mean().sqrt()
    out_rms = wav.pow(2).mean().sqrt()
    if out_rms > 1e-6:
        wav = wav * (in_rms / out_rms)
    return wav


# ── architecture explanation ──────────────────────────────────────────────────

ARCH_EXPLANATION = """
### How the network works

**Goal:** squeeze a 1-second sound through a small bottleneck, then rebuild it.
If the reconstruction sounds close to the original, the bottleneck learned something real.

---

**Encoder** — compresses audio in two ways at once:

| Layer | Input | Output | What happens |
|---|---|---|---|
| Conv1D | 1 ch × 800 | 16 ch × 400 | 16 filters learn simple patterns (edges, rumbles...) |
| Conv1D | 16 ch × 400 | 32 ch × 200 | 32 filters combine those patterns |
| Conv1D | 32 ch × 200 | 64 ch × 100 | richer features, shorter sequence |
| Conv1D | 64 ch × 100 | **N ch × 50** | **latent — this is the bottleneck** |

Each layer uses **stride 2** — the filter jumps two steps at a time, halving the time axis.
The filters start random and are shaped by backprop during training.

---

**Latent space** — `N channels × 50 time positions`

- N is the latent channel count you set before training.
- Each of the 50 positions corresponds to a 1ms window of the 50ms clip.
- Total units = N × 50. Smaller N = more compression = harder reconstruction.

---

**Decoder** — mirror image of the encoder, using ConvTranspose1D to upsample:

```
N ch × 50  →  64 ch × 100  →  32 ch × 200  →  16 ch × 400  →  1 ch × 800
```

The decoder is **not** undoing the encoder — it's a separate set of learned filters
that figures out how to rebuild a plausible waveform from the compressed representation.

---

**Loss function — L1 (Mean Absolute Error)**

At each training step:
```
loss = mean( |reconstructed_sample - original_sample| )
```

For every one of the 16,000 output samples, we measure how far off it is.
The average of all those errors is the loss.

Backprop then nudges every weight in both encoder and decoder slightly toward
whatever would have made that loss smaller.

- **L1** (absolute difference) penalises all errors equally.
- **L2** (squared difference) punishes large errors more, often produces blurrier audio.
- L1 tends to give crisper reconstructions for waveforms.

---

**What to listen for when comparing models:**

| Latent channels | Effect |
|---|---|
| 2–4 | Heavy compression — expect muffled, blurry output |
| 8 | Moderate compression — details start to emerge |
| 16–32 | Light compression — closer to original, less interesting |

More training steps improve reconstruction at any latent size,
but a tiny latent will always lose some detail — that's the point.
"""


# ── public datasets ───────────────────────────────────────────────────────────

DATASETS = {
    "LibriSpeech test-clean — 346 MB — English speech, multiple speakers": {
        "url": "https://www.openslr.org/resources/12/test-clean.tar.gz",
        "ext": (".flac", ".wav"),
    },
    "Google Speech Commands v0.01 — 1.4 GB — spoken words (yes/no/stop/go...)": {
        "url": "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz",
        "ext": (".wav",),
    },
    "NSynth test — 1.4 GB — musical instrument notes": {
        "url": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",
        "ext": (".wav",),
    },
}


def download_dataset(name, max_files):
    cfg = DATASETS[name]
    url, exts = cfg["url"], cfg["ext"]

    with tempfile.TemporaryDirectory() as tmpdir:
        archive = os.path.join(tmpdir, "dataset.tar.gz")

        bar = st.progress(0.0, text="Downloading…")
        r = requests.get(url, stream=True, timeout=300)
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(archive, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    bar.progress(downloaded / total * 0.5,
                                 text=f"Downloading… {downloaded // 1_000_000} / {total // 1_000_000} MB")

        bar.progress(0.5, text="Extracting…")
        extract_dir = os.path.join(tmpdir, "extracted")
        os.makedirs(extract_dir)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(extract_dir)

        found = []
        for root, _, files in os.walk(extract_dir):
            for fname in files:
                if fname.lower().endswith(exts):
                    found.append(os.path.join(root, fname))

        if max_files < len(found):
            found = random.sample(found, max_files)

        existing = len([f for f in os.listdir(DATA_DIR)
                        if f.lower().endswith((".wav", ".flac"))])
        for i, src in enumerate(found):
            ext = os.path.splitext(src)[1]
            dst = os.path.join(DATA_DIR, f"ds_{existing + i:06d}{ext}")
            shutil.copy2(src, dst)
            bar.progress(0.5 + (i + 1) / len(found) * 0.5,
                         text=f"Copying {i + 1}/{len(found)} files…")

        bar.progress(1.0, text="Done.")
        return len(found)


def _slerp(za, zb, alpha):
    # flatten to vectors, slerp, reshape back
    shape = za.shape
    a = za.reshape(1, -1).float()
    b = zb.reshape(1, -1).float()
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    dot = (a_norm * b_norm).sum(dim=-1).clamp(-1, 1)
    omega = dot.acos()
    sin_omega = omega.sin()
    # fall back to linear when vectors are nearly parallel
    if sin_omega.abs() < 1e-6:
        return ((1 - alpha) * za + alpha * zb)
    return ((torch.sin((1 - alpha) * omega) / sin_omega) * a +
            (torch.sin(alpha * omega) / sin_omega) * b).reshape(shape)


_PGHI_GAMMA = 0.25645 * N_FFT ** 2   # time-frequency ratio of a Hann window


def pghi(mag, tol=1e-6):
    """Phase Gradient Heap Integration (Prusa et al. 2017).

    Estimates a consistent STFT phase from magnitude alone, via the
    phase-magnitude relations of a Gaussian-like window. Non-iterative.
    mag: (bins, frames) numpy magnitude spectrogram -> phase in radians.
    """
    bins, frames = mag.shape
    logs = np.log(np.maximum(mag, 1e-300))
    logs = np.maximum(logs, logs.max() - 11.0)          # limit dynamic range

    dm = np.zeros_like(logs)                            # d log|S| / d bin
    dn = np.zeros_like(logs)                            # d log|S| / d frame
    dm[1:-1, :] = (logs[2:, :] - logs[:-2, :]) / 2
    dn[:, 1:-1] = (logs[:, 2:] - logs[:, :-2]) / 2

    m_idx = np.arange(bins)[:, None]
    # phase advance per hop / per bin; the -pi accounts for torch.stft
    # placing the window origin at the frame start rather than its center
    tgrad = (HOP_LENGTH * N_FFT / _PGHI_GAMMA) * dm + 2 * np.pi * HOP_LENGTH * m_idx / N_FFT
    fgrad = -(_PGHI_GAMMA / (HOP_LENGTH * N_FFT)) * dn - np.pi

    phase = np.zeros_like(mag)
    done = mag <= tol * mag.max()
    phase[done] = np.random.default_rng(0).uniform(0, 2 * np.pi, done.sum())

    heap = []
    start = np.unravel_index(np.argmax(mag), mag.shape)
    heapq.heappush(heap, (-mag[start], start))
    assigned = done.copy()
    assigned[start] = True
    while heap:
        _, (m, n) = heapq.heappop(heap)
        for dmm, dnn, grad, sign in ((0, 1, tgrad, 1), (0, -1, tgrad, -1),
                                     (1, 0, fgrad, 1), (-1, 0, fgrad, -1)):
            mm, nn = m + dmm, n + dnn
            if 0 <= mm < bins and 0 <= nn < frames and not assigned[mm, nn]:
                phase[mm, nn] = phase[m, n] + sign * (grad[m, n] + grad[mm, nn]) / 2
                assigned[mm, nn] = True
                heapq.heappush(heap, (-mag[mm, nn], (mm, nn)))
    return phase


# ── sound square helpers ─────────────────────────────────────────────────────

SQUARE_SECONDS = 2.0
SQUARE_SAMPLES = int(SQUARE_SECONDS * SAMPLE_RATE)
SQUARE_PX      = 380   # canvas size in pixels


def fix_length(waveform, n_samples=SQUARE_SAMPLES):
    """Trim or zero-pad (1, samples) to exactly n_samples."""
    if waveform.shape[1] >= n_samples:
        return waveform[:, :n_samples]
    return F.pad(waveform, (0, n_samples - waveform.shape[1]))


def encode_frames(model, waveform):
    """(1, samples) → (T, ...) latent sequence, one latent per mel frame."""
    mel = waveform_to_mel(waveform)                       # (1, N_MELS, T)
    frames = mel[0].T.unsqueeze(1).unsqueeze(-1)          # (T, 1, N_MELS, 1)
    with torch.no_grad():
        return model.encode(frames.to(device))            # (T, ...)


def decode_frames(model, z_seq):
    """(T, ...) latent sequence → (1, samples) via PGHI synthesis."""
    with torch.no_grad():
        mel_out = model.decode(z_seq).cpu()               # (T, 1, N_MELS, 1)
    mel_mat  = mel_out.squeeze(-1).squeeze(1).T           # (N_MELS, T)
    full_mag = _inverse_mel(mel_mat.exp()).clamp(min=0).sqrt()   # (bins, T)
    phase    = pghi(full_mag.numpy().astype(np.float64))
    full_stft = full_mag * torch.tensor(np.exp(1j * phase), dtype=torch.complex64)
    wav = torch.istft(full_stft, n_fft=N_FFT, hop_length=HOP_LENGTH,
                      window=_window, length=full_mag.shape[1] * HOP_LENGTH)
    return wav.unsqueeze(0)


def idw_weights(points, qx, qy, power):
    """Inverse-distance weights for query (qx, qy) against [(x, y), ...]."""
    d = np.array([np.hypot(qx - x, qy - y) for x, y in points])
    nearest = int(d.argmin())
    if d[nearest] < 1e-3:                                 # clicked on a sound
        w = np.zeros(len(d)); w[nearest] = 1.0
        return w
    w = 1.0 / d ** power
    return w / w.sum()


def draw_square(sounds, pending=None, cursor=None):
    img = Image.new("RGB", (SQUARE_PX, SQUARE_PX), "#1a1a2e")
    dr = ImageDraw.Draw(img)
    for i in range(1, 4):                                 # grid lines
        p = SQUARE_PX * i // 4
        dr.line([(p, 0), (p, SQUARE_PX)], fill="#2a2a45")
        dr.line([(0, p), (SQUARE_PX, p)], fill="#2a2a45")
    dr.rectangle([0, 0, SQUARE_PX - 1, SQUARE_PX - 1], outline="#44446a")
    r = 9
    for i, s in enumerate(sounds):
        x, y = s["x"] * SQUARE_PX, s["y"] * SQUARE_PX
        dr.ellipse([x - r, y - r, x + r, y + r], fill="#4f9dde", outline="white")
        dr.text((x + r + 3, y - r), str(i + 1), fill="white")
    if pending is not None:
        x, y = pending[0] * SQUARE_PX, pending[1] * SQUARE_PX
        dr.ellipse([x - r, y - r, x + r, y + r], outline="#e05555", width=2)
    if cursor is not None:
        x, y = cursor[0] * SQUARE_PX, cursor[1] * SQUARE_PX
        dr.line([(x - r, y), (x + r, y)], fill="#57d992", width=2)
        dr.line([(x, y - r), (x, y + r)], fill="#57d992", width=2)
    return img


def interpolate_sounds(model, source_a, source_b, n_steps, use_slerp=False):
    mel_a = waveform_to_mel(load_waveform(source_a))
    mel_b = waveform_to_mel(load_waveform(source_b))

    min_frames = min(mel_a.shape[2], mel_b.shape[2])
    mel_a = mel_a[:, :, :min_frames]
    mel_b = mel_b[:, :, :min_frames]

    pad = (CLIP_FRAMES - min_frames % CLIP_FRAMES) % CLIP_FRAMES
    if pad:
        mel_a = F.pad(mel_a, (0, pad))
        mel_b = F.pad(mel_b, (0, pad))

    model.eval()
    with torch.no_grad():
        z_a, z_b = [], []
        for start in range(0, mel_a.shape[2], CLIP_FRAMES):
            z_a.append(model.encode(mel_a[:, :, start:start + CLIP_FRAMES].unsqueeze(0).to(device)))
            z_b.append(model.encode(mel_b[:, :, start:start + CLIP_FRAMES].unsqueeze(0).to(device)))

        results = []
        for i in range(n_steps):
            alpha = i / (n_steps - 1) if n_steps > 1 else 0.0

            # Decode all chunks to magnitudes, then estimate a consistent
            # phase for the whole spectrogram with PGHI.
            all_mags = []
            for za, zb in zip(z_a, z_b):
                z       = _slerp(za, zb, alpha) if use_slerp else (1 - alpha) * za + alpha * zb
                mel_out = model.decode(z).squeeze(0).cpu()           # (1, N_MELS, CLIP_FRAMES)
                stft_mag = _inverse_mel(mel_out[0].exp()).clamp(min=0).sqrt()  # (bins, CLIP_FRAMES)
                all_mags.append(stft_mag)

            full_mag = torch.cat(all_mags, dim=1)[:, :min_frames]
            phase    = pghi(full_mag.numpy().astype(np.float64))
            full_stft = (full_mag * torch.tensor(np.exp(1j * phase), dtype=torch.complex64))
            wav = torch.istft(full_stft, n_fft=N_FFT, hop_length=HOP_LENGTH,
                              window=_window, length=min_frames * HOP_LENGTH).unsqueeze(0)
            results.append((alpha, wav))

    return results


# ── ui ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Generative Sound", layout="wide")
st.title("Waveform Autoencoder")
st.caption(f"Device: {device}  |  Sample rate: {SAMPLE_RATE} Hz  |  Clip length: 50ms")

with st.expander("How does this network work? (architecture + loss function)"):
    st.markdown(ARCH_EXPLANATION)

for _key in ("recon_a", "recon_b", "last_trained", "interp_src_a", "interp_src_b",
             "sq_pending", "sq_last_rec", "sq_last_click", "sq_result"):
    if _key not in st.session_state:
        st.session_state[_key] = None
if "sq_sounds" not in st.session_state:
    st.session_state.sq_sounds = []   # [{"x", "y", "wav" (1, samples) tensor}]

for _key, _fname in (("interp_src_a", "interp_a.wav"), ("interp_src_b", "interp_b.wav")):
    if st.session_state[_key] is None:
        _path = os.path.join(DEBUG_DIR, _fname)
        if os.path.exists(_path):
            with open(_path, "rb") as _f:
                st.session_state[_key] = _f.read()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["1 · Data", "2 · Train", "3 · Compare & Reconstruct",
                                        "4 · Interpolate", "5 · Sound Square"])


# ── tab 1: data ───────────────────────────────────────────────────────────────

with tab1:
    st.subheader("Add audio to your training set")
    st.write("Upload a file or record directly. Recordings are chopped into 1-second clips.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Upload a file**")
        uploaded = st.file_uploader("Audio file", type=["wav", "flac", "aif", "aiff"],
                                    label_visibility="collapsed")
        if uploaded:
            st.audio(uploaded)
            if st.button("Add uploaded file"):
                waveform = load_waveform(uploaded.read())
                path = save_recording(waveform)
                n = waveform.shape[1] // CLIP_SAMPLES
                if n == 0:
                    st.warning("Recording is shorter than 50 ms — no clips were added.")
                else:
                    st.success(f"Saved {waveform.shape[1] / SAMPLE_RATE:.1f}s → {n} clip(s)")

    with col2:
        st.markdown("**Record from microphone**")
        audio_bytes = audio_recorder(text="Click to record", pause_threshold=3.0, key="recorder")
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            if st.button("Add recording"):
                waveform = load_waveform(audio_bytes)
                n = waveform.shape[1] // CLIP_SAMPLES
                if n == 0:
                    st.warning(
                        f"Recording is {waveform.shape[1] / SAMPLE_RATE:.2f}s — "
                        "needs to be at least 50 ms to produce a clip."
                    )
                else:
                    save_recording(waveform)
                    st.success(f"Saved {waveform.shape[1] / SAMPLE_RATE:.1f}s → {n} clip(s)")

    st.divider()

    clips = load_clips()
    files = [f for f in os.listdir(DATA_DIR)
             if f.lower().endswith(('.wav', '.flac', '.aif', '.aiff'))]

    c1, c2, c3 = st.columns([1, 1, 1])
    c1.metric("Files in dataset", len(files))
    c2.metric("50ms clips", len(clips))
    clip_secs = CLIP_SAMPLES / SAMPLE_RATE
    c3.metric("Total audio", f"{len(clips) * clip_secs:.0f}s")

    st.divider()
    st.markdown("**Download a public dataset**")
    st.write("Downloads audio files and copies them into your training set.")

    ds_col1, ds_col2 = st.columns([3, 1])
    ds_choice = ds_col1.selectbox("Dataset", list(DATASETS.keys()), label_visibility="collapsed")
    max_files = ds_col2.number_input("Max files", min_value=10, max_value=10000,
                                     value=500, step=100)
    if st.button("Download & add to dataset"):
        try:
            n = download_dataset(ds_choice, int(max_files))
            st.success(f"Added {n} files to data/raw/")
            st.rerun()
        except Exception as e:
            st.error(f"Download failed: {e}")

    st.divider()
    st.markdown("**Danger zone**")
    if st.button("Clear dataset", type="secondary"):
        if "confirm_clear" not in st.session_state:
            st.session_state.confirm_clear = True
        st.rerun()

    if st.session_state.get("confirm_clear"):
        st.warning("This will delete all recordings from data/raw/. Are you sure?")
        col_yes, col_no, _ = st.columns([1, 1, 4])
        if col_yes.button("Yes, delete all", type="primary"):
            for f in os.listdir(DATA_DIR):
                if f.lower().endswith(('.wav', '.flac', '.aif', '.aiff')):
                    os.remove(os.path.join(DATA_DIR, f))
            st.session_state.confirm_clear = False
            st.success("Dataset cleared.")
            st.rerun()
        if col_no.button("Cancel"):
            st.session_state.confirm_clear = False
            st.rerun()


# ── tab 2 helpers ─────────────────────────────────────────────────────────────

def _next_model_name():
    existing = glob.glob(os.path.join(CHECKPOINT_DIR, "*.json"))
    return f"run_{len(existing) + 1:03d}"


def _model_details(cfg):
    is_vae = cfg.get("model_type", "ae") == "vae"
    arch   = cfg.get("arch", "conv")
    latent = (f"{cfg['latent_ch']} dims" if arch == "mlp"
              else f"{cfg['latent_ch']} ch × {LATENT_H}×{LATENT_W} = {cfg['latent_units']} units")
    d = {
        "Type":       ("VAE" if is_vae else "Autoencoder") + f" ({arch.upper()})",
        "Domain":     cfg.get("domain", "mel"),
        "Input":      f"{cfg.get('clip_frames', CLIP_FRAMES)} mel frame × {cfg.get('n_mels', N_MELS)} bins  ({cfg.get('clip_ms', round(CLIP_FRAMES * HOP_LENGTH / SAMPLE_RATE * 1000))} ms)",
        "Latent":     latent,
        "Steps":      cfg["steps"],
        "Clips used": cfg["clips_used"],
    }
    if is_vae and cfg.get("beta") is not None:
        d["β (KL weight)"] = cfg["beta"]
        d["Free bits"]     = cfg.get("free_bits", "—")
    d["Reconstruction"] = ("Original STFT phase + single full ISTFT — no chunk boundaries"
                           if cfg.get("domain", "mel") == "mel"
                           else "Direct waveform decode")
    d["Generation / interp"] = ("Phase propagation: Δφ_k = 2π·k·H/N per hop, "
                                "continuous across chunks, single ISTFT")
    return d


def _loss_history_chart(cfg):
    """Render the persisted loss history of a saved model, if any."""
    hist = cfg.get("loss_history") or []
    if not hist:
        st.caption("No loss history recorded for this model.")
        return
    st.line_chart(pd.DataFrame(hist).set_index("step"), height=180)


def _run_training_loop(model, is_vae, train_loader, val_loader,
                       num_steps, start_step, beta, free_bits):
    """Shared training loop used by both Train and Continue. Returns loss_history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    st.caption("Reconstruction loss (train + val)")
    recon_chart = st.line_chart({"recon": [], "val recon": []}, height=180)
    if is_vae:
        st.caption("KL loss")
        kl_chart = st.line_chart({"kl": []}, height=150)

    progress  = st.progress(0.0, text="Starting…")
    log_area  = st.empty()
    log_lines = []
    loss_history = []

    report_every = max(1, num_steps // 20)
    step = start_step
    total = start_step + num_steps
    recon_loss_val = kl_loss_val = 0.0

    while step < total:
        model.train()
        for batch in train_loader:
            if step >= total:
                break
            batch = batch.to(device)

            if is_vae:
                recon, mu, logvar = model(batch)
                recon_loss_val = F.l1_loss(recon, batch)
                kl_raw    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                reduce_dims = tuple(i for i in range(kl_raw.dim()) if i != 1)
                kl_per_ch = kl_raw.mean(dim=reduce_dims)
                kl_loss_val = kl_per_ch.clamp(min=free_bits).mean()
                loss = recon_loss_val + beta * kl_loss_val
            else:
                loss = F.l1_loss(model(batch), batch)
                recon_loss_val = loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % report_every == 0 or step == total:
                model.eval()
                with torch.no_grad():
                    v_recon = []
                    for b in val_loader:
                        b = b.to(device)
                        out = model(b)[0] if is_vae else model(b)
                        v_recon.append(F.l1_loss(out, b).item())
                val_recon = sum(v_recon) / len(v_recon)
                model.train()

                recon_chart.add_rows({"recon": [recon_loss_val.item()], "val recon": [val_recon]})
                if is_vae:
                    kl_chart.add_rows({"kl": [kl_loss_val.item()]})

                log_line = (f"step {step:5d}/{total} | "
                            f"recon {recon_loss_val.item():.4f} | val {val_recon:.4f}")
                if is_vae:
                    log_line += f" | kl {kl_loss_val.item():.4f}"
                log_lines.append(log_line)
                progress.progress((step - start_step) / num_steps,
                                  text=f"Step {step}/{total}")
                log_area.code("\n".join(log_lines[-12:]))

                entry = {"step": step,
                         "recon": round(recon_loss_val.item(), 5),
                         "val_recon": round(val_recon, 5)}
                if is_vae:
                    entry["kl"] = round(kl_loss_val.item(), 5)
                loss_history.append(entry)

    progress.progress(1.0, text="Done!")
    return loss_history


# ── tab 2: train ──────────────────────────────────────────────────────────────

with tab2:
    st.subheader("Train a model")

    # ── rename last trained model ──────────────────────────────────────────────
    if st.session_state.last_trained:
        last = st.session_state.last_trained
        st.success(f"Trained: **{last}**  — rename it if you like, then dismiss.")
        _last_cfg_path = os.path.join(CHECKPOINT_DIR, f"{last}.json")
        if os.path.exists(_last_cfg_path):
            with open(_last_cfg_path) as f:
                _last_cfg = json.load(f)
            st.caption("Training loss")
            _loss_history_chart(_last_cfg)
        ren_col1, ren_col2, ren_col3 = st.columns([3, 1, 1])
        new_name = ren_col1.text_input("New name", value=last, key="rename_input",
                                       label_visibility="collapsed")
        if ren_col2.button("Save name"):
            new_name = new_name.strip().replace(" ", "_")
            if new_name and new_name != last:
                old_pt   = os.path.join(CHECKPOINT_DIR, f"{last}.pt")
                old_json = os.path.join(CHECKPOINT_DIR, f"{last}.json")
                new_pt   = os.path.join(CHECKPOINT_DIR, f"{new_name}.pt")
                new_json = os.path.join(CHECKPOINT_DIR, f"{new_name}.json")
                if os.path.exists(old_pt):
                    os.rename(old_pt, new_pt)
                if os.path.exists(old_json):
                    with open(old_json) as f:
                        saved_cfg = json.load(f)
                    saved_cfg["name"]     = new_name
                    saved_cfg["filename"] = f"{new_name}.pt"
                    with open(new_json, "w") as f:
                        json.dump(saved_cfg, f, indent=2)
                    os.remove(old_json)
                if st.session_state.get("interp_model") == last:
                    st.session_state["interp_model"] = new_name
            st.session_state.last_trained = None
            st.rerun()
        if ren_col3.button("Dismiss"):
            st.session_state.last_trained = None
            st.rerun()
        st.divider()

    # ── new training run ───────────────────────────────────────────────────────
    auto_name = _next_model_name()
    st.caption(f"Will be saved as **{auto_name}** — rename after training.")

    col1, col2, col3 = st.columns(3)
    latent_ch     = col1.select_slider("Latent channels",
                                       options=[2, 4, 8, 16, 32], value=16)
    num_steps     = col2.slider("Training steps", 100, 20000, 10000, 100)
    batch_size    = col3.slider("Batch size", 4, 64, 16, 4)

    col3b, col4, col5 = st.columns([1, 1, 2])
    arch_label       = col3b.radio("Architecture", ["MLP", "Conv"], horizontal=True)
    model_type_label = col4.radio("Model type", ["Autoencoder", "VAE"], horizontal=True)
    is_mlp = arch_label == "MLP"
    is_vae = model_type_label == "VAE"

    beta = 0.001
    free_bits = 0.5
    if is_vae:
        v1, v2 = col5.columns(2)
        beta      = v1.select_slider("β",
                                     options=[0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
                                     value=0.001)
        free_bits = v2.select_slider("Free bits",
                                     options=[0.0, 0.1, 0.5, 1.0, 2.0, 4.0],
                                     value=0.5)

    if is_mlp:
        latent_units = latent_ch
    else:
        latent_units = latent_ch * LATENT_H * LATENT_W
    mel_units      = N_MELS * CLIP_FRAMES
    compression    = round(mel_units / latent_units, 1)
    clip_ms        = round(CLIP_FRAMES * HOP_LENGTH / SAMPLE_RATE * 1000)
    model_type_str = "vae" if is_vae else "ae"
    arch_str       = "mlp" if is_mlp else "conv"
    latent_desc    = (f"{latent_ch} dims" if is_mlp
                      else f"{latent_ch} ch × {LATENT_H}×{LATENT_W} = {latent_units} units")
    st.info(
        f"{'VAE' if is_vae else 'Autoencoder'} ({arch_label})  |  Mel spectrogram  |  "
        f"Input: {CLIP_FRAMES} frame × {N_MELS} bins ({clip_ms} ms)  |  "
        f"Latent: {latent_desc}  |  {compression}× compression"
        + (f"  |  β={beta}  λ={free_bits}" if is_vae else "")
    )

    if st.button("Train", type="primary"):
        clips = load_clips()
        if not clips:
            st.error("No clips found. Add recordings in the Data tab first.")
        else:
            data = torch.stack(clips)
            val_size = max(1, len(clips) // 10)
            perm = torch.randperm(len(clips))
            train_loader = DataLoader(data[perm[val_size:]], batch_size=int(batch_size),
                                      shuffle=True, drop_last=len(clips) - val_size >= int(batch_size))
            val_loader   = DataLoader(data[perm[:val_size]], batch_size=int(batch_size))

            if is_mlp:
                model = (MLPVAE if is_vae else MLPAutoencoder)(latent_ch=latent_ch).to(device)
            else:
                model = (VAE if is_vae else Autoencoder)(latent_ch=latent_ch).to(device)
            loss_history = _run_training_loop(model, is_vae, train_loader, val_loader,
                                              num_steps, 0, beta, free_bits)

            pt_file = f"{auto_name}.pt"
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, pt_file))
            cfg = {
                "name": auto_name, "filename": pt_file,
                "domain": "mel", "model_type": model_type_str, "arch": arch_str,
                "latent_ch": latent_ch, "latent_units": latent_units,
                "clip_frames": CLIP_FRAMES, "n_mels": N_MELS, "clip_ms": clip_ms,
                "steps": num_steps, "clips_used": len(clips),
                "beta": beta if is_vae else None,
                "free_bits": free_bits if is_vae else None,
                "loss_history": loss_history,
            }
            with open(os.path.join(CHECKPOINT_DIR, f"{auto_name}.json"), "w") as f:
                json.dump(cfg, f, indent=2)
            st.session_state.last_trained = auto_name
            st.session_state["interp_model"] = auto_name
            st.rerun()

    st.divider()

    # ── continue training ──────────────────────────────────────────────────────
    st.markdown("**Continue training a saved model**")
    saved_for_cont = list_saved_models()
    if not saved_for_cont:
        st.caption("No saved models yet.")
    else:
        cc1, cc2, cc3 = st.columns([2, 1, 1])
        cont_name = cc1.selectbox("Model", [m["name"] for m in saved_for_cont],
                                  key="cont_model")
        cont_steps = cc2.slider("Additional steps", 100, 20000, 2000, 100,
                                key="cont_steps")
        cont_cfg = next(m for m in saved_for_cont if m["name"] == cont_name)
        if cc3.button("Continue", type="secondary"):
            clips = load_clips()
            if not clips:
                st.error("No clips found.")
            else:
                data = torch.stack(clips)
                val_size = max(1, len(clips) // 10)
                perm = torch.randperm(len(clips))
                train_loader = DataLoader(data[perm[val_size:]], batch_size=16,
                                          shuffle=True, drop_last=len(clips) - val_size >= 16)
                val_loader   = DataLoader(data[perm[:val_size]], batch_size=16)

                cont_m = load_model(cont_cfg)
                cont_m.train()
                cont_is_vae   = cont_cfg.get("model_type", "ae") == "vae"
                cont_beta      = cont_cfg.get("beta") or 0.001
                cont_free_bits = cont_cfg.get("free_bits") or 0.5
                start_step     = cont_cfg.get("steps", 0)

                new_hist = _run_training_loop(cont_m, cont_is_vae, train_loader, val_loader,
                                              cont_steps, start_step, cont_beta, cont_free_bits)
                torch.save(cont_m.state_dict(),
                           os.path.join(CHECKPOINT_DIR, cont_cfg["filename"]))
                cont_cfg["steps"] = start_step + cont_steps
                cont_cfg["loss_history"] = cont_cfg.get("loss_history", []) + new_hist
                with open(os.path.join(CHECKPOINT_DIR, f"{cont_name}.json"), "w") as f:
                    json.dump(cont_cfg, f, indent=2)
                st.success(f"'{cont_name}' trained for {cont_steps} more steps.")
                st.session_state["interp_model"] = cont_name
                st.rerun()

    st.divider()

    # ── saved models table ─────────────────────────────────────────────────────
    st.markdown("**Saved models**")
    saved = list_saved_models()
    if not saved:
        st.write("No models saved yet.")
    else:
        rows = [
            {
                "Name":         m["name"],
                "Type":         m.get("model_type", "ae").upper(),
                "Domain":       m.get("domain", "mel"),
                "Input":        f"{m.get('clip_frames', '?')}fr × {m.get('n_mels', N_MELS)}mel ({m.get('clip_ms', '?')}ms)",
                "Latent":       f"{m['latent_ch']}ch = {m['latent_units']}u",
                "β":            m.get("beta") or "—",
                "Steps":        m["steps"],
                "Clips":        m["clips_used"],
            }
            for m in saved
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

        with st.expander("Loss history"):
            view_name = st.selectbox("Model", [m["name"] for m in saved], key="loss_view")
            _loss_history_chart(next(m for m in saved if m["name"] == view_name))


# ── tab 3: compare & reconstruct ─────────────────────────────────────────────

def rms(waveform):
    return float(waveform.pow(2).mean().sqrt())


def _waveform_fig(waveform, label="", save_as=None):
    wav = waveform.squeeze().numpy()
    t   = np.arange(len(wav)) / SAMPLE_RATE
    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.plot(t, wav, linewidth=0.3, color="steelblue")
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("s", fontsize=7)
    ax.set_ylim(-1, 1)
    ax.tick_params(labelsize=7)
    if label:
        ax.set_title(label, fontsize=8)
    fig.tight_layout(pad=0.3)
    if save_as:
        fig.savefig(os.path.join(DEBUG_DIR, save_as), dpi=150)
    return fig



with tab3:
    st.subheader("Compare models")
    st.write("Select up to two trained models and hear how each reconstructs the same sound.")

    saved = list_saved_models()

    if not saved:
        st.warning("No trained models yet. Train at least one in the Train tab.")
    else:
        model_names = [m["name"] for m in saved]

        col_src, _ = st.columns([2, 1])
        with col_src:
            st.markdown("**Input sound**")
            r_upload = st.file_uploader("Upload to reconstruct", type=["wav", "flac"],
                                        key="r_upload", label_visibility="collapsed")
            r_recorded = audio_recorder(text="Or record", pause_threshold=3.0, key="r_recorder")

        source = None
        source_waveform = None
        if r_upload:
            source = r_upload.read()
        elif r_recorded:
            source = r_recorded

        if source:
            source_waveform = load_waveform(source)
            src_rms = rms(source_waveform)
            st.audio(source, format="audio/wav")
            fig = _waveform_fig(source_waveform, "Input", save_as="waveform_input.png")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption(f"Input amplitude (RMS): **{src_rms:.4f}**")

        st.divider()

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Model A — reconstruction**")
            sel_a = st.selectbox("Select model A", model_names, key="sel_a")
            cfg_a = next(m for m in saved if m["name"] == sel_a)
            with st.expander("Model details"):
                st.json(_model_details(cfg_a))
            if source and st.button("Reconstruct with A", type="primary"):
                try:
                    m = load_model(cfg_a)
                    out = reconstruct(m, source)
                    st.session_state.recon_a = (tensor_to_bytes(out), rms(out))
                except Exception as e:
                    st.error(f"Error: {e}")

            if st.session_state.recon_a:
                audio_bytes, out_rms = st.session_state.recon_a
                st.audio(audio_bytes, format="audio/wav")
                recon_wav_a = load_waveform(audio_bytes)
                fig = _waveform_fig(recon_wav_a, "Reconstruction A", save_as="waveform_recon_a.png")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                src_rms_val = rms(source_waveform) if source_waveform is not None else 0
                st.caption(f"Output amplitude (RMS): **{out_rms:.4f}**  "
                           f"(input was {src_rms_val:.4f})")
                if src_rms_val > 0.001 and out_rms < 0.001:
                    st.warning("Output is near-silence — the model may have been trained on silence.")

        with col_b:
            st.markdown("**Model B — reconstruction**")
            default_b = min(1, len(model_names) - 1)
            sel_b = st.selectbox("Select model B", model_names, index=default_b, key="sel_b")
            cfg_b = next(m for m in saved if m["name"] == sel_b)
            with st.expander("Model details"):
                st.json(_model_details(cfg_b))
            if source and st.button("Reconstruct with B", type="primary"):
                try:
                    m = load_model(cfg_b)
                    out = reconstruct(m, source)
                    st.session_state.recon_b = (tensor_to_bytes(out), rms(out))
                except Exception as e:
                    st.error(f"Error: {e}")

            if st.session_state.recon_b:
                audio_bytes, out_rms = st.session_state.recon_b
                st.audio(audio_bytes, format="audio/wav")
                recon_wav_b = load_waveform(audio_bytes)
                fig = _waveform_fig(recon_wav_b, "Reconstruction B", save_as="waveform_recon_b.png")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                src_rms_val = rms(source_waveform) if source_waveform is not None else 0
                st.caption(f"Output amplitude (RMS): **{out_rms:.4f}**  "
                           f"(input was {src_rms_val:.4f})")
                if src_rms_val > 0.001 and out_rms < 0.001:
                    st.warning("Output is near-silence — the model may have been trained on silence.")


# ── tab 4: interpolate ────────────────────────────────────────────────────────

with tab4:
    st.subheader("Interpolate between two sounds")
    st.write(
        "Each sound is encoded clip by clip (50 ms chunks). "
        "Latents are interpolated at each α step and decoded directly to waveform. "
        "Chunks are stitched for continuity. The longer sound is truncated to match the shorter."
    )

    saved = list_saved_models()
    if not saved:
        st.warning("No trained models yet. Train at least one in the Train tab.")
    else:
        col_cfg, _ = st.columns([2, 1])
        with col_cfg:
            interp_model_name = st.selectbox("Model", [m["name"] for m in saved], key="interp_model")
            n_steps = st.slider("Number of steps", min_value=3, max_value=10, value=5,
                                help="Includes α=0 (sound A) and α=1 (sound B)")
            use_slerp = st.toggle("Spherical interpolation (slerp)",
                                  help="Follows the curvature of the latent space instead of a straight line. Often sounds smoother.")

        st.divider()
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Sound A** (α = 0.0)")
            up_a  = st.file_uploader("Upload A", type=["wav", "flac"], key="interp_a",
                                     label_visibility="collapsed")
            rec_a = audio_recorder(text="Or record A", pause_threshold=3.0, key="interp_rec_a")
            if up_a:
                new_a = up_a.read()
                with open(os.path.join(DEBUG_DIR, "interp_a.wav"), "wb") as f:
                    f.write(new_a)
                st.session_state.interp_src_a = new_a
            elif rec_a:
                with open(os.path.join(DEBUG_DIR, "interp_a.wav"), "wb") as f:
                    f.write(rec_a)
                st.session_state.interp_src_a = rec_a
            source_a = st.session_state.interp_src_a
            if source_a:
                st.audio(source_a, format="audio/wav")

        with col_b:
            st.markdown("**Sound B** (α = 1.0)")
            up_b  = st.file_uploader("Upload B", type=["wav", "flac"], key="interp_b",
                                     label_visibility="collapsed")
            rec_b = audio_recorder(text="Or record B", pause_threshold=3.0, key="interp_rec_b")
            if up_b:
                new_b = up_b.read()
                with open(os.path.join(DEBUG_DIR, "interp_b.wav"), "wb") as f:
                    f.write(new_b)
                st.session_state.interp_src_b = new_b
            elif rec_b:
                with open(os.path.join(DEBUG_DIR, "interp_b.wav"), "wb") as f:
                    f.write(rec_b)
                st.session_state.interp_src_b = rec_b
            source_b = st.session_state.interp_src_b
            if source_b:
                st.audio(source_b, format="audio/wav")

        if source_a and source_b and st.button("Interpolate", type="primary"):
            try:
                cfg = next(m for m in saved if m["name"] == interp_model_name)
                m = load_model(cfg)
                results = interpolate_sounds(m, source_a, source_b, n_steps, use_slerp=use_slerp)

                st.divider()
                st.markdown("**Results** — play each step to hear the path through latent space")
                cols = st.columns(len(results))
                for col, (alpha, audio) in zip(cols, results):
                    with col:
                        label = "A" if alpha == 0.0 else ("B" if alpha == 1.0 else f"α={alpha:.2f}")
                        st.caption(label)
                        st.audio(tensor_to_bytes(audio), format="audio/wav")
            except Exception as e:
                st.error(f"Error: {e}")


# ── tab 5: sound square ───────────────────────────────────────────────────────

with tab5:
    st.subheader("Sound Square")
    st.write(
        "**Training:** click the square, then record — a 2-second sound gets pinned "
        "to that spot. Add as many as you like. "
        "**Inference:** click anywhere and the sounds are blended by their distance "
        "to your click (inverse-distance weighting on the frame-level latents)."
    )

    sq_saved = list_saved_models()
    if not sq_saved:
        st.warning("No trained models yet. Train one in the Train tab.")
    else:
        c1, c2, c3 = st.columns([2, 2, 1])
        sq_model_name = c1.selectbox("Model", [m["name"] for m in sq_saved], key="sq_model")
        sq_mode = c2.radio("Mode", ["Add sounds", "Generate"], horizontal=True, key="sq_mode")
        sq_power = c3.slider("Sharpness", 0.5, 8.0, 2.0, 0.5,
                             help="IDW exponent: low = smooth global blend, high = nearest sound dominates")

        sounds = st.session_state.sq_sounds
        col_sq, col_side = st.columns([1, 1])

        with col_sq:
            img = draw_square(sounds,
                              pending=st.session_state.sq_pending,
                              cursor=st.session_state.sq_last_click if sq_mode == "Generate" else None)
            click = streamlit_image_coordinates(img, key="sq_canvas")

        if click is not None:
            cx, cy = click["x"] / SQUARE_PX, click["y"] / SQUARE_PX
            if (cx, cy) != (st.session_state.sq_last_click or (None, None)):
                st.session_state.sq_last_click = (cx, cy)
                if sq_mode == "Add sounds":
                    st.session_state.sq_pending = (cx, cy)
                    st.rerun()
                else:
                    if len(sounds) < 2:
                        st.warning("Add at least 2 sounds first.")
                    else:
                        try:
                            cfg = next(m for m in sq_saved if m["name"] == sq_model_name)
                            m = load_model(cfg)
                            latents = [encode_frames(m, s["wav"]) for s in sounds]
                            n_frames = min(z.shape[0] for z in latents)
                            w = idw_weights([(s["x"], s["y"]) for s in sounds], cx, cy, sq_power)
                            z_mix = sum(float(wi) * z[:n_frames] for wi, z in zip(w, latents))
                            wav = decode_frames(m, z_mix)
                            st.session_state.sq_result = (tensor_to_bytes(wav), w)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

        with col_side:
            if sq_mode == "Add sounds":
                if st.session_state.sq_pending is not None:
                    px, py = st.session_state.sq_pending
                    st.markdown(f"**Record 2 s for point ({px:.2f}, {py:.2f})**")
                    rec = audio_recorder(text="Record", pause_threshold=2.5, key="sq_rec")
                    if rec and rec != st.session_state.sq_last_rec:
                        st.session_state.sq_last_rec = rec
                        wav = fix_length(load_waveform(rec))
                        sounds.append({"x": px, "y": py, "wav": wav})
                        st.session_state.sq_pending = None
                        st.rerun()
                else:
                    st.caption("Click the square to choose where the next sound goes.")

                if sounds:
                    st.markdown(f"**{len(sounds)} sound(s) placed**")
                    for i, s in enumerate(sounds):
                        cc1, cc2 = st.columns([4, 1])
                        with cc1:
                            st.caption(f"{i + 1} — ({s['x']:.2f}, {s['y']:.2f})")
                            st.audio(tensor_to_bytes(s["wav"]), format="audio/wav")
                        if cc2.button("✕", key=f"sq_del_{i}"):
                            sounds.pop(i)
                            st.rerun()
                    if st.button("Clear all", key="sq_clear"):
                        st.session_state.sq_sounds = []
                        st.session_state.sq_pending = None
                        st.session_state.sq_result = None
                        st.rerun()
            else:
                if st.session_state.sq_result:
                    audio_bytes, w = st.session_state.sq_result
                    st.markdown("**Generated sound**")
                    st.audio(audio_bytes, format="audio/wav", autoplay=True)
                    st.caption("Weights: " +
                               ", ".join(f"{i + 1}: {wi:.2f}" for i, wi in enumerate(w)))
                else:
                    st.caption("Click the square to generate a sound.")
