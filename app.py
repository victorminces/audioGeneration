import os
import io
import json
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
import streamlit as st
from audio_recorder_streamlit import audio_recorder

from model import Autoencoder
from dataset import SAMPLE_RATE, CLIP_LENGTH

DATA_DIR = "data/raw"
CHECKPOINT_DIR = "checkpoints"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── helpers ───────────────────────────────────────────────────────────────────

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
            for start in range(0, waveform.shape[1] - CLIP_LENGTH + 1, CLIP_LENGTH):
                clips.append(waveform[:, start:start + CLIP_LENGTH])
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
    m = Autoencoder(latent_ch=cfg["latent_ch"]).to(device)
    pt_path = os.path.join(CHECKPOINT_DIR, cfg["filename"])
    m.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
    m.eval()
    return m


def reconstruct(model, source):
    waveform = load_waveform(source)
    total = waveform.shape[1]
    pad = (CLIP_LENGTH - total % CLIP_LENGTH) % CLIP_LENGTH
    if pad:
        waveform = F.pad(waveform, (0, pad))
    chunks = []
    with torch.no_grad():
        for start in range(0, waveform.shape[1], CLIP_LENGTH):
            chunk = waveform[:, start:start + CLIP_LENGTH].unsqueeze(0).to(device)
            chunks.append(model(chunk).squeeze(0).cpu())
    return torch.cat(chunks, dim=1)[:, :total]


# ── architecture explanation ──────────────────────────────────────────────────

ARCH_EXPLANATION = """
### How the network works

**Goal:** squeeze a 1-second sound through a small bottleneck, then rebuild it.
If the reconstruction sounds close to the original, the bottleneck learned something real.

---

**Encoder** — compresses audio in two ways at once:

| Layer | Input | Output | What happens |
|---|---|---|---|
| Conv1D | 1 ch × 16,000 | 16 ch × 8,000 | 16 filters learn simple patterns (edges, rumbles...) |
| Conv1D | 16 ch × 8,000 | 32 ch × 4,000 | 32 filters combine those patterns |
| Conv1D | 32 ch × 4,000 | 64 ch × 2,000 | richer features, shorter sequence |
| Conv1D | 64 ch × 2,000 | 32 ch × 1,000 | starts narrowing channels |
| Conv1D | 32 ch × 1,000 | **N ch × 500** | **latent — this is the bottleneck** |

Each layer uses **stride 2** — the filter jumps two steps at a time, halving the time axis.
The filters start random and are shaped by backprop during training.

---

**Latent space** — `N channels × 500 time positions`

- N is the latent channel count you set before training.
- Each of the 500 positions corresponds to a ~32ms window of the original audio.
- Total units = N × 500. Smaller N = more compression = harder reconstruction.

---

**Decoder** — mirror image of the encoder, using ConvTranspose1D to upsample:

```
N ch × 500  →  32 ch × 1,000  →  64 ch × 2,000  →  32 ch × 4,000
            →  16 ch × 8,000  →   1 ch × 16,000
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


# ── ui ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Generative Sound", layout="wide")
st.title("Waveform Autoencoder")
st.caption(f"Device: {device}  |  Sample rate: {SAMPLE_RATE} Hz  |  Clip length: 1 second")

with st.expander("How does this network work? (architecture + loss function)"):
    st.markdown(ARCH_EXPLANATION)

tab1, tab2, tab3 = st.tabs(["1 · Data", "2 · Train", "3 · Compare & Reconstruct"])


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
                n = waveform.shape[1] // CLIP_LENGTH
                if n == 0:
                    st.warning("Recording is shorter than 1 second — no clips were added.")
                else:
                    st.success(f"Saved {waveform.shape[1] / SAMPLE_RATE:.1f}s → {n} clip(s)")

    with col2:
        st.markdown("**Record from microphone**")
        audio_bytes = audio_recorder(text="Click to record", pause_threshold=3.0, key="recorder")
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            if st.button("Add recording"):
                waveform = load_waveform(audio_bytes)
                n = waveform.shape[1] // CLIP_LENGTH
                if n == 0:
                    st.warning(
                        f"Recording is {waveform.shape[1] / SAMPLE_RATE:.2f}s — "
                        "needs to be at least 1 second to produce a clip."
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
    c2.metric("1-second clips", len(clips))
    c3.metric("Total audio", f"{len(clips):.0f}s")

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


# ── tab 2: train ──────────────────────────────────────────────────────────────

with tab2:
    st.subheader("Train a model")
    st.write("Give it a name so you can compare different configs later.")

    col1, col2, col3 = st.columns(3)
    model_name = col1.text_input("Model name", value="model_01",
                                 help="Used as the filename. No spaces.")
    latent_ch = col2.select_slider("Latent channels (N)",
                                   options=[2, 4, 8, 16, 32],
                                   value=8,
                                   help="N × 500 = total latent units. Smaller = more compression.")
    num_steps = col3.slider("Training steps", 100, 10000, 1000, 100)

    col4, _ = st.columns([1, 2])
    batch_size = col4.slider("Batch size", 4, 64, 16, 4)

    latent_units = latent_ch * 500
    compression = round(16000 / latent_units, 1)
    st.info(
        f"Latent: **{latent_ch} ch × 500 steps = {latent_units} units**  "
        f"({compression}× compression of 16,000-sample input)"
    )

    if st.button("Train", type="primary"):
        name_clean = model_name.strip().replace(" ", "_") or "model"
        clips = load_clips()
        if not clips:
            st.error("No clips found. Add recordings in the Data tab first.")
        else:
            data = torch.stack(clips)
            val_size = max(1, len(clips) // 10)
            perm = torch.randperm(len(clips))
            train_data = data[perm[val_size:]]
            val_data = data[perm[:val_size]]

            bs = int(batch_size)
            train_loader = DataLoader(train_data, batch_size=bs, shuffle=True,
                                      drop_last=len(train_data) >= bs)
            val_loader = DataLoader(val_data, batch_size=bs)

            model = Autoencoder(latent_ch=latent_ch).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            progress = st.progress(0.0, text="Starting...")
            log_area = st.empty()
            log_lines = [f"Found {len(clips)} clips | latent {latent_ch} ch | device {device}"]

            report_every = max(1, num_steps // 20)
            step = 0

            while step < num_steps:
                model.train()
                for batch in train_loader:
                    if step >= num_steps:
                        break
                    batch = batch.to(device)
                    loss = F.l1_loss(model(batch), batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                    if step % report_every == 0 or step == num_steps:
                        model.eval()
                        with torch.no_grad():
                            v = [F.l1_loss(model(b.to(device)), b.to(device)).item()
                                 for b in val_loader]
                        val_loss = sum(v) / len(v)
                        model.train()
                        log_lines.append(
                            f"step {step:5d}/{num_steps} | "
                            f"train {loss.item():.4f} | val {val_loss:.4f}"
                        )
                        progress.progress(step / num_steps,
                                          text=f"Step {step}/{num_steps}")
                        log_area.code("\n".join(log_lines))

            pt_file = f"{name_clean}.pt"
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, pt_file))
            cfg = {
                "name": name_clean,
                "filename": pt_file,
                "latent_ch": latent_ch,
                "latent_units": latent_units,
                "steps": num_steps,
                "clips_used": len(clips),
            }
            with open(os.path.join(CHECKPOINT_DIR, f"{name_clean}.json"), "w") as f:
                json.dump(cfg, f, indent=2)

            progress.progress(1.0, text="Done!")
            log_lines.append(f"\nSaved as '{name_clean}'")
            log_area.code("\n".join(log_lines))
            st.success(f"Model '{name_clean}' saved. Go to Compare & Reconstruct.")

    st.divider()
    st.markdown("**Saved models**")
    saved = list_saved_models()
    if not saved:
        st.write("No models saved yet.")
    else:
        rows = [
            {
                "Name": m["name"],
                "Latent ch": m["latent_ch"],
                "Latent units": m["latent_units"],
                "Steps": m["steps"],
                "Trained on (clips)": m["clips_used"],
            }
            for m in saved
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


# ── tab 3: compare & reconstruct ─────────────────────────────────────────────

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
        if r_upload:
            source = r_upload.read()
            st.audio(source, format="audio/wav")
        elif r_recorded:
            source = r_recorded
            st.audio(source, format="audio/wav")

        st.divider()

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Model A**")
            sel_a = st.selectbox("Select model A", model_names, key="sel_a")
            cfg_a = next(m for m in saved if m["name"] == sel_a)
            st.caption(
                f"Latent: {cfg_a['latent_ch']} ch × 500 = {cfg_a['latent_units']} units  |  "
                f"Steps: {cfg_a['steps']}"
            )
            if source and st.button("Reconstruct with A", type="primary"):
                try:
                    m = load_model(cfg_a)
                    out = reconstruct(m, source)
                    st.audio(tensor_to_bytes(out), format="audio/wav")
                except Exception as e:
                    st.error(f"Error: {e}")

        with col_b:
            st.markdown("**Model B**")
            default_b = min(1, len(model_names) - 1)
            sel_b = st.selectbox("Select model B", model_names, index=default_b, key="sel_b")
            cfg_b = next(m for m in saved if m["name"] == sel_b)
            st.caption(
                f"Latent: {cfg_b['latent_ch']} ch × 500 = {cfg_b['latent_units']} units  |  "
                f"Steps: {cfg_b['steps']}"
            )
            if source and st.button("Reconstruct with B", type="primary"):
                try:
                    m = load_model(cfg_b)
                    out = reconstruct(m, source)
                    st.audio(tensor_to_bytes(out), format="audio/wav")
                except Exception as e:
                    st.error(f"Error: {e}")
