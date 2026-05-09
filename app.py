import os
import io
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
CHECKPOINT = os.path.join(CHECKPOINT_DIR, "model.pt")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── helpers ───────────────────────────────────────────────────────────────────

def load_waveform(source):
    """Load a waveform from a file path, file-like object, or bytes."""
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
        if not fname.lower().endswith(('.wav', '.mp3', '.flac', '.aif', '.aiff')):
            continue
        try:
            waveform = load_waveform(os.path.join(DATA_DIR, fname))
            for start in range(0, waveform.shape[1] - CLIP_LENGTH + 1, CLIP_LENGTH):
                clips.append(waveform[:, start:start + CLIP_LENGTH])
        except Exception as e:
            st.warning(f"Skipping {fname}: {e}")
    return clips


def save_audio(waveform):
    existing = [f for f in os.listdir(DATA_DIR) if f.startswith("recording_")]
    path = os.path.join(DATA_DIR, f"recording_{len(existing):04d}.wav")
    sf.write(path, waveform.numpy().T, SAMPLE_RATE)
    return path


def get_model():
    m = Autoencoder().to(device)
    if os.path.exists(CHECKPOINT):
        m.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    m.eval()
    return m


def tensor_to_bytes(waveform):
    buf = io.BytesIO()
    sf.write(buf, waveform.numpy().T, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return buf.read()


# ── ui ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Generative Sound", layout="wide")
st.title("Waveform Autoencoder")
st.caption(
    f"Record or upload sounds → train a compression model → hear what the bottleneck keeps. "
    f"**Latent:** 8 ch × 500 steps = 4,000 units  |  **Device:** {device}"
)

tab1, tab2, tab3 = st.tabs(["1 · Data", "2 · Train", "3 · Reconstruct"])


# ── tab 1: data ───────────────────────────────────────────────────────────────

with tab1:
    st.subheader("Add audio to your training set")
    st.write("Upload a file or record directly. Long recordings are chopped into 1-second clips.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Upload a file**")
        uploaded = st.file_uploader("Audio file", type=["wav", "mp3", "flac", "aif", "aiff"],
                                    label_visibility="collapsed")
        if uploaded:
            st.audio(uploaded)
            if st.button("Add uploaded file", key="add_upload"):
                waveform = load_waveform(uploaded.read())
                path = save_audio(waveform)
                n = waveform.shape[1] // CLIP_LENGTH
                st.success(f"Saved {waveform.shape[1] / SAMPLE_RATE:.1f}s → {n} clip(s)")

    with col2:
        st.markdown("**Record from microphone**")
        audio_bytes = audio_recorder(text="Click to record", pause_threshold=3.0, key="recorder")
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            if st.button("Add recording", key="add_rec"):
                waveform = load_waveform(audio_bytes)
                path = save_audio(waveform)
                n = waveform.shape[1] // CLIP_LENGTH
                st.success(f"Saved {waveform.shape[1] / SAMPLE_RATE:.1f}s → {n} clip(s)")

    st.divider()
    clips = load_clips()
    files = [f for f in os.listdir(DATA_DIR)
             if f.lower().endswith(('.wav', '.mp3', '.flac', '.aif', '.aiff'))]
    c1, c2 = st.columns(2)
    c1.metric("Files in dataset", len(files))
    c2.metric("1-second clips", len(clips))


# ── tab 2: train ──────────────────────────────────────────────────────────────

with tab2:
    st.subheader("Train the autoencoder")
    st.write("Loss should decrease over steps. Lower = better reconstruction.")

    col1, col2 = st.columns(2)
    num_steps = col1.slider("Steps", 100, 10000, 1000, 100)
    batch_size = col2.slider("Batch size", 4, 64, 16, 4)

    if st.button("Train", type="primary"):
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

            model = Autoencoder().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            progress = st.progress(0.0, text="Starting...")
            log_area = st.empty()
            log_lines = [f"Found {len(clips)} clips. Training on {device}..."]

            report_every = max(1, num_steps // 20)
            step = 0

            while step < num_steps:
                model.train()
                for batch in train_loader:
                    if step >= num_steps:
                        break
                    batch = batch.to(device)
                    recon = model(batch)
                    loss = F.l1_loss(recon, batch)
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
                            f"step {step:5d}/{num_steps} | train {loss.item():.4f} | val {val_loss:.4f}"
                        )
                        progress.progress(step / num_steps,
                                          text=f"Step {step}/{num_steps} — loss {loss.item():.4f}")
                        log_area.code("\n".join(log_lines))

            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT)
            progress.progress(1.0, text="Done!")
            log_lines.append(f"\nCheckpoint saved to {CHECKPOINT}")
            log_area.code("\n".join(log_lines))
            st.success("Training complete. Go to the Reconstruct tab to hear the results.")


# ── tab 3: reconstruct ────────────────────────────────────────────────────────

with tab3:
    st.subheader("Hear what the bottleneck kept")
    st.write("Upload or record a sound. The model compresses and reconstructs it.")

    if not os.path.exists(CHECKPOINT):
        st.warning("No trained model found. Train the model first.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Input**")
            r_upload = st.file_uploader("Upload to reconstruct", type=["wav", "mp3", "flac"],
                                        key="r_upload", label_visibility="collapsed")
            r_recorded = audio_recorder(text="Or record", pause_threshold=3.0, key="r_recorder")

            source = None
            if r_upload:
                source = r_upload.read()
                st.audio(source, format="audio/wav")
            elif r_recorded:
                source = r_recorded
                st.audio(source, format="audio/wav")

        with col2:
            st.markdown("**Reconstructed**")
            if source and st.button("Reconstruct", type="primary"):
                try:
                    waveform = load_waveform(source)
                    total = waveform.shape[1]
                    pad = (CLIP_LENGTH - total % CLIP_LENGTH) % CLIP_LENGTH
                    if pad:
                        waveform = F.pad(waveform, (0, pad))

                    model = get_model()
                    chunks_out = []
                    with torch.no_grad():
                        for start in range(0, waveform.shape[1], CLIP_LENGTH):
                            chunk = waveform[:, start:start + CLIP_LENGTH].unsqueeze(0).to(device)
                            chunks_out.append(model(chunk).squeeze(0).cpu())

                    output = torch.cat(chunks_out, dim=1)[:, :total]
                    st.audio(tensor_to_bytes(output), format="audio/wav")
                    st.caption(f"{total / SAMPLE_RATE:.2f}s reconstructed in {len(chunks_out)} chunk(s)")
                except Exception as e:
                    st.error(f"Error: {e}")
