import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio
import torchaudio.transforms as T
import numpy as np
import gradio as gr

from model import Autoencoder
from dataset import SAMPLE_RATE, CLIP_LENGTH

DATA_DIR = "data/raw"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT = os.path.join(CHECKPOINT_DIR, "model.pt")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
if os.path.exists(CHECKPOINT):
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()


# ── helpers ──────────────────────────────────────────────────────────────────

def filepath_to_tensor(path):
    """Load an audio file path to a mono float32 tensor [1, samples]."""
    waveform, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        waveform = T.Resample(sr, SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def load_clips():
    """Load all audio files in data/raw/ and slice into 1-second clips."""
    clips = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.lower().endswith(('.wav', '.mp3', '.flac', '.aif', '.aiff')):
            continue
        try:
            waveform = filepath_to_tensor(os.path.join(DATA_DIR, fname))
            for start in range(0, waveform.shape[1] - CLIP_LENGTH + 1, CLIP_LENGTH):
                clips.append(waveform[:, start:start + CLIP_LENGTH])
        except Exception as e:
            print(f"Skipping {fname}: {e}")
    return clips


def dataset_summary():
    files = [f for f in os.listdir(DATA_DIR)
             if f.lower().endswith(('.wav', '.mp3', '.flac', '.aif', '.aiff'))]
    clips = load_clips()
    return f"{len(files)} file(s) → {len(clips)} one-second clip(s)"


# ── tab 1: data ───────────────────────────────────────────────────────────────

def add_audio(path):
    # Gradio 6 passes a file path string for Audio inputs
    if path is None:
        return "No audio provided.", dataset_summary()
    try:
        waveform = filepath_to_tensor(path)
        existing = [f for f in os.listdir(DATA_DIR) if f.startswith("recording_")]
        out_path = os.path.join(DATA_DIR, f"recording_{len(existing):04d}.wav")
        torchaudio.save(out_path, waveform, SAMPLE_RATE)
        n_clips = waveform.shape[1] // CLIP_LENGTH
        msg = f"Saved {waveform.shape[1] / SAMPLE_RATE:.1f}s → {n_clips} clip(s)"
        return msg, dataset_summary()
    except Exception as e:
        return f"Error: {e}", dataset_summary()


# ── tab 2: train ──────────────────────────────────────────────────────────────

def train_model(num_steps, batch_size):
    global model

    clips = load_clips()
    if not clips:
        yield "No clips found. Add recordings in the Data tab first."
        return

    log = f"Found {len(clips)} clips. Training on {device}...\n"
    yield log

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

    report_every = max(1, int(num_steps) // 20)
    step = 0

    while step < int(num_steps):
        model.train()
        for batch in train_loader:
            if step >= int(num_steps):
                break
            batch = batch.to(device)
            recon = model(batch)
            loss = F.l1_loss(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % report_every == 0 or step == int(num_steps):
                model.eval()
                with torch.no_grad():
                    v = [F.l1_loss(model(b.to(device)), b.to(device)).item()
                         for b in val_loader]
                val_loss = sum(v) / len(v)
                model.train()
                log += f"step {step:5d}/{int(num_steps)} | train {loss.item():.4f} | val {val_loss:.4f}\n"
                yield log

    torch.save(model.state_dict(), CHECKPOINT)
    log += f"\nDone. Checkpoint saved to {CHECKPOINT}"
    yield log


# ── tab 3: reconstruct ────────────────────────────────────────────────────────

def reconstruct_audio(path):
    if path is None:
        return None, "No audio provided."
    if not os.path.exists(CHECKPOINT):
        return None, "No trained model found. Train the model first."

    try:
        waveform = filepath_to_tensor(path)
        total = waveform.shape[1]
        pad = (CLIP_LENGTH - total % CLIP_LENGTH) % CLIP_LENGTH
        if pad:
            waveform = F.pad(waveform, (0, pad))

        model.eval()
        chunks_out = []
        with torch.no_grad():
            for start in range(0, waveform.shape[1], CLIP_LENGTH):
                chunk = waveform[:, start:start + CLIP_LENGTH].unsqueeze(0).to(device)
                recon = model(chunk).squeeze(0).cpu()
                chunks_out.append(recon)

        output = torch.cat(chunks_out, dim=1)[:, :total]
        out_np = output.squeeze(0).numpy()
        return (SAMPLE_RATE, out_np), f"Reconstructed {total / SAMPLE_RATE:.2f}s in {len(chunks_out)} chunk(s)."
    except Exception as e:
        return None, f"Error: {e}"


# ── ui ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Generative Sound") as demo:
    gr.Markdown("# Waveform Autoencoder")
    gr.Markdown(
        "Record or upload sounds → train a compression model → hear what the bottleneck keeps.\n\n"
        f"**Latent:** 8 ch × 500 time steps = 4,000 units  |  **Device:** {device}"
    )

    with gr.Tab("1 · Data"):
        gr.Markdown(
            "Upload files or record directly. Long recordings are chopped into 1-second clips automatically."
        )
        audio_in = gr.Audio(sources=["upload", "microphone"], label="Audio", type="filepath")
        add_btn = gr.Button("Add to training data", variant="primary")
        add_msg = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            info_box = gr.Textbox(label="Dataset", value=dataset_summary(), interactive=False)
            refresh_btn = gr.Button("Refresh", scale=0)

        add_btn.click(add_audio, inputs=audio_in, outputs=[add_msg, info_box])
        refresh_btn.click(dataset_summary, outputs=info_box)

    with gr.Tab("2 · Train"):
        gr.Markdown("Train the autoencoder on your recordings. Loss should decrease over steps.")
        with gr.Row():
            steps_sl = gr.Slider(100, 10000, value=1000, step=100, label="Steps")
            batch_sl = gr.Slider(4, 64, value=16, step=4, label="Batch size")
        train_btn = gr.Button("Train", variant="primary")
        train_log = gr.Textbox(label="Log", lines=18, interactive=False, max_lines=18)
        train_btn.click(train_model, inputs=[steps_sl, batch_sl], outputs=train_log)

    with gr.Tab("3 · Reconstruct"):
        gr.Markdown(
            "Upload or record a sound. The model compresses it through the bottleneck and reconstructs it. "
            "Listen to what changed."
        )
        with gr.Row():
            recon_in = gr.Audio(sources=["upload", "microphone"], label="Input", type="filepath")
            recon_out = gr.Audio(label="Reconstructed", interactive=False)
        recon_btn = gr.Button("Reconstruct", variant="primary")
        recon_msg = gr.Textbox(label="Info", interactive=False)
        recon_btn.click(reconstruct_audio, inputs=recon_in, outputs=[recon_out, recon_msg])

demo.launch()
