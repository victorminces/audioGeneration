"""Train the DDSP timbre model used by the gesture instrument.

Analyzes clips from data/raw/ into (f0, loudness, mel) once, caches the
result (slow: torchcrepe pitch tracking), then trains ddsp.DDSP end-to-end
against a multi-scale spectral loss. Saves a checkpoint that
mapper.load_latest_ddsp() will pick up automatically (newest "domain": "ddsp"
config in checkpoints/).

Usage:
    python download_data.py --dataset librispeech   # populate data/raw/ first
    python train_ddsp.py                             # defaults: 20000 steps, 2s clips
    python train_ddsp.py --steps 200 --batch-size 8   # quick smoke test
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from torch.utils.data import DataLoader, Dataset

import ddsp

PROJ = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJ, "data", "raw")
CHECKPOINT_DIR = os.path.join(PROJ, "checkpoints")


def _iter_clips(data_dir, clip_samples):
    """Yield (1, clip_samples) waveform tensors: each file split into as many
    non-overlapping clip-length windows as fit."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.flac")) +
                   glob.glob(os.path.join(data_dir, "*.wav")))
    for p in paths:
        wav, sr = sf.read(p, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        if sr != ddsp.SR:
            wav_t = AF.resample(wav_t, sr, ddsp.SR)
        n_clips = wav_t.shape[1] // clip_samples
        for i in range(n_clips):
            yield wav_t[:, i * clip_samples:(i + 1) * clip_samples]


def build_cache(data_dir, clip_seconds, cache_path):
    """Analyze every clip in data_dir into (wav, f0, loud, mel) and save as .npz."""
    clip_samples = int(clip_seconds * ddsp.SR)
    n_frames = clip_samples // ddsp.HOP

    wavs, f0s, louds, mels = [], [], [], []
    for wav in _iter_clips(data_dir, clip_samples):
        f0 = ddsp.extract_f0(wav, n_frames)
        loud = ddsp.extract_loudness(wav, n_frames)
        mel = ddsp.mel_frames(wav, n_frames)
        wavs.append(wav[0].numpy())
        f0s.append(f0.numpy())
        louds.append(loud.numpy())
        mels.append(mel.numpy())
        print(f"\r  analyzed {len(wavs)} clips", end="", flush=True)
    print()

    if not wavs:
        raise RuntimeError(f"No audio clips found in {data_dir} (need files >= "
                           f"{clip_seconds}s — run download_data.py first).")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, wav=np.stack(wavs), f0=np.stack(f0s),
             loud=np.stack(louds), mel=np.stack(mels))
    print(f"Cached {len(wavs)} clips -> {cache_path}")


class ClipDataset(Dataset):
    def __init__(self, wav, f0, loud, mel):
        self.wav, self.f0, self.loud, self.mel = wav, f0, loud, mel

    def __len__(self):
        return len(self.wav)

    def __getitem__(self, i):
        return self.wav[i], self.f0[i], self.loud[i], self.mel[i]


def _next_name():
    taken = [0]
    for p in glob.glob(os.path.join(CHECKPOINT_DIR, "ddsp_*.json")):
        m = re.match(r"ddsp_(\d+)$", os.path.splitext(os.path.basename(p))[0])
        if m:
            taken.append(int(m.group(1)))
    return f"ddsp_{max(taken) + 1:03d}"


def train(args):
    cache_path = args.cache or os.path.join(CHECKPOINT_DIR, "ddsp_cache.npz")
    if args.rebuild_cache or not os.path.exists(cache_path):
        build_cache(args.data_dir, args.clip_seconds, cache_path)

    data = np.load(cache_path)
    wav, f0, loud, mel = (torch.from_numpy(data[k]) for k in ("wav", "f0", "loud", "mel"))
    n = len(wav)

    n_val = max(1, int(n * args.val_frac))
    perm = torch.randperm(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = ClipDataset(wav[train_idx], f0[train_idx], loud[train_idx], mel[train_idx])
    val_ds = ClipDataset(wav[val_idx], f0[val_idx], loud[val_idx], mel[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=len(train_idx) >= args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ddsp.DDSP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps,
                                                           eta_min=1e-5)

    loss_history = []
    report_every = max(1, args.steps // 40)
    step = 0
    recon_loss_val = 0.0

    while step < args.steps:
        model.train()
        for wav_b, f0_b, loud_b, mel_b in train_loader:
            if step >= args.steps:
                break
            wav_b, f0_b, loud_b, mel_b = (t.to(device) for t in (wav_b, f0_b, loud_b, mel_b))

            recon = model(mel_b, f0_b, loud_b)
            loss = ddsp.multiscale_spectral_loss(recon, wav_b)
            recon_loss_val = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if step % report_every == 0 or step == args.steps:
                model.eval()
                with torch.no_grad():
                    v_losses = []
                    for wav_v, f0_v, loud_v, mel_v in val_loader:
                        wav_v, f0_v, loud_v, mel_v = (t.to(device)
                                                      for t in (wav_v, f0_v, loud_v, mel_v))
                        v_recon = model(mel_v, f0_v, loud_v)
                        v_losses.append(ddsp.multiscale_spectral_loss(v_recon, wav_v).item())
                val_loss = sum(v_losses) / len(v_losses)
                model.train()

                print(f"step {step:6d}/{args.steps} | recon {recon_loss_val:.4f} | "
                      f"val {val_loss:.4f}")
                loss_history.append({"step": step, "recon": round(recon_loss_val, 5),
                                     "val_recon": round(val_loss, 5)})

    name = args.out_name or _next_name()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    pt_path = os.path.join(CHECKPOINT_DIR, f"{name}.pt")
    torch.save(model.state_dict(), pt_path)

    cfg = {
        "name": name, "filename": f"{name}.pt",
        "domain": "ddsp", "model_type": "ddsp", "arch": "ddsp",
        "latent_ch": ddsp.Z_DIM, "latent_units": ddsp.Z_DIM,
        "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
        "clip_seconds": args.clip_seconds, "val_frac": args.val_frac,
        "clips_used": n, "loss_history": loss_history,
        "note": f"DDSP harmonic+noise, timbre z={ddsp.Z_DIM}, trained via train_ddsp.py",
    }
    with open(os.path.join(CHECKPOINT_DIR, f"{name}.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Saved {pt_path} ({name}.json)")
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--cache", default=None,
                        help="Feature cache path (default: checkpoints/ddsp_cache.npz)")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Recompute the feature cache even if it already exists")
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--out-name", default=None,
                        help="Checkpoint name (default: auto-incrementing ddsp_NNN)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
