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
    python train_ddsp.py --resume ddsp_003 --steps 2000   # train 2000 more steps
                                                            # on top of ddsp_003
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


def synth_tones(n, clip_samples, rng, air_max=0.03, f_lo=400.0):
    """Whistle-like clips spanning the upper pitch range.

    LibriSpeech f0 sits around 80-300 Hz, which after scaling occupies only
    the bottom of the model's f0 input. Train on it alone and the top of a
    widened range is never exercised — the network has no idea what to do
    there, which is exactly why whistling produced nothing. These fill it in.

    They are deliberately whistle-shaped: near-sinusoidal, with vibrato, a
    weak second partial and a little breath. The model already renders pure
    tones well at pitches it *has* seen, so what it needs is examples at the
    pitches, not a new kind of timbre.
    """
    out = []
    t = np.arange(clip_samples) / ddsp.SR
    for _ in range(n):
        f0 = float(np.exp(rng.uniform(np.log(f_lo), np.log(ddsp.F0_DETECT_MAX * 0.99))))
        drift = f0 * (1 + rng.uniform(-0.25, 0.25) * t / max(t[-1], 1e-9))
        vib = drift * (1 + rng.uniform(0.002, 0.012)
                       * np.sin(2 * np.pi * rng.uniform(3.5, 7.0) * t))
        ph = 2 * np.pi * np.cumsum(vib) / ddsp.SR
        x = np.sin(ph) + rng.uniform(0.02, 0.15) * np.sin(2 * ph)
        # Air down to exactly zero. Starting this at 0.005 was the bug behind
        # breathy whistles: nothing the model ever saw was cleaner than 0.5%
        # noise, so a real whistle fell off the end of its experience and the
        # amplitude head answered with no tone at all — 220x collapse at
        # 1800 Hz. A whistle is close to a pure sine; the training set has to
        # contain that.
        x += rng.uniform(0.0, air_max) * rng.normal(0, 1, clip_samples)
        # fade in/out so the take has attacks rather than clicks
        env = np.ones(clip_samples)
        k = int(0.05 * ddsp.SR)
        env[:k] = np.linspace(0, 1, k); env[-k:] = np.linspace(1, 0, k)
        x = x * env * rng.uniform(0.25, 0.7) / (np.abs(x).max() + 1e-9)
        out.append(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
    return out


def analyze_clips(clips, n_frames):
    """(1, samples) waveforms -> stacked (wav, f0, loud, mel) arrays."""
    wavs, f0s, louds, mels = [], [], [], []
    for wav in clips:
        wavs.append(wav[0].numpy())
        f0s.append(ddsp.extract_f0(wav, n_frames).numpy())
        louds.append(ddsp.extract_loudness(wav, n_frames).numpy())
        mels.append(ddsp.mel_frames(wav, n_frames).numpy())
        print(f"\r  analyzed {len(wavs)} clips", end="", flush=True)
    print()
    return [np.stack(a) for a in (wavs, f0s, louds, mels)]


def append_tones(cache_path, n, clip_seconds, air_max, f_lo, seed):
    """Add synthetic tones to an existing cache without re-analysing the rest.

    The speech clips take far longer to analyse than the tones do (pitch
    tracking dominates), and they have not changed — so filling a gap in the
    tone coverage should not cost a full rebuild.
    """
    clip_samples = int(clip_seconds * ddsp.SR)
    n_frames = clip_samples // ddsp.HOP
    rng = np.random.default_rng(seed)
    print(f"  generating {n} tones: air 0-{air_max}, {f_lo:.0f}-"
          f"{ddsp.F0_DETECT_MAX * 0.99:.0f} Hz")
    fresh = analyze_clips(synth_tones(n, clip_samples, rng, air_max, f_lo), n_frames)
    with np.load(cache_path) as old:
        n_before = len(old["wav"])
        merged = {k: np.concatenate([old[k], f]) for k, f in
                  zip(("wav", "f0", "loud", "mel"), fresh)}
    # np.load is lazy, so the handle must be closed before overwriting the same
    # path — and nothing may read from it afterwards
    np.savez(cache_path, **merged)
    print(f"  cache {n_before} -> {len(merged['wav'])} clips")


def build_cache(data_dir, clip_seconds, cache_path, synth_frac=0.0, seed=0):
    """Analyze every clip in data_dir into (wav, f0, loud, mel) and save as .npz."""
    clip_samples = int(clip_seconds * ddsp.SR)
    n_frames = clip_samples // ddsp.HOP

    clips = list(_iter_clips(data_dir, clip_samples))
    if synth_frac > 0:
        rng = np.random.default_rng(seed)
        n = int(round(len(clips) * synth_frac))
        clips += synth_tones(n, clip_samples, rng)
        print(f"  + {n} synthetic high-pitch clips (400-{ddsp.F0_DETECT_MAX:.0f} Hz)")

    wavs, f0s, louds, mels = [], [], [], []
    for wav in clips:
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
    def __init__(self, wav, f0, rel, mel):
        self.wav, self.f0, self.rel, self.mel = wav, f0, rel, mel

    def __len__(self):
        return len(self.wav)

    def __getitem__(self, i):
        return self.wav[i], self.f0[i], self.rel[i], self.mel[i]


def derive_features(wav):
    """Cached waveforms -> (loudness, log-mel), recomputed rather than trusted.

    `wav` and `f0` are what the cache exists for: crepe is nearly all of the
    build time. Loudness and log-mel are two cheap transforms, so deriving them
    on load costs seconds and means they can never drift out of step with
    ddsp.py — which they silently did when the analysis floors changed from
    absolute to relative, and would have again on any future change.

    Written into preallocated tensors rather than accumulated in lists: this
    machine runs the training with about 2 GB free, and a list of 2419 tensors
    plus the stack that doubles it was enough to segfault the process.
    """
    n = len(wav)
    n_frames = wav.shape[1] // ddsp.HOP
    loud = torch.empty(n, n_frames)
    mel = torch.empty(n, n_frames, ddsp.N_MELS)
    for i in range(n):
        w = wav[i].unsqueeze(0)
        loud[i] = ddsp.extract_loudness(w, n_frames)
        mel[i] = ddsp.mel_frames(w, n_frames)
    return loud, mel


def adapt_checkpoint(model, old_pt, device):
    """Warm-start the factored model from a checkpoint that still took loudness.

    The old decoder computed W @ [f0, loud, z] + b. Deleting the loudness column
    is the same as pinning `loud` to a constant, and which constant is ours to
    pick: folding W[:, i_loud] * scale_loud(REF_LOUD) into the bias makes the
    17-input decoder exactly the old 18-input one held at normal singing level.
    Not an approximation — the same function. (A plain column drop would pin it
    at 0 in scaled units, i.e. log-RMS -8, the near-silence operating point
    where the old model already misbehaved.)

    Everything else transfers unchanged except head_amp, which is deleted
    because level is no longer predicted, and head_mix, which is new: nothing in
    the old model corresponds to it, so it starts at a constant mostly-harmonic
    balance and learns from there.
    """
    old = torch.load(old_pt, map_location=device, weights_only=True)
    new = model.state_dict()
    scaled_ref = (ddsp.REF_LOUD + 8.0) / 8.0        # the old module's scale_loud
    OLD_IN_DIM, I_LOUD = 2 + ddsp.Z_DIM, 1          # x was [f0, loud, z...]
    kept, folded, fresh = [], [], []

    for k, v in new.items():
        if k not in old:
            fresh.append(k)
            continue
        o = old[k]
        if o.shape == v.shape:
            new[k] = o
            kept.append(k)
        elif o.dim() == 2 and o.shape[0] == v.shape[0] and o.shape[1] == v.shape[1] + 1:
            # x is the trailing block of the input: it is the whole input for
            # mlp_in.0, and the last OLD_IN_DIM columns of cat([h, x]) for
            # mlp_out.0. So the loudness column is always this far in.
            col = o.shape[1] - OLD_IN_DIM + I_LOUD
            bias_key = k.replace(".weight", ".bias")
            new[k] = torch.cat([o[:, :col], o[:, col + 1:]], dim=1)
            if bias_key in new and bias_key in old:
                new[bias_key] = old[bias_key] + o[:, col] * scaled_ref
            folded.append(f"{k} col {col}")
        else:
            fresh.append(f"{k} (shape {tuple(o.shape)} -> {tuple(v.shape)})")

    if "decoder.head_mix.weight" in new:
        new["decoder.head_mix.weight"] = torch.zeros_like(new["decoder.head_mix.weight"])
        new["decoder.head_mix.bias"] = torch.full_like(
            new["decoder.head_mix.bias"], -2.197)     # sigmoid -> 0.10 noise
    if "decoder.head_env.weight" in new:
        # A filter-mode model cannot inherit head_harm: that is indexed by
        # harmonic number and this by frequency, and no fixed map between them
        # exists without knowing f0. So it starts from a plausible spectrum
        # instead of noise — falling about 5 dB per octave, roughly what a
        # voiced source does. Worth remembering when comparing the two modes:
        # the amplitude model inherits a *trained* head here and this one does
        # not, which flatters the amplitude model early in a run.
        new["decoder.head_env.weight"] = torch.zeros_like(new["decoder.head_env.weight"])
        tilt = torch.logspace(0, -2, ddsp.N_ENV)      # 1.0 down to 0.01
        new["decoder.head_env.bias"] = torch.logit(
            (tilt / 2.0) ** (1 / 2.3)).to(new["decoder.head_env.bias"].dtype)
    model.load_state_dict(new)
    dropped = [k for k in old if k not in new]
    print(f"  adapted: {len(kept)} tensors copied, {len(folded)} folded "
          f"({', '.join(folded)}), {len(fresh)} fresh ({', '.join(fresh)})")
    print(f"  dropped from the old checkpoint: {', '.join(dropped) or 'none'}")


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
        build_cache(args.data_dir, args.clip_seconds, cache_path,
                    synth_frac=args.synth_frac, seed=args.seed)
    elif args.append_tones:
        append_tones(cache_path, args.append_tones, args.clip_seconds,
                     args.append_air_max, args.append_f_lo, args.seed + 1)

    with np.load(cache_path) as data:
        wav = torch.from_numpy(data["wav"])   # ~310 MB; everything below is
        f0 = torch.from_numpy(data["f0"])     # careful not to copy it again
    n = len(wav)
    print(f"deriving loudness and log-mel for {n} clips...", flush=True)
    loud, mel = derive_features(wav)

    # Train on level-normalised clips. Every clip is scaled to the same
    # performed level and the resulting per-frame RMS is handed to the model
    # directly, so the level term of the loss is matched by construction and
    # what is left to reduce is spectral shape. Note this makes val losses
    # incomparable with ddsp_001..005, which were scored on absolute level too.
    ref = ddsp.take_level(loud)                             # (n,)
    rel = loud - ref[:, None]                               # (n, T)
    wav.mul_(torch.exp(ddsp.REF_LOUD - ref)[:, None])       # in place: see above
    print(f"  performed level p5 {np.percentile(ref, 5):+.2f}  "
          f"p50 {np.percentile(ref, 50):+.2f}  p95 {np.percentile(ref, 95):+.2f}"
          f"  -> all normalised to {ddsp.REF_LOUD:+.2f}", flush=True)

    n_val = max(1, int(n * args.val_frac))
    # Seeded so the split is the same on every run. Without this, two runs over
    # the same cache validate against different clips and their val losses are
    # not comparable — including a run and its own --resume continuation.
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = ClipDataset(wav[train_idx], f0[train_idx], rel[train_idx], mel[train_idx])
    val_ds = ClipDataset(wav[val_idx], f0[val_idx], rel[val_idx], mel[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=len(train_idx) >= args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_step = 0
    loss_history = []
    f0_min, f0_max = ddsp.F0_SCALE_MIN, ddsp.F0_SCALE_MAX
    name = args.out_name or args.resume or _next_name()
    prev_cfg = None
    if args.resume:
        resume_json = os.path.join(CHECKPOINT_DIR, f"{args.resume}.json")
        with open(resume_json) as f:
            prev_cfg = json.load(f)
        start_step = prev_cfg.get("steps_done", prev_cfg.get("steps", 0))
        loss_history = prev_cfg.get("loss_history", [])
        # The scale is baked into the weights being resumed, so it has to come
        # from the checkpoint rather than today's default. Without this, loading
        # a 60-2000 model under a 60-600 default would misread every pitch by
        # nearly an octave and a half and quietly train on the error.
        f0_min = float(prev_cfg.get("f0_min", 60.0))
        f0_max = float(prev_cfg.get("f0_max", 600.0))

    harm_mode = (prev_cfg or {}).get("harm_mode", args.harm_mode)
    model = ddsp.DDSP(f0_min=f0_min, f0_max=f0_max, harm_mode=harm_mode).to(device)
    print(f"harmonic mode: {harm_mode}"
          + ("  (direct per-harmonic amplitudes)" if harm_mode == "amp" else
             "  (spectral envelope sampled at k*f0)"))
    if args.resume:
        resume_pt = os.path.join(CHECKPOINT_DIR, f"{args.resume}.pt")
        model.load_state_dict(torch.load(resume_pt, map_location=device))
        print(f"Resuming from {args.resume} at step {start_step} "
              f"(f0 scale {f0_min:.0f}-{f0_max:.0f} Hz, from its config)")
    elif args.adapt:
        adapt_pt = os.path.join(CHECKPOINT_DIR, f"{args.adapt}.pt")
        print(f"Warm-starting from pre-factoring checkpoint {args.adapt}:")
        adapt_checkpoint(model, adapt_pt, device)

    target_step = start_step + args.steps

    # Fresh optimizer + a full cosine anneal over just this phase's steps
    # (matches this codebase's existing Train/Continue convention).
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps,
                                                           eta_min=1e-5)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def save(step_done):
        pt_path = os.path.join(CHECKPOINT_DIR, f"{name}.pt")
        torch.save(model.state_dict(), pt_path)
        cfg = {
            "name": name, "filename": f"{name}.pt",
            "domain": "ddsp", "model_type": "ddsp", "arch": ddsp.ARCH,
            "latent_ch": ddsp.Z_DIM, "latent_units": ddsp.Z_DIM,
            "steps": target_step, "steps_done": step_done,
            "batch_size": args.batch_size, "lr": args.lr,
            "clip_seconds": args.clip_seconds, "val_frac": args.val_frac,
            "f0_min": model.f0_min, "f0_max": model.f0_max,
            "harm_mode": model.harm_mode,
            "synth_frac": args.synth_frac,
            "clips_used": n, "loss_history": loss_history,
            "note": f"DDSP harmonic+noise, timbre z={ddsp.Z_DIM}, trained via train_ddsp.py"
                   + ("" if step_done == target_step else " (in progress)"),
        }
        with open(os.path.join(CHECKPOINT_DIR, f"{name}.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        return pt_path

    report_every = max(1, args.steps // 40)
    step = start_step
    recon_loss_val = 0.0

    while step < target_step:
        model.train()
        for wav_b, f0_b, rel_b, mel_b in train_loader:
            if step >= target_step:
                break
            wav_b, f0_b, rel_b, mel_b = (t.to(device) for t in (wav_b, f0_b, rel_b, mel_b))

            recon = model(mel_b, f0_b, ddsp.level_to_amp(rel_b))
            loss = ddsp.multiscale_spectral_loss(recon, wav_b)
            recon_loss_val = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if step % report_every == 0 or step == target_step:
                model.eval()
                with torch.no_grad():
                    v_losses = []
                    for wav_v, f0_v, rel_v, mel_v in val_loader:
                        wav_v, f0_v, rel_v, mel_v = (t.to(device)
                                                     for t in (wav_v, f0_v, rel_v, mel_v))
                        v_recon = model(mel_v, f0_v, ddsp.level_to_amp(rel_v))
                        v_losses.append(ddsp.multiscale_spectral_loss(v_recon, wav_v).item())
                val_loss = sum(v_losses) / len(v_losses)
                model.train()

                print(f"step {step:6d}/{target_step} | recon {recon_loss_val:.4f} | "
                      f"val {val_loss:.4f}")
                loss_history.append({"step": step, "recon": round(recon_loss_val, 5),
                                     "val_recon": round(val_loss, 5)})

                pt_path = save(step)
                print(f"  saved {pt_path} (steps_done={step})")

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
    parser.add_argument("--steps", type=int, default=20000,
                        help="Steps to train this phase (additional steps, if --resume is set)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--synth-frac", type=float, default=0.30,
                        help="synthetic high-pitch clips, as a fraction of the "
                             "real ones, so the top of the f0 range is trained")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-name", default=None,
                        help="Checkpoint name (default: --resume's name, or an "
                             "auto-incrementing ddsp_NNN for a fresh run)")
    parser.add_argument("--append-tones", type=int, default=0,
                        help="add N synthetic tones to the existing cache and "
                             "reuse the rest, instead of rebuilding it")
    parser.add_argument("--append-air-max", type=float, default=0.008,
                        help="upper bound on added noise for --append-tones; the "
                             "lower bound is always 0, which is the point")
    parser.add_argument("--append-f-lo", type=float, default=700.0,
                        help="lowest pitch for --append-tones")
    parser.add_argument("--resume", default=None,
                        help="Existing checkpoint name to continue training from "
                             "(e.g. ddsp_003)")
    parser.add_argument("--harm-mode", default="amp", choices=list(ddsp.HARM_MODES),
                        help="How harmonic amplitudes are produced. 'amp': the "
                             "decoder predicts all 60 directly. 'filter': it "
                             "predicts a spectral envelope on absolute "
                             "frequency and they are read off it at k*f0 "
                             "(source-filter). Ignored with --resume, which "
                             "takes the mode from the checkpoint.")
    parser.add_argument("--adapt", default=None,
                        help="Warm-start from a pre-factoring checkpoint (e.g. "
                             "ddsp_004): fold its loudness input into the biases "
                             "at the reference level and keep everything else. "
                             "Use --resume for checkpoints of this architecture.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
