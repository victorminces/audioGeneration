import os
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
import soundfile as sf


SAMPLE_RATE = 16000
N_MELS     = 80
N_FFT      = 1024
HOP_LENGTH = 256
CLIP_FRAMES  = 1        #  1 STFT frame per clip (256 / 16000 ≈ 16 ms)
CLIP_SAMPLES = 800      # 800 / 16000 = 50 ms of raw waveform

_mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)


def waveform_to_mel(waveform):
    """(1, samples) → (1, N_MELS, frames) log-mel spectrogram."""
    return (_mel_transform(waveform) + 1e-6).log()


class AudioDataset(Dataset):
    def __init__(self, folder):
        self.clips = []
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(('.wav', '.mp3', '.flac', '.aif', '.aiff')):
                self._load_file(os.path.join(folder, fname))
        print(f"Dataset: {len(self.clips)} clips from {folder}")

    def _load_file(self, path):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.tensor(data.T)
        if sr != SAMPLE_RATE:
            waveform = T.Resample(sr, SAMPLE_RATE)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        mel = waveform_to_mel(waveform)           # (1, N_MELS, frames)
        total = mel.shape[2]
        for start in range(0, total - CLIP_FRAMES + 1, CLIP_FRAMES):
            self.clips.append(mel[:, :, start:start + CLIP_FRAMES])

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        return self.clips[idx]
