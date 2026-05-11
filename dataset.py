import os
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
import soundfile as sf


SAMPLE_RATE = 16000
CLIP_LENGTH = 800    # 50 ms


class AudioDataset(Dataset):
    def __init__(self, folder):
        self.clips = []
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(('.wav', '.mp3', '.flac', '.aif', '.aiff')):
                self._load_file(os.path.join(folder, fname))
        print(f"Dataset: {len(self.clips)} clips from {folder}")

    def _load_file(self, path):
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.tensor(data.T)  # (channels, samples)
        if sr != SAMPLE_RATE:
            waveform = T.Resample(sr, SAMPLE_RATE)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        total = waveform.shape[1]
        for start in range(0, total - CLIP_LENGTH + 1, CLIP_LENGTH):
            clip = waveform[:, start:start + CLIP_LENGTH]
            self.clips.append(clip)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        return self.clips[idx]
