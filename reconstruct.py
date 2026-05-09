import os
import torch
import torchaudio

from dataset import AudioDataset, SAMPLE_RATE
from model import Autoencoder


DATA_DIR = "data/raw"
OUTPUT_DIR = "outputs"
CHECKPOINT = "checkpoints/model.pt"
NUM_CLIPS = 10


def reconstruct():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    dataset = AudioDataset(DATA_DIR)
    n = min(NUM_CLIPS, len(dataset))

    with torch.no_grad():
        for i in range(n):
            clip = dataset[i]
            x = clip.unsqueeze(0).to(device)
            recon = model(x).squeeze(0).cpu()

            torchaudio.save(os.path.join(OUTPUT_DIR, f"original_{i:02d}.wav"), clip, SAMPLE_RATE)
            torchaudio.save(os.path.join(OUTPUT_DIR, f"reconstructed_{i:02d}.wav"), recon, SAMPLE_RATE)
            print(f"Saved clip {i:02d}")

    print(f"Done. Check the outputs/ folder.")


if __name__ == "__main__":
    reconstruct()
