import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import AudioDataset
from model import Autoencoder


DATA_DIR = "data/raw"
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_STEPS = 1000
VAL_CLIPS = 10


def evaluate(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            total += F.l1_loss(recon, batch).item()
            count += 1
    return total / count if count > 0 else 0.0


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    dataset = AudioDataset(DATA_DIR)
    if len(dataset) == 0:
        print("No audio clips found. Add WAV files to data/raw/ and try again.")
        return

    val_size = min(VAL_CLIPS, max(1, len(dataset) // 5))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    step = 0
    while step < NUM_STEPS:
        model.train()
        for batch in train_loader:
            if step >= NUM_STEPS:
                break
            batch = batch.to(device)
            recon = model(batch)
            loss = F.l1_loss(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 100 == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"step {step:5d} | train loss {loss.item():.4f} | val loss {val_loss:.4f}")

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    train()
