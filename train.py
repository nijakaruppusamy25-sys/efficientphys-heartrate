import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from data.dataset_loader import EfficientPhysDataset
from models.efficientphys_model import EfficientPhys_Conv


def train():

    print("Loading Dataset...")

    dataset = EfficientPhysDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    print("Dataset Size:", len(dataset))

    print("Loading Model...")

    model = EfficientPhys_Conv(
        frame_depth=config.FRAME_DEPTH,
        img_size=config.IMG_SIZE
    ).to(config.DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    best_loss = float("inf")

    print("\nTraining Started...\n")

    for epoch in range(config.EPOCHS):

        model.train()

        running_loss = 0.0

        for frames, bpm in dataloader:

            frames = frames.to(config.DEVICE)
            bpm = bpm.float().to(config.DEVICE)

            # (B,T,C,H,W) → (B*T,C,H,W)
            b, t, c, h, w = frames.shape
            frames = frames.reshape(b * t, c, h, w)

            optimizer.zero_grad()

            outputs = model(frames)

            outputs = outputs.reshape(b, t, -1).mean(dim=1).squeeze()

            print("Predictions:", outputs.detach().cpu().numpy())
            print("Ground Truth:", bpm.detach().cpu().numpy())
            print("=" * 50)

            loss = criterion(outputs, bpm)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)

        print(
            f"Epoch [{epoch+1}/{config.EPOCHS}] "
            f"Loss : {epoch_loss:.4f}"
        )

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            torch.save(
                model.state_dict(),
                config.MODEL_PATH
            )

            print("Best model saved.")

    print("\nTraining Completed.")


if __name__ == "__main__":
    train()