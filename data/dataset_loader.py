import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config


class EfficientPhysDataset(Dataset):

    def __init__(self):
        self.labels = pd.read_csv(config.LABELS_FILE)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        video_name = self.labels.iloc[index]["video"]
        bpm = float(self.labels.iloc[index]["bpm"])

        npy_file = video_name.replace(".avi", ".npy")
        npy_path = os.path.join(config.SEQUENCES_DIR, npy_file)

        frames = np.load(npy_path).astype(np.float32)

        if frames.max() > 1:
            frames /= 255.0

        # ---------------------------------
        # Select one 20-frame clip
        # ---------------------------------
        total_frames = len(frames)

        if total_frames >= config.FRAME_DEPTH:
            start = random.randint(
                0,
                total_frames - config.FRAME_DEPTH
            )
            frames = frames[start:start + config.FRAME_DEPTH]

        else:
            pad = config.FRAME_DEPTH - total_frames
            frames = np.concatenate(
                [frames, np.repeat(frames[-1:], pad, axis=0)],
                axis=0
            )

        # (T,H,W,C) → (T,C,H,W)
        frames = np.transpose(frames, (0, 3, 1, 2))

        frames = torch.tensor(frames, dtype=torch.float32)
        bpm = torch.tensor(bpm, dtype=torch.float32)

        print("Frames shape:", frames.shape)
        print("Min:", frames.min().item())
        print("Max:", frames.max().item())
        print("BPM:", bpm.item())
        print("-" * 40)

        return frames, bpm