import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from tqdm import tqdm
import config

INPUT_DIR = config.PROCESSED_FACES_DIR
OUTPUT_DIR = config.SEQUENCES_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess_video(video_folder):

    frames = []

    images = sorted(os.listdir(video_folder))

    for image in images:

        image_path = os.path.join(video_folder, image)

        frame = cv2.imread(image_path)

        if frame is None:
            continue

        # Resize to EfficientPhys input size
        frame = cv2.resize(frame, (config.IMG_SIZE, config.IMG_SIZE))

        # Convert BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = frame.astype(np.float32)

        # Normalize
        frame /= 255.0

        frames.append(frame)

    if len(frames) == 0:
        return None

    frames = np.array(frames)

    # Keep only first MAX_FRAMES
    if len(frames) > config.MAX_FRAMES:
        frames = frames[:config.MAX_FRAMES]

    # Pad short videos
    while len(frames) < config.MAX_FRAMES:
        frames = np.concatenate(
            (frames, frames[-1][None]),
            axis=0
        )

    return frames


print("Creating frame sequences...")

folders = sorted(os.listdir(INPUT_DIR))

for folder in tqdm(folders):

    folder_path = os.path.join(INPUT_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    sequence = preprocess_video(folder_path)

    if sequence is None:
        continue

    save_path = os.path.join(
        OUTPUT_DIR,
        folder + ".npy"
    )

    np.save(save_path, sequence)

print("Done!")