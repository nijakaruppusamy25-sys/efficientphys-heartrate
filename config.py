import torch
import os

# ==============================
# Dataset Paths
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
VIDEOS_DIR = os.path.join(DATASET_DIR, "videos")
PROCESSED_FACES_DIR = os.path.join(DATASET_DIR, "processed_faces")
SEQUENCES_DIR = os.path.join(DATASET_DIR, "sequences")
LABELS_FILE = os.path.join(DATASET_DIR, "labels.csv")

# ==============================
# Output Paths
# ==============================
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# Image Settings
# ==============================
IMG_SIZE = 36          # EfficientPhys default
CHANNELS = 3

# ==============================
# Video Settings
# ==============================
FRAME_DEPTH = 20       # EfficientPhys default
MAX_FRAMES = 150       # Number of frames extracted per video
FPS = 30

# ==============================
# Training Settings
# ==============================
BATCH_SIZE = 4
EPOCHS = 1

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# ==============================
# Device
# ==============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# Model
# ==============================
MODEL_NAME = "EfficientPhys"

# ==============================
# Checkpoint
# ==============================
MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "efficientphys_best.pth"
)