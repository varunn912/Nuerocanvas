import os
import torch
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# EEG Processing settings
EEG_SAMPLING_RATE = int(os.getenv("EEG_SAMPLING_RATE", 256))
EEG_LOW_CUT = float(os.getenv("EEG_LOW_CUT", 1.0))
EEG_HIGH_CUT = float(os.getenv("EEG_HIGH_CUT", 50.0))

# Model settings
FUSION_MODEL_PATH = MODELS_DIR / "eeg_text_fusion_model.pth"
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
SD_MODEL_NAME = "stabilityai/stable-diffusion-2-1"

# Training settings
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
EPOCHS = int(os.getenv("EPOCHS", 10))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))

# UI settings
UI_SHARE = os.getenv("UI_SHARE", "False").lower() == "true"
UI_DEBUG = os.getenv("UI_DEBUG", "False").lower() == "true"

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')