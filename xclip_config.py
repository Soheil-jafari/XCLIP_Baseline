"""
Central configuration file for the X-CLIP baseline experiments.
This file is adapted directly from your `project_config.py` to ensure all
paths and settings are correct for your ML server environment.
"""
import os
import torch

class Config:
    # --- Base Directories on ML Server (from your config) ---
    ML_SERVER_HOME = "/users/2/240331715/"
    UNIFIED_MEDICAL_VIDEOS_DIR = os.path.join(ML_SERVER_HOME, "data", "unified_medical_videos")
    PROJECT_ROOT = os.path.join(ML_SERVER_HOME, "project_folder", "Language-Guided-Endoscopy-Localization")

    # --- Data Paths ---
    EXTRACTED_FRAMES_DIR = os.path.join(UNIFIED_MEDICAL_VIDEOS_DIR, "extracted_frames")
    TRAIN_TRIPLETS_CSV_PATH = os.path.join(UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets", "cholec80_train_triplets.csv")
    VAL_TRIPLETS_CSV_PATH = os.path.join(UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets", "cholec80_val_triplets.csv")
    TEST_TRIPLETS_CSV_PATH = os.path.join(UNIFIED_MEDICAL_VIDEOS_DIR, "final_triplets", "cholec80_test_triplets.csv")

    # --- Output & Checkpoint Paths for X-CLIP Experiments ---
    # We will create a dedicated directory for these experiments to keep things clean.
    XCLIP_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "xclip_baselines")
    XCLIP_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "xclip_baselines")

    # --- Core Model & Data Settings ---
    MODEL_NAME = "microsoft/xclip-base-patch32" # The official X-CLIP model for your paper
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 8  # Set this based on your server's CPU cores for fast data loading
    FRAME_RATE = 30
    FRAME_GLOB = "*.jpg" # e.g., "frame_%06d.jpg" or "*.jpg"

    # --- Training Hyperparameters ---
    # These can be overridden by command-line arguments in the scripts.
    
    # For Fine-Tuning
    FT_EPOCHS = 5
    FT_BATCH_SIZE = 16
    FT_LEARNING_RATE = 1e-5

    # For Linear-Probe
    LP_EPOCHS = 10
    LP_BATCH_SIZE = 32
    LP_LEARNING_RATE = 1e-3
    
    # --- Evaluation Settings ---
    EVAL_BATCH_SIZE = 64
    EVAL_NUM_FRAMES = 16 # How many frames per window
    EVAL_STRIDE = 8      # Sliding window stride

# Instantiate the config for easy import in other scripts
config = Config()

