"""
X-CLIP Baseline Experiment 1: Zero-Shot Evaluation

This script loads the pre-trained X-CLIP model directly from Hugging Face
and evaluates its performance on your test set without any training.
This establishes the "out-of-the-box" baseline for your paper.
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local module imports
from xclip_config import config
from xclip_module.data import VideoWindowDataset, load_triplets_for_eval
from xclip_module.metrics import calculate_and_save_metrics
from xclip_module.utils import setup_output_dir, get_model_and_processor

def main():
    parser = argparse.ArgumentParser(description="X-CLIP Zero-Shot Evaluation")
    parser.add_argument("--tag", type=str, default="zeroshot", help="A tag for the output directory.")
    args = parser.parse_args()

    # --- Setup ---
    output_dir = setup_output_dir(config.XCLIP_OUTPUT_DIR, args.tag)
    device = torch.device(config.DEVICE)
    print(f"[INFO] Starting Zero-Shot Evaluation. Results will be saved to: {output_dir}")
    print(f"[INFO] Using device: {device}")

    # --- Load Model ---
    # We load the official pre-trained model from Hugging Face.
    model, processor = get_model_and_processor(config.MODEL_NAME, device)
    model.eval()

    # --- Load Data ---
    # This function reads your test CSV and prepares it for evaluation.
    eval_items = load_triplets_for_eval(config.TEST_TRIPLETS_CSV_PATH)
    
    dataset = VideoWindowDataset(
        items=eval_items,
        config=config,
        processor=processor,
        num_frames=config.EVAL_NUM_FRAMES,
        stride=config.EVAL_STRIDE
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.EVAL_BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"[INFO] Loaded {len(eval_items)} video-query pairs from {config.TEST_TRIPLETS_CSV_PATH}")
    print(f"[INFO] Created dataset with {len(dataset)} windows to evaluate.")

    # --- Run Inference ---
    window_scores_by_item = defaultdict(lambda: {"starts": [], "scores": []})
    amp_dtype = torch.float16 if "cuda" in device.type else None

    for batch in tqdm(dataloader, desc="🚀 Evaluating Windows (Zero-Shot)"):
        inputs, videos, queries, start_frames = batch
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(amp_dtype is not None), dtype=amp_dtype):
            logits = model(**inputs)
            # For DataParallel, output is a list of tensors that needs gathering
            if isinstance(logits, list):
                logits = torch.cat(logits, dim=0)
            bscores = logits.diag().detach().cpu().tolist()

        for i in range(len(videos)):
            key = (videos[i], queries[i])
            window_scores_by_item[key]["starts"].append(start_frames[i].item())
            window_scores_by_item[key]["scores"].append(bscores[i])

    # --- Calculate and Save Metrics ---
    calculate_and_save_metrics(
        output_dir=output_dir,
        eval_items=eval_items,
        window_scores_by_item=window_scores_by_item,
        eval_config={
            "mode": "zero-shot",
            "model_name": config.MODEL_NAME,
            "num_frames": config.EVAL_NUM_FRAMES,
            "stride": config.EVAL_STRIDE,
        }
    )
    print(f"\n[SUCCESS] Zero-Shot evaluation complete. Metrics saved in {output_dir}")


if __name__ == "__main__":
    main()

