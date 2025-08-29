"""
Evaluation metrics for video-language tasks, including AUROC, AUPRC.
"""
import numpy as np
import json
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
from typing import Dict, List
import pandas as pd

def assign_window_scores_to_frames(n_frames, window_starts, window_scores, window_len):
    scores = np.zeros(n_frames, dtype=float)
    counts = np.zeros(n_frames, dtype=float)
    for s, sc in zip(window_starts, window_scores):
        e = min(n_frames, s + window_len)
        scores[s:e] += sc
        counts[s:e] += 1.0
    counts[counts == 0] = 1.0
    return scores / counts

def calculate_and_save_metrics(output_dir: Path, eval_items: Dict, window_scores_by_item: Dict, eval_config: Dict):
    """
    Aggregates window scores to frame scores and computes all required metrics.
    Saves results to JSON and TXT files.
    """
    frame_level_metrics = []

    for (video, query), bundle in eval_items.items():
        key = (video, query)
        if key not in window_scores_by_item: continue

        results = window_scores_by_item[key]
        if not results['starts']: continue
        
        # Estimate n_frames from the last window start + stride
        n_frames = max(s for s in results['starts']) + eval_config['num_frames']
        
        frame_scores = assign_window_scores_to_frames(
            n_frames, results["starts"], results["scores"], eval_config["num_frames"]
        )
        
        y_true = np.zeros(n_frames, dtype=int)
        for (s, e) in bundle["gts"]:
            y_true[max(0, int(s)) : min(n_frames, int(e))] = 1
        
        # Guard against cases with no positive labels, which crashes metrics
        if np.unique(y_true).size < 2: continue

        try:
            auroc = roc_auc_score(y_true, frame_scores)
            ap = average_precision_score(y_true, frame_scores)
            frame_level_metrics.append({"video": video, "query": query, "auroc": auroc, "ap": ap})
        except ValueError as e:
            print(f"Skipping metrics for {key} due to error: {e}")

    macro_auroc = np.mean([m["auroc"] for m in frame_level_metrics]) if frame_level_metrics else 0.0
    macro_ap = np.mean([m["ap"] for m in frame_level_metrics]) if frame_level_metrics else 0.0

    metrics = {
        "frame_level": {"macro_AUROC": macro_auroc, "macro_AP": macro_ap},
        "config": {k: v for k, v in eval_config.items() if isinstance(v, (str, int, float, bool))},
        "per_item_metrics": frame_level_metrics
    }

    # Save files
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    summary_config = json.dumps(metrics['config'], indent=2)
    summary = (
        f"Evaluation Summary ({eval_config.get('mode', 'N/A')})\n"
        f"==================================\n"
        f"Macro AUROC: {macro_auroc:.4f}\n"
        f"Macro AP (AUPRC): {macro_ap:.4f}\n"
        f"==================================\n"
        f"Configuration:\n{summary_config}"
    )
    with open(output_dir / "metrics.txt", "w") as f:
        f.write(summary)
    
    print("\n" + summary)

