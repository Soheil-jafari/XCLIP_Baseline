"""
X-CLIP Baseline Experiment 3: Full Fine-Tuning & Evaluation

This script fine-tunes the entire pre-trained X-CLIP model on your
training data using a contrastive loss. This is the most comprehensive
training method and should yield the best performance.
"""
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

# Local module imports
from xclip_config import config
from xclip_module.data import TripletDataset, VideoWindowDataset, load_triplets_for_eval
from xclip_module.metrics import calculate_and_save_metrics
from xclip_module.utils import setup_output_dir, get_model_and_processor

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """InfoNCE loss for a batch of (video, text) pairs."""
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def main():
    parser = argparse.ArgumentParser(description="X-CLIP Full Fine-Tuning and Evaluation")
    parser.add_argument("--epochs", type=int, default=config.FT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.FT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.FT_LEARNING_RATE)
    parser.add_argument("--tag", type=str, default="finetune", help="A tag for the output directory.")
    args = parser.parse_args()

    # --- Setup ---
    output_dir = setup_output_dir(config.XCLIP_CHECKPOINT_DIR, args.tag)
    device = torch.device(config.DEVICE)
    print(f"[INFO] Starting Full Fine-Tuning. Checkpoints and results will be saved to: {output_dir}")
    print(f"[INFO] Using device: {device}")

    # --- Load Model ---
    model, processor = get_model_and_processor(config.MODEL_NAME, device)

    # --- Load Data ---
    train_dataset = TripletDataset(
        csv_path=config.TRAIN_TRIPLETS_CSV_PATH,
        processor=processor,
        config=config,
        num_frames=config.EVAL_NUM_FRAMES,
        is_contrastive=True 
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
        shuffle=True
    )
    print(f"[INFO] Loaded {len(train_dataset)} training samples for fine-tuning.")

    # --- Training Loop ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    print("[INFO] Starting full model fine-tuning...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            inputs = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits = model(**inputs)
                if isinstance(logits, list): # From DataParallel
                    logits = torch.cat(logits, dim=0)
                loss = contrastive_loss(logits)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} - Average Contrastive Loss: {avg_loss:.4f}")

        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
        torch.save(model_to_save.state_dict(), output_dir / f"finetuned_epoch_{epoch+1}.pt")
    
    print("[INFO] Fine-tuning complete. Final model saved.")

    # --- Evaluation ---
    print("\n[INFO] Starting evaluation with the fine-tuned model...")
    model.eval()

    eval_items = load_triplets_for_eval(config.TEST_TRIPLETS_CSV_PATH)
    eval_dataset = VideoWindowDataset(eval_items, config, processor, config.EVAL_NUM_FRAMES, config.EVAL_STRIDE)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.EVAL_BATCH_SIZE, num_workers=config.NUM_WORKERS)
    
    window_scores_by_item = defaultdict(lambda: {"starts": [], "scores": []})

    for batch in tqdm(eval_dataloader, desc="🚀 Evaluating Windows (Fine-Tuned)"):
        inputs, videos, queries, start_frames = batch
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = model(**inputs)
            if isinstance(logits, list):
                logits = torch.cat(logits, dim=0)
            bscores = logits.diag().detach().cpu().tolist()

        for i in range(len(videos)):
            key = (videos[i], queries[i])
            window_scores_by_item[key]["starts"].append(start_frames[i].item())
            window_scores_by_item[key]["scores"].append(bscores[i])

    calculate_and_save_metrics(
        output_dir=output_dir,
        eval_items=eval_items,
        window_scores_by_item=window_scores_by_item,
        eval_config={"mode": "fine-tune", **vars(args)}
    )
    print(f"\n[SUCCESS] Fine-tuning experiment complete. Results saved in {output_dir}")


if __name__ == "__main__":
    main()

