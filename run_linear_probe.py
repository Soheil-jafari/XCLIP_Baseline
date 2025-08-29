"""
X-CLIP Baseline Experiment 2: Linear-Probe (Few-Shot) Training & Evaluation

This script freezes the entire pre-trained X-CLIP model and trains only a new,
small linear layer on top of the features. This is a very fast and efficient
way to adapt the model to your specific dataset.
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

# Local module imports
from xclip_config import config
from xclip_module.model import LinearProbeHead
from xclip_module.data import TripletDataset, VideoWindowDataset, load_triplets_for_eval
from xclip_module.metrics import calculate_and_save_metrics
from xclip_module.utils import setup_output_dir, get_model_and_processor

def main():
    parser = argparse.ArgumentParser(description="X-CLIP Linear-Probe Training and Evaluation")
    parser.add_argument("--epochs", type=int, default=config.LP_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.LP_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LP_LEARNING_RATE)
    parser.add_argument("--tag", type=str, default="linear_probe", help="A tag for the output directory.")
    args = parser.parse_args()

    # --- Setup ---
    output_dir = setup_output_dir(config.XCLIP_CHECKPOINT_DIR, args.tag)
    device = torch.device(config.DEVICE)
    print(f"[INFO] Starting Linear-Probe experiment. Checkpoints and results will be saved to: {output_dir}")
    print(f"[INFO] Using device: {device}")

    # --- Load Model & Freeze Backbone ---
    base_model, processor = get_model_and_processor(config.MODEL_NAME, device)
    
    # IMPORTANT: Freeze all parameters of the base model
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()

    # Create the new linear probe head that we will train
    # Get embed_dim from the model config, handling DataParallel wrapper
    model_config = base_model.module.model.config if isinstance(base_model, nn.DataParallel) else base_model.model.config
    probe_head = LinearProbeHead(embed_dim=model_config.vision_config.hidden_size).to(device)
    
    # --- Load Data ---
    train_dataset = TripletDataset(
        csv_path=config.TRAIN_TRIPLETS_CSV_PATH,
        processor=processor,
        config=config,
        num_frames=config.EVAL_NUM_FRAMES
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=config.NUM_WORKERS,
        shuffle=True
    )
    print(f"[INFO] Loaded {len(train_dataset)} training triplets.")

    # --- Training Loop ---
    optimizer = optim.AdamW(probe_head.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss() 

    print("[INFO] Starting training of the linear probe head...")
    for epoch in range(args.epochs):
        probe_head.train()
        total_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            
            with torch.no_grad():
                # Extract video features directly from the underlying transformer model
                video_features = base_model.module.model.vision_model(**inputs)[1]
                video_embeds = base_model.module.model.video_projection(video_features)
            
            logits = probe_head(video_embeds)
            
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} - Average Loss: {avg_loss:.4f}")

        torch.save(probe_head.state_dict(), output_dir / f"linear_probe_epoch_{epoch+1}.pt")
    
    print("[INFO] Training complete. Final checkpoint saved.")
    
    # --- Evaluation ---
    print("\n[INFO] Starting evaluation with the trained linear probe...")
    probe_head.eval()

    eval_items = load_triplets_for_eval(config.TEST_TRIPLETS_CSV_PATH)
    eval_dataset = VideoWindowDataset(eval_items, config, processor, config.EVAL_NUM_FRAMES, config.EVAL_STRIDE)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.EVAL_BATCH_SIZE, num_workers=config.NUM_WORKERS)
    
    window_scores_by_item = defaultdict(lambda: {"starts": [], "scores": []})

    for batch in tqdm(eval_dataloader, desc="🚀 Evaluating Windows (Linear Probe)"):
        inputs, videos, queries, start_frames = batch
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        with torch.no_grad():
            video_features = base_model.module.model.vision_model(**inputs)[1]
            video_embeds = base_model.module.model.video_projection(video_features)
            logits = probe_head(video_embeds)
            scores = torch.sigmoid(logits).squeeze().cpu().tolist()

        if not isinstance(scores, list): scores = [scores]

        for i in range(len(videos)):
            key = (videos[i], queries[i])
            window_scores_by_item[key]["starts"].append(start_frames[i].item())
            window_scores_by_item[key]["scores"].append(scores[i])
    
    calculate_and_save_metrics(
        output_dir=output_dir,
        eval_items=eval_items,
        window_scores_by_item=window_scores_by_item,
        eval_config={"mode": "linear-probe", **vars(args)}
    )
    print(f"\n[SUCCESS] Linear-Probe experiment complete. Results saved in {output_dir}")

if __name__ == "__main__":
    main()

