"""
Data loading utilities and PyTorch Dataset classes for X-CLIP experiments.
Designed for efficiency to prevent bottlenecks.
"""
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple

def get_frame_paths(base_dir: Path, video_id: str, glob_pattern: str) -> List[Path]:
    """Finds and sorts frame paths for a given video ID."""
    video_dir = base_dir / str(video_id)
    if not video_dir.is_dir():
        return []
    return sorted(list(video_dir.glob(glob_pattern)))

def load_triplets_for_eval(csv_path: str) -> Dict:
    """
    Loads triplets from a CSV and groups them by (video, query) for evaluation.
    This structure is needed to calculate metrics correctly.
    """
    df = pd.read_csv(csv_path)
    items = defaultdict(lambda: {"gts": [], "fps": None})
    
    video_col, text_col = ('video_id', 'text') if 'video_id' in df.columns else ('video', 'query')
    sf_col, ef_col = 'start_frame', 'end_frame'

    for _, row in df.iterrows():
        key = (str(row[video_col]), row[text_col])
        items[key]["gts"].append((row[sf_col], row[ef_col]))
        if items[key]["fps"] is None:
            items[key]["fps"] = row.get('fps', 30)
    return dict(items)


class TripletDataset(Dataset):
    """
    A PyTorch Dataset for training. It reads the triplets CSV.
    - For contrastive training, it yields positive (video, text) pairs.
    - For classification (linear probe), it yields (video, label) pairs.
    """
    def __init__(self, csv_path, processor, config, num_frames, is_contrastive=False):
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.config = config
        self.num_frames = num_frames
        self.is_contrastive = is_contrastive
        self.base_dir = Path(config.EXTRACTED_FRAMES_DIR)
        
        video_col = 'video_id' if 'video_id' in self.df.columns else 'video'
        self.df['video_path_exists'] = self.df[video_col].apply(lambda x: (self.base_dir / str(x)).exists())
        self.df = self.df[self.df['video_path_exists']].reset_index(drop=True)
        self.video_col = video_col
        self.text_col = 'text' if 'text' in self.df.columns else 'query'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id, text = row[self.video_col], row[self.text_col]
        start_frame, end_frame = row['start_frame'], row['end_frame']

        frame_paths = get_frame_paths(self.base_dir, video_id, self.config.FRAME_GLOB)
        if not frame_paths: return self.__getitem__((idx + 1) % len(self))
        
        indices = torch.linspace(start_frame, end_frame - 1, self.num_frames).long()
        indices = torch.clamp(indices, 0, len(frame_paths) - 1)
        
        clip_paths = [frame_paths[i] for i in indices]
        try:
            clip_images = [Image.open(p).convert("RGB") for p in clip_paths]
        except (IOError, FileNotFoundError, UnidentifiedImageError):
            return self.__getitem__((idx + 1) % len(self))

        if self.is_contrastive:
            inputs = self.processor(text=[text], videos=[clip_images], return_tensors="pt", padding=True)
            return {k: v.squeeze(0) for k, v in inputs.items()}
        else:
            inputs = self.processor(videos=[clip_images], return_tensors="pt")
            label = torch.tensor(1.0)
            return ({k: v.squeeze(0) for k, v in inputs.items() if k != 'text'}, label)


class VideoWindowDataset(Dataset):
    """
    Efficient Dataset for evaluation. Pre-calculates all sliding windows.
    """
    def __init__(self, items, config, processor, num_frames, stride):
        self.windows = []
        base_dir = Path(config.EXTRACTED_FRAMES_DIR)

        for (video, query), bundle in items.items():
            frame_paths = get_frame_paths(base_dir, video, config.FRAME_GLOB)
            if not frame_paths: continue

            n_frames = len(frame_paths)
            window_starts = list(range(0, n_frames - num_frames + 1, stride))
            if not window_starts and n_frames > 0: window_starts = [0]

            for start in window_starts:
                self.windows.append({
                    "video": video, "query": query, "start": start,
                    "frame_paths": frame_paths, "num_frames": n_frames
                })
        
        self.processor = processor
        self.num_frames_in_window = num_frames
        
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        win = self.windows[idx]
        start, end = win['start'], win['start'] + self.num_frames_in_window
        clip_paths = win['frame_paths'][start:end]

        try:
            images = [Image.open(p).convert("RGB") for p in clip_paths]
            if len(images) < self.num_frames_in_window:
                images.extend([images[-1]] * (self.num_frames_in_window - len(images)))
        except (IOError, FileNotFoundError, UnidentifiedImageError):
            images = [Image.new('RGB', (224, 224))] * self.num_frames_in_window

        inputs = self.processor(text=[win['query']], videos=[images], return_tensors="pt", padding=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        return inputs, win['video'], win['query'], win['start']

