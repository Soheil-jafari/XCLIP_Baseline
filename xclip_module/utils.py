"""
General utility functions for the X-CLIP experiments.
"""
import torch
import random
import numpy as np
from pathlib import Path
import datetime

from .model import XCLIPWrapper

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_output_dir(base_path: str, tag: str) -> Path:
    """Creates a unique timestamped output directory for a run."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_path) / f"{tag}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_model_and_processor(model_name, device):
    """Loads the model and wraps it for multi-GPU if available."""
    wrapper = XCLIPWrapper(model_name)
    processor = wrapper.processor
    
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = torch.nn.DataParallel(wrapper)
    else:
        model = wrapper
        
    model.to(device)
    return model, processor

