"""
Defines the PyTorch modules for the X-CLIP model.
Includes the main wrapper and a separate head for linear probing.
"""
import torch
import torch.nn as nn
from transformers import XCLIPProcessor, XCLIPModel

class XCLIPWrapper(nn.Module):
    """
    A clean wrapper around the Hugging Face X-CLIP model.
    This handles the model and processor together and ensures compatibility
    with `torch.nn.DataParallel` for multi-GPU training.
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.model = XCLIPModel.from_pretrained(model_name)
        self.processor = XCLIPProcessor.from_pretrained(model_name)

    def forward(self, **inputs):
        """
        The forward pass returns the diagonal of the logits matrix,
        representing the similarity score for each (video, text) pair in the batch.
        """
        # We only need the logits for video-text similarity
        return self.model(**inputs).logits_per_video

class LinearProbeHead(nn.Module):
    """
    A simple linear layer to be placed on top of frozen X-CLIP features.
    This is used for the "linear-probe" or "few-shot" experiment.
    """
    def __init__(self, embed_dim: int, num_classes: int = 1):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, video_embeds: torch.Tensor):
        """
        Takes pooled video embeddings and returns a classification logit.
        """
        return self.classifier(video_embeds)

