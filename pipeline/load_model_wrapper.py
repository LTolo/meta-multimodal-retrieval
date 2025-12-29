import sys
import os

# Sys.path Option A
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../meta_models/core")))

from vision_encoder import VisionEncoder
from transformer import TransformerModel

def load_model(model_type="image_text"):
    """
    Safe wrapper for Meta perception_models
    Returns vision_model, text_model
    """
    if model_type == "image_text":
        vision_model = VisionEncoder(pretrained=True)
        vision_model.eval()
        text_model = TransformerModel(pretrained=True)
        text_model.eval()
        return vision_model, text_model
    else:
        raise ValueError(f"Unknown model_type {model_type}")
