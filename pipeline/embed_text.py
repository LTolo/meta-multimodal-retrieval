from load_model_wrapper import load_model
import torch

_, text_model = load_model("image_text")

def embed_text(text):
    """
    Returns numpy array embedding for text
    """
    with torch.no_grad():
        return text_model.encode_text(text).cpu().numpy()
