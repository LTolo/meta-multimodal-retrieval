import torch
from load_model_wrapper import load_model
from sam3 import build_sam_model

# Modelle laden
vision_model, text_model = load_model("image_text")

# SAM-3 Segmentation
sam_model = build_sam_model("vit_h")
sam_model.eval()

def embed_image(image):
    """
    Returns tuple: (Meta Vision embedding, SAM-3 mask embedding)
    """
    with torch.no_grad():
        meta_emb = vision_model.encode(image).cpu().numpy()
        mask_emb = sam_model.generate_mask_embedding(image).cpu().numpy()
    return meta_emb, mask_emb
