from meta_models.models import load_model as load_meta_model
from sam3 import build_sam_model

# Meta perception model
meta_model = load_meta_model("image_text")
meta_model.eval()

# SAM-3 (Segmentation)
sam_model = build_sam_model("vit_h")
sam_model.eval()

def embed_image(image):
    meta_embedding = meta_model.encode_image(image)
    # Optional: SAM Mask Embedding
    mask_embedding = sam_model.generate_mask_embedding(image)
    return meta_embedding, mask_embedding
