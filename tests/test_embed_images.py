import sys
import os

# --- FORCE project root into sys.path ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- now imports ALWAYS work ---
from pipeline.embed_images import embed_image
import cv2

image = cv2.imread("tests/sample.jpg")
meta_emb, mask_emb = embed_image(image)

print("Meta embedding shape:", meta_emb.shape)
print("SAM embedding shape:", mask_emb.shape)
