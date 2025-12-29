import cv2
import numpy as np
from pipeline.embed_images import embed_image

# Testbild laden
image_path = "tests/sample.jpg"  # lege ein kleines Testbild hier ab
image = cv2.imread(image_path)

meta_emb, mask_emb = embed_image(image)

print("Meta embedding shape:", meta_emb.shape)
print("SAM-3 mask embedding shape:", mask_emb.shape)
