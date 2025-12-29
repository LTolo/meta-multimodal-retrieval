import numpy as np
from extract_frames import extract_frames
from embed_images import embed_image

def embed_video(video_path, fps=1):
    """
    Returns tuple: (Meta video embedding, SAM-3 mask embedding)
    """
    frames = extract_frames(video_path, fps=fps)
    meta_embeddings = []
    mask_embeddings = []

    for frame in frames:
        meta_emb, mask_emb = embed_image(frame)
        meta_embeddings.append(meta_emb)
        mask_embeddings.append(mask_emb)

    video_meta_emb = np.mean(np.stack(meta_embeddings), axis=0)
    video_mask_emb = np.mean(np.stack(mask_embeddings), axis=0)

    return video_meta_emb, video_mask_emb
