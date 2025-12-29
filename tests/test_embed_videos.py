
from pipeline.embed_videos import embed_video

video_path = "tests/sample.mp4"  # kleines Testvideo (1-2 Sekunden)

video_meta_emb, video_mask_emb = embed_video(video_path, fps=1)

print("Video meta embedding shape:", video_meta_emb.shape)
print("Video mask embedding shape:", video_mask_emb.shape)
