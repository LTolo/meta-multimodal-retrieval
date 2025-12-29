from pipeline.embed_text import embed_text

text = "This is a test sentence."

text_emb = embed_text(text)
print("Text embedding shape:", text_emb.shape)
