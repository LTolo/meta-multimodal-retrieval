import streamlit as st
import numpy as np
from pipeline.embed_text import embed_text
from retrieval.faiss_index import FaissIndex

st.title("Meta Multimodal Retrieval Demo")

query = st.text_input("Enter your text query:")

# Dummy embeddings for demo
dummy_embeddings = np.random.rand(10, 512).astype('float32')
dummy_ids = [f"video_{i}.mp4" for i in range(10)]
index = FaissIndex(dim=512)
index.add(dummy_embeddings, dummy_ids)

if query:
    query_emb = embed_text(query).reshape(1, -1).astype('float32')
    results, distances = index.search(query_emb, k=5)
    st.write("Top matching videos/images:")
    for res, dist in zip(results, distances[0]):
        st.write(f"{res} (score: {dist:.3f})")
