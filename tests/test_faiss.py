import numpy as np
from retrieval.faiss_index import FaissIndex

# Dummy Vektoren
vectors = np.random.rand(5, 512).astype("float32")
ids = [f"id_{i}" for i in range(5)]

index = FaissIndex(dim=512)
index.add(vectors, ids)

query = np.random.rand(1, 512).astype("float32")
results, distances = index.search(query, k=3)

print("Top results:", results)
print("Distances:", distances)
