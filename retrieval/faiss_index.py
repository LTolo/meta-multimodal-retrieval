import faiss
import numpy as np

class FaissIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.ids = []

    def add(self, vectors, ids=None):
        self.index.add(vectors)
        if ids:
            self.ids.extend(ids)

    def search(self, query_vector, k=5):
        distances, indices = self.index.search(query_vector, k)
        if self.ids:
            results = [self.ids[i] for i in indices[0]]
        else:
            results = indices[0].tolist()
        return results, distances
