import faiss
import numpy as np

class FaissIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
    def add(self, vectors):
        self.index.add(vectors)
    def search(self, query_vector, k=5):
        return self.index.search(query_vector, k)
