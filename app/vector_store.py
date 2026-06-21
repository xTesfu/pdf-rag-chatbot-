import faiss
import numpy as np

def build_index(vectors):
    dim = len(vectors[0])

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))

    return index, vectors