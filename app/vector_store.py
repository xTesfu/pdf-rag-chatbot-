import faiss
import numpy as np
import pickle
import os

INDEX_PATH = "data/faiss_index.bin"
CHUNKS_PATH = "data/chunks.pkl"


def build_index(vectors):
    dim = vectors.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))

    return index


def save_index(index):
    faiss.write_index(index, INDEX_PATH)


def load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)

    return None


def save_chunks(chunks):
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)


def load_chunks():
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "rb") as f:
            return pickle.load(f)

    return None


def clear_cache():
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)

    if os.path.exists(CHUNKS_PATH):
        os.remove(CHUNKS_PATH)