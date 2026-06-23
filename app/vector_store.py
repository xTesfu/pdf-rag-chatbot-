import hashlib
import pickle
from pathlib import Path

import faiss
import numpy as np

DATA_DIR = Path("data")

def get_doc_id(pdf_bytes):
    return hashlib.md5(pdf_bytes).hexdigest()


def get_doc_path(doc_id):
    path = DATA_DIR / doc_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_index(vectors):
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(np.array(vectors).astype("float32"))

    return index


def save_index(index, doc_id):
    path = get_doc_path(doc_id)
    faiss.write_index(
        index,
        str(path / "index.bin")
    )


def load_index(doc_id):
    path = get_doc_path(doc_id)
    index_file = path / "index.bin"

    if index_file.exists():
        return faiss.read_index(str(index_file))

    return None


def save_chunks(chunks, doc_id):
    path = get_doc_path(doc_id)

    with open(path / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


def load_chunks(doc_id):
    path = get_doc_path(doc_id)
    chunk_file = path / "chunks.pkl"

    if chunk_file.exists():
        with open(chunk_file, "rb") as f:
            return pickle.load(f)

    return None


def clear_document(doc_id):
    path = get_doc_path(doc_id)

    for file in path.iterdir():
        file.unlink()

    path.rmdir()


def get_all_documents():
    if not DATA_DIR.exists():
        return []

    return [
        folder.name
        for folder in DATA_DIR.iterdir()
        if folder.is_dir()
    ]