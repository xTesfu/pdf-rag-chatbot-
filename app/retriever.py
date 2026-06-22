# from app.embeddings import build_vector

# def retrieve(query, index, chunks, k=3):
#     q_vec = build_vector([query]).astype("float32")
#     _, ids = index.search(q_vec, k)
#     return [chunks[i] for i in ids[0]]


from app.embeddings import build_vector
from app.vector_store import get_all_documents, load_chunks, load_index

# import numpy as np


def retrieve(query, k=3):
    """
    Search across ALL PDFs and return top-k most relevant chunks.
    """

    q_vec = build_vector([query]).astype("float32")

    all_results = []

    doc_ids = get_all_documents()

    for doc_id in doc_ids:

        index = load_index(doc_id)
        chunks = load_chunks(doc_id)

        if index is None or chunks is None:
            continue

        distances, indices = index.search(q_vec, k)

        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            all_results.append({
                "text": chunks[idx],
                "score": float(distances[0][i]),
                "doc_id": doc_id
            })

    # sort by best score (FAISS L2 → lower is better)
    all_results.sort(key=lambda x: x["score"])

    return [r["text"] for r in all_results[:k]]