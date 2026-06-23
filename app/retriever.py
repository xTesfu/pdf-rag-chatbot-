from app.embeddings import build_vector
from app.vector_store import get_all_documents, load_chunks, load_index


def retrieve(query, k=5):

    q_vec = build_vector([{"text": query}]).astype("float32")

    results = []

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

            chunk = chunks[idx]

            results.append({
                "text": chunk["text"],
                "document": chunk["document"],
                "page": chunk["page"],
                "score": float(distances[0][i])
            })

    # best first (FAISS IP → higher is better)
    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:k]