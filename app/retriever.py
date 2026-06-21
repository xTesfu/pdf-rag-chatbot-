from app.embeddings import build_vector

def retrieve(query, index, chunks, k=3):
    q_vec = build_vector([query]).astype("float32")
    _, ids = index.search(q_vec, k)
    return [chunks[i] for i in ids[0]]