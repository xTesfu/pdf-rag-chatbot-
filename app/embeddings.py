from sentence_transformers import SentenceTransformer

# embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')


def build_vector(chunks):
    # vectors = embedder.encode(chunks)
    vectors = embedder.encode(chunks, normalize_embeddings=True)
    return vectors