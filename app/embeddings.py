from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')


def build_vector(chunks):
    vectors = embedder.encode(chunks, normalize_embeddings=True)
    return vectors