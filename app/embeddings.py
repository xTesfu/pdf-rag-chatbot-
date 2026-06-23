from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")


def build_vector(chunks):
    texts = [c["text"] for c in chunks]

    vectors = embedder.encode(
        texts,
        normalize_embeddings=True
    )

    return vectors