from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def build_vector(chunks):
    vectors = embedder.encode(chunks)
    return vectors