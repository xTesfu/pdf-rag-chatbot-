import os
import faiss
import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv() 

api_key = os.getenv("QWEN_API_KEY")
model_name = os.getenv("MODEL_NAME")

# =========================
# 1. CONFIG (Ollama + Qwen)
# =========================
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key=api_key
)

MODEL = model_name

# =========================
# 2. EMBEDDING MODEL
# =========================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# # =========================
# # 3. PDF LOADING
# # =========================
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text

# =========================
# 4. CHUNKING
# =========================
def chunk_text(text, chunk_size=400):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

# =========================
# 5. EMBEDDINGS + FAISS
# =========================
def build_index(chunks):
    vectors = embedder.encode(chunks)
    dim = len(vectors[0])

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))

    return index, vectors

# =========================
# 6. RETRIEVAL
# =========================
def retrieve(query, index, chunks, k=3):
    q_vec = embedder.encode([query]).astype("float32")
    _, ids = index.search(q_vec, k)
    return [chunks[i] for i in ids[0]]

# =========================
# 7. LLM ANSWER (QWEN)
# =========================
def ask_llm(context, question):
    prompt = f"""
Use the context to answer the question.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

# =========================
# 8. MAIN RAG PIPELINE
# =========================
def main(pdf_path):
    print("Loading PDF...")
    text = load_pdf(pdf_path)

    print("Chunking text...")
    chunks = chunk_text(text)

    print("Building vector index...")
    index, _ = build_index(chunks)

    print("\n✅ Ready! Ask questions about your PDF.\n")

    while True:
        query = input("Ask: ")

        if query.lower() in ["exit", "quit"]:
            break

        context = retrieve(query, index, chunks)
        context_text = "\n\n".join(context)

        answer = ask_llm(context_text, query)

        print("\n--- Answer ---")
        print(answer)
        print("\n")

# =========================
# 9. RUN
# =========================
if __name__ == "__main__":
    main("your_file.pdf")
    #ollama run qwen2.5:7b