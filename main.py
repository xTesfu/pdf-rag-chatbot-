from app.chunker import chunk_text
from app.pdf_loader import load_pdf
from app.embeddings import build_vector
from app.retriever import retrieve
from app.llm import ask_llm
from app.vector_store import (
    build_index,
    save_index,
    load_index,
    save_chunks,
    load_chunks
)

def main(pdf_path):
    print("Loading PDF...")
    text = load_pdf(pdf_path)

    print("Chunking text...")
    chunks = chunk_text(text)

    index = load_index()
    saved_chunks = load_chunks()

    if index is None or saved_chunks is None:

        print("Building vectors...")
        vectors = build_vector(chunks)

        print("Building vector index...")
        index = build_index(vectors)

        save_index(index)
        save_chunks(chunks)

    else:
        chunks = saved_chunks
        print("Loaded existing FAISS index.")

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
    main("data/your_file.pdf")
    #ollama run qwen2.5:7b