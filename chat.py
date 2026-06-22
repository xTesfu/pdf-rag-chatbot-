import streamlit as st
import tempfile
import os

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
    load_chunks,
    clear_cache
)

st.set_page_config(
    page_title="PDF-RAG Chat",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Chat with your PDF")
st.markdown("Upload a PDF and ask questions about it.")

# ---------------------------------------------------
# Session State
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None


# ---------------------------------------------------
# Upload PDF
# ---------------------------------------------------
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    if st.session_state.index is None:

        with st.spinner("Processing PDF..."):

            # Save temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:
                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name

            # Load PDF
            text = load_pdf(pdf_path)

            # Chunk text
            chunks = chunk_text(text)

            index = load_index()
            saved_chunks = load_chunks()

            if index is None or saved_chunks is None:

                vectors = build_vector(chunks)
                index = build_index(vectors)

                save_index(index)
                save_chunks(chunks)

                st.session_state.index = index
                st.session_state.chunks = chunks

            else:
                st.session_state.index = index
                st.session_state.chunks = saved_chunks

            os.remove(pdf_path)

        st.sidebar.success("PDF processed successfully!")

# ---------------------------------------------------
# Display chat history
# ---------------------------------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------
# Chat input
# ---------------------------------------------------
if prompt := st.chat_input("Ask something about the PDF..."):

    if st.session_state.index is None:
        st.warning("Please upload a PDF first.")
        st.stop()

    # Display user message
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Retrieve context
    context = retrieve(
        prompt,
        st.session_state.index,
        st.session_state.chunks
    )

    context_text = "\n\n".join(context)

    # Ask LLM
    with st.spinner("Thinking..."):
        answer = ask_llm(
            context_text,
            prompt
        )

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

# with st.expander("Retrieved Context"):
#     st.write(context_text)


# if st.sidebar.button("Reset PDF"):
#     st.session_state.index = None
#     st.session_state.chunks = None
#     st.session_state.messages = []

if st.sidebar.button("Reset PDF"):
    clear_cache()

    st.session_state.index = None
    st.session_state.chunks = None
    st.session_state.messages = []


# with st.sidebar:
#     st.header("Settings")
#     st.write(
#         f"Chunks loaded: "
#         f"{len(st.session_state.chunks) if st.session_state.chunks else 0}"
#     )