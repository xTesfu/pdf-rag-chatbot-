from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(pages, document_name=None):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    all_chunks = []

    for page in pages:

        chunks = splitter.split_text(page["text"])

        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "document": document_name,
                "page": page["page"]
            })

    return all_chunks