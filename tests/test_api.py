from fastapi.testclient import TestClient

from api import app


def test_upload_pdfs_builds_new_index(monkeypatch):
    client = TestClient(app)

    # Fake PDF loader
    monkeypatch.setattr(
        "api.load_pdf",
        lambda path: [
            {"page": 1, "text": "Hello world"}
        ],
    )

    monkeypatch.setattr(
        "api.chunk_text",
        lambda pages: [
            {"text": "Hello world", "page": 1, "document": "file.pdf"}
        ],
    )

    monkeypatch.setattr(
        "api.build_vector",
        lambda chunks: [[0.1, 0.2]],
    )

    class FakeIndex:
        pass

    monkeypatch.setattr(
        "api.build_index",
        lambda vectors: FakeIndex(),
    )

    monkeypatch.setattr(
        "api.load_index",
        lambda doc_id: None,
    )

    monkeypatch.setattr(
        "api.load_chunks",
        lambda doc_id: None,
    )

    monkeypatch.setattr(
        "api.save_index",
        lambda index, doc_id: None,
    )

    monkeypatch.setattr(
        "api.save_chunks",
        lambda chunks, doc_id: None,
    )

    # Fake file upload
    files = [
        ("files", ("test.pdf", b"fake-pdf-content", "application/pdf"))
    ]

    response = client.post("/upload", files=files)

    assert response.status_code == 200
    data = response.json()

    assert data["message"] == "PDF(s) uploaded successfully"
    assert data["count"] == 1
    assert len(data["documents"]) == 1

def test_upload_pdfs_uses_cache(monkeypatch):
    client = TestClient(app)

    monkeypatch.setattr(
        "api.load_index",
        lambda doc_id: object(),  # cache exists
    )

    monkeypatch.setattr(
        "api.load_chunks",
        lambda doc_id: [{"text": "cached"}],
    )

    monkeypatch.setattr("api.load_pdf", lambda path: None)
    monkeypatch.setattr("api.chunk_text", lambda x: None)
    monkeypatch.setattr("api.build_vector", lambda x: None)
    monkeypatch.setattr("api.build_index", lambda x: None)

    files = [
        ("files", ("test.pdf", b"fake", "application/pdf"))
    ]

    response = client.post("/upload", files=files)

    assert response.status_code == 200

def test_ask_endpoint_returns_answer(monkeypatch):
    client = TestClient(app)

    monkeypatch.setattr(
        "api.retrieve",
        lambda q: [
            {
                "text": "Python is a language",
                "document": "file1.pdf",
                "page": 1,
                "score": 0.9,
            }
        ],
    )

    captured = {}

    def fake_ask_llm(context, question):
        captured["context"] = context
        captured["question"] = question
        return "Fake Answer"

    monkeypatch.setattr(
        "api.ask_llm",
        fake_ask_llm,
    )

    response = client.post(
        "/ask",
        json={"question": "What is Python?"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "What is Python?"
    assert data["answer"] == "Fake Answer"
    assert data["sources_found"] == 1

    # IMPORTANT: verify structured context format
    assert "file1.pdf" in captured["context"]
    assert "page 1" in captured["context"]
    assert "Python is a language" in captured["context"]

def test_documents_endpoint(monkeypatch):
    client = TestClient(app)

    monkeypatch.setattr(
        "api.get_all_documents",
        lambda: ["doc1", "doc2"],
    )

    response = client.get("/documents")

    assert response.status_code == 200

    data = response.json()

    assert data["documents"] == ["doc1", "doc2"]
    assert data["count"] == 2