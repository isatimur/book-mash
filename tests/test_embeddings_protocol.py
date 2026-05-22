from book_mash.judges.embeddings import EmbeddingClient, pick_embedding_client


def test_pick_returns_openai_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    client = pick_embedding_client()
    assert isinstance(client, EmbeddingClient)


def test_pick_raises_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    import pytest
    with pytest.raises(RuntimeError, match="No embedding API key"):
        pick_embedding_client()
