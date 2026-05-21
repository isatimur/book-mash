import pytest

from book_mash.judges.embeddings import EmbeddingClient, pick_embedding_client


def test_pick_returns_a_client(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "stub")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = pick_embedding_client()
    assert isinstance(client, EmbeddingClient)


def test_pick_falls_back_to_openai(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    client = pick_embedding_client()
    assert isinstance(client, EmbeddingClient)


def test_pick_raises_without_either(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No embedding API key"):
        pick_embedding_client()
