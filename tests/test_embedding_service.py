import requests
from modules.embedding_service import EmbeddingService


class DummyResponse:
    def __init__(self, data, status=200):
        self._data = data
        self.status_code = status
    def raise_for_status(self):
        pass
    def json(self):
        return self._data


def test_embed_updates_stats(monkeypatch):
    def mock_post(url, json=None, timeout=None):
        return DummyResponse({"embedding": [0.1] * 768})
    monkeypatch.setattr(requests, "post", mock_post)
    service = EmbeddingService(batch_size=2, show_progress=False)
    embeddings = service.embed(["a", "b", "c"])
    assert len(embeddings) == 3
    stats = service.get_statistics()
    assert stats["total_requests"] == 3
    assert stats["successful_requests"] == 3
    assert stats["failed_requests"] == 0


def test_health_check(monkeypatch):
    def mock_get(url, timeout=None):
        return DummyResponse({"models": [{"name": "nomic-embed-text"}]})
    def mock_post(url, json=None, timeout=None):
        return DummyResponse({"embedding": [0.0] * 768})
    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.setattr(requests, "post", mock_post)
    service = EmbeddingService(show_progress=False)
    health = service.health_check()
    assert health["service_available"] is True
    assert health["model_loaded"] is True
    assert health["embedding_test"] is True
