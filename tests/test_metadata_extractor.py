import requests
from modules.metadata_extractor import MetadataExtractor


class DummyResponse:
    def __init__(self, data):
        self._data = data
    def raise_for_status(self):
        pass
    def json(self):
        return self._data


def test_extract_metadata_success(monkeypatch):
    def mock_post(url, json, timeout):
        return DummyResponse({"response": '{"title":"My Title","authors":"A","publishers":"P","year":"2020","doi":"123"}'})
    monkeypatch.setattr(requests, "post", mock_post)
    extractor = MetadataExtractor()
    meta = extractor.extract_metadata_with_ollama("sample", "sample.pdf", max_retries=0)
    assert meta["title"] == "My Title"


def test_extract_metadata_fallback(monkeypatch):
    def mock_post(url, json, timeout):
        return DummyResponse({"response": 'not json'})
    monkeypatch.setattr(requests, "post", mock_post)
    extractor = MetadataExtractor()
    meta = extractor.extract_metadata_with_ollama("sample", "Title -- Author -- 2021 -- Pub.pdf", max_retries=0)
    assert meta["publishers"] == "Pub"
    assert meta["year"] == "2021"

