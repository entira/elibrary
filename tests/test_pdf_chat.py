import sys
import json
import types
from pathlib import Path
from pdf_chat import MultiLibraryRetriever, PDFLibraryChat


class DummyRetriever:
    def __init__(self, video_file, index_file):
        self.video_file = video_file
        self.index_file = index_file
    def search(self, query, top_k=5):
        return ["A", "B"]


def test_multi_library_search_and_citations(monkeypatch, tmp_path):
    # create two libraries
    libs = []
    for i, text in enumerate(["A", "B"], 1):
        lib_dir = tmp_path / str(i)
        lib_dir.mkdir()
        index_path = lib_dir / "library_index.json"
        data = {"metadata": [{"text": text, "enhanced_metadata": {"title": f"T{i}", "page_reference": str(i)}}]}
        index_path.write_text(json.dumps(data))
        video_path = lib_dir / "library.mp4"
        video_path.write_text("vid")
        libs.append({"library_id": str(i), "name": f"Library {i}", "video_file": str(video_path), "index_file": str(index_path), "chunks": 1, "files": 1})

    dummy_mod = types.SimpleNamespace(MemvidRetriever=DummyRetriever)
    monkeypatch.setitem(sys.modules, "memvid", dummy_mod)

    retriever = MultiLibraryRetriever(libs)
    results = retriever.search("query", top_k=2)
    assert len(results) == 2
    assert results[0]["library_id"] == "1"

    chat_stub = PDFLibraryChat.__new__(PDFLibraryChat)
    chat_stub.chunk_citation_map = {"A": "[T1, page 1 - Library 1]", "B": "[T2, page 2 - Library 2]"}
    cited = PDFLibraryChat._add_citations_to_context(chat_stub, ["A", "B"])
    assert cited[0].endswith("Library 1]")
    assert cited[1].endswith("Library 2]")
