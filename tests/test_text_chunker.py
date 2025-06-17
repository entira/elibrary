import sys, types, tiktoken
import pytest
from modules.text_chunker import TextChunker


def test_create_enhanced_chunks_basic(monkeypatch):
    dummy = types.SimpleNamespace(encode=lambda x, **k: [ord(c) for c in x], decode=lambda l: "".join(chr(i) for i in l))
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda name: dummy)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: dummy)
    page_texts = {
        1: "This is page one. " * 20,
        2: "Second page text. " * 20,
    }
    chunker = TextChunker(chunk_size=50, overlap_percentage=0.0)
    chunks = chunker.create_enhanced_chunks(page_texts)
    assert len(chunks) > 0
    first = chunks[0]
    assert first.start_page == 1
    stats = chunker.get_chunk_stats(chunks)
    assert stats["total_chunks"] == len(chunks)
