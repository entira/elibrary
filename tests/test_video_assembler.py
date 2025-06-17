import sys
import types
from pathlib import Path
from modules.video_assembler import VideoAssembler


class DummyEncoder:
    def __init__(self):
        self.text_data = []
    def add_text(self, text):
        self.text_data.append({"text": text, "metadata": {}})
    def build_video(self, path, fps=30, quality="medium", compression=True):
        Path(path).write_bytes(b"video")
    def get_index(self):
        return self.text_data


def test_assemble_video(monkeypatch, tmp_path):
    dummy_mod = types.SimpleNamespace(MemvidEncoder=DummyEncoder)
    monkeypatch.setitem(sys.modules, "memvid", dummy_mod)

    chunks = [
        {"text": "a", "metadata": {"file_name": "file.pdf", "token_count": 5, "num_pages": 1}},
        {"text": "b", "metadata": {"file_name": "file.pdf", "token_count": 4, "num_pages": 1}},
    ]
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    video_path = tmp_path / "out.mp4"
    index_path = tmp_path / "index.json"

    assembler = VideoAssembler()
    result = assembler.assemble_video(frames_dir, chunks, video_path, index_path)
    assert result["success"] is True
    assert video_path.exists()
    assert index_path.exists()
    validation = assembler.validate_index_output(index_path)
    assert validation["valid"] is True
