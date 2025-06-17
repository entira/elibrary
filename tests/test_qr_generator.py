import sys
import types
from pathlib import Path
from modules.qr_generator import QRGenerator


class DummyEncoder:
    def add_text(self, text):
        pass
    def _build_qr_frame(self, path, index):
        Path(path).write_text("frame")


def test_generate_qr_frames_sequential(monkeypatch, tmp_path):
    dummy_mod = types.SimpleNamespace(MemvidEncoder=DummyEncoder)
    monkeypatch.setitem(sys.modules, "memvid", dummy_mod)

    chunks = [{"text": "one"}, {"text": "two"}]
    generator = QRGenerator(n_workers=1, show_progress=False)
    frames_dir, stats = generator.generate_qr_frames_sequential(chunks, tmp_path)
    assert stats["total_frames"] == 2
    validation = generator.validate_frames(frames_dir)
    assert validation["valid"] is True
