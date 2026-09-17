import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import dog_face_hub as hub


class FakeYOLO:
    def __init__(self, path):
        self.path = path
        self.task = "detect"
        self.overrides = {}

    def to(self, device):
        self.device = str(device)
        return self


def test_local_checkpoint_loader_sets_device(tmp_path, monkeypatch):
    path = tmp_path / "custom.pt"
    path.write_bytes(b"fake")
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    detector = hub.load_model(model_path=path)
    assert detector.path == str(path)
    assert detector.device == "cpu"
    assert detector.overrides["device"] == "cpu"


def test_checksum_rejects_corrupt_file(tmp_path):
    path = tmp_path / "bad.pt"
    path.write_bytes(b"not the released checkpoint")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        hub._verify_checkpoint(path)


def test_missing_local_model_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    with pytest.raises(FileNotFoundError):
        hub.load_model(model_path=tmp_path / "missing.pt")
