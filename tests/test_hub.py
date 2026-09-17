import hashlib

import pytest

torch = pytest.importorskip("torch")

import dog_emotion_hub as hub


def test_tensor_model_returns_four_logits():
    model = hub.DogEmotionModel(torch.nn.Linear(3, 4)).eval()
    with torch.inference_mode():
        output = model(torch.ones(2, 3))
    assert output.shape == (2, 4)
    assert output.dtype == torch.float32
    assert model.class_names == ("angry", "happy", "relaxed", "sad")


def test_sha256(tmp_path):
    path = tmp_path / "model.pkl"
    path.write_bytes(b"test artifact")
    assert hub._sha256(path) == hashlib.sha256(b"test artifact").hexdigest()


def test_corrupt_cached_pickle_is_rejected_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    cached = tmp_path / "checkpoints" / "dog-emotion-v1.0.0" / "convnext-fp16.pkl"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"not the released model")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        hub._download_model("fp16")


def test_invalid_precision_is_rejected():
    with pytest.raises(ValueError, match="precision"):
        hub.load_model(precision="int8")
