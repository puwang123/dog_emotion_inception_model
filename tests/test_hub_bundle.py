"""Offline tests for the combined entry point; no torch/model downloads needed."""

import hubconf


def test_bundle_forwards_options_and_returns_named_components(monkeypatch):
    detector, model, transform = object(), object(), object()
    calls = []

    def fake_face(**kwargs):
        calls.append(("face", kwargs))
        return detector

    def fake_emotion(**kwargs):
        calls.append(("emotion", kwargs))
        return model, transform

    monkeypatch.setattr(hubconf, "dog_face_yolov8n", fake_face)
    monkeypatch.setattr(hubconf, "dog_emotion_convnext", fake_emotion)
    bundle = hubconf.dog_models(
        device="cuda:0", precision="fp32",
        face_model_path="face.pt", emotion_model_path="emotion.pkl",
    )
    assert bundle == {
        "face_detector": detector,
        "emotion_model": model,
        "emotion_transform": transform,
    }
    assert calls == [
        ("face", {"device": "cuda:0", "model_path": "face.pt"}),
        ("emotion", {
            "device": "cuda:0", "precision": "fp32", "model_path": "emotion.pkl",
        }),
    ]
