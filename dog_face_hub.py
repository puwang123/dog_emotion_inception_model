"""Local/release-backed YOLOv8n dog-face detector for Torch Hub."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

_FILENAME = "dog_face_yolov8n.pt"
_SHA256 = "457ffa4c43ada20d9bbbcd0e2fd3dc678f71f8464a0ee5a687716648c43757bf"
_URL = (
    "https://github.com/puwang123/dog_emotion_inception_model/releases/"
    "download/v1.0.0/dog_face_yolov8n.pt"
)


def _verify_checkpoint(path: Path) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != _SHA256:
        raise RuntimeError(f"Dog-face checkpoint checksum mismatch: {path}")


def _default_checkpoint() -> Path:
    local = Path(__file__).resolve().parent / "model" / _FILENAME
    if local.is_file():
        _verify_checkpoint(local)
        return local

    cache = Path(torch.hub.get_dir()) / "checkpoints" / "dog-face-v1.0.0"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / _FILENAME
    if not path.exists():
        try:
            torch.hub.download_url_to_file(_URL, str(path), hash_prefix=_SHA256)
        except Exception as exc:
            raise RuntimeError(
                "Could not download the dog-face checkpoint. Upload "
                "model/dog_face_yolov8n.pt to the v1.0.0 GitHub release, "
                "or supply model_path pointing to a local trusted checkpoint."
            ) from exc
    _verify_checkpoint(path)
    return path


def load_model(device="cpu", model_path=None):
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA devices are supported")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use device='cpu' or check your driver")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install dog-face dependencies: pip install -r requirements-face.txt") from exc

    path = Path(model_path) if model_path is not None else _default_checkpoint()
    if not path.is_file():
        raise FileNotFoundError(path)
    # Ultralytics .pt checkpoints may deserialize Python objects. Trust the
    # checkpoint source even when its checksum is verified.
    detector = YOLO(str(path))
    if detector.task != "detect":
        raise ValueError(f"Expected a detection checkpoint, got task={detector.task!r}")
    detector.to(target)
    detector.overrides["device"] = str(target)
    return detector
