"""Release-backed model and preprocessing adapters for Torch Hub."""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

_ASSETS = {
    "fp16": (
        "convnext-fp16.pkl",
        "3d12b39a7832529a5bb412c9ebee4cb14fba41e6fe66b74c2a6041ce21519247",
    ),
    "fp32": (
        "convnext.pkl",
        "64ad7b0562a28bc20e5edb7916ed50aa5878d0fd630de1f3b4dc065f814cb620",
    ),
}
_RELEASE = "https://github.com/puwang123/dog_emotion_inception_model/releases/download/v1.0.0"
_CLASS_NAMES = ("angry", "happy", "relaxed", "sad")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_model(precision: str) -> Path:
    filename, digest = _ASSETS[precision]
    cache = Path(torch.hub.get_dir()) / "checkpoints" / "dog-emotion-v1.0.0"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / filename
    if not path.exists():
        torch.hub.download_url_to_file(
            f"{_RELEASE}/{filename}", str(path), hash_prefix=digest
        )
    if _sha256(path) != digest:
        raise RuntimeError(
            f"Cached model checksum mismatch: {path}. Remove this file and retry."
        )
    return path


class DogEmotionModel(nn.Module):
    """Tensor interface to the trained network; forward returns logits."""

    class_names = _CLASS_NAMES

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        parameter = next(self.network.parameters())
        context = (
            torch.autocast("cuda", dtype=torch.float16)
            if parameter.is_cuda and parameter.dtype == torch.float16
            else nullcontext()
        )
        with context:
            return self.network(images.to(dtype=parameter.dtype)).float()


class DogEmotionTransform:
    """Apply the export's validation transforms, including normalization."""

    def __init__(self, dataloaders):
        self.dataloaders = dataloaders

    def __call__(self, image) -> torch.Tensor:
        from fastai.vision.core import PILImage

        loader = self.dataloaders.test_dl(
            [PILImage.create(image)], bs=1, num_workers=0
        )
        batch = loader.one_batch()
        return batch[0][0].as_subclass(torch.Tensor)


def load_model(device="cpu", precision="fp16", model_path=None):
    if precision not in _ASSETS:
        raise ValueError("precision must be 'fp16' or 'fp32'")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA devices are supported")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use device='cpu' or check your driver")

    path = Path(model_path) if model_path is not None else _download_model(precision)
    if not path.is_file():
        raise FileNotFoundError(path)

    # These release exports are pickle files. Only load trusted artifacts.
    from fastai.learner import load_learner

    learner = load_learner(path, cpu=True)
    vocabulary = tuple(str(name) for name in learner.dls.vocab)
    if vocabulary != _CLASS_NAMES:
        raise ValueError(f"Unexpected model vocabulary: {vocabulary}")
    learner.dls.to(torch.device("cpu"))
    transform = DogEmotionTransform(learner.dls)
    dtype = torch.float16 if precision == "fp16" and target.type == "cuda" else torch.float32
    network = learner.model.eval().to(device=target, dtype=dtype)
    model = DogEmotionModel(network).eval()
    return model, transform
