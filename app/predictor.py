"""GPU-backed fastai dog-emotion prediction."""

from __future__ import annotations

import io
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import BinaryIO, Callable, ContextManager

import numpy as np
from PIL import Image, UnidentifiedImageError

EXPECTED_CLASS_NAMES = ("angry", "happy", "relaxed", "sad")


class PredictionError(RuntimeError):
    pass


class ModelUnavailableError(RuntimeError):
    pass


class DogEmotionPredictor:
    def __init__(self, model_path: str, *, require_gpu: bool = True):
        path = Path(model_path)
        if not path.is_file():
            raise ModelUnavailableError(f"Model file not found at {path}.")

        try:
            import torch
            from fastai.learner import load_learner
            from fastai.vision.core import PILImage

            gpu_available = torch.cuda.is_available()
            if require_gpu and not gpu_available:
                raise ModelUnavailableError(
                    "No CUDA GPU detected. Run the container with --gpus all, or use "
                    "--allow-cpu for an explicit CPU fallback."
                )

            use_gpu = gpu_available
            self._learner = load_learner(path, cpu=not use_gpu)
            if use_gpu:
                device = torch.device("cuda")
                self._learner.model.to(device)
                self._learner.dls.to(device)
                model_dtype = next(self._learner.model.parameters()).dtype
                self._autocast_context: Callable[[], ContextManager] = (
                    lambda: torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=model_dtype == torch.float16,
                    )
                )
            else:
                # Many CPU operations do not support FP16. Explicit CPU
                # fallback therefore restores FP32 weights before inference.
                self._learner.model.float()
                self._autocast_context = nullcontext

            class_names = tuple(str(name) for name in self._learner.dls.vocab)
            if class_names != EXPECTED_CLASS_NAMES:
                raise ModelUnavailableError(
                    f"Model classes are {class_names}; expected {EXPECTED_CLASS_NAMES}."
                )

            self._class_names = class_names
            self._to_model_image: Callable[[Image.Image], object] = PILImage.create
            self._inference_context: Callable[[], ContextManager] = torch.inference_mode
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(f"Could not load the prediction model: {exc}") from exc

        self._lock = threading.Lock()

    def predict(self, stream: BinaryIO) -> dict:
        try:
            raw = stream.read()
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise PredictionError("The input file is not a valid image.") from exc

        try:
            with (
                self._lock,
                self._inference_context(),
                self._autocast_context(),
            ):
                _label, _index, scores = self._learner.predict(
                    self._to_model_image(image)
                )
        except Exception as exc:
            raise PredictionError(f"Prediction failed: {exc}") from exc

        if hasattr(scores, "detach"):
            scores = scores.detach().float().cpu().numpy()
        probabilities = np.asarray(scores, dtype=np.float64).reshape(-1)
        if probabilities.size != len(self._class_names):
            raise PredictionError(
                f"Model returned {probabilities.size} classes; "
                f"expected {len(self._class_names)}."
            )
        if not np.all(np.isfinite(probabilities)):
            raise PredictionError("Model returned non-finite probabilities.")

        best = int(np.argmax(probabilities))
        return {
            "emotion": self._class_names[best],
            "confidence": round(float(probabilities[best]), 6),
            "probabilities": {
                name: round(float(score), 6)
                for name, score in zip(self._class_names, probabilities, strict=True)
            },
        }
