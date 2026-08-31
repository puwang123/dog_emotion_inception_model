"""Run dog-emotion prediction for a local image."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .predictor import DogEmotionPredictor, ModelUnavailableError, PredictionError


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a dog's visible expression.")
    parser.add_argument("image", type=Path, help="JPG, PNG, or WebP image to classify")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.getenv("MODEL_PATH", "/app/model/convnext.pkl")),
        help="fastai exported learner (default: MODEL_PATH or /app/model/convnext.pkl)",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit CPU inference when CUDA is unavailable",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if not args.image.is_file():
        print(f"Image file not found: {args.image}", file=sys.stderr)
        return 2

    try:
        predictor = DogEmotionPredictor(str(args.model), require_gpu=not args.allow_cpu)
        with args.image.open("rb") as image:
            result = predictor.predict(image)
    except (ModelUnavailableError, PredictionError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
