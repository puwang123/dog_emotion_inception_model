"""Measure cold-start and warm inference performance."""

from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from .predictor import DogEmotionPredictor, ModelUnavailableError, PredictionError


def _positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark dog-emotion model loading and inference."
    )
    parser.add_argument("image", type=Path, help="representative input image")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.getenv("MODEL_PATH", "/app/model/convnext.pkl")),
        help="fastai exported learner",
    )
    parser.add_argument("--warmup", type=_positive_integer, default=5)
    parser.add_argument("--iterations", type=_positive_integer, default=50)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit benchmarking without CUDA",
    )
    return parser.parse_args()


def _synchronize_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize_latencies(seconds: list[float]) -> dict[str, float]:
    milliseconds = np.asarray(seconds, dtype=np.float64) * 1000.0
    return {
        "min_ms": round(float(milliseconds.min()), 3),
        "mean_ms": round(float(milliseconds.mean()), 3),
        "median_ms": round(float(np.median(milliseconds)), 3),
        "p95_ms": round(float(np.percentile(milliseconds, 95)), 3),
        "max_ms": round(float(milliseconds.max()), 3),
        "stddev_ms": round(float(milliseconds.std()), 3),
        "throughput_images_per_second": round(1.0 / statistics.mean(seconds), 3),
    }


def main() -> int:
    args = _arguments()
    if not args.image.is_file():
        print(f"Image file not found: {args.image}", file=sys.stderr)
        return 2

    try:
        import torch

        image_data = args.image.read_bytes()

        load_started = time.perf_counter()
        predictor = DogEmotionPredictor(
            str(args.model), require_gpu=not args.allow_cpu
        )
        _synchronize_cuda()
        load_seconds = time.perf_counter() - load_started

        result = None
        for _ in range(args.warmup):
            result = predictor.predict(io.BytesIO(image_data))
        _synchronize_cuda()

        latencies: list[float] = []
        for _ in range(args.iterations):
            _synchronize_cuda()
            started = time.perf_counter()
            result = predictor.predict(io.BytesIO(image_data))
            _synchronize_cuda()
            latencies.append(time.perf_counter() - started)

        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        report = {
            "device": device,
            "pytorch_version": torch.__version__,
            "pytorch_cuda_version": torch.version.cuda,
            "image": str(args.image),
            "model_load_ms": round(load_seconds * 1000.0, 3),
            "warmup_iterations": args.warmup,
            "measured_iterations": args.iterations,
            "latency": summarize_latencies(latencies),
            "last_prediction": result,
        }
    except (ModelUnavailableError, PredictionError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
