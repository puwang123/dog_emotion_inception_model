#!/usr/bin/env python3
"""Convert an exported fastai learner's model weights to FP16."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a trusted fastai learner export to FP16 weights."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="source fastai learner, for example model/convnext.pkl",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="output path, for example model/convnext-fp16.pkl",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the output file if it already exists",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def size_mib(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def export_inference_learner(learner, output: Path) -> None:
    """Serialize an inference-only learner without recreating an optimizer."""
    import torch

    # This is the inference-relevant portion of Learner.export(). Calling the
    # public method also recreates its training optimizer after saving, which
    # fails for this third-party learner's older fastcore objects.
    learner._end_cleanup()
    learner.dls = learner.dls.new_empty()
    learner.opt = None

    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        torch.save(learner, temporary, pickle_protocol=2)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    if not source.is_file():
        raise SystemExit(f"Source model not found: {source}")
    if source == output:
        raise SystemExit("Source and output paths must be different.")
    if output.exists() and not args.force:
        raise SystemExit(f"Output already exists: {output} (use --force to replace it)")

    # load_learner uses pickle. Only convert an artifact from a trusted source.
    from fastai.learner import load_learner

    print(f"Loading trusted learner: {source}")
    learner = load_learner(source, cpu=True)
    learner.model.half()

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting FP16 learner: {output}")
    export_inference_learner(learner, output)

    original_size = size_mib(source)
    converted_size = size_mib(output)
    reduction = (1.0 - converted_size / original_size) * 100.0

    print(f"Original size:  {original_size:.1f} MiB")
    print(f"FP16 size:      {converted_size:.1f} MiB")
    print(f"Size reduction: {reduction:.1f}%")
    print(f"SHA-256:        {sha256(output)}")
    print("Conversion complete. Compare predictions before replacing the FP32 model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
