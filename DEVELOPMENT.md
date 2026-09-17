# Development guide

`app/__main__.py` implements the CLI and `app/predictor.py` owns fastai learner loading, GPU validation, and inference. The Docker entry point is `python -m app`; no network port or HTTP service is used.

PyTorch discovers CUDA at startup. GPU inference is mandatory unless the caller explicitly supplies `--allow-cpu`.

PyTorch is pinned to `2.8.0+cu129` with torchvision `0.23.0+cu129` through
PyTorch's CUDA 12.9 wheel index. Do not remove that index or leave PyTorch
unpinned: current PyPI resolution may install a CUDA 13 wheel that requires a
newer NVIDIA driver.

The bundled `model/convnext.pkl` is an exported fastai learner trained with a
ConvNeXt Tiny backbone. The export contains its 400 x 400 padded-image
transforms, normalization, vocabulary, and native four-class probabilities.
Its class order is `angry`, `happy`, `relaxed`, `sad`.

Because `load_learner` uses Python pickle, never replace the model with an
artifact from an untrusted source. The bundled file's SHA-256 is
`64ad7b0562a28bc20e5edb7916ed50aa5878d0fd630de1f3b4dc065f814cb620`.

## Torch Hub development

`hubconf.dog_models` loads both model families in one call and returns
`face_detector`, `emotion_model`, and `emotion_transform` in a dictionary.
The individual loaders remain available. `pytest -q tests/test_hub_bundle.py`
checks argument forwarding without importing PyTorch or downloading models.

The root `hubconf.py` exposes `dog_emotion_convnext`, which returns a PyTorch
module and a preprocessing callable. `dog_emotion_hub.py` downloads immutable
`v1.0.0` assets, validates their SHA-256 digests, and adapts the exported learner.
Use the README's `source="local"` example to validate changes before publishing.
Run `pytest -q tests/test_hub.py` for offline adapter/cache checks; these tests
do not download a model. Validate predictions against `python -m app --model`
on representative images before publishing a new release.

`dog_face_hub.py` adds a separate `dog_face_yolov8n` entry point using
Ultralytics' native prediction API. Install `requirements-face.txt` alongside
the CUDA-pinned base requirements to use both models. The ignored local
`model/dog_face_yolov8n.pt` is preferred when present; publish that exact file as
a `v1.0.0` release asset to enable hosted loading. Its digest is pinned in the
loader. `pytest -q tests/test_face_hub.py` tests the loader with a fake YOLO
object without loading real weights or downloading anything.

## Docker development workflow

Confirm Docker can access the NVIDIA GPU:

```bash
docker run --rm --gpus all \
  nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04 nvidia-smi
```

Rebuild the image after changing the code or dependencies. Use `--no-cache`
when dependency or model-loading changes must be installed from scratch:

```bash
docker build --no-cache -t dog-emotion .
```

Run a prediction for an image stored in the local `images` directory:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/images:/images:ro" \
  dog-emotion /images/angry_dog1.png
```

The equivalent Compose command is:

```bash
docker compose run --rm dog-emotion /images/angry_dog1.png
```

Open an interactive shell in a new GPU-enabled container:

```bash
docker run --rm -it --gpus all \
  -v "$(pwd)/app:/app/app:ro" \
  -v "$(pwd)/images:/images:ro" \
  --entrypoint /bin/bash \
  dog-emotion
```

From inside that container, run prediction with:

```bash
python -m app /images/angry_dog1.png
```

## Performance benchmark

The benchmark measures model loading separately, performs unmeasured warmup
runs, synchronizes CUDA around every measured inference, and reports min, mean,
median, p95, max, standard deviation, and throughput:

```bash
python -m app.benchmark /images/angry_dog1.png \
  --warmup 5 \
  --iterations 50
```

Run it directly through Docker with:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/images:/images:ro" \
  --entrypoint python \
  dog-emotion -m app.benchmark /images/angry_dog1.png \
  --warmup 5 --iterations 50
```

Use a representative production image and avoid running other GPU workloads
during the benchmark. The first model load is intentionally reported as cold
startup; warm inference statistics exclude model loading and warmup.

## Convert the model to FP16

Mount the model directory as writable, enter the container, and run:

```bash
python scripts/convert_model_fp16.py \
  /app/model/convnext.pkl \
  /app/model/convnext-fp16.pkl
```

If only `app/` is mounted during development, also mount `scripts/`:

```bash
-v "$(pwd)/scripts:/app/scripts:ro"
```

The converter refuses to overwrite an existing output unless `--force` is
provided. A fastai learner is a Python pickle, so only convert a model from a
trusted source. Test predictions and benchmark the FP16 output before replacing
the original FP32 model. If a previous conversion attempt already created the
output file, rerun with `--force`. The predictor detects FP16 weights and uses
CUDA autocast so float32 image tensors are safely accepted by the FP16 model.

Run fast unit tests with `pytest -q`. They use a fake model and do not require a GPU.

When replacing the model, use a compatible exported fastai image-classification learner with exactly the vocabulary `angry`, `happy`, `relaxed`, `sad`. The learner export should contain all required inference transforms.
