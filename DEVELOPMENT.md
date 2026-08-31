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

Run fast unit tests with `pytest -q`. They use a fake model and do not require a GPU.

When replacing the model, use a compatible exported fastai image-classification learner with exactly the vocabulary `angry`, `happy`, `relaxed`, `sad`. The learner export should contain all required inference transforms.
