# Dog Emotion Predictor

A command-line image classifier that uses a fastai/PyTorch ConvNeXt Tiny model and an NVIDIA GPU to estimate a dog's visible expression as **angry**, **happy**, **relaxed**, or **sad**. There is no HTTP server or web interface.

The included `model/convnext.pkl` comes from Mohit Agarwal's [Dog Emotion with Clean Dataset notebook](https://www.kaggle.com/code/mohitagarwal17/dog-emotion-with-clean-dataset-93-76-accuracy/notebook). Its class order is `angry`, `happy`, `relaxed`, `sad`, and its SHA-256 is `64ad7b0562a28bc20e5edb7916ed50aa5878d0fd630de1f3b4dc065f814cb620`.

The model is a Python pickle. Only load it if you trust its Kaggle source; loading an untrusted pickle can execute arbitrary code.

## Torch Hub

Install the dependencies first (Torch Hub checks them but does not install them):

```bash
pip install -r requirements.txt
```

Load the release model and its exact saved preprocessing, similar to Gazelle:

```python
import torch
from PIL import Image

model, transform = torch.hub.load(
    "puwang123/dog_emotion_inception_model:main",
    "dog_emotion_convnext",
    device="cuda",
    precision="fp16",
    trust_repo=True,
)

image = Image.open("images/angry_dog1.png").convert("RGB")
batch = transform(image).unsqueeze(0).to("cuda")
with torch.inference_mode():
    probabilities = model(batch).softmax(dim=-1)[0]

index = int(probabilities.argmax())
print(model.class_names[index], float(probabilities[index]))
```

The default FP16 download uses the `v1.0.0/convnext-fp16.pkl` release asset
and verifies its SHA-256 before deserialization. Files are cached beneath
`torch.hub.get_dir()`. Use `precision="fp32"` for the full-precision asset,
or `device="cpu"` for CPU inference (weights are promoted to FP32).
The returned model accepts normalized NCHW batches and returns logits in
`angry`, `happy`, `relaxed`, `sad` order; apply softmax once for probabilities.
The transform accepts a filename or PIL image and returns a CHW CPU tensor.

Both Torch Hub repository code and the fastai pickle must be trusted.
`trust_repo=True` acknowledges execution of repository code; checksum verification
detects a changed artifact but does not sandbox pickle deserialization.

Test unpublished local changes without downloading repository code:

```python
model, transform = torch.hub.load(
    ".", "dog_emotion_convnext", source="local",
    model_path="model/convnext-fp16.pkl", device="cuda",
)
```

Commit and push `hubconf.py`, `dog_emotion_hub.py`, and the documentation to
`main` before using the GitHub-loading example. If repository code is already
cached, add `force_reload=True` once to refresh it. No model re-upload is required.

## Requirements

- An NVIDIA GPU and compatible host driver
- Docker Engine
- NVIDIA Container Toolkit configured for Docker

The Python dependencies pin PyTorch 2.8.0 with its CUDA 12.9 wheel, matching
the CUDA 12.9 container and avoiding a CUDA 13 driver requirement.

Confirm that Docker can see the GPU before building this project:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04 nvidia-smi
```

## Predict with Docker

Build the image:

```bash
docker build -t dog-emotion .
```

Mount the directory containing the input image and pass its container path:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/images:/images:ro" \
  dog-emotion /images/dog.jpg
```

The script writes the predicted emotion, confidence, and all class probabilities as JSON. CUDA is required by default, so a missing GPU produces a clear error instead of silently running on the CPU.

With Compose, put the image in `./images` and run:

```bash
docker compose run --rm dog-emotion /images/dog.jpg
```

To explicitly permit CPU fallback (mainly for development), append `--allow-cpu`. To use another compatible exported fastai learner, mount it and pass `--model /path/model.pkl`.

Benchmark warm GPU inference and cold model loading:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/images:/images:ro" \
  --entrypoint python \
  dog-emotion -m app.benchmark /images/dog.jpg \
  --warmup 5 --iterations 50
```

### Reference benchmark

Measured on an NVIDIA GeForce RTX 4060 Ti using PyTorch 2.8.0+cu129 and CUDA
12.9, with 5 warmup runs and 200 measured predictions of
`/images/angry_dog1.png`:

| Measurement | Result |
|---|---:|
| Cold model load | 4,004.513 ms |
| Minimum latency | 27.182 ms |
| Mean latency | 31.543 ms |
| Median latency | 31.121 ms |
| p95 latency | 35.840 ms |
| Maximum latency | 42.267 ms |
| Latency standard deviation | 2.478 ms |
| Throughput | 31.703 images/second |

The final measured prediction was `angry` with 95.2428% confidence. Results
will vary with GPU model, driver, power state, host load, storage, container
cache state, and input image.

Full benchmark output:

```json
{
  "device": "NVIDIA GeForce RTX 4060 Ti",
  "pytorch_version": "2.8.0+cu129",
  "pytorch_cuda_version": "12.9",
  "image": "/images/angry_dog1.png",
  "model_load_ms": 4004.513,
  "warmup_iterations": 5,
  "measured_iterations": 200,
  "latency": {
    "min_ms": 27.182,
    "mean_ms": 31.543,
    "median_ms": 31.121,
    "p95_ms": 35.84,
    "max_ms": 42.267,
    "stddev_ms": 2.478,
    "throughput_images_per_second": 31.703
  },
  "last_prediction": {
    "emotion": "angry",
    "confidence": 0.952428,
    "probabilities": {
      "angry": 0.952428,
      "happy": 0.014476,
      "relaxed": 0.024824,
      "sad": 0.008272
    }
  }
}
```

## Local development

Install the dependencies and invoke the same module directly:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m app --model model/convnext.pkl path/to/dog.jpg
pytest -q
```

The exported learner owns its 400 x 400 padded-image preprocessing, pretrained-model normalization, class vocabulary, and multiclass probability conversion. The result is an image-classification estimate, not a reliable account of an animal's internal emotional state or veterinary advice.
