"""Torch Hub entry points for dog-emotion classification."""

dependencies = ["torch", "torchvision", "fastai", "timm", "numpy", "PIL"]


def dog_emotion_convnext(device="cpu", precision="fp16", model_path=None):
    """Return (model, transform) using the trusted v1.0.0 fastai export.

    transform accepts a PIL image or filename and returns a CHW CPU tensor.
    model accepts normalized NCHW tensors and returns four float32 logits.
    Class order: angry, happy, relaxed, sad. CPU weights are promoted to FP32.
    model_path optionally selects an existing trusted fastai learner export.
    """
    from dog_emotion_hub import load_model

    return load_model(device=device, precision=precision, model_path=model_path)
