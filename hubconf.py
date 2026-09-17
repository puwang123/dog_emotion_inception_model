"""Individual and combined Torch Hub loaders for dog emotion and face detection."""

# Keep this list minimal: Hub checks every global dependency for every entry
# point. Model-specific dependencies are imported lazily by each loader.
dependencies = ["torch"]


def dog_emotion_convnext(device="cpu", precision="fp16", model_path=None):
    """Return (model, transform) using the trusted v1.0.0 fastai export.

    transform accepts a PIL image or filename and returns a CHW CPU tensor.
    model accepts normalized NCHW tensors and returns four float32 logits.
    Class order: angry, happy, relaxed, sad. CPU weights are promoted to FP32.
    model_path optionally selects an existing trusted fastai learner export.
    """
    from dog_emotion_hub import load_model

    return load_model(device=device, precision=precision, model_path=model_path)


def dog_face_yolov8n(device="cpu", model_path=None):
    """Return an Ultralytics YOLO dog-face detector with built-in preprocessing.

    Accepts filenames, PIL images, arrays, and batches through predict().
    Returns native Ultralytics Results, including boxes.xyxy/conf/cls.
    Install requirements-face.txt first. Only load trusted checkpoints.
    """
    from dog_face_hub import load_model

    return load_model(device=device, model_path=model_path)


def dog_models(
    device="cpu",
    precision="fp16",
    face_model_path=None,
    emotion_model_path=None,
):
    """Load both models in one Hub call and return named components.

    precision controls the emotion model only. Optional local checkpoint paths
    bypass release downloads. Detection does not automatically run emotion
    classification; compose the returned components explicitly.
    """
    detector = dog_face_yolov8n(device=device, model_path=face_model_path)
    model, transform = dog_emotion_convnext(
        device=device, precision=precision, model_path=emotion_model_path
    )
    return {
        "face_detector": detector,
        "emotion_model": model,
        "emotion_transform": transform,
    }
