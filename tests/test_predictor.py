import io
import threading
from contextlib import nullcontext

import numpy as np
import pytest
from PIL import Image

from app.predictor import DogEmotionPredictor, PredictionError


class FakeLearner:
    def __init__(self, scores):
        self.scores = scores

    def predict(self, _image):
        return "happy", 1, self.scores


def predictor_with_scores(scores):
    predictor = DogEmotionPredictor.__new__(DogEmotionPredictor)
    predictor._learner = FakeLearner(scores)
    predictor._class_names = ("angry", "happy", "relaxed", "sad")
    predictor._to_model_image = lambda image: image
    predictor._inference_context = nullcontext
    predictor._autocast_context = nullcontext
    predictor._lock = threading.Lock()
    return predictor


def image_bytes():
    image = io.BytesIO()
    Image.new("RGB", (8, 8), "gold").save(image, "JPEG")
    image.seek(0)
    return image


def test_prediction_uses_native_multiclass_probabilities():
    predictor = predictor_with_scores(np.array([0.05, 0.8, 0.1, 0.05]))

    result = predictor.predict(image_bytes())

    assert result["emotion"] == "happy"
    assert result["confidence"] == 0.8
    assert tuple(result["probabilities"]) == ("angry", "happy", "relaxed", "sad")


def test_rejects_invalid_image():
    predictor = predictor_with_scores(np.array([0.05, 0.8, 0.1, 0.05]))

    with pytest.raises(PredictionError, match="valid image"):
        predictor.predict(io.BytesIO(b"not an image"))
