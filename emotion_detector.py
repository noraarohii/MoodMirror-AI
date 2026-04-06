import cv2
import numpy as np
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

EMOTION_MAP = {
    "Happy":    "happy",
    "Sad":      "sad",
    "Angry":    "angry",
    "Neutral":  "neutral",
    "Fear":     "sad",
    "Disgust":  "angry",
    "Surprise": "happy",
    "Contempt": "angry",
}

_recognizer = None

def get_recognizer():
    global _recognizer
    if _recognizer is None:
        _recognizer = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")
    return _recognizer

def detect_emotion(face_img):
    try:
        rec = get_recognizer()
        emotion, scores = rec.predict_emotions(face_img, logits=False)
        confidence = float(max(scores)) * 100
        mapped = EMOTION_MAP.get(emotion, "neutral")
        return mapped, confidence
    except Exception as e:
        print(f"[EMOTION ERROR] {e}")
        return "neutral", 0.0