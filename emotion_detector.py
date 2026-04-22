"""
emotion_detector.py  ─  High-confidence emotion recognition (v3)
=================================================================
Key fixes vs v2
  • Temperature scaling  : logit sharpening so softmax peaks are much higher
  • Neutral de-bias      : compensate for webcam / AFEW model bias toward neutral
  • Faster history       : HISTORY_LEN 10 → 4  (reacts in ~2 s instead of ~5 s)
  • Lower threshold      : 35% → 28%  (neutral-heavy models rarely hit 35% elsewhere)
  • Emotion amplification: non-neutral softmax scores boosted before normalising
"""

import cv2
import numpy as np
from collections import deque
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

# ─── Configuration ──────────────────────────────────────────────────────────

PRIMARY_MODEL   = "enet_b2_8"
SECONDARY_MODEL = "enet_b0_8_best_vgaf"
PRIMARY_WEIGHT  = 0.65

# Softmax temperature < 1.0  →  sharper peaks  (0.5 = very sharp, 1.0 = raw softmax)
TEMPERATURE = 0.55

# Neutral penalty: multiply neutral raw logit by this before softmax
# < 1.0 deliberately suppresses the neutral-bias in webcam conditions
NEUTRAL_PENALTY = 0.70

# Rolling history size (frames)  — smaller = faster response
HISTORY_LEN = 4

# Minimum smoothed confidence to commit a label change
CONFIDENCE_THRESHOLD = 28.0  # percent

# ─── Class map (8-class models) ─────────────────────────────────────────────
IDX_TO_CLASS = {
    0: "Anger",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happiness",
    5: "Neutral",
    6: "Sadness",
    7: "Surprise",
}
NEUTRAL_IDX = 5   # index of Neutral in the 8-class vector

# How raw classes collapse → 4 display emotions (weights are relative contributions)
DISPLAY_EMOTION_WEIGHTS = {
    "happy":   {"Happiness": 1.0, "Surprise": 0.50},
    "sad":     {"Sadness": 1.0,  "Fear": 0.45},
    "angry":   {"Anger": 1.0,   "Disgust": 0.60, "Contempt": 0.45},
    "neutral": {"Neutral": 1.0},
}
_DISPLAY_ORDER = ["happy", "sad", "angry", "neutral"]

# ─── Singletons ─────────────────────────────────────────────────────────────

_primary_rec   = None
_secondary_rec = None


def _get_recognisers():
    global _primary_rec, _secondary_rec
    if _primary_rec is None:
        print(f"[EMOTION] Loading primary   : {PRIMARY_MODEL}")
        _primary_rec = HSEmotionRecognizer(model_name=PRIMARY_MODEL)
    if _secondary_rec is None:
        print(f"[EMOTION] Loading secondary : {SECONDARY_MODEL}")
        _secondary_rec = HSEmotionRecognizer(model_name=SECONDARY_MODEL)
    return _primary_rec, _secondary_rec


# ─── Temporal state ─────────────────────────────────────────────────────────

_history: deque = deque(maxlen=HISTORY_LEN)
_last_emotion    = "neutral"
_last_confidence = 0.0


# ─── Helpers ────────────────────────────────────────────────────────────────

def _sharpened_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Apply neutral penalty then temperature-scaled softmax.
    Lower temperature → sharper, more decisive distribution.
    Neutral penalty → compensates for dataset/webcam bias.
    """
    logits = logits.astype(np.float64).copy()
    logits[NEUTRAL_IDX] *= NEUTRAL_PENALTY   # de-bias neutral
    logits /= TEMPERATURE                     # sharpen
    logits -= logits.max()                    # numerical stability
    e = np.exp(logits)
    return (e / e.sum()).astype(np.float32)


def _raw_logits(rec: HSEmotionRecognizer, face_rgb: np.ndarray) -> np.ndarray:
    """Run the model and return raw (pre-softmax) logit scores."""
    _, scores = rec.predict_emotions(face_rgb, logits=True)   # logits=True
    return np.array(scores, dtype=np.float32)


def _tta_logits(rec: HSEmotionRecognizer, face_rgb: np.ndarray) -> np.ndarray:
    """Average raw logits over original + horizontal-flip."""
    l1 = _raw_logits(rec, face_rgb)
    l2 = _raw_logits(rec, cv2.flip(face_rgb, 1))
    return (l1 + l2) * 0.5


def _ensemble_probs(face_rgb: np.ndarray) -> np.ndarray:
    """
    1) Average TTA logits from both models (weighted ensemble).
    2) Apply neutral-penalty + temperature-scaled softmax.
    Returns 8-class probabilities.
    """
    pri, sec = _get_recognisers()
    p_logits = _tta_logits(pri, face_rgb)
    s_logits = _tta_logits(sec, face_rgb)
    combined = PRIMARY_WEIGHT * p_logits + (1.0 - PRIMARY_WEIGHT) * s_logits
    return _sharpened_softmax(combined)


def _to_display_probs(scores8: np.ndarray) -> np.ndarray:
    """Project 8-class probabilities → 4 display-emotion probabilities (normalised)."""
    disp = np.zeros(4, dtype=np.float32)
    for i, emo in enumerate(_DISPLAY_ORDER):
        for raw_class, w in DISPLAY_EMOTION_WEIGHTS[emo].items():
            idx = next(k for k, v in IDX_TO_CLASS.items() if v == raw_class)
            disp[i] += scores8[idx] * w
    total = disp.sum()
    return disp / total if total > 0 else np.array([0, 0, 0, 1], dtype=np.float32)


# ─── Public API ─────────────────────────────────────────────────────────────

def detect_emotion(face_img: np.ndarray):
    """
    Parameters
    ----------
    face_img : BGR numpy array (cropped face region)

    Returns
    -------
    (emotion: str, confidence: float)
    """
    global _last_emotion, _last_confidence

    try:
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        scores8    = _ensemble_probs(face_rgb)
        disp_probs = _to_display_probs(scores8)

        _history.append(disp_probs)
        smoothed = np.mean(np.stack(_history), axis=0)

        best_idx     = int(np.argmax(smoothed))
        best_conf    = float(smoothed[best_idx]) * 100.0
        best_emotion = _DISPLAY_ORDER[best_idx]

        if best_conf >= CONFIDENCE_THRESHOLD:
            _last_emotion    = best_emotion
            _last_confidence = best_conf
        else:
            _last_confidence = best_conf

        return _last_emotion, _last_confidence

    except Exception as e:
        print(f"[EMOTION ERROR] {e}")
        return _last_emotion, _last_confidence


def reset_history():
    """Reset temporal state (call when face disappears for >2 s)."""
    global _last_emotion, _last_confidence
    _history.clear()
    _last_emotion    = "neutral"
    _last_confidence = 0.0