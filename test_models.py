import cv2
from deepface import DeepFace

img = cv2.imread("temp_face.jpg")

# Test different emotion models explicitly
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "SFace"]

for model in models:
    try:
        r = DeepFace.analyze(
            img,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            model_name=model,
        )
        if isinstance(r, list):
            r = r[0]
        scores = r.get("emotion", {})
        dominant = r.get("dominant_emotion", "?")
        happy = scores.get("happy", 0)
        print(f"{model:15s} → dominant={dominant:10s} happy={happy:.1f}%")
    except Exception as e:
        print(f"{model:15s} → FAILED: {e}")