import cv2
import numpy as np
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier

EMOTION_PYTHON = r".\deepface_env\Scripts\python.exe"
SAVE_PATH = "emotion_model.pkl"
SAMPLES_PER_EMOTION = 30

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def preprocess(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def extract_features(face_img):
    face = cv2.resize(face_img, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = preprocess(cv2.cvtColor(face, cv2.COLOR_GRAY2BGR))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return face.flatten().astype(np.float32) / 255.0

def get_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=8,
        minSize=(80, 80), maxSize=(400, 400)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    pad_x, pad_y = int(w * 0.3), int(h * 0.3)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)
    return frame[y1:y2, x1:x2]

def collect_samples(cap, emotion_label, emotion_name):
    samples = []
    print(f"\n>>> Make your {emotion_name.upper()} face and press SPACE to collect {SAMPLES_PER_EMOTION} samples. Press Q to skip.")

    while len(samples) < SAMPLES_PER_EMOTION:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        face = get_face(frame)
        if face is not None:
            cv2.putText(display, "Face detected!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(display,
                    f"{emotion_name.upper()}: {len(samples)}/{SAMPLES_PER_EMOTION} — SPACE to capture, Q to skip",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and face is not None:
            feat = extract_features(face)
            samples.append(feat)
            print(f"  Captured {len(samples)}/{SAMPLES_PER_EMOTION}")
        elif key == ord('q'):
            print(f"  Skipped {emotion_name}")
            return [], []

    return samples, [emotion_label] * len(samples)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    all_features = []
    all_labels = []

    emotions = [
        ("happy",   "happy 😊 — smile big!"),
        ("sad",     "sad 😢 — frown, look down"),
        ("angry",   "angry 😠 — furrowed brows, stern face"),
        ("neutral", "neutral 😐 — relaxed, no expression"),
    ]

    print("=== EMOTION CALIBRATION ===")
    print("You'll collect samples for each emotion.")
    print("Press SPACE to capture each sample.\n")

    for label, name in emotions:
        samples, labels = collect_samples(cap, label, name)
        all_features.extend(samples)
        all_labels.extend(labels)

    cap.release()
    cv2.destroyAllWindows()

    if len(all_features) < 10:
        print("Not enough samples collected. Try again.")
        return

    print(f"\nTraining on {len(all_features)} samples...")
    clf = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    clf.fit(all_features, all_labels)

    with open(SAVE_PATH, "wb") as f:
        pickle.dump(clf, f)

    print(f"Saved model to {SAVE_PATH}")
    print("Now run main.py — it will use your personal emotion model!")

if __name__ == "__main__":
    main()