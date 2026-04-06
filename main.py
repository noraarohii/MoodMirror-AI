import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time
import threading
import urllib.request
import numpy as np
from emotion_detector import detect_emotion

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MEME_WIDTH = 200
MEME_HEIGHT = 200

SCREENSHOT_FOLDER = "screenshots"
os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)

MEME_PATHS = {
    "happy":   "memes/happy.jpg",
    "sad":     "memes/sad.jpg",
    "angry":   "memes/angry.jpg",
    "neutral": "memes/neutral.jpg",
}

HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

emotion_check_interval = 0.3
last_emotion_check = 0
current_emotion = "neutral"
current_confidence = 0.0
emotion_busy = False

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

MP_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


def download_model():
    if not os.path.exists(HAND_MODEL_PATH):
        print("[INFO] Downloading hand landmarker model...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        print("[INFO] Download complete.")


def preprocess_face(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def load_memes():
    memes = {}
    for emotion, path in MEME_PATHS.items():
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (MEME_WIDTH, MEME_HEIGHT))
                memes[emotion] = img
            else:
                memes[emotion] = None
                print(f"[ERROR] Could not read image: {path}")
        else:
            memes[emotion] = None
            print(f"[ERROR] File not found: {path}")
    return memes


def emotion_worker_thread(face_img):
    global current_emotion, current_confidence, emotion_busy

    try:
        emotion, confidence = detect_emotion(face_img)
        current_emotion = emotion
        current_confidence = confidence
        print(f"[DETECTED] {emotion} ({confidence:.1f}%)")
    except Exception as e:
        print("[ERROR]", e)

    emotion_busy = False


def start_emotion_detection(frame):
    global emotion_busy

    if emotion_busy:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(80, 80),
        maxSize=(400, 400),
    )

    if len(faces) == 0:
        return

    x, y, w, h = faces[0]
    pad_x = int(w * 0.5)
    pad_y = int(h * 0.6)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)
    face_img = frame[y1:y2, x1:x2]

    if face_img.size == 0:
        return

    emotion_busy = True
    threading.Thread(
        target=emotion_worker_thread,
        args=(face_img.copy(),),
        daemon=True
    ).start()


def count_fingers(lm):
    tips = [4, 8, 12, 16, 20]
    pip_joints = [3, 6, 10, 14, 18]
    fingers = []
    fingers.append(1 if lm[tips[0]].x < lm[pip_joints[0]].x else 0)
    for i in range(1, 5):
        fingers.append(1 if lm[tips[i]].y < lm[pip_joints[i]].y else 0)
    return sum(fingers)


def overlay_meme(frame, meme, emotion):
    if meme is None:
        cv2.putText(frame, f"{emotion} meme not found",
                    (frame.shape[1] - 220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return frame

    fh, fw = frame.shape[:2]
    mh, mw = meme.shape[:2]
    x1 = fw - mw - 10
    y1 = 10
    x2 = x1 + mw
    y2 = y1 + mh

    if x1 < 0 or y1 < 0 or x2 > fw or y2 > fh:
        return frame

    frame[y1:y2, x1:x2] = meme
    cv2.rectangle(frame, (x1, y2 + 5), (x2, y2 + 30), (0, 0, 0), -1)
    cv2.putText(frame, f"MEME: {emotion}",
                (x1 + 5, y2 + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def draw_confidence_bar(frame, emotion, confidence):
    bar_x, bar_y = 10, 80
    bar_w, bar_h = 160, 10

    color_map = {
        "happy":   (0, 220, 100),
        "sad":     (200, 100, 50),
        "angry":   (0, 50, 220),
        "neutral": (160, 160, 160),
    }
    color = color_map.get(emotion, (160, 160, 160))

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    fill_w = int(bar_w * min(confidence, 100.0) / 100.0)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1)
    cv2.putText(frame, f"{confidence:.0f}%",
                (bar_x + bar_w + 6, bar_y + 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def save_screenshot(frame):
    filename = os.path.join(
        SCREENSHOT_FOLDER, f"screenshot_{int(time.time())}.png")
    cv2.imwrite(filename, frame)
    print("[INFO] Saved:", filename)


def main():
    global last_emotion_check

    download_model()
    memes = load_memes()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Camera could not open")
        return

    base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=vision.RunningMode.IMAGE,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    show_meme = False
    countdown = False
    countdown_start = 0
    captured = False
    fingers = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=8,
            minSize=(80, 80), maxSize=(400, 400),
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        fingers = 0

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            h_px, w_px = frame.shape[:2]
            pts = [(int(p.x * w_px), int(p.y * h_px)) for p in lm]
            for a, b in MP_HAND_CONNECTIONS:
                cv2.line(display, pts[a], pts[b], (255, 255, 255), 2)
            for pt in pts:
                cv2.circle(display, pt, 4, (255, 0, 0), -1)

            fingers = count_fingers(lm)

            if fingers == 1:
                show_meme = True
                countdown = False
                captured = False
            elif fingers == 2:
                show_meme = False
                if not countdown:
                    countdown = True
                    countdown_start = time.time()
                    captured = False
            else:
                show_meme = False
                countdown = False
                captured = False
        else:
            show_meme = False
            countdown = False
            captured = False

        if show_meme and (time.time() - last_emotion_check > emotion_check_interval):
            start_emotion_detection(frame)
            last_emotion_check = time.time()

        if show_meme:
            display = overlay_meme(
                display, memes.get(current_emotion), current_emotion)

        cv2.putText(display, f"Emotion: {current_emotion}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        draw_confidence_bar(display, current_emotion, current_confidence)
        cv2.putText(display, f"Fingers: {fingers}",
                    (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if countdown:
            elapsed = time.time() - countdown_start
            remaining = max(1, 3 - int(elapsed))
            if elapsed < 3:
                cv2.putText(display, f"Screenshot in {remaining}",
                            (140, 240), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)
            elif not captured:
                save_screenshot(display)
                captured = True
                countdown = False

        cv2.putText(display,
                    "1 finger = meme | 2 fingers = screenshot | q = quit",
                    (10, FRAME_HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("AI Meme Camera", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()