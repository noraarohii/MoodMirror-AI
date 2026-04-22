import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time
import threading
import urllib.request
import numpy as np
from flask import Flask, render_template, Response, send_from_directory
from emotion_detector import detect_emotion

app = Flask(__name__)

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MEME_WIDTH = 180
MEME_HEIGHT = 180

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
show_meme = False
current_fingers = 0

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
        gray, scaleFactor=1.1, minNeighbors=8,
        minSize=(80, 80), maxSize=(400, 400),
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


def draw_ui(frame, emotion, confidence, fingers):
    emotion_colors = {
        "happy":   (0, 220, 100),
        "sad":     (200, 100, 50),
        "angry":   (0, 50, 220),
        "neutral": (160, 160, 160),
    }
    color = emotion_colors.get(emotion, (160, 160, 160))

    # Emotion label
    cv2.putText(frame, f"{emotion.upper()}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Confidence bar background
    cv2.rectangle(frame, (10, 45), (170, 58), (50, 50, 50), -1)
    fill_w = int(160 * min(confidence, 100.0) / 100.0)
    if fill_w > 0:
        cv2.rectangle(frame, (10, 45), (10 + fill_w, 58), color, -1)
    cv2.putText(frame, f"{confidence:.0f}%",
                (175, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Finger count
    cv2.putText(frame, f"fingers: {fingers}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def save_screenshot(frame):
    filename = os.path.join(
        SCREENSHOT_FOLDER, f"screenshot_{int(time.time())}.png")
    cv2.imwrite(filename, frame)
    print("[INFO] Saved:", filename)
    return filename


def generate_frames(memes, landmarker):
    global last_emotion_check, show_meme, current_fingers

    countdown = False
    countdown_start = 0
    captured = False

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        # Face box
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=8,
            minSize=(80, 80), maxSize=(400, 400),
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 100, 100), 1)

        # Hand detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        fingers = 0

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            h_px, w_px = frame.shape[:2]
            pts = [(int(p.x * w_px), int(p.y * h_px)) for p in lm]
            for a, b in MP_HAND_CONNECTIONS:
                cv2.line(display, pts[a], pts[b], (200, 200, 200), 1)
            for pt in pts:
                cv2.circle(display, pt, 3, (100, 180, 255), -1)

            fingers = count_fingers(lm)
            current_fingers = fingers

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
            current_fingers = 0

        # Emotion detection
        if show_meme and (time.time() - last_emotion_check > emotion_check_interval):
            start_emotion_detection(frame)
            last_emotion_check = time.time()

        if show_meme:
            display = overlay_meme(display, memes.get(current_emotion), current_emotion)

        # Countdown
        if countdown:
            elapsed = time.time() - countdown_start
            remaining = max(1, 3 - int(elapsed))
            if elapsed < 3:
                cv2.putText(display, f"{remaining}",
                            (300, 260), cv2.FONT_HERSHEY_SIMPLEX,
                            4, (0, 100, 255), 6)
            elif not captured:
                save_screenshot(display)
                captured = True
                countdown = False

        display = draw_ui(display, current_emotion, current_confidence, fingers)

        _, buffer = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()


memes = load_memes()
download_model()

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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(memes, landmarker),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/screenshots/<filename>")
def screenshot(filename):
    return send_from_directory(SCREENSHOT_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=False, threaded=True)