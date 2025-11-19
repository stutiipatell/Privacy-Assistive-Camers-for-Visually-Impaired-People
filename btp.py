
btp>btp_env\Scripts\activate

import cv2
from deepface import DeepFace
import tempfile
import os
import logging
import contextlib
import sys
import time
import numpy as np
import threading
import queue
import pytesseract

# -----------------------------
# TESSERACT CONFIGURATION
# -----------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Suppress TensorFlow and DeepFace logs
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('deepface').setLevel(logging.ERROR)

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Temporarily suppress console output from noisy libraries."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# -----------------------------
# CONFIGURATION
# -----------------------------
USE_WEBCAM = False
VIDEO_PATH = "http://192.168.29.186:8080/video"
DARKNESS_THRESHOLD = 50
PROCESS_EVERY_N_FRAMES = 5
ANNOUNCE_COOLDOWN = 5     # seconds per person
DOCUMENT_COOLDOWN = 10    # seconds

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Queue for announcements
announce_queue = queue.Queue()

# Store last announcement times per person/document
last_announced_times = {}
lock = threading.Lock()

# -----------------------------
# WINDOWS TEXT-TO-SPEECH (TTS)
# -----------------------------
def speak(text: str):
    try:
        os.system(
            f'powershell -Command "Add-Type –AssemblyName System.Speech; '
            f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\');"'
        )
    except Exception as e:
        print(f"[TTS ERROR] {e}")

def tts_worker():
    """Background thread that speaks announcements in order."""
    global last_announced_times
    while True:
        msg = announce_queue.get()
        if msg == "QUIT":
            announce_queue.task_done()
            break
        try:
            key = "document" if "document" in msg.lower() or "card" in msg.lower() else msg.split(" ")[0]
            now = time.time()

            # Always speak documents immediately, cooldown only for faces
            if "document" in key or "card" in key:
                print(f"[TTS] Speaking (document): {msg}")
                speak(msg)
                with lock:
                    last_announced_times[key] = now

            elif (key not in last_announced_times) or (now - last_announced_times[key] > ANNOUNCE_COOLDOWN):
                print(f"[TTS] Speaking: {msg}")
                speak(msg)
                with lock:
                    last_announced_times[key] = now
            else:
                print(f"[TTS] Skipped (cooldown active) for {key}")

        except Exception as e:
            print(f"[TTS ERROR] {e}")
        announce_queue.task_done()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# -----------------------------
# VIDEO SOURCE
# -----------------------------
if USE_WEBCAM:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Using webcam...")
else:
    cap = cv2.VideoCapture(VIDEO_PATH)
    print(f"Using video stream: {VIDEO_PATH}")

if not cap.isOpened():
    print("Cannot open video source")
    exit()

cv2.namedWindow("Face & Document Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face & Document Recognition", 800, 600)

frame_count = 0
print("Starting face and document/card recognition...")

# -----------------------------
# MAIN LOOP
# -----------------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % PROCESS_EVERY_N_FRAMES != 0:
            continue

        frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < DARKNESS_THRESHOLD:
            continue

        frame_h, frame_w = frame.shape[:2]

        # -----------------------------
        # FACE DETECTION & RECOGNITION
        # -----------------------------
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                x_full, y_full, w_full, h_full = x*2, y*2, w*2, h*2
                face_img = frame[y_full:y_full+h_full, x_full:x_full+w_full]
                name = None
                temp_filename = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                        temp_filename = tmp_file.name
                        cv2.imwrite(temp_filename, face_img)

                    with suppress_stdout_stderr():
                        result = DeepFace.find(img_path=temp_filename, db_path="face_db", enforce_detection=True)

                    if isinstance(result, list) and len(result) > 0 and not result[0].empty:
                        identity_path = result[0]['identity'].values[0]
                        name = os.path.basename(os.path.dirname(identity_path))
                except Exception as e:
                    print(f"[DeepFace ERROR] {e}")
                finally:
                    if temp_filename and os.path.exists(temp_filename):
                        os.remove(temp_filename)

                if name and name.lower() != "unknown":
                    cv2.rectangle(frame, (x_full, y_full), (x_full+w_full, y_full+h_full), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x_full, y_full - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    face_center_x = x_full + w_full // 2
                    if face_center_x < frame_w / 3:
                        horiz = "left"
                    elif face_center_x > 2 * frame_w / 3:
                        horiz = "right"
                    else:
                        horiz = "center"
                    message = f"{name} is on the {horiz}"
                    print(f"[MAIN] Detected {name}, enqueueing announcement: {message}")
                    announce_queue.put(message)

        # -----------------------------
        # ROBUST DOCUMENT / CARD DETECTION (LENIENT)
        # -----------------------------
        try:
            gray_doc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray_doc, (5,5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), iterations=1)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            frame_area = frame_h * frame_w
            doc_detected = False
            detected_text = False

            # Sort contours by area, largest first
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 0.01*frame_area or area > 0.9*frame_area:
                    continue

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02*peri, True)

                # Allow polygons with 4–8 sides
                if len(approx) < 4 or len(approx) > 8:
                    continue

                hull = cv2.convexHull(cnt)
                solidity = float(area)/cv2.contourArea(hull)

                if 0.3 < solidity <= 1.0:  # lenient solidity check
                    x, y, w, h = cv2.boundingRect(approx)
                    cv2.drawContours(frame, [approx], -1, (255,0,0), 2)

                    roi = frame[y:y+h, x:x+w]
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi_gray = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    text = pytesseract.image_to_string(roi_gray, config="--psm 6").strip()

                    if len(text.split()) >= 2 or len(text) > 5:  # lenient text detection
                        detected_text = True

                    doc_detected = True
                    break  # Only detect largest valid document

            now = time.time()
            if doc_detected:
                with lock:
                    last_time = last_announced_times.get("document", 0)
                    if now - last_time > DOCUMENT_COOLDOWN:
                        message = "A clear document or ID card detected." if detected_text else "A document or ID card detected."
                        print(f"[DOC] {message}")
                        announce_queue.put(message)
                        last_announced_times["document"] = now

        except Exception as e:
            print(f"[DOC ERROR] {e}")

        # -----------------------------
        # DISPLAY FRAME
        # -----------------------------
        cv2.imshow("Face & Document Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break	

finally:
    print("Exiting...")
    cap.release()
    cv2.destroyAllWindows()
    announce_queue.put("QUIT")
    announce_queue.join()
    tts_thread.join()
    print("Exited cleanly.")
