import os
import cv2
import numpy as np
import time
import argparse
import sqlite3

import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from keras.models import load_model
from preprocess import preprocess_points_to_28x28

try:
    import pyautogui
except Exception:
    pyautogui = None

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")


def init_db(path):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transcripts(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            text TEXT
        )
    """)
    conn.commit()
    return conn


def is_only_index_up(landmarks, image_h):
    """Detect if only the index finger is raised."""
    lm = landmarks
    index_up = lm[8].y * image_h < lm[6].y * image_h
    middle_up = lm[12].y * image_h < lm[10].y * image_h
    ring_up = lm[16].y * image_h < lm[14].y * image_h
    pinky_up = lm[20].y * image_h < lm[18].y * image_h
    return index_up and not (middle_up or ring_up or pinky_up)


def draw_landmarks_on_frame(frame, landmarks, w, h):
    """Draw hand landmarks and connections on the frame."""
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17)
    ]
    for s, e in connections:
        x1, y1 = int(landmarks[s].x * w), int(landmarks[s].y * h)
        x2, y2 = int(landmarks[e].x * w), int(landmarks[e].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)


def main():
    parser = argparse.ArgumentParser(description="Real-time air-writing character recognition")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.h5)")
    parser.add_argument("--labels", type=str, required=True, help="Path to labels file (.npy)")
    parser.add_argument("--save_db", type=str, default=None,
                        help="Optional: path to SQLite DB to store text stream")
    parser.add_argument("--pause", type=float, default=0.45,
                        help="Pause duration (seconds) before prediction (default: 0.45)")
    parser.add_argument("--confidence", type=float, default=0.50,
                        help="Minimum confidence threshold for accepting predictions (default: 0.50)")
    args = parser.parse_args()

    char_model = load_model(args.model)
    labels = np.load(args.labels)
    n_classes = len(labels)

    conn = init_db(args.save_db) if args.save_db else None

    # Set up HandLandmarker with VIDEO mode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    landmarker = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    points = []
    last_draw_ts = time.time()
    drawing = False
    debug = True
    typed_buffer = ""
    auto_type = False
    frame_ts = 0

    print("[INFO] Controls: SPACE=space, BACKSPACE=delete, ENTER=commit+space, T=toggle autotype, D=debug, ESC=quit")
    print(f"[INFO] Pause timeout: {args.pause}s | Confidence threshold: {args.confidence}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33  # ~30fps increment in ms
        result = landmarker.detect_for_video(mp_image, frame_ts)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            if is_only_index_up(landmarks, h):
                idx = landmarks[8]
                x, y = int(idx.x * w), int(idx.y * h)
                points.append([x, y])
                last_draw_ts = time.time()
                drawing = True
            else:
                drawing = False

            if debug:
                draw_landmarks_on_frame(frame, landmarks, w, h)

        # If paused after drawing -> predict
        if len(points) > 4 and (time.time() - last_draw_ts > args.pause) and not drawing:
            img = preprocess_points_to_28x28(points)  # (28,28,1)
            inp = img[None, ...]
            pred = char_model.predict(inp, verbose=0)[0]
            ci = int(np.argmax(pred))
            conf = float(pred[ci])
            ch = str(labels[ci])
            if conf > args.confidence:
                typed_buffer += ch
                if auto_type and pyautogui is not None:
                    pyautogui.typewrite(ch)
            points = []

        # HUD
        cv2.putText(frame, f"Typed: {typed_buffer[-40:]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"AutoType: {'ON' if auto_type else 'OFF'}  Debug: {'ON' if debug else 'OFF'}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "SPACE=space  BACKSPACE=del  ENTER=commit  T=autotype  D=debug  ESC=quit",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

        # Preview of the current stroke
        if len(points) > 1:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

        cv2.imshow("Air Writing - Recognize", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord(' '):
            typed_buffer += ' '
            if auto_type and pyautogui is not None:
                pyautogui.typewrite(' ')
        elif key == 8:
            if len(typed_buffer) > 0:
                typed_buffer = typed_buffer[:-1]
                if auto_type and pyautogui is not None:
                    pyautogui.press('backspace')
        elif key == ord('\r') or key == 13:
            commit = typed_buffer.strip()
            if commit:
                if auto_type and pyautogui is not None:
                    pyautogui.typewrite('\n')
                if conn is not None:
                    cur = conn.cursor()
                    cur.execute("INSERT INTO transcripts(ts, text) VALUES(datetime('now'), ?)", (commit,))
                    conn.commit()
            typed_buffer += ' '
        elif key == ord('t'):
            auto_type = not auto_type
        elif key == ord('d'):
            debug = not debug

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    if conn is not None:
        conn.close()


if __name__ == "__main__":
    main()