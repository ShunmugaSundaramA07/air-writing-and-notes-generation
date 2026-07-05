import os
import cv2
import numpy as np
import time
import argparse

import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from preprocess import preprocess_points_to_28x28

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
os.makedirs(DATASET_DIR, exist_ok=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")


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
    # Draw landmarks as circles
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    # Draw connections
    connections = [
        (0,1),(1,2),(2,3),(3,4),      # thumb
        (0,5),(5,6),(6,7),(7,8),      # index
        (5,9),(9,10),(10,11),(11,12), # middle
        (9,13),(13,14),(14,15),(15,16), # ring
        (13,17),(17,18),(18,19),(19,20), # pinky
        (0,17)
    ]
    for s, e in connections:
        x1, y1 = int(landmarks[s].x * w), int(landmarks[s].y * h)
        x2, y2 = int(landmarks[e].x * w), int(landmarks[e].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)


def ensure_label_folder(label):
    label = label.upper()
    folder = os.path.join(DATASET_DIR, label)
    os.makedirs(folder, exist_ok=True)
    return folder


def main():
    parser = argparse.ArgumentParser(description="Collect air-writing dataset samples")
    parser.add_argument("--pause", type=float, default=0.4,
                        help="Pause duration (seconds) before generating preview (default: 0.4)")
    args = parser.parse_args()

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
    current_label = "A"
    points = []
    last_draw_ts = time.time()
    drawing = False
    save_preview = None
    frame_ts = 0

    print("[INFO] Controls:")
    print(" - Press A..Z or 0..9 to select current label")
    print(" - Raise ONLY index finger to draw (pen down). Lower to lift (pen up).")
    print(" - Press / to save current sample")
    print(" - Press ] to discard sample")
    print(" - ESC to quit")

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

            draw_landmarks_on_frame(frame, landmarks, w, h)

        # Auto-generate preview when paused after drawing
        if len(points) > 4 and (time.time() - last_draw_ts > args.pause) and not drawing:
            img = preprocess_points_to_28x28(points)
            preview = (img[..., 0] * 255).astype(np.uint8)
            save_preview = preview
            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
            preview_bgr = cv2.resize(preview_bgr, (140, 140), interpolation=cv2.INTER_NEAREST)
            frame[10:150, 10:150] = preview_bgr
        elif save_preview is not None:
            preview_bgr = cv2.cvtColor(save_preview, cv2.COLOR_GRAY2BGR)
            preview_bgr = cv2.resize(preview_bgr, (140, 140), interpolation=cv2.INTER_NEAREST)
            frame[10:150, 10:150] = preview_bgr

        # HUD
        cv2.putText(frame, f"Label: {current_label}", (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Controls: [A..Z/0..9]=label  /=save  ]=new  ESC=quit", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Collect Dataset - AirWriting", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        if key in range(ord('a'), ord('z') + 1) or key in range(ord('A'), ord('Z') + 1):
            current_label = chr(key).upper()
            ensure_label_folder(current_label)
        elif key in range(ord('0'), ord('9') + 1):
            current_label = chr(key)
            ensure_label_folder(current_label)
        elif key == ord(']'):
            points = []
            save_preview = None
        elif key == ord('/'):
            if save_preview is not None:
                folder = ensure_label_folder(current_label)
                count = len([f for f in os.listdir(folder) if f.lower().endswith(".png")])
                out_path = os.path.join(folder, f"img_{count + 1:05d}.png")
                cv2.imwrite(out_path, save_preview)
                print(f"[SAVED] {out_path}")
                points = []
                save_preview = None

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()