import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import argparse

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_features(landmarks):
    # Extract 21 points
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    # Make translation invariant by subtracting wrist (point 0)
    pts = pts - pts[0]
    # Make scale invariant by dividing by distance from wrist to middle finger MCP (point 9)
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        scale = 1.0
    pts = pts / scale
    return pts.flatten().tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, required=True, help="Label for the gesture (e.g., 'Hello')")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to collect")
    parser.add_argument("--out", type=str, default="gestures.csv", help="Output CSV file")
    args = parser.parse_args()

    out_file = os.path.join(os.path.dirname(__file__), args.out)
    
    # Write header if file doesn't exist
    if not os.path.exists(out_file):
        with open(out_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ["label"] + [f"p{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']]
            writer.writerow(header)

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    
    samples_collected = 0
    print(f"Collecting {args.num_samples} samples for '{args.label}'. Press SPACE to start capturing.")
    
    capturing = False
    while cap.isOpened() and samples_collected < args.num_samples:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        result = hands.process(rgb)
        
        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            
            if capturing:
                features = extract_features(landmarks)
                with open(out_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([args.label] + features)
                samples_collected += 1
                
        cv2.putText(frame, f"Label: {args.label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {samples_collected}/{args.num_samples}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if not capturing:
            cv2.putText(frame, "Press SPACE to start", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        cv2.imshow("Collect Gestures", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break
        elif key == ord(' '):
            capturing = True

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"[DONE] Collected {samples_collected} samples for {args.label} -> {out_file}")

if __name__ == "__main__":
    main()
