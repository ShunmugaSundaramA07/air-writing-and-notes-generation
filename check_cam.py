import numpy as np
import cv2

def check_cameras():
    print("Checking for available cameras...")
    available_cameras = []
    for i in range(5):  # Check first 5 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  [FOUND] Camera at index {i}")
                available_cameras.append(i)
            else:
                print(f"  [ERROR] Camera at index {i} opened but could not read frame.")
            cap.release()
        else:
            print(f"  [NOT FOUND] No camera at index {i}")
    
    if not available_cameras:
        print("\n!!! NO FUNCTIONAL CAMERAS DETECTED !!!")
        print("Please ensure your webcam is plugged in and not being used by another application.")
    return available_cameras

if __name__ == "__main__":
    check_cameras()
