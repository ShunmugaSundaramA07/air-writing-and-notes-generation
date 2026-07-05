import cv2

def test_camera():
    for index in range(3):
        print(f"Checking camera index {index}...")
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"Camera index {index} could not be opened.")
            continue
        
        ret, frame = cap.read()
        if not ret:
            print(f"Camera index {index} opened but could not read a frame.")
            cap.release()
            continue
            
        print(f"Success! Camera index {index} is working. Press 'q' in the window to exit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(f"Camera Test - Index {index}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return

    print("No working camera found on indices 0, 1, or 2.")

if __name__ == "__main__":
    test_camera()
