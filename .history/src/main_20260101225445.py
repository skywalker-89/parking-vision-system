import cv2
import os

# CONFIGURATION
# ---------------------------------------------------------
VIDEO_PATH = os.path.join("videos", "demo.mp4")
WINDOW_NAME = "Parking Vision System (Person 3)"
# ---------------------------------------------------------


def main():
    # 1. Load the Video
    # -----------------------------------------------------
    print(f"Attempting to load video from: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Safety Check: Did the video load?
    if not cap.isOpened():
        print(f"❌ Error: Could not open video. Check path: {VIDEO_PATH}")
        return
    else:
        print(f"✅ Video loaded successfully. Press 'q' to quit.")

    # 2. The Frame Loop (The "Heartbeat" of the system)
    # -----------------------------------------------------
    while True:
        # Read a single frame
        success, frame = cap.read()

        # If no frame is read, the video is over or broken
        if not success:
            print("End of video stream or cannot fetch frame.")
            break

        # -------------------------------------------------
        # 🚧 TODO: Person 1 will plug Detection here
        # detections = detector.detect(frame)
        # -------------------------------------------------

        # -------------------------------------------------
        # 🚧 TODO: Person 2 will plug Logic here
        # status = parking_logic.check_parking(detections)
        # -------------------------------------------------

        # -------------------------------------------------
        # 🚧 TODO: You (Person 3) will plug Visualization here
        # frame = visualization.draw(frame, detections)
        # -------------------------------------------------

        # OPTIONAL: Resize frame if CCTV footage is too big for your screen
        # frame = cv2.resize(frame, (1280, 720))

        # 3. Display the Frame
        cv2.imshow(WINDOW_NAME, frame)

        # 4. Keyboard Controls (Press 'q' to exit)
        # waitKey(1) controls playback speed.
        # 1 = 1ms delay (fast as possible), 25 = approx real-time for 30fps
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("User requested exit.")
            break

    # 5. Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("System Shutdown.")


if __name__ == "__main__":
    main()
