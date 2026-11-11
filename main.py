"""
Motion-Triggered Security Camera with Face Detection
CSC 126 Computer Vision Midterm Project

Techniques used:
  (a) Image Operations   - grayscale, Gaussian blur, thresholding, morphology
  (c) Haar Cascade       - face detection inside motion events
  (e) Motion & Tracking  - background subtraction, contours, centroid tracking

Usage:     python main.py [--source 0 | path/to/video.mp4]
Controls:  q = quit    s = save screenshot
"""

import argparse
import cv2
import math
import os
from datetime import datetime

# ----------------------------- CONFIGURATION -----------------------------
FRAME_WIDTH = 640          # resize frames for speed
MIN_CONTOUR_AREA = 1500    # ignore motion blobs smaller than this (noise)
WARMUP_FRAMES = 30         # let the background model learn before alerting
ALERT_COOLDOWN = 3         # seconds between saved snapshots
SNAPSHOT_DIR = "snapshots"
SCREENSHOT_DIR = "screenshots"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


# ----------------------------- OBJECT TRACKER ----------------------------
class CentroidTracker:
    """Gives each moving object an ID and follows it by nearest centroid."""

    def __init__(self, max_distance=80, max_missing=15):
        self.next_id = 0
        self.objects = {}       # id -> (cx, cy)
        self.missing = {}       # id -> frames since last seen
        self.max_distance = max_distance
        self.max_missing = max_missing

    def update(self, centroids):
        unmatched = list(centroids)

        # Try to match each known object to the closest new centroid
        for oid, (ox, oy) in list(self.objects.items()):
            if unmatched:
                best = min(unmatched, key=lambda c: math.hypot(c[0] - ox, c[1] - oy))
                if math.hypot(best[0] - ox, best[1] - oy) <= self.max_distance:
                    self.objects[oid] = best
                    self.missing[oid] = 0
                    unmatched.remove(best)
                    continue
            # Not matched this frame
            self.missing[oid] += 1
            if self.missing[oid] > self.max_missing:
                del self.objects[oid]
                del self.missing[oid]

        # Any leftover centroids are new objects
        for c in unmatched:
            self.objects[self.next_id] = c
            self.missing[self.next_id] = 0
            self.next_id += 1

        return self.objects


# ----------------------------- HELPER FUNCTIONS --------------------------
def parse_args():
    """Read the video source from the command line."""
    parser = argparse.ArgumentParser(description="Motion-Triggered Security Camera")
    parser.add_argument(
        "--source", default="0",
        help="0 for webcam (default), 1 for second camera, or a video file path"
    )
    args = parser.parse_args()
    # Camera indexes arrive as text, so convert them to numbers
    return int(args.source) if args.source.isdigit() else args.source


def load_face_cascade():
    """Load the Haar cascade that ships with OpenCV."""
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError("Could not load Haar cascade file.")
    return cascade


def preprocess(frame):
    """(a) Image operations: grayscale + Gaussian blur to reduce noise."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    return gray, blurred


def get_motion_mask(blurred, bg_subtractor):
    """(e) Background subtraction, then clean the mask with morphology."""
    mask = bg_subtractor.apply(blurred)
    # Shadows are marked 127 by MOG2; keep only strong foreground (255)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)    # remove small specks
    mask = cv2.dilate(mask, kernel, iterations=3)   # fill gaps in objects
    return mask


def find_moving_objects(mask):
    """(e) Find contours and return bounding boxes of big-enough blobs."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA:
            boxes.append(cv2.boundingRect(cnt))     # (x, y, w, h)
    return boxes


def detect_faces(gray, cascade):
    """(c) Haar cascade face detection."""
    return cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )


def draw_overlay(frame, boxes, tracked, faces, alert, fps_text):
    """Draw boxes, labels, IDs, and the status banner on the frame."""
    # Moving objects (green) + their tracking IDs
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Motion", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for oid, (cx, cy) in tracked.items():
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(frame, f"ID {oid}", (cx + 8, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Faces (red)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Face", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Status banner
    if alert:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (0, 0, 255), -1)
        cv2.putText(frame, "ALERT: MOTION DETECTED" + (" + FACE" if len(faces) else ""),
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (0, 120, 0), -1)
        cv2.putText(frame, "Status: Monitoring - No motion",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, fps_text, (frame.shape[1] - 110, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


# ----------------------------- MAIN PROGRAM ------------------------------
def main():
    source = parse_args()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("ERROR: Cannot open video source:", source)
        return

    face_cascade = load_face_cascade()
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=True
    )
    tracker = CentroidTracker()

    frame_count = 0
    last_alert_time = 0.0
    motion_events = 0
    tick_start = cv2.getTickCount()

    print("Running... press 'q' to quit, 's' to save a screenshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video / no frame received.")
            break

        # Resize to a fixed width, keep aspect ratio
        scale = FRAME_WIDTH / frame.shape[1]
        frame = cv2.resize(frame, (FRAME_WIDTH, int(frame.shape[0] * scale)))
        frame_count += 1

        # 1) Preprocess
        gray, blurred = preprocess(frame)

        # 2) Motion detection
        mask = get_motion_mask(blurred, bg_subtractor)
        boxes = find_moving_objects(mask) if frame_count > WARMUP_FRAMES else []

        # 3) Track moving objects
        centroids = [(x + w // 2, y + h // 2) for (x, y, w, h) in boxes]
        tracked = tracker.update(centroids)

        # 4) Face detection only when there is motion
        alert = len(boxes) > 0
        faces = detect_faces(gray, face_cascade) if alert else []

        # 5) Save a snapshot when an alert fires (with cooldown)
        now = cv2.getTickCount() / cv2.getTickFrequency()
        if alert and (now - last_alert_time) > ALERT_COOLDOWN:
            last_alert_time = now
            motion_events += 1
            name = datetime.now().strftime("alert_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(os.path.join(SNAPSHOT_DIR, name), frame)
            print(f"[{datetime.now():%H:%M:%S}] Motion event #{motion_events} "
                  f"- faces: {len(faces)} - saved {name}")

        # 6) Draw results
        elapsed = (cv2.getTickCount() - tick_start) / cv2.getTickFrequency()
        fps_text = f"FPS: {frame_count / elapsed:.1f}"
        output = draw_overlay(frame.copy(), boxes, tracked, faces, alert, fps_text)

        cv2.imshow("Security Feed", output)
        cv2.imshow("Motion Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            name = datetime.now().strftime("screenshot_%H%M%S.png")
            cv2.imwrite(os.path.join(SCREENSHOT_DIR, name), output)
            cv2.imwrite(os.path.join(SCREENSHOT_DIR, "mask_" + name), mask)
            print("Saved", name)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Total motion events: {motion_events}")


if __name__ == "__main__":
    main()