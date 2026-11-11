# Motion-Triggered Security Camera with Face Detection

## Members
- Demart D. Cogtas
- Eleanore Barata
- Queenith Jean Bigaran

CSC 126 Computer Vision Midterm Project


## Description
A computer vision application that monitors a live webcam feed or a recorded
video. It detects motion using background subtraction, tracks each moving
object with an ID, and runs Haar cascade face detection whenever motion is
found. When an alert is triggered, the app labels the frame, shows the motion
mask, and saves a snapshot with a timestamp.

**Real-world use case:** a simple home or room security monitor that alerts
when someone enters the view and checks whether a face is visible.

## OpenCV Techniques Used
- **(a) Image Operations:** grayscale conversion, Gaussian blur, thresholding,
  erosion and dilation (morphology)
- **(c) Haar Cascade Classifier:** frontal face detection with `detectMultiScale`
- **(e) Motion and Object Tracking:** MOG2 background subtraction, contour
  detection, bounding boxes, and a centroid-based tracker that assigns IDs

## Features
- Works with a live webcam or a recorded video file
- Green boxes for motion, red boxes for faces, blue dots with tracking IDs
- Status banner (Monitoring / ALERT), timestamp, and FPS counter
- Separate window showing the motion mask
- Automatic snapshots saved to `snapshots/` when an alert fires

## Requirements
- Python 3.9 or newer
- A webcam (optional, if you use a video file)

## Setup
```
git clone https://github.com/demartcogtas/motion-security-cam.git
cd motion-security-cam
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run
```
python main.py                              # live webcam (default)
python main.py --source samples/test.mp4    # recorded video
```

**Controls:** press `q` to quit, `s` to save a screenshot to `screenshots/`.

## Project Structure
```
motion-security-cam/
├── main.py              # main application
├── requirements.txt     # dependencies
├── README.md
├── samples/             # sample input video
├── screenshots/         # output screenshots
└── snapshots/           # auto-saved alert snapshots
```

## Settings
These can be adjusted at the top of `main.py`:
- `MIN_CONTOUR_AREA`: minimum size of motion to count (raise to reduce false alerts)
- `WARMUP_FRAMES`: frames used to learn the background
- `ALERT_COOLDOWN`: seconds between saved snapshots

## Limitations
- Lighting changes and shadows can cause false motion.
- Haar cascades only detect front-facing faces and can miss tilted or turned faces.
- Fast-moving or overlapping people can swap tracking IDs.



