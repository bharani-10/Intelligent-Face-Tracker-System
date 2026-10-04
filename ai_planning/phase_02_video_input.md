# Phase 2: Video Input

## 1. Goal
Implement a robust video input layer that reads video frames using OpenCV, supporting both local files and RTSP streams interchangeably, while remaining entirely decoupled from any downstream AI logic.

## 2. Requirements
- Implement `app/video_input.py` supporting local videos and RTSP.
- Expose methods: `open`, `read`, `is_opened`, `release`.
- Expose properties: `width`, `height`, `fps`, `frame_count`.
- Handle errors gracefully: invalid files, missing paths, connection failures.
- Avoid logging RTSP passwords.
- Provide a simple test in `app/main.py`.

## 3. Files changed
- Created `app/video_input.py`.
- Created `app/main.py`.
- Modified `config.json` (no changes needed since it already has `videos/sample.mp4`).

## 4. Architecture decision
The `VideoInput` class acts as an abstraction layer over OpenCV's `VideoCapture`. The rest of the pipeline will request frames from `VideoInput` without caring whether the source is a static file or a live camera stream.

## 5. Implementation approach
- Use `cv2.VideoCapture`.
- Pass `source_type` and `source` from configuration.
- Implement property decorators for video metadata.
- Sanitize URLs in logging using `urllib.parse` to strip out passwords from RTSP URLs if present.
- Build a lightweight loop in `main.py` that reads frames and displays them using `cv2.imshow`, falling back cleanly in headless environments.

## 6. Libraries used
- `cv2` (OpenCV): Video decoding.
- `urllib.parse`: URL sanitization for logs.
- `logging`: Event logging.

## 7. Important functions/classes
- `VideoInput`: Wraps `cv2.VideoCapture` and exposes state.
- `VideoInput._sanitize_source_for_logging`: Masks `password` in `rtsp://user:password@ip...` URIs.
- `main()` in `main.py`: Orchestrates the testing loop.

## 8. Input
Video files or RTSP streams configured in `config.json`. (Using a dummy generated video for testing).

## 9. Output
A stream of frames (numpy arrays) alongside video metadata (resolution, FPS). Logging outputs into `logs/events.log`.

## 10. Error handling
- Checking `os.path.exists` for local files.
- Safe-failing `cap.isOpened()` when RTSP fails.
- Safe logging of connections (password redaction).
- Catching display errors in headless environments.

## 11. Test cases
- Run `app.main` with a valid local video. Ensure frames are read and metadata printed.
- Run `app.main` with a missing file. Ensure it exits gracefully with an error log.

## 12. Actual test results
A dummy 120-frame video was generated to test the local environment. `main.py` successfully opened the dummy video, printed dimensions (640x480) and FPS (30.0), and smoothly read 100 frames before safely exiting as expected.

## 13. Known limitations
The real video from Google Drive needs to be downloaded manually due to automated browser tool constraints with Google Drive links.

## 14. Hackathon requirement mapping
- "Support local video files. Design the interface so RTSP can be supported later."
- "Expose: open, read, is_opened, release, width, height, FPS, frame count."
- "Do not expose or log RTSP passwords."
- "Create a simple test in main.py."
