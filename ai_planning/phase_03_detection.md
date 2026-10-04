# Phase 3: YOLO Face Detection

## 1. Goal
Integrate a face-specific YOLO model (`yolov8n-face`) to detect faces in individual video frames, without processing tracking or recognition yet. 

## 2. Requirements
- Implement `app/detector.py`.
- Load YOLO model only once.
- Accept OpenCV frames and return structured bounding boxes (`[x1, y1, x2, y2]`) and confidences.
- Configurable confidence threshold.
- Handle frames with no detections.
- Visualization (drawing boxes on frames).

## 3. Files changed
- Created `app/detector.py`.
- Modified `app/main.py`.
- Created `ai_planning/phase_03_detection.md`.

## 4. Architecture decision
The detector is isolated in `FaceDetector`. It only cares about single frames and has no temporal awareness (no tracking). We explicitly use a `yolov8n-face.pt` model instead of a generic YOLOv8 model, because generic YOLO models are optimized for detecting "person" (the entire body) and might miss cropped faces, which is detrimental to the ArcFace recognition phase later.

## 5. Implementation approach
- `FaceDetector` initializes and caches the YOLO model during instantiation.
- An initial "warmup" inference is run on a dummy frame so the first real frame doesn't lag.
- The `detect` method processes the frame and formats the `ultralytics` results into a clean list of dictionaries.
- `draw_detections` provides visual testing capability.
- `main.py` is updated to download the pretrained `yolov8n-face.pt` model if it doesn't exist.

## 6. Libraries used
- `ultralytics`: YOLOv8 framework.
- `cv2`, `numpy`: Image processing.

## 7. Important functions/classes
- `FaceDetector`: Encapsulates YOLO.
- `FaceDetector.detect()`: Core inference loop.
- `FaceDetector.draw_detections()`: Visualization helper.

## 8. Input
- OpenCV video frames (NumPy arrays).
- Configuration values (`model_path`, `conf_threshold`).

## 9. Output
- List of dictionaries: `[{"bbox": [x1, y1, x2, y2], "confidence": float, "class_id": int}]`.

## 10. Error handling
- Checking if frame is None or empty.
- Graceful exception handling if the `.pt` model file is corrupted or missing.
- Catching exceptions during model initialization.

## 11. Test cases
- Run the video feed through the detector.
- Ensure bounding boxes are drawn on faces.
- Ensure the application doesn't crash on frames with zero faces.

## 12. Actual test results
Tested on local video sample. The model downloaded successfully, warmed up, and processed the video frame-by-frame, logging the number of faces detected and outputting clean JSON-like detections.

## 13. Known limitations
The current detector finds faces independently on every frame. Without tracking (Phase 7), a face on Frame 1 and the same face on Frame 2 are completely unrelated to the system.

## 14. Hackathon requirement mapping
- "Use a YOLO-based model for real-time face detection."
- "Output format should be clean and consistent."
- "Load model only once."
