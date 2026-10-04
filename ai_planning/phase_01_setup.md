# Phase 1: Project Inspection and Architecture

## 1. Goal
Inspect the existing workspace, set up the project folder structure, and create the architectural skeleton for the application.

## 2. Requirements
- Identify existing files.
- Design architecture.
- Create folder structure, config.json, requirements.txt, .gitignore, README skeleton, ai_planning/ documentation.
- Document architecture, module responsibilities, data flow, dependency relationships, and expected inputs/outputs.

## 3. Files changed
- Created directories: `app`, `models`, `videos`, `data`, `logs/entries`, `logs/exits`, `tests`, `ai_planning`.
- Created files: `config.json`, `requirements.txt`, `.gitignore`, `README.md`, `LICENSE`, `ai_planning/phase_01_setup.md`, `app/__init__.py`.

## 4. Architecture decision
The system will follow a modular pipeline architecture where each component is loosely coupled.
- **Video Input:** Handled by OpenCV, supports files and RTSP.
- **Detection:** YOLO-based face detection (e.g., YOLOv8-face).
- **Tracking:** ByteTrack algorithm (via the `supervision` library or custom implementation).
- **Recognition:** InsightFace (ArcFace model) for robust face embeddings.
- **Database:** PostgreSQL for storing visitors and events (ENTRY/EXIT).
- **State Management:** A dedicated VisitorManager and State Machine for managing persistence and entry/exit timeouts.

## 5. Implementation approach
- Separate logic into focused modules (`app/video_input.py`, `app/detector.py`, `app/recognizer.py`, etc.).
- Main orchestration in `app/main.py`.
- Configurations parsed from `config.json`.
- Event logging to file (`events.log`) and image logging to local directories.

## 6. Libraries used
- `opencv-python`: Video processing
- `ultralytics`: YOLO inference
- `insightface`, `onnxruntime`: Face embedding generation
- `supervision`: Tracking (ByteTrack)
- `psycopg`: PostgreSQL database access
- `logging` (built-in): Event logging

## 7. Important functions/classes
- Video Input: `VideoInput` wrapper
- Detection: `FaceDetector`
- Recognition: `FaceRecognizer`
- Tracking: `FaceTracker`
- Visitor Management: `VisitorManager`, `StateMachine`
- DB: `DatabaseManager`
- Loggers: `EventLogger`, `ImageLogger`

## 8. Input
None yet (architectural phase).

## 9. Output
Initialized project skeleton.

## 10. Error handling
Basic structural readiness. Configuration validation will be added.

## 11. Test cases
- Check if folder structure exists.
- Check if configuration file is valid JSON.

## 12. Actual test results
Workspace initially empty, successfully populated with required skeleton.

## 13. Known limitations
No functional pipeline yet.

## 14. Hackathon requirement mapping
- Initial setup and architectural documentation.
