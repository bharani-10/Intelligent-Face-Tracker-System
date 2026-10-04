# Intelligent Face Tracker with Auto-Registration and Visitor Counting

## 1. Project Overview

This project detects faces in a local video or RTSP stream, tracks them with ByteTrack, generates InsightFace/ArcFace embeddings, registers persistent visitor identities, and records ENTRY/EXIT events with local images and PostgreSQL metadata.

## 2. Problem Statement

Frame counts and temporary tracker IDs cannot identify a returning person. The system separates persistent `visitor_id` identity from temporary `track_id` tracking state.

## 3. Objective

Automatically register new people, recognize returning visitors, create exactly one ENTRY and EXIT per visit, and maintain a database-backed unique visitor count.

## 4. Features

- YOLO face detection with configurable confidence and frame skipping.
- InsightFace `buffalo_l` ArcFace embeddings.
- PostgreSQL visitor and event persistence.
- ByteTrack temporary tracking IDs.
- Cosine-similarity re-identification.
- Missing-frame ENTRY/EXIT state management.
- Timestamped local entry and exit images.
- `logs/events.log` file logging.
- Local MP4 and RTSP input abstraction.
- Streamlit dashboard backed by PostgreSQL.

## 5. Architecture

`VideoInput` supplies frames. `FaceDetector` produces boxes. `FaceTracker` assigns temporary IDs. `FaceRecognizer` crops faces and generates embeddings. `VisitorManager` matches persistent IDs and manages visits. `ImageLogger` stores images. `DatabaseManager` owns all PostgreSQL queries. Streamlit only reads the service/database layer.

## 6. Architecture Diagram

```mermaid
flowchart TD
    A[MP4 or RTSP] --> B[VideoInput]
    B --> C[Frame Scheduler]
    C --> D[YOLO FaceDetector]
    D --> E[ByteTrack]
    E --> F[Face Crop]
    F --> G[InsightFace ArcFace]
    G --> H[Cosine Similarity]
    H --> I[VisitorManager]
    I --> J[ENTRY or EXIT State]
    J --> K[Local ImageLogger]
    J --> L[PostgreSQL DatabaseManager]
    I --> M[events.log]
    L --> N[Streamlit Dashboard]
```

## 7. Complete Data Flow

1. Open the configured local file or RTSP source.
2. Run YOLO on scheduled frames.
3. Keep tracker output between skipped detection cycles.
4. Crop new tracks and generate ArcFace embeddings.
5. Compare embeddings with PostgreSQL records.
6. Reuse a matching `visitor_id` or register a new visitor.
7. Create one ENTRY when a visitor becomes present.
8. Create EXIT only after the missing-frame timeout.
9. Save an image locally and insert event metadata in PostgreSQL.

## 8. Technology Stack

Python, OpenCV, Ultralytics YOLO, InsightFace/ArcFace, ONNX Runtime, supervision ByteTrack, PostgreSQL, psycopg, NumPy, Streamlit, and Python logging.

## 9. Folder Structure

```text
app/                 Backend runtime modules
frontend/app.py      Streamlit dashboard
ai_planning/         Phase planning documents
logs/entries/        Entry images by date
logs/exits/          Exit images by date
logs/events.log      Application log
models/              YOLO model files
tests/               Automated tests
videos/              Local development videos
config.json          Runtime configuration
.env.example         PostgreSQL URL template
requirements.txt     Python dependencies
```

## 10. Setup

Use the project virtual environment from the repository root. The current machine has PostgreSQL service `postgresql-x64-18` running, but valid credentials must be supplied by the developer.

## 11. Installation

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

InsightFace may download its model pack on first use. The configured YOLO model must exist at `models/yolov8n-face.pt`.

## 12. PostgreSQL Setup

Create a database and role using pgAdmin or a PostgreSQL administration shell. Then set a connection URL in the environment:

```powershell
$env:DATABASE_URL = "postgresql://username:password@localhost:5432/visitor_tracker"
```

The application creates the `visitors` and `events` tables automatically. The local `.env` is ignored and is used only on this machine; no password is committed to the repository.

## 13. Environment Variables

Copy `.env.example` to a local environment mechanism and set a real value:

```text
DATABASE_URL=postgresql://username:password@localhost:5432/visitor_tracker
```

`.env` is ignored by Git and is loaded automatically with `python-dotenv`.

## 14. Configuration

All important runtime values are loaded from `config.json`. Startup validates source type, confidence, threshold, frame skip, tracking age, and paths.

```json
{
  "input": { "source_type": "file", "source": "videos/video_sample1.mp4" },
  "detection": { "model": "models/yolov8n-face.pt", "confidence": 0.5, "frame_skip": 2 },
  "recognition": { "threshold": 0.4 },
  "tracking": { "max_age": 30 },
  "database": { "url_env": "DATABASE_URL" },
  "logging": { "path": "logs/events.log" }
}
```

## 15. Google Drive Video Setup

Download one representative face-containing video manually from the provided source:

https://drive.google.com/drive/folders/15YCN3CYb97GyIoNUV6NJxGNIf-rUBFUJ?usp=sharing

Place it under `videos/` and set `input.source` to its actual local filename. The runtime does not depend on Google Drive.

## 16. Available Video Audit

The configured file is `videos/sample.mp4`: 120 frames, 640×480, 30 FPS. YOLO produced zero detections at sampled frames, confirming it is a dummy/empty development video.

The repository also contains `videos/video_sample1.mp4`: 240 frames, 3840×2160, approximately 29.97 FPS. YOLO produced 8, 13, 16, 16, and 13 detections at sampled checkpoints. The full pipeline processed all 240 frames with PostgreSQL enabled.

## 17. Running the Backend

After setting `DATABASE_URL`:

```powershell
python -m app.main
```

Models are loaded once. Video, database, logging, and display resources are released during shutdown.

## 18. Running the Dashboard

```powershell
streamlit run frontend/app.py
```

The dashboard reads actual PostgreSQL values and provides Start/Stop controls that launch the existing backend pipeline. It does not duplicate detection or recognition logic.

The dashboard also supports:

- Dynamic selection of supported sample videos from `videos/`.
- Uploading MP4, AVI, MOV, or MKV files.
- UUID-based upload storage under `data/uploads/` without overwriting samples.
- Asynchronous `Start Processing` execution through `app.main`.
- Per-run JSON statistics under `data/runs/`.
- Recent events filtered to the current run.
- Actual PostgreSQL visitor totals and event totals.
- Entry and exit image galleries from the local image store.
- Read-only display of detection, recognition, frame-skip, and timeout settings.
- `New Test` reset of the current dashboard session without deleting persistent visitor identities.

The dashboard shows global persistent visitor totals and also reports current-run event deltas, so results from separate tests are not silently mixed.

## 19. YOLO Detection

`FaceDetector` loads the configured face model once, warms it up, applies the confidence threshold, and returns bounding boxes, confidence, and class ID.

## 20. InsightFace Recognition

`FaceRecognizer` uses InsightFace `buffalo_l` and ArcFace embeddings. It crops YOLO boxes with padding and rejects invalid or very small crops. CPU execution is currently selected by default.

## 21. Tracking

ByteTrack assigns temporary `track_id` values. A `track_id` is not a persistent visitor identity. Skipped detection frames retain the last tracker output; actual empty detection cycles advance the tracker and allow tracks to expire.

## 22. Auto Registration

If the best cosine similarity is below the configured threshold, a new visitor ID is inserted with its embedding and timestamps. Matching embeddings reuse the existing `visitor_id` even when the `track_id` changes.

## 23. ENTRY/EXIT State Machine

A visitor enters `PRESENT` once and receives one ENTRY. Temporary misses become `MISSING` and do not immediately create EXIT. EXIT is emitted only after `missing_frames > tracking.max_age`. A later appearance creates another ENTRY for the same visitor ID.

## 24. Unique Visitor Counting

The `visitors` table is the source of truth. The count does not increase for repeated frames, repeated visits, or changed tracker IDs. PostgreSQL queries also expose total entries, total exits, currently present visitors, and visitor history.

## 25. PostgreSQL Schema

`visitors`:

- `id BIGSERIAL PRIMARY KEY`
- `visitor_id TEXT UNIQUE`
- `embedding BYTEA`
- `first_seen TIMESTAMPTZ`
- `last_seen TIMESTAMPTZ`

`events`:

- `event_id BIGSERIAL PRIMARY KEY`
- `visitor_id` foreign key
- `event_type` with ENTRY/EXIT check
- `timestamp TIMESTAMPTZ`
- `image_path TEXT`
- optional `track_id`

Visitors and events are separate because one persistent identity can have many visits and events while its embedding is stored once.

## 26. Image Storage

Images are stored locally under:

```text
logs/entries/YYYY-MM-DD/
logs/exits/YYYY-MM-DD/
```

Filenames include visitor ID, event type, timestamp, and a UUID suffix. PostgreSQL stores the image path, not the JPEG bytes.

## 27. Logging

`logs/events.log` records startup, video status, model setup, detections, embeddings, recognition, registration, tracking, ENTRY, EXIT, database errors, image errors, and shutdown/benchmark information.

## 28. Frame Skipping

With `frame_skip = N`, detection runs on frames 1, `1 + N`, `1 + 2N`, and so on. Larger values reduce YOLO cost but can reduce motion accuracy and increase missed detections. ByteTrack continues using the last tracked output between detection cycles.

## 29. Performance Benchmark

Measured during the full run on `video_sample1.mp4`:

| Metric | Result |
|---|---:|
| Frames processed | 240 |
| Input FPS | 29.97 |
| Processing FPS | 0.64 |
| Total runtime | 373.39 seconds |
| Average YOLO time | 105.4 ms |
| Average InsightFace time | 3260.9 ms |
| Unique visitors | 28 |

InsightFace embedding generation was the measured bottleneck. CPU/RAM/GPU telemetry remains TODO.

## 30. Testing

```powershell
python -m pytest -q
```

Current result with the configured local PostgreSQL service: `6 passed`. Tests cover schema creation, embedding persistence, event insertion, retrieval, counts, configuration validation, and visitor state behavior.

## 31. Security

Do not commit `.env`, PostgreSQL passwords, RTSP credentials, API keys, or tokens. `.env.example` contains placeholders only. RTSP passwords are masked by the video input logging path.

## 32. Limitations

- RTSP reconnection behavior is not implemented as a configurable retry loop.
- Dedicated CPU/RAM/GPU telemetry is not implemented.
- Threshold quality depends on camera angle, lighting, crop quality, and validation data.

## 33. Demo Video

TODO: add a Loom or YouTube demonstration URL before submission.

## AI Planning

- **Problem understanding:** Persistent visitor identity must remain separate from temporary track identity.
- **Requirements:** YOLO, ArcFace, ByteTrack, PostgreSQL, automatic registration, ENTRY/EXIT, images, logs, counting, configuration, RTSP, dashboard, tests, and benchmark reporting.
- **Architecture decisions:** Focused modules with `main.py` as coordinator, PostgreSQL isolated in `app/database.py`, and UI isolated in `frontend/app.py`.
- **Model choices:** Face-specific YOLO for boxes and InsightFace `buffalo_l` ArcFace for embeddings.
- **Database choice:** PostgreSQL provides transactional persistent storage and supports the required dashboard queries.
- **Tracking choice:** ByteTrack provides lightweight temporary IDs and missing-track aging.
- **Frame skipping strategy:** Detect on a predictable schedule while retaining tracker output between detection cycles.
- **Testing strategy:** Dependency-light state tests plus PostgreSQL integration tests that require `DATABASE_URL`.
- **Performance optimization:** Model warmup, frame skipping, one embedding per track, and lightweight tracking.
- **Limitations:** The real PostgreSQL credentials, end-to-end face video run, RTSP test, and full telemetry are still pending.

## Submission Checklist

- [x] Phases 1-4 modules inspected and retained
- [x] YOLO, InsightFace, and ByteTrack modules present
- [x] PostgreSQL database abstraction implemented
- [x] Environment-based database configuration
- [x] Streamlit dashboard added
- [x] Image and event logging modules present
- [x] Configuration and state tests present
- [ ] Supply valid `DATABASE_URL`
- [ ] Run PostgreSQL integration tests
- [ ] Run full pipeline on `video_sample1.mp4`
- [ ] Capture actual visitor/event/image output
- [ ] Add CPU/RAM/GPU telemetry
- [ ] Test RTSP with an available camera
- [ ] Add demo video URL

This project is a part of a hackathon run by [https://katomaran.com](https://katomaran.com/)