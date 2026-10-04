# Phase 4: Face Embedding Generation

## 1. Goal
Generate a numerical facial embedding for each detected face using InsightFace / ArcFace, while preserving the YOLO detection pipeline.

## 2. Requirements
- Implement `app/recognizer.py`.
- Do NOT use the `face_recognition` library.
- Load the InsightFace model exactly once.
- Accept a cropped face image and return a normalized numerical embedding.
- Handle small/low-quality crops without crashing.
- Write a test in `main.py` that confirms the dimension of the embedding (usually 512).

## 3. Files changed
- Created `app/recognizer.py`.
- Modified `app/main.py`.
- Created `ai_planning/phase_04_recognition.md`.

## 4. Concepts Explained
- **What is a Face Embedding?**
  A face embedding is a high-dimensional mathematical vector (an array of numbers) that represents the unique features of a face. Think of it as a "digital fingerprint" for a human face.
- **Why not compare raw images?**
  Raw pixels change drastically due to lighting, angle, makeup, and camera quality. It is mathematically impossible to consistently identify a person by directly comparing pixel RGB values.
- **What does ArcFace do?**
  ArcFace is a State-Of-The-Art (SOTA) neural network architecture designed to project images of faces into a 512-dimensional sphere. It maps pictures of the *same* person very close to each other, and pictures of *different* people far apart.
- **Cosine Similarity:**
  Because ArcFace maps faces onto a sphere and automatically normalizes the vectors, we compare two faces by measuring the angle between their vectors. If the angle is very small (cosine similarity approaches 1.0), it's the same person.
- **Recognition Threshold:**
  If the similarity between two embeddings is higher than the threshold (e.g. `0.4`), we confidently conclude it is the same person. If it is lower, it's a new person.

## 5. Implementation approach
- `FaceRecognizer` initializes the InsightFace `buffalo_l` pack.
- We use the `crop_face` method to crop the YOLO bounding box with an extra 20% padding. Padding ensures that InsightFace has enough visual context to find the facial landmarks (eyes, nose, mouth) required to properly align the face before feeding it to ArcFace.
- `get_embedding` passes the crop to InsightFace, returning a 512-D numpy array.

## 6. Libraries used
- `insightface`: SOTA Face Recognition.
- `numpy`: Array manipulations.
- `cv2`: Image cropping.

## 7. Important functions/classes
- `FaceRecognizer`: Wrapper class.
- `FaceRecognizer.crop_face`: Crops the frame using YOLO's `x1, y1, x2, y2` coordinates with padding.
- `FaceRecognizer.get_embedding`: Executes ArcFace on the padded crop.

## 8. Input
A full `frame` and a `bbox` from YOLO.

## 9. Output
A normalized numpy array of shape `(512,)`.

## 10. Error handling
- Very small crops (`< 20px`) are immediately discarded to prevent garbage embeddings.
- If InsightFace's internal aligner fails to find facial landmarks in a bad crop, it returns `None`.

## 11. Test cases
- Run the video feed through the pipeline.
- Assert that when YOLO detects a face, InsightFace generates an embedding of shape `(512,)`.

## 12. Actual test results
The test successfully loaded both YOLO and InsightFace. When run against an image containing a face, it printed `Successfully generated embedding. Shape: (512,)`. 

## 13. Known limitations
Generating embeddings for every face on every frame is extremely slow. We are doing it for the test now, but later (in Phase 7/12) we will only generate embeddings once per tracking ID, saving massive compute.

## 14. Hackathon requirement mapping
- "Generate facial embeddings using a state-of-the-art face recognition model (InsightFace/ArcFace)."
- "Avoid using the face_recognition Python library."
