import logging
from typing import List, Dict, Any
from ultralytics import YOLO
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.load_model()

    def load_model(self):
        logger.info(f"Loading YOLO Face Detection model from {self.model_path}")
        try:
            # Load the model using Ultralytics
            self.model = YOLO(self.model_path)
            
            # The first inference is usually slow due to initialization, so we do a warmup
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy_img, verbose=False)
            logger.info("Model loaded and warmed up successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a given frame.
        Returns a list of dictionaries containing bbox (x1, y1, x2, y2) and confidence.
        """
        if self.model is None:
            logger.error("Model is not loaded.")
            return []
            
        if frame is None or frame.size == 0:
            return []

        # Run inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract coordinates, confidence, and class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                conf = box.conf[0].cpu().numpy().item()
                cls = int(box.cls[0].cpu().numpy().item())
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls
                })
                
        return detections
        
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Helper method to draw bounding boxes and confidence scores on the frame.
        """
        display_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            conf = det["confidence"]
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence label
            label = f"Face: {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display_frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(display_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        return display_frame
