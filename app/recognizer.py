import logging
from typing import Optional
import numpy as np
import cv2
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

class FaceRecognizer:
    def __init__(self, model_name: str = "buffalo_l", providers: list = None):
        """
        Initializes the InsightFace model once.
        buffalo_l contains the ArcFace model for generating embeddings.
        """
        if providers is None:
            # We explicitly allow CPU execution for maximum compatibility during the hackathon
            providers = ['CPUExecutionProvider']
            
        logger.info(f"Loading InsightFace model: {model_name}")
        try:
            self.app = FaceAnalysis(name=model_name, providers=providers)
            # Using a smaller internal detection size since we only pass pre-cropped faces
            self.app.prepare(ctx_id=0, det_size=(160, 160)) 
            logger.info("InsightFace loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise

    def get_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Takes an OpenCV BGR face crop, detects the exact landmarks internally, 
        aligns it, and generates a 512-dimensional normalized embedding using ArcFace.
        """
        if face_crop is None or face_crop.size == 0:
            return None
            
        # Filter out extremely small crops that would produce garbage embeddings
        h, w = face_crop.shape[:2]
        if h < 20 or w < 20:
            logger.debug("Face crop too small for reliable recognition.")
            return None

        # InsightFace processes the image
        faces = self.app.get(face_crop)
        
        if not faces:
            # The internal detector couldn't find the landmarks in the YOLO crop
            return None
            
        # Return the embedding of the first face found in the crop
        # Note: ArcFace automatically L2-normalizes the output embedding!
        return faces[0].embedding
        
    def crop_face(self, frame: np.ndarray, bbox: list, padding: float = 0.2) -> Optional[np.ndarray]:
        """
        Safely crops the face from the frame using a bounding box, adding padding 
        so InsightFace has enough context to properly align the face.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        
        box_w = x2 - x1
        box_h = y2 - y1
        
        # Add padding
        pad_x = int(box_w * padding)
        pad_y = int(box_h * padding)
        
        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y)
        nx2 = min(w, x2 + pad_x)
        ny2 = min(h, y2 + pad_y)
        
        crop = frame[ny1:ny2, nx1:nx2]
        if crop.size == 0:
            return None
            
        return crop
