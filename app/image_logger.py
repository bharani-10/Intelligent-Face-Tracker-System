import os
import cv2
import logging
import numpy as np
from uuid import uuid4

logger = logging.getLogger(__name__)

class ImageLogger:
    def __init__(self, base_dir: str = "logs"):
        self.entries_dir = os.path.join(base_dir, "entries")
        self.exits_dir = os.path.join(base_dir, "exits")
        os.makedirs(self.entries_dir, exist_ok=True)
        os.makedirs(self.exits_dir, exist_ok=True)

    def save_event_image(self, face_crop: np.ndarray, visitor_id: str, event_type: str, timestamp_str: str) -> str:
        date_str = timestamp_str.split("T")[0]
        safe_time = timestamp_str.replace(":", "-").replace(".", "-")
        
        target_dir = os.path.join(self.entries_dir if event_type == "ENTRY" else self.exits_dir, date_str)
        os.makedirs(target_dir, exist_ok=True)
        
        filename = f"{visitor_id}_{event_type}_{safe_time}_{uuid4().hex[:8]}.jpg"
        filepath = os.path.join(target_dir, filename)
        
        try:
            if face_crop is None or face_crop.size == 0 or not cv2.imwrite(filepath, face_crop):
                raise OSError("OpenCV could not write the image")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save image for {visitor_id}: {e}")
            return ""
