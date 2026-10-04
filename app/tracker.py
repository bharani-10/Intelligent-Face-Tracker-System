import logging
from typing import List, Dict, Any
import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)

class FaceTracker:
    def __init__(self, max_age: int = 30):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=max_age,
            minimum_matching_threshold=0.8,
            frame_rate=30,
        )
        self.last_tracks = []
        logger.info(f"Initialized ByteTrack with max_age={max_age}")

    def update(self, detections: List[Dict[str, Any]], detection_performed: bool = True) -> List[Dict[str, Any]]:
        if not detection_performed:
            return list(self.last_tracks)

        if not detections:
            empty_dets = sv.Detections.empty()
            self.tracker.update_with_detections(empty_dets)
            self.last_tracks = []
            return []
            
        xyxy = np.array([d['bbox'] for d in detections])
        confidence = np.array([d['confidence'] for d in detections])
        class_id = np.array([d['class_id'] for d in detections])
        
        sv_dets = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        tracked = self.tracker.update_with_detections(sv_dets)
        
        result = []
        for i in range(len(tracked.xyxy)):
            result.append({
                "bbox": tracked.xyxy[i].tolist(),
                "confidence": tracked.confidence[i] if tracked.confidence is not None else 1.0,
                "class_id": tracked.class_id[i] if tracked.class_id is not None else 0,
                "track_id": tracked.tracker_id[i]
            })
        self.last_tracks = result
        return result
