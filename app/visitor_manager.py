import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
from app.database import DatabaseManager

if TYPE_CHECKING:
    from app.image_logger import ImageLogger

logger = logging.getLogger(__name__)

class VisitorManager:
    def __init__(self, db: DatabaseManager, image_logger: ImageLogger, similarity_threshold: float = 0.4, missing_timeout_frames: int = 30):
        self.db = db
        self.image_logger = image_logger
        self.threshold = similarity_threshold
        self.missing_timeout_frames = missing_timeout_frames

        self.known_visitors = []
        for visitor_id, embedding in self.db.get_visitor_embeddings():
            if embedding.ndim != 1 or embedding.size != 512:
                logger.warning(
                    "Ignoring visitor %s with invalid embedding shape %s",
                    visitor_id,
                    embedding.shape,
                )
                continue
            self.known_visitors.append((visitor_id, embedding))
        self.visitor_count = self.db.get_unique_visitor_count()
        self.active_visits = {}
        self.track_visitors = {}

    @staticmethod
    def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        emb1_norm = np.linalg.norm(emb1)
        emb2_norm = np.linalg.norm(emb2)
        if emb1_norm == 0 or emb2_norm == 0:
            return -1.0
        return float(np.dot(emb1, emb2) / (emb1_norm * emb2_norm))

    def identify_visitor(self, new_embedding: np.ndarray, timestamp: Optional[str] = None) -> str:
        best_match_id = None
        best_sim = -1.0
        for vid, known_emb in self.known_visitors:
            sim = self._cosine_similarity(new_embedding, known_emb)
            if sim > best_sim:
                best_sim = sim
                best_match_id = vid
                
        if best_match_id is not None and best_sim >= self.threshold:
            logger.info("Recognized visitor %s (similarity %.3f)", best_match_id, best_sim)
            return best_match_id

        new_id = f"VISITOR_{self.visitor_count + 1:04d}"
        self.db.register_visitor(new_id, new_embedding, timestamp or "")
        self.known_visitors.append((new_id, np.asarray(new_embedding, dtype=np.float32)))
        self.visitor_count = self.db.get_unique_visitor_count()
        logger.info("Registered visitor %s", new_id)
        return new_id

    def _save_event(self, visitor_id: str, event_type: str, timestamp: str, crop, track_id: int) -> None:
        image_path = self.image_logger.save_event_image(crop, visitor_id, event_type, timestamp)
        if not image_path:
            logger.error("Image save failed for %s %s; recording event with an empty image path", event_type, visitor_id)
        try:
            self.db.log_event(visitor_id, event_type, timestamp, image_path, track_id)
        except Exception:
            logger.exception("Database event insert failed for %s %s", event_type, visitor_id)
            raise
        logger.info("%s: %s (Track %s)", event_type, visitor_id, track_id)

    def process_frame(self, frame_number: int, timestamp: str, tracked_faces: list, face_crops: dict, embeddings: dict):
        seen_this_frame = set()

        for det in tracked_faces:
            tid = int(det["track_id"])
            vid = self.track_visitors.get(tid)
            if vid is None and tid in embeddings:
                vid = self.identify_visitor(embeddings[tid], timestamp)
                self.track_visitors[tid] = vid
            if vid is None:
                continue

            seen_this_frame.add(vid)
            det["visitor_id"] = vid
            crop = face_crops.get(tid)
            if vid not in self.active_visits:
                self.active_visits[vid] = {
                    'last_seen_frame': frame_number,
                    'state': 'PRESENT',
                    'track_id': tid,
                    'last_crop': crop,
                }
                self._save_event(vid, "ENTRY", timestamp, crop, tid)
            else:
                self.active_visits[vid]['last_seen_frame'] = frame_number
                self.active_visits[vid]['state'] = 'PRESENT'
                self.active_visits[vid]['track_id'] = tid
                if crop is not None:
                    self.active_visits[vid]['last_crop'] = crop
            self.db.update_last_seen(vid, timestamp)

        for vid in list(self.active_visits.keys()):
            if vid not in seen_this_frame:
                visit_data = self.active_visits[vid]
                frames_missing = frame_number - visit_data['last_seen_frame']

                if frames_missing <= self.missing_timeout_frames:
                    visit_data['state'] = 'MISSING'
                    continue

                self._save_event(vid, "EXIT", timestamp, visit_data['last_crop'], visit_data['track_id'])
                self.track_visitors.pop(visit_data['track_id'], None)
                del self.active_visits[vid]
