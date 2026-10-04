import argparse
import json
import logging
import cv2
import os
import sys
import urllib.request
import time
from datetime import datetime

from app.config import ConfigurationError, load_config
from app.event_logger import configure_logging
from app.video_input import VideoInput
from app.detector import FaceDetector
from app.recognizer import FaceRecognizer
from app.tracker import FaceTracker
from app.database import DatabaseManager
from app.image_logger import ImageLogger
from app.visitor_manager import VisitorManager

def download_model_if_needed(model_path):
    logger = logging.getLogger(__name__)
    if not os.path.exists(model_path):
        logger.info(f"Model {model_path} not found. Downloading Face YOLOv8...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
        try:
            urllib.request.urlretrieve(url, model_path)
            logger.info("Model downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            sys.exit(1)

def _write_run_stats(stats_path, stats):
    if not stats_path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(stats_path)), exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as stats_file:
            json.dump(stats, stats_file, indent=2)
    except OSError:
        logging.getLogger(__name__).exception("Unable to write run statistics")


def main(source_override=None, stats_path=None, display=True):
    try:
        config = load_config("config.json")
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        return

    if source_override:
        config["input"]["source"] = source_override
        config["input"]["source_type"] = "file"

    configure_logging(config['logging']['path'])
    logger = logging.getLogger(__name__)
    
    logger.info("Application started. Running Final Integrated Pipeline.")
    
    input_cfg = config.get('input', {})
    det_cfg = config.get('detection', {})
    rec_cfg = config.get('recognition', {})
    trk_cfg = config.get('tracking', {})
    
    source = input_cfg.get('source', '')
    source_type = input_cfg.get('source_type', 'file')
    model_path = det_cfg.get('model', 'models/yolov8n-face.pt')
    conf_threshold = det_cfg.get('confidence', 0.5)
    frame_skip = det_cfg.get('frame_skip', 2)
    
    download_model_if_needed(model_path)
    
    try:
        db = DatabaseManager(env_var=config['database'].get('url_env', 'DATABASE_URL'))
    except Exception as error:
        logger.error("Unable to initialize PostgreSQL: %s", error)
        return
    visitors_before = db.get_unique_visitor_count()
    entries_before = db.get_total_entry_events()
    exits_before = db.get_total_exit_events()
    image_logger = ImageLogger(os.path.dirname(config['logging']['path']) or 'logs')
    
    video = VideoInput(source=source, source_type=source_type)
    if not video.open():
        logger.error("Could not open video.")
        return
        
    try:
        detector = FaceDetector(model_path=model_path, conf_threshold=conf_threshold)
        tracker = FaceTracker(max_age=trk_cfg.get('max_age', 30))
        recognizer = FaceRecognizer()
        manager = VisitorManager(
            db=db, 
            image_logger=image_logger, 
            similarity_threshold=rec_cfg['threshold'],
            missing_timeout_frames=trk_cfg.get('max_age', 30)
        )
    except Exception as e:
        logger.error(f"Failed to initialize AI modules: {e}")
        return
        
    logger.info(f"Video details: {video.width}x{video.height} @ {video.fps} FPS. Total frames: {video.frame_count}")
    
    frames_read = 0
    detection_frames = 0
    window_name = "Intelligent Face Tracker"
    
    # Tracking embeddings dictionary: track_id -> embedding (compute only once per track)
    track_embeddings = {}
    
    # Benchmark stats
    start_time = time.time()
    yolo_times = []
    recognizer_times = []
    
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break
                
            frames_read += 1
            timestamp = datetime.now().isoformat()
            
            # Frame skipping logic
            detection_due = (frames_read - 1) % frame_skip == 0
            if detection_due:
                detection_frames += 1
                t1 = time.time()
                detections = detector.detect(frame)
                yolo_times.append(time.time() - t1)
            else:
                detections = []
                
            tracked_faces = tracker.update(detections, detection_performed=detection_due)
            
            face_crops = {}
            
            for det in tracked_faces:
                try:
                    tid = det["track_id"]
                    if tid not in track_embeddings:
                        crop = recognizer.crop_face(frame, det["bbox"], padding=0.2)
                        if crop is not None:
                            t2 = time.time()
                            emb = recognizer.get_embedding(crop)
                            if emb is not None:
                                recognizer_times.append(time.time() - t2)
                                track_embeddings[tid] = emb
                                face_crops[tid] = crop
                except Exception:
                    logger.exception("Failed to process face on track %s", det.get("track_id"))
            
            # Combine known track embeddings with new ones for processing
            active_embs = {t["track_id"]: track_embeddings[t["track_id"]] for t in tracked_faces if t["track_id"] in track_embeddings}
            
            manager.process_frame(frames_read, timestamp, tracked_faces, face_crops, active_embs)
            
            # Visualization
            display_frame = frame.copy()
            for det in tracked_faces:
                x1, y1, x2, y2 = map(int, det["bbox"])
                vid = det.get("visitor_id", f"Track_{det['track_id']}")
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(display_frame, vid, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
            cv2.putText(display_frame, f"Unique Visitors: {manager.visitor_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display_frame, f"FPS: {frames_read / (time.time() - start_time):.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if display:
                try:
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                except Exception:
                    logger.debug("Display unavailable; continuing headlessly", exc_info=True)
                
    except Exception as e:
        logger.error(f"Error during video processing: {e}")
    finally:
        if 'manager' in locals() and frames_read:
            manager.process_frame(
                frames_read + trk_cfg.get('max_age', 30) + 1,
                datetime.now().isoformat(),
                [],
                {},
                {},
            )
        video.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        total_time = time.time() - start_time
        fps = frames_read / total_time if total_time > 0 else 0
        avg_yolo = sum(yolo_times) / len(yolo_times) if yolo_times else 0
        avg_rec = sum(recognizer_times) / len(recognizer_times) if recognizer_times else 0
        
        logger.info(f"--- Benchmark Results ---")
        logger.info(f"Total frames: {frames_read}")
        logger.info(f"Processing time: {total_time:.2f}s")
        logger.info(f"Overall FPS: {fps:.2f}")
        logger.info(f"Avg YOLO Time: {avg_yolo*1000:.1f}ms")
        logger.info(f"Avg InsightFace Time: {avg_rec*1000:.1f}ms")
        logger.info(f"Total Unique Visitors: {manager.visitor_count}")
        _write_run_stats(
            stats_path,
            {
                "status": "completed",
                "source": source,
                "frames_processed": frames_read,
                "detection_frames": detection_frames,
                "skipped_frames": max(0, frames_read - detection_frames),
                "processing_time_seconds": total_time,
                "processing_fps": fps,
                "input_fps": video.fps,
                "average_yolo_ms": avg_yolo * 1000,
                "average_insightface_ms": avg_rec * 1000,
                "unique_visitors": manager.visitor_count,
                "total_entries": db.get_total_entry_events(),
                "total_exits": db.get_total_exit_events(),
                "currently_present": len(db.get_currently_present_visitors()),
                "visitors_registered_this_run": max(0, manager.visitor_count - visitors_before),
                "entries_this_run": max(0, db.get_total_entry_events() - entries_before),
                "exits_this_run": max(0, db.get_total_exit_events() - exits_before),
            },
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the intelligent face tracker pipeline")
    parser.add_argument("--source", help="Override the configured local video source")
    parser.add_argument("--stats-path", help="Write per-run JSON statistics to this path")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV display windows")
    arguments = parser.parse_args()
    main(arguments.source, arguments.stats_path, not arguments.no_display)
