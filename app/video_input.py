import cv2
import logging
from typing import Tuple, Optional
import urllib.parse

logger = logging.getLogger(__name__)

class VideoInput:
    def __init__(self, source: str, source_type: str = "file"):
        self.source = source
        self.source_type = source_type
        self.cap = None
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._frame_count = 0

    def _sanitize_source_for_logging(self, source_url: str) -> str:
        """Removes passwords from RTSP URLs for safe logging."""
        if self.source_type == "rtsp" and "@" in source_url:
            try:
                parsed = urllib.parse.urlparse(source_url)
                if parsed.password:
                    # Replace password with ***
                    safe_netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
                    safe_url = parsed._replace(netloc=safe_netloc).geturl()
                    return safe_url
            except Exception:
                pass
        return source_url

    def open(self) -> bool:
        safe_source = self._sanitize_source_for_logging(self.source)
        logger.info(f"Attempting to open video source ({self.source_type}): {safe_source}")
        
        # In OpenCV, providing a string source works for both files and RTSP URLs
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {safe_source}")
            return False
            
        self._width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Frame count is often not available or 0 for RTSP streams
        if self.source_type == "file":
            self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self._frame_count = -1
            
        logger.info(f"Successfully opened video source. Resolution: {self._width}x{self._height}, FPS: {self._fps:.2f}")
        return True

    def read(self) -> Tuple[bool, Optional[any]]:
        if not self.is_opened():
            logger.error("Attempted to read from an unopened video source.")
            return False, None
            
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            # End of video or temporary RTSP drop
            return False, None
            
        return True, frame

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            safe_source = self._sanitize_source_for_logging(self.source)
            logger.info(f"Released video source: {safe_source}")

    @property
    def width(self) -> int:
        return self._width
        
    @property
    def height(self) -> int:
        return self._height
        
    @property
    def fps(self) -> float:
        return self._fps
        
    @property
    def frame_count(self) -> int:
        return self._frame_count
