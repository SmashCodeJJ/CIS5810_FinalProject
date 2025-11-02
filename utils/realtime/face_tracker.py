"""
Face tracking module for real-time face swapping
Uses OpenCV trackers to avoid expensive face detection every frame
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List
import time


class FaceTracker:
    """
    Face tracker that alternates between detection and tracking
    Detection is expensive (~20ms), tracking is cheap (~2ms)
    """
    
    def __init__(self, 
                 detector,
                 detect_interval: int = 5,
                 tracker_type: str = 'CSRT',
                 confidence_threshold: float = 0.6):
        """
        Args:
            detector: Face detection model (app from insightface)
            detect_interval: Run full detection every N frames
            tracker_type: 'CSRT', 'KCF', or 'MOSSE'
            confidence_threshold: Minimum confidence for face detection
        """
        self.detector = detector
        self.detect_interval = detect_interval
        self.confidence_threshold = confidence_threshold
        self.frame_count = 0
        self.tracker = None
        self.bbox = None
        self.tracker_type = tracker_type
        self.last_detection_time = 0
        
    def _create_tracker(self):
        """Create OpenCV tracker based on type"""
        if self.tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        elif self.tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        elif self.tracker_type == 'MOSSE':
            return cv2.TrackerMOSSE_create()
        else:
            raise ValueError(f"Unknown tracker type: {self.tracker_type}")
    
    def _detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Run full face detection
        Returns: (x, y, w, h) bounding box or None
        """
        try:
            # Use det_model directly (the internal detection model)
            bboxes, _ = self.detector.det_model.detect(frame, max_num=1, metric='default')
            
            if bboxes.shape[0] == 0:
                return None
            
            # Filter by confidence
            keep = bboxes[:, 4] >= self.confidence_threshold
            bboxes = bboxes[keep]
            
            if bboxes.shape[0] == 0:
                return None
            
            # Get best face (highest confidence)
            best_bbox = bboxes[0]
            x, y, w, h = int(best_bbox[0]), int(best_bbox[1]), \
                        int(best_bbox[2] - best_bbox[0]), int(best_bbox[3] - best_bbox[1])
            
            # Ensure valid bounding box
            if w > 0 and h > 0:
                return (x, y, w, h)
            return None
            
        except Exception as e:
            print(f"Face detection error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _track_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Update face position using tracker
        Returns: (x, y, w, h) bounding box or None if tracking failed
        """
        if self.tracker is None:
            return None
        
        try:
            success, bbox = self.tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                # Validate bbox
                if w > 0 and h > 0 and x >= 0 and y >= 0:
                    return (x, y, w, h)
            return None
        except Exception as e:
            print(f"Tracking error: {e}")
            return None
    
    def update(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Update tracker/detector and return current face bounding box
        Returns: (x, y, w, h) or None if no face found
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Determine if we should run detection
        should_detect = (
            self.frame_count % self.detect_interval == 0 or  # Interval-based
            self.bbox is None or  # No face currently tracked
            self.tracker is None or  # Tracker not initialized
            (current_time - self.last_detection_time) > 1.0  # Force detection every 1 second
        )
        
        if should_detect:
            # Run full detection
            self.bbox = self._detect_face(frame)
            
            if self.bbox is not None:
                # Initialize tracker with new detection
                self.tracker = self._create_tracker()
                x, y, w, h = self.bbox
                self.tracker.init(frame, (x, y, w, h))
                self.last_detection_time = current_time
            else:
                # No face found, reset tracker
                self.tracker = None
                self.bbox = None
                
        else:
            # Use tracking instead of detection
            self.bbox = self._track_face(frame)
            
            # If tracking failed, reset for detection next frame
            if self.bbox is None:
                self.tracker = None
        
        return self.bbox
    
    def reset(self):
        """Reset tracker state"""
        self.tracker = None
        self.bbox = None
        self.frame_count = 0
        self.last_detection_time = 0
    
    def get_stats(self) -> dict:
        """Get tracker statistics"""
        return {
            'frame_count': self.frame_count,
            'detections': self.frame_count // self.detect_interval,
            'tracking': self.frame_count - (self.frame_count // self.detect_interval),
            'has_face': self.bbox is not None
        }

