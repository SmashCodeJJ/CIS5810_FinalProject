"""
Camera capture utilities for real-time face swapping
Handles webcam access and frame capture
"""
import cv2
import numpy as np
from typing import Optional, Tuple


class CameraCapture:
    """
    Webcam capture wrapper with error handling
    """
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """
        Args:
            camera_id: Camera device ID (usually 0 for default)
            width: Frame width
            height: Frame height
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.is_opened = False
        
    def open(self) -> bool:
        """Open camera connection"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                return False
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Set buffer size to 1 for low latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.is_opened = True
            return True
        except Exception as e:
            print(f"Error opening camera: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame from camera
        Returns: (success, frame)
        """
        if not self.is_opened or self.cap is None:
            return False, None
        
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return True, frame
            return False, None
        except Exception as e:
            print(f"Error reading frame: {e}")
            return False, None
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
    
    def __enter__(self):
        """Context manager entry"""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()

