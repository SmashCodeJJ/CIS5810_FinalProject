"""
Performance monitoring for real-time face swapping
Tracks FPS, latency, and other metrics
"""
import time
from collections import deque
from typing import Optional


class PerformanceMonitor:
    """
    Monitors and reports performance metrics for real-time processing
    """
    
    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Number of recent frames to average over
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.detection_times = deque(maxlen=window_size)
        self.generator_times = deque(maxlen=window_size)
        
        self.frame_count = 0
        self.start_time = time.time()
        self.frame_start_time = None
        
    def start_frame(self):
        """Mark the start of frame processing"""
        self.frame_start_time = time.time()
        self.frame_count += 1
    
    def end_frame(self):
        """Mark the end of frame processing"""
        if self.frame_start_time is not None:
            elapsed = time.time() - self.frame_start_time
            self.frame_times.append(elapsed)
            self.frame_start_time = None
    
    def record_detection_time(self, time_ms: float):
        """Record face detection/tracking time"""
        self.detection_times.append(time_ms)
    
    def record_generator_time(self, time_ms: float):
        """Record generator inference time"""
        self.generator_times.append(time_ms)
    
    def record_processing_time(self, time_ms: float):
        """Record total processing time"""
        self.processing_times.append(time_ms)
    
    def get_fps(self) -> float:
        """Get average FPS over recent frames"""
        if len(self.frame_times) == 0:
            return 0.0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_avg_latency(self) -> float:
        """Get average latency in milliseconds"""
        if len(self.processing_times) == 0:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)
    
    def get_avg_detection_time(self) -> float:
        """Get average detection time in milliseconds"""
        if len(self.detection_times) == 0:
            return 0.0
        return sum(self.detection_times) / len(self.detection_times)
    
    def get_avg_generator_time(self) -> float:
        """Get average generator time in milliseconds"""
        if len(self.generator_times) == 0:
            return 0.0
        return sum(self.generator_times) / len(self.generator_times)
    
    def get_stats(self) -> dict:
        """Get all performance statistics"""
        return {
            'fps': self.get_fps(),
            'avg_latency_ms': self.get_avg_latency(),
            'avg_detection_ms': self.get_avg_detection_time(),
            'avg_generator_ms': self.get_avg_generator_time(),
            'total_frames': self.frame_count,
            'elapsed_time': time.time() - self.start_time
        }
    
    def print_stats(self):
        """Print current statistics"""
        stats = self.get_stats()
        print(f"FPS: {stats['fps']:.1f} | "
              f"Latency: {stats['avg_latency_ms']:.1f}ms | "
              f"Detection: {stats['avg_detection_ms']:.1f}ms | "
              f"Generator: {stats['avg_generator_ms']:.1f}ms")
    
    def reset(self):
        """Reset all statistics"""
        self.frame_times.clear()
        self.processing_times.clear()
        self.detection_times.clear()
        self.generator_times.clear()
        self.frame_count = 0
        self.start_time = time.time()

