"""
Computer Vision Processing Pipeline - Real-time Contour Extraction & Tracking
Task 4: Adaptive thresholding, contour detection, object tracking, and metrics evaluation
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time
from collections import defaultdict, deque
import json


@dataclass
class ContourMetrics:
    """Metrics for a detected contour"""
    contour_id: int
    area: float
    perimeter: float
    center: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    aspect_ratio: float
    circularity: float
    timestamp: float
    frame_number: int


@dataclass
class FrameMetrics:
    """Per-frame inference metrics"""
    frame_number: int
    timestamp: float
    total_contours_detected: int
    total_objects_tracked: int
    processing_time_ms: float
    contour_metrics: List[ContourMetrics] = field(default_factory=list)
    adaptive_threshold_range: Tuple[int, int] = (0, 255)


class ObjectTracker:
    """Tracks objects across frames using centroid tracking"""
    
    def __init__(self, max_distance: float = 50.0, max_disappeared: int = 5):
        self.next_object_id = 0
        self.objects = {}  # id -> centroid history
        self.disappeared = {}  # id -> frames disappeared count
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.tracks = defaultdict(list)  # id -> list of centroids
    
    def register(self, centroid: Tuple[int, int]) -> int:
        """Register a new object"""
        obj_id = self.next_object_id
        self.next_object_id += 1
        self.objects[obj_id] = centroid
        self.disappeared[obj_id] = 0
        self.tracks[obj_id] = [centroid]
        return obj_id
    
    def deregister(self, obj_id: int):
        """Deregister an object"""
        del self.objects[obj_id]
        del self.disappeared[obj_id]
    
    def update(self, centroids: List[Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
        """Update tracker with new centroids"""
        if len(centroids) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects
        
        # Match new centroids to existing objects
        if len(self.objects) == 0:
            for centroid in centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Compute distances between existing centroids and new centroids
            distances = np.zeros((len(object_centroids), len(centroids)))
            for i, obj_centroid in enumerate(object_centroids):
                for j, new_centroid in enumerate(centroids):
                    distances[i, j] = np.linalg.norm(
                        np.array(obj_centroid) - np.array(new_centroid)
                    )
            
            # Match objects to centroids
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if distances[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = centroids[col]
                self.disappeared[object_id] = 0
                self.tracks[object_id].append(centroids[col])
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Register new centroids
            unused_cols = set(range(len(centroids))) - used_cols
            for col in unused_cols:
                self.register(centroids[col])
            
            # Handle disappeared objects
            for row in set(range(len(object_ids))) - used_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        
        return self.objects


class AdaptiveThresholdAnalyzer:
    """Analyzes optimal threshold parameters"""
    
    @staticmethod
    def analyze_histogram(gray_image: np.ndarray) -> Dict[str, int]:
        """Analyze image histogram to find optimal threshold"""
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Find peak (mode)
        peak = np.argmax(hist)
        
        # Find valleys
        valleys = []
        for i in range(1, len(hist) - 1):
            if hist[i-1] > hist[i] < hist[i+1]:
                valleys.append(i)
        
        return {
            'peak': int(peak),
            'mean': int(np.mean(gray_image)),
            'median': int(np.median(gray_image)),
            'std': int(np.std(gray_image)),
            'valley_count': len(valleys)
        }


class CVPipeline:
    """Main Computer Vision Processing Pipeline"""
    
    def __init__(self, 
                 blur_kernel_size: int = 5,
                 morph_kernel_size: int = 5,
                 min_contour_area: float = 100.0,
                 max_contour_area: float = 50000.0):
        
        self.blur_kernel_size = blur_kernel_size
        self.morph_kernel_size = morph_kernel_size
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        
        self.tracker = ObjectTracker()
        self.frame_metrics_history = []
        self.current_frame_number = 0
        
        # Morphological kernels
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (morph_kernel_size, morph_kernel_size)
        )
    
    def apply_gaussian_filter(self, image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian blur for noise reduction"""
        kernel_size = max(self.blur_kernel_size, 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def apply_adaptive_threshold(self, 
                                gray_image: np.ndarray,
                                method: str = 'gaussian') -> Tuple[np.ndarray, Dict]:
        """Apply adaptive thresholding"""
        if method == 'gaussian':
            thresholded = cv2.adaptiveThreshold(
                gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        else:  # mean
            thresholded = cv2.adaptiveThreshold(
                gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        
        # Analyze threshold parameters
        analysis = AdaptiveThresholdAnalyzer.analyze_histogram(gray_image)
        
        return thresholded, analysis
    
    def apply_morphological_operations(self, binary_image: np.ndarray) -> np.ndarray:
        """Apply morphological opening and closing"""
        # Opening: removes small noise
        opened = cv2.morphologyEx(
            binary_image, cv2.MORPH_OPEN, self.morph_kernel, iterations=1
        )
        # Closing: closes small holes
        closed = cv2.morphologyEx(
            opened, cv2.MORPH_CLOSE, self.morph_kernel, iterations=1
        )
        return closed
    
    def detect_contours(self, processed_image: np.ndarray) -> List[np.ndarray]:
        """Detect contours in processed image"""
        contours, _ = cv2.findContours(
            processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours by area
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < self.max_contour_area:
                filtered_contours.append(contour)
        
        return filtered_contours
    
    def extract_contour_metrics(self, 
                               contour: np.ndarray,
                               contour_id: int,
                               frame_number: int,
                               timestamp: float) -> ContourMetrics:
        """Extract metrics from a contour"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Circularity (4*pi*area / perimeter^2)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        
        return ContourMetrics(
            contour_id=contour_id,
            area=area,
            perimeter=perimeter,
            center=(cx, cy),
            bounding_box=(x, y, w, h),
            aspect_ratio=aspect_ratio,
            circularity=circularity,
            timestamp=timestamp,
            frame_number=frame_number
        )
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, FrameMetrics]:
        """Process a single frame"""
        start_time = time.time()
        self.current_frame_number += 1
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Gaussian blur
        blurred = self.apply_gaussian_filter(gray, sigma=1.0)
        
        # Adaptive threshold
        thresholded, threshold_analysis = self.apply_adaptive_threshold(blurred)
        
        # Morphological operations
        processed = self.apply_morphological_operations(thresholded)
        
        # Detect contours
        contours = self.detect_contours(processed)
        
        # Extract metrics and track objects
        contour_metrics_list = []
        centroids = []
        
        for i, contour in enumerate(contours):
            metrics = self.extract_contour_metrics(
                contour, i, self.current_frame_number, time.time()
            )
            contour_metrics_list.append(metrics)
            centroids.append(metrics.center)
        
        # Update tracker
        tracked_objects = self.tracker.update(centroids)
        
        # Create visualization frame
        output_frame = frame.copy()
        
        # Draw tracked objects
        for obj_id, (cx, cy) in tracked_objects.items():
            # Draw circle at centroid
            cv2.circle(output_frame, (cx, cy), 4, (0, 255, 0), -1)
            
            # Draw track history
            if obj_id in self.tracker.tracks and len(self.tracker.tracks[obj_id]) > 1:
                track = self.tracker.tracks[obj_id]
                for i in range(1, len(track)):
                    cv2.line(output_frame, track[i-1], track[i], (0, 255, 0), 2)
            
            # Draw text label
            text = f"ID:{obj_id}"
            cv2.putText(output_frame, text, (cx - 10, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw all detected contours
        for i, contour in enumerate(contours):
            cv2.drawContours(output_frame, [contour], 0, (255, 0, 0), 2)
        
        # Add frame information
        frame_info = f"Frame: {self.current_frame_number} | Objects: {len(tracked_objects)} | Contours: {len(contours)}"
        cv2.putText(output_frame, frame_info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        processing_time = (time.time() - start_time) * 1000
        
        frame_metrics = FrameMetrics(
            frame_number=self.current_frame_number,
            timestamp=time.time(),
            total_contours_detected=len(contours),
            total_objects_tracked=len(tracked_objects),
            processing_time_ms=processing_time,
            contour_metrics=contour_metrics_list,
            adaptive_threshold_range=(
                threshold_analysis['mean'] - threshold_analysis['std'],
                threshold_analysis['mean'] + threshold_analysis['std']
            )
        )
        
        self.frame_metrics_history.append(frame_metrics)
        
        return output_frame, frame_metrics
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics"""
        if not self.frame_metrics_history:
            return {}
        
        processing_times = [m.processing_time_ms for m in self.frame_metrics_history]
        contour_counts = [m.total_contours_detected for m in self.frame_metrics_history]
        object_counts = [m.total_objects_tracked for m in self.frame_metrics_history]
        
        return {
            'total_frames_processed': len(self.frame_metrics_history),
            'total_contours_detected': sum(contour_counts),
            'total_objects_tracked': sum(object_counts),
            'avg_processing_time_ms': np.mean(processing_times),
            'min_processing_time_ms': np.min(processing_times),
            'max_processing_time_ms': np.max(processing_times),
            'avg_contours_per_frame': np.mean(contour_counts),
            'avg_objects_per_frame': np.mean(object_counts),
            'fps': 1000.0 / np.mean(processing_times) if processing_times else 0
        }
    
    def export_metrics(self, filename: str):
        """Export metrics to JSON file"""
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'total_frames': len(self.frame_metrics_history),
            'performance_summary': self.get_performance_summary(),
            'frame_metrics': [
                {
                    'frame_number': m.frame_number,
                    'timestamp': m.timestamp,
                    'total_contours': m.total_contours_detected,
                    'total_objects': m.total_objects_tracked,
                    'processing_time_ms': m.processing_time_ms,
                    'contours': [
                        {
                            'id': c.contour_id,
                            'area': c.area,
                            'perimeter': c.perimeter,
                            'center': c.center,
                            'bbox': c.bounding_box,
                            'aspect_ratio': c.aspect_ratio,
                            'circularity': c.circularity
                        }
                        for c in m.contour_metrics
                    ]
                }
                for m in self.frame_metrics_history
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=2)
