"""
Test script for Computer Vision Pipeline
Tests with synthetic video frames and real-time processing
"""

import cv2
import numpy as np
from cv_pipeline import CVPipeline, FrameMetrics
import os
import json
from datetime import datetime


def create_synthetic_frames(num_frames: int = 30, width: int = 640, height: int = 480) -> list:
    """Create synthetic video frames with moving shapes"""
    frames = []
    
    for frame_num in range(num_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray background
        
        # Draw moving circles
        for obj_id in range(3):
            # Calculate position with animation
            center_x = int(width * (0.3 + 0.4 * np.sin(2 * np.pi * (frame_num + obj_id) / num_frames)))
            center_y = int(height * (0.3 + 0.4 * np.cos(2 * np.pi * (frame_num + obj_id) / num_frames)))
            radius = 40 + int(20 * np.sin(2 * np.pi * frame_num / num_frames))
            
            color = (50 + obj_id * 70, 100 + obj_id * 50, 150 - obj_id * 30)
            cv2.circle(frame, (center_x, center_y), radius, color, -1)
        
        # Add some rectangles
        for obj_id in range(2):
            x = int(100 + obj_id * 300 + 50 * np.sin(2 * np.pi * frame_num / num_frames))
            y = int(250 + 50 * np.cos(2 * np.pi * (frame_num + obj_id) / num_frames))
            w = 60
            h = 80
            color = (100 + obj_id * 80, 150 - obj_id * 40, 100)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        
        # Add some noise
        noise = np.random.normal(0, 5, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)
        
        frames.append(frame)
    
    return frames


def test_cv_pipeline(output_folder: str = '.'):
    """Test the CV pipeline with synthetic frames"""
    
    print("="*80)
    print("COMPUTER VISION PROCESSING PIPELINE - TEST SUITE")
    print("="*80)
    print()
    
    # Create pipeline
    pipeline = CVPipeline(
        blur_kernel_size=5,
        morph_kernel_size=5,
        min_contour_area=100.0,
        max_contour_area=50000.0
    )
    
    print("[OK] Pipeline initialized")
    print()
    
    # Generate synthetic frames
    print("Generating synthetic video frames...")
    frames = create_synthetic_frames(num_frames=30, width=640, height=480)
    print(f"[OK] Generated {len(frames)} frames")
    print()
    
    # Process frames
    print("Processing frames...")
    output_frames = []
    
    for i, frame in enumerate(frames):
        output_frame, metrics = pipeline.process_frame(frame)
        output_frames.append(output_frame)
        
        if (i + 1) % 10 == 0:
            print(f"  Frame {i+1}/{len(frames)}: {metrics.total_contours_detected} contours, " +
                  f"{metrics.total_objects_tracked} objects, {metrics.processing_time_ms:.2f}ms")
    
    print(f"[OK] Processed {len(frames)} frames")
    print()
    
    # Get performance summary
    print("="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    summary = pipeline.get_performance_summary()
    
    for key, value in summary.items():
        if 'time' in key or 'fps' in key:
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    print()
    
    # Export metrics
    metrics_file = os.path.join(output_folder, 'cv_metrics.json')
    pipeline.export_metrics(metrics_file)
    print(f"[OK] Metrics exported to: {metrics_file}")
    print()
    
    # Generate detailed report
    print("="*80)
    print("DETAILED FRAME-BY-FRAME ANALYSIS")
    print("="*80)
    print()
    print(f"{'Frame':<8} {'Contours':<12} {'Objects':<12} {'Time (ms)':<12} {'Avg Area':<12}")
    print("-" * 60)
    
    for metrics in pipeline.frame_metrics_history:
        avg_area = np.mean([c.area for c in metrics.contour_metrics]) if metrics.contour_metrics else 0
        print(f"{metrics.frame_number:<8} {metrics.total_contours_detected:<12} " +
              f"{metrics.total_objects_tracked:<12} {metrics.processing_time_ms:<12.2f} {avg_area:<12.1f}")
    
    print()
    
    # Test adaptive thresholding analysis
    print("="*80)
    print("ADAPTIVE THRESHOLD ANALYSIS (First Frame)")
    print("="*80)
    if pipeline.frame_metrics_history:
        first_frame_metrics = pipeline.frame_metrics_history[0]
        print(f"Threshold Range: {first_frame_metrics.adaptive_threshold_range}")
    print()
    
    # Statistics
    print("="*80)
    print("CONTOUR STATISTICS")
    print("="*80)
    
    all_areas = []
    all_perimeters = []
    all_aspect_ratios = []
    all_circularities = []
    
    for metrics in pipeline.frame_metrics_history:
        for contour_metric in metrics.contour_metrics:
            all_areas.append(contour_metric.area)
            all_perimeters.append(contour_metric.perimeter)
            all_aspect_ratios.append(contour_metric.aspect_ratio)
            all_circularities.append(contour_metric.circularity)
    
    if all_areas:
        print(f"Total Contours Detected: {len(all_areas)}")
        print(f"Area Stats:")
        print(f"  Mean: {np.mean(all_areas):.1f}")
        print(f"  Std Dev: {np.std(all_areas):.1f}")
        print(f"  Min: {np.min(all_areas):.1f}")
        print(f"  Max: {np.max(all_areas):.1f}")
        print()
        print(f"Aspect Ratio Stats:")
        print(f"  Mean: {np.mean(all_aspect_ratios):.2f}")
        print(f"  Std Dev: {np.std(all_aspect_ratios):.2f}")
        print()
        print(f"Circularity Stats:")
        print(f"  Mean: {np.mean(all_circularities):.2f}")
        print(f"  Std Dev: {np.std(all_circularities):.2f}")
    
    print()
    print("="*80)
    print("TEST SUITE COMPLETION: SUCCESS")
    print("="*80)
    
    # Save output frames
    video_file = os.path.join(output_folder, 'cv_output.avi')
    if output_frames:
        h, w = output_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(video_file, fourcc, 30.0, (w, h))
        for frame in output_frames:
            out.write(frame)
        out.release()
        print(f"[OK] Output video saved to: {video_file}")
    
    return pipeline, summary


def test_frame_processing_details():
    """Test individual frame processing steps"""
    print("\n" + "="*80)
    print("DETAILED FRAME PROCESSING ANALYSIS")
    print("="*80 + "\n")
    
    # Create a simple test frame
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
    cv2.circle(frame, (200, 200), 50, (50, 100, 150), -1)
    cv2.circle(frame, (400, 300), 60, (100, 50, 100), -1)
    cv2.rectangle(frame, (50, 350), (150, 450), (150, 100, 50), -1)
    
    pipeline = CVPipeline()
    
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(f"[OK] Converted to grayscale: {gray.shape}")
    
    # Step 2: Apply Gaussian blur
    blurred = pipeline.apply_gaussian_filter(gray, sigma=1.0)
    print(f"[OK] Applied Gaussian blur (kernel={pipeline.blur_kernel_size})")
    
    # Step 3: Apply adaptive threshold
    thresholded, analysis = pipeline.apply_adaptive_threshold(blurred)
    print(f"[OK] Applied adaptive threshold")
    print(f"     Histogram Analysis:")
    for key, value in analysis.items():
        print(f"       {key}: {value}")
    
    # Step 4: Morphological operations
    processed = pipeline.apply_morphological_operations(thresholded)
    print(f"[OK] Applied morphological operations (kernel={pipeline.morph_kernel_size}x{pipeline.morph_kernel_size})")
    
    # Step 5: Detect contours
    contours = pipeline.detect_contours(processed)
    print(f"[OK] Detected {len(contours)} contours")
    
    # Step 6: Extract metrics
    for i, contour in enumerate(contours):
        metrics = pipeline.extract_contour_metrics(contour, i, 1, 0.0)
        print(f"\n     Contour {i}:")
        print(f"       Area: {metrics.area:.1f}")
        print(f"       Perimeter: {metrics.perimeter:.1f}")
        print(f"       Center: {metrics.center}")
        print(f"       Aspect Ratio: {metrics.aspect_ratio:.2f}")
        print(f"       Circularity: {metrics.circularity:.3f}")
    
    print()


if __name__ == '__main__':
    # Create Q4 output folder
    output_folder = r'c:\Users\ASA\Desktop\Progree Tasks\Q4'
    os.makedirs(output_folder, exist_ok=True)
    
    # Run main test
    pipeline, summary = test_cv_pipeline(output_folder)
    
    # Run detailed analysis
    test_frame_processing_details()
    
    print("\n[OK] All tests completed successfully!")
