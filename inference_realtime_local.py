"""
Local Real-Time Face Swapping Script
=====================================
Run this on your local computer for true continuous streaming.

Features:
- True continuous webcam streaming (no permission delays)
- Higher FPS than Colab (10-30 FPS)
- OpenCV window display
- Performance monitoring
- Press 'q' to quit

Usage:
    python inference_realtime_local.py \
      --source_path examples/images/mark.jpg \
      --camera_index 0 \
      --num_blocks 2 \
      --detect_interval 5

Requirements:
- Webcam connected
- GPU recommended (works on CPU but slower)
- All models downloaded (run download_models.sh)
"""

import argparse
import sys
import cv2
import torch
import time
import numpy as np

# Add project to path
sys.path.insert(0, '.')

from inference_realtime import init_models, load_source_face
from utils.realtime.face_tracker import FaceTracker
from utils.inference.realtime_processing import process_single_frame
from utils.realtime.performance_monitor import PerformanceMonitor


def main():
    parser = argparse.ArgumentParser(description='Real-time face swapping with webcam')
    
    # Required arguments
    parser.add_argument('--source_path', type=str, required=True,
                       help='Path to source face image')
    parser.add_argument('--camera_index', type=int, default=0,
                       help='Camera index (0 for default webcam)')
    
    # Model arguments
    parser.add_argument('--G_path', type=str, default='weights/G_unet_2blocks.pth',
                       help='Path to generator model')
    parser.add_argument('--backbone', type=str, default='unet', choices=['unet', 'linknet'],
                       help='Generator backbone')
    parser.add_argument('--num_blocks', type=int, default=2, choices=[1, 2, 3],
                       help='Number of generator blocks (1=faster, 2=balanced, 3=best quality)')
    
    # Processing arguments
    parser.add_argument('--crop_size', type=int, default=224,
                       help='Face crop size')
    parser.add_argument('--detect_interval', type=int, default=5,
                       help='Face detection interval (frames between full detections)')
    parser.add_argument('--tracker_type', type=str, default='CSRT',
                       choices=['CSRT', 'KCF', 'MOSSE'],
                       help='Tracker type')
    parser.add_argument('--confidence_threshold', type=float, default=0.6,
                       help='Face detection confidence threshold')
    parser.add_argument('--half', action='store_true',
                       help='Use FP16 for faster inference')
    
    # Display arguments
    parser.add_argument('--display_width', type=int, default=1280,
                       help='Display window width')
    parser.add_argument('--display_height', type=int, default=720,
                       help='Display window height')
    parser.add_argument('--show_stats', action='store_true', default=True,
                       help='Show performance statistics on display')
    parser.add_argument('--save_output', type=str, default=None,
                       help='Path to save output video (optional)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🎭 Real-Time Face Swapping - Local")
    print("="*60)
    print(f"Source face: {args.source_path}")
    print(f"Camera index: {args.camera_index}")
    print(f"Generator: {args.backbone} with {args.num_blocks} blocks")
    print(f"Detection interval: {args.detect_interval} frames")
    print("="*60)
    
    # Check if source image exists
    import os
    if not os.path.exists(args.source_path):
        print(f"❌ Error: Source image not found: {args.source_path}")
        sys.exit(1)
    
    # Initialize models
    print("\n📦 Loading models... This may take 1-2 minutes...")
    try:
        app, G, netArc, handler = init_models(args)
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load source face embedding
    print(f"\n📸 Loading source face: {args.source_path}")
    try:
        source_embed = load_source_face(
            args.source_path,
            app,
            netArc,
            args.crop_size
        )
        print("✅ Source face loaded!")
    except Exception as e:
        print(f"❌ Error loading source face: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Initialize face tracker
    print("\n🔍 Initializing face tracker...")
    tracker = FaceTracker(
        detector=app,
        detect_interval=args.detect_interval,
        tracker_type=args.tracker_type,
        confidence_threshold=args.confidence_threshold
    )
    
    # Initialize performance monitor
    monitor = PerformanceMonitor(window_size=30)
    
    # Initialize camera
    print(f"\n📹 Opening camera {args.camera_index}...")
    cap = cv2.VideoCapture(args.camera_index)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {args.camera_index}")
        print("   Check if camera is connected and not used by another app")
        sys.exit(1)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.display_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.display_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual camera properties
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ Camera opened: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
    
    # Initialize video writer if saving
    video_writer = None
    if args.save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.save_output,
            fourcc,
            30.0,
            (actual_width, actual_height)
        )
        print(f"💾 Saving output to: {args.save_output}")
    
    print("\n" + "="*60)
    print("🎬 Starting real-time face swapping...")
    print("="*60)
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("Press 'r' to reset tracker (force re-detection)")
    print("="*60 + "\n")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Failed to read frame from camera")
                break
            
            # Flip horizontally for mirror effect (optional)
            frame = cv2.flip(frame, 1)
            
            # Start monitoring
            monitor.start_frame()
            
            # Update tracker
            bbox = tracker.update(frame)
            
            # Process frame
            det_time = 0
            gen_time = 0
            result = None
            
            if bbox is not None:
                # Face detected - apply face swap
                try:
                    result, det_time, gen_time = process_single_frame(
                        frame=frame,
                        source_embed=source_embed,
                        netArc=netArc,
                        G=G,
                        app=app,
                        handler=handler,
                        bbox=bbox,
                        crop_size=args.crop_size,
                        half=args.half
                    )
                except Exception as e:
                    print(f"⚠️  Error processing frame: {e}")
                    result = frame
            else:
                # No face detected - show original frame
                result = frame.copy()
                cv2.putText(result, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Record metrics
            monitor.record_detection_time(det_time)
            monitor.record_generator_time(gen_time)
            total_time = (time.time() - monitor.frame_start_time) * 1000 if monitor.frame_start_time else 0
            monitor.record_processing_time(total_time)
            monitor.end_frame()
            
            # Draw bounding box if face detected
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(result, "Face Detected", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display statistics
            if args.show_stats:
                stats = monitor.get_stats()
                y_offset = 30
                line_height = 25
                
                # FPS
                cv2.putText(result, f"FPS: {stats['fps']:.1f}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += line_height
                
                # Latency
                cv2.putText(result, f"Latency: {stats['avg_latency_ms']:.1f}ms", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += line_height
                
                # Detection time
                cv2.putText(result, f"Detection: {stats['avg_detection_ms']:.1f}ms", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += line_height
                
                # Generator time
                cv2.putText(result, f"Generator: {stats['avg_generator_ms']:.1f}ms", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += line_height
                
                # Frame count
                cv2.putText(result, f"Frames: {frame_count}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Save frame to video if requested
            if video_writer is not None:
                video_writer.write(result)
            
            # Display frame
            cv2.imshow('Real-Time Face Swap', result)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('s'):
                # Save current frame
                filename = f'output_frame_{int(time.time())}.jpg'
                cv2.imwrite(filename, result)
                print(f"💾 Saved frame to: {filename}")
            elif key == ord('r'):
                # Reset tracker
                tracker.reset()
                print("🔄 Tracker reset - re-detecting face...")
            
            frame_count += 1
            
            # Print stats every 30 frames
            if frame_count % 30 == 0:
                stats = monitor.get_stats()
                elapsed = time.time() - start_time
                print(f"Frame {frame_count} | FPS: {stats['fps']:.1f} | "
                      f"Latency: {stats['avg_latency_ms']:.1f}ms | "
                      f"Detection: {stats['avg_detection_ms']:.1f}ms | "
                      f"Generator: {stats['avg_generator_ms']:.1f}ms")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
        
        # Print final statistics
        print("\n" + "="*60)
        print("📊 Final Statistics")
        print("="*60)
        stats = monitor.get_stats()
        elapsed = time.time() - start_time
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Average FPS: {stats['fps']:.2f}")
        print(f"Average latency: {stats['avg_latency_ms']:.1f}ms")
        print(f"Average detection time: {stats['avg_detection_ms']:.1f}ms")
        print(f"Average generator time: {stats['avg_generator_ms']:.1f}ms")
        print("="*60)


if __name__ == '__main__':
    main()

