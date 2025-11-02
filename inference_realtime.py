"""
Real-time face swapping from webcam
Optimized for 15-20 FPS performance
"""
import argparse
import cv2
import torch
import numpy as np
import time
import sys
import os

from utils.inference.image_processing import crop_face, normalize_and_torch
from utils.inference.realtime_processing import process_single_frame
from utils.realtime.face_tracker import FaceTracker
from utils.realtime.performance_monitor import PerformanceMonitor
from utils.realtime.camera_capture import CameraCapture

from network.AEI_Net import AEI_Net
from coordinate_reg.image_infer import Handler
from insightface_func.face_detect_crop_multi import Face_detect_crop
from arcface_model.iresnet import iresnet100


def init_models(args):
    """Initialize all models for face swapping"""
    print("Loading models...")
    
    # Face detection model
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    # Use smaller detection size for faster processing
    det_size = (320, 320) if args.fast_mode else (640, 640)
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=det_size)
    
    # Generator model
    G = AEI_Net(args.backbone, num_blocks=args.num_blocks, c_id=512)
    G.eval()
    G.load_state_dict(torch.load(args.G_path, map_location=torch.device('cpu')))
    G = G.cuda()
    G = G.half()  # Use FP16 for speed
    
    # ArcFace model for embeddings
    netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load('arcface_model/backbone.pth'))
    netArc = netArc.cuda()
    netArc.eval()
    
    # Landmark detection handler
    handler = Handler('./coordinate_reg/model/2d106det', 0, ctx_id=0, det_size=640)
    
    print("Models loaded successfully!")
    return app, G, netArc, handler


def load_source_face(source_path: str, app, netArc, crop_size: int = 224):
    """Load and process source face, return embedding"""
    print(f"Loading source face from {source_path}...")
    
    source_img = cv2.imread(source_path)
    if source_img is None:
        raise ValueError(f"Could not read source image: {source_path}")
    
    # Crop face
    face_crop = crop_face(source_img, app, crop_size)
    if not face_crop or len(face_crop) == 0:
        raise ValueError(f"No face detected in source image: {source_path}")
    
    # Get embedding (compute once, reuse for all frames)
    source_normalized = normalize_and_torch(face_crop[0])
    import torch.nn.functional as F
    source_embed = netArc(F.interpolate(source_normalized, scale_factor=0.5, mode='bilinear', align_corners=True))
    
    print("Source face loaded and embedded!")
    return source_embed


def draw_info(frame: np.ndarray, monitor: PerformanceMonitor, tracker: FaceTracker):
    """Draw performance info and face status on frame"""
    stats = monitor.get_stats()
    tracker_stats = tracker.get_stats()
    
    # Background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # FPS
    fps_color = (0, 255, 0) if stats['fps'] >= 15 else (0, 165, 255) if stats['fps'] >= 10 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {stats['fps']:.1f}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
    
    # Latency
    cv2.putText(frame, f"Latency: {stats['avg_latency_ms']:.1f}ms", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Detection time
    cv2.putText(frame, f"Detection: {stats['avg_detection_ms']:.1f}ms", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Face status
    face_status = "Face: Detected" if tracker_stats['has_face'] else "Face: Searching..."
    face_color = (0, 255, 0) if tracker_stats['has_face'] else (0, 0, 255)
    cv2.putText(frame, face_status, (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 1)
    
    return frame


def main(args):
    """Main real-time face swapping loop"""
    # Initialize models
    app, G, netArc, handler = init_models(args)
    
    # Load source face and get embedding (compute once)
    source_embed = load_source_face(args.source_path, app, netArc, args.crop_size)
    
    # Initialize tracker
    tracker = FaceTracker(
        detector=app,
        detect_interval=args.detect_interval,
        tracker_type=args.tracker_type,
        confidence_threshold=0.6
    )
    
    # Initialize performance monitor
    monitor = PerformanceMonitor(window_size=30)
    
    # Open camera
    print(f"Opening camera {args.camera_id}...")
    camera = CameraCapture(camera_id=args.camera_id, width=args.width, height=args.height)
    
    if not camera.open():
        print(f"Error: Could not open camera {args.camera_id}")
        print("Try different camera IDs: --camera_id 0, 1, 2, ...")
        return
    
    print("\n" + "="*50)
    print("Real-time Face Swapping Started!")
    print("="*50)
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reset tracker")
    print("  - Press 's' to save current frame")
    print("="*50 + "\n")
    
    frame_count = 0
    saved_count = 0
    
    try:
        while True:
            monitor.start_frame()
            
            # Read frame
            ret, frame = camera.read()
            if not ret or frame is None:
                print("Error reading frame from camera")
                break
            
            # Update tracker
            bbox = tracker.update(frame)
            
            # Process frame
            det_time = 0
            gen_time = 0
            if bbox is not None:
                result, det_time, gen_time = process_single_frame(
                    frame=frame,
                    source_embed=source_embed,
                    netArc=netArc,
                    G=G,
                    app=app,
                    handler=handler,
                    bbox=bbox,
                    crop_size=args.crop_size,
                    half=True
                )
                
                if result is not None:
                    # Draw bbox
                    x, y, w, h = bbox
                    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    display_frame = result
                else:
                    display_frame = frame
            else:
                display_frame = frame
            
            # Record metrics
            monitor.record_detection_time(det_time)
            monitor.record_generator_time(gen_time)
            total_time = (time.time() - monitor.frame_start_time) * 1000 if monitor.frame_start_time else 0
            monitor.record_processing_time(total_time)
            monitor.end_frame()
            
            # Draw info
            display_frame = draw_info(display_frame, monitor, tracker)
            
            # Display
            cv2.imshow('Real-time Face Swap', display_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                tracker.reset()
                print("Tracker reset!")
            elif key == ord('s'):
                save_path = f'examples/results/realtime_frame_{saved_count:04d}.jpg'
                cv2.imwrite(save_path, display_frame)
                saved_count += 1
                print(f"Frame saved to {save_path}")
            
            frame_count += 1
            
            # Print stats every 30 frames
            if frame_count % 30 == 0:
                monitor.print_stats()
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera.release()
        cv2.destroyAllWindows()
        
        # Final stats
        print("\n" + "="*50)
        print("Final Statistics:")
        print("="*50)
        monitor.print_stats()
        print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time face swapping from webcam')
    
    # Model parameters
    parser.add_argument('--G_path', default='weights/G_unet_2blocks.pth', type=str,
                        help='Path to generator weights')
    parser.add_argument('--backbone', default='unet', choices=['unet', 'linknet', 'resnet'],
                        help='Generator backbone')
    parser.add_argument('--num_blocks', default=2, type=int,
                        help='Number of AddBlocks (1=faster, 2=better quality, 3=best quality)')
    
    # Source face
    parser.add_argument('--source_path', default='examples/images/mark.jpg', type=str,
                        help='Path to source face image')
    
    # Camera settings
    parser.add_argument('--camera_id', default=0, type=int,
                        help='Camera device ID')
    parser.add_argument('--width', default=640, type=int,
                        help='Camera frame width')
    parser.add_argument('--height', default=480, type=int,
                        help='Camera frame height')
    
    # Processing settings
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Face crop size (don\'t change)')
    parser.add_argument('--detect_interval', default=5, type=int,
                        help='Run full detection every N frames (higher = faster)')
    parser.add_argument('--tracker_type', default='CSRT', choices=['CSRT', 'KCF', 'MOSSE'],
                        help='Face tracker type (CSRT=best, KCF=faster, MOSSE=fastest)')
    parser.add_argument('--fast_mode', action='store_true',
                        help='Use faster detection (320x320 instead of 640x640)')
    
    args = parser.parse_args()
    
    # Validate source image exists
    if not os.path.exists(args.source_path):
        print(f"Error: Source image not found: {args.source_path}")
        sys.exit(1)
    
    main(args)

