"""
Performance benchmark script for real-time face swapping
Tests and reports FPS, latency, and component timings
"""
import time
import cv2
import numpy as np
import sys
import argparse
from typing import List, Dict

from utils.inference.realtime_processing import process_single_frame
from utils.realtime.face_tracker import FaceTracker
from inference_realtime import init_models, load_source_face


def benchmark_static_image(
    source_path: str,
    test_image_path: str,
    num_frames: int = 100,
    fast_mode: bool = True,
    num_blocks: int = 2,
    detect_interval: int = 5
) -> Dict:
    """Benchmark with static image (simulates video frames)"""
    print("="*60)
    print("🎯 Real-Time Face Swapping Performance Benchmark")
    print("="*60)
    print(f"Source: {source_path}")
    print(f"Test Image: {test_image_path}")
    print(f"Frames: {num_frames}")
    print(f"Fast Mode: {fast_mode}")
    print(f"Generator Blocks: {num_blocks}")
    print("="*60)
    
    # Initialize models
    print("\n[1/4] Initializing models...")
    class Args:
        def __init__(self):
            self.G_path = 'weights/G_unet_2blocks.pth'
            self.backbone = 'unet'
            self.num_blocks = num_blocks
            self.fast_mode = fast_mode
            self.crop_size = 224
            self.detect_interval = detect_interval
            self.tracker_type = 'CSRT'
    
    args = Args()
    try:
        app, G, netArc, handler = init_models(args)
        print("✅ Models initialized!")
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        return None
    
    # Load source face
    print("[2/4] Loading source face...")
    try:
        source_embed = load_source_face(source_path, app, netArc, args.crop_size)
        print("✅ Source face loaded!")
    except Exception as e:
        print(f"❌ Error loading source face: {e}")
        return None
    
    # Load test image
    print("[3/4] Loading test image...")
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print(f"❌ Error: Could not read test image: {test_image_path}")
        return None
    print(f"✅ Test image loaded! Shape: {test_image.shape}")
    
    # Initialize tracker
    print("[4/4] Initializing tracker...")
    tracker = FaceTracker(
        detector=app,
        detect_interval=detect_interval,
        tracker_type='CSRT',
        confidence_threshold=0.6
    )
    print("✅ Tracker initialized!")
    
    # Run benchmark
    print(f"\n🚀 Running benchmark with {num_frames} frames...")
    print("="*60)
    
    times = []
    detection_times = []
    generator_times = []
    successful_frames = 0
    
    for i in range(num_frames):
        # Update tracker
        bbox = tracker.update(test_image)
        
        if bbox is not None:
            start = time.time()
            try:
                _, det_time, gen_time = process_single_frame(
                    frame=test_image,
                    source_embed=source_embed,
                    netArc=netArc,
                    G=G,
                    app=app,
                    handler=handler,
                    bbox=bbox,
                    crop_size=args.crop_size,
                    half=True
                )
                total_time = (time.time() - start) * 1000
                
                times.append(total_time)
                detection_times.append(det_time)
                generator_times.append(gen_time)
                successful_frames += 1
                
            except Exception as e:
                print(f"⚠️  Error processing frame {i+1}: {e}")
        else:
            print(f"⚠️  No face detected in frame {i+1}")
        
        # Progress update
        if (i + 1) % 10 == 0:
            if times:
                avg_fps = 1000 / np.mean(times)
                print(f"Progress: {i+1}/{num_frames} frames | Avg FPS: {avg_fps:.1f}")
    
    # Calculate statistics
    if not times:
        print("\n❌ No frames were successfully processed!")
        return None
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    avg_fps = 1000 / avg_time
    min_fps = 1000 / max_time
    max_fps = 1000 / min_time
    
    avg_det = np.mean(detection_times)
    avg_gen = np.mean(generator_times)
    
    # Print results
    print("\n" + "="*60)
    print("📊 BENCHMARK RESULTS")
    print("="*60)
    print(f"Frames Processed: {successful_frames}/{num_frames}")
    print(f"Success Rate: {100*successful_frames/num_frames:.1f}%")
    print("")
    print("FPS Statistics:")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Min FPS:     {min_fps:.2f}")
    print(f"  Max FPS:     {max_fps:.2f}")
    print("")
    print("Latency Statistics:")
    print(f"  Average: {avg_time:.1f}ms")
    print(f"  Std Dev: {std_time:.1f}ms")
    print(f"  Min:     {min_time:.1f}ms")
    print(f"  Max:     {max_time:.1f}ms")
    print("")
    print("Component Times:")
    print(f"  Detection: {avg_det:.1f}ms (avg)")
    print(f"  Generator: {avg_gen:.1f}ms (avg)")
    print(f"  Other:      {avg_time - avg_det - avg_gen:.1f}ms (avg)")
    print("="*60)
    
    # Performance assessment
    print("\n📈 PERFORMANCE ASSESSMENT:")
    if avg_fps >= 15:
        print("✅ EXCELLENT: Real-time performance achieved!")
    elif avg_fps >= 12:
        print("✅ GOOD: Acceptable for demos and presentations")
    elif avg_fps >= 8:
        print("⚠️  ACCEPTABLE: Usable but noticeable lag")
    else:
        print("❌ NEEDS OPTIMIZATION: Too slow for real-time")
    
    print(f"\nTarget: 15-18 FPS | Your Result: {avg_fps:.1f} FPS")
    print("="*60)
    
    return {
        'avg_fps': avg_fps,
        'min_fps': min_fps,
        'max_fps': max_fps,
        'avg_latency_ms': avg_time,
        'avg_detection_ms': avg_det,
        'avg_generator_ms': avg_gen,
        'success_rate': successful_frames / num_frames,
        'frames_processed': successful_frames
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark real-time face swapping performance')
    
    parser.add_argument('--source_path', default='examples/images/mark.jpg', type=str,
                        help='Path to source face image')
    parser.add_argument('--test_image', default='examples/images/beckham.jpg', type=str,
                        help='Path to test image (target)')
    parser.add_argument('--num_frames', default=100, type=int,
                        help='Number of frames to process')
    parser.add_argument('--fast_mode', action='store_true',
                        help='Enable fast mode (smaller detection)')
    parser.add_argument('--num_blocks', default=2, type=int,
                        help='Number of generator blocks (1=faster, 2=better quality)')
    parser.add_argument('--detect_interval', default=5, type=int,
                        help='Detection interval for tracking')
    
    args = parser.parse_args()
    
    result = benchmark_static_image(
        source_path=args.source_path,
        test_image_path=args.test_image,
        num_frames=args.num_frames,
        fast_mode=args.fast_mode,
        num_blocks=args.num_blocks,
        detect_interval=args.detect_interval
    )
    
    if result is None:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()

