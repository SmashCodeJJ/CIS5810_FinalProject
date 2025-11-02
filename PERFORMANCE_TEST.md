# 🎯 Real-Time Performance Testing Guide

## Overview

This guide explains how to test and measure the performance of real-time face swapping.

---

## 📊 Performance Metrics

### Key Metrics to Measure:
- **FPS (Frames Per Second)**: Target 15-20 FPS
- **Latency**: Target <70ms per frame
- **Detection Time**: Should be ~4ms (with tracking) or ~20ms (full detection)
- **Generator Time**: ~90-120ms (main bottleneck)
- **Memory Usage**: GPU VRAM consumption
- **CPU Usage**: Should be minimal (GPU does heavy lifting)

---

## 🧪 Method 1: Local Testing (Webcam)

### Prerequisites
- Webcam connected
- GPU available (recommended)
- Models downloaded

### Test Script

```python
# test_realtime_performance.py
python inference_realtime.py \
    --source_path examples/images/mark.jpg \
    --camera_id 0 \
    --fast_mode \
    --num_blocks 2
```

### What to Look For:
- **FPS Counter**: Should show 12-18 FPS on screen
- **Latency**: Displayed in top-left corner
- **Smoothness**: Video should be relatively smooth (not stuttering)

### Expected Performance:

| Configuration | FPS | Latency | Quality |
|--------------|-----|---------|---------|
| Default (2 blocks, 640x640) | 12-15 | 60-80ms | High |
| Fast mode | 15-18 | 50-70ms | High |
| 1 block + fast | 18-22 | 40-60ms | Medium |
| Lower resolution (320x240) | 20-25 | 35-50ms | Medium |

---

## 🧪 Method 2: Colab Testing (Frame-by-Frame)

### Step 1: Open Colab Notebook
```
SberSwap_Realtime_Colab.ipynb
```

### Step 2: Run Performance Test Cell

```python
import time
import sys
sys.path.insert(0, '/content/sber-swap')

from utils.inference.realtime_processing import process_single_frame
from utils.realtime.face_tracker import FaceTracker
from utils.realtime.performance_monitor import PerformanceMonitor
from inference_realtime import init_models, load_source_face

# Initialize models (run once)
class Args:
    def __init__(self):
        self.G_path = 'weights/G_unet_2blocks.pth'
        self.backbone = 'unet'
        self.num_blocks = 2
        self.fast_mode = True
        self.crop_size = 224
        self.detect_interval = 5
        self.tracker_type = 'CSRT'

args = Args()
app, G, netArc, handler = init_models(args)
source_embed = load_source_face('/content/sber-swap/examples/images/mark.jpg', app, netArc, 224)
tracker = FaceTracker(detector=app, detect_interval=5, tracker_type='CSRT', confidence_threshold=0.6)
monitor = PerformanceMonitor(window_size=30)

print("✅ Models loaded!")
```

### Step 3: Capture and Test Multiple Frames

```python
# Test with 10 frames
test_results = []

for i in range(10):
    # Capture frame
    frame = take_photo()
    
    # Start timing
    monitor.start_frame()
    start_time = time.time()
    
    # Process
    bbox = tracker.update(frame)
    if bbox is not None:
        result, det_time, gen_time = process_single_frame(
            frame=frame,
            source_embed=source_embed,
            netArc=netArc,
            G=G,
            app=app,
            handler=handler,
            bbox=bbox,
            crop_size=224,
            half=True
        )
        
        total_time = (time.time() - start_time) * 1000
        
        monitor.record_detection_time(det_time)
        monitor.record_generator_time(gen_time)
        monitor.record_processing_time(total_time)
        monitor.end_frame()
        
        test_results.append({
            'frame': i+1,
            'total_ms': total_time,
            'detection_ms': det_time,
            'generator_ms': gen_time,
            'fps': 1000/total_time if total_time > 0 else 0
        })
        
        print(f"Frame {i+1}: {total_time:.1f}ms ({1000/total_time:.1f} FPS)")

# Print summary
if test_results:
    avg_fps = sum(r['fps'] for r in test_results) / len(test_results)
    avg_latency = sum(r['total_ms'] for r in test_results) / len(test_results)
    avg_det = sum(r['detection_ms'] for r in test_results) / len(test_results)
    avg_gen = sum(r['generator_ms'] for r in test_results) / len(test_results)
    
    print("\n" + "="*50)
    print("📊 PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Average Latency: {avg_latency:.1f}ms")
    print(f"Average Detection: {avg_det:.1f}ms")
    print(f"Average Generator: {avg_gen:.1f}ms")
    print("="*50)
```

---

## 🧪 Method 3: Automated Performance Benchmark

### Create Benchmark Script

```python
# benchmark_realtime.py
import time
import cv2
import numpy as np
import sys
from utils.inference.realtime_processing import process_single_frame
from utils.realtime.face_tracker import FaceTracker
from inference_realtime import init_models, load_source_face

class Args:
    def __init__(self):
        self.G_path = 'weights/G_unet_2blocks.pth'
        self.backbone = 'unet'
        self.num_blocks = 2
        self.fast_mode = True
        self.crop_size = 224
        self.detect_interval = 5
        self.tracker_type = 'CSRT'

def benchmark(num_frames=100):
    """Run performance benchmark"""
    print("Initializing models...")
    args = Args()
    app, G, netArc, handler = init_models(args)
    source_embed = load_source_face('examples/images/mark.jpg', app, netArc, 224)
    tracker = FaceTracker(detector=app, detect_interval=5, tracker_type='CSRT')
    
    # Load test video frame or create synthetic face image
    test_image = cv2.imread('examples/images/beckham.jpg')
    
    if test_image is None:
        print("Error: Test image not found")
        return
    
    print(f"\nRunning benchmark with {num_frames} frames...")
    
    times = []
    detection_times = []
    generator_times = []
    
    for i in range(num_frames):
        # Update tracker
        bbox = tracker.update(test_image)
        
        if bbox is not None:
            start = time.time()
            _, det_time, gen_time = process_single_frame(
                frame=test_image,
                source_embed=source_embed,
                netArc=netArc,
                G=G,
                app=app,
                handler=handler,
                bbox=bbox,
                crop_size=224,
                half=True
            )
            total_time = (time.time() - start) * 1000
            
            times.append(total_time)
            detection_times.append(det_time)
            generator_times.append(gen_time)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{num_frames} frames...")
    
    # Calculate statistics
    if times:
        avg_time = np.mean(times)
        avg_fps = 1000 / avg_time
        min_fps = 1000 / np.max(times)
        max_fps = 1000 / np.min(times)
        
        print("\n" + "="*60)
        print("📊 BENCHMARK RESULTS")
        print("="*60)
        print(f"Frames Processed: {len(times)}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Min FPS: {min_fps:.2f}")
        print(f"Max FPS: {max_fps:.2f}")
        print(f"\nAverage Latency: {avg_time:.1f}ms")
        print(f"Min Latency: {np.min(times):.1f}ms")
        print(f"Max Latency: {np.max(times):.1f}ms")
        print(f"\nAverage Detection Time: {np.mean(detection_times):.1f}ms")
        print(f"Average Generator Time: {np.mean(generator_times):.1f}ms")
        print("="*60)

if __name__ == "__main__":
    benchmark(num_frames=100)
```

### Run Benchmark:
```bash
python benchmark_realtime.py
```

---

## 🧪 Method 4: Using Test Script with Video File

### Convert Video to Frames and Test

```python
import cv2
import time
from inference_realtime import init_models, load_source_face
from utils.inference.realtime_processing import process_single_frame
from utils.realtime.face_tracker import FaceTracker

# Initialize
args = Args()
app, G, netArc, handler = init_models(args)
source_embed = load_source_face('examples/images/mark.jpg', app, netArc, 224)
tracker = FaceTracker(detector=app, detect_interval=5, tracker_type='CSRT')

# Load video
cap = cv2.VideoCapture('examples/videos/test_video.mp4')
frame_count = 0
times = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    start = time.time()
    
    # Process frame
    bbox = tracker.update(frame)
    if bbox is not None:
        result, _, _ = process_single_frame(
            frame=frame,
            source_embed=source_embed,
            netArc=netArc,
            G=G,
            app=app,
            handler=handler,
            bbox=bbox,
            crop_size=224,
            half=True
        )
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
    
    if frame_count >= 100:  # Test 100 frames
        break

cap.release()

# Print results
if times:
    avg_fps = 1000 / (sum(times) / len(times))
    print(f"Average FPS: {avg_fps:.2f}")
```

---

## 📈 Performance Analysis

### What Good Performance Looks Like:

```
✅ GOOD:
- FPS: 15-20
- Latency: 50-70ms
- Detection: 4-8ms (with tracking)
- Generator: 90-120ms
- Smooth video playback

⚠️ ACCEPTABLE:
- FPS: 10-15
- Latency: 70-100ms
- Some stuttering but usable

❌ NEEDS OPTIMIZATION:
- FPS: <10
- Latency: >100ms
- Severe stuttering
```

### Bottleneck Analysis:

1. **If Detection Time > 20ms**: Use fast mode or increase detect_interval
2. **If Generator Time > 120ms**: Use 1-block model or lower resolution
3. **If Total Time > 100ms**: Check GPU utilization, reduce resolution

---

## 🔧 Optimization Tips

### If FPS is Low:

1. **Enable Fast Mode**: `--fast_mode`
2. **Use 1-Block Generator**: `--num_blocks 1`
3. **Increase Detection Interval**: `--detect_interval 10`
4. **Lower Resolution**: `--width 320 --height 240`
5. **Use KCF Tracker**: `--tracker_type KCF` (faster than CSRT)

### If Quality is Poor:

1. **Use 2-Block Generator**: `--num_blocks 2`
2. **Higher Resolution**: `--width 640 --height 480`
3. **Lower Detection Interval**: `--detect_interval 3`
4. **Use CSRT Tracker**: `--tracker_type CSRT` (more accurate)

---

## 📝 Performance Report Template

After testing, document your results:

```
System: [CPU/GPU specs]
Config: [Fast mode, 2 blocks, etc.]
Frames Tested: [number]

Results:
- Average FPS: [X]
- Average Latency: [X]ms
- Detection Time: [X]ms
- Generator Time: [X]ms
- Min FPS: [X]
- Max FPS: [X]

Notes: [Any issues or observations]
```

---

## 🚀 Quick Test Commands

### Local (Webcam):
```bash
python inference_realtime.py --source_path examples/images/mark.jpg --fast_mode
```

### Local (Static Image Test):
```bash
python test_realtime.py --source_path examples/images/mark.jpg --test_image examples/images/beckham.jpg
```

### Colab:
Use `SberSwap_Realtime_Colab.ipynb` and run the performance test cells.

---

## ✅ Success Criteria

**Target Performance (Phase 2 Goal):**
- ✅ 15-18 FPS average
- ✅ <70ms latency
- ✅ Smooth video playback
- ✅ Acceptable quality

**Stretch Goal (Phase 3):**
- ✅ 20-25 FPS average
- ✅ <50ms latency
- ✅ Professional quality

---

See `REALTIME_README.md` for more details!

