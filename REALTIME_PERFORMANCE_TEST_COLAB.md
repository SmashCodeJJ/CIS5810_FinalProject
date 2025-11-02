# 🚀 Real-Time Performance Testing in Colab

## Overview

This guide shows how to test real-time face swapping performance in Google Colab, including FPS measurement and optimization testing.

---

## ⚡ Quick Performance Test

### Step 1: Setup (Run Once)

```python
# Clone repository with real-time implementation
!git clone -b Youxin https://github.com/SmashCodeJJ/CIS5810_FinalProject.git sber-swap
%cd sber-swap

# Install dependencies
%pip install -q -r requirements.txt

print("✅ Setup complete! Restart runtime, then continue.")
```

**⚠️ RESTART RUNTIME** after this cell!

### Step 2: Load Models and Test Performance

```python
import sys
import torch
import time
import numpy as np
import cv2
from IPython.display import display, Image, clear_output
import matplotlib.pyplot as plt

# Add to path
sys.path.insert(0, '/content/sber-swap')

# Import modules
from inference_realtime import init_models, load_source_face
from utils.realtime.face_tracker import FaceTracker
from utils.inference.realtime_processing import process_single_frame
from utils.realtime.performance_monitor import PerformanceMonitor

# Initialize models
print("Loading models... This may take 1-2 minutes...")

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

# Load source face
source_path = '/content/sber-swap/examples/images/mark.jpg'
source_embed = load_source_face(source_path, app, netArc, args.crop_size)

# Initialize tracker and monitor
tracker = FaceTracker(detector=app, detect_interval=5, tracker_type='CSRT', confidence_threshold=0.6)
monitor = PerformanceMonitor(window_size=30)

print("✅ Models loaded! Ready for performance testing.")
```

### Step 3: Performance Test with Webcam

```python
import base64
import io
from PIL import Image
from IPython.display import Javascript
from google.colab.output import eval_js

def take_photo():
    """Capture photo from webcam"""
    js = Javascript('''
        async function takePhoto() {
            const video = document.createElement('video');
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            video.srcObject = stream;
            video.play();
            await new Promise(resolve => {
                video.onloadedmetadata = () => {
                    video.setAttribute('width', video.videoWidth);
                    video.setAttribute('height', video.videoHeight);
                    resolve();
                }
            });
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            video.srcObject.getTracks().forEach(track => track.stop());
            return canvas.toDataURL('image/jpeg', 0.95);
        }
        takePhoto();
    ''')
    data = eval_js(js)
    image_bytes = base64.b64decode(data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes))
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

# Test multiple frames
print("📸 Capture multiple frames to test performance...")
print("Click this cell multiple times (recommended: 10-20 times)")

# Capture and process frame
frame = take_photo()

# Start monitoring
monitor.start_frame()

# Update tracker
bbox = tracker.update(frame)

# Process frame
det_time = 0
gen_time = 0
result = None

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

# Record metrics
monitor.record_detection_time(det_time)
monitor.record_generator_time(gen_time)
total_time = (time.time() - monitor.frame_start_time) * 1000 if monitor.frame_start_time else 0
monitor.record_processing_time(total_time)
monitor.end_frame()

# Display result
if result is not None:
    # Add stats overlay
    stats = monitor.get_stats()
    cv2.putText(result, f"FPS: {stats['fps']:.1f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(result, f"Latency: {stats['avg_latency_ms']:.1f}ms", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    display_frame = result
else:
    display_frame = frame
    cv2.putText(display_frame, "No face detected", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Save and display
cv2.imwrite('/content/sber-swap/examples/results/test_frame.jpg', display_frame)
display(Image('/content/sber-swap/examples/results/test_frame.jpg'))

# Print performance stats
stats = monitor.get_stats()
print("\n" + "="*60)
print("📊 PERFORMANCE STATISTICS")
print("="*60)
print(f"FPS:              {stats['fps']:.2f} frames/second")
print(f"Avg Latency:      {stats['avg_latency_ms']:.1f} ms")
print(f"Detection Time:   {stats['avg_detection_ms']:.1f} ms")
print(f"Generator Time:  {stats['avg_generator_ms']:.1f} ms")
print(f"Total Frames:    {stats['total_frames']}")
print("="*60)

# Performance evaluation
if stats['fps'] >= 15:
    print("✅ EXCELLENT - Real-time performance achieved!")
elif stats['fps'] >= 10:
    print("✅ GOOD - Acceptable for real-time use")
elif stats['fps'] >= 5:
    print("⚠️  FAIR - Some lag may be noticeable")
else:
    print("❌ SLOW - Not suitable for real-time")

print("\n💡 Tip: Run this cell multiple times (10-20x) for accurate average FPS")
```

---

## 📊 Batch Performance Test (More Accurate)

Test with multiple frames from webcam for better statistics:

```python
# Batch performance test
num_frames = 10  # Test with 10 frames
print(f"🧪 Testing performance with {num_frames} frames...")
print("You'll need to capture frames interactively...")

frames_processed = 0
total_start = time.time()

for i in range(num_frames):
    print(f"\n📸 Frame {i+1}/{num_frames} - Please allow webcam access...")
    
    # Capture frame
    frame = take_photo()
    
    # Process
    monitor.start_frame()
    bbox = tracker.update(frame)
    
    if bbox is not None:
        result, det_time, gen_time = process_single_frame(
            frame=frame, source_embed=source_embed, netArc=netArc,
            G=G, app=app, handler=handler, bbox=bbox,
            crop_size=args.crop_size, half=True
        )
    else:
        det_time = 0
        gen_time = 0
    
    monitor.record_detection_time(det_time)
    monitor.record_generator_time(gen_time)
    total_time = (time.time() - monitor.frame_start_time) * 1000 if monitor.frame_start_time else 0
    monitor.record_processing_time(total_time)
    monitor.end_frame()
    
    frames_processed += 1
    
    # Show progress
    stats = monitor.get_stats()
    print(f"   Current FPS: {stats['fps']:.1f} | Latency: {stats['avg_latency_ms']:.1f}ms")

total_time_elapsed = time.time() - total_start

# Final statistics
final_stats = monitor.get_stats()
print("\n" + "="*60)
print("📊 FINAL PERFORMANCE REPORT")
print("="*60)
print(f"Frames Processed:    {frames_processed}")
print(f"Total Time:          {total_time_elapsed:.2f} seconds")
print(f"Average FPS:         {final_stats['fps']:.2f} fps")
print(f"Average Latency:     {final_stats['avg_latency_ms']:.1f} ms")
print(f"Detection Time:      {final_stats['avg_detection_ms']:.1f} ms")
print(f"Generator Time:     {final_stats['avg_generator_ms']:.1f} ms")
print("="*60)

# Performance breakdown
print("\n📈 Performance Breakdown:")
print(f"   Detection:    {final_stats['avg_detection_ms']:.1f}ms ({final_stats['avg_detection_ms']/final_stats['avg_latency_ms']*100:.1f}%)")
print(f"   Generator:    {final_stats['avg_generator_ms']:.1f}ms ({final_stats['avg_generator_ms']/final_stats['avg_latency_ms']*100:.1f}%)")
print(f"   Other:        {final_stats['avg_latency_ms'] - final_stats['avg_detection_ms'] - final_stats['avg_generator_ms']:.1f}ms")

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['Detection', 'Generator', 'Other']
times = [
    final_stats['avg_detection_ms'],
    final_stats['avg_generator_ms'],
    final_stats['avg_latency_ms'] - final_stats['avg_detection_ms'] - final_stats['avg_generator_ms']
]
colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
bars = ax.bar(categories, times, color=colors)
ax.set_ylabel('Time (ms)', fontsize=12)
ax.set_title('Processing Time Breakdown', fontsize=14, weight='bold')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}ms', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# Performance evaluation
print("\n" + "="*60)
if final_stats['fps'] >= 15:
    print("✅ EXCELLENT - Real-time performance achieved (15+ FPS)")
    print("   Ready for smooth real-time face swapping!")
elif final_stats['fps'] >= 10:
    print("✅ GOOD - Acceptable performance (10-15 FPS)")
    print("   Usable for real-time with minor lag")
elif final_stats['fps'] >= 5:
    print("⚠️  FAIR - Moderate performance (5-10 FPS)")
    print("   Noticeable lag, may need optimization")
else:
    print("❌ SLOW - Poor performance (<5 FPS)")
    print("   Not suitable for real-time. Check GPU or optimize settings.")
print("="*60)
```

---

## 🎯 Quick Performance Benchmark (Static Image)

Test processing speed with a static image (no webcam needed):

```python
# Quick benchmark with static image
print("🚀 Quick Performance Benchmark (Static Image)")

test_image_path = '/content/sber-swap/examples/images/beckham.jpg'
test_image = cv2.imread(test_image_path)

if test_image is None:
    print(f"❌ Could not load test image: {test_image_path}")
else:
    print(f"✅ Testing with image: {test_image.shape}")
    
    # Reset monitor
    monitor.reset()
    
    # Run multiple iterations
    num_iterations = 10
    print(f"\nRunning {num_iterations} iterations...")
    
    for i in range(num_iterations):
        monitor.start_frame()
        
        # Process frame
        bbox = tracker.update(test_image)
        if bbox is not None:
            result, det_time, gen_time = process_single_frame(
                frame=test_image, source_embed=source_embed,
                netArc=netArc, G=G, app=app, handler=handler,
                bbox=bbox, crop_size=args.crop_size, half=True
            )
        else:
            det_time = 0
            gen_time = 0
        
        monitor.record_detection_time(det_time)
        monitor.record_generator_time(gen_time)
        total_time = (time.time() - monitor.frame_start_time) * 1000 if monitor.frame_start_time else 0
        monitor.record_processing_time(total_time)
        monitor.end_frame()
        
        if (i + 1) % 5 == 0:
            stats = monitor.get_stats()
            print(f"  Iteration {i+1}: FPS = {stats['fps']:.1f}")
    
    # Final results
    final_stats = monitor.get_stats()
    print("\n" + "="*60)
    print("📊 BENCHMARK RESULTS")
    print("="*60)
    print(f"Average FPS:      {final_stats['fps']:.2f}")
    print(f"Average Latency:  {final_stats['avg_latency_ms']:.1f} ms")
    print(f"Detection:        {final_stats['avg_detection_ms']:.1f} ms")
    print(f"Generator:        {final_stats['avg_generator_ms']:.1f} ms")
    print("="*60)
```

---

## ⚙️ Optimization Testing

Test different configurations to find optimal settings:

```python
# Test different configurations
configs = [
    {"fast_mode": False, "num_blocks": 2, "detect_interval": 5, "name": "Default"},
    {"fast_mode": True, "num_blocks": 2, "detect_interval": 5, "name": "Fast Mode"},
    {"fast_mode": True, "num_blocks": 1, "detect_interval": 5, "name": "Fast + 1-block"},
    {"fast_mode": True, "num_blocks": 2, "detect_interval": 10, "name": "Fast + Track 10"},
]

results = []

for config in configs:
    print(f"\n🧪 Testing: {config['name']}")
    
    # Reinitialize with new config
    args.fast_mode = config['fast_mode']
    args.num_blocks = config['num_blocks']
    args.detect_interval = config['detect_interval']
    
    # Reload models if needed
    if config['num_blocks'] != 2:
        G_path = f'weights/G_unet_{config["num_blocks"]}block.pth'
        if os.path.exists(G_path):
            args.G_path = G_path
            app, G, netArc, handler = init_models(args)
    
    # Update tracker
    tracker = FaceTracker(detector=app, detect_interval=config['detect_interval'],
                         tracker_type='CSRT', confidence_threshold=0.6)
    
    # Test with static image
    test_image = cv2.imread('/content/sber-swap/examples/images/beckham.jpg')
    monitor.reset()
    
    for i in range(5):  # 5 iterations per config
        monitor.start_frame()
        bbox = tracker.update(test_image)
        if bbox is not None:
            result, det_time, gen_time = process_single_frame(
                frame=test_image, source_embed=source_embed,
                netArc=netArc, G=G, app=app, handler=handler,
                bbox=bbox, crop_size=args.crop_size, half=True
            )
        monitor.record_detection_time(det_time if 'det_time' in locals() else 0)
        monitor.record_generator_time(gen_time if 'gen_time' in locals() else 0)
        total_time = (time.time() - monitor.frame_start_time) * 1000 if monitor.frame_start_time else 0
        monitor.record_processing_time(total_time)
        monitor.end_frame()
    
    stats = monitor.get_stats()
    results.append({
        'name': config['name'],
        'fps': stats['fps'],
        'latency': stats['avg_latency_ms']
    })
    print(f"   Result: {stats['fps']:.1f} FPS, {stats['avg_latency_ms']:.1f}ms latency")

# Display comparison
print("\n" + "="*60)
print("📊 CONFIGURATION COMPARISON")
print("="*60)
print(f"{'Configuration':<20} {'FPS':<10} {'Latency (ms)':<15}")
print("-" * 60)
for r in results:
    print(f"{r['name']:<20} {r['fps']:<10.1f} {r['latency']:<15.1f}")
print("="*60)

# Find best config
best = max(results, key=lambda x: x['fps'])
print(f"\n🏆 Best Configuration: {best['name']} ({best['fps']:.1f} FPS)")
```

---

## 📈 Expected Performance (Colab T4 GPU)

| Configuration | Expected FPS | Latency |
|--------------|--------------|---------|
| Default (2-block, 640x640) | 12-15 | 65-80ms |
| Fast Mode (320x320) | 15-18 | 55-65ms |
| Fast + 1-block | 18-22 | 45-55ms |
| Fast + Track 10 | 20-25 | 40-50ms |

---

## 🐛 Troubleshooting

### Low FPS
1. **Check GPU**: `torch.cuda.is_available()` should be `True`
2. **Enable Fast Mode**: `args.fast_mode = True`
3. **Use 1-block model**: `args.num_blocks = 1`
4. **Increase tracking interval**: `args.detect_interval = 10`

### High Latency
1. **GPU not active**: Restart runtime with GPU enabled
2. **First frame slow**: This is normal (model loading)
3. **Too many frames**: Average improves after 5+ frames

### Webcam Issues
1. **Permission denied**: Allow browser camera access
2. **No face detected**: Ensure good lighting and frontal face
3. **Use static image test**: Alternative if webcam doesn't work

---

## ✅ Performance Checklist

- [ ] GPU enabled (Runtime → Change runtime type → GPU)
- [ ] Models loaded successfully
- [ ] Source face embedded
- [ ] Test with 10+ frames for accuracy
- [ ] Check FPS is 10+ (15+ is ideal)
- [ ] Latency is <70ms (ideally <60ms)

---

## 📖 More Information

- See `REALTIME_README.md` for full documentation
- See `SberSwap_Realtime_Colab.ipynb` for interactive notebook

