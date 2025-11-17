# 🎥 Real-Time Face Swapping Guide

## Overview

This implementation provides **real-time face swapping from webcam** with optimized performance targeting **15-20 FPS**.

## Features

✅ **Face Tracking** - Avoids expensive detection every frame  
✅ **Source Embedding Caching** - Compute once, reuse for all frames  
✅ **Performance Monitoring** - Real-time FPS and latency tracking  
✅ **Optimized Pipeline** - Single-frame processing with minimal overhead  

## Quick Start

### Basic Usage

```bash
# Default settings (camera 0, source face from examples/images/mark.jpg)
python inference_realtime.py

# Specify custom source face
python inference_realtime.py --source_path path/to/your/face.jpg

# Fast mode (smaller detection, ~20% faster)
python inference_realtime.py --fast_mode

# Faster generator (1 block instead of 2, trade quality for speed)
python inference_realtime.py --num_blocks 1
```

### Advanced Options

```bash
python inference_realtime.py \
    --source_path examples/images/mark.jpg \
    --camera_id 0 \
    --width 640 \
    --height 480 \
    --detect_interval 5 \
    --tracker_type CSRT \
    --num_blocks 2 \
    --fast_mode
```

## Controls

- **'q'** - Quit application
- **'r'** - Reset face tracker
- **'s'** - Save current frame to `examples/results/`

## Performance Tips

### Target: 15-20 FPS

1. **Use Fast Mode**: `--fast_mode` reduces detection time by ~40%
2. **Adjust Detection Interval**: `--detect_interval 10` detects every 10 frames (default: 5)
3. **Use 1-Block Generator**: `--num_blocks 1` faster but lower quality
4. **Lower Resolution**: `--width 320 --height 240` for maximum speed

### Expected Performance

| Configuration | FPS | Quality | Use Case |
|--------------|-----|---------|----------|
| Default (2 blocks, 640x640) | 12-15 | High | Demos, presentations |
| Fast mode | 15-18 | High | Smooth real-time |
| 1 block + fast mode | 18-22 | Medium | Maximum speed |
| Lower resolution | 20-25 | Medium | Extreme speed needs |

## Troubleshooting

### Camera Not Opening

```bash
# Try different camera IDs
python inference_realtime.py --camera_id 1
python inference_realtime.py --camera_id 2
```

### Low FPS (< 10)

1. Check GPU is available: `nvidia-smi`
2. Enable fast mode: `--fast_mode`
3. Use smaller generator: `--num_blocks 1`
4. Reduce resolution: `--width 320 --height 240`

### Face Not Detected

1. Ensure good lighting
2. Face should be frontal (0-30° angle)
3. Reset tracker: Press 'r'
4. Check detection threshold in code (default: 0.6)

### Colab Usage

For Google Colab, webcam access requires special setup:

```python
# Use JavaScript-based webcam capture
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import cv2

def take_photo():
    js = Javascript('''
        async function takePhoto() {
            const video = document.createElement('video');
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            video.srcObject = stream;
            video.play();
            await new Promise(resolve => video.onloadedmetadata = () => {
                video.setAttribute('width', video.videoWidth);
                video.setAttribute('height', video.videoHeight);
                resolve();
            });
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            video.srcObject.getTracks().forEach(track => track.stop());
            return canvas.toDataURL('image/jpeg', 0.8);
        }
    ''')
    data = eval_js(js)
    # Convert to OpenCV format and process...
```

## Architecture

```
Camera → Face Tracker → Face Detection → Face Processing → Generator → Blending → Display
          (fast)          (slow, every N)   (every frame)    (GPU)       (CPU)
```

**Key Optimizations:**
- **Tracking**: ~4ms (vs detection ~20ms)
- **Cached Source Embedding**: 0ms (vs 50ms every frame)
- **Single Frame Processing**: No batch overhead
- **FP16 Precision**: Faster GPU computation

## Performance Metrics

The application displays real-time metrics:
- **FPS**: Frames per second (target: 15+)
- **Latency**: End-to-end processing time (target: <70ms)
- **Detection**: Face detection/tracking time
- **Generator**: Model inference time

## File Structure

```
sber-swap/
├── inference_realtime.py           # Main real-time script
├── utils/
│   ├── inference/
│   │   └── realtime_processing.py  # Single-frame processing
│   └── realtime/
│       ├── face_tracker.py         # OpenCV face tracking
│       ├── performance_monitor.py  # FPS/latency tracking
│       └── camera_capture.py       # Webcam handling
└── REALTIME_README.md              # This file
```

## Next Steps

For multi-threading (20-25 FPS), see `REALTIME_REQUIREMENTS.md` Phase 3 implementation guide.

