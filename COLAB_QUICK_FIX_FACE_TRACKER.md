# 🔧 Quick Fix for Face Detection Error in Colab

## Error
```
Face detection error: 'Face_detect_crop' object has no attribute 'detect'
```

## ✅ Quick Fix (Copy-Paste into Colab)

If you're getting this error and haven't pulled the latest code, add this cell **BEFORE** loading models:

```python
# Quick fix for face tracker
import sys
sys.path.insert(0, '/content/sber-swap')

# Patch the face tracker file directly
face_tracker_path = '/content/sber-swap/utils/realtime/face_tracker.py'

# Read current file
with open(face_tracker_path, 'r') as f:
    content = f.read()

# Replace the problematic line
old_line = "            bboxes, _ = self.detector.detect(frame, max_num=1)"
new_line = "            bboxes, _ = self.detector.det_model.detect(frame, max_num=1, metric='default')"

if old_line in content:
    content = content.replace(old_line, new_line)
    with open(face_tracker_path, 'w') as f:
        f.write(content)
    print("✅ Face tracker fixed!")
else:
    # Check if already fixed or has different format
    if "det_model.detect" in content:
        print("✅ Face tracker already fixed!")
    else:
        print("⚠️  Could not auto-fix. Please pull latest code.")

# Reload the module
if 'utils.realtime.face_tracker' in sys.modules:
    del sys.modules['utils.realtime.face_tracker']
```

---

## 🎯 Alternative: Direct Fix in Code

Or manually update the face tracker detection method:

```python
# After importing, patch the method directly
from utils.realtime.face_tracker import FaceTracker

# Monkey patch the detection method
original_detect = FaceTracker._detect_face

def fixed_detect_face(self, frame):
    try:
        # Use det_model directly
        bboxes, _ = self.detector.det_model.detect(frame, max_num=1, metric='default')
        
        if bboxes.shape[0] == 0:
            return None
        
        keep = bboxes[:, 4] >= self.confidence_threshold
        bboxes = bboxes[keep]
        
        if bboxes.shape[0] == 0:
            return None
        
        best_bbox = bboxes[0]
        x, y, w, h = int(best_bbox[0]), int(best_bbox[1]), \
                    int(best_bbox[2] - best_bbox[0]), int(best_bbox[3] - best_bbox[1])
        
        if w > 0 and h > 0:
            return (x, y, w, h)
        return None
    except Exception as e:
        print(f"Face detection error: {e}")
        return None

# Apply the patch
FaceTracker._detect_face = fixed_detect_face
print("✅ Face tracker patched!")
```

---

## 📋 Complete Working Cell (All Fixes)

Use this complete cell that includes all fixes:

```python
import sys
import torch
import time
import numpy as np
import cv2
from IPython.display import display, Image as IPImage
import base64
import io
from PIL import Image
from google.colab.output import eval_js

# Add to path
sys.path.insert(0, '/content/sber-swap')

# Fix face tracker if needed
try:
    from utils.realtime.face_tracker import FaceTracker
    
    # Check if fix needed
    import inspect
    source = inspect.getsource(FaceTracker._detect_face)
    if 'detector.detect(' in source and 'det_model.detect(' not in source:
        # Need to patch
        original_method = FaceTracker._detect_face
        
        def fixed_detect(self, frame):
            try:
                bboxes, _ = self.detector.det_model.detect(frame, max_num=1, metric='default')
                if bboxes.shape[0] == 0:
                    return None
                keep = bboxes[:, 4] >= self.confidence_threshold
                bboxes = bboxes[keep]
                if bboxes.shape[0] == 0:
                    return None
                best_bbox = bboxes[0]
                x, y, w, h = int(best_bbox[0]), int(best_bbox[1]), \
                            int(best_bbox[2] - best_bbox[0]), int(best_bbox[3] - best_bbox[1])
                if w > 0 and h > 0:
                    return (x, y, w, h)
                return None
            except Exception as e:
                print(f"Face detection error: {e}")
                return None
        
        FaceTracker._detect_face = fixed_detect
        print("✅ Face tracker patched!")
    else:
        print("✅ Face tracker already correct")
except Exception as e:
    print(f"⚠️  Could not check/patch tracker: {e}")

# Webcam function
def take_photo():
    js_code = '''
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
    '''
    data = eval_js(js_code + 'takePhoto()')
    image_bytes = base64.b64decode(data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes))
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

# Import other modules
from inference_realtime import init_models, load_source_face
from utils.inference.realtime_processing import process_single_frame
from utils.realtime.performance_monitor import PerformanceMonitor

print("✅ All imports and fixes ready!")
```

---

## 🔄 Best Solution: Pull Latest Code

The easiest solution is to pull the latest code that already has the fix:

```python
# Pull latest code with fix
%cd /content/sber-swap
!git pull origin Youxin

print("✅ Latest code pulled! Restart runtime and continue.")
```

---

## ✅ Verification

After applying the fix, verify it works:

```python
# Test the tracker
from utils.realtime.face_tracker import FaceTracker
from insightface_func.face_detect_crop_multi import Face_detect_crop

# Create detector
app = Face_detect_crop(name='antelope', root='./insightface_func/models')
app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

# Create tracker
tracker = FaceTracker(detector=app, detect_interval=5)

# Test with a simple image
test_img = cv2.imread('/content/sber-swap/examples/images/beckham.jpg')
bbox = tracker.update(test_img)

if bbox is not None:
    print(f"✅ Face detection works! Bbox: {bbox}")
else:
    print("⚠️  No face detected (might be OK if image has no face)")
```

