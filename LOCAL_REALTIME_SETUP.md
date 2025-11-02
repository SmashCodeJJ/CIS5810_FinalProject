# 🖥️ Local Real-Time Setup Guide

## Issues Found

From testing, we found:
1. **NumPy Version Issue**: NumPy 2.2.6 incompatible with PyTorch (needs NumPy <2.0)
2. **Camera Permission**: macOS requires camera permission in System Settings
3. **No GPU**: Running on CPU (slow, but will work)

---

## 🔧 Fix Setup Issues

### Step 1: Fix NumPy Version

```bash
# Install compatible NumPy version
pip install "numpy<2.0"

# Or create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### Step 2: Grant Camera Permission (macOS)

**On macOS:**
1. Go to **System Settings** → **Privacy & Security** → **Camera**
2. Check if your terminal/IDE has camera access
3. If not, add it manually

**Or run this to check:**
```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera works!' if cap.isOpened() else 'Camera blocked'); cap.release()"
```

### Step 3: Check GPU (Optional but Recommended)

```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# If False, you can still run on CPU (but slower)
```

---

## 🚀 Quick Test (No Camera Needed)

Test with static image first:

```bash
cd "/Users/apple/Documents/Upenn Academy/CIS5810/Final Project/sber-swap"

# Test with static image
python test_realtime.py \
    --source_path examples/images/mark.jpg \
    --test_image examples/images/beckham.jpg \
    --output_image examples/results/test_local_output.jpg
```

This will:
- ✅ Test all models load correctly
- ✅ Verify face swap works
- ✅ Show performance metrics
- ✅ No camera needed

---

## 📹 Full Real-Time Test (With Camera)

Once camera permission is granted:

```bash
cd "/Users/apple/Documents/Upenn Academy/CIS5810/Final Project/sber-swap"

# Run real-time face swapping
python inference_realtime.py \
    --source_path examples/images/mark.jpg \
    --camera_id 0 \
    --fast_mode
```

**Controls:**
- `q` - Quit
- `r` - Reset tracker
- `s` - Save current frame

---

## 🔍 Troubleshooting

### Camera Permission Denied (macOS)

**Error**: `OpenCV: not authorized to capture video`

**Fix:**
1. Open **System Settings**
2. Go to **Privacy & Security** → **Camera**
3. Enable access for:
   - Terminal (if running from terminal)
   - Python (if running from IDE)
   - Your IDE (VS Code, PyCharm, etc.)

**Test:**
```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('✅ Camera OK' if cap.isOpened() else '❌ Camera blocked'); cap.release()"
```

### NumPy Version Error

**Error**: `NumPy 2.x incompatible`

**Fix:**
```bash
pip install "numpy<2.0"
# Then restart Python/IDE
```

### Camera ID Not Found

**Error**: `Could not open camera (ID 0)`

**Try different IDs:**
```bash
python inference_realtime.py --camera_id 1  # Try camera 1
python inference_realtime.py --camera_id 2  # Try camera 2
```

### Slow Performance (No GPU)

**On CPU, expect:**
- 2-5 FPS (vs 15-18 FPS on GPU)
- Still works, but noticeable lag

**To improve:**
- Use `--fast_mode` flag
- Use `--num_blocks 1` (smaller model)
- Lower resolution: `--width 320 --height 240`

---

## ✅ Expected Performance

| Hardware | Expected FPS | Quality |
|----------|--------------|---------|
| GPU (NVIDIA) | 15-18 FPS | High |
| GPU (M1/M2 Mac) | 8-12 FPS | High |
| CPU Only | 2-5 FPS | Medium |

---

## 🎯 Quick Start Checklist

- [ ] Fix NumPy: `pip install "numpy<2.0"`
- [ ] Grant camera permission (macOS Settings)
- [ ] Test static image: `python test_realtime.py`
- [ ] Test camera: `python -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'Failed'); cap.release()"`
- [ ] Run real-time: `python inference_realtime.py --source_path examples/images/mark.jpg`

---

## 📝 Example Commands

### Test Static Image (No Camera)
```bash
python test_realtime.py \
    --source_path examples/images/mark.jpg \
    --test_image examples/images/beckham.jpg
```

### Real-Time with Default Settings
```bash
python inference_realtime.py \
    --source_path examples/images/mark.jpg
```

### Real-Time with Fast Mode
```bash
python inference_realtime.py \
    --source_path examples/images/mark.jpg \
    --fast_mode \
    --num_blocks 1
```

### Real-Time with Custom Camera
```bash
python inference_realtime.py \
    --source_path examples/images/mark.jpg \
    --camera_id 1 \
    --width 640 \
    --height 480
```

---

## 🐛 Common Issues

### Issue: "Camera failed to initialize"
**Solution**: Grant camera permission in System Settings

### Issue: NumPy version error
**Solution**: `pip install "numpy<2.0"` and restart

### Issue: "No module named 'utils'"
**Solution**: Make sure you're in the `sber-swap` directory

### Issue: Models not found
**Solution**: Run `bash download_models.sh` or download manually

---

## 💡 Pro Tips

1. **Test with static image first** - Verify everything works before trying camera
2. **Use fast_mode** - Better performance, minimal quality loss
3. **Check camera ID** - Try 0, 1, 2 if default doesn't work
4. **CPU is slow** - Expect 2-5 FPS, not 15-18 FPS
5. **Grant permission once** - After granting, camera works for all sessions

---

## 🎉 Success Indicators

When it's working, you should see:
- ✅ "Models loaded successfully!"
- ✅ Camera window opens
- ✅ Face detection shows green bounding box
- ✅ Real-time face swapping visible
- ✅ FPS counter shows 10+ (GPU) or 2-5 (CPU)
- ✅ Controls work (q to quit, etc.)

---

## 📖 More Help

See:
- `REALTIME_README.md` - Full real-time documentation
- `QUICK_START_REALTIME.md` - Quick start guide
- `COLAB_CAMERA_PERMISSIONS.md` - Colab limitations explained

