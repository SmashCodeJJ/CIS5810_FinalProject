# 🚀 Quick Start: Run Real-Time Locally

## ✅ Your System Status

From diagnostics:
- ✅ **Camera**: Accessible
- ✅ **Models**: Found
- ⚠️ **NumPy**: 2.2.6 (should be <2.0 - fix below)
- ⚠️ **CUDA**: Not available (will run on CPU, slower)

---

## 🔧 Step 1: Fix NumPy Version

```bash
cd "/Users/apple/Documents/Upenn Academy/CIS5810/Final Project/sber-swap"
pip install "numpy<2.0"
```

**⚠️ Important**: Restart your terminal/IDE after installing!

---

## 🧪 Step 2: Test with Static Image First (Recommended)

Before testing with camera, verify everything works:

```bash
python test_realtime.py \
    --source_path examples/images/mark.jpg \
    --test_image examples/images/beckham.jpg
```

This tests:
- ✅ Model loading
- ✅ Face detection
- ✅ Face swapping
- ✅ Performance metrics

**No camera needed!**

---

## 📹 Step 3: Run Real-Time with Camera

Once static test works:

```bash
python inference_realtime.py \
    --source_path examples/images/mark.jpg \
    --camera_id 0 \
    --fast_mode
```

**Controls:**
- **`q`** - Quit
- **`r`** - Reset face tracker
- **`s`** - Save current frame

---

## 📊 Expected Performance

### On Your System (CPU):
- **FPS**: 2-5 FPS (slow but works)
- **Latency**: 200-500ms per frame
- **Quality**: High (but laggy)

### With GPU (if available):
- **FPS**: 15-18 FPS (smooth)
- **Latency**: 50-70ms per frame
- **Quality**: High

---

## 🔍 Quick Diagnostics

Run this to check everything:

```bash
bash RUN_REALTIME_LOCAL.sh
```

Or manually:
```bash
# Check camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('✅ Camera OK' if cap.isOpened() else '❌ Camera blocked'); cap.release()"

# Check NumPy
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 🐛 Common Issues & Fixes

### Issue 1: NumPy Version Error
**Error**: `NumPy 2.x incompatible`

**Fix:**
```bash
pip install "numpy<2.0"
# Restart terminal/IDE
```

### Issue 2: Camera Permission
**Error**: `OpenCV: not authorized to capture video`

**Fix (macOS):**
1. System Settings → Privacy & Security → Camera
2. Enable access for Terminal/Python/IDE
3. Restart terminal/IDE

### Issue 3: Slow Performance
**CPU is slow** - This is expected:
- Use `--fast_mode` flag
- Use `--num_blocks 1` (smaller model)
- Lower resolution: `--width 320 --height 240`

### Issue 4: Import Errors
**Error**: `No module named 'utils'`

**Fix:**
```bash
# Make sure you're in the right directory
cd "/Users/apple/Documents/Upenn Academy/CIS5810/Final Project/sber-swap"
pwd  # Should show .../sber-swap
```

---

## ✅ Quick Test Commands

### Test 1: Static Image (No Camera)
```bash
python test_realtime.py \
    --source_path examples/images/mark.jpg \
    --test_image examples/images/beckham.jpg
```

### Test 2: Real-Time (With Camera)
```bash
python inference_realtime.py \
    --source_path examples/images/mark.jpg \
    --fast_mode
```

### Test 3: Real-Time (Optimized for CPU)
```bash
python inference_realtime.py \
    --source_path examples/images/mark.jpg \
    --fast_mode \
    --num_blocks 1 \
    --width 320 \
    --height 240
```

---

## 🎯 What to Expect

### When It Works:
1. ✅ Models load (takes 10-30 seconds)
2. ✅ Camera window opens
3. ✅ Face detection shows green box
4. ✅ Face swapping appears in real-time
5. ✅ FPS counter shows in top-left
6. ✅ Controls work (q, r, s keys)

### Performance Indicators:
- **Good**: 10+ FPS (GPU) or 3+ FPS (CPU)
- **Fair**: 5-10 FPS (GPU) or 2-3 FPS (CPU)
- **Poor**: <5 FPS (GPU) or <2 FPS (CPU)

---

## 💡 Pro Tips

1. **Test static image first** - Verify models work
2. **Use fast_mode** - Better performance
3. **CPU is slow** - Expect 2-5 FPS, not 15-18
4. **Fix NumPy first** - Prevents many errors
5. **Grant camera permission** - One-time setup on macOS

---

## 📝 Full Command Reference

```bash
# Basic
python inference_realtime.py --source_path examples/images/mark.jpg

# Fast mode
python inference_realtime.py --source_path examples/images/mark.jpg --fast_mode

# Optimized for CPU
python inference_realtime.py \
    --source_path examples/images/mark.jpg \
    --fast_mode \
    --num_blocks 1 \
    --width 320 \
    --height 240 \
    --detect_interval 10

# Custom camera
python inference_realtime.py \
    --source_path examples/images/mark.jpg \
    --camera_id 1
```

---

## 🎉 Success!

When working correctly:
- Camera window shows your face
- Green box around detected face
- Face swapping happens in real-time
- FPS counter visible
- Smooth operation (GPU) or slightly laggy (CPU)

All fixes and guides are ready! 🚀

