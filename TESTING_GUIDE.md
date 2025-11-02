# 🧪 Testing Guide: Real-Time Face Swapping

## 🎯 Quick Test in Colab

Follow these steps to verify real-time face swapping works:

---

## Step 1: Setup Colab Environment

1. **Open Colab Notebook**
   - Go to: [Google Colab](https://colab.research.google.com/)
   - Upload: `SberSwap_Realtime_Colab.ipynb`
   - Or clone from GitHub:
     ```python
     !git clone -b Youxin https://github.com/SmashCodeJJ/CIS5810_FinalProject.git
     %cd CIS5810_FinalProject
     ```

2. **Enable GPU**
   - Runtime → Change runtime type → GPU (T4)

3. **Run Installation Cell**
   - Run the first few cells to install dependencies
   - Wait for "✅ Installation complete"

---

## Step 2: Test Real-Time Processing

### Option A: Quick Single Frame Test (Recommended First)

1. **Run Setup Cells**
   - Upload source face image
   - Load models (wait ~1-2 minutes)

2. **Test Single Frame Capture**
   - Run the "Single Frame Capture" cell
   - **Expected**: 
     - Camera permission dialog appears
     - Frame is captured
     - Face swap is applied
     - Result is displayed
   - ✅ **If this works**: Real-time loop should work too

3. **Check Output**
   - ✅ Face detected → Green bounding box
   - ✅ Face swapped → Your source face appears
   - ✅ Stats displayed → FPS, latency shown

### Option B: Continuous Loop Test

1. **Run Continuous Loop Cell**
   - Click "Start Streaming" section
   - **Expected Behavior**:
     - Camera permission each frame (~0.5-1 sec delay)
     - Frame captured and processed
     - Display updates continuously
     - FPS ~1-2 (limited by permissions)

2. **Monitor Output**
   - Watch for:
     - ✅ "Capturing frame X..." messages
     - ✅ Face detection working (green box)
     - ✅ Face swapping applied
     - ✅ Performance stats updating

3. **Stop Test**
   - Click "Stop" button or Ctrl+C
   - Check final stats

---

## Step 3: Verify It's Working

### ✅ Success Indicators:

1. **Face Detection Works**
   - Green bounding box appears around face
   - "Face detected" message in output

2. **Face Swapping Works**
   - Source face appears in place of target face
   - Result looks different from original

3. **Performance Metrics**
   - FPS: > 0 (usually 1-2 in Colab due to permissions)
   - Latency: < 1000ms per frame
   - Detection time: < 200ms
   - Generator time: < 300ms

4. **No Errors**
   - No `AttributeError`
   - No `TypeError`
   - No camera access errors

---

## Step 4: Run Locally (After Colab Test)

Once Colab test passes, run locally:

### Local Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Models**
   ```bash
   bash download_models.sh
   ```

3. **Run Local Script**
   ```bash
   python inference_realtime_local.py \
     --source_path examples/images/mark.jpg \
     --camera_index 0
   ```

### Local Features (Better than Colab):
- ✅ **True continuous streaming** (no permission delays)
- ✅ **Higher FPS** (10-30 FPS depending on hardware)
- ✅ **No browser limitations**
- ✅ **Better performance**

---

## 🔍 Troubleshooting

### Issue: "No face detected"

**Possible causes:**
- Face too far from camera
- Face at extreme angle
- Poor lighting

**Solutions:**
- Move face closer to camera
- Face camera directly
- Improve lighting
- Check camera is working

### Issue: "Camera permission denied"

**In Colab:**
- Click "Allow" when prompted
- If blocked, check browser settings
- Try refreshing and allowing permissions

**Locally:**
- Grant camera permissions in OS settings
- Check camera isn't used by another app

### Issue: "AttributeError: no attribute 'detect'"

**Solution:**
- Make sure you're using the latest code from GitHub
- Check `utils/realtime/face_tracker.py` has correct implementation
- Pull latest changes: `git pull origin Youxin`

### Issue: Low FPS

**In Colab:**
- Normal (due to permission delays)
- Expect 1-2 FPS
- Use video processing for better performance

**Locally:**
- Check GPU is being used: `nvidia-smi`
- Reduce `detect_interval` (more tracking, less detection)
- Use `num_blocks=1` for faster processing

---

## 📊 Expected Performance

### Colab (with permissions):
- **FPS**: 1-2 FPS
- **Latency**: 500-1000ms per frame
- **Detection**: 50-100ms
- **Generator**: 200-400ms
- **Bottleneck**: Browser camera permissions

### Local (direct camera access):
- **FPS**: 10-30 FPS (depends on GPU)
- **Latency**: 30-100ms per frame
- **Detection**: 20-50ms (with tracking)
- **Generator**: 50-200ms
- **Bottleneck**: GPU processing speed

---

## ✅ Test Checklist

Before moving to local, verify:

- [ ] Colab notebook runs without errors
- [ ] Models load successfully
- [ ] Single frame capture works
- [ ] Face detection works (green box)
- [ ] Face swapping works (source face appears)
- [ ] Performance stats display correctly
- [ ] Can process multiple frames
- [ ] Stop button works

**If all checked**: ✅ Ready for local deployment!

---

## 🚀 Next Steps

1. ✅ **Test in Colab first** (this guide)
2. ✅ **Verify it works** (checklist above)
3. ✅ **Download local script** (`inference_realtime_local.py`)
4. ✅ **Run on local computer** (better performance)
5. ✅ **Test with local webcam**

---

## 💡 Tips

1. **Start with single frame** in Colab to verify setup
2. **Use video processing** in Colab for better quality
3. **Run locally** for true real-time experience
4. **Check GPU usage**: `nvidia-smi` (local) or Colab GPU indicator
5. **Monitor memory**: Watch for OOM errors

---

## 📞 Need Help?

If Colab test fails:
1. Check error messages
2. Verify all cells ran successfully
3. Check camera permissions in browser
4. Try single frame first
5. Check GitHub issues or documentation

If local test fails:
1. Verify camera access
2. Check GPU drivers
3. Verify dependencies installed
4. Check model files downloaded
5. Review error logs

---

**Good luck with testing! 🎉**

