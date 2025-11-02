# ⚡ Quick Test: Real-Time Face Swapping in Colab

## 🎯 5-Minute Test

Follow these steps to quickly verify everything works:

---

## Step 1: Open Colab (2 minutes)

1. Go to [Google Colab](https://colab.research.google.com/)
2. File → Upload notebook
3. Upload: `SberSwap_Realtime_Colab.ipynb`
   - Or clone from GitHub:
   ```python
   !git clone -b Youxin https://github.com/SmashCodeJJ/CIS5810_FinalProject.git
   %cd CIS5810_FinalProject
   ```
4. Runtime → Change runtime type → **GPU (T4)**

---

## Step 2: Run Setup (1 minute)

Run cells in order:
1. **Install dependencies** - Wait for "✅ Installation complete"
2. **Upload source face** - Upload any face image (or use example)
3. **Load models** - Wait 1-2 minutes (shows "✅ Models loaded!")

---

## Step 3: Quick Test (30 seconds)

### ✅ Test 1: Single Frame (Easiest)

1. Run the **"Single Frame Capture"** cell
2. **Expected:**
   - Camera permission dialog appears → Click "Allow"
   - Frame captured
   - Face detected (green box)
   - Face swapped (source face appears)
   - Stats displayed

**If this works** ✅ → Real-time loop will work too!

---

### ✅ Test 2: Continuous Loop (Optional)

1. Run the **"Continuous Real-Time Processing Loop"** cell
2. **Expected:**
   - Captures frames continuously
   - Shows "Capturing frame X..."
   - Face swapping applied
   - Stats updating

**Note**: Each frame requires permission (~0.5-1 sec delay)

---

## ✅ Success Indicators

You'll know it's working if you see:

1. ✅ **No errors** in output
2. ✅ **Green bounding box** around face
3. ✅ **Source face appears** (swapped)
4. ✅ **Stats display**: FPS, latency, etc.
5. ✅ **Result looks different** from original

---

## 🚀 If Test Passes: Run Locally

Once Colab test works, run on your local computer:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download models
bash download_models.sh

# 3. Run local script
python inference_realtime_local.py \
  --source_path examples/images/mark.jpg \
  --camera_index 0
```

**Local advantages:**
- ✅ True continuous streaming (no permission delays)
- ✅ Higher FPS (10-30 FPS vs 1-2 FPS in Colab)
- ✅ Better performance

---

## ❌ Troubleshooting

### "No face detected"
- Move face closer to camera
- Face camera directly
- Improve lighting

### "Camera permission denied"
- Click "Allow" when prompted
- Check browser camera settings

### "AttributeError" or other errors
- Make sure all cells ran successfully
- Check you're using latest code from GitHub
- Try restarting runtime and re-running cells

---

## 📊 Expected Results

**Colab:**
- FPS: 1-2 (limited by permissions)
- Latency: 500-1000ms per frame
- Works but slow due to browser security

**Local:**
- FPS: 10-30 (much better!)
- Latency: 30-100ms per frame
- True real-time experience

---

**That's it! If single frame works, you're ready for local deployment! 🎉**

