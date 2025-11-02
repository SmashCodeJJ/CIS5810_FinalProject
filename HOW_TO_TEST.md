# 🧪 How to Test Real-Time Face Swapping

## Quick Answer: Test in Colab First, Then Run Locally

---

## ✅ Step 1: Test in Colab (5 minutes)

### Why Colab first?
- ✅ Quick setup (no local installation)
- ✅ Free GPU access
- ✅ Verify everything works before local setup
- ✅ Catch any issues early

### How to test:

1. **Open Colab Notebook**
   ```
   File → Upload → SberSwap_Realtime_Colab.ipynb
   Runtime → Change runtime type → GPU (T4)
   ```

2. **Run Setup Cells** (in order):
   - Install dependencies
   - Upload source face image
   - Load models (wait ~1-2 minutes)

3. **Test Single Frame** (easiest test):
   - Run "Single Frame Capture" cell
   - **Expected**: 
     - Camera permission → Click "Allow"
     - Green box around face ✅
     - Face swapped ✅
     - Stats displayed ✅

4. **If single frame works** → ✅ **Ready for local!**

---

## ✅ Step 2: Run Locally (Better Performance)

Once Colab test passes:

```bash
# 1. Clone repository
git clone -b Youxin https://github.com/SmashCodeJJ/CIS5810_FinalProject.git
cd CIS5810_FinalProject

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download models
bash download_models.sh

# 4. Run local script
python inference_realtime_local.py \
  --source_path examples/images/mark.jpg \
  --camera_index 0
```

### Local advantages:
- ✅ **True continuous streaming** (no permission delays)
- ✅ **Higher FPS** (10-30 FPS vs 1-2 FPS in Colab)
- ✅ **Better performance**
- ✅ **Full control**

---

## 🎯 What to Look For

### ✅ Success Indicators:

1. **Models load** → "✅ Models loaded!"
2. **Face detected** → Green bounding box appears
3. **Face swapped** → Source face appears in place of target
4. **Stats display** → FPS, latency shown
5. **No errors** → Clean output

### ❌ If something fails:

- **"No face detected"** → Move closer, better lighting
- **"Camera error"** → Check permissions, camera access
- **"Model error"** → Verify models downloaded
- **"AttributeError"** → Make sure using latest code

---

## 📊 Expected Performance

| Environment | FPS | Latency | Notes |
|------------|-----|---------|-------|
| **Colab** | 1-2 | 500-1000ms | Permission delays |
| **Local** | 10-30 | 30-100ms | True real-time |

---

## 🚀 Quick Test Checklist

Before moving to local, verify:

- [ ] Colab notebook runs without errors
- [ ] Single frame capture works
- [ ] Face detection works (green box)
- [ ] Face swapping works (source face appears)
- [ ] Performance stats display
- [ ] Can process multiple frames

**If all checked** → ✅ **Ready for local deployment!**

---

## 💡 Tips

1. **Start with single frame** in Colab (easiest)
2. **Verify everything works** before local setup
3. **Use local for best performance** after testing
4. **Check GPU**: `nvidia-smi` (local) or Colab GPU indicator

---

## 📝 Summary

**Test in Colab** → Verify it works → **Run locally** → Enjoy true real-time!

**Files:**
- `SberSwap_Realtime_Colab.ipynb` - Colab notebook
- `inference_realtime_local.py` - Local script
- `TESTING_GUIDE.md` - Detailed guide
- `QUICK_TEST_COLAB.md` - 5-minute test

**That's it! 🎉**

