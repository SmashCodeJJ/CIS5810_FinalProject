# 🚀 Push to GitHub - Step-by-Step Guide

Your project is ready to push to GitHub! Follow these simple steps:

---

## Step 1: Create GitHub Repository

1. Go to: **https://github.com/new**
2. Fill in:
   - **Repository name**: `sber-swap-colab`
   - **Description**: `Sber-Swap face swapping updated for Google Colab (Python 3.12, PyTorch 2.x)`
   - **Visibility**: ✅ **Public** (so Colab can clone it)
   - **DON'T** check "Initialize with README"
3. Click **"Create repository"**

---

## Step 2: Clean Up and Push

Open **Terminal** on your Mac and run:

```bash
cd "/Users/apple/Documents/Upenn Academy/CIS5810/Final Project/sber-swap"

# Clean up temporary files
./cleanup_before_push.sh

# Push to GitHub
./push_to_github.sh
```

That's it! The scripts will:
- Remove temporary debug files
- Initialize git repository
- Commit all your updated files
- Push to: `https://github.com/SmashCodeJJ/sber-swap-colab`

---

## Step 3: Use in Google Colab

### Method 1: Direct Notebook Link
After pushing, open this URL in your browser:
```
https://colab.research.google.com/github/SmashCodeJJ/sber-swap-colab/blob/main/SberSwap_Colab_Ready.ipynb
```

### Method 2: Manual Clone
In a new Colab notebook:

```python
# Cell 1: Install (then restart runtime)
!git clone https://github.com/SmashCodeJJ/sber-swap-colab.git
%cd sber-swap-colab
!git submodule init && git submodule update
%pip install -q -r requirements.txt
!bash download_models.sh
print("✅ Restart runtime now!")

# Cell 2: Run (after restart)
%cd /content/sber-swap-colab
!python inference.py \
  --image_to_image True \
  --target_image examples/images/beckham.jpg \
  --source_paths examples/images/mark.jpg \
  --out_image_name examples/results/result.png

from IPython.display import Image, display
display(Image('examples/results/result.png'))
```

---

## 📦 What Will Be Pushed

**Updated Files** (10 files):
- ✅ `requirements.txt` - Modern dependencies
- ✅ `coordinate_reg/image_infer.py` - ONNX Runtime
- ✅ `insightface_func/face_detect_crop_single.py` - API fix
- ✅ `insightface_func/face_detect_crop_multi.py` - API fix
- ✅ `utils/inference/image_processing.py` - Matrix fixes
- ✅ `utils/inference/video_processing.py` - Matrix fixes
- ✅ `utils/inference/core.py` - Validation
- ✅ `utils/inference/util.py` - Regex fix
- ✅ `models/networks/normalization.py` - Regex fix
- ✅ `inference.py` - Error handling

**New Files**:
- ✅ `SberSwap_Colab_Ready.ipynb` - Ready-to-use notebook
- ✅ `README_COLAB.md` - Documentation
- ✅ `.gitignore` - Ignore rules

**Will NOT be pushed** (cleaned up):
- ❌ Debug scripts (debug_*.py, test_*.py)
- ❌ Temporary docs (COLAB_*.md, etc.)
- ❌ Duplicate notebooks

---

## 🎯 After Pushing

Your repo will be live at:
**https://github.com/SmashCodeJJ/sber-swap-colab**

You can:
1. ⭐ Star your own repo (why not!)
2. 📝 Edit README_COLAB.md on GitHub if needed
3. 🔗 Share the Colab link with others
4. 🔄 Update anytime with `git push`

---

## 🐛 Troubleshooting

### "Permission denied (publickey)"
You need to authenticate. Run:
```bash
# Use HTTPS (easier, will prompt for password/token)
git remote set-url origin https://github.com/SmashCodeJJ/sber-swap-colab.git
git push -u origin main
```

### "Updates were rejected"
The remote has changes you don't have. Run:
```bash
git pull --rebase origin main
git push -u origin main
```

### Need to use a Personal Access Token?
1. Go to: https://github.com/settings/tokens
2. Generate new token (classic)
3. Use token as password when pushing

---

## 📞 Need Help?

If you encounter any issues:
1. Check that you created the repo with the exact name: `sber-swap-colab`
2. Make sure the repo is **Public** (not Private)
3. Try HTTPS URL instead of SSH

---

**Ready? Run the commands in Step 2 above!** 🚀

