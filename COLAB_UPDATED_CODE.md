# 📝 Updated Colab Code for Your Setup

## ✅ Fixed Issues

1. **Use `%cd` instead of `!cd`** - `!cd` doesn't persist in Colab
2. **Specify branch**: `-b Youxin` to get real-time implementation
3. **Correct repo name**: `CIS5810_FinalProject` (not `sber-swap-colab`)
4. **Fixed path structure**: Clone to `sber-swap` directory
5. **Removed unnecessary submodule commands** (not needed for our repo)

---

## 🚀 Updated Code (Copy & Paste)

### Cell 1: Installation (Run once, then RESTART RUNTIME)

```python
# Clone repository with real-time implementation (Youxin branch)
!git clone -b Youxin https://github.com/SmashCodeJJ/CIS5810_FinalProject.git sber-swap
%cd sber-swap

# Install dependencies
%pip install -q -r requirements.txt

# Download models (if needed)
import os
if not os.path.exists('weights/G_unet_2blocks.pth'):
    print("Downloading models...")
    !bash download_models.sh 2>/dev/null || echo "Note: Models may need manual download"

print("\n" + "="*50)
print("✅ Installation complete!")
print("="*50)
print("⚠️  IMPORTANT: Go to Runtime → Restart runtime")
print("    Then skip this cell and run Cell 2 below.")
print("="*50)
```

### Cell 2: Verify (Run after restart)

```python
# Change to project directory (use %cd, not !cd)
%cd /content/sber-swap

# Verify environment
import torch
import numpy as np
import cv2

print("="*50)
print("✅ Environment Check")
print("="*50)
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("="*50)
```

### Cell 3: Run Face Swap (Image to Image)

```python
# Make sure we're in the right directory
%cd /content/sber-swap

# Run inference
!python inference.py \
  --image_to_image True \
  --target_image examples/images/beckham.jpg \
  --source_paths examples/images/mark.jpg \
  --out_image_name examples/results/ghost_result.png

print("✅ Face swap complete!")
```

### Cell 4: Display Results

```python
from IPython.display import Image, display
import matplotlib.pyplot as plt
import cv2
import os

# Check if result exists
result_path = '/content/sber-swap/examples/results/ghost_result.png'
if not os.path.exists(result_path):
    print("❌ Result file not found. Check for errors above.")
else:
    # Display comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Source
    source_path = '/content/sber-swap/examples/images/mark.jpg'
    if os.path.exists(source_path):
        source = cv2.imread(source_path)
        axes[0].imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Source (Mark)', fontsize=14)
        axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, 'Source not found', ha='center')
        axes[0].axis('off')

    # Target
    target_path = '/content/sber-swap/examples/images/beckham.jpg'
    if os.path.exists(target_path):
        target = cv2.imread(target_path)
        axes[1].imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Target (Beckham)', fontsize=14)
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'Target not found', ha='center')
        axes[1].axis('off')

    # Result
    result = cv2.imread(result_path)
    axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[2].set_title('RESULT - Face Swapped', fontsize=14, color='green', weight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    # Check if files are the same
    import filecmp
    if os.path.exists(target_path):
        same = filecmp.cmp(target_path, result_path)
        if same:
            print("❌ RESULT IS IDENTICAL TO TARGET - Face swap didn't work")
            print("   This usually means no face was detected in source or target image.")
        else:
            print("✅ RESULT IS DIFFERENT - Face swap appears to have worked!")
    else:
        print("⚠️  Could not compare files (target not found)")
```

---

## 🔑 Key Changes from Your Original Code

### ❌ Old (Wrong):
```python
!git clone https://github.com/SmashCodeJJ/CIS5810_FinalProject.git
!cd sber-swap              # ❌ !cd doesn't persist
!git submodule init        # ❌ Not needed for this repo
!git submodule update      # ❌ Not needed for this repo
%cd CIS5810_FinalProject   # ❌ Wrong directory name
```

### ✅ New (Correct):
```python
!git clone -b Youxin https://github.com/SmashCodeJJ/CIS5810_FinalProject.git sber-swap
%cd sber-swap              # ✅ %cd persists in Colab
# No submodule commands needed
# Directory is correctly named 'sber-swap'
```

---

## 📋 What Changed

1. **Branch specification**: `-b Youxin` gets the real-time implementation
2. **Directory naming**: Clone to `sber-swap` (cleaner, matches original)
3. **Path consistency**: Always use `/content/sber-swap` after cloning
4. **Removed submodules**: Not needed for our repository structure
5. **Fixed cd commands**: Use `%cd` for persistence, `!cd` only for shell commands

---

## 🎯 Alternative: Use the Real-Time Notebook

For real-time face swapping from webcam, use the dedicated notebook:
```
SberSwap_Realtime_Colab.ipynb
```

This includes:
- JavaScript webcam capture
- Real-time frame processing
- Performance metrics
- Step-by-step instructions

---

## 🐛 Troubleshooting

### Issue: "Directory not found"
**Fix**: Make sure you use `%cd` not `!cd`, and the directory is `/content/sber-swap`

### Issue: "Models not found"
**Fix**: Run `!bash download_models.sh` manually, or download models separately

### Issue: "Face swap didn't work"
**Fix**: 
- Check that both images have clear faces
- Ensure images exist at the specified paths
- Check error messages in output

### Issue: "Import errors after restart"
**Fix**: Make sure you restarted runtime after installation!

---

## ✅ Quick Checklist

- [ ] Enable GPU: Runtime → Change runtime type → GPU (T4)
- [ ] Run Cell 1 (Installation)
- [ ] Restart Runtime
- [ ] Run Cell 2 (Verify)
- [ ] Run Cell 3 (Face Swap)
- [ ] Run Cell 4 (Display Results)

---

## 📖 More Information

See `REALTIME_README.md` for real-time face swapping documentation.

