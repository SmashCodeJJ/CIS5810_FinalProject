# 🎭 Simple Image-to-Image Face Swapping in Colab

## ✅ Use This Instead!

Since you want **image swapping** (not real-time camera), use the **Simple Notebook**:

**Notebook**: `SberSwap_Simple_Colab.ipynb`  
**Purpose**: Image-to-image face swapping (no webcam needed)

---

## 🚀 Quick Start (Simple Image Swap)

### Cell 1: Installation (Run once, then RESTART)
```python
!git clone -b Youxin https://github.com/SmashCodeJJ/CIS5810_FinalProject.git sber-swap
%cd sber-swap
%pip install -q -r requirements.txt

print("✅ Restart runtime now!")
```

### Cell 2: Run Face Swap (After restart)
```python
%cd /content/sber-swap

!python inference.py \
  --image_to_image True \
  --target_image examples/images/beckham.jpg \
  --source_paths examples/images/mark.jpg \
  --out_image_name examples/results/ghost_result.png

print("✅ Face swap complete!")
```

### Cell 3: Display Results
```python
from IPython.display import display, Image as IPImage
import matplotlib.pyplot as plt
import cv2
import filecmp
import os

result_path = '/content/sber-swap/examples/results/ghost_result.png'
target_path = '/content/sber-swap/examples/images/beckham.jpg'
source_path = '/content/sber-swap/examples/images/mark.jpg'

if os.path.exists(result_path):
    # Display comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Source
    source = cv2.imread(source_path)
    axes[0].imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Source (Mark)', fontsize=14)
    axes[0].axis('off')
    
    # Target
    target = cv2.imread(target_path)
    axes[1].imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Target (Beckham)', fontsize=14)
    axes[1].axis('off')
    
    # Result
    result = cv2.imread(result_path)
    axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[2].set_title('RESULT - Face Swapped', fontsize=14, color='green', weight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Check if swap worked
    same = filecmp.cmp(target_path, result_path)
    if same:
        print("❌ RESULT IS IDENTICAL - Face swap didn't work")
    else:
        print("✅ RESULT IS DIFFERENT - Face swap worked!")
```

---

## 📝 What Changed

### ✅ Fixed Issues:
1. **Face Tracker Error**: Fixed `det_model` access (uses models dict fallback)
2. **Image Import**: Uses `IPImage` alias to avoid conflicts
3. **Simple Notebook**: Created `SberSwap_Simple_Colab.ipynb` for image-only swapping

### 🎯 For Image Swapping:
- ✅ Use `inference.py` (not `inference_realtime.py`)
- ✅ Use `SberSwap_Simple_Colab.ipynb` (not real-time notebook)
- ✅ No webcam/camera needed
- ✅ Just works with static images

---

## 📊 Notebook Comparison

| Notebook | Purpose | Use Case |
|----------|---------|----------|
| `SberSwap_Simple_Colab.ipynb` | Image-to-image | ✅ **Use this for image swapping** |
| `SberSwap_Realtime_Colab.ipynb` | Real-time webcam | Use only if you need webcam |

---

## 🔧 Your Current Code (Fixed)

Here's your exact code with fixes:

```python
# Cell 1: Setup
!git clone -b Youxin https://github.com/SmashCodeJJ/CIS5810_FinalProject.git
%cd sber-swap  # ✅ Use %cd, not !cd
%pip install -q -r requirements.txt

# Cell 2: After restart
%cd /content/sber-swap  # ✅ Make sure we're in right directory

# Run face swap
!python inference.py \
  --image_to_image True \
  --target_image examples/images/beckham.jpg \
  --source_paths examples/images/mark.jpg \
  --out_image_name examples/results/ghost_result.png

# Cell 3: Display (with fixed import)
from IPython.display import display, Image as IPImage  # ✅ Use IPImage
import matplotlib.pyplot as plt
import cv2
import filecmp
import os

result_path = '/content/sber-swap/examples/results/ghost_result.png'
target_path = '/content/sber-swap/examples/images/beckham.jpg'
source_path = '/content/sber-swap/examples/images/mark.jpg'

if os.path.exists(result_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    if os.path.exists(source_path):
        source = cv2.imread(source_path)
        axes[0].imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Source (Mark)', fontsize=14)
        axes[0].axis('off')
    
    if os.path.exists(target_path):
        target = cv2.imread(target_path)
        axes[1].imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Target (Beckham)', fontsize=14)
        axes[1].axis('off')
    
    result = cv2.imread(result_path)
    axes[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[2].set_title('RESULT - Face Swapped', fontsize=14, color='green', weight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Check if swap worked
    if os.path.exists(target_path):
        same = filecmp.cmp(target_path, result_path)
        if same:
            print("❌ RESULT IS IDENTICAL TO TARGET - Face swap didn't work")
        else:
            print("✅ RESULT IS DIFFERENT - Face swap worked!")
```

---

## 🐛 About That Face Detection Error

The error you saw (`'Face_detect_crop' object has no attribute 'detect'`) was from the **real-time face tracker**, which you don't need for simple image swapping.

**Solution**: 
- ✅ Don't use `inference_realtime.py` for image swapping
- ✅ Use `inference.py` instead (it's simpler and doesn't need face tracker)
- ✅ The face tracker error won't appear with `inference.py`

---

## ✅ Recommended Approach

1. **For Image Swapping** → Use `SberSwap_Simple_Colab.ipynb`
   - Simple, straightforward
   - No real-time components
   - Just image-to-image swapping

2. **For Real-Time** → Use `SberSwap_Realtime_Colab.ipynb`
   - Only if you need webcam
   - Has face tracker (now fixed)
   - More complex setup

---

## 🎯 Summary

**For your use case (image swapping, not camera)**:
- ✅ Use `inference.py` (not real-time version)
- ✅ Use the simple notebook or your fixed code above
- ✅ Ignore face tracker errors (they're from real-time code you're not using)
- ✅ Focus on getting image swap working first

All fixes are pushed to GitHub!

