# 🔧 Quick Fix for Colab Code

## ❌ The Problem

You had this (WRONG):
```python
/content %cd /content/sber-swap  # ✅ Always use %cd
```

The `/content` at the beginning is being interpreted as a command!

## ✅ The Fix

Just use this (CORRECT):
```python
%cd /content/sber-swap
```

**Remove the `/content` at the beginning of the line!**

---

## 📋 Complete Corrected Code

### Cell 1: Installation (Run once, then RESTART RUNTIME)
```python
# Clone repository with real-time implementation
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
print("⚠️  IMPORTANT: Go to Runtime → Restart runtime")
print("="*50)
```

### Cell 2: Verify (Run AFTER restart)
```python
%cd /content/sber-swap

import torch
import numpy as np
import cv2

print("="*50)
print("✅ Environment Check")
print("="*50)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("="*50)
```

### Cell 3: Run Face Swap
```python
%cd /content/sber-swap

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
import filecmp
import os

result_path = '/content/sber-swap/examples/results/ghost_result.png'
target_path = '/content/sber-swap/examples/images/beckham.jpg'
source_path = '/content/sber-swap/examples/images/mark.jpg'

if os.path.exists(result_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Source
    if os.path.exists(source_path):
        source = cv2.imread(source_path)
        axes[0].imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Source (Mark)', fontsize=14)
        axes[0].axis('off')
    
    # Target
    if os.path.exists(target_path):
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
    
    # Check if same
    if os.path.exists(target_path):
        same = filecmp.cmp(target_path, result_path)
        if same:
            print("❌ RESULT IS IDENTICAL - Face swap didn't work")
        else:
            print("✅ RESULT IS DIFFERENT - Face swap worked!")
else:
    print("❌ Result file not found. Check for errors above.")
```

---

## 🎯 Key Points

1. **`%cd /content/sber-swap`** - Just this, nothing before it!
2. **No `/content` prefix** - Don't put `/content` at the start of the `%cd` line
3. **Use `%cd` not `!cd`** - `%cd` persists, `!cd` doesn't
4. **Path is `/content/sber-swap`** - This is correct after cloning

---

## ✅ Quick Test

After running Cell 2, you can verify the path:
```python
!pwd  # Should show: /content/sber-swap
!ls   # Should show: inference.py, weights/, examples/, etc.
```

