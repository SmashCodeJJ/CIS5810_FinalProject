# 🔧 Fix: ModuleNotFoundError for insightface

## 🐛 The Problem

You're getting:
```
ModuleNotFoundError: No module named 'insightface'
```

This usually means:
1. **Runtime wasn't restarted** after installation
2. **Dependencies didn't install correctly**
3. **Package name issue** (insightface vs insightface-python)

---

## ✅ Solution 1: Reinstall Dependencies (Recommended)

Run this cell **BEFORE** running inference:

```python
%cd /content/sber-swap

# Install/upgrade insightface
!pip install -q insightface==0.7.3

# Verify installation
import insightface
print(f"✅ InsightFace version: {insightface.__version__}")

# Check all critical imports
try:
    from insightface.utils import face_align
    print("✅ face_align imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Try alternative import
    import insightface
    print(f"InsightFace location: {insightface.__file__}")
```

---

## ✅ Solution 2: Complete Dependency Reinstall

If Solution 1 doesn't work, reinstall all dependencies:

```python
%cd /content/sber-swap

# Reinstall all dependencies
!pip install -q --upgrade -r requirements.txt

# Specifically install insightface
!pip install -q insightface==0.7.3

# Verify
import insightface
from insightface.utils import face_align
print("✅ All imports successful!")
```

---

## ✅ Solution 3: Fresh Start (Nuclear Option)

If nothing works, restart from scratch:

```python
# Cell 1: Clean installation
!rm -rf /content/sber-swap
!git clone -b Youxin https://github.com/SmashCodeJJ/CIS5810_FinalProject.git sber-swap
%cd sber-swap

# Install dependencies
%pip install -q -r requirements.txt

# Specifically ensure insightface is installed
!pip install -q insightface==0.7.3

print("✅ Installation complete!")
print("⚠️  RESTART RUNTIME NOW!")
```

Then after restart:
```python
%cd /content/sber-swap

# Verify imports
import torch
import insightface
from insightface.utils import face_align
import cv2
import numpy as np

print("✅ All imports successful!")
print(f"InsightFace: {insightface.__version__}")
print(f"PyTorch: {torch.__version__}")
```

---

## 🔍 Diagnostic Code

Run this to diagnose the issue:

```python
%cd /content/sber-swap

import sys
print(f"Python path: {sys.executable}")
print(f"Python version: {sys.version}")

# Check if insightface is installed
try:
    import insightface
    print(f"✅ InsightFace found: {insightface.__version__}")
    print(f"   Location: {insightface.__file__}")
except ImportError:
    print("❌ InsightFace not found")
    print("   Installing now...")
    !pip install -q insightface==0.7.3
    import insightface
    print(f"✅ Installed: {insightface.__version__}")

# Check specific imports
try:
    from insightface.utils import face_align
    print("✅ face_align import successful")
except ImportError as e:
    print(f"❌ face_align import failed: {e}")
    
# Check other critical imports
imports_to_check = [
    'torch',
    'cv2',
    'numpy',
    'onnxruntime',
]

for module in imports_to_check:
    try:
        m = __import__(module)
        version = getattr(m, '__version__', 'unknown')
        print(f"✅ {module}: {version}")
    except ImportError:
        print(f"❌ {module}: NOT FOUND")
```

---

## 📋 Corrected Complete Setup

### Step 1: Installation Cell
```python
!git clone -b Youxin https://github.com/SmashCodeJJ/CIS5810_FinalProject.git sber-swap
%cd sber-swap

# Install dependencies
%pip install -q -r requirements.txt

# Ensure insightface is installed
!pip install -q insightface==0.7.3

print("✅ Installation complete!")
print("⚠️  RESTART RUNTIME NOW!")
```

### Step 2: After Restart - Verify
```python
%cd /content/sber-swap

# Verify all imports work
import torch
import insightface
import cv2
import numpy as np
import onnxruntime as ort

print("="*50)
print("✅ All imports successful!")
print(f"PyTorch: {torch.__version__}")
print(f"InsightFace: {insightface.__version__}")
print(f"ONNX Runtime: {ort.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print("="*50)
```

### Step 3: Run Inference
```python
%cd /content/sber-swap

!python inference.py \
  --image_to_image True \
  --target_image examples/images/beckham.jpg \
  --source_paths examples/images/mark.jpg \
  --out_image_name examples/results/ghost_result.png
```

---

## 🎯 Quick Fix (Copy This Now)

Just run this cell to fix the immediate issue:

```python
%cd /content/sber-swap

# Install insightface
!pip install -q insightface==0.7.3

# Verify
try:
    import insightface
    from insightface.utils import face_align
    print("✅ InsightFace installed and imported successfully!")
    print(f"   Version: {insightface.__version__}")
except Exception as e:
    print(f"❌ Error: {e}")
    print("   Try restarting runtime and running again")
```

Then run your inference command again!

---

## 💡 Common Causes

1. **Didn't restart runtime** - Most common! Always restart after installation
2. **Installation failed silently** - Check for error messages in installation cell
3. **Wrong Python environment** - Colab might be using different Python
4. **Package version conflict** - Try specific version: `insightface==0.7.3`

---

## ✅ Checklist

- [ ] Installed dependencies with `%pip install -r requirements.txt`
- [ ] **RESTARTED RUNTIME** after installation
- [ ] Verified imports with diagnostic code
- [ ] Installed insightface specifically: `!pip install insightface==0.7.3`
- [ ] Running inference from correct directory: `/content/sber-swap`

