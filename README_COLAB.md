# Sber-Swap for Google Colab

**Modern fork of [sberbank-ai/sber-swap](https://github.com/sberbank-ai/sber-swap)** updated for Google Colab compatibility with Python 3.12 and PyTorch 2.x.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SmashCodeJJ/sber-swap-colab/blob/main/SberSwap_Colab_Ready.ipynb)

## 🎯 What's New in This Fork?

### ✅ Major Updates
- **Python 3.12** compatible (original: Python 3.7)
- **PyTorch 2.2.0** (original: 1.6.0)
- **ONNX Runtime** instead of deprecated MXNet
- **InsightFace 0.7+** API compatibility
- **NumPy 2.0** compatible
- Fixed all transformation matrix issues
- Robust error handling throughout

### 🔧 Files Modified
1. `requirements.txt` - Updated all dependencies
2. `coordinate_reg/image_infer.py` - Replaced MXNet with ONNX Runtime
3. `insightface_func/face_detect_crop_single.py` - Fixed InsightFace API
4. `insightface_func/face_detect_crop_multi.py` - Fixed InsightFace API
5. `utils/inference/image_processing.py` - Fixed transformation matrices
6. `utils/inference/video_processing.py` - Fixed transformation matrices
7. `utils/inference/core.py` - Added validation checks
8. `utils/inference/util.py` - Fixed regex warnings
9. `models/networks/normalization.py` - Fixed regex warnings
10. `inference.py` - Enhanced error messages

---

## 🚀 Quick Start (Google Colab)

### Method 1: Using Jupyter Notebook
Click the badge above or open [SberSwap_Colab_Ready.ipynb](https://colab.research.google.com/github/SmashCodeJJ/sber-swap-colab/blob/main/SberSwap_Colab_Ready.ipynb) in Colab.

### Method 2: Manual Setup

#### Cell 1: Clone and Install
```python
# Clone repository
!git clone https://github.com/SmashCodeJJ/sber-swap-colab.git
%cd sber-swap-colab

# Initialize submodules
!git submodule init
!git submodule update

# Install dependencies
%pip install -q -r requirements.txt

# Download pre-trained models
!bash download_models.sh

print("✅ Installation complete!")
print("⚠️  IMPORTANT: Click 'Runtime > Restart runtime' NOW")
print("    Then run Cell 2 for face swapping")
```

#### Cell 2: Run Face Swap (After Restart!)
```python
%cd /content/sber-swap-colab

# Basic example: Swap faces
!python inference.py \
  --image_to_image True \
  --target_image examples/images/beckham.jpg \
  --source_paths examples/images/mark.jpg \
  --out_image_name examples/results/result.png

# Display result
from IPython.display import Image, display
display(Image('examples/results/result.png'))
```

---

## 📦 Dependencies

| Package | Original | Updated |
|---------|----------|---------|
| Python | 3.7 | 3.12 |
| PyTorch | 1.6.0+cu101 | 2.2.0 |
| torchvision | 0.7.0+cu101 | 0.17.0 |
| mxnet | mxnet-cu101mkl | ❌ Removed (replaced with ONNX) |
| onnxruntime-gpu | 1.4.0 | 1.16.3+ |
| insightface | 0.2.1 | 0.7.3+ |
| kornia | 0.5.4 | 0.6.12+ |
| numpy | - | Compatible with 2.0+ |

---

## 🎬 Usage

### Image-to-Image Face Swap
```bash
python inference.py \
  --image_to_image True \
  --target_image path/to/target.jpg \
  --source_paths path/to/source.jpg \
  --out_image_name output.png
```

### Video Face Swap
```bash
python inference.py \
  --target_path path/to/video.mp4 \
  --source_paths path/to/face.jpg \
  --output_path output.mp4
```

### Multi-Face Swap
```bash
python inference.py \
  --target_path path/to/video.mp4 \
  --source_paths face1.jpg face2.jpg face3.jpg \
  --output_path output.mp4
```

---

## 🐛 Troubleshooting

### "Bad source images!" Error
- Ensure images are valid and readable
- Check that faces are clearly visible
- Try images with frontal face poses

### "No valid frames after resizing"
- Face detection failed on target image
- Try different images or adjust detection threshold

### Import Errors After Installation
- **ALWAYS** restart the Colab runtime after installation
- Don't skip the restart step!

### torchaudio Warning
- Safe to ignore: `torchaudio 2.8.0+cu126 requires torch==2.8.0`
- Or run: `!pip uninstall torchaudio -y`

---

## 📝 Technical Details

### Why These Changes?

1. **MXNet → ONNX Runtime**: MXNet is deprecated and incompatible with Python 3.10+
2. **InsightFace API**: Removed `threshold` parameter (API changed in v0.7+)
3. **Face Alignment**: Fixed `mode='None'` parameter causing matrix shape errors
4. **Matrix Validation**: Added checks for transformation matrix shapes (2,3)
5. **NumPy 2.0**: Replaced deprecated `np.bool` usage

### Architecture
- **Face Detection**: InsightFace SCRFD model
- **Face Recognition**: ArcFace (ResNet100)
- **Face Swapping**: Modified FaceShifter with AAD layers
- **Landmark Detection**: ONNX models (fallback to InsightFace)

---

## 🙏 Credits

- Original: [sberbank-ai/sber-swap](https://github.com/sberbank-ai/sber-swap)
- InsightFace: [deepinsight/insightface](https://github.com/deepinsight/insightface)
- FaceShifter: [Paper](https://arxiv.org/abs/1908.05932)

---

## 📄 License

Same as original sber-swap repository. See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Found a bug or want to improve compatibility? PRs welcome!

---

**Made with ❤️ for Google Colab users**  
Last updated: October 2025


