#!/bin/bash
# Quick script to run real-time face swapping locally
# Usage: ./RUN_REALTIME_LOCAL.sh

echo "🚀 Local Real-Time Face Swapping Setup"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -f "inference_realtime.py" ]; then
    echo "❌ Error: inference_realtime.py not found!"
    echo "   Please run this script from the sber-swap directory"
    exit 1
fi

# Check Python
echo "✅ Checking Python..."
python --version

# Check NumPy version
echo ""
echo "🔍 Checking NumPy version..."
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "   NumPy: $NUMPY_VERSION"
    if [[ $(echo "$NUMPY_VERSION >= 2.0" | bc -l 2>/dev/null || echo "1") == "1" ]]; then
        echo "   ⚠️  Warning: NumPy >= 2.0 may cause issues"
        echo "   Consider: pip install 'numpy<2.0'"
    fi
else
    echo "   ❌ NumPy not installed"
fi

# Check PyTorch and CUDA
echo ""
echo "🔍 Checking PyTorch..."
python -c "import torch; print(f'   PyTorch: {torch.__version__}'); print(f'   CUDA: {torch.cuda.is_available()}')" 2>/dev/null

# Check camera
echo ""
echo "🔍 Checking camera access..."
python -c "import cv2; cap = cv2.VideoCapture(0); print('   ✅ Camera accessible!' if cap.isOpened() else '   ❌ Camera not accessible'); cap.release()" 2>/dev/null

# Check models
echo ""
echo "🔍 Checking models..."
if [ -f "weights/G_unet_2blocks.pth" ]; then
    echo "   ✅ Generator model found"
else
    echo "   ❌ Generator model missing: weights/G_unet_2blocks.pth"
fi

if [ -f "arcface_model/backbone.pth" ]; then
    echo "   ✅ ArcFace model found"
else
    echo "   ❌ ArcFace model missing: arcface_model/backbone.pth"
fi

echo ""
echo "========================================"
echo "📝 To run real-time face swapping:"
echo ""
echo "  python inference_realtime.py \\"
echo "    --source_path examples/images/mark.jpg \\"
echo "    --camera_id 0 \\"
echo "    --fast_mode"
echo ""
echo "Controls:"
echo "  - Press 'q' to quit"
echo "  - Press 'r' to reset tracker"
echo "  - Press 's' to save frame"
echo "========================================"

