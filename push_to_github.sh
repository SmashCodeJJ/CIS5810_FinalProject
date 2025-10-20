#!/bin/bash
# Push sber-swap-colab to GitHub
# Author: SmashCodeJJ

echo "🚀 Pushing sber-swap-colab to GitHub..."

# Navigate to project directory
cd "$(dirname "$0")"

# Initialize git if needed
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
fi

# Add all files
echo "➕ Adding files..."
git add .

# Commit
echo "💾 Creating commit..."
git commit -m "Initial commit: Sber-Swap updated for Colab compatibility

- Updated requirements.txt for Python 3.12 and PyTorch 2.x
- Replaced MXNet with ONNX Runtime
- Fixed InsightFace API compatibility (removed threshold parameter)
- Fixed face_align transformation matrix issues
- Added robust error handling and validation
- All dependencies compatible with Google Colab"

# Add remote (update if already exists)
echo "🔗 Adding GitHub remote..."
git remote remove origin 2>/dev/null
git remote add origin https://github.com/SmashCodeJJ/sber-swap-colab.git

# Push to GitHub
echo "📤 Pushing to GitHub..."
git branch -M main
git push -u origin main

echo "✅ Done! Your code is now at: https://github.com/SmashCodeJJ/sber-swap-colab"
echo "📋 Next: Open Google Colab and clone your repo!"

