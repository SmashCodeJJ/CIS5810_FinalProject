#!/bin/bash
# Clean up temporary files before pushing to GitHub
# Author: SmashCodeJJ

echo "🧹 Cleaning up temporary files..."

cd "$(dirname "$0")"

# Remove temporary documentation files
echo "📄 Removing temporary documentation..."
rm -f COLAB_*.md
rm -f MODERNIZATION_*.md
rm -f UPDATED_*.md
rm -f ALL_FIXES_*.md
rm -f FINAL_*.md
rm -f COLAB_*.txt

# Remove debug and test scripts
echo "🐛 Removing debug scripts..."
rm -f debug_*.py
rm -f test_*.py
rm -f convert_mxnet_to_onnx.py

# Remove Complete_Update_Summary.ipynb (we have SberSwap_Colab_Ready.ipynb)
echo "📓 Removing duplicate notebook..."
rm -f Complete_Update_Summary.ipynb
rm -f Complete_Colab_Update_Guide.ipynb

# Keep only the cleaned files
echo "✅ Cleanup complete!"
echo ""
echo "Files ready for GitHub:"
echo "  ✓ SberSwap_Colab_Ready.ipynb (main notebook)"
echo "  ✓ README_COLAB.md (documentation)"
echo "  ✓ .gitignore (ignore rules)"
echo "  ✓ All updated Python files"
echo ""
echo "Next step: Run ./push_to_github.sh"

