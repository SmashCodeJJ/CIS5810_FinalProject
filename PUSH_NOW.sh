#!/bin/bash
# Simple push script for CIS5810_FinalProject
# Author: SmashCodeJJ

echo "======================================"
echo "Pushing to GitHub"
echo "======================================"
echo ""
echo "Repository: https://github.com/SmashCodeJJ/CIS5810_FinalProject"
echo ""

cd "$(dirname "$0")"

# Check if we have commits
if ! git log &>/dev/null; then
    echo "❌ No commits found!"
    exit 1
fi

# Show what we're about to push
echo "📦 Files to push:"
git ls-files | wc -l | xargs echo "   Total files:"
git count-objects -vH | grep "size:" 

echo ""
echo "🚀 Pushing to GitHub..."
echo ""
echo "You will be prompted for:"
echo "  Username: SmashCodeJJ"
echo "  Password: Your Personal Access Token"
echo ""
echo "Don't have a token? Get one at:"
echo "  👉 https://github.com/settings/tokens/new"
echo "     (Check 'repo' scope, then generate)"
echo ""

# Try to push
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "✅ SUCCESS!"
    echo "======================================"
    echo ""
    echo "Your code is now at:"
    echo "👉 https://github.com/SmashCodeJJ/CIS5810_FinalProject"
    echo ""
else
    echo ""
    echo "======================================"
    echo "❌ Push failed!"
    echo "======================================"
    echo ""
    echo "Common issues:"
    echo "1. Wrong password - Use Personal Access Token, not GitHub password"
    echo "2. Token missing 'repo' scope - Create new token with full repo access"
    echo "3. Network issues - Check your internet connection"
    echo ""
    echo "Get a new token at: https://github.com/settings/tokens/new"
    echo ""
fi


