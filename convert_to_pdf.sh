#!/bin/bash

# Script to convert PROJECT_PROPOSAL.md to PDF
# Requires: pandoc and texlive-latex-base

echo "Converting PROJECT_PROPOSAL.md to PDF..."

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null
then
    echo "Error: pandoc is not installed"
    echo "Install with: sudo apt-get install pandoc texlive-latex-base"
    echo "Or on Mac: brew install pandoc basictex"
    exit 1
fi

# Convert Markdown to PDF with custom settings for 3-page limit
pandoc PROJECT_PROPOSAL.md \
    -o PROJECT_PROPOSAL.pdf \
    --pdf-engine=pdflatex \
    -V geometry:margin=0.75in \
    -V fontsize=11pt \
    -V linestretch=1.0 \
    --toc=false \
    2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ PDF created successfully: PROJECT_PROPOSAL.pdf"
    echo "📄 Checking page count..."
    
    # Check page count (requires pdfinfo)
    if command -v pdfinfo &> /dev/null; then
        pages=$(pdfinfo PROJECT_PROPOSAL.pdf | grep "Pages:" | awk '{print $2}')
        echo "📊 Total pages: $pages"
        
        if [ "$pages" -gt 3 ]; then
            echo "⚠️  WARNING: Document exceeds 3 pages ($pages pages)"
            echo "💡 Consider using PROJECT_PROPOSAL_SHORT.md for a condensed version"
        else
            echo "✅ Page count is within limit (≤3 pages)"
        fi
    fi
    
    # Show file size
    size=$(du -h PROJECT_PROPOSAL.pdf | cut -f1)
    echo "💾 File size: $size"
    
else
    echo "❌ Error: PDF conversion failed"
    echo "Trying alternative method..."
    
    # Alternative: Use markdown-pdf (npm package)
    if command -v markdown-pdf &> /dev/null; then
        markdown-pdf PROJECT_PROPOSAL.md
        echo "✅ PDF created with markdown-pdf"
    else
        echo "❌ Failed. Please install pandoc or use an online converter:"
        echo "   - https://www.markdowntopdf.com/"
        echo "   - https://cloudconvert.com/md-to-pdf"
    fi
fi

echo ""
echo "📝 Alternative conversion methods:"
echo "   1. Google Docs: File → Import → Upload .md → Download as PDF"
echo "   2. VS Code: Install 'Markdown PDF' extension"
echo "   3. Online: Upload to https://www.markdowntopdf.com/"

