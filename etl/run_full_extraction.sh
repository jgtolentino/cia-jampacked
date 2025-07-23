#!/bin/bash

echo "🎯 JamPacked + Claude Desktop Full Campaign Extraction"
echo "====================================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Create output directories
mkdir -p output/real_campaigns_extraction
mkdir -p output/jampacked_claude_extraction
mkdir -p logs

# Option 1: Run standalone extraction (no Google Drive required)
echo "1️⃣ Running standalone feature extraction..."
echo "   This extracts features from 32 real campaigns"
echo ""
python3 etl/real_campaigns_full_extraction.py

if [ $? -eq 0 ]; then
    echo "✅ Standalone extraction completed!"
else
    echo "❌ Standalone extraction failed"
fi

echo ""
echo "2️⃣ For Google Drive integration with Claude Desktop:"
echo "   - Ensure Claude Desktop is running with MCP Google Drive server"
echo "   - Run: python3 etl/jampacked_claude_desktop_integration.py"
echo ""
echo "   Or if you have credentials:"
echo "   - export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json"
echo "   - python3 etl/google_drive_full_etl.py"

echo ""
echo "📊 Results will be saved to:"
echo "   - output/real_campaigns_extraction/ (standalone)"
echo "   - output/jampacked_claude_extraction/ (integrated)"
echo "   - output/gdrive_etl/ (Google Drive)"

echo ""
echo "🚀 Next steps:"
echo "   1. Check extraction results in output directories"
echo "   2. Load data into JamPacked platform"
echo "   3. Run predictive models and analysis"
echo "   4. Generate insights dashboard"