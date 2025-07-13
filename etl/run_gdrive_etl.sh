#!/bin/bash
# Google Drive ETL Script for Claude Desktop

echo "🚀 Starting Google Drive ETL Pipeline"
echo "📁 Folder ID: 0AJMhu01UUQKoUk9PVA"

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the ETL
python3 etl/google_drive_full_etl.py 0AJMhu01UUQKoUk9PVA

# Check if successful
if [ $? -eq 0 ]; then
    echo "✅ ETL completed successfully!"
    echo "📊 Check output/gdrive_etl/ for results"
else
    echo "❌ ETL failed. Check logs/gdrive_etl.log for details"
fi
