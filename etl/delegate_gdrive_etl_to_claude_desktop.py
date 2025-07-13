#!/usr/bin/env python3
"""
Delegate Google Drive ETL to Claude Desktop
This script creates the ETL request that Claude Desktop can execute with proper authentication
"""

import json
import os
from datetime import datetime
from pathlib import Path

def create_etl_request(folder_id: str = "0AJMhu01UUQKoUk9PVA"):
    """Create ETL request for Claude Desktop"""
    
    request = {
        "task": "google_drive_full_etl",
        "description": "Extract all campaign data from Google Drive folder and apply comprehensive feature engineering",
        "folder_id": folder_id,
        "timestamp": datetime.now().isoformat(),
        "steps": [
            {
                "step": 1,
                "action": "authenticate",
                "description": "Authenticate with Google Drive API using available credentials"
            },
            {
                "step": 2,
                "action": "list_files",
                "description": "Recursively list all files in the folder",
                "parameters": {
                    "folder_id": folder_id,
                    "recursive": True,
                    "include_metadata": True
                }
            },
            {
                "step": 3,
                "action": "download_data_files",
                "description": "Download all CSV, Excel, and Google Sheets files",
                "parameters": {
                    "file_types": ["csv", "xlsx", "xls", "google-sheets"],
                    "export_google_sheets_as": "csv"
                }
            },
            {
                "step": 4,
                "action": "extract_features",
                "description": "Extract 80+ comprehensive features from campaign data",
                "feature_categories": [
                    "visual_features",
                    "text_features",
                    "audio_features",
                    "campaign_features",
                    "cultural_features",
                    "temporal_features",
                    "network_features",
                    "hierarchical_features",
                    "survival_features",
                    "performance_metrics"
                ]
            },
            {
                "step": 5,
                "action": "save_results",
                "description": "Save processed data in multiple formats",
                "outputs": {
                    "formats": ["csv", "parquet", "json"],
                    "include_raw_data": True,
                    "include_feature_engineered_data": True,
                    "include_summary_report": True
                }
            }
        ],
        "expected_outputs": {
            "raw_campaigns_csv": "gdrive_raw_campaigns_TIMESTAMP.csv",
            "campaign_features_csv": "gdrive_campaign_features_TIMESTAMP.csv",
            "campaign_features_parquet": "gdrive_campaign_features_TIMESTAMP.parquet",
            "campaign_features_json": "gdrive_campaign_features_TIMESTAMP.json",
            "etl_summary_json": "etl_summary_TIMESTAMP.json"
        },
        "claude_desktop_instructions": """
To execute this ETL in Claude Desktop:

1. Ensure you have Google Drive API access configured
2. Use the MCP Google Drive server if available
3. Run the following Python script with proper authentication:
   python etl/google_drive_full_etl.py {folder_id}
   
4. The script will:
   - Authenticate with Google Drive
   - Download all campaign data files
   - Extract 80+ features per campaign
   - Save results in multiple formats
   
5. Check the output/gdrive_etl directory for results
""".format(folder_id=folder_id)
    }
    
    # Save request for Claude Desktop
    request_path = Path("etl/claude_desktop_etl_request.json")
    with open(request_path, 'w') as f:
        json.dump(request, f, indent=2)
    
    print("=" * 60)
    print("📋 GOOGLE DRIVE ETL REQUEST FOR CLAUDE DESKTOP")
    print("=" * 60)
    print(f"\n📁 Target Folder ID: {folder_id}")
    print("\n🎯 ETL Pipeline Steps:")
    for step in request['steps']:
        print(f"   {step['step']}. {step['description']}")
    
    print("\n📊 Expected Outputs:")
    for output_type, filename in request['expected_outputs'].items():
        print(f"   - {filename}")
    
    print("\n💡 Next Steps:")
    print("1. Open Claude Desktop")
    print("2. Ensure Google Drive MCP server is connected")
    print("3. Run: python etl/google_drive_full_etl.py")
    print(f"4. Results will be in: output/gdrive_etl/")
    
    print(f"\n✅ Request saved to: {request_path}")
    
    # Also create a simple bash script for Claude Desktop
    bash_script = """#!/bin/bash
# Google Drive ETL Script for Claude Desktop

echo "🚀 Starting Google Drive ETL Pipeline"
echo "📁 Folder ID: {folder_id}"

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the ETL
python3 etl/google_drive_full_etl.py {folder_id}

# Check if successful
if [ $? -eq 0 ]; then
    echo "✅ ETL completed successfully!"
    echo "📊 Check output/gdrive_etl/ for results"
else
    echo "❌ ETL failed. Check logs/gdrive_etl.log for details"
fi
""".format(folder_id=folder_id)
    
    bash_path = Path("etl/run_gdrive_etl.sh")
    with open(bash_path, 'w') as f:
        f.write(bash_script)
    os.chmod(bash_path, 0o755)
    
    print(f"\n🔧 Bash script created: {bash_path}")
    print("   Run with: ./etl/run_gdrive_etl.sh")
    
    return request

def main():
    """Create the ETL request"""
    import sys
    folder_id = sys.argv[1] if len(sys.argv) > 1 else "0AJMhu01UUQKoUk9PVA"
    
    request = create_etl_request(folder_id)
    
    print("\n" + "=" * 60)
    print("📌 IMPORTANT: This ETL requires Google Drive API authentication")
    print("   Claude Desktop can handle this with its MCP Google Drive server")
    print("=" * 60)

if __name__ == "__main__":
    main()