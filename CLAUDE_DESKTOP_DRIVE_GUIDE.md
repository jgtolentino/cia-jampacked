# 🚀 Claude Desktop Google Drive Integration Guide

## Overview

This guide shows how to use Claude Desktop's native Google Drive integration to extract Pulser awards data without requiring service accounts or API keys.

## ✅ Advantages

- **No Authentication Setup** - Uses Claude Desktop's existing Google account
- **Native Integration** - Direct access to Google Drive through Claude
- **Visual Interface** - Can see and navigate files directly
- **No API Limits** - Doesn't count against Google API quotas
- **Simpler Workflow** - No service account JSON files to manage

## 📋 How It Works

### 1. **Automated Extraction via Task Queue**

```javascript
// The system creates a task for Claude Desktop
const taskId = await extractor.createExtractionTask({
  folder_id: '0AJMhu01UUQKoUk9PVA',
  instructions: 'Extract award data from Pulser folder'
});
```

### 2. **Claude Desktop Processes the Task**

When Claude Desktop picks up the task, it will:
1. Access the Google Drive folder using its native integration
2. Read spreadsheet files containing award data
3. Extract and structure the information
4. Return results to the task queue

### 3. **Manual Extraction (Alternative)**

You can also ask Claude Desktop directly:

```
Please extract award data from this Google Drive folder:
https://drive.google.com/drive/folders/0AJMhu01UUQKoUk9PVA

Look for spreadsheets with:
- Campaign names
- Client/Brand information  
- Award shows (Cannes, D&AD, One Show, etc.)
- Award levels (Gold, Silver, Bronze)
- Years

Format the results as a CSV or table that I can import.
```

## 🔧 Setup Instructions

### 1. **Ensure Claude Desktop Has Google Drive Access**

Claude Desktop should already have access to Google Drive. If not:
- Open Claude Desktop
- Check if you can access Google Drive files
- Follow any prompts to connect your Google account

### 2. **Share Pulser Folder (if needed)**

If the folder isn't accessible:
- Ask the folder owner to share it with your Google account
- Or request view access to folder ID: `0AJMhu01UUQKoUk9PVA`

### 3. **Test the Integration**

Run a test extraction:
```bash
# Create a test task
node -e "
const { ClaudeDesktopDriveExtractor } = require('./mcp-integration/claude-desktop-drive-extractor');
const extractor = new ClaudeDesktopDriveExtractor();
extractor.initialize().then(() => {
  extractor.createExtractionTask({ test: true });
});
"
```

## 📊 Expected Output Format

Claude Desktop will return data in this format:

```json
{
  "files_processed": [
    "Cannes Lions 2024.xlsx",
    "D&AD Winners 2024.csv"
  ],
  "awards": [
    {
      "campaign_name": "EcoFuture Campaign",
      "client": "GreenTech Corp",
      "brand": "EcoSmart",
      "agency": "Pulser",
      "award_show": "cannes_lions",
      "award_year": 2024,
      "award_category": "Film",
      "award_level": "Gold",
      "country": "USA"
    }
  ]
}
```

## 🤖 Integration with JamPacked

The system automatically:
1. Creates extraction tasks in the queue
2. Claude Desktop processes them when available
3. Results are stored in the SQLite database
4. JamPacked can then analyze the award data

## 📝 SQL Alternative

You can also use Claude Desktop's SQL interface directly:

```sql
-- Ask Claude Desktop to execute this after extracting data
INSERT INTO campaign_awards 
(award_id, campaign_id, award_show, award_year, award_category, 
 award_level, award_title, client, brand, agency, country)
VALUES 
-- Claude will fill in the actual values from Google Drive
(?, ?, ?, ?, ?, ?, ?, ?, ?, 'Pulser', ?);
```

## 🔄 Scheduled Extraction

The system can automatically schedule extraction tasks:
- Weekly on Sundays at 2 AM
- Or manually trigger with: `npm run extract-awards`

## ❓ Troubleshooting

**Issue: Claude Desktop can't access the folder**
- Solution: Ensure the folder is shared with your Google account

**Issue: Extraction task times out**
- Solution: Break into smaller tasks by year or award show

**Issue: Data format inconsistencies**
- Solution: Claude Desktop handles various formats automatically

## 🎯 Benefits Over Service Account

| Feature | Service Account | Claude Desktop |
|---------|----------------|----------------|
| Setup Required | Complex (JSON keys, IAM) | None |
| Authentication | Service Account | Your Google Account |
| Access Method | API calls | Native UI |
| Rate Limits | Yes | No |
| Visual Verification | No | Yes |
| Error Handling | Manual | Built-in |

---

**Ready to use!** The Claude Desktop integration makes Google Drive extraction simple and reliable without any authentication hassles.