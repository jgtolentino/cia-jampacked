# Google Drive API Service Account Setup

1. Go to Google Cloud Console: https://console.cloud.google.com
2. Create a new project or select existing one
3. Enable Google Drive API and Google Sheets API
4. Create a service account:
   - Go to "IAM & Admin" > "Service Accounts"
   - Click "Create Service Account"
   - Name it "jampacked-drive-access"
   - Grant it "Viewer" role
5. Create and download JSON key:
   - Click on the service account
   - Go to "Keys" tab
   - Add Key > Create New Key > JSON
   - Save as "service-account-key.json" in this directory
6. Share Pulser awards folder with service account email
   - Get email from service account (ends with @*.iam.gserviceaccount.com)
   - Share folder with this email as "Viewer"

Folder ID: 0AJMhu01UUQKoUk9PVA
