const { google } = require('googleapis');
const fs = require('fs').promises;
const path = require('path');

/**
 * Google Drive API Setup for Pulser Awards Database
 * Folder ID: 0AJMhu01UUQKoUk9PVA
 * 
 * NOTE: This module is deprecated in favor of Claude Desktop integration
 * See: claude-desktop-drive-extractor.js for the new approach
 * 
 * @deprecated Use ClaudeDesktopDriveExtractor instead
 */
class GoogleDriveIntegration {
  constructor() {
    this.drive = null;
    this.auth = null;
    this.Pulser_AWARDS_FOLDER_ID = '0AJMhu01UUQKoUk9PVA';
  }

  /**
   * Initialize Google Drive API with service account or OAuth2
   */
  async initialize() {
    try {
      // Try service account first (recommended for server applications)
      const serviceAccountPath = path.join(__dirname, 'service-account-key.json');
      
      if (await this.fileExists(serviceAccountPath)) {
        // Use service account
        const serviceAccount = JSON.parse(await fs.readFile(serviceAccountPath, 'utf8'));
        
        this.auth = new google.auth.GoogleAuth({
          credentials: serviceAccount,
          scopes: ['https://www.googleapis.com/auth/drive.readonly']
        });
      } else {
        // Use OAuth2 (requires user interaction)
        this.auth = await this.setupOAuth2();
      }

      this.drive = google.drive({ version: 'v3', auth: this.auth });
      console.log('✅ Google Drive API initialized successfully');
      
      return true;
    } catch (error) {
      console.error('❌ Failed to initialize Google Drive API:', error.message);
      throw error;
    }
  }

  /**
   * Setup OAuth2 authentication
   */
  async setupOAuth2() {
    const OAuth2 = google.auth.OAuth2;
    
    // These would typically come from environment variables
    const CLIENT_ID = process.env.GOOGLE_CLIENT_ID || 'your-client-id';
    const CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET || 'your-client-secret';
    const REDIRECT_URI = process.env.GOOGLE_REDIRECT_URI || 'http://localhost:3000/oauth2callback';
    
    const oauth2Client = new OAuth2(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI);
    
    // Check for stored tokens
    const tokenPath = path.join(__dirname, 'tokens.json');
    if (await this.fileExists(tokenPath)) {
      const tokens = JSON.parse(await fs.readFile(tokenPath, 'utf8'));
      oauth2Client.setCredentials(tokens);
    } else {
      // Generate auth URL for first-time setup
      const authUrl = oauth2Client.generateAuthUrl({
        access_type: 'offline',
        scope: ['https://www.googleapis.com/auth/drive.readonly']
      });
      
      console.log('🔐 Authorize this app by visiting this url:', authUrl);
      throw new Error('OAuth2 setup required. Please visit the auth URL and configure tokens.');
    }
    
    return oauth2Client;
  }

  /**
   * List all files in Pulser awards folder
   */
  async listAwardsFiles() {
    if (!this.drive) {
      throw new Error('Google Drive API not initialized. Call initialize() first.');
    }

    try {
      const response = await this.drive.files.list({
        q: `'${this.Pulser_AWARDS_FOLDER_ID}' in parents and trashed = false`,
        fields: 'nextPageToken, files(id, name, mimeType, modifiedTime, size)',
        pageSize: 1000
      });

      const files = response.data.files;
      console.log(`📁 Found ${files.length} files in Pulser awards folder`);
      
      return files;
    } catch (error) {
      console.error('❌ Error listing files:', error.message);
      throw error;
    }
  }

  /**
   * Download a specific file
   */
  async downloadFile(fileId, fileName) {
    if (!this.drive) {
      throw new Error('Google Drive API not initialized');
    }

    try {
      const destPath = path.join(__dirname, 'downloads', fileName);
      await this.ensureDirectory(path.dirname(destPath));

      const dest = fs.createWriteStream(destPath);
      
      const response = await this.drive.files.get(
        { fileId: fileId, alt: 'media' },
        { responseType: 'stream' }
      );

      return new Promise((resolve, reject) => {
        response.data
          .on('end', () => {
            console.log(`✅ Downloaded: ${fileName}`);
            resolve(destPath);
          })
          .on('error', err => {
            console.error(`❌ Error downloading ${fileName}:`, err);
            reject(err);
          })
          .pipe(dest);
      });
    } catch (error) {
      console.error('❌ Download error:', error.message);
      throw error;
    }
  }

  /**
   * Get file metadata including sheets for Google Sheets
   */
  async getFileMetadata(fileId) {
    if (!this.drive) {
      throw new Error('Google Drive API not initialized');
    }

    try {
      const response = await this.drive.files.get({
        fileId: fileId,
        fields: 'id, name, mimeType, modifiedTime, size, properties, appProperties'
      });

      return response.data;
    } catch (error) {
      console.error('❌ Error getting file metadata:', error.message);
      throw error;
    }
  }

  /**
   * Read Google Sheets data directly
   */
  async readGoogleSheet(spreadsheetId) {
    const sheets = google.sheets({ version: 'v4', auth: this.auth });
    
    try {
      // Get all sheets in the spreadsheet
      const metadata = await sheets.spreadsheets.get({
        spreadsheetId: spreadsheetId
      });

      const sheetNames = metadata.data.sheets.map(sheet => sheet.properties.title);
      console.log(`📊 Found sheets: ${sheetNames.join(', ')}`);

      // Read data from all sheets
      const allData = {};
      
      for (const sheetName of sheetNames) {
        const response = await sheets.spreadsheets.values.get({
          spreadsheetId: spreadsheetId,
          range: `${sheetName}!A:Z` // Read all columns
        });

        allData[sheetName] = response.data.values || [];
        console.log(`✅ Read ${allData[sheetName].length} rows from sheet: ${sheetName}`);
      }

      return allData;
    } catch (error) {
      console.error('❌ Error reading Google Sheet:', error.message);
      throw error;
    }
  }

  /**
   * Extract award data from various file types
   */
  async extractAwardData(file) {
    const { id, name, mimeType } = file;
    
    console.log(`📄 Processing: ${name} (${mimeType})`);

    try {
      switch (mimeType) {
        case 'application/vnd.google-apps.spreadsheet':
          // Google Sheets
          return await this.readGoogleSheet(id);
          
        case 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        case 'application/vnd.ms-excel':
          // Excel files - download first
          const excelPath = await this.downloadFile(id, name);
          return await this.parseExcelFile(excelPath);
          
        case 'text/csv':
          // CSV files
          const csvPath = await this.downloadFile(id, name);
          return await this.parseCSVFile(csvPath);
          
        case 'application/pdf':
          // PDF files - might contain award certificates
          console.log(`⚠️ PDF file detected: ${name} - manual processing may be required`);
          return { type: 'pdf', name, id, requiresManualProcessing: true };
          
        default:
          console.log(`⚠️ Unsupported file type: ${mimeType} for file: ${name}`);
          return { type: 'unsupported', name, id, mimeType };
      }
    } catch (error) {
      console.error(`❌ Error processing ${name}:`, error.message);
      return { type: 'error', name, id, error: error.message };
    }
  }

  /**
   * Helper functions
   */
  async fileExists(filePath) {
    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }

  async ensureDirectory(dirPath) {
    try {
      await fs.mkdir(dirPath, { recursive: true });
    } catch (error) {
      console.error('Error creating directory:', error);
    }
  }

  async parseExcelFile(filePath) {
    // Would use a library like 'xlsx' to parse Excel files
    // For now, return placeholder
    return { type: 'excel', path: filePath, parsed: false };
  }

  async parseCSVFile(filePath) {
    // Would use a library like 'csv-parse' to parse CSV files
    // For now, return placeholder
    return { type: 'csv', path: filePath, parsed: false };
  }
}

/**
 * Create service account setup instructions
 */
async function createServiceAccountInstructions() {
  const instructions = `# Google Drive API Service Account Setup

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
`;

  await fs.writeFile(
    path.join(__dirname, 'GOOGLE_DRIVE_SETUP.md'),
    instructions
  );
  
  console.log('📝 Created Google Drive setup instructions');
}

// Export for use in other modules
module.exports = {
  GoogleDriveIntegration,
  createServiceAccountInstructions
};

// Test function
async function testGoogleDriveConnection() {
  const drive = new GoogleDriveIntegration();
  
  try {
    await drive.initialize();
    const files = await drive.listAwardsFiles();
    
    console.log('\n📊 Awards Database Files:');
    files.forEach((file, index) => {
      console.log(`${index + 1}. ${file.name} (${file.mimeType})`);
    });
    
    return files;
  } catch (error) {
    console.error('Test failed:', error);
    
    // Create setup instructions if auth fails
    if (error.message.includes('OAuth2') || error.message.includes('service account')) {
      await createServiceAccountInstructions();
    }
  }
}

// Run test if called directly
if (require.main === module) {
  testGoogleDriveConnection();
}