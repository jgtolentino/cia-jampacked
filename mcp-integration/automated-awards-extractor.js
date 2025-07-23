const { ClaudeDesktopDriveExtractor } = require('./claude-desktop-drive-extractor');
const sqlite3 = require('sqlite3').verbose();
const cron = require('node-cron');
const path = require('path');
const fs = require('fs').promises;

/**
 * Automated Awards Data Extraction from Google Drive
 * Processes Pulser awards database and updates SQLite
 */
class AwardsDataExtractor {
  constructor(dbPath) {
    this.dbPath = dbPath || '/Users/tbwa/Documents/GitHub/mcp-sqlite-server/data/database.sqlite';
    this.claudeExtractor = new ClaudeDesktopDriveExtractor(dbPath);
    this.db = null;
    this.processedFiles = new Set();
  }

  /**
   * Initialize database connection and Claude Desktop extractor
   */
  async initialize() {
    // Initialize database
    this.db = new sqlite3.Database(this.dbPath);
    
    // Create necessary tables if they don't exist
    await this.createTables();
    
    // Load processed files history
    await this.loadProcessedFiles();
    
    // Initialize Claude Desktop extractor
    await this.claudeExtractor.initialize();
    
    console.log('✅ Awards extractor initialized with Claude Desktop integration');
  }

  /**
   * Create database tables for awards data
   */
  async createTables() {
    const queries = [
      // Awards metadata table
      `CREATE TABLE IF NOT EXISTS award_metadata (
        file_id TEXT PRIMARY KEY,
        file_name TEXT NOT NULL,
        file_type TEXT,
        last_processed DATETIME,
        processing_status TEXT,
        error_message TEXT
      )`,
      
      // Campaign awards table (if not exists)
      `CREATE TABLE IF NOT EXISTS campaign_awards (
        award_id TEXT PRIMARY KEY,
        campaign_id TEXT NOT NULL,
        award_show TEXT NOT NULL,
        award_year INTEGER,
        award_category TEXT,
        award_level TEXT,
        award_title TEXT,
        client TEXT,
        brand TEXT,
        agency TEXT,
        country TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id)
      )`,
      
      // Award shows reference table
      `CREATE TABLE IF NOT EXISTS award_shows (
        show_id TEXT PRIMARY KEY,
        show_name TEXT NOT NULL,
        show_type TEXT,
        prestige_tier INTEGER,
        website TEXT
      )`,
      
      // Processing log
      `CREATE TABLE IF NOT EXISTS extraction_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        extraction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        files_processed INTEGER,
        records_extracted INTEGER,
        errors_count INTEGER,
        duration_seconds INTEGER
      )`
    ];

    for (const query of queries) {
      await this.runQuery(query);
    }

    // Insert award shows if not exists
    await this.populateAwardShows();
  }

  /**
   * Populate award shows reference data
   */
  async populateAwardShows() {
    const awardShows = [
      ['cannes_lions', 'Cannes Lions', 'Global', 1, 'https://www.canneslions.com'],
      ['dad_pencils', 'D&AD Pencils', 'Global', 1, 'https://www.dandad.org'],
      ['one_show', 'One Show', 'Global', 1, 'https://www.oneshow.org'],
      ['effie', 'Effie Awards', 'Effectiveness', 1, 'https://www.effie.org'],
      ['clio', 'Clio Awards', 'Global', 2, 'https://clios.com'],
      ['webby', 'Webby Awards', 'Digital', 2, 'https://www.webbyawards.com'],
      ['adfest', 'ADFEST', 'Regional', 2, 'https://www.adfest.com'],
      ['spikes_asia', 'Spikes Asia', 'Regional', 2, 'https://www.spikes.asia']
    ];

    for (const show of awardShows) {
      await this.runQuery(
        `INSERT OR IGNORE INTO award_shows (show_id, show_name, show_type, prestige_tier, website) 
         VALUES (?, ?, ?, ?, ?)`,
        show
      );
    }
  }

  /**
   * Main extraction process - delegates to Claude Desktop
   */
  async extractAwardsData() {
    const startTime = Date.now();

    try {
      console.log('🚀 Starting awards data extraction via Claude Desktop...');
      
      // Create extraction task for Claude Desktop
      const taskId = await this.claudeExtractor.createExtractionTask({
        instructions: 'Extract all award data from Pulser Google Drive folder',
        year_range: '2020-2024',
        focus_shows: ['Cannes Lions', 'D&AD', 'One Show', 'Effie', 'Clio']
      });
      
      console.log(`📋 Created extraction task: ${taskId}`);
      console.log('⏳ Waiting for Claude Desktop to process Google Drive files...');
      console.log('💡 Claude Desktop will use its native Google Drive integration');
      
      // Monitor task completion
      const result = await this.claudeExtractor.monitorExtractionTask(taskId, 600000); // 10 minute timeout
      
      // Process the results
      const processed = await this.claudeExtractor.processExtractionResults(taskId);
      
      // Log extraction summary
      const duration = Math.round((Date.now() - startTime) / 1000);
      await this.logExtraction(
        processed.files_processed.length,
        processed.awards_count,
        processed.errors.length,
        duration
      );

      console.log('\n📊 Extraction Summary:');
      console.log(`Files processed: ${processed.files_processed.length}`);
      console.log(`Records extracted: ${processed.awards_count}`);
      console.log(`Errors: ${processed.errors.length}`);
      console.log(`Duration: ${duration} seconds`);

      return {
        success: true,
        filesProcessed: processed.files_processed.length,
        recordsExtracted: processed.awards_count,
        errorsCount: processed.errors.length,
        duration
      };

    } catch (error) {
      console.error('❌ Extraction failed:', error);
      throw error;
    }
  }

  /**
   * Process individual award file
   */
  async processAwardFile(file) {
    console.log(`\n📄 Processing: ${file.name}`);
    
    // Extract data based on file type
    const data = await this.drive.extractAwardData(file);
    
    if (!data || data.requiresManualProcessing) {
      console.log(`⚠️ File requires manual processing: ${file.name}`);
      return { recordsCount: 0 };
    }

    // Process based on data type
    if (data.type === 'unsupported' || data.type === 'error') {
      return { recordsCount: 0 };
    }

    // Parse award records from data
    const awards = await this.parseAwardRecords(data, file.name);
    
    // Insert awards into database
    let recordsCount = 0;
    for (const award of awards) {
      try {
        await this.insertAwardRecord(award);
        recordsCount++;
      } catch (error) {
        console.error(`Error inserting award record:`, error.message);
      }
    }

    console.log(`✅ Extracted ${recordsCount} award records from ${file.name}`);
    return { recordsCount };
  }

  /**
   * Parse award records from extracted data
   */
  async parseAwardRecords(data, fileName) {
    const awards = [];
    
    // Detect award show from filename
    const awardShow = this.detectAwardShow(fileName);
    
    if (data && typeof data === 'object') {
      // Process Google Sheets data
      for (const [sheetName, rows] of Object.entries(data)) {
        if (!Array.isArray(rows) || rows.length < 2) continue;
        
        const headers = rows[0];
        const awardRecords = this.parseSheetData(rows, headers, awardShow);
        awards.push(...awardRecords);
      }
    }

    return awards;
  }

  /**
   * Parse sheet data into award records
   */
  parseSheetData(rows, headers, defaultAwardShow) {
    const awards = [];
    
    // Map common column names
    const columnMap = {
      'campaign': 'campaign_name',
      'campaign name': 'campaign_name',
      'title': 'campaign_name',
      'client': 'client',
      'advertiser': 'client',
      'brand': 'brand',
      'product': 'brand',
      'agency': 'agency',
      'agency name': 'agency',
      'award': 'award_level',
      'award level': 'award_level',
      'metal': 'award_level',
      'category': 'award_category',
      'year': 'award_year',
      'country': 'country',
      'region': 'country'
    };

    // Find column indices
    const indices = {};
    headers.forEach((header, index) => {
      const normalizedHeader = header.toLowerCase().trim();
      for (const [key, value] of Object.entries(columnMap)) {
        if (normalizedHeader.includes(key)) {
          indices[value] = index;
        }
      }
    });

    // Process data rows
    for (let i = 1; i < rows.length; i++) {
      const row = rows[i];
      if (!row || row.length === 0) continue;

      const award = {
        award_id: `award_${Date.now()}_${i}`,
        campaign_id: this.generateCampaignId(row[indices.campaign_name] || '', row[indices.client] || ''),
        award_show: defaultAwardShow,
        award_year: this.extractYear(row[indices.award_year]),
        award_category: row[indices.award_category] || '',
        award_level: this.normalizeAwardLevel(row[indices.award_level] || ''),
        award_title: row[indices.campaign_name] || '',
        client: row[indices.client] || '',
        brand: row[indices.brand] || '',
        agency: row[indices.agency] || 'Pulser',
        country: row[indices.country] || ''
      };

      // Only add if we have minimum required data
      if (award.campaign_id && award.award_show) {
        awards.push(award);
      }
    }

    return awards;
  }

  /**
   * Insert award record into database
   */
  async insertAwardRecord(award) {
    // First, ensure campaign exists
    await this.ensureCampaignExists(award);

    // Insert award
    const query = `
      INSERT OR REPLACE INTO campaign_awards 
      (award_id, campaign_id, award_show, award_year, award_category, 
       award_level, award_title, client, brand, agency, country)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;

    await this.runQuery(query, [
      award.award_id,
      award.campaign_id,
      award.award_show,
      award.award_year,
      award.award_category,
      award.award_level,
      award.award_title,
      award.client,
      award.brand,
      award.agency,
      award.country
    ]);
  }

  /**
   * Ensure campaign exists in campaigns table
   */
  async ensureCampaignExists(award) {
    const query = `
      INSERT OR IGNORE INTO campaigns 
      (campaign_id, campaign_name, client, brand, year)
      VALUES (?, ?, ?, ?, ?)
    `;

    await this.runQuery(query, [
      award.campaign_id,
      award.award_title,
      award.client,
      award.brand,
      award.award_year || new Date().getFullYear()
    ]);
  }

  /**
   * Helper functions
   */
  
  detectAwardShow(fileName) {
    const fileNameLower = fileName.toLowerCase();
    
    if (fileNameLower.includes('cannes')) return 'cannes_lions';
    if (fileNameLower.includes('d&ad') || fileNameLower.includes('dad')) return 'dad_pencils';
    if (fileNameLower.includes('one show')) return 'one_show';
    if (fileNameLower.includes('effie')) return 'effie';
    if (fileNameLower.includes('clio')) return 'clio';
    if (fileNameLower.includes('webby')) return 'webby';
    if (fileNameLower.includes('adfest')) return 'adfest';
    if (fileNameLower.includes('spikes')) return 'spikes_asia';
    
    return 'unknown';
  }

  normalizeAwardLevel(level) {
    const levelLower = level.toLowerCase();
    
    if (levelLower.includes('grand') || levelLower.includes('best')) return 'Grand Prix';
    if (levelLower.includes('gold')) return 'Gold';
    if (levelLower.includes('silver')) return 'Silver';
    if (levelLower.includes('bronze')) return 'Bronze';
    if (levelLower.includes('shortlist') || levelLower.includes('finalist')) return 'Shortlist';
    
    return level;
  }

  extractYear(yearValue) {
    if (!yearValue) return new Date().getFullYear();
    
    const yearStr = yearValue.toString();
    const yearMatch = yearStr.match(/20\d{2}/);
    
    return yearMatch ? parseInt(yearMatch[0]) : new Date().getFullYear();
  }

  generateCampaignId(campaignName, client) {
    const cleanName = (campaignName || '').replace(/[^a-zA-Z0-9]/g, '').toLowerCase();
    const cleanClient = (client || '').replace(/[^a-zA-Z0-9]/g, '').toLowerCase();
    return `${cleanClient}_${cleanName}`.substring(0, 50);
  }

  shouldReprocess(file) {
    // Reprocess if file was modified recently (within last 7 days)
    const modifiedDate = new Date(file.modifiedTime);
    const daysSinceModified = (Date.now() - modifiedDate) / (1000 * 60 * 60 * 24);
    return daysSinceModified < 7;
  }

  /**
   * Database helper functions
   */
  
  runQuery(query, params = []) {
    return new Promise((resolve, reject) => {
      this.db.run(query, params, function(err) {
        if (err) reject(err);
        else resolve({ lastID: this.lastID, changes: this.changes });
      });
    });
  }

  async loadProcessedFiles() {
    return new Promise((resolve, reject) => {
      this.db.all(
        'SELECT file_id FROM award_metadata WHERE processing_status = "success"',
        (err, rows) => {
          if (err) reject(err);
          else {
            rows.forEach(row => this.processedFiles.add(row.file_id));
            resolve();
          }
        }
      );
    });
  }

  async markFileProcessed(fileId, fileName, status, errorMessage = null) {
    const query = `
      INSERT OR REPLACE INTO award_metadata 
      (file_id, file_name, processing_status, last_processed, error_message)
      VALUES (?, ?, ?, datetime('now'), ?)
    `;
    
    await this.runQuery(query, [fileId, fileName, status, errorMessage]);
  }

  async logExtraction(filesProcessed, recordsExtracted, errorsCount, duration) {
    const query = `
      INSERT INTO extraction_log 
      (files_processed, records_extracted, errors_count, duration_seconds)
      VALUES (?, ?, ?, ?)
    `;
    
    await this.runQuery(query, [filesProcessed, recordsExtracted, errorsCount, duration]);
  }
}

/**
 * Schedule automated extraction
 */
function scheduleAutomatedExtraction(dbPath) {
  const extractor = new AwardsDataExtractor(dbPath);
  
  // Run every Sunday at 2 AM
  cron.schedule('0 2 * * 0', async () => {
    console.log('🕒 Starting scheduled awards extraction...');
    
    try {
      await extractor.initialize();
      await extractor.extractAwardsData();
      console.log('✅ Scheduled extraction completed');
    } catch (error) {
      console.error('❌ Scheduled extraction failed:', error);
    }
  });
  
  console.log('📅 Automated extraction scheduled for every Sunday at 2 AM');
}

// Export for use
module.exports = {
  AwardsDataExtractor,
  scheduleAutomatedExtraction
};

// Run extraction if called directly
if (require.main === module) {
  const extractor = new AwardsDataExtractor();
  
  extractor.initialize()
    .then(() => extractor.extractAwardsData())
    .then(result => {
      console.log('\n✅ Extraction completed:', result);
      process.exit(0);
    })
    .catch(error => {
      console.error('\n❌ Extraction failed:', error);
      process.exit(1);
    });
}