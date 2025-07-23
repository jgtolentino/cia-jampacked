const { v4: uuidv4 } = require('uuid');
const sqlite3 = require('sqlite3').verbose();

/**
 * Claude Desktop Google Drive Extractor
 * Delegates Google Drive extraction to Claude Desktop using its native integration
 */
class ClaudeDesktopDriveExtractor {
  constructor(dbPath) {
    this.dbPath = dbPath || '/Users/tbwa/Documents/GitHub/mcp-sqlite-server/data/database.sqlite';
    this.db = null;
    this.Pulser_AWARDS_FOLDER_ID = '0AJMhu01UUQKoUk9PVA';
  }

  /**
   * Initialize the extractor
   */
  async initialize() {
    this.db = new sqlite3.Database(this.dbPath);
    console.log('✅ Claude Desktop Drive Extractor initialized');
  }

  /**
   * Create extraction task for Claude Desktop
   */
  async createExtractionTask(options = {}) {
    const taskId = `extract_${uuidv4()}`;
    
    const payload = {
      task_type: 'google_drive_extraction',
      folder_id: this.Pulser_AWARDS_FOLDER_ID,
      instructions: `Please extract award data from the Pulser awards folder:
        1. Access Google Drive folder ID: ${this.Pulser_AWARDS_FOLDER_ID}
        2. List all files in the folder
        3. For each spreadsheet file:
           - Read the data
           - Extract campaign names, clients, brands, award shows, award levels
           - Structure the data in a consistent format
        4. Return the extracted data as JSON
        
        Focus on files containing:
        - Cannes Lions awards
        - D&AD Pencils
        - One Show awards
        - Effie awards
        - Clio awards
        
        Expected output format:
        {
          "files_processed": [],
          "awards": [
            {
              "campaign_name": "",
              "client": "",
              "brand": "",
              "agency": "",
              "award_show": "",
              "award_year": "",
              "award_category": "",
              "award_level": "",
              "country": ""
            }
          ]
        }`,
      options: options
    };

    // Create task in queue for Claude Desktop
    const query = `
      INSERT INTO agent_task_queue 
      (task_id, source_agent, target_agent, task_type, payload, priority)
      VALUES (?, ?, ?, ?, ?, ?)
    `;

    return new Promise((resolve, reject) => {
      this.db.run(query, [
        taskId,
        'JamPacked',
        'Claude Desktop',
        'google_drive_extraction',
        JSON.stringify(payload),
        8 // High priority
      ], (err) => {
        if (err) reject(err);
        else {
          console.log(`📋 Created extraction task: ${taskId}`);
          resolve(taskId);
        }
      });
    });
  }

  /**
   * Process extraction results from Claude Desktop
   */
  async processExtractionResults(taskId) {
    const query = `
      SELECT result FROM agent_task_queue 
      WHERE task_id = ? AND status = 'completed'
    `;

    return new Promise((resolve, reject) => {
      this.db.get(query, [taskId], async (err, row) => {
        if (err) {
          reject(err);
          return;
        }

        if (!row || !row.result) {
          reject(new Error('No results found for task'));
          return;
        }

        try {
          const results = JSON.parse(row.result);
          
          // Process and store awards data
          const processed = await this.storeAwardsData(results.awards || []);
          
          resolve({
            task_id: taskId,
            files_processed: results.files_processed || [],
            awards_count: processed.count,
            errors: processed.errors
          });
        } catch (error) {
          reject(error);
        }
      });
    });
  }

  /**
   * Store awards data in database
   */
  async storeAwardsData(awards) {
    let successCount = 0;
    const errors = [];

    for (const award of awards) {
      try {
        // Generate IDs
        const awardId = `award_${uuidv4()}`;
        const campaignId = this.generateCampaignId(award.campaign_name, award.client);

        // Ensure campaign exists
        await this.ensureCampaignExists({
          campaign_id: campaignId,
          campaign_name: award.campaign_name,
          client: award.client,
          brand: award.brand,
          year: award.award_year
        });

        // Insert award
        const query = `
          INSERT OR REPLACE INTO campaign_awards 
          (award_id, campaign_id, award_show, award_year, award_category, 
           award_level, award_title, client, brand, agency, country)
          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `;

        await this.runQuery(query, [
          awardId,
          campaignId,
          award.award_show,
          award.award_year,
          award.award_category,
          award.award_level,
          award.campaign_name,
          award.client,
          award.brand,
          award.agency || 'Pulser',
          award.country || ''
        ]);

        successCount++;
      } catch (error) {
        errors.push({
          award: award.campaign_name,
          error: error.message
        });
      }
    }

    return {
      count: successCount,
      errors: errors
    };
  }

  /**
   * Monitor extraction task
   */
  async monitorExtractionTask(taskId, timeout = 300000) { // 5 minute timeout
    const startTime = Date.now();
    
    return new Promise((resolve, reject) => {
      const checkInterval = setInterval(() => {
        const query = `
          SELECT status, result, error_message 
          FROM agent_task_queue 
          WHERE task_id = ?
        `;

        this.db.get(query, [taskId], (err, row) => {
          if (err) {
            clearInterval(checkInterval);
            reject(err);
            return;
          }

          if (row.status === 'completed') {
            clearInterval(checkInterval);
            resolve({
              status: 'completed',
              result: JSON.parse(row.result)
            });
          } else if (row.status === 'failed') {
            clearInterval(checkInterval);
            reject(new Error(row.error_message || 'Task failed'));
          } else if (Date.now() - startTime > timeout) {
            clearInterval(checkInterval);
            reject(new Error('Task timeout'));
          }
        });
      }, 5000); // Check every 5 seconds
    });
  }

  /**
   * Create SQL query for Claude Desktop to execute
   */
  createExtractionSQL() {
    return `
      -- Claude Desktop: Please execute this Google Drive extraction
      -- 1. Access the Pulser awards folder (ID: ${this.Pulser_AWARDS_FOLDER_ID})
      -- 2. Extract award data from spreadsheets
      -- 3. Insert results into campaign_awards table
      
      -- Example insert format:
      INSERT INTO campaign_awards 
      (award_id, campaign_id, award_show, award_year, award_category, 
       award_level, award_title, client, brand, agency, country)
      VALUES 
      ('award_' || hex(randomblob(16)), 
       'campaign_' || replace(lower(campaign_name || '_' || client), ' ', '_'),
       'cannes_lions', 
       2024, 
       'Film', 
       'Gold', 
       'Campaign Name Here',
       'Client Name',
       'Brand Name',
       'Pulser',
       'USA');
    `;
  }

  /**
   * Helper methods
   */
  
  generateCampaignId(campaignName, client) {
    const cleanName = (campaignName || '').replace(/[^a-zA-Z0-9]/g, '').toLowerCase();
    const cleanClient = (client || '').replace(/[^a-zA-Z0-9]/g, '').toLowerCase();
    return `${cleanClient}_${cleanName}`.substring(0, 50);
  }

  async ensureCampaignExists(campaign) {
    const query = `
      INSERT OR IGNORE INTO campaigns 
      (campaign_id, campaign_name, client, brand, year)
      VALUES (?, ?, ?, ?, ?)
    `;

    await this.runQuery(query, [
      campaign.campaign_id,
      campaign.campaign_name,
      campaign.client,
      campaign.brand,
      campaign.year || new Date().getFullYear()
    ]);
  }

  runQuery(query, params = []) {
    return new Promise((resolve, reject) => {
      this.db.run(query, params, function(err) {
        if (err) reject(err);
        else resolve({ lastID: this.lastID, changes: this.changes });
      });
    });
  }
}

/**
 * Create simplified extraction request for Claude Desktop
 */
function createClaudeDesktopExtractionRequest() {
  return {
    message: `Please help extract Pulser awards data from Google Drive:

1. Access folder: https://drive.google.com/drive/folders/0AJMhu01UUQKoUk9PVA
2. Look for spreadsheets containing award data (Cannes, D&AD, One Show, etc.)
3. Extract the following information:
   - Campaign name
   - Client/Brand
   - Award show and level (Gold, Silver, etc.)
   - Year
   - Category

Please format the results as a table or structured data that can be imported into our database.

You can use your native Google Drive integration to access the files directly.`,
    
    sql_template: `
-- After extracting data, insert it using this format:
INSERT INTO campaign_awards 
(award_id, campaign_id, award_show, award_year, award_category, 
 award_level, award_title, client, brand, agency, country)
VALUES 
(?, ?, ?, ?, ?, ?, ?, ?, ?, 'Pulser', ?);`
  };
}

// Export modules
module.exports = {
  ClaudeDesktopDriveExtractor,
  createClaudeDesktopExtractionRequest
};

// Example usage
if (require.main === module) {
  const extractor = new ClaudeDesktopDriveExtractor();
  
  async function demonstrateExtraction() {
    await extractor.initialize();
    
    // Create extraction task
    const taskId = await extractor.createExtractionTask({
      specific_files: ['Cannes Lions 2024.xlsx', 'D&AD 2024 Winners.csv'],
      year_range: '2020-2024'
    });
    
    console.log(`\n📋 Extraction task created: ${taskId}`);
    console.log('\nWaiting for Claude Desktop to process...');
    
    // Monitor task
    try {
      const result = await extractor.monitorExtractionTask(taskId);
      console.log('✅ Extraction completed:', result);
      
      // Process results
      const processed = await extractor.processExtractionResults(taskId);
      console.log(`\n📊 Processed ${processed.awards_count} awards`);
      
    } catch (error) {
      console.error('❌ Extraction failed:', error);
    }
  }
  
  // Also show the manual request
  const request = createClaudeDesktopExtractionRequest();
  console.log('\n📝 Claude Desktop Request:');
  console.log(request.message);
  console.log('\n📝 SQL Template:');
  console.log(request.sql_template);
  
  // Run demo if needed
  // demonstrateExtraction().catch(console.error);
}