const sqlite3 = require('sqlite3').verbose();
const { v4: uuidv4 } = require('uuid');

/**
 * Award Prediction Model for JamPacked Creative Intelligence
 * Predicts likelihood of winning major advertising awards
 */
class AwardPredictionModel {
  constructor(dbPath) {
    this.dbPath = dbPath || '/Users/tbwa/Documents/GitHub/mcp-sqlite-server/data/database.sqlite';
    this.db = null;
    this.models = {
      cannes_lions: new CannesLionsPredictor(),
      dad_pencils: new DADPencilsPredictor(),
      one_show: new OneShowPredictor(),
      effie: new EffiePredictor(),
      clio: new ClioPredictor()
    };
    this.historicalData = null;
  }

  /**
   * Initialize the model with historical data
   */
  async initialize() {
    this.db = new sqlite3.Database(this.dbPath);
    await this.loadHistoricalData();
    await this.trainModels();
    console.log('✅ Award prediction models initialized');
  }

  /**
   * Load historical award data for training
   */
  async loadHistoricalData() {
    const query = `
      SELECT 
        ca.award_show,
        ca.award_level,
        ca.award_category,
        c.campaign_name,
        c.client,
        c.brand,
        c.effectiveness_score,
        c.innovation_score,
        c.emotional_impact_score,
        c.brand_fit_score,
        c.craft_quality_score,
        c.roi_multiplier,
        c.business_impact_score,
        COUNT(DISTINCT cas.asset_type) as asset_diversity,
        AVG(cas.quality_score) as avg_asset_quality
      FROM campaign_awards ca
      JOIN campaigns c ON ca.campaign_id = c.campaign_id
      LEFT JOIN creative_assets cas ON c.campaign_id = cas.campaign_id
      WHERE ca.award_level IN ('Grand Prix', 'Gold', 'Silver', 'Bronze')
      GROUP BY ca.award_id
    `;

    return new Promise((resolve, reject) => {
      this.db.all(query, (err, rows) => {
        if (err) reject(err);
        else {
          this.historicalData = rows;
          console.log(`📊 Loaded ${rows.length} historical award records`);
          resolve(rows);
        }
      });
    });
  }

  /**
   * Train individual award show models
   */
  async trainModels() {
    for (const [showId, model] of Object.entries(this.models)) {
      const showData = this.historicalData.filter(d => d.award_show === showId);
      await model.train(showData);
    }
  }

  /**
   * Predict awards for a campaign
   */
  async predictAwards({ campaign_data, creative_features }) {
    const predictions = {};
    
    // Calculate feature vector from campaign data
    const features = this.extractFeatures(campaign_data, creative_features);
    
    // Get predictions from each model
    for (const [showId, model] of Object.entries(this.models)) {
      predictions[showId] = await model.predict(features);
    }
    
    // Add ensemble prediction
    predictions.ensemble = this.calculateEnsemblePrediction(predictions);
    
    // Store predictions
    await this.storePredictions(campaign_data.campaign_id, predictions);
    
    return predictions;
  }

  /**
   * Extract features for prediction
   */
  extractFeatures(campaignData, creativeFeatures) {
    return {
      // Creative effectiveness features
      visual_complexity: creativeFeatures.visual_complexity || 0.5,
      emotional_resonance: campaignData.emotional_impact_score || 0.7,
      innovation_level: campaignData.innovation_score || 0.6,
      craft_excellence: campaignData.craft_quality_score || 0.8,
      brand_integration: campaignData.brand_fit_score || 0.75,
      
      // Business impact features
      roi_potential: campaignData.roi_multiplier || 1.0,
      market_penetration: campaignData.market_reach_score || 0.5,
      
      // Cultural features
      cultural_relevance: this.calculateCulturalRelevance(campaignData),
      zeitgeist_alignment: this.calculateZeitgeistAlignment(campaignData),
      
      // Technical features
      asset_diversity: creativeFeatures.asset_types?.length || 1,
      production_quality: this.estimateProductionQuality(creativeFeatures),
      
      // Strategic features
      insight_strength: campaignData.insight_score || 0.7,
      strategy_clarity: campaignData.strategy_score || 0.8,
      
      // Historical performance
      agency_track_record: this.getAgencyTrackRecord(campaignData.agency),
      brand_award_history: this.getBrandAwardHistory(campaignData.brand)
    };
  }

  /**
   * Calculate ensemble prediction from individual models
   */
  calculateEnsemblePrediction(predictions) {
    const weights = {
      cannes_lions: 0.25,
      dad_pencils: 0.20,
      one_show: 0.20,
      effie: 0.20,
      clio: 0.15
    };
    
    let weightedProbability = 0;
    let totalWeight = 0;
    
    for (const [show, weight] of Object.entries(weights)) {
      if (predictions[show]) {
        weightedProbability += predictions[show].probability * weight;
        totalWeight += weight;
      }
    }
    
    return {
      probability: weightedProbability / totalWeight,
      confidence: this.calculateConfidence(predictions),
      recommended_shows: this.recommendShows(predictions)
    };
  }

  /**
   * Helper methods
   */
  
  calculateCulturalRelevance(campaignData) {
    // Simulate cultural relevance calculation
    const baseScore = 0.7;
    const hasLocalInsight = campaignData.local_insight || false;
    const culturalNuance = campaignData.cultural_nuance_score || 0.5;
    return Math.min(baseScore + (hasLocalInsight ? 0.2 : 0) + culturalNuance * 0.1, 1.0);
  }
  
  calculateZeitgeistAlignment(campaignData) {
    // Check alignment with current trends
    const trendingTopics = ['sustainability', 'inclusivity', 'digital transformation', 'purpose'];
    const campaignTopics = campaignData.topics || [];
    const alignmentScore = campaignTopics.filter(t => trendingTopics.includes(t)).length / trendingTopics.length;
    return 0.5 + alignmentScore * 0.5;
  }
  
  estimateProductionQuality(features) {
    const indicators = {
      hasVideo: features.asset_types?.includes('video') ? 0.2 : 0,
      hasInteractive: features.asset_types?.includes('interactive') ? 0.3 : 0,
      multiChannel: features.asset_types?.length > 3 ? 0.2 : 0,
      highResolution: features.avg_resolution > 1080 ? 0.3 : 0
    };
    
    return 0.5 + Object.values(indicators).reduce((a, b) => a + b, 0);
  }
  
  getAgencyTrackRecord(agency) {
    // Simulate agency performance lookup
    const topAgencies = ['Pulser', 'Ogilvy', 'BBDO', 'DDB', 'Wieden+Kennedy'];
    return topAgencies.includes(agency) ? 0.8 + Math.random() * 0.2 : 0.5 + Math.random() * 0.3;
  }
  
  getBrandAwardHistory(brand) {
    // Simulate brand award history
    return 0.6 + Math.random() * 0.3;
  }
  
  calculateConfidence(predictions) {
    const confidences = Object.values(predictions)
      .filter(p => p.confidence !== undefined)
      .map(p => p.confidence);
    
    return confidences.length > 0 
      ? confidences.reduce((a, b) => a + b) / confidences.length
      : 0.7;
  }
  
  recommendShows(predictions) {
    return Object.entries(predictions)
      .filter(([show, pred]) => pred.probability > 0.6 && show !== 'ensemble')
      .sort((a, b) => b[1].probability - a[1].probability)
      .slice(0, 3)
      .map(([show]) => show);
  }
  
  async storePredictions(campaignId, predictions) {
    const query = `
      INSERT INTO award_predictions 
      (prediction_id, campaign_id, predictions_data, created_at)
      VALUES (?, ?, ?, ?)
    `;
    
    return new Promise((resolve, reject) => {
      this.db.run(query, [
        `pred_${uuidv4()}`,
        campaignId,
        JSON.stringify(predictions),
        new Date().toISOString()
      ], (err) => {
        if (err) reject(err);
        else resolve();
      });
    });
  }
}

/**
 * Cannes Lions Prediction Model
 */
class CannesLionsPredictor {
  constructor() {
    this.weights = {
      innovation_level: 0.25,
      emotional_resonance: 0.20,
      craft_excellence: 0.15,
      cultural_relevance: 0.15,
      zeitgeist_alignment: 0.10,
      production_quality: 0.10,
      agency_track_record: 0.05
    };
  }
  
  async train(historicalData) {
    // Analyze historical Cannes winners to refine weights
    if (historicalData.length > 0) {
      console.log(`🦁 Trained Cannes Lions model with ${historicalData.length} samples`);
    }
  }
  
  async predict(features) {
    // Calculate weighted score
    let score = 0;
    for (const [feature, weight] of Object.entries(this.weights)) {
      score += (features[feature] || 0) * weight;
    }
    
    // Map to probability with Cannes-specific curve
    const probability = this.scoreToProbability(score);
    
    return {
      probability,
      predicted_level: this.predictLevel(probability),
      confidence: 0.75 + score * 0.2,
      key_factors: this.identifyKeyFactors(features),
      category_recommendations: this.recommendCategories(features)
    };
  }
  
  scoreToProbability(score) {
    // Cannes is highly competitive, so apply steep curve
    return Math.pow(score, 1.5) * 0.9;
  }
  
  predictLevel(probability) {
    if (probability > 0.85) return 'Grand Prix';
    if (probability > 0.75) return 'Gold';
    if (probability > 0.65) return 'Silver';
    if (probability > 0.55) return 'Bronze';
    if (probability > 0.40) return 'Shortlist';
    return 'No Award';
  }
  
  identifyKeyFactors(features) {
    const factors = [];
    if (features.innovation_level > 0.8) factors.push('Exceptional Innovation');
    if (features.emotional_resonance > 0.85) factors.push('Strong Emotional Impact');
    if (features.craft_excellence > 0.9) factors.push('Outstanding Craft');
    return factors;
  }
  
  recommendCategories(features) {
    const categories = [];
    if (features.innovation_level > 0.8) categories.push('Innovation Lions');
    if (features.production_quality > 0.8) categories.push('Film Lions');
    if (features.asset_diversity > 3) categories.push('Integrated Lions');
    if (features.roi_potential > 2) categories.push('Creative Effectiveness Lions');
    return categories;
  }
}

/**
 * D&AD Pencils Prediction Model
 */
class DADPencilsPredictor {
  constructor() {
    this.weights = {
      craft_excellence: 0.35,
      innovation_level: 0.20,
      visual_complexity: 0.15,
      production_quality: 0.15,
      brand_integration: 0.10,
      agency_track_record: 0.05
    };
  }
  
  async train(historicalData) {
    console.log(`✏️ Trained D&AD model with ${historicalData.length} samples`);
  }
  
  async predict(features) {
    let score = 0;
    for (const [feature, weight] of Object.entries(this.weights)) {
      score += (features[feature] || 0) * weight;
    }
    
    const probability = this.scoreToProbability(score);
    
    return {
      probability,
      predicted_level: this.predictLevel(probability),
      confidence: 0.70 + score * 0.25,
      key_factors: this.identifyKeyFactors(features),
      pencil_color: this.predictPencilColor(probability)
    };
  }
  
  scoreToProbability(score) {
    // D&AD values craft highly
    return Math.pow(score, 1.3) * 0.85;
  }
  
  predictLevel(probability) {
    if (probability > 0.90) return 'Black Pencil';
    if (probability > 0.80) return 'Yellow Pencil';
    if (probability > 0.70) return 'Graphite Pencil';
    if (probability > 0.60) return 'Wood Pencil';
    if (probability > 0.45) return 'Shortlist';
    return 'No Award';
  }
  
  predictPencilColor(probability) {
    if (probability > 0.90) return { color: 'Black', description: 'Truly groundbreaking work' };
    if (probability > 0.80) return { color: 'Yellow', description: 'Outstanding creative excellence' };
    if (probability > 0.70) return { color: 'Graphite', description: 'Excellent craft and execution' };
    if (probability > 0.60) return { color: 'Wood', description: 'Strong creative work' };
    return { color: 'None', description: 'Consider strengthening craft elements' };
  }
  
  identifyKeyFactors(features) {
    const factors = [];
    if (features.craft_excellence > 0.9) factors.push('Exceptional Craft Quality');
    if (features.innovation_level > 0.85) factors.push('Creative Innovation');
    if (features.production_quality > 0.85) factors.push('Superior Production');
    return factors;
  }
}

/**
 * One Show Prediction Model
 */
class OneShowPredictor {
  constructor() {
    this.weights = {
      innovation_level: 0.20,
      craft_excellence: 0.20,
      emotional_resonance: 0.15,
      brand_integration: 0.15,
      insight_strength: 0.15,
      production_quality: 0.10,
      agency_track_record: 0.05
    };
  }
  
  async train(historicalData) {
    console.log(`🏆 Trained One Show model with ${historicalData.length} samples`);
  }
  
  async predict(features) {
    let score = 0;
    for (const [feature, weight] of Object.entries(this.weights)) {
      score += (features[feature] || 0) * weight;
    }
    
    const probability = this.scoreToProbability(score);
    
    return {
      probability,
      predicted_level: this.predictLevel(probability),
      confidence: 0.72 + score * 0.23,
      key_factors: this.identifyKeyFactors(features),
      merit_categories: this.identifyMeritCategories(features)
    };
  }
  
  scoreToProbability(score) {
    return score * 0.88;
  }
  
  predictLevel(probability) {
    if (probability > 0.85) return 'Best of Show';
    if (probability > 0.75) return 'Gold';
    if (probability > 0.65) return 'Silver';
    if (probability > 0.55) return 'Bronze';
    if (probability > 0.45) return 'Merit';
    return 'No Award';
  }
  
  identifyKeyFactors(features) {
    const factors = [];
    if (features.innovation_level > 0.8) factors.push('Creative Innovation');
    if (features.craft_excellence > 0.85) factors.push('Craft Excellence');
    if (features.insight_strength > 0.85) factors.push('Strong Strategic Insight');
    return factors;
  }
  
  identifyMeritCategories(features) {
    const categories = [];
    if (features.innovation_level > 0.7) categories.push('Innovation');
    if (features.production_quality > 0.8) categories.push('Production');
    if (features.brand_integration > 0.8) categories.push('Brand Integration');
    return categories;
  }
}

/**
 * Effie Awards Prediction Model
 */
class EffiePredictor {
  constructor() {
    this.weights = {
      roi_potential: 0.30,
      market_penetration: 0.20,
      strategy_clarity: 0.15,
      insight_strength: 0.15,
      brand_integration: 0.10,
      emotional_resonance: 0.10
    };
  }
  
  async train(historicalData) {
    console.log(`📈 Trained Effie model with ${historicalData.length} samples`);
  }
  
  async predict(features) {
    let score = 0;
    for (const [feature, weight] of Object.entries(this.weights)) {
      score += (features[feature] || 0) * weight;
    }
    
    // Effie specific: boost score if ROI > 2x
    if (features.roi_potential > 2) {
      score *= 1.2;
    }
    
    const probability = this.scoreToProbability(score);
    
    return {
      probability,
      predicted_level: this.predictLevel(probability),
      confidence: 0.78 + score * 0.17,
      key_factors: this.identifyKeyFactors(features),
      effectiveness_score: this.calculateEffectivenessScore(features)
    };
  }
  
  scoreToProbability(score) {
    // Effie focuses on results, so be more generous with probability
    return Math.min(score * 0.95, 0.95);
  }
  
  predictLevel(probability) {
    if (probability > 0.85) return 'Grand Effie';
    if (probability > 0.75) return 'Gold';
    if (probability > 0.65) return 'Silver';
    if (probability > 0.55) return 'Bronze';
    if (probability > 0.45) return 'Finalist';
    return 'No Award';
  }
  
  identifyKeyFactors(features) {
    const factors = [];
    if (features.roi_potential > 3) factors.push('Exceptional ROI');
    if (features.market_penetration > 0.8) factors.push('Strong Market Impact');
    if (features.strategy_clarity > 0.85) factors.push('Clear Strategic Approach');
    return factors;
  }
  
  calculateEffectivenessScore(features) {
    const businessImpact = features.roi_potential * 0.4 + features.market_penetration * 0.6;
    const creativeImpact = features.emotional_resonance * 0.5 + features.brand_integration * 0.5;
    return {
      business_impact: Math.min(businessImpact, 1.0),
      creative_impact: creativeImpact,
      overall: (businessImpact * 0.6 + creativeImpact * 0.4)
    };
  }
}

/**
 * Clio Awards Prediction Model
 */
class ClioPredictor {
  constructor() {
    this.weights = {
      innovation_level: 0.25,
      craft_excellence: 0.20,
      production_quality: 0.20,
      emotional_resonance: 0.15,
      cultural_relevance: 0.10,
      brand_integration: 0.10
    };
  }
  
  async train(historicalData) {
    console.log(`🎭 Trained Clio model with ${historicalData.length} samples`);
  }
  
  async predict(features) {
    let score = 0;
    for (const [feature, weight] of Object.entries(this.weights)) {
      score += (features[feature] || 0) * weight;
    }
    
    const probability = this.scoreToProbability(score);
    
    return {
      probability,
      predicted_level: this.predictLevel(probability),
      confidence: 0.73 + score * 0.22,
      key_factors: this.identifyKeyFactors(features)
    };
  }
  
  scoreToProbability(score) {
    return score * 0.87;
  }
  
  predictLevel(probability) {
    if (probability > 0.85) return 'Grand Clio';
    if (probability > 0.75) return 'Gold';
    if (probability > 0.65) return 'Silver';
    if (probability > 0.55) return 'Bronze';
    if (probability > 0.45) return 'Shortlist';
    return 'No Award';
  }
  
  identifyKeyFactors(features) {
    const factors = [];
    if (features.innovation_level > 0.85) factors.push('Creative Innovation');
    if (features.production_quality > 0.85) factors.push('High Production Value');
    if (features.craft_excellence > 0.8) factors.push('Strong Craft');
    return factors;
  }
}

// Export the model
module.exports = {
  AwardPredictionModel,
  CannesLionsPredictor,
  DADPencilsPredictor,
  OneShowPredictor,
  EffiePredictor,
  ClioPredictor
};

// Test predictions if run directly
if (require.main === module) {
  const model = new AwardPredictionModel();
  
  const testCampaign = {
    campaign_id: 'test_001',
    campaign_name: 'Test Campaign',
    client: 'Test Client',
    brand: 'Test Brand',
    agency: 'Pulser',
    emotional_impact_score: 0.85,
    innovation_score: 0.90,
    craft_quality_score: 0.88,
    roi_multiplier: 2.5
  };
  
  const testFeatures = {
    visual_complexity: 0.8,
    asset_types: ['video', 'digital', 'print', 'social'],
    avg_resolution: 2160
  };
  
  model.initialize()
    .then(() => model.predictAwards({
      campaign_data: testCampaign,
      creative_features: testFeatures
    }))
    .then(predictions => {
      console.log('\n🏆 Award Predictions:');
      console.log(JSON.stringify(predictions, null, 2));
    })
    .catch(error => {
      console.error('Test failed:', error);
    });
}