-- JamPacked Creative Intelligence Database Schema
-- WARC Effective 100 Enhanced Version
-- Supports comprehensive creative effectiveness measurement

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS creative;
CREATE SCHEMA IF NOT EXISTS campaigns;
CREATE SCHEMA IF NOT EXISTS effectiveness;
CREATE SCHEMA IF NOT EXISTS warc_standards;

-- =====================================================
-- CREATIVE ASSETS SCHEMA
-- =====================================================

-- Creative Assets Master Table
CREATE TABLE creative.assets (
    asset_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID NOT NULL,
    asset_type VARCHAR(100) NOT NULL CHECK (asset_type IN ('image', 'video', 'audio', 'copy', 'interactive', 'ar_vr', 'ai_generated')),
    asset_name VARCHAR(500) NOT NULL,
    asset_url VARCHAR(1000) NOT NULL,
    file_hash VARCHAR(256) UNIQUE NOT NULL,
    file_size_bytes BIGINT,
    duration_seconds DECIMAL(10,2),
    dimensions JSONB,
    
    -- Metadata
    asset_metadata JSONB NOT NULL DEFAULT '{}',
    platform_specs JSONB DEFAULT '{}',
    production_details JSONB DEFAULT '{}',
    
    -- Analysis Results
    visual_analysis JSONB DEFAULT '{}',
    copy_analysis JSONB DEFAULT '{}',
    audio_analysis JSONB DEFAULT '{}',
    talent_analysis JSONB DEFAULT '{}',
    
    -- Distinctive Assets
    brand_elements JSONB DEFAULT '{}',
    distinctive_assets JSONB DEFAULT '{}',
    memory_encoding_score DECIMAL(5,2),
    
    -- Effectiveness Scores
    attention_score DECIMAL(5,2),
    engagement_score DECIMAL(5,2),
    recall_score DECIMAL(5,2),
    persuasion_score DECIMAL(5,2),
    
    -- Embeddings for similarity search
    visual_embedding vector(1536),
    semantic_embedding vector(1536),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    analyzed_at TIMESTAMP WITH TIME ZONE,
    
    -- Indexes
    CONSTRAINT fk_campaign FOREIGN KEY (campaign_id) REFERENCES campaigns.campaigns(campaign_id)
);

CREATE INDEX idx_creative_assets_campaign ON creative.assets(campaign_id);
CREATE INDEX idx_creative_assets_type ON creative.assets(asset_type);
CREATE INDEX idx_creative_assets_visual_embedding ON creative.assets USING ivfflat (visual_embedding vector_cosine_ops);
CREATE INDEX idx_creative_assets_semantic_embedding ON creative.assets USING ivfflat (semantic_embedding vector_cosine_ops);

-- =====================================================
-- CAMPAIGNS SCHEMA
-- =====================================================

-- Campaigns Master Table
CREATE TABLE campaigns.campaigns (
    campaign_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID NOT NULL,
    campaign_name VARCHAR(500) NOT NULL,
    campaign_code VARCHAR(100) UNIQUE,
    
    -- Strategic Information
    campaign_objectives JSONB NOT NULL,
    target_audience JSONB NOT NULL,
    brand_positioning TEXT,
    competitive_context JSONB,
    
    -- Budget and Timing
    total_budget_usd DECIMAL(15,2),
    budget_allocation JSONB,
    media_channels JSONB NOT NULL,
    launch_date DATE NOT NULL,
    end_date DATE,
    
    -- Performance Tracking
    performance_metrics JSONB DEFAULT '{}',
    kpi_targets JSONB NOT NULL,
    success_criteria JSONB,
    
    -- Effectiveness Scores
    creative_effectiveness_score DECIMAL(5,2),
    warc_effectiveness_score DECIMAL(5,2),
    roi_multiple DECIMAL(10,2),
    
    -- Long-term Impact
    brand_equity_impact DECIMAL(10,2),
    mental_availability_lift DECIMAL(10,2),
    
    -- Attribution
    roi_attribution JSONB DEFAULT '{}',
    incrementality_results JSONB DEFAULT '{}',
    
    -- Purpose & Cultural Impact
    purpose_alignment_score DECIMAL(5,2),
    cultural_relevance_score DECIMAL(5,2),
    social_impact_metrics JSONB DEFAULT '{}',
    
    -- Status
    status VARCHAR(50) DEFAULT 'planning' CHECK (status IN ('planning', 'active', 'completed', 'archived')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_campaigns_client ON campaigns.campaigns(client_id);
CREATE INDEX idx_campaigns_status ON campaigns.campaigns(status);
CREATE INDEX idx_campaigns_dates ON campaigns.campaigns(launch_date, end_date);

-- =====================================================
-- EFFECTIVENESS MEASUREMENT SCHEMA
-- =====================================================

-- WARC-Standard Effectiveness Scorecard
CREATE TABLE effectiveness.scorecards (
    scorecard_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID NOT NULL REFERENCES campaigns.campaigns(campaign_id),
    evaluation_date DATE NOT NULL,
    
    -- WARC Five Dimensions
    strategic_planning_score DECIMAL(5,2) NOT NULL CHECK (strategic_planning_score BETWEEN 0 AND 100),
    creative_idea_score DECIMAL(5,2) NOT NULL CHECK (creative_idea_score BETWEEN 0 AND 100),
    business_results_score DECIMAL(5,2) NOT NULL CHECK (business_results_score BETWEEN 0 AND 100),
    brand_building_score DECIMAL(5,2) NOT NULL CHECK (brand_building_score BETWEEN 0 AND 100),
    cultural_impact_score DECIMAL(5,2) NOT NULL CHECK (cultural_impact_score BETWEEN 0 AND 100),
    
    -- Aggregate Scores
    overall_effectiveness_score DECIMAL(5,2) NOT NULL CHECK (overall_effectiveness_score BETWEEN 0 AND 100),
    warc_percentile_rank DECIMAL(5,2),
    
    -- Creative Effectiveness Components
    creative_innovation_score DECIMAL(5,2),
    storytelling_impact_score DECIMAL(5,2),
    distinctive_asset_usage_score DECIMAL(5,2),
    platform_optimization_score DECIMAL(5,2),
    
    -- Performance Predictions
    predicted_engagement_lift DECIMAL(10,2),
    predicted_brand_recall_lift DECIMAL(10,2),
    predicted_roi_impact DECIMAL(10,2),
    prediction_confidence_level DECIMAL(5,2),
    
    -- Statistical Confidence
    confidence_intervals JSONB NOT NULL,
    statistical_significance JSONB,
    sample_size_adequacy JSONB,
    
    -- Detailed Analysis
    contributing_factors JSONB NOT NULL,
    improvement_recommendations JSONB NOT NULL,
    competitive_comparison JSONB,
    category_benchmarks JSONB,
    
    -- Award Potential
    award_prediction_score DECIMAL(5,2),
    award_submission_readiness JSONB,
    
    -- Metadata
    evaluated_by VARCHAR(255),
    evaluation_methodology JSONB,
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_scorecards_campaign ON effectiveness.scorecards(campaign_id);
CREATE INDEX idx_scorecards_date ON effectiveness.scorecards(evaluation_date);
CREATE INDEX idx_scorecards_overall_score ON effectiveness.scorecards(overall_effectiveness_score DESC);

-- Long-term Brand Building Measurement
CREATE TABLE effectiveness.brand_building (
    measurement_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID NOT NULL REFERENCES campaigns.campaigns(campaign_id),
    measurement_date DATE NOT NULL,
    
    -- Mental Availability
    unaided_awareness DECIMAL(5,2),
    aided_awareness DECIMAL(5,2),
    consideration DECIMAL(5,2),
    preference DECIMAL(5,2),
    mental_market_share DECIMAL(5,2),
    
    -- Category Entry Points
    cep_coverage DECIMAL(5,2),
    cep_strength JSONB,
    category_associations JSONB,
    
    -- Distinctive Assets
    asset_recognition JSONB,
    asset_attribution JSONB,
    asset_uniqueness JSONB,
    asset_strength_index DECIMAL(5,2),
    
    -- Brand Equity Components
    brand_salience DECIMAL(5,2),
    brand_meaning JSONB,
    brand_response JSONB,
    brand_resonance DECIMAL(5,2),
    
    -- Memory Structure
    memory_encoding_effectiveness DECIMAL(5,2),
    retrieval_cue_strength JSONB,
    
    -- Long-term Tracking
    baseline_metrics JSONB,
    lift_vs_baseline JSONB,
    decay_curve_parameters JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_brand_building_campaign ON effectiveness.brand_building(campaign_id);
CREATE INDEX idx_brand_building_date ON effectiveness.brand_building(measurement_date);

-- =====================================================
-- ECONOMETRIC MODELING SCHEMA
-- =====================================================

-- Media Mix Modeling Results
CREATE TABLE effectiveness.media_mix_models (
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID NOT NULL REFERENCES campaigns.campaigns(campaign_id),
    model_period_start DATE NOT NULL,
    model_period_end DATE NOT NULL,
    
    -- Model Components
    base_sales DECIMAL(15,2),
    media_contribution JSONB NOT NULL,
    promotion_impact JSONB,
    seasonality_factors JSONB,
    competitive_effects JSONB,
    external_factors JSONB,
    
    -- Adstock & Saturation
    adstock_parameters JSONB NOT NULL,
    saturation_curves JSONB NOT NULL,
    carryover_effects JSONB,
    
    -- Channel Performance
    channel_roi JSONB NOT NULL,
    channel_efficiency JSONB,
    marginal_roi JSONB,
    optimal_budget_allocation JSONB,
    
    -- Cross-effects
    synergy_effects JSONB,
    cannibalization_effects JSONB,
    halo_effects JSONB,
    
    -- Model Quality
    model_fit_statistics JSONB NOT NULL,
    validation_metrics JSONB,
    prediction_accuracy DECIMAL(5,2),
    
    -- Metadata
    model_type VARCHAR(100),
    modeling_approach TEXT,
    assumptions JSONB,
    limitations JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255)
);

CREATE INDEX idx_mmm_campaign ON effectiveness.media_mix_models(campaign_id);
CREATE INDEX idx_mmm_period ON effectiveness.media_mix_models(model_period_start, model_period_end);

-- =====================================================
-- PURPOSE & CULTURAL IMPACT SCHEMA
-- =====================================================

-- Purpose Effectiveness Measurement
CREATE TABLE effectiveness.purpose_impact (
    impact_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID NOT NULL REFERENCES campaigns.campaigns(campaign_id),
    measurement_date DATE NOT NULL,
    
    -- Authenticity Measurement
    brand_values_alignment DECIMAL(5,2),
    action_credibility DECIMAL(5,2),
    consumer_belief_score DECIMAL(5,2),
    employee_alignment_score DECIMAL(5,2),
    
    -- Stakeholder Impact
    consumer_impact JSONB,
    employee_impact JSONB,
    community_impact JSONB,
    investor_impact JSONB,
    
    -- Social Metrics
    social_awareness_lift DECIMAL(10,2),
    attitude_shift_measurement JSONB,
    behavior_change_tracking JSONB,
    advocacy_creation_rate DECIMAL(10,2),
    
    -- Business Connection
    purpose_premium_impact DECIMAL(10,2),
    loyalty_enhancement DECIMAL(10,2),
    talent_attraction_boost DECIMAL(10,2),
    esg_score_improvement DECIMAL(10,2),
    
    -- Cultural Relevance
    cultural_zeitgeist_score DECIMAL(5,2),
    social_conversation_share DECIMAL(10,2),
    earned_media_value DECIMAL(15,2),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_purpose_campaign ON effectiveness.purpose_impact(campaign_id);
CREATE INDEX idx_purpose_date ON effectiveness.purpose_impact(measurement_date);

-- =====================================================
-- COMPETITIVE INTELLIGENCE SCHEMA
-- =====================================================

-- Competitive Campaign Tracking
CREATE TABLE effectiveness.competitive_campaigns (
    competitive_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    competitor_name VARCHAR(255) NOT NULL,
    campaign_name VARCHAR(500),
    category VARCHAR(255) NOT NULL,
    market VARCHAR(100) NOT NULL,
    
    -- Campaign Details
    estimated_budget DECIMAL(15,2),
    media_channels JSONB,
    creative_themes JSONB,
    
    -- Performance Estimates
    estimated_reach DECIMAL(15,0),
    estimated_engagement JSONB,
    share_of_voice DECIMAL(5,2),
    
    -- Effectiveness Analysis
    creative_quality_score DECIMAL(5,2),
    innovation_score DECIMAL(5,2),
    predicted_effectiveness DECIMAL(5,2),
    
    -- Awards & Recognition
    awards_won JSONB,
    industry_recognition JSONB,
    
    tracking_start_date DATE NOT NULL,
    tracking_end_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_competitive_category ON effectiveness.competitive_campaigns(category);
CREATE INDEX idx_competitive_market ON effectiveness.competitive_campaigns(market);

-- =====================================================
-- AWARD TRACKING SCHEMA
-- =====================================================

-- Award Submissions and Results
CREATE TABLE warc_standards.award_tracking (
    award_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID NOT NULL REFERENCES campaigns.campaigns(campaign_id),
    award_body VARCHAR(255) NOT NULL,
    award_year INTEGER NOT NULL,
    
    -- Submission Details
    category VARCHAR(500) NOT NULL,
    submission_date DATE,
    case_study_url VARCHAR(1000),
    
    -- Results
    award_status VARCHAR(50) CHECK (award_status IN ('submitted', 'shortlisted', 'bronze', 'silver', 'gold', 'grand_prix', 'not_awarded')),
    award_level VARCHAR(100),
    
    -- Scoring & Feedback
    judge_scores JSONB,
    judge_feedback TEXT,
    category_rank INTEGER,
    
    -- Predictive Analysis
    predicted_win_probability DECIMAL(5,2),
    prediction_factors JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_awards_campaign ON warc_standards.award_tracking(campaign_id);
CREATE INDEX idx_awards_body_year ON warc_standards.award_tracking(award_body, award_year);

-- =====================================================
-- GLOBAL STANDARDIZATION SCHEMA
-- =====================================================

-- Cross-Market Normalization
CREATE TABLE warc_standards.global_benchmarks (
    benchmark_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    market VARCHAR(100) NOT NULL,
    category VARCHAR(255) NOT NULL,
    benchmark_period VARCHAR(20) NOT NULL,
    
    -- Market Factors
    market_size_index DECIMAL(10,2),
    media_cost_index DECIMAL(10,2),
    competitive_intensity_index DECIMAL(10,2),
    digital_maturity_index DECIMAL(10,2),
    
    -- Performance Benchmarks
    awareness_benchmarks JSONB,
    engagement_benchmarks JSONB,
    conversion_benchmarks JSONB,
    roi_benchmarks JSONB,
    
    -- Cultural Factors
    cultural_dimensions JSONB,
    local_preferences JSONB,
    regulatory_constraints JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_benchmarks_market_category ON warc_standards.global_benchmarks(market, category);

-- =====================================================
-- AUDIT AND VERSIONING
-- =====================================================

-- Campaign Analysis Audit Trail
CREATE TABLE effectiveness.analysis_audit (
    audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID NOT NULL REFERENCES campaigns.campaigns(campaign_id),
    analysis_type VARCHAR(100) NOT NULL,
    analysis_version INTEGER NOT NULL,
    
    -- Analysis Details
    analysis_parameters JSONB NOT NULL,
    analysis_results JSONB NOT NULL,
    confidence_metrics JSONB,
    
    -- Metadata
    analyzed_by VARCHAR(255),
    analysis_tool VARCHAR(255),
    analysis_duration_seconds INTEGER,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(campaign_id, analysis_type, analysis_version)
);

CREATE INDEX idx_audit_campaign_type ON effectiveness.analysis_audit(campaign_id, analysis_type);

-- =====================================================
-- TRIGGERS AND FUNCTIONS
-- =====================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update trigger to relevant tables
CREATE TRIGGER update_campaigns_updated_at BEFORE UPDATE ON campaigns.campaigns
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_creative_assets_updated_at BEFORE UPDATE ON creative.assets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_award_tracking_updated_at BEFORE UPDATE ON warc_standards.award_tracking
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- MATERIALIZED VIEWS FOR PERFORMANCE
-- =====================================================

-- Campaign Performance Dashboard View
CREATE MATERIALIZED VIEW effectiveness.campaign_performance_summary AS
SELECT 
    c.campaign_id,
    c.campaign_name,
    c.client_id,
    c.launch_date,
    c.end_date,
    c.total_budget_usd,
    s.overall_effectiveness_score,
    s.warc_percentile_rank,
    s.predicted_roi_impact,
    COUNT(DISTINCT ca.asset_id) as total_assets,
    AVG(ca.attention_score) as avg_attention_score,
    MAX(bb.mental_market_share) as peak_mental_share,
    MAX(mmm.prediction_accuracy) as model_accuracy
FROM campaigns.campaigns c
LEFT JOIN effectiveness.scorecards s ON c.campaign_id = s.campaign_id
LEFT JOIN creative.assets ca ON c.campaign_id = ca.campaign_id
LEFT JOIN effectiveness.brand_building bb ON c.campaign_id = bb.campaign_id
LEFT JOIN effectiveness.media_mix_models mmm ON c.campaign_id = mmm.campaign_id
GROUP BY c.campaign_id, c.campaign_name, c.client_id, c.launch_date, 
         c.end_date, c.total_budget_usd, s.overall_effectiveness_score,
         s.warc_percentile_rank, s.predicted_roi_impact;

CREATE INDEX idx_perf_summary_campaign ON effectiveness.campaign_performance_summary(campaign_id);
CREATE INDEX idx_perf_summary_score ON effectiveness.campaign_performance_summary(overall_effectiveness_score DESC);

-- Refresh materialized view periodically
CREATE OR REPLACE FUNCTION refresh_campaign_performance_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY effectiveness.campaign_performance_summary;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- INITIAL DATA SEEDS
-- =====================================================

-- Insert WARC benchmark categories
INSERT INTO warc_standards.global_benchmarks (market, category, benchmark_period, market_size_index, media_cost_index, competitive_intensity_index, digital_maturity_index)
VALUES 
    ('USA', 'CPG', '2024', 100.0, 100.0, 85.0, 95.0),
    ('Philippines', 'Telecom', '2024', 45.0, 35.0, 90.0, 75.0),
    ('Philippines', 'QSR', '2024', 45.0, 35.0, 95.0, 75.0),
    ('Global', 'Technology', '2024', 100.0, 120.0, 95.0, 100.0);

-- Grant appropriate permissions
GRANT USAGE ON SCHEMA creative TO jampacked_app;
GRANT USAGE ON SCHEMA campaigns TO jampacked_app;
GRANT USAGE ON SCHEMA effectiveness TO jampacked_app;
GRANT USAGE ON SCHEMA warc_standards TO jampacked_app;

GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA creative TO jampacked_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA campaigns TO jampacked_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA effectiveness TO jampacked_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA warc_standards TO jampacked_app;