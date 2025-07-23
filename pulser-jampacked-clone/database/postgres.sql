-- PostgreSQL Schema for JamPacked Creative Intelligence Clone
-- Database: campaign_db

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";

-- Campaigns table
CREATE TABLE campaigns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_name VARCHAR(255) NOT NULL,
    brand VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    campaign_objective VARCHAR(50) CHECK (campaign_objective IN ('brand_awareness', 'consideration', 'conversion', 'retention')),
    start_date DATE,
    end_date DATE,
    budget DECIMAL(12, 2),
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Creative assets table
CREATE TABLE creative_assets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    asset_type VARCHAR(50) CHECK (asset_type IN ('video', 'image', 'audio', 'text', 'interactive')),
    asset_url TEXT NOT NULL,
    platform VARCHAR(100),
    dimensions VARCHAR(50),
    duration_seconds INTEGER,
    file_size_mb DECIMAL(10, 2),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Campaign performance metrics
CREATE TABLE campaign_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    impressions INTEGER DEFAULT 0,
    clicks INTEGER DEFAULT 0,
    conversions INTEGER DEFAULT 0,
    spend DECIMAL(10, 2) DEFAULT 0,
    engagement_rate DECIMAL(5, 4),
    conversion_rate DECIMAL(5, 4),
    brand_recall_score DECIMAL(5, 2),
    consideration_lift DECIMAL(5, 4),
    effectiveness_score DECIMAL(5, 2),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(campaign_id, date)
);

-- Creative analyses results
CREATE TABLE creative_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID REFERENCES campaigns(id),
    campaign_objective VARCHAR(50),
    industry VARCHAR(100),
    brand_positioning TEXT,
    effectiveness_score DECIMAL(5, 2),
    feature_scores JSONB,
    performance_forecast JSONB,
    optimization_recommendations JSONB,
    award_predictions JSONB,
    processing_time_seconds DECIMAL(6, 2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Optimization plans
CREATE TABLE optimization_plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    current_performance JSONB,
    predicted_performance JSONB,
    tactics JSONB,
    objectives JSONB,
    applied_tactics TEXT[],
    applied_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP
);

-- A/B test results
CREATE TABLE ab_test_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    test_name VARCHAR(255),
    hypothesis TEXT,
    variant_a JSONB,
    variant_b JSONB,
    start_date DATE,
    end_date DATE,
    sample_size_a INTEGER,
    sample_size_b INTEGER,
    metric_results JSONB,
    winner VARCHAR(1),
    confidence_level DECIMAL(3, 2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Industry benchmarks
CREATE TABLE industry_benchmarks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    industry VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10, 4),
    percentile INTEGER,
    sample_size INTEGER,
    date_range_start DATE,
    date_range_end DATE,
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(industry, metric_name, percentile)
);

-- Award predictions tracking
CREATE TABLE award_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    award_show VARCHAR(100),
    category VARCHAR(255),
    predicted_probability DECIMAL(3, 2),
    confidence_level DECIMAL(3, 2),
    prediction_date TIMESTAMP DEFAULT NOW(),
    actual_result VARCHAR(50),
    result_date DATE
);

-- Indexes for performance
CREATE INDEX idx_campaigns_brand ON campaigns(brand);
CREATE INDEX idx_campaigns_industry ON campaigns(industry);
CREATE INDEX idx_campaigns_status ON campaigns(status);
CREATE INDEX idx_campaign_performance_campaign_date ON campaign_performance(campaign_id, date DESC);
CREATE INDEX idx_creative_analyses_campaign ON creative_analyses(campaign_id);
CREATE INDEX idx_creative_analyses_score ON creative_analyses(effectiveness_score DESC);
CREATE INDEX idx_optimization_plans_campaign ON optimization_plans(campaign_id);
CREATE INDEX idx_optimization_plans_expires ON optimization_plans(expires_at);
CREATE INDEX idx_ab_test_campaign ON ab_test_results(campaign_id);
CREATE INDEX idx_benchmarks_industry ON industry_benchmarks(industry, metric_name);

-- Materialized view for campaign summaries
CREATE MATERIALIZED VIEW campaign_summaries AS
SELECT 
    c.id,
    c.campaign_name,
    c.brand,
    c.industry,
    c.campaign_objective,
    COUNT(DISTINCT ca.id) as asset_count,
    AVG(cp.effectiveness_score) as avg_effectiveness_score,
    SUM(cp.impressions) as total_impressions,
    SUM(cp.clicks) as total_clicks,
    SUM(cp.conversions) as total_conversions,
    AVG(cp.engagement_rate) as avg_engagement_rate,
    AVG(cp.conversion_rate) as avg_conversion_rate,
    MAX(analysis.effectiveness_score) as latest_analysis_score,
    COUNT(DISTINCT opt.id) as optimization_count,
    COUNT(DISTINCT ab.id) as ab_test_count
FROM campaigns c
LEFT JOIN creative_assets ca ON c.id = ca.campaign_id
LEFT JOIN campaign_performance cp ON c.id = cp.campaign_id
LEFT JOIN creative_analyses analysis ON c.id = analysis.campaign_id
LEFT JOIN optimization_plans opt ON c.id = opt.campaign_id
LEFT JOIN ab_test_results ab ON c.id = ab.campaign_id
GROUP BY c.id;

-- Refresh materialized view function
CREATE OR REPLACE FUNCTION refresh_campaign_summaries()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY campaign_summaries;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_campaigns_updated_at BEFORE UPDATE ON campaigns
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Sample data for testing
INSERT INTO industry_benchmarks (industry, metric_name, metric_value, percentile, sample_size) VALUES
('telecom', 'engagement_rate', 0.025, 25, 1000),
('telecom', 'engagement_rate', 0.035, 50, 1000),
('telecom', 'engagement_rate', 0.048, 75, 1000),
('telecom', 'conversion_rate', 0.008, 25, 1000),
('telecom', 'conversion_rate', 0.012, 50, 1000),
('telecom', 'conversion_rate', 0.018, 75, 1000),
('retail', 'engagement_rate', 0.032, 25, 1500),
('retail', 'engagement_rate', 0.045, 50, 1500),
('retail', 'engagement_rate', 0.062, 75, 1500),
('retail', 'conversion_rate', 0.015, 25, 1500),
('retail', 'conversion_rate', 0.022, 50, 1500),
('retail', 'conversion_rate', 0.035, 75, 1500);

-- Grant permissions (adjust as needed)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO creative_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO creative_user;