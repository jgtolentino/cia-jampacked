# CES JamPacked Agentic - MCP Integration Guide

## 🔗 Seamless Integration with Model Context Protocol

CES JamPacked Agentic integrates directly with your existing MCP SQLite server for AI-powered advertising optimization:
- **Database Path**: `/Users/tbwa/Documents/GitHub/mcp-sqlite-server/data/database.sqlite`
- **MCP Server**: `/Users/tbwa/Documents/GitHub/mcp-sqlite-server/dist/index.js`

## 🚀 Quick Setup

1. **Initialize JamPacked tables in your existing MCP database**:
```bash
cd /Users/tbwa/Documents/GitHub/jampacked-creative-intelligence
python setup_mcp_integration.py
```

This creates JamPacked tables in your existing SQLite database:
- `jampacked_creative_analysis`
- `jampacked_pattern_discoveries`
- `jampacked_cultural_insights`
- `jampacked_optimizations`
- `jampacked_sessions`

## 📊 How It Works

### 1. **Run JamPacked Analysis (Python)**
```python
from jampacked_sqlite_integration import analyze_campaign_via_mcp

# Analyze campaign
results = await analyze_campaign_via_mcp(
    materials={'text': ['Ad copy'], 'images': [image_data]},
    campaign_context={'campaign_name': 'Q4 Launch', 'target_cultures': ['us', 'uk']}
)

# Results are automatically stored in your MCP SQLite database
print(f"Campaign ID: {results['campaign_id']}")
```

### 2. **Query Results in Claude Desktop (SQL)**
Since the data is in your MCP SQLite database, use standard SQL queries:

```sql
-- Get latest campaign analysis
SELECT * FROM jampacked_creative_analysis 
WHERE campaign_id = 'your_campaign_id'
ORDER BY created_at DESC;

-- Find novel patterns
SELECT * FROM jampacked_pattern_discoveries
WHERE novelty_score > 0.8
ORDER BY novelty_score DESC;

-- Get cultural insights
SELECT * FROM jampacked_cultural_insights
WHERE campaign_id = 'your_campaign_id';
```

### 3. **Access from Claude Code**
Claude Code can access the same data through the shared SQLite database:
```bash
# Use the same queries or Python integration
# All data is shared between Claude Desktop and Claude Code
```

## 🎯 Key Benefits

### **No Duplication**
- Uses your existing MCP SQLite infrastructure
- No separate databases or servers needed
- All JamPacked data stored alongside your other MCP data

### **Unified Access**
- **Claude Desktop**: Query via SQL using MCP tools
- **Claude Code**: Access via Python or SQL
- **Shared Sessions**: Work seamlessly between interfaces

### **Real Examples**

#### Example 1: Analyze McDonald's Campaign
```python
# In Python/Claude Code
materials = {
    'images': [hero_image, banner_image],
    'text': ['Lovin\' It', 'Campaign tagline'],
    'videos': [tv_commercial]
}

context = {
    'campaign_name': 'McDonalds Holiday 2024',
    'target_cultures': ['us', 'japan', 'brazil'],
    'business_objectives': ['brand_awareness', 'sales_lift']
}

results = await analyze_campaign_via_mcp(materials, context)
```

#### Example 2: Query in Claude Desktop
```sql
-- Get McDonald's campaign effectiveness
SELECT 
    campaign_name,
    creative_effectiveness_score,
    cultural_alignment_score,
    recommendations
FROM jampacked_creative_analysis
WHERE campaign_name LIKE '%McDonalds%'
ORDER BY created_at DESC;

-- Find optimization opportunities
SELECT * FROM jampacked_optimizations
WHERE campaign_id IN (
    SELECT campaign_id FROM jampacked_creative_analysis
    WHERE campaign_name LIKE '%McDonalds%'
)
ORDER BY priority_score DESC;
```

## 📋 Available SQL Tables

### `jampacked_creative_analysis`
- `campaign_id`: Unique campaign identifier
- `creative_effectiveness_score`: Overall effectiveness (0-1)
- `attention_score`: Attention prediction score
- `emotion_score`: Emotional impact score
- `brand_recall_score`: Brand memorability score
- `cultural_alignment_score`: Cultural appropriateness
- `analysis_results`: Full JSON analysis
- `recommendations`: JSON array of recommendations

### `jampacked_pattern_discoveries`
- `pattern_type`: Type of pattern discovered
- `novelty_score`: How novel/unique the pattern is (0-1)
- `business_impact`: Predicted business impact
- `pattern_data`: Full pattern details in JSON

### `jampacked_cultural_insights`
- `culture`: Target culture code (us, uk, japan, etc.)
- `effectiveness_score`: Cultural effectiveness (0-1)
- `adaptation_recommendations`: Localization suggestions
- `risk_assessment`: Cultural risks identified

### `jampacked_optimizations`
- `optimization_type`: Category of optimization
- `predicted_impact`: Expected performance lift
- `implementation_effort`: low/medium/high
- `priority_score`: Recommended priority (0-1)
- `ab_test_plan`: A/B testing recommendations

## 🔧 Advanced Usage

### Cross-Session Continuity
```python
# Start analysis in Claude Code
session_id = 'abc123'
results = await analyze_campaign_via_mcp(materials, context, session_id)

# Continue in Claude Desktop with same session
# SELECT * FROM jampacked_sessions WHERE session_id = 'abc123';
```

### Batch Analysis
```python
# Analyze multiple campaigns
campaigns = [campaign1, campaign2, campaign3]
for campaign in campaigns:
    await analyze_campaign_via_mcp(campaign['materials'], campaign['context'])

# Query all results in Claude Desktop
# SELECT * FROM jampacked_creative_analysis ORDER BY created_at DESC;
```

### Performance Tracking
```sql
-- Track JamPacked performance over time
SELECT 
    DATE(created_at) as analysis_date,
    COUNT(*) as campaigns_analyzed,
    AVG(creative_effectiveness_score) as avg_effectiveness
FROM jampacked_creative_analysis
GROUP BY DATE(created_at)
ORDER BY analysis_date DESC;
```

## 🎉 Summary

JamPacked seamlessly integrates with your existing MCP SQLite server:
- **No new infrastructure** - Uses your existing database
- **Unified data access** - Same data in Claude Desktop and Code
- **Standard SQL queries** - No special tools needed
- **Full JamPacked intelligence** - All features available

Just run the setup script and start analyzing campaigns! All results are automatically available in both Claude interfaces through your existing MCP SQLite server.