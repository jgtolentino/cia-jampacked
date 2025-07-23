-- sql/enhanced_youtube_analysis_with_drive.sql
-- Enhanced Claude Desktop SQL Trigger for YouTube + Google Drive Analysis
-- Combines video analysis with campaign intelligence from Pulser Drive folder

-- Example usage from Claude Desktop:
-- sqlite3 data/database.sqlite < sql/enhanced_youtube_analysis_with_drive.sql

-- Insert enhanced YouTube analysis task with Google Drive context
INSERT INTO agent_task_queue (
    task_id,
    source_agent,
    target_agent,
    task_type,
    payload,
    status,
    priority,
    created_at
) VALUES (
    'enhanced_yt_001',
    'claude_desktop',
    'pulser',
    'analyze_youtube',
    json('{
        "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "analysis_focus": "enhanced_creative_effectiveness",
        "client_context": {
            "brand": "McDonalds",
            "client": "McDonalds Philippines", 
            "campaign_objective": "brand_awareness_local_market",
            "target_audience": "filipino_families_millennials",
            "industry": "quick_service_restaurant",
            "market": "philippines",
            "campaign_type": "brand_building",
            "brand_values": ["family", "joy", "accessibility", "local_connection"],
            "positioning": "bringing families together through food and joy",
            "distinctive_assets": ["golden_arches", "im_lovin_it", "red_yellow_branding"]
        },
        "google_drive_integration": {
            "enabled": true,
            "campaign_root_folder": "0AJMhu01UUQKoUk9PVA",
            "extract_context": true,
            "search_for": [
                "creative_briefs",
                "brand_guidelines",
                "campaign_strategy",
                "historical_performance",
                "competitive_analysis"
            ]
        },
        "warc_dimensions": [
            "strategic_planning_rigor",
            "creative_idea_excellence", 
            "business_results_delivery",
            "brand_building_impact",
            "cultural_social_value"
        ],
        "award_shows": [
            "cannes_lions",
            "effie_awards", 
            "one_show",
            "dad_pencils",
            "clio_awards"
        ],
        "analysis_enhancements": [
            "historical_campaign_benchmarking",
            "competitive_industry_analysis", 
            "brand_guidelines_compliance",
            "cultural_relevance_philippines",
            "award_prediction_enhancement"
        ],
        "output_format": "comprehensive_enhanced_report",
        "urgency": "high"
    }'),
    'pending',
    9,
    datetime('now')
);

-- Verify enhanced task was created
SELECT 
    task_id,
    source_agent,
    target_agent,
    task_type,
    status,
    priority,
    created_at,
    json_extract(payload, '$.client_context.brand') as brand,
    json_extract(payload, '$.google_drive_integration.enabled') as drive_enabled
FROM agent_task_queue 
WHERE task_id = 'enhanced_yt_001';

-- Monitor all enhanced YouTube analysis tasks
SELECT 
    task_id,
    target_agent,
    task_type,
    status,
    priority,
    created_at,
    json_extract(payload, '$.client_context.brand') as brand,
    json_extract(payload, '$.analysis_focus') as focus,
    CASE 
        WHEN status = 'completed' THEN '✅'
        WHEN status = 'in_progress' THEN '⏳'
        WHEN status = 'failed' THEN '❌'
        ELSE '⏸️'
    END as status_icon
FROM agent_task_queue 
WHERE task_type IN ('analyze_youtube', 'analyze_transcript', 'campaign_context_extraction')
AND json_extract(payload, '$.google_drive_integration.enabled') = 'true'
ORDER BY priority DESC, created_at DESC;

-- Track the complete enhanced workflow
WITH enhanced_workflow_tracking AS (
    SELECT 
        task_id,
        target_agent,
        task_type,
        status,
        priority,
        created_at,
        json_extract(payload, '$.client_context.brand') as brand,
        CASE task_type
            WHEN 'analyze_youtube' THEN 1
            WHEN 'campaign_context_extraction' THEN 2
            WHEN 'analyze_transcript' THEN 3
            ELSE 4
        END as workflow_step
    FROM agent_task_queue 
    WHERE task_id LIKE 'enhanced_%' 
    OR json_extract(payload, '$.google_drive_integration.enabled') = 'true'
    OR task_type = 'campaign_context_extraction'
)
SELECT 
    workflow_step,
    CASE 
        WHEN workflow_step = 1 THEN '🎬 Video Processing'
        WHEN workflow_step = 2 THEN '📁 Drive Context Extraction'
        WHEN workflow_step = 3 THEN '🧠 Enhanced Creative Analysis'
        ELSE '📊 Results Integration'
    END as workflow_stage,
    task_id,
    target_agent,
    brand,
    CASE status
        WHEN 'completed' THEN '✅ Done'
        WHEN 'in_progress' THEN '⏳ Working'
        WHEN 'failed' THEN '❌ Error'
        ELSE '⏸️ Queued'
    END as status,
    priority,
    created_at
FROM enhanced_workflow_tracking
ORDER BY workflow_step, priority DESC, created_at;

-- Enhanced status dashboard
.mode table
.headers on
SELECT 
    '🎯 Enhanced YouTube Analysis Dashboard' as component,
    CASE 
        WHEN EXISTS(
            SELECT 1 FROM agent_task_queue 
            WHERE task_id = 'enhanced_yt_001' AND status = 'completed'
        ) THEN '✅ Analysis Complete'
        WHEN EXISTS(
            SELECT 1 FROM agent_task_queue 
            WHERE task_id = 'enhanced_yt_001' AND status = 'in_progress'
        ) THEN '⏳ Processing...'
        WHEN EXISTS(
            SELECT 1 FROM agent_task_queue 
            WHERE task_id = 'enhanced_yt_001' AND status = 'failed'
        ) THEN '❌ Failed'
        ELSE '⏸️ Pending'
    END as status;

-- Get enhanced results when ready
SELECT 
    '🎯 Enhanced Analysis Results for enhanced_yt_001:' as info,
    json_extract(result, '$.campaign_intelligence.historical_campaigns') as historical_data,
    json_extract(result, '$.award_predictions') as award_predictions,
    json_extract(result, '$.competitive_analysis') as competitive_analysis,
    json_extract(result, '$.strategic_recommendations') as recommendations
FROM agent_task_queue 
WHERE task_id = 'enhanced_yt_001' 
AND status = 'completed';

-- Campaign context extraction status
SELECT 
    '📁 Google Drive Context Extraction:' as info,
    task_id,
    status,
    json_extract(payload, '$.search_terms') as search_terms,
    json_extract(result, '$.files_processed') as files_found
FROM agent_task_queue 
WHERE task_type = 'campaign_context_extraction'
AND json_extract(payload, '$.folder_id') = '0AJMhu01UUQKoUk9PVA'
ORDER BY created_at DESC
LIMIT 5;

-- Performance metrics
WITH performance_metrics AS (
    SELECT 
        COUNT(*) as total_enhanced_tasks,
        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_tasks,
        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_tasks,
        AVG(CASE 
            WHEN status = 'completed' THEN 
                (julianday(updated_at) - julianday(created_at)) * 24 * 60 
            ELSE NULL 
        END) as avg_processing_time_minutes
    FROM agent_task_queue 
    WHERE json_extract(payload, '$.google_drive_integration.enabled') = 'true'
    OR task_type = 'campaign_context_extraction'
)
SELECT 
    '📊 Enhanced Analysis Performance:' as metric_category,
    total_enhanced_tasks as total_tasks,
    completed_tasks,
    failed_tasks,
    ROUND(avg_processing_time_minutes, 2) as avg_time_minutes,
    ROUND((completed_tasks * 100.0 / total_enhanced_tasks), 1) as success_rate_percent
FROM performance_metrics;

-- Quick test: Create a simpler enhanced analysis task
INSERT INTO agent_task_queue (
    task_id,
    source_agent,
    target_agent,
    task_type,
    payload,
    status,
    priority,
    created_at
) VALUES (
    'test_enhanced_' || strftime('%s', 'now'),
    'claude_desktop_test',
    'pulser',
    'analyze_youtube',
    json('{
        "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "analysis_focus": "enhanced_creative_effectiveness",
        "client_context": {
            "brand": "TestBrand",
            "client": "Test Client",
            "industry": "technology"
        },
        "google_drive_integration": {
            "enabled": true,
            "campaign_root_folder": "0AJMhu01UUQKoUk9PVA"
        },
        "test_mode": true
    }'),
    'pending',
    8,
    datetime('now')
);

-- Show what tasks are ready for processing
SELECT 
    '🚀 Ready for Processing:' as status,
    COUNT(*) as pending_tasks
FROM agent_task_queue 
WHERE status = 'pending' 
AND (
    json_extract(payload, '$.google_drive_integration.enabled') = 'true'
    OR task_type = 'campaign_context_extraction'
);

-- Display recent enhanced analysis tasks
SELECT 
    task_id,
    SUBSTR(task_id, 1, 20) as short_id,
    target_agent,
    json_extract(payload, '$.client_context.brand') as brand,
    status,
    CASE priority
        WHEN 9 THEN '🔥 Critical'
        WHEN 8 THEN '⚡ High'
        WHEN 7 THEN '📈 Medium'
        ELSE '📋 Normal'
    END as priority_level,
    datetime(created_at, 'localtime') as created_local
FROM agent_task_queue 
WHERE (
    json_extract(payload, '$.google_drive_integration.enabled') = 'true'
    OR task_type = 'campaign_context_extraction'
)
ORDER BY created_at DESC
LIMIT 10;