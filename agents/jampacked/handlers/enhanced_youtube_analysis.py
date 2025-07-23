# agents/jampacked/handlers/enhanced_youtube_analysis.py
"""
Enhanced YouTube Analysis with Google Drive Campaign Intelligence
Combines video analysis with campaign context from Pulser Drive folder
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
import re

class EnhancedYouTubeAnalyzer:
    """
    Enhanced JamPacked YouTube Analysis with Google Drive Integration
    Provides campaign context and historical intelligence for better predictions
    """
    
    def __init__(self, db_path="/Users/tbwa/Documents/GitHub/mcp-sqlite-server/data/database.sqlite"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.DRIVE_CAMPAIGN_ROOT_ID = "0AJMhu01UUQKoUk9PVA"
        
    def analyze_youtube_with_context(self, task_data):
        """
        Enhanced analysis combining video content with campaign intelligence
        """
        try:
            transcript = task_data.get('transcript', '')
            video_metadata = task_data.get('metadata', {})
            client_context = task_data.get('client_context', {})
            
            # Step 1: Extract campaign context from Google Drive
            campaign_intelligence = self.extract_campaign_context(client_context, video_metadata)
            
            # Step 2: Enhanced WARC analysis with campaign context
            warc_analysis = self.extract_enhanced_warc_dimensions(
                transcript, video_metadata, campaign_intelligence
            )
            
            # Step 3: Historical award prediction with campaign data
            award_predictions = self.predict_awards_with_context(
                warc_analysis, campaign_intelligence
            )
            
            # Step 4: Competitive benchmarking
            competitive_analysis = self.perform_competitive_analysis(
                warc_analysis, campaign_intelligence
            )
            
            # Step 5: Strategic recommendations
            strategic_recommendations = self.generate_strategic_recommendations(
                warc_analysis, award_predictions, competitive_analysis
            )
            
            # Compile comprehensive analysis
            enhanced_analysis = {
                'video_analysis': {
                    'transcript': transcript,
                    'metadata': video_metadata,
                    'warc_dimensions': warc_analysis
                },
                'campaign_intelligence': campaign_intelligence,
                'award_predictions': award_predictions,
                'competitive_analysis': competitive_analysis,
                'strategic_recommendations': strategic_recommendations,
                'confidence_metrics': self.calculate_enhanced_confidence(
                    warc_analysis, campaign_intelligence
                ),
                'analysis_timestamp': datetime.now().isoformat(),
                'integration_version': '2.0_drive_enhanced'
            }
            
            # Store enhanced results
            self.store_enhanced_analysis(enhanced_analysis)
            
            return enhanced_analysis
            
        except Exception as e:
            self.logger.error(f"Enhanced analysis failed: {str(e)}")
            return {'error': str(e), 'status': 'failed'}
    
    def extract_campaign_context(self, client_context, video_metadata):
        """
        Extract relevant campaign context from Google Drive
        """
        # Create Drive extraction task for campaign context
        drive_task_id = self.request_drive_extraction(client_context, video_metadata)
        
        # Get existing campaign intelligence from database
        historical_data = self.get_historical_campaign_data(client_context)
        
        return {
            'drive_extraction_task_id': drive_task_id,
            'historical_campaigns': historical_data,
            'brand_context': self.extract_brand_context(client_context),
            'industry_benchmarks': self.get_industry_benchmarks(client_context.get('industry')),
            'campaign_objectives': self.map_campaign_objectives(client_context),
            'target_audience_insights': self.get_audience_insights(client_context.get('target_audience'))
        }
    
    def request_drive_extraction(self, client_context, video_metadata):
        """
        Create task for Claude Desktop to extract relevant campaign materials
        """
        drive_task_id = f"drive_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Construct intelligent search query for Drive
        search_terms = []
        if client_context.get('brand'):
            search_terms.append(client_context['brand'])
        if client_context.get('client'):
            search_terms.append(client_context['client'])
        if video_metadata.get('title'):
            # Extract potential campaign names from video title
            search_terms.extend(self.extract_campaign_keywords(video_metadata['title']))
        
        payload = {
            'task_type': 'campaign_context_extraction',
            'folder_id': self.DRIVE_CAMPAIGN_ROOT_ID,
            'search_terms': search_terms,
            'client_context': client_context,
            'video_metadata': video_metadata,
            'extraction_focus': [
                'creative_briefs',
                'brand_guidelines', 
                'campaign_strategies',
                'target_audience_definitions',
                'competitive_analyses',
                'historical_performance_data'
            ],
            'instructions': f"""
            Extract campaign context for enhanced YouTube analysis:
            
            1. Search folder {self.DRIVE_CAMPAIGN_ROOT_ID} for materials related to:
               - Brand: {client_context.get('brand', 'Unknown')}
               - Client: {client_context.get('client', 'Unknown')}
               - Keywords: {', '.join(search_terms)}
            
            2. Look for these document types:
               - Creative briefs (.docx, .pdf)
               - Brand guidelines
               - Campaign strategies
               - Target audience research
               - Competitive analysis reports
               - Previous campaign performance data
            
            3. Extract key information:
               - Brand positioning and values
               - Campaign objectives and KPIs
               - Target audience demographics/psychographics
               - Competitive landscape
               - Historical campaign performance
               - Award-winning campaign patterns
            
            4. Return structured data for analysis enhancement
            """
        }
        
        # Insert drive extraction task
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO agent_task_queue 
            (task_id, source_agent, target_agent, task_type, payload, priority, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            drive_task_id,
            'jampacked_enhanced',
            'claude_desktop',
            'campaign_context_extraction',
            json.dumps(payload),
            9,  # Very high priority
            'pending',
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Created Drive context extraction task: {drive_task_id}")
        return drive_task_id
    
    def extract_enhanced_warc_dimensions(self, transcript, metadata, campaign_intelligence):
        """
        Enhanced WARC analysis with campaign context
        """
        base_analysis = self.extract_warc_dimensions(transcript, metadata)
        
        # Enhance with campaign intelligence
        if campaign_intelligence.get('brand_context'):
            brand_context = campaign_intelligence['brand_context']
            
            # Enhanced Strategic Planning with brand context
            base_analysis['strategic_planning']['brand_alignment'] = self.score_brand_alignment(
                transcript, brand_context
            )
            base_analysis['strategic_planning']['objective_achievement'] = self.score_objective_achievement(
                transcript, metadata, campaign_intelligence.get('campaign_objectives', {})
            )
            
            # Enhanced Creative Excellence with historical benchmarks
            if campaign_intelligence.get('historical_campaigns'):
                base_analysis['creative_excellence']['innovation_vs_benchmark'] = self.compare_innovation_to_history(
                    base_analysis['creative_excellence']['innovation_level'],
                    campaign_intelligence['historical_campaigns']
                )
            
            # Enhanced Brand Building with guidelines compliance
            base_analysis['brand_building']['guidelines_compliance'] = self.score_guidelines_compliance(
                transcript, brand_context.get('guidelines', {})
            )
            base_analysis['brand_building']['distinctive_assets_usage'] = self.score_distinctive_assets_usage(
                transcript, brand_context.get('distinctive_assets', [])
            )
        
        return base_analysis
    
    def predict_awards_with_context(self, warc_analysis, campaign_intelligence):
        """
        Enhanced award prediction using campaign intelligence
        """
        base_predictions = self.predict_awards(warc_analysis)
        
        # Enhance predictions with historical data
        historical_campaigns = campaign_intelligence.get('historical_campaigns', [])
        brand_context = campaign_intelligence.get('brand_context', {})
        
        for award_show in base_predictions:
            if award_show == 'ensemble':
                continue
                
            prediction = base_predictions[award_show]
            
            # Apply historical performance boost
            historical_boost = self.calculate_historical_performance_boost(
                award_show, historical_campaigns
            )
            
            # Apply brand equity boost
            brand_boost = self.calculate_brand_equity_boost(
                award_show, brand_context
            )
            
            # Apply category expertise boost
            category_boost = self.calculate_category_expertise_boost(
                award_show, campaign_intelligence.get('industry_benchmarks', {})
            )
            
            # Enhanced probability calculation
            enhanced_probability = min(
                prediction['probability'] * (1 + historical_boost + brand_boost + category_boost),
                0.98
            )
            
            base_predictions[award_show].update({
                'enhanced_probability': enhanced_probability,
                'historical_boost': historical_boost,
                'brand_boost': brand_boost,
                'category_boost': category_boost,
                'confidence_factors': {
                    'historical_data_available': len(historical_campaigns) > 0,
                    'brand_context_available': bool(brand_context),
                    'industry_benchmarks_available': bool(campaign_intelligence.get('industry_benchmarks'))
                }
            })
        
        return base_predictions
    
    def perform_competitive_analysis(self, warc_analysis, campaign_intelligence):
        """
        Competitive analysis using industry benchmarks
        """
        industry_benchmarks = campaign_intelligence.get('industry_benchmarks', {})
        
        return {
            'performance_vs_industry': self.compare_to_industry_benchmarks(
                warc_analysis, industry_benchmarks
            ),
            'competitive_advantages': self.identify_competitive_advantages(
                warc_analysis, industry_benchmarks
            ),
            'improvement_opportunities': self.identify_improvement_opportunities(
                warc_analysis, industry_benchmarks
            ),
            'market_positioning': self.assess_market_positioning(
                warc_analysis, campaign_intelligence
            )
        }
    
    def generate_strategic_recommendations(self, warc_analysis, award_predictions, competitive_analysis):
        """
        Generate strategic recommendations based on comprehensive analysis
        """
        recommendations = {
            'immediate_optimizations': [],
            'award_submission_strategy': [],
            'long_term_improvements': [],
            'competitive_response': []
        }
        
        # Immediate optimizations
        for dimension, scores in warc_analysis.items():
            if isinstance(scores, dict):
                for metric, score in scores.items():
                    if isinstance(score, (int, float)) and score < 0.7:
                        recommendations['immediate_optimizations'].append({
                            'dimension': dimension,
                            'metric': metric,
                            'current_score': score,
                            'target_score': 0.8,
                            'recommendation': self.get_improvement_recommendation(dimension, metric)
                        })
        
        # Award submission strategy
        for award_show, prediction in award_predictions.items():
            if award_show != 'ensemble' and prediction.get('enhanced_probability', 0) > 0.6:
                recommendations['award_submission_strategy'].append({
                    'award_show': award_show,
                    'probability': prediction.get('enhanced_probability'),
                    'recommended_categories': prediction.get('recommended_categories', []),
                    'submission_priority': 'high' if prediction.get('enhanced_probability') > 0.8 else 'medium'
                })
        
        return recommendations
    
    # Helper methods for enhanced analysis
    def extract_campaign_keywords(self, title):
        """Extract potential campaign keywords from video title"""
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', title.lower())
        return [word for word in words if len(word) > 3 and word not in stop_words]
    
    def get_historical_campaign_data(self, client_context):
        """Retrieve historical campaign data from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT c.*, ca.award_show, ca.award_level, ca.award_year
            FROM campaigns c
            LEFT JOIN campaign_awards ca ON c.campaign_id = ca.campaign_id
            WHERE (c.client = ? OR c.brand = ?)
            ORDER BY ca.award_year DESC, ca.award_level
            LIMIT 50
        """
        
        cursor.execute(query, (
            client_context.get('client', ''),
            client_context.get('brand', '')
        ))
        
        results = cursor.fetchall()
        conn.close()
        
        return [dict(zip([col[0] for col in cursor.description], row)) for row in results]
    
    def extract_brand_context(self, client_context):
        """Extract and structure brand context"""
        return {
            'brand_name': client_context.get('brand'),
            'client_name': client_context.get('client'),
            'industry': client_context.get('industry'),
            'campaign_objective': client_context.get('campaign_objective'),
            'target_audience': client_context.get('target_audience'),
            'brand_values': client_context.get('brand_values', []),
            'positioning': client_context.get('positioning', ''),
            'distinctive_assets': client_context.get('distinctive_assets', [])
        }
    
    def get_industry_benchmarks(self, industry):
        """Get industry performance benchmarks"""
        # This would typically query a benchmarks database
        # For now, return industry-specific baseline scores
        industry_benchmarks = {
            'technology': {
                'innovation_baseline': 0.8,
                'emotional_resonance_baseline': 0.6,
                'viral_potential_baseline': 0.7
            },
            'retail': {
                'innovation_baseline': 0.6,
                'emotional_resonance_baseline': 0.8,
                'viral_potential_baseline': 0.6
            },
            'automotive': {
                'innovation_baseline': 0.7,
                'emotional_resonance_baseline': 0.7,
                'viral_potential_baseline': 0.5
            }
        }
        
        return industry_benchmarks.get(industry, industry_benchmarks['technology'])
    
    def calculate_historical_performance_boost(self, award_show, historical_campaigns):
        """Calculate performance boost based on historical award wins"""
        award_count = len([c for c in historical_campaigns if c.get('award_show') == award_show])
        recent_wins = len([c for c in historical_campaigns 
                          if c.get('award_show') == award_show and 
                          c.get('award_year', 0) >= datetime.now().year - 3])
        
        return min(award_count * 0.05 + recent_wins * 0.1, 0.3)
    
    # Additional placeholder methods would be implemented here...
    def extract_warc_dimensions(self, transcript, metadata):
        """Base WARC dimension extraction"""
        # This would call the original analysis method
        from .analyze_transcript import YouTubeTranscriptAnalyzer
        base_analyzer = YouTubeTranscriptAnalyzer(self.db_path)
        return base_analyzer.extract_warc_dimensions(transcript, metadata)
    
    def predict_awards(self, warc_analysis):
        """Base award prediction"""
        from .analyze_transcript import YouTubeTranscriptAnalyzer
        base_analyzer = YouTubeTranscriptAnalyzer(self.db_path)
        return base_analyzer.predict_awards(warc_analysis)
    
    # Placeholder methods for demonstration
    def score_brand_alignment(self, transcript, brand_context): return 0.8
    def score_objective_achievement(self, transcript, metadata, objectives): return 0.7
    def compare_innovation_to_history(self, innovation_score, historical): return innovation_score * 1.1
    def score_guidelines_compliance(self, transcript, guidelines): return 0.9
    def score_distinctive_assets_usage(self, transcript, assets): return 0.8
    def calculate_brand_equity_boost(self, award_show, brand_context): return 0.1
    def calculate_category_expertise_boost(self, award_show, benchmarks): return 0.05
    def compare_to_industry_benchmarks(self, analysis, benchmarks): return {}
    def identify_competitive_advantages(self, analysis, benchmarks): return []
    def identify_improvement_opportunities(self, analysis, benchmarks): return []
    def assess_market_positioning(self, analysis, intelligence): return {}
    def get_improvement_recommendation(self, dimension, metric): return f"Improve {metric} in {dimension}"
    def map_campaign_objectives(self, context): return {}
    def get_audience_insights(self, audience): return {}
    def calculate_enhanced_confidence(self, warc, intelligence): return 0.85
    def store_enhanced_analysis(self, analysis): pass

# Enhanced task runner
def run_enhanced_task_runner():
    """
    Enhanced task runner that handles both YouTube analysis and Drive integration
    """
    db_path = "/Users/tbwa/Documents/GitHub/mcp-sqlite-server/data/database.sqlite"
    analyzer = EnhancedYouTubeAnalyzer(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("🚀 Enhanced YouTube + Drive Analyzer started")
    
    while True:
        try:
            # Poll for enhanced analysis tasks
            cursor.execute("""
                SELECT task_id, payload FROM agent_task_queue 
                WHERE target_agent = 'jampacked' 
                AND task_type IN ('analyze_transcript', 'enhanced_youtube_analysis')
                AND status = 'pending'
                LIMIT 1
            """)
            
            task = cursor.fetchone()
            if task:
                task_id, payload = task
                task_data = json.loads(payload)
                
                print(f"🧠 Processing enhanced analysis task {task_id}")
                
                # Update status
                cursor.execute("""
                    UPDATE agent_task_queue 
                    SET status = 'in_progress', updated_at = ?
                    WHERE task_id = ?
                """, (datetime.now().isoformat(), task_id))
                conn.commit()
                
                try:
                    # Run enhanced analysis
                    if task_data.get('client_context') or task_data.get('original_request', {}).get('client_context'):
                        result = analyzer.analyze_youtube_with_context(task_data)
                        print(f"✅ Enhanced analysis completed for {task_id}")
                    else:
                        # Fall back to basic analysis
                        from .analyze_transcript import YouTubeTranscriptAnalyzer
                        basic_analyzer = YouTubeTranscriptAnalyzer(db_path)
                        result = basic_analyzer.analyze_transcript(task_data)
                        print(f"✅ Basic analysis completed for {task_id}")
                    
                    # Update with results
                    cursor.execute("""
                        UPDATE agent_task_queue 
                        SET status = 'completed', result = ?, updated_at = ?
                        WHERE task_id = ?
                    """, (json.dumps(result), datetime.now().isoformat(), task_id))
                    
                except Exception as e:
                    # Update with error
                    cursor.execute("""
                        UPDATE agent_task_queue 
                        SET status = 'failed', result = ?, updated_at = ?
                        WHERE task_id = ?
                    """, (json.dumps({'error': str(e)}), datetime.now().isoformat(), task_id))
                    
                    print(f"❌ Failed enhanced analysis for task {task_id}: {e}")
                
                conn.commit()
            
            import time
            time.sleep(5)  # Poll every 5 seconds
            
        except KeyboardInterrupt:
            print("\n🛑 Enhanced analyzer stopped")
            break
        except Exception as e:
            print(f"❌ Enhanced analyzer error: {e}")
            import time
            time.sleep(10)
    
    conn.close()

if __name__ == "__main__":
    run_enhanced_task_runner()