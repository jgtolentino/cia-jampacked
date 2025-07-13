#!/usr/bin/env python3
"""
Enhanced Real Campaign Loader for JamPacked Creative Intelligence
Loads 32 real campaigns with comprehensive features into the JamPacked platform
"""

import json
import asyncio
import aiohttp
import sqlite3
from pathlib import Path
from datetime import datetime
import subprocess
import sys

class EnhancedCampaignLoader:
    def __init__(self, api_base="http://localhost:8080", local_db="real_campaigns_features.db"):
        self.api_base = api_base
        self.local_db = local_db
        self.extracted_data = None
        
    async def run_feature_extraction(self):
        """Run the comprehensive feature extraction first"""
        print("🔧 Running comprehensive feature extraction...")
        
        extraction_script = Path(__file__).parent / "enhanced-real-campaign-extraction.py"
        
        try:
            result = subprocess.run([sys.executable, str(extraction_script)], 
                                  capture_output=True, text=True, check=True)
            print("✅ Feature extraction completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Feature extraction failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    def load_extracted_data(self):
        """Load the extracted comprehensive dataset"""
        try:
            # Connect to the local database
            conn = sqlite3.connect(self.local_db)
            
            # Load the comprehensive dataset
            query = """
            SELECT * FROM real_campaigns_comprehensive 
            ORDER BY creative_effectiveness_score DESC
            """
            
            self.extracted_data = []
            cursor = conn.execute(query)
            columns = [description[0] for description in cursor.description]
            
            for row in cursor.fetchall():
                campaign_dict = dict(zip(columns, row))
                self.extracted_data.append(campaign_dict)
            
            conn.close()
            
            print(f"✅ Loaded {len(self.extracted_data)} campaigns with comprehensive features")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load extracted data: {e}")
            return False
    
    async def check_platform_health(self):
        """Check if JamPacked platform is running"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base}/health", timeout=5) as response:
                    if response.status == 200:
                        print("✅ JamPacked platform is running")
                        return True
                    else:
                        print(f"⚠️ Platform responded with status {response.status}")
                        return False
        except Exception as e:
            print(f"❌ Platform health check failed: {e}")
            print("💡 Make sure to run: ./deploy.sh")
            return False
    
    async def load_campaign_to_platform(self, session, campaign_data):
        """Load individual campaign to JamPacked platform"""
        try:
            # Prepare API payload
            payload = {
                "campaign": {
                    "id": campaign_data["campaign_id"],
                    "name": campaign_data["name"],
                    "brand": campaign_data["brand"],
                    "industry": campaign_data["industry"],
                    "year": campaign_data["year"],
                    "award_show": campaign_data["award_show"],
                    "award_category": campaign_data["award_category"]
                },
                "features": {
                    "award_recognition": {
                        "award_prestige_score": campaign_data.get("award_prestige_score", 0),
                        "award_potential_score": campaign_data.get("award_potential_score", 0),
                        "award_status_binary": campaign_data.get("award_status_binary", 0)
                    },
                    "csr_analysis": {
                        "csr_presence_binary": campaign_data.get("csr_presence_binary", 0),
                        "csr_message_prominence": campaign_data.get("csr_message_prominence", 0),
                        "csr_authenticity_score": campaign_data.get("csr_authenticity_score", 0),
                        "csr_category": campaign_data.get("csr_category", "None")
                    },
                    "visual_features": {
                        "visual_complexity_score": campaign_data.get("visual_complexity_score", 0),
                        "aesthetic_score": campaign_data.get("aesthetic_score", 0),
                        "memorability_score": campaign_data.get("memorability_score", 0),
                        "brand_asset_visibility": campaign_data.get("brand_asset_visibility", 0)
                    },
                    "text_features": {
                        "message_sentiment": campaign_data.get("message_sentiment", 0),
                        "message_clarity": campaign_data.get("message_clarity", 0),
                        "headline_strength": campaign_data.get("headline_strength", 0)
                    },
                    "innovation": {
                        "innovation_level": campaign_data.get("innovation_level", 0),
                        "technology_integration": campaign_data.get("technology_integration", 0),
                        "creative_distinctiveness": campaign_data.get("creative_distinctiveness", 0)
                    },
                    "cultural_context": {
                        "cultural_relevance_score": campaign_data.get("cultural_relevance_score", 0),
                        "cultural_authenticity": campaign_data.get("cultural_authenticity", 0),
                        "localization_depth": campaign_data.get("localization_depth", "regional")
                    }
                },
                "effectiveness": {
                    "creative_effectiveness_score": campaign_data.get("creative_effectiveness_score", 0),
                    "roi_multiplier": campaign_data.get("roi_multiplier", 1.0),
                    "brand_lift_percentage": campaign_data.get("brand_lift_percentage", 0),
                    "engagement_rate": campaign_data.get("engagement_rate", 0)
                },
                "metadata": {
                    "source": "google_drive_real",
                    "extraction_timestamp": campaign_data.get("extraction_timestamp"),
                    "verified": True
                }
            }
            
            # Send to platform API
            async with session.post(f"{self.api_base}/api/v1/campaigns", json=payload) as response:
                if response.status in [200, 201]:
                    print(f"✅ Loaded: {campaign_data['name']} (CES: {campaign_data.get('creative_effectiveness_score', 0):.1f})")
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ Failed to load {campaign_data['name']}: {error_text}")
                    return False
                    
        except Exception as e:
            print(f"❌ Error loading {campaign_data['name']}: {e}")
            return False
    
    async def load_all_campaigns(self):
        """Load all campaigns to the platform"""
        if not self.extracted_data:
            print("❌ No extracted data available")
            return False
        
        print(f"\n🚀 Loading {len(self.extracted_data)} campaigns to JamPacked platform...")
        
        successful_loads = 0
        failed_loads = 0
        
        async with aiohttp.ClientSession() as session:
            for campaign in self.extracted_data:
                success = await self.load_campaign_to_platform(session, campaign)
                if success:
                    successful_loads += 1
                else:
                    failed_loads += 1
                
                # Small delay to avoid overwhelming the API
                await asyncio.sleep(0.1)
        
        print(f"\n📊 Loading Summary:")
        print(f"   ✅ Successful: {successful_loads}")
        print(f"   ❌ Failed: {failed_loads}")
        print(f"   📈 Success Rate: {(successful_loads/(successful_loads+failed_loads)*100):.1f}%")
        
        return successful_loads > 0
    
    async def create_analysis_dashboard(self):
        """Create analysis dashboards and views"""
        try:
            async with aiohttp.ClientSession() as session:
                # Create award recognition analysis
                award_analysis_payload = {
                    "analysis_type": "award_recognition_impact",
                    "parameters": {
                        "group_by": "award_show",
                        "metrics": ["creative_effectiveness_score", "roi_multiplier", "brand_lift_percentage"]
                    }
                }
                
                async with session.post(f"{self.api_base}/api/v1/analytics/create", 
                                      json=award_analysis_payload) as response:
                    if response.status == 200:
                        print("✅ Created award recognition analysis dashboard")
                
                # Create CSR effectiveness analysis
                csr_analysis_payload = {
                    "analysis_type": "csr_effectiveness_analysis",
                    "parameters": {
                        "compare_groups": ["csr_presence_binary"],
                        "metrics": ["creative_effectiveness_score", "engagement_rate", "brand_lift_percentage"]
                    }
                }
                
                async with session.post(f"{self.api_base}/api/v1/analytics/create", 
                                      json=csr_analysis_payload) as response:
                    if response.status == 200:
                        print("✅ Created CSR effectiveness analysis dashboard")
                
                # Create cultural relevance analysis
                cultural_analysis_payload = {
                    "analysis_type": "cultural_impact_analysis",
                    "parameters": {
                        "group_by": "localization_depth",
                        "metrics": ["cultural_relevance_score", "creative_effectiveness_score"]
                    }
                }
                
                async with session.post(f"{self.api_base}/api/v1/analytics/create", 
                                      json=cultural_analysis_payload) as response:
                    if response.status == 200:
                        print("✅ Created cultural impact analysis dashboard")
                        
        except Exception as e:
            print(f"⚠️ Dashboard creation warning: {e}")
    
    async def verify_loading(self):
        """Verify that campaigns were loaded correctly"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_base}/api/v1/campaigns/count") as response:
                    if response.status == 200:
                        data = await response.json()
                        campaign_count = data.get("count", 0)
                        print(f"✅ Verification: {campaign_count} campaigns loaded in platform")
                        return campaign_count > 0
                    else:
                        print("⚠️ Could not verify campaign count")
                        return False
        except Exception as e:
            print(f"⚠️ Verification warning: {e}")
            return False
    
    def display_summary(self):
        """Display loading summary and next steps"""
        if not self.extracted_data:
            return
        
        # Calculate summary statistics
        total_campaigns = len(self.extracted_data)
        csr_campaigns = len([c for c in self.extracted_data if c.get('csr_presence_binary', 0) == 1])
        high_performers = len([c for c in self.extracted_data if c.get('creative_effectiveness_score', 0) > 75])
        avg_effectiveness = sum(c.get('creative_effectiveness_score', 0) for c in self.extracted_data) / total_campaigns
        
        # Award show breakdown
        award_shows = {}
        for campaign in self.extracted_data:
            show = campaign.get('award_show', 'Unknown')
            award_shows[show] = award_shows.get(show, 0) + 1
        
        print("\n" + "="*80)
        print("📊 REAL CAMPAIGNS LOADED - SUMMARY REPORT")
        print("="*80)
        
        print(f"\n🎯 DATASET OVERVIEW:")
        print(f"   Total Campaigns: {total_campaigns}")
        print(f"   CSR Campaigns: {csr_campaigns} ({csr_campaigns/total_campaigns*100:.1f}%)")
        print(f"   High Performers (CES > 75): {high_performers} ({high_performers/total_campaigns*100:.1f}%)")
        print(f"   Average Effectiveness Score: {avg_effectiveness:.1f}")
        
        print(f"\n🏆 AWARD SHOW DISTRIBUTION:")
        for show, count in sorted(award_shows.items(), key=lambda x: x[1], reverse=True):
            print(f"   {show}: {count} campaigns")
        
        print(f"\n🌟 TOP PERFORMING CAMPAIGNS:")
        top_campaigns = sorted(self.extracted_data, 
                             key=lambda x: x.get('creative_effectiveness_score', 0), reverse=True)[:5]
        for campaign in top_campaigns:
            ces = campaign.get('creative_effectiveness_score', 0)
            print(f"   {ces:.1f} - {campaign['name']} ({campaign['brand']})")
        
        print(f"\n💡 CSR IMPACT INSIGHTS:")
        if csr_campaigns > 0:
            csr_avg = sum(c.get('creative_effectiveness_score', 0) for c in self.extracted_data 
                         if c.get('csr_presence_binary', 0) == 1) / csr_campaigns
            non_csr_avg = sum(c.get('creative_effectiveness_score', 0) for c in self.extracted_data 
                            if c.get('csr_presence_binary', 0) == 0) / (total_campaigns - csr_campaigns)
            print(f"   CSR campaigns avg CES: {csr_avg:.1f}")
            print(f"   Non-CSR campaigns avg CES: {non_csr_avg:.1f}")
            print(f"   CSR Impact: {'+' if csr_avg > non_csr_avg else ''}{csr_avg - non_csr_avg:.1f} points")
        else:
            print("   No CSR campaigns detected in dataset")
        
        print(f"\n🚀 PLATFORM ACCESS:")
        print(f"   API Base: {self.api_base}")
        print(f"   Campaigns API: {self.api_base}/api/v1/campaigns")
        print(f"   Analytics API: {self.api_base}/api/v1/analytics")
        print(f"   Dashboard: http://localhost:3000")
        
        print(f"\n✅ JAMPACKED READY FOR:")
        print(f"   • Creative Effectiveness Analysis")
        print(f"   • Award Recognition Modeling")
        print(f"   • CSR Impact Assessment")
        print(f"   • Cultural Relevance Scoring")
        print(f"   • Innovation Impact Analysis")
        print(f"   • Multi-dimensional Predictions")
    
    async def run_complete_loading(self):
        """Run the complete loading pipeline"""
        print("🚀 ENHANCED REAL CAMPAIGN LOADING PIPELINE")
        print("="*60)
        
        # Step 1: Run feature extraction
        print("\n📊 STEP 1: Feature Extraction")
        if not await self.run_feature_extraction():
            print("❌ Feature extraction failed - aborting")
            return False
        
        # Step 2: Load extracted data
        print("\n📥 STEP 2: Loading Extracted Data")
        if not self.load_extracted_data():
            print("❌ Data loading failed - aborting")
            return False
        
        # Step 3: Check platform health
        print("\n🏥 STEP 3: Platform Health Check")
        if not await self.check_platform_health():
            print("❌ Platform not available - deploy with ./deploy.sh")
            return False
        
        # Step 4: Load campaigns to platform
        print("\n🔄 STEP 4: Loading to Platform")
        if not await self.load_all_campaigns():
            print("❌ Campaign loading failed")
            return False
        
        # Step 5: Create analysis dashboards
        print("\n📈 STEP 5: Creating Analytics Dashboards")
        await self.create_analysis_dashboard()
        
        # Step 6: Verify loading
        print("\n✅ STEP 6: Verification")
        await self.verify_loading()
        
        # Step 7: Display summary
        print("\n📋 STEP 7: Summary Report")
        self.display_summary()
        
        print("\n🎉 LOADING COMPLETE! JamPacked is ready with real campaign data.")
        return True


async def main():
    """Main execution function"""
    loader = EnhancedCampaignLoader()
    success = await loader.run_complete_loading()
    
    if success:
        print("\n🚀 Next steps:")
        print("   1. Access API: curl http://localhost:8080/api/v1/campaigns")
        print("   2. View dashboard: open http://localhost:3000")
        print("   3. Run analysis: python scripts/analyze-campaigns.py")
        print("   4. Generate insights: python scripts/generate-insights.py")
    else:
        print("\n❌ Loading failed. Check logs and try again.")


if __name__ == "__main__":
    asyncio.run(main())
