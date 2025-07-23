"""
TikTok Smart+ Campaign Integration
Automated campaign creation and optimization for TikTok's AI-powered advertising
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp
import json
from datetime import datetime


class CampaignObjective(Enum):
    TRAFFIC = "traffic"
    CONVERSIONS = "conversions"
    APP_INSTALLS = "app_installs"
    VIDEO_VIEWS = "video_views"
    LEAD_GENERATION = "lead_generation"
    CATALOG_SALES = "catalog_sales"


class OptimizationGoal(Enum):
    CLICKS = "clicks"
    CONVERSIONS = "conversions"
    VALUE = "value"
    IMPRESSIONS = "impressions"
    REACH = "reach"


class CreativeType(Enum):
    SINGLE_VIDEO = "single_video"
    CAROUSEL = "carousel"
    COLLECTION = "collection"
    SPARK_ADS = "spark_ads"
    DYNAMIC_SHOWCASE = "dynamic_showcase"


@dataclass
class SmartPlusCampaign:
    """TikTok Smart+ campaign configuration"""
    name: str
    objective: CampaignObjective
    optimization_goal: OptimizationGoal
    daily_budget: float
    target_cpa: Optional[float] = None
    target_roas: Optional[float] = None
    creative_assets: List[Dict] = None
    audience_signals: Dict = None
    
    def to_api_payload(self) -> Dict:
        """Convert to TikTok API format"""
        payload = {
            "campaign_name": self.name,
            "objective": self.objective.value,
            "daily_budget": self.daily_budget * 100,  # Convert to cents
            "optimization_goal": self.optimization_goal.value,
            "smart_optimization": True,
            "campaign_type": "SMART_PLUS"
        }
        
        if self.target_cpa:
            payload["target_cpa"] = self.target_cpa * 100
        
        if self.target_roas:
            payload["target_roas"] = self.target_roas
        
        if self.audience_signals:
            payload["audience_signals"] = self.audience_signals
        
        return payload


class TikTokSmartPlusAPI:
    """TikTok Smart+ API integration"""
    
    def __init__(self, access_token: str, advertiser_id: str):
        self.access_token = access_token
        self.advertiser_id = advertiser_id
        self.base_url = "https://business-api.tiktok.com/open_api/v1.3"
        self.headers = {
            "Access-Token": access_token,
            "Content-Type": "application/json"
        }
    
    async def create_smart_campaign(self, campaign: SmartPlusCampaign) -> Dict:
        """Create Smart+ campaign with full automation"""
        # Create campaign
        campaign_payload = campaign.to_api_payload()
        campaign_payload["advertiser_id"] = self.advertiser_id
        
        async with aiohttp.ClientSession() as session:
            # Create campaign
            campaign_response = await self._make_request(
                session, 
                "POST", 
                "/campaign/create/",
                campaign_payload
            )
            
            if campaign_response.get("code") == 0:
                campaign_id = campaign_response["data"]["campaign_id"]
                
                # Create Smart+ ad group with automation
                adgroup_response = await self._create_smart_adgroup(
                    session, campaign_id, campaign
                )
                
                # Upload and create creatives
                if campaign.creative_assets:
                    creative_response = await self._create_smart_creatives(
                        session, 
                        adgroup_response["data"]["adgroup_id"],
                        campaign.creative_assets
                    )
                
                return {
                    "success": True,
                    "campaign_id": campaign_id,
                    "adgroup_id": adgroup_response["data"]["adgroup_id"],
                    "performance_estimates": self._calculate_performance_estimates(campaign)
                }
            
            return {"success": False, "error": campaign_response}
    
    async def _create_smart_adgroup(self, 
                                   session: aiohttp.ClientSession,
                                   campaign_id: str,
                                   campaign: SmartPlusCampaign) -> Dict:
        """Create Smart+ optimized ad group"""
        adgroup_payload = {
            "advertiser_id": self.advertiser_id,
            "campaign_id": campaign_id,
            "adgroup_name": f"{campaign.name}_SmartAdGroup",
            "placement_type": "PLACEMENT_TYPE_AUTOMATIC",  # Let Smart+ decide
            "budget_mode": "BUDGET_MODE_DAY",
            "budget": campaign.daily_budget * 100,
            "schedule_type": "SCHEDULE_FROM_NOW",
            "optimization_goal": campaign.optimization_goal.value,
            "bid_type": "BID_TYPE_SMART",  # Smart+ bidding
            "deep_bid_type": "DEEP_BID_TYPE_SMART_OPTIMIZE"  # Full automation
        }
        
        # Add audience signals if provided
        if campaign.audience_signals:
            adgroup_payload["targeting"] = self._build_smart_targeting(
                campaign.audience_signals
            )
        
        return await self._make_request(session, "POST", "/adgroup/create/", adgroup_payload)
    
    def _build_smart_targeting(self, audience_signals: Dict) -> Dict:
        """Build Smart+ targeting from audience signals"""
        targeting = {
            "audience_mode": "SMART_MODE",  # Let TikTok optimize
        }
        
        # Add seed audiences if provided
        if "seed_audiences" in audience_signals:
            targeting["include_custom_audience"] = {
                "custom_audience_ids": audience_signals["seed_audiences"]
            }
        
        # Add interest signals
        if "interests" in audience_signals:
            targeting["interest_category_ids"] = audience_signals["interests"]
        
        # Demographics are optional - Smart+ will find best audience
        if "age_range" in audience_signals:
            targeting["age_groups"] = audience_signals["age_range"]
        
        return targeting
    
    async def _create_smart_creatives(self,
                                    session: aiohttp.ClientSession,
                                    adgroup_id: str,
                                    creative_assets: List[Dict]) -> Dict:
        """Create Smart+ optimized creatives"""
        creative_payload = {
            "advertiser_id": self.advertiser_id,
            "adgroup_id": adgroup_id,
            "creatives": [],
            "creative_optimization": "SMART_CREATIVE_ENABLED"
        }
        
        for asset in creative_assets:
            creative = {
                "creative_name": asset.get("name", "Smart+ Creative"),
                "creative_type": CreativeType.DYNAMIC_SHOWCASE.value,
                "dynamic_creative_elements": {
                    "videos": asset.get("videos", []),
                    "images": asset.get("images", []),
                    "texts": asset.get("texts", []),
                    "call_to_actions": asset.get("ctas", ["Shop Now", "Learn More"]),
                    "display_names": asset.get("display_names", [])
                }
            }
            creative_payload["creatives"].append(creative)
        
        return await self._make_request(session, "POST", "/creative/create/", creative_payload)
    
    async def _make_request(self, 
                           session: aiohttp.ClientSession,
                           method: str,
                           endpoint: str,
                           data: Dict) -> Dict:
        """Make API request to TikTok"""
        url = f"{self.base_url}{endpoint}"
        
        async with session.request(method, url, headers=self.headers, json=data) as response:
            return await response.json()
    
    def _calculate_performance_estimates(self, campaign: SmartPlusCampaign) -> Dict:
        """Calculate performance estimates based on Smart+ benchmarks"""
        # Based on 53% ROAS improvement and 50% CPA reduction
        base_estimates = {
            "estimated_daily_impressions": campaign.daily_budget * 1000,
            "estimated_daily_clicks": campaign.daily_budget * 50,
            "estimated_conversions": campaign.daily_budget * 2
        }
        
        # Apply Smart+ improvements
        smart_multipliers = {
            "roas_improvement": 1.53,
            "cpa_reduction": 0.50,
            "reach_expansion": 2.1
        }
        
        return {
            "daily_impressions": int(base_estimates["estimated_daily_impressions"] * 
                                   smart_multipliers["reach_expansion"]),
            "daily_clicks": int(base_estimates["estimated_daily_clicks"] * 1.3),
            "daily_conversions": int(base_estimates["estimated_conversions"] * 
                                   smart_multipliers["roas_improvement"]),
            "estimated_cpa": (campaign.target_cpa or campaign.daily_budget / 2) * 
                           smart_multipliers["cpa_reduction"],
            "estimated_roas": (campaign.target_roas or 3.0) * 
                            smart_multipliers["roas_improvement"]
        }
    
    async def get_campaign_performance(self, campaign_id: str) -> Dict:
        """Get real-time performance data"""
        params = {
            "advertiser_id": self.advertiser_id,
            "campaign_ids": [campaign_id],
            "metrics": [
                "spend", "impressions", "clicks", "conversions",
                "cost_per_conversion", "conversion_rate", "ctr", "cpm"
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            return await self._make_request(
                session, "GET", "/campaign/get/", params
            )
    
    async def optimize_campaign(self, campaign_id: str, performance_data: Dict) -> Dict:
        """Apply Smart+ AI optimizations based on performance"""
        optimization_actions = []
        
        # Check performance thresholds
        current_roas = performance_data.get("roas", 0)
        current_cpa = performance_data.get("cpa", float('inf'))
        
        if current_roas < 2.0:
            optimization_actions.append({
                "action": "expand_audience",
                "reason": "ROAS below target, expanding audience for better matches"
            })
        
        if current_cpa > performance_data.get("target_cpa", 50):
            optimization_actions.append({
                "action": "adjust_bidding",
                "reason": "CPA above target, optimizing bid strategy"
            })
        
        # Apply optimizations
        for action in optimization_actions:
            if action["action"] == "expand_audience":
                await self._expand_audience_reach(campaign_id)
            elif action["action"] == "adjust_bidding":
                await self._optimize_bidding(campaign_id, performance_data)
        
        return {
            "optimizations_applied": len(optimization_actions),
            "actions": optimization_actions,
            "projected_improvement": self._calculate_optimization_impact(optimization_actions)
        }
    
    async def _expand_audience_reach(self, campaign_id: str) -> Dict:
        """Expand audience using Smart+ lookalike modeling"""
        # In production, this would call TikTok API to expand audience
        return {"audience_expansion": "enabled", "expansion_level": "balanced"}
    
    async def _optimize_bidding(self, campaign_id: str, performance_data: Dict) -> Dict:
        """Optimize bidding strategy using Smart+ AI"""
        # In production, this would adjust bid strategies
        return {"bid_optimization": "applied", "strategy": "target_cpa_adjusted"}
    
    def _calculate_optimization_impact(self, actions: List[Dict]) -> Dict:
        """Calculate expected impact of optimizations"""
        impact = {
            "expected_roas_lift": 0,
            "expected_cpa_reduction": 0
        }
        
        for action in actions:
            if action["action"] == "expand_audience":
                impact["expected_roas_lift"] += 0.15
            elif action["action"] == "adjust_bidding":
                impact["expected_cpa_reduction"] += 0.10
        
        return impact


class SmartPlusCreativeOptimizer:
    """Optimize creatives using TikTok's AI"""
    
    def __init__(self, api_client: TikTokSmartPlusAPI):
        self.api_client = api_client
    
    async def generate_creative_variations(self, base_creative: Dict) -> List[Dict]:
        """Generate AI-powered creative variations"""
        variations = []
        
        # Text variations
        base_text = base_creative.get("text", "")
        text_variations = [
            base_text,
            f"🔥 {base_text}",
            f"{base_text} - Limited Time!",
            f"Don't Miss: {base_text}",
            f"{base_text} ✨"
        ]
        
        # CTA variations
        cta_variations = [
            "Shop Now",
            "Get Yours",
            "Learn More",
            "Sign Up",
            "Discover More"
        ]
        
        # Generate combinations
        for text in text_variations[:3]:
            for cta in cta_variations[:2]:
                variation = base_creative.copy()
                variation["text"] = text
                variation["cta"] = cta
                variation["variation_id"] = f"var_{len(variations)}"
                variations.append(variation)
        
        return variations
    
    async def test_creative_performance(self, variations: List[Dict]) -> Dict:
        """A/B test creative variations using Smart+ optimization"""
        # In production, this would create split tests
        test_results = {
            "test_id": f"creative_test_{datetime.now().timestamp()}",
            "variations": len(variations),
            "estimated_test_duration": "3-5 days",
            "optimization_method": "multi_armed_bandit"
        }
        
        return test_results


class SmartPlusCampaignBuilder:
    """Builder for creating optimized Smart+ campaigns"""
    
    @staticmethod
    def ecommerce_campaign(product_catalog_id: str, budget: float) -> SmartPlusCampaign:
        """Pre-built e-commerce campaign template"""
        return SmartPlusCampaign(
            name="Smart+ E-commerce Campaign",
            objective=CampaignObjective.CATALOG_SALES,
            optimization_goal=OptimizationGoal.VALUE,
            daily_budget=budget,
            target_roas=3.5,
            audience_signals={
                "product_categories": ["apparel", "accessories"],
                "purchase_intent": "high"
            }
        )
    
    @staticmethod
    def lead_gen_campaign(industry: str, budget: float) -> SmartPlusCampaign:
        """Pre-built lead generation campaign template"""
        return SmartPlusCampaign(
            name=f"Smart+ {industry} Lead Gen",
            objective=CampaignObjective.LEAD_GENERATION,
            optimization_goal=OptimizationGoal.CONVERSIONS,
            daily_budget=budget,
            target_cpa=25.0,
            audience_signals={
                "interests": [industry],
                "intent_signals": ["in_market"]
            }
        )
    
    @staticmethod
    def app_install_campaign(app_id: str, budget: float) -> SmartPlusCampaign:
        """Pre-built app install campaign template"""
        return SmartPlusCampaign(
            name="Smart+ App Install Campaign",
            objective=CampaignObjective.APP_INSTALLS,
            optimization_goal=OptimizationGoal.CONVERSIONS,
            daily_budget=budget,
            target_cpa=2.0,
            audience_signals={
                "app_behaviors": ["gaming", "shopping"],
                "lookalike_source": "existing_users"
            }
        )


# Example usage
async def main():
    """Example Smart+ campaign creation"""
    # Initialize API client
    api = TikTokSmartPlusAPI(
        access_token="your_access_token",
        advertiser_id="your_advertiser_id"
    )
    
    # Create e-commerce campaign
    campaign = SmartPlusCampaignBuilder.ecommerce_campaign(
        product_catalog_id="catalog_123",
        budget=1000.0
    )
    
    # Add creative assets
    campaign.creative_assets = [{
        "name": "Product Showcase",
        "videos": ["video_id_1", "video_id_2"],
        "texts": [
            "Discover our latest collection",
            "Transform your style today",
            "Exclusive online deals"
        ],
        "ctas": ["Shop Now", "View Collection"]
    }]
    
    # Create campaign
    result = await api.create_smart_campaign(campaign)
    
    print("Smart+ Campaign Created:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())