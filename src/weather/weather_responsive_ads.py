"""
Weather-Responsive Advertising System
Integrates weather data to trigger contextual advertising campaigns
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta
import aiohttp


class WeatherCondition(Enum):
    SUNNY = "sunny"
    RAINY = "rainy"
    SNOWY = "snowy"
    CLOUDY = "cloudy"
    WINDY = "windy"
    HOT = "hot"
    COLD = "cold"
    MILD = "mild"


class AdTriggerType(Enum):
    TEMPERATURE_BASED = "temperature"
    CONDITION_BASED = "condition"
    FORECAST_BASED = "forecast"
    COMBINATION = "combination"


@dataclass
class WeatherTrigger:
    """Define weather conditions that trigger specific ads"""
    name: str
    trigger_type: AdTriggerType
    conditions: Dict
    ad_creative_id: str
    boost_percentage: float
    
    def should_trigger(self, weather_data: Dict) -> bool:
        """Check if weather conditions match trigger criteria"""
        if self.trigger_type == AdTriggerType.TEMPERATURE_BASED:
            temp = weather_data.get("temperature", 0)
            min_temp = self.conditions.get("min_temperature", -100)
            max_temp = self.conditions.get("max_temperature", 100)
            return min_temp <= temp <= max_temp
            
        elif self.trigger_type == AdTriggerType.CONDITION_BASED:
            current_condition = weather_data.get("condition", "").lower()
            target_conditions = [c.lower() for c in self.conditions.get("conditions", [])]
            return current_condition in target_conditions
            
        elif self.trigger_type == AdTriggerType.FORECAST_BASED:
            forecast = weather_data.get("forecast", [])
            forecast_condition = self.conditions.get("forecast_condition")
            days_ahead = self.conditions.get("days_ahead", 1)
            
            if len(forecast) > days_ahead:
                return forecast[days_ahead].get("condition") == forecast_condition
                
        elif self.trigger_type == AdTriggerType.COMBINATION:
            # Check multiple conditions
            temp_check = True
            condition_check = True
            
            if "temperature" in self.conditions:
                temp = weather_data.get("temperature", 0)
                temp_range = self.conditions["temperature"]
                temp_check = temp_range["min"] <= temp <= temp_range["max"]
            
            if "conditions" in self.conditions:
                current = weather_data.get("condition", "").lower()
                condition_check = current in self.conditions["conditions"]
            
            return temp_check and condition_check
            
        return False


class WeatherDataProvider:
    """Interface for weather data providers"""
    
    async def get_current_weather(self, location: Dict) -> Dict:
        """Get current weather for location"""
        # Simulated weather data - in production, integrate with weather API
        return {
            "temperature": 75,
            "condition": "sunny",
            "humidity": 65,
            "wind_speed": 10,
            "uv_index": 7,
            "precipitation": 0,
            "forecast": [
                {"day": 1, "condition": "sunny", "temp_high": 78, "temp_low": 65},
                {"day": 2, "condition": "cloudy", "temp_high": 72, "temp_low": 60},
                {"day": 3, "condition": "rainy", "temp_high": 68, "temp_low": 55}
            ]
        }
    
    async def get_weather_forecast(self, location: Dict, days: int = 3) -> List[Dict]:
        """Get weather forecast"""
        weather = await self.get_current_weather(location)
        return weather.get("forecast", [])[:days]


class WeatherAdsAPI:
    """WeatherAds.io style API integration"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://api.weatherads.io/v1"
        self.weather_provider = WeatherDataProvider()
    
    async def create_weather_campaign(self, campaign_config: Dict) -> Dict:
        """Create weather-triggered campaign"""
        return {
            "campaign_id": f"weather_{datetime.now().timestamp()}",
            "status": "active",
            "triggers": campaign_config.get("triggers", []),
            "created_at": datetime.now().isoformat()
        }
    
    async def get_triggered_ads(self, location: Dict) -> List[Dict]:
        """Get ads triggered by current weather conditions"""
        weather_data = await self.weather_provider.get_current_weather(location)
        triggered_ads = []
        
        # Check all registered triggers
        for trigger in self._get_active_triggers():
            if trigger.should_trigger(weather_data):
                triggered_ads.append({
                    "ad_id": trigger.ad_creative_id,
                    "trigger_name": trigger.name,
                    "boost_percentage": trigger.boost_percentage,
                    "weather_context": weather_data
                })
        
        return triggered_ads
    
    def _get_active_triggers(self) -> List[WeatherTrigger]:
        """Get all active weather triggers"""
        # In production, this would fetch from database
        return WeatherTriggerLibrary.get_all_triggers()


class WeatherTriggerLibrary:
    """Pre-built weather trigger templates"""
    
    @staticmethod
    def sunny_weather_trigger() -> WeatherTrigger:
        """Trigger for sunny weather - great for outdoor products"""
        return WeatherTrigger(
            name="sunny_day_outdoor",
            trigger_type=AdTriggerType.COMBINATION,
            conditions={
                "temperature": {"min": 70, "max": 95},
                "conditions": ["sunny", "clear", "partly cloudy"]
            },
            ad_creative_id="outdoor_summer_creative",
            boost_percentage=65.6  # Based on Stella Artois results
        )
    
    @staticmethod
    def rainy_weather_trigger() -> WeatherTrigger:
        """Trigger for rainy weather - indoor activities, delivery"""
        return WeatherTrigger(
            name="rainy_day_indoor",
            trigger_type=AdTriggerType.CONDITION_BASED,
            conditions={"conditions": ["rainy", "stormy", "drizzle"]},
            ad_creative_id="indoor_comfort_creative",
            boost_percentage=45.0
        )
    
    @staticmethod
    def cold_weather_trigger() -> WeatherTrigger:
        """Trigger for cold weather - winter products"""
        return WeatherTrigger(
            name="cold_weather_comfort",
            trigger_type=AdTriggerType.TEMPERATURE_BASED,
            conditions={"min_temperature": -50, "max_temperature": 45},
            ad_creative_id="winter_warmth_creative",
            boost_percentage=80.0
        )
    
    @staticmethod
    def hot_weather_trigger() -> WeatherTrigger:
        """Trigger for hot weather - cooling products"""
        return WeatherTrigger(
            name="hot_weather_cooling",
            trigger_type=AdTriggerType.TEMPERATURE_BASED,
            conditions={"min_temperature": 85, "max_temperature": 120},
            ad_creative_id="summer_cooling_creative",
            boost_percentage=600.0  # Based on Bravissimo swimwear results
        )
    
    @staticmethod
    def weekend_sunny_trigger() -> WeatherTrigger:
        """Weekend + sunny weather combination"""
        return WeatherTrigger(
            name="weekend_outdoor_perfect",
            trigger_type=AdTriggerType.COMBINATION,
            conditions={
                "temperature": {"min": 65, "max": 85},
                "conditions": ["sunny", "clear"],
                "day_of_week": ["saturday", "sunday"]
            },
            ad_creative_id="weekend_adventure_creative",
            boost_percentage=120.0
        )
    
    @staticmethod
    def get_all_triggers() -> List[WeatherTrigger]:
        """Get all available triggers"""
        return [
            WeatherTriggerLibrary.sunny_weather_trigger(),
            WeatherTriggerLibrary.rainy_weather_trigger(),
            WeatherTriggerLibrary.cold_weather_trigger(),
            WeatherTriggerLibrary.hot_weather_trigger(),
            WeatherTriggerLibrary.weekend_sunny_trigger()
        ]


class WeatherResponseOptimizer:
    """Optimize ad campaigns based on weather patterns"""
    
    def __init__(self):
        self.performance_data = []
        self.weather_api = WeatherAdsAPI()
    
    async def optimize_campaign(self, 
                               campaign_id: str,
                               locations: List[Dict]) -> Dict:
        """Optimize campaign based on weather patterns across locations"""
        optimization_results = {
            "campaign_id": campaign_id,
            "optimizations": [],
            "projected_improvement": 0
        }
        
        for location in locations:
            # Get weather data and triggered ads
            triggered_ads = await self.weather_api.get_triggered_ads(location)
            
            if triggered_ads:
                optimization = {
                    "location": location,
                    "triggered_ads": triggered_ads,
                    "recommended_budget_shift": self._calculate_budget_shift(triggered_ads),
                    "timing_recommendation": self._get_timing_recommendation(location)
                }
                optimization_results["optimizations"].append(optimization)
        
        # Calculate overall projected improvement
        if optimization_results["optimizations"]:
            avg_boost = sum(
                opt["triggered_ads"][0]["boost_percentage"] 
                for opt in optimization_results["optimizations"]
            ) / len(optimization_results["optimizations"])
            optimization_results["projected_improvement"] = avg_boost
        
        return optimization_results
    
    def _calculate_budget_shift(self, triggered_ads: List[Dict]) -> Dict:
        """Calculate recommended budget shifts based on triggers"""
        if not triggered_ads:
            return {"shift_percentage": 0}
        
        # Use highest boost percentage from triggered ads
        max_boost = max(ad["boost_percentage"] for ad in triggered_ads)
        
        return {
            "shift_percentage": min(max_boost / 10, 50),  # Cap at 50% shift
            "reasoning": f"Weather conditions support {max_boost}% performance boost"
        }
    
    def _get_timing_recommendation(self, location: Dict) -> Dict:
        """Get timing recommendations based on weather patterns"""
        # In production, analyze historical weather patterns
        return {
            "best_hours": [10, 11, 12, 13, 14],  # Peak sunny hours
            "best_days": ["saturday", "sunday"],  # Weekend preference
            "avoid_hours": [18, 19, 20, 21]  # Evening hours
        }


class WeatherCampaignManager:
    """Manage weather-responsive advertising campaigns"""
    
    def __init__(self):
        self.optimizer = WeatherResponseOptimizer()
        self.active_campaigns = {}
    
    async def create_weather_campaign(self,
                                    product_category: str,
                                    base_budget: float,
                                    target_locations: List[Dict]) -> Dict:
        """Create complete weather-responsive campaign"""
        
        # Select appropriate triggers based on product category
        triggers = self._select_triggers_for_category(product_category)
        
        campaign = {
            "id": f"weather_campaign_{datetime.now().timestamp()}",
            "product_category": product_category,
            "base_budget": base_budget,
            "target_locations": target_locations,
            "triggers": triggers,
            "status": "active",
            "created_at": datetime.now().isoformat()
        }
        
        self.active_campaigns[campaign["id"]] = campaign
        
        # Get initial optimization recommendations
        optimization = await self.optimizer.optimize_campaign(
            campaign["id"],
            target_locations
        )
        
        campaign["optimization"] = optimization
        
        return campaign
    
    def _select_triggers_for_category(self, category: str) -> List[Dict]:
        """Select appropriate weather triggers for product category"""
        category_triggers = {
            "outdoor_equipment": [
                WeatherTriggerLibrary.sunny_weather_trigger(),
                WeatherTriggerLibrary.weekend_sunny_trigger()
            ],
            "swimwear": [
                WeatherTriggerLibrary.hot_weather_trigger(),
                WeatherTriggerLibrary.sunny_weather_trigger()
            ],
            "winter_clothing": [
                WeatherTriggerLibrary.cold_weather_trigger()
            ],
            "food_delivery": [
                WeatherTriggerLibrary.rainy_weather_trigger(),
                WeatherTriggerLibrary.cold_weather_trigger()
            ],
            "beverages": [
                WeatherTriggerLibrary.hot_weather_trigger(),
                WeatherTriggerLibrary.sunny_weather_trigger()
            ]
        }
        
        triggers = category_triggers.get(category, [WeatherTriggerLibrary.sunny_weather_trigger()])
        
        return [
            {
                "name": t.name,
                "type": t.trigger_type.value,
                "conditions": t.conditions,
                "boost": t.boost_percentage
            }
            for t in triggers
        ]


# Example usage
async def main():
    """Example weather-responsive campaign setup"""
    manager = WeatherCampaignManager()
    
    # Create campaign for swimwear
    campaign = await manager.create_weather_campaign(
        product_category="swimwear",
        base_budget=10000,
        target_locations=[
            {"city": "Miami", "state": "FL", "country": "US"},
            {"city": "Los Angeles", "state": "CA", "country": "US"},
            {"city": "Phoenix", "state": "AZ", "country": "US"}
        ]
    )
    
    print("Weather-Responsive Campaign Created:")
    print(json.dumps(campaign, indent=2))


if __name__ == "__main__":
    asyncio.run(main())