"""
Microsoft 7-Element Prompt Structure for Advertising AI Agents
Implements the gold standard framework for AI advertising prompts
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import json


class PromptElement(Enum):
    ROLE = "role"
    TASK_GOAL = "task_goal"
    SOURCE = "source"
    CONTEXT = "context"
    TONE = "tone"
    TARGET_AUDIENCE = "target_audience"
    INCLUSIVE_MODIFIERS = "inclusive_modifiers"


@dataclass
class AdvertisingPrompt:
    """Structured advertising prompt following Microsoft's 7-element framework"""
    role: str
    task_goal: str
    source: str
    context: str
    tone: str
    target_audience: str
    inclusive_modifiers: str
    
    def to_prompt(self) -> str:
        """Generate optimized prompt from elements"""
        return (
            f"You are {self.role} (role), "
            f"{self.task_goal} (task goal), "
            f"using {self.source} (source) "
            f"for {self.context} (context), "
            f"sound {self.tone} (tone) "
            f"to {self.target_audience} (target audience). "
            f"{self.inclusive_modifiers} (inclusive modifiers)."
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API usage"""
        return {
            "role": self.role,
            "task_goal": self.task_goal,
            "source": self.source,
            "context": self.context,
            "tone": self.tone,
            "target_audience": self.target_audience,
            "inclusive_modifiers": self.inclusive_modifiers
        }


class PromptTemplateLibrary:
    """Library of pre-built advertising prompt templates"""
    
    @staticmethod
    def search_ad_template() -> AdvertisingPrompt:
        return AdvertisingPrompt(
            role="an ad copy writer for paid search",
            task_goal="create 3 unique versions of ad copy, 85 characters or less",
            source="the latest research on best performing ads",
            context="selling children's shoes",
            tone="empathetic and inspiring",
            target_audience="parents' hopes for their purchase",
            inclusive_modifiers="Ensure it is inclusive of all children, regardless of abilities"
        )
    
    @staticmethod
    def social_media_template() -> AdvertisingPrompt:
        return AdvertisingPrompt(
            role="a social media marketing expert",
            task_goal="create engaging social media posts with CTAs",
            source="trending topics and viral content patterns",
            context="promoting sustainable fashion brand",
            tone="authentic and environmentally conscious",
            target_audience="Gen Z consumers aged 18-25",
            inclusive_modifiers="Include diverse body types and cultural backgrounds"
        )
    
    @staticmethod
    def display_ad_template() -> AdvertisingPrompt:
        return AdvertisingPrompt(
            role="a visual advertising strategist",
            task_goal="design display ad concepts with compelling headlines",
            source="high-performing display ad benchmarks",
            context="luxury travel experiences",
            tone="aspirational yet accessible",
            target_audience="affluent millennials planning vacations",
            inclusive_modifiers="Feature diverse travelers and accessible destinations"
        )
    
    @staticmethod
    def video_ad_template() -> AdvertisingPrompt:
        return AdvertisingPrompt(
            role="a video advertising creative director",
            task_goal="script a 15-second video ad with strong hook",
            source="TikTok and YouTube Shorts best practices",
            context="fitness app for beginners",
            tone="motivational and supportive",
            target_audience="people starting their fitness journey",
            inclusive_modifiers="Show all fitness levels and body types succeeding"
        )
    
    @staticmethod
    def email_campaign_template() -> AdvertisingPrompt:
        return AdvertisingPrompt(
            role="an email marketing specialist",
            task_goal="write personalized email subject lines and preview text",
            source="email marketing performance data",
            context="e-commerce seasonal sale",
            tone="urgent yet friendly",
            target_audience="loyal customers and subscribers",
            inclusive_modifiers="Use gender-neutral language and diverse examples"
        )


class PromptOptimizer:
    """Optimize prompts based on performance metrics"""
    
    def __init__(self):
        self.performance_history = []
    
    def track_performance(self, prompt: AdvertisingPrompt, metrics: Dict):
        """Track prompt performance for optimization"""
        self.performance_history.append({
            "prompt": prompt.to_dict(),
            "metrics": metrics,
            "timestamp": self._get_timestamp()
        })
    
    def optimize_prompt(self, base_prompt: AdvertisingPrompt) -> AdvertisingPrompt:
        """Optimize prompt based on historical performance"""
        # Analyze performance history to identify best-performing elements
        if not self.performance_history:
            return base_prompt
        
        # Find best performing tone and target audience combinations
        best_performance = max(
            self.performance_history,
            key=lambda x: x["metrics"].get("conversion_rate", 0)
        )
        
        # Apply learnings to optimize prompt
        optimized = AdvertisingPrompt(
            role=base_prompt.role,
            task_goal=base_prompt.task_goal,
            source=base_prompt.source,
            context=base_prompt.context,
            tone=best_performance["prompt"].get("tone", base_prompt.tone),
            target_audience=best_performance["prompt"].get("target_audience", base_prompt.target_audience),
            inclusive_modifiers=base_prompt.inclusive_modifiers
        )
        
        return optimized
    
    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()


class DynamicPromptEngine:
    """Real-time prompt adjustment based on performance data"""
    
    def __init__(self):
        self.optimizer = PromptOptimizer()
        self.active_prompts = {}
    
    def create_dynamic_prompt(self, 
                            campaign_id: str,
                            base_template: AdvertisingPrompt,
                            performance_threshold: float = 0.02) -> str:
        """Create dynamically optimized prompt"""
        # Check if we have performance data for this campaign
        if campaign_id in self.active_prompts:
            current_metrics = self._get_current_metrics(campaign_id)
            
            # Optimize if performance is below threshold
            if current_metrics.get("ctr", 0) < performance_threshold:
                optimized = self.optimizer.optimize_prompt(base_template)
                self.active_prompts[campaign_id] = optimized
                return optimized.to_prompt()
        else:
            self.active_prompts[campaign_id] = base_template
        
        return self.active_prompts[campaign_id].to_prompt()
    
    def _get_current_metrics(self, campaign_id: str) -> Dict:
        """Simulate getting real-time metrics"""
        # In production, this would connect to analytics APIs
        return {
            "ctr": 0.015,
            "conversion_rate": 0.023,
            "roas": 3.2
        }


# Example usage
if __name__ == "__main__":
    # Create a search ad prompt
    search_prompt = PromptTemplateLibrary.search_ad_template()
    print("Search Ad Prompt:")
    print(search_prompt.to_prompt())
    print("\n")
    
    # Create dynamic prompt engine
    engine = DynamicPromptEngine()
    
    # Generate optimized prompt for campaign
    optimized = engine.create_dynamic_prompt(
        campaign_id="CAMP_001",
        base_template=search_prompt
    )
    print("Optimized Prompt:")
    print(optimized)