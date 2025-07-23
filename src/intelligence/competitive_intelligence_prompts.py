"""
Comprehensive Prompt Library for Competitive Intelligence
Structured prompts for market analysis, competitor monitoring, and strategic insights
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime


class IntelligenceType(Enum):
    MARKET_ANALYSIS = "market_analysis"
    COMPETITOR_MONITORING = "competitor_monitoring"
    CREATIVE_BENCHMARKING = "creative_benchmarking"
    CRISIS_RESPONSE = "crisis_response"
    TREND_IDENTIFICATION = "trend_identification"
    PRICING_INTELLIGENCE = "pricing_intelligence"
    PRODUCT_ANALYSIS = "product_analysis"
    CUSTOMER_SENTIMENT = "customer_sentiment"


class AnalysisDepth(Enum):
    SURFACE = "surface"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    EXECUTIVE = "executive"


@dataclass
class CompetitiveIntelligencePrompt:
    """Structured prompt for competitive intelligence"""
    name: str
    intelligence_type: IntelligenceType
    analysis_depth: AnalysisDepth
    prompt_template: str
    required_inputs: List[str]
    ethical_constraints: List[str]
    output_format: str
    
    def generate(self, inputs: Dict[str, Any]) -> str:
        """Generate prompt with specific inputs"""
        # Validate required inputs
        for required in self.required_inputs:
            if required not in inputs:
                raise ValueError(f"Missing required input: {required}")
        
        # Fill template with inputs
        prompt = self.prompt_template
        for key, value in inputs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        
        # Add ethical constraints
        if self.ethical_constraints:
            prompt += "\n\nETHICAL CONSTRAINTS:\n"
            for constraint in self.ethical_constraints:
                prompt += f"- {constraint}\n"
        
        # Add output format
        prompt += f"\n\nOUTPUT FORMAT: {self.output_format}"
        
        return prompt


class CompetitiveIntelligenceLibrary:
    """Comprehensive library of competitive intelligence prompts"""
    
    @staticmethod
    def market_analysis_prompts() -> List[CompetitiveIntelligencePrompt]:
        """Market analysis prompt templates"""
        return [
            CompetitiveIntelligencePrompt(
                name="market_size_estimation",
                intelligence_type=IntelligenceType.MARKET_ANALYSIS,
                analysis_depth=AnalysisDepth.COMPREHENSIVE,
                prompt_template="""
Analyze the market size and growth potential for {industry} in {region}.

Consider the following factors:
1. Current market size in revenue and unit volume
2. Historical growth rates (past 5 years)
3. Key market drivers and inhibitors
4. Demographic and economic factors
5. Technology adoption rates
6. Regulatory environment impact

Focus on {specific_segment} if applicable.
Time frame: {time_frame}

Provide data sources and confidence levels for each estimate.
""",
                required_inputs=["industry", "region", "specific_segment", "time_frame"],
                ethical_constraints=[
                    "Use only publicly available information",
                    "Cite all data sources",
                    "Distinguish between estimates and confirmed data"
                ],
                output_format="Structured report with executive summary, detailed analysis, and data appendix"
            ),
            
            CompetitiveIntelligencePrompt(
                name="market_share_analysis",
                intelligence_type=IntelligenceType.MARKET_ANALYSIS,
                analysis_depth=AnalysisDepth.DETAILED,
                prompt_template="""
Calculate market share distribution for top competitors in {industry}.

Analyze:
1. Revenue-based market share
2. Unit/volume-based market share
3. Geographic distribution of market share
4. Market share trends over {time_period}
5. Factors driving market share changes

Companies to analyze: {competitor_list}
Primary metric: {primary_metric}
""",
                required_inputs=["industry", "competitor_list", "time_period", "primary_metric"],
                ethical_constraints=[
                    "Use only public financial reports and industry analyses",
                    "Note any estimates or assumptions clearly"
                ],
                output_format="Market share table with trend analysis and visual charts"
            )
        ]
    
    @staticmethod
    def competitor_monitoring_prompts() -> List[CompetitiveIntelligencePrompt]:
        """Competitor monitoring prompt templates"""
        return [
            CompetitiveIntelligencePrompt(
                name="competitor_strategy_analysis",
                intelligence_type=IntelligenceType.COMPETITOR_MONITORING,
                analysis_depth=AnalysisDepth.COMPREHENSIVE,
                prompt_template="""
Analyze {competitor_name}'s current business strategy and recent strategic moves.

Examine:
1. Recent product launches and announcements
2. Marketing campaigns and messaging changes
3. Pricing strategy shifts
4. Distribution channel changes
5. Partnership and acquisition activity
6. Executive statements and earnings calls insights
7. Patent filings and R&D focus areas

Time frame: {analysis_period}
Industry context: {industry}

Identify strategic patterns and likely future moves.
""",
                required_inputs=["competitor_name", "analysis_period", "industry"],
                ethical_constraints=[
                    "Use only public information",
                    "No attempts to access confidential data",
                    "Respect intellectual property",
                    "Focus on business strategy, not personal information"
                ],
                output_format="Strategic analysis report with SWOT, key findings, and predictive insights"
            ),
            
            CompetitiveIntelligencePrompt(
                name="competitive_positioning_map",
                intelligence_type=IntelligenceType.COMPETITOR_MONITORING,
                analysis_depth=AnalysisDepth.DETAILED,
                prompt_template="""
Create a competitive positioning analysis for {company} versus key competitors.

Dimensions to analyze:
1. Price positioning: {price_range}
2. Quality/feature positioning
3. Target customer segments
4. Brand perception and values
5. Distribution strategy
6. Innovation level

Competitors: {competitor_list}
Primary differentiators: {key_differentiators}
""",
                required_inputs=["company", "competitor_list", "price_range", "key_differentiators"],
                ethical_constraints=[
                    "Base analysis on publicly available information",
                    "Use customer reviews and public feedback ethically"
                ],
                output_format="Positioning matrix with detailed competitive advantages/disadvantages"
            )
        ]
    
    @staticmethod
    def creative_benchmarking_prompts() -> List[CompetitiveIntelligencePrompt]:
        """Creative benchmarking prompt templates"""
        return [
            CompetitiveIntelligencePrompt(
                name="advertising_effectiveness_benchmark",
                intelligence_type=IntelligenceType.CREATIVE_BENCHMARKING,
                analysis_depth=AnalysisDepth.DETAILED,
                prompt_template="""
Benchmark advertising creative effectiveness for {brand} against top competitors.

Analyze:
1. Creative themes and messaging strategies
2. Emotional engagement scores (if available)
3. Channel-specific creative performance
4. A/B test insights (from public case studies)
5. Award recognition and industry acclaim
6. Social media engagement metrics
7. Estimated media spend efficiency

Campaigns to analyze: {campaign_period}
Channels: {channels}
Competitive set: {competitors}

Focus on {specific_metrics} as primary success indicators.
""",
                required_inputs=["brand", "campaign_period", "channels", "competitors", "specific_metrics"],
                ethical_constraints=[
                    "Use only publicly visible creative assets",
                    "Respect copyright and creative ownership",
                    "Focus on strategic insights, not copying"
                ],
                output_format="Creative effectiveness scorecard with best practices and recommendations"
            ),
            
            CompetitiveIntelligencePrompt(
                name="content_strategy_analysis",
                intelligence_type=IntelligenceType.CREATIVE_BENCHMARKING,
                analysis_depth=AnalysisDepth.SURFACE,
                prompt_template="""
Quick analysis of {competitor}'s content strategy across {platforms}.

Review:
1. Content types and formats used
2. Publishing frequency and timing
3. Engagement rates by content type
4. Key themes and topics
5. Influencer partnerships
6. User-generated content usage

Time frame: {time_frame}
""",
                required_inputs=["competitor", "platforms", "time_frame"],
                ethical_constraints=[
                    "Analyze only public social media and content",
                    "No scraping of private or protected content"
                ],
                output_format="Content strategy summary with top performing content examples"
            )
        ]
    
    @staticmethod
    def crisis_response_prompts() -> List[CompetitiveIntelligencePrompt]:
        """Crisis response prompt templates"""
        return [
            CompetitiveIntelligencePrompt(
                name="competitive_threat_assessment",
                intelligence_type=IntelligenceType.CRISIS_RESPONSE,
                analysis_depth=AnalysisDepth.EXECUTIVE,
                prompt_template="""
URGENT: Assess competitive threat from {threat_source} regarding {threat_type}.

Immediate analysis needed:
1. Severity and credibility of threat
2. Potential impact on our {affected_area}
3. Competitor's likely next moves
4. Our vulnerable points
5. Recommended defensive actions
6. Opportunities to counter

Context: {situation_context}
Time sensitivity: {urgency_level}
""",
                required_inputs=["threat_source", "threat_type", "affected_area", "situation_context", "urgency_level"],
                ethical_constraints=[
                    "Maintain ethical standards even under pressure",
                    "No retaliatory actions that could escalate",
                    "Focus on defensive strategies"
                ],
                output_format="Executive brief with immediate actions and risk mitigation plan"
            ),
            
            CompetitiveIntelligencePrompt(
                name="market_innovation_analysis",
                intelligence_type=IntelligenceType.CRISIS_RESPONSE,
                analysis_depth=AnalysisDepth.COMPREHENSIVE,
                prompt_template="""
Analyze market innovation potential from {innovateor} in {market_segment}.

Assess:
1. Innovation timeline and phases
2. Customer segments most at risk
3. Our competitive advantages that remain relevant
4. Required strategic pivots
5. Partnership or acquisition opportunities
6. Defensive positioning strategies

Innovation type: {innovation_type}
Our current position: {market_position}
""",
                required_inputs=["innovateor", "market_segment", "innovation_type", "market_position"],
                ethical_constraints=[
                    "Focus on market dynamics, not undermining competitors",
                    "Propose ethical competitive responses only"
                ],
                output_format="Innovation response playbook with scenario planning"
            )
        ]
    
    @staticmethod
    def trend_identification_prompts() -> List[CompetitiveIntelligencePrompt]:
        """Trend identification prompt templates"""
        return [
            CompetitiveIntelligencePrompt(
                name="emerging_trends_analysis",
                intelligence_type=IntelligenceType.TREND_IDENTIFICATION,
                analysis_depth=AnalysisDepth.COMPREHENSIVE,
                prompt_template="""
Identify emerging trends in {industry} that could impact competitive dynamics.

Analyze:
1. Technology trends affecting the industry
2. Consumer behavior shifts
3. Regulatory changes on the horizon
4. New business models emerging
5. Cross-industry influences
6. Generational preference changes

Geographic focus: {geography}
Time horizon: {time_horizon}
Probability threshold: {probability_threshold}

Rank trends by potential impact and likelihood.
""",
                required_inputs=["industry", "geography", "time_horizon", "probability_threshold"],
                ethical_constraints=[
                    "Base predictions on observable data",
                    "Acknowledge uncertainty levels",
                    "Avoid speculation presented as fact"
                ],
                output_format="Trend report with impact/probability matrix and strategic implications"
            )
        ]
    
    @staticmethod
    def pricing_intelligence_prompts() -> List[CompetitiveIntelligencePrompt]:
        """Pricing intelligence prompt templates"""
        return [
            CompetitiveIntelligencePrompt(
                name="competitive_pricing_analysis",
                intelligence_type=IntelligenceType.PRICING_INTELLIGENCE,
                analysis_depth=AnalysisDepth.DETAILED,
                prompt_template="""
Analyze pricing strategies for {product_category} across key competitors.

Examine:
1. Base pricing and discount structures
2. Promotional frequency and depth
3. Bundle strategies
4. Geographic pricing variations
5. Channel-specific pricing
6. Price-to-value positioning

Competitors: {competitor_list}
Time period: {analysis_period}
Key features for comparison: {feature_list}
""",
                required_inputs=["product_category", "competitor_list", "analysis_period", "feature_list"],
                ethical_constraints=[
                    "Use only publicly advertised prices",
                    "No attempts to access private pricing",
                    "Respect confidential B2B pricing"
                ],
                output_format="Pricing comparison matrix with strategic insights and recommendations"
            )
        ]
    
    @staticmethod
    def generate_custom_prompt(
        intelligence_type: IntelligenceType,
        specific_objective: str,
        constraints: List[str]
    ) -> CompetitiveIntelligencePrompt:
        """Generate custom competitive intelligence prompt"""
        return CompetitiveIntelligencePrompt(
            name="custom_intelligence_prompt",
            intelligence_type=intelligence_type,
            analysis_depth=AnalysisDepth.DETAILED,
            prompt_template=f"""
Custom Competitive Intelligence Analysis

Objective: {specific_objective}

Conduct analysis following best practices for {intelligence_type.value} while respecting all ethical and legal boundaries.

Provide actionable insights with supporting evidence.
""",
            required_inputs=["specific_objective"],
            ethical_constraints=constraints + [
                "Adhere to all applicable laws and regulations",
                "Respect competitor intellectual property",
                "Use only ethical information gathering methods"
            ],
            output_format="Structured analysis with findings, evidence, and recommendations"
        )


class CompetitiveIntelligenceOrchestrator:
    """Orchestrate complex competitive intelligence workflows"""
    
    def __init__(self):
        self.prompt_library = CompetitiveIntelligenceLibrary()
        self.analysis_history = []
    
    def create_intelligence_workflow(self,
                                   company: str,
                                   objectives: List[str],
                                   depth: AnalysisDepth) -> List[CompetitiveIntelligencePrompt]:
        """Create workflow of prompts for comprehensive intelligence gathering"""
        workflow_prompts = []
        
        # Map objectives to appropriate prompts
        for objective in objectives:
            if "market" in objective.lower():
                workflow_prompts.extend(self.prompt_library.market_analysis_prompts())
            elif "competitor" in objective.lower() or "competitive" in objective.lower():
                workflow_prompts.extend(self.prompt_library.competitor_monitoring_prompts())
            elif "creative" in objective.lower() or "advertising" in objective.lower():
                workflow_prompts.extend(self.prompt_library.creative_benchmarking_prompts())
            elif "crisis" in objective.lower() or "threat" in objective.lower():
                workflow_prompts.extend(self.prompt_library.crisis_response_prompts())
            elif "trend" in objective.lower():
                workflow_prompts.extend(self.prompt_library.trend_identification_prompts())
            elif "pricing" in objective.lower() or "price" in objective.lower():
                workflow_prompts.extend(self.prompt_library.pricing_intelligence_prompts())
        
        # Filter by requested depth
        workflow_prompts = [p for p in workflow_prompts if p.analysis_depth == depth or depth == AnalysisDepth.COMPREHENSIVE]
        
        # Record workflow
        self.analysis_history.append({
            "timestamp": datetime.now(),
            "company": company,
            "objectives": objectives,
            "prompts_generated": len(workflow_prompts)
        })
        
        return workflow_prompts
    
    def create_competitive_dashboard_prompts(self, 
                                           company: str,
                                           competitors: List[str],
                                           update_frequency: str) -> Dict[str, CompetitiveIntelligencePrompt]:
        """Create prompts for ongoing competitive dashboard"""
        dashboard_prompts = {
            "market_position": CompetitiveIntelligenceLibrary.generate_custom_prompt(
                IntelligenceType.MARKET_ANALYSIS,
                f"Track {company}'s market position versus {', '.join(competitors)} with {update_frequency} updates",
                ["Use only public data sources", "Maintain consistent methodology"]
            ),
            
            "creative_performance": CompetitiveIntelligenceLibrary.generate_custom_prompt(
                IntelligenceType.CREATIVE_BENCHMARKING,
                f"Monitor creative performance metrics for {company} and competitors {update_frequency}",
                ["Track only public campaign metrics", "Focus on strategic insights"]
            ),
            
            "strategic_moves": CompetitiveIntelligenceLibrary.generate_custom_prompt(
                IntelligenceType.COMPETITOR_MONITORING,
                f"Alert on significant strategic moves by {', '.join(competitors)}",
                ["Monitor only public announcements", "Verify from multiple sources"]
            ),
            
            "trend_alerts": CompetitiveIntelligenceLibrary.generate_custom_prompt(
                IntelligenceType.TREND_IDENTIFICATION,
                f"Identify emerging trends that could impact {company}'s competitive position",
                ["Focus on early indicators", "Validate trend significance"]
            )
        }
        
        return dashboard_prompts


class EthicalIntelligenceGuidelines:
    """Ensure all competitive intelligence follows ethical guidelines"""
    
    @staticmethod
    def get_ethical_boundaries() -> List[str]:
        """Core ethical boundaries for competitive intelligence"""
        return [
            "Never attempt to access confidential or proprietary information",
            "Do not impersonate or misrepresent identity",
            "Respect all intellectual property rights",
            "Use only publicly available information sources",
            "Do not engage in any form of corporate espionage",
            "Maintain competitor employee privacy",
            "Avoid any actions that could be considered unfair competition",
            "Document all information sources transparently",
            "Do not spread misinformation about competitors",
            "Focus on learning and improving, not undermining"
        ]
    
    @staticmethod
    def validate_prompt_ethics(prompt: CompetitiveIntelligencePrompt) -> bool:
        """Validate that prompt follows ethical guidelines"""
        ethical_keywords = [
            "public", "ethical", "respect", "legal", "transparent",
            "available", "documented", "fair"
        ]
        
        unethical_keywords = [
            "hack", "steal", "confidential", "secret", "insider",
            "spy", "infiltrate", "sabotage"
        ]
        
        prompt_text = prompt.prompt_template.lower()
        
        # Check for ethical keywords
        ethical_score = sum(1 for keyword in ethical_keywords if keyword in prompt_text)
        
        # Check for unethical keywords
        unethical_score = sum(1 for keyword in unethical_keywords if keyword in prompt_text)
        
        # Ensure ethical constraints are present
        has_constraints = len(prompt.ethical_constraints) >= 2
        
        return ethical_score > 0 and unethical_score == 0 and has_constraints


# Example usage
def main():
    """Example competitive intelligence prompt usage"""
    # Create orchestrator
    orchestrator = CompetitiveIntelligenceOrchestrator()
    
    # Create intelligence workflow
    workflow = orchestrator.create_intelligence_workflow(
        company="Our Company",
        objectives=[
            "Understand market positioning",
            "Monitor competitor pricing strategies",
            "Track creative performance"
        ],
        depth=AnalysisDepth.DETAILED
    )
    
    print("Competitive Intelligence Workflow Created:")
    print(f"Number of prompts: {len(workflow)}")
    
    # Example: Generate a specific prompt
    if workflow:
        example_prompt = workflow[0]
        inputs = {
            "industry": "E-commerce Fashion",
            "region": "North America",
            "specific_segment": "Sustainable Fashion",
            "time_frame": "2024-2026"
        }
        
        try:
            generated = example_prompt.generate(inputs)
            print(f"\nExample Generated Prompt:\n{generated}")
        except ValueError as e:
            print(f"Error: {e}")
    
    # Create competitive dashboard
    dashboard = orchestrator.create_competitive_dashboard_prompts(
        company="Our Brand",
        competitors=["Competitor A", "Competitor B", "Competitor C"],
        update_frequency="weekly"
    )
    
    print(f"\nDashboard Prompts Created: {len(dashboard)}")
    
    # Validate ethics
    guidelines = EthicalIntelligenceGuidelines()
    print(f"\nEthical Guidelines:")
    for guideline in guidelines.get_ethical_boundaries()[:5]:
        print(f"- {guideline}")


if __name__ == "__main__":
    main()