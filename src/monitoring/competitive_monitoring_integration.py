"""
Competitive Intelligence Tools Integration
Integrates Brand24, Brandwatch, and other monitoring platforms
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import aiohttp
import json
from datetime import datetime, timedelta
import numpy as np


class MonitoringPlatform(Enum):
    BRAND24 = "brand24"
    BRANDWATCH = "brandwatch"
    MENTION = "mention"
    TALKWALKER = "talkwalker"
    SPRINKLR = "sprinklr"
    HOOTSUITE = "hootsuite"


class DataSource(Enum):
    SOCIAL_MEDIA = "social_media"
    NEWS = "news"
    BLOGS = "blogs"
    FORUMS = "forums"
    REVIEWS = "reviews"
    VIDEO = "video"
    PODCASTS = "podcasts"


class SentimentType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass
class CompetitiveMention:
    """Single competitive mention/insight"""
    platform: MonitoringPlatform
    source: DataSource
    timestamp: datetime
    brand: str
    competitor: Optional[str]
    content: str
    sentiment: SentimentType
    reach: int
    engagement: int
    url: Optional[str]
    author_influence: float
    
    @property
    def impact_score(self) -> float:
        """Calculate impact score based on reach and engagement"""
        return (self.reach * 0.3 + self.engagement * 0.7) * self.author_influence


@dataclass
class CompetitiveInsight:
    """Aggregated competitive insight"""
    insight_type: str
    brands: List[str]
    trend: str
    magnitude: float
    confidence: float
    supporting_mentions: List[CompetitiveMention]
    recommendations: List[str]


class Brand24Integration:
    """Brand24 monitoring platform integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.brand24.com/v3"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def fetch_mentions(self,
                           project_id: str,
                           start_date: datetime,
                           end_date: datetime) -> List[CompetitiveMention]:
        """Fetch mentions from Brand24"""
        # In production, use actual Brand24 API
        # Simulated data for example
        mentions = []
        
        brands = ["OurBrand", "Competitor1", "Competitor2", "Competitor3"]
        sources = list(DataSource)
        sentiments = list(SentimentType)
        
        for _ in range(50):  # Generate 50 sample mentions
            mention = CompetitiveMention(
                platform=MonitoringPlatform.BRAND24,
                source=np.random.choice(sources),
                timestamp=start_date + timedelta(
                    seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
                ),
                brand=np.random.choice(brands),
                competitor=np.random.choice(brands) if np.random.random() > 0.5 else None,
                content=self._generate_sample_content(),
                sentiment=np.random.choice(sentiments),
                reach=np.random.randint(100, 10000),
                engagement=np.random.randint(10, 1000),
                url=f"https://example.com/mention_{np.random.randint(1000, 9999)}",
                author_influence=np.random.random()
            )
            mentions.append(mention)
        
        return mentions
    
    def _generate_sample_content(self) -> str:
        """Generate sample mention content"""
        templates = [
            "Just tried {brand} and it's amazing! Much better than {competitor}",
            "Disappointed with {brand} service today. Switching to {competitor}",
            "{brand} launched new feature - game changer for the industry",
            "Why is {brand} pricing so high compared to {competitor}?",
            "Love the new campaign from {brand}! So creative!"
        ]
        return np.random.choice(templates)
    
    async def get_sentiment_analysis(self, project_id: str) -> Dict:
        """Get sentiment analysis for project"""
        # Simulated sentiment data
        return {
            "positive": 45,
            "negative": 20,
            "neutral": 35,
            "sentiment_score": 0.25,  # -1 to 1 scale
            "trend": "improving"
        }
    
    async def get_reach_metrics(self, project_id: str) -> Dict:
        """Get reach and influence metrics"""
        return {
            "total_reach": 1500000,
            "unique_authors": 3500,
            "avg_influence_score": 0.65,
            "top_influencers": [
                {"name": "TechInfluencer", "reach": 50000, "mentions": 5},
                {"name": "MarketingGuru", "reach": 35000, "mentions": 3}
            ]
        }


class BrandwatchIntegration:
    """Brandwatch (now Cision) integration"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.brandwatch.com/v2"
    
    async def fetch_mentions(self,
                           query_id: str,
                           start_date: datetime,
                           end_date: datetime) -> List[CompetitiveMention]:
        """Fetch mentions from Brandwatch"""
        # Similar structure to Brand24, different API
        mentions = []
        
        # Brandwatch specific data enrichment
        for _ in range(30):
            mention = CompetitiveMention(
                platform=MonitoringPlatform.BRANDWATCH,
                source=np.random.choice(list(DataSource)),
                timestamp=start_date + timedelta(hours=np.random.randint(0, 24)),
                brand="OurBrand",
                competitor=np.random.choice(["Competitor1", "Competitor2"]),
                content="Brandwatch detected mention with advanced NLP",
                sentiment=np.random.choice(list(SentimentType)),
                reach=np.random.randint(500, 50000),
                engagement=np.random.randint(50, 5000),
                url=None,
                author_influence=np.random.uniform(0.3, 1.0)
            )
            mentions.append(mention)
        
        return mentions
    
    async def get_competitive_analysis(self, 
                                     brands: List[str]) -> Dict:
        """Get competitive analysis across brands"""
        analysis = {}
        
        for brand in brands:
            analysis[brand] = {
                "share_of_voice": np.random.uniform(15, 35),
                "sentiment_score": np.random.uniform(-0.5, 0.8),
                "engagement_rate": np.random.uniform(1, 5),
                "growth_rate": np.random.uniform(-10, 25)
            }
        
        return analysis
    
    async def get_trending_topics(self, industry: str) -> List[Dict]:
        """Get trending topics in industry"""
        topics = [
            {"topic": "AI Innovation", "volume": 15000, "growth": 45},
            {"topic": "Sustainability", "volume": 12000, "growth": 30},
            {"topic": "Customer Experience", "volume": 8000, "growth": 20}
        ]
        return topics


class CompetitiveIntelligenceHub:
    """Central hub for all competitive monitoring platforms"""
    
    def __init__(self):
        self.platforms: Dict[MonitoringPlatform, Any] = {}
        self.cached_mentions: List[CompetitiveMention] = []
        self.insights_cache: List[CompetitiveInsight] = []
        self.last_sync = None
    
    def add_platform(self, 
                    platform: MonitoringPlatform,
                    credentials: Dict):
        """Add monitoring platform"""
        if platform == MonitoringPlatform.BRAND24:
            self.platforms[platform] = Brand24Integration(
                api_key=credentials.get("api_key")
            )
        elif platform == MonitoringPlatform.BRANDWATCH:
            self.platforms[platform] = BrandwatchIntegration(
                api_key=credentials.get("api_key"),
                api_secret=credentials.get("api_secret")
            )
        # Add other platforms as needed
    
    async def sync_all_platforms(self,
                               start_date: datetime,
                               end_date: datetime,
                               queries: Dict[MonitoringPlatform, str]) -> List[CompetitiveMention]:
        """Sync data from all platforms"""
        all_mentions = []
        
        # Fetch from each platform concurrently
        tasks = []
        for platform, integration in self.platforms.items():
            if platform in queries:
                task = integration.fetch_mentions(
                    queries[platform], start_date, end_date
                )
                tasks.append(task)
        
        # Gather all results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_mentions.extend(result)
            else:
                print(f"Error fetching mentions: {result}")
        
        self.cached_mentions = all_mentions
        self.last_sync = datetime.now()
        
        return all_mentions
    
    def analyze_competitive_landscape(self) -> Dict:
        """Analyze overall competitive landscape"""
        if not self.cached_mentions:
            return {"error": "No data available"}
        
        landscape = {
            "total_mentions": len(self.cached_mentions),
            "brands": {},
            "sentiment_distribution": {},
            "source_distribution": {},
            "temporal_analysis": {}
        }
        
        # Analyze by brand
        brand_mentions = {}
        for mention in self.cached_mentions:
            if mention.brand not in brand_mentions:
                brand_mentions[mention.brand] = []
            brand_mentions[mention.brand].append(mention)
        
        for brand, mentions in brand_mentions.items():
            landscape["brands"][brand] = {
                "mention_count": len(mentions),
                "avg_reach": np.mean([m.reach for m in mentions]),
                "avg_engagement": np.mean([m.engagement for m in mentions]),
                "sentiment": self._calculate_sentiment_distribution(mentions),
                "share_of_voice": len(mentions) / len(self.cached_mentions) * 100
            }
        
        # Overall sentiment distribution
        sentiment_counts = {}
        for mention in self.cached_mentions:
            sentiment_counts[mention.sentiment.value] = sentiment_counts.get(
                mention.sentiment.value, 0
            ) + 1
        
        landscape["sentiment_distribution"] = {
            k: v / len(self.cached_mentions) * 100 
            for k, v in sentiment_counts.items()
        }
        
        return landscape
    
    def _calculate_sentiment_distribution(self, 
                                        mentions: List[CompetitiveMention]) -> Dict:
        """Calculate sentiment distribution for mentions"""
        sentiment_counts = {s.value: 0 for s in SentimentType}
        
        for mention in mentions:
            sentiment_counts[mention.sentiment.value] += 1
        
        total = len(mentions)
        return {
            k: (v / total * 100) if total > 0 else 0 
            for k, v in sentiment_counts.items()
        }
    
    def generate_competitive_insights(self) -> List[CompetitiveInsight]:
        """Generate actionable competitive insights"""
        insights = []
        
        if not self.cached_mentions:
            return insights
        
        # Insight 1: Sentiment shifts
        sentiment_insight = self._analyze_sentiment_shifts()
        if sentiment_insight:
            insights.append(sentiment_insight)
        
        # Insight 2: Emerging competitors
        emerging_insight = self._analyze_emerging_competitors()
        if emerging_insight:
            insights.append(emerging_insight)
        
        # Insight 3: Campaign effectiveness
        campaign_insight = self._analyze_campaign_effectiveness()
        if campaign_insight:
            insights.append(campaign_insight)
        
        # Insight 4: Crisis detection
        crisis_insight = self._detect_potential_crisis()
        if crisis_insight:
            insights.append(crisis_insight)
        
        self.insights_cache = insights
        return insights
    
    def _analyze_sentiment_shifts(self) -> Optional[CompetitiveInsight]:
        """Analyze significant sentiment shifts"""
        # Group mentions by brand and time
        brand_sentiments = {}
        
        for mention in self.cached_mentions:
            if mention.brand not in brand_sentiments:
                brand_sentiments[mention.brand] = []
            
            sentiment_value = 1 if mention.sentiment == SentimentType.POSITIVE else (
                -1 if mention.sentiment == SentimentType.NEGATIVE else 0
            )
            brand_sentiments[mention.brand].append({
                "time": mention.timestamp,
                "sentiment": sentiment_value
            })
        
        # Find significant shifts
        for brand, sentiments in brand_sentiments.items():
            if len(sentiments) < 10:
                continue
            
            # Calculate rolling average
            sentiments_sorted = sorted(sentiments, key=lambda x: x["time"])
            recent_avg = np.mean([s["sentiment"] for s in sentiments_sorted[-5:]])
            overall_avg = np.mean([s["sentiment"] for s in sentiments_sorted])
            
            shift = recent_avg - overall_avg
            
            if abs(shift) > 0.3:  # Significant shift
                return CompetitiveInsight(
                    insight_type="sentiment_shift",
                    brands=[brand],
                    trend="improving" if shift > 0 else "declining",
                    magnitude=abs(shift),
                    confidence=0.8,
                    supporting_mentions=self.cached_mentions[-5:],
                    recommendations=[
                        f"{'Capitalize on positive momentum' if shift > 0 else 'Address negative sentiment'} for {brand}",
                        "Monitor competitor responses to sentiment changes",
                        "Adjust messaging strategy based on sentiment drivers"
                    ]
                )
        
        return None
    
    def _analyze_emerging_competitors(self) -> Optional[CompetitiveInsight]:
        """Identify emerging competitors"""
        # Track mention growth by competitor
        competitor_growth = {}
        
        for mention in self.cached_mentions:
            if mention.competitor:
                if mention.competitor not in competitor_growth:
                    competitor_growth[mention.competitor] = 0
                competitor_growth[mention.competitor] += 1
        
        # Find fast-growing competitors
        if competitor_growth:
            avg_mentions = np.mean(list(competitor_growth.values()))
            
            emerging = [
                comp for comp, count in competitor_growth.items() 
                if count > avg_mentions * 1.5
            ]
            
            if emerging:
                return CompetitiveInsight(
                    insight_type="emerging_competitors",
                    brands=emerging,
                    trend="growing",
                    magnitude=max(competitor_growth.values()) / avg_mentions,
                    confidence=0.7,
                    supporting_mentions=[
                        m for m in self.cached_mentions 
                        if m.competitor in emerging
                    ][:10],
                    recommendations=[
                        f"Monitor {', '.join(emerging)} for competitive threats",
                        "Analyze their successful strategies",
                        "Strengthen differentiation against emerging players"
                    ]
                )
        
        return None
    
    def _analyze_campaign_effectiveness(self) -> Optional[CompetitiveInsight]:
        """Analyze campaign effectiveness based on mentions"""
        # Detect campaign-related mentions
        campaign_mentions = [
            m for m in self.cached_mentions 
            if any(keyword in m.content.lower() 
                  for keyword in ["campaign", "launch", "new", "announcing"])
        ]
        
        if len(campaign_mentions) > 5:
            # Calculate engagement metrics
            avg_engagement = np.mean([m.engagement for m in campaign_mentions])
            avg_reach = np.mean([m.reach for m in campaign_mentions])
            
            # Compare to non-campaign mentions
            regular_mentions = [
                m for m in self.cached_mentions 
                if m not in campaign_mentions
            ]
            
            if regular_mentions:
                regular_engagement = np.mean([m.engagement for m in regular_mentions])
                lift = (avg_engagement - regular_engagement) / regular_engagement * 100
                
                if lift > 20:  # Significant lift
                    return CompetitiveInsight(
                        insight_type="campaign_effectiveness",
                        brands=list(set(m.brand for m in campaign_mentions)),
                        trend="high_performance",
                        magnitude=lift,
                        confidence=0.85,
                        supporting_mentions=campaign_mentions[:5],
                        recommendations=[
                            f"Campaign driving {lift:.0f}% engagement lift",
                            "Analyze creative elements for replication",
                            "Consider similar campaign approach"
                        ]
                    )
        
        return None
    
    def _detect_potential_crisis(self) -> Optional[CompetitiveInsight]:
        """Detect potential PR crisis situations"""
        # Look for negative mention spikes
        negative_mentions = [
            m for m in self.cached_mentions 
            if m.sentiment == SentimentType.NEGATIVE
        ]
        
        if len(negative_mentions) > len(self.cached_mentions) * 0.4:
            # High proportion of negative mentions
            affected_brands = list(set(m.brand for m in negative_mentions))
            
            # Find common themes
            crisis_keywords = ["scandal", "boycott", "fail", "disaster", "controversy"]
            crisis_mentions = [
                m for m in negative_mentions
                if any(kw in m.content.lower() for kw in crisis_keywords)
            ]
            
            if crisis_mentions:
                return CompetitiveInsight(
                    insight_type="crisis_detection",
                    brands=affected_brands,
                    trend="crisis_emerging",
                    magnitude=len(crisis_mentions) / len(self.cached_mentions),
                    confidence=0.9,
                    supporting_mentions=crisis_mentions[:5],
                    recommendations=[
                        "Immediate crisis monitoring required",
                        "Prepare response strategy",
                        "Monitor competitor crisis handling for insights"
                    ]
                )
        
        return None
    
    def get_real_time_alerts(self) -> List[Dict]:
        """Get real-time alerts based on monitoring"""
        alerts = []
        
        # Check for sudden mention spikes
        recent_mentions = [
            m for m in self.cached_mentions 
            if m.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        if len(recent_mentions) > len(self.cached_mentions) * 0.2:
            alerts.append({
                "type": "mention_spike",
                "severity": "high",
                "message": f"Sudden spike in mentions: {len(recent_mentions)} in last hour",
                "action": "investigate_immediately"
            })
        
        # Check for influencer mentions
        high_influence_mentions = [
            m for m in self.cached_mentions 
            if m.author_influence > 0.8
        ]
        
        for mention in high_influence_mentions:
            alerts.append({
                "type": "influencer_mention",
                "severity": "medium",
                "message": f"High-influence mention about {mention.brand}",
                "action": "engage_if_appropriate",
                "mention": mention
            })
        
        return alerts


class CompetitiveReportGenerator:
    """Generate comprehensive competitive intelligence reports"""
    
    def __init__(self, hub: CompetitiveIntelligenceHub):
        self.hub = hub
    
    def generate_executive_report(self) -> Dict:
        """Generate executive-level competitive report"""
        landscape = self.hub.analyze_competitive_landscape()
        insights = self.hub.generate_competitive_insights()
        
        report = {
            "report_date": datetime.now().isoformat(),
            "summary": {
                "total_mentions": landscape.get("total_mentions", 0),
                "sentiment_score": self._calculate_overall_sentiment(),
                "top_competitor": self._identify_top_competitor(landscape),
                "key_insights": len(insights)
            },
            "competitive_position": self._analyze_competitive_position(landscape),
            "opportunities": self._identify_opportunities(insights),
            "threats": self._identify_threats(insights),
            "recommendations": self._generate_recommendations(landscape, insights)
        }
        
        return report
    
    def _calculate_overall_sentiment(self) -> float:
        """Calculate overall sentiment score"""
        if not self.hub.cached_mentions:
            return 0.0
        
        sentiment_values = []
        for mention in self.hub.cached_mentions:
            if mention.sentiment == SentimentType.POSITIVE:
                sentiment_values.append(1)
            elif mention.sentiment == SentimentType.NEGATIVE:
                sentiment_values.append(-1)
            else:
                sentiment_values.append(0)
        
        return np.mean(sentiment_values)
    
    def _identify_top_competitor(self, landscape: Dict) -> str:
        """Identify top competitor by share of voice"""
        brands = landscape.get("brands", {})
        if not brands:
            return "Unknown"
        
        # Exclude our brand and find top competitor
        competitors = {
            k: v for k, v in brands.items() 
            if k != "OurBrand"
        }
        
        if competitors:
            return max(
                competitors.items(), 
                key=lambda x: x[1].get("share_of_voice", 0)
            )[0]
        
        return "None identified"
    
    def _analyze_competitive_position(self, landscape: Dict) -> Dict:
        """Analyze our competitive position"""
        our_metrics = landscape.get("brands", {}).get("OurBrand", {})
        
        position = {
            "share_of_voice_rank": 1,  # Calculate actual rank
            "sentiment_rank": 1,
            "engagement_rank": 1,
            "overall_position": "strong"  # strong/moderate/weak
        }
        
        # Compare to competitors
        if our_metrics:
            our_sov = our_metrics.get("share_of_voice", 0)
            if our_sov > 30:
                position["overall_position"] = "strong"
            elif our_sov > 20:
                position["overall_position"] = "moderate"
            else:
                position["overall_position"] = "weak"
        
        return position
    
    def _identify_opportunities(self, insights: List[CompetitiveInsight]) -> List[Dict]:
        """Identify strategic opportunities"""
        opportunities = []
        
        for insight in insights:
            if insight.insight_type == "sentiment_shift" and insight.trend == "declining":
                # Competitor weakness
                opportunities.append({
                    "type": "competitor_weakness",
                    "description": f"{insight.brands[0]} experiencing negative sentiment",
                    "action": "Highlight our strengths in this area"
                })
            
            elif insight.insight_type == "campaign_effectiveness":
                opportunities.append({
                    "type": "successful_strategy",
                    "description": "High-performing campaign format identified",
                    "action": "Adapt successful elements to our campaigns"
                })
        
        return opportunities
    
    def _identify_threats(self, insights: List[CompetitiveInsight]) -> List[Dict]:
        """Identify competitive threats"""
        threats = []
        
        for insight in insights:
            if insight.insight_type == "emerging_competitors":
                threats.append({
                    "type": "new_competition",
                    "description": f"Emerging competitors: {', '.join(insight.brands)}",
                    "severity": "medium",
                    "action": "Strengthen market position"
                })
            
            elif insight.insight_type == "crisis_detection":
                if "OurBrand" in insight.brands:
                    threats.append({
                        "type": "reputation_risk",
                        "description": "Potential crisis situation detected",
                        "severity": "high",
                        "action": "Activate crisis management protocol"
                    })
        
        return threats
    
    def _generate_recommendations(self, 
                                landscape: Dict,
                                insights: List[CompetitiveInsight]) -> List[str]:
        """Generate strategic recommendations"""
        recommendations = []
        
        # Based on position
        position = self._analyze_competitive_position(landscape)
        if position["overall_position"] == "weak":
            recommendations.append("Increase brand visibility through targeted campaigns")
            recommendations.append("Analyze competitor strategies for improvement areas")
        
        # Based on insights
        for insight in insights:
            recommendations.extend(insight.recommendations[:2])
        
        # General recommendations
        recommendations.append("Continue monitoring competitive landscape weekly")
        recommendations.append("Set up automated alerts for significant changes")
        
        return list(set(recommendations))[:5]  # Top 5 unique recommendations


# Example usage
async def main():
    """Example competitive monitoring usage"""
    # Create hub
    hub = CompetitiveIntelligenceHub()
    
    # Add platforms
    hub.add_platform(
        MonitoringPlatform.BRAND24,
        {"api_key": "brand24_api_key"}
    )
    hub.add_platform(
        MonitoringPlatform.BRANDWATCH,
        {"api_key": "bw_key", "api_secret": "bw_secret"}
    )
    
    # Sync data
    queries = {
        MonitoringPlatform.BRAND24: "project_123",
        MonitoringPlatform.BRANDWATCH: "query_456"
    }
    
    mentions = await hub.sync_all_platforms(
        datetime.now() - timedelta(days=7),
        datetime.now(),
        queries
    )
    
    print(f"Synced {len(mentions)} mentions")
    
    # Analyze landscape
    landscape = hub.analyze_competitive_landscape()
    print(f"\nCompetitive Landscape:")
    print(f"Total mentions: {landscape['total_mentions']}")
    print(f"Brands tracked: {len(landscape['brands'])}")
    
    # Generate insights
    insights = hub.generate_competitive_insights()
    print(f"\nGenerated {len(insights)} insights:")
    for insight in insights:
        print(f"- {insight.insight_type}: {insight.brands} ({insight.trend})")
    
    # Generate report
    report_gen = CompetitiveReportGenerator(hub)
    report = report_gen.generate_executive_report()
    
    print(f"\nExecutive Report:")
    print(f"Overall sentiment: {report['summary']['sentiment_score']:.2f}")
    print(f"Top competitor: {report['summary']['top_competitor']}")
    print(f"Opportunities: {len(report['opportunities'])}")
    print(f"Threats: {len(report['threats'])}")
    
    # Get alerts
    alerts = hub.get_real_time_alerts()
    print(f"\nReal-time alerts: {len(alerts)}")


if __name__ == "__main__":
    asyncio.run(main())