"""
Cross-Platform Measurement Dashboard
Unified campaign tracking and analysis across all advertising platforms
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json


class Platform(Enum):
    GOOGLE = "google"
    META = "meta"
    TIKTOK = "tiktok"
    AMAZON = "amazon"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    SNAPCHAT = "snapchat"
    PINTEREST = "pinterest"
    PROGRAMMATIC = "programmatic"


class MetricType(Enum):
    IMPRESSIONS = "impressions"
    CLICKS = "clicks"
    CONVERSIONS = "conversions"
    SPEND = "spend"
    REVENUE = "revenue"
    CTR = "ctr"
    CPC = "cpc"
    CPA = "cpa"
    ROAS = "roas"
    CPM = "cpm"


class DashboardView(Enum):
    EXECUTIVE = "executive"
    PERFORMANCE = "performance"
    ATTRIBUTION = "attribution"
    COMPETITIVE = "competitive"
    PREDICTIVE = "predictive"
    REALTIME = "realtime"


@dataclass
class CampaignMetrics:
    """Standardized metrics across platforms"""
    platform: Platform
    campaign_id: str
    campaign_name: str
    timestamp: datetime
    impressions: int
    clicks: int
    conversions: int
    spend: float
    revenue: float
    
    @property
    def ctr(self) -> float:
        return self.clicks / self.impressions if self.impressions > 0 else 0
    
    @property
    def cpc(self) -> float:
        return self.spend / self.clicks if self.clicks > 0 else 0
    
    @property
    def cpa(self) -> float:
        return self.spend / self.conversions if self.conversions > 0 else 0
    
    @property
    def roas(self) -> float:
        return self.revenue / self.spend if self.spend > 0 else 0
    
    @property
    def cpm(self) -> float:
        return (self.spend / self.impressions) * 1000 if self.impressions > 0 else 0


class PlatformConnector:
    """Base class for platform API connectors"""
    
    def __init__(self, platform: Platform, credentials: Dict):
        self.platform = platform
        self.credentials = credentials
        self.rate_limit_remaining = 1000
        self.last_sync = None
    
    async def fetch_metrics(self, 
                          start_date: datetime,
                          end_date: datetime,
                          campaign_ids: Optional[List[str]] = None) -> List[CampaignMetrics]:
        """Fetch metrics from platform API"""
        # This would be implemented per platform
        # Simulated data for example
        metrics = []
        
        num_campaigns = 5 if campaign_ids is None else len(campaign_ids)
        
        for i in range(num_campaigns):
            campaign_id = campaign_ids[i] if campaign_ids else f"{self.platform.value}_campaign_{i}"
            
            # Generate realistic metrics
            base_impressions = np.random.randint(10000, 100000)
            ctr = np.random.uniform(0.01, 0.05)
            cvr = np.random.uniform(0.01, 0.03)
            cpc = np.random.uniform(0.5, 3.0)
            
            clicks = int(base_impressions * ctr)
            conversions = int(clicks * cvr)
            spend = clicks * cpc
            revenue = conversions * np.random.uniform(50, 200)
            
            metrics.append(CampaignMetrics(
                platform=self.platform,
                campaign_id=campaign_id,
                campaign_name=f"{self.platform.value.title()} Campaign {i+1}",
                timestamp=datetime.now(),
                impressions=base_impressions,
                clicks=clicks,
                conversions=conversions,
                spend=spend,
                revenue=revenue
            ))
        
        self.last_sync = datetime.now()
        return metrics
    
    async def validate_connection(self) -> bool:
        """Validate API connection"""
        # In production, this would test API credentials
        return True
    
    def get_rate_limit_status(self) -> Dict:
        """Get current rate limit status"""
        return {
            "remaining": self.rate_limit_remaining,
            "reset_time": datetime.now() + timedelta(hours=1),
            "limit": 1000
        }


class GoogleAdsConnector(PlatformConnector):
    """Google Ads specific connector"""
    
    def __init__(self, credentials: Dict):
        super().__init__(Platform.GOOGLE, credentials)
    
    async def fetch_metrics(self,
                          start_date: datetime,
                          end_date: datetime,
                          campaign_ids: Optional[List[str]] = None) -> List[CampaignMetrics]:
        """Fetch Google Ads metrics"""
        # In production, use Google Ads API
        metrics = await super().fetch_metrics(start_date, end_date, campaign_ids)
        
        # Add Google-specific processing
        for metric in metrics:
            # Simulate Quality Score impact
            quality_score = np.random.uniform(5, 10)
            metric.spend *= (10 / quality_score)  # Better QS = lower costs
        
        return metrics


class MetaAdsConnector(PlatformConnector):
    """Meta (Facebook/Instagram) Ads connector"""
    
    def __init__(self, credentials: Dict):
        super().__init__(Platform.META, credentials)
    
    async def fetch_metrics(self,
                          start_date: datetime,
                          end_date: datetime,
                          campaign_ids: Optional[List[str]] = None) -> List[CampaignMetrics]:
        """Fetch Meta Ads metrics"""
        # In production, use Meta Marketing API
        metrics = await super().fetch_metrics(start_date, end_date, campaign_ids)
        
        # Add Meta-specific processing
        for metric in metrics:
            # Simulate social engagement boost
            engagement_rate = np.random.uniform(0.02, 0.08)
            metric.clicks = int(metric.clicks * (1 + engagement_rate))
        
        return metrics


class TikTokAdsConnector(PlatformConnector):
    """TikTok Ads connector"""
    
    def __init__(self, credentials: Dict):
        super().__init__(Platform.TIKTOK, credentials)
    
    async def fetch_metrics(self,
                          start_date: datetime,
                          end_date: datetime,
                          campaign_ids: Optional[List[str]] = None) -> List[CampaignMetrics]:
        """Fetch TikTok Ads metrics"""
        # In production, use TikTok Marketing API
        metrics = await super().fetch_metrics(start_date, end_date, campaign_ids)
        
        # Add TikTok-specific processing
        for metric in metrics:
            # Simulate higher engagement for video platform
            metric.clicks = int(metric.clicks * 1.3)
            metric.conversions = int(metric.conversions * 1.2)
        
        return metrics


class UnifiedDashboard:
    """Main cross-platform dashboard controller"""
    
    def __init__(self):
        self.connectors: Dict[Platform, PlatformConnector] = {}
        self.cached_data: Dict[str, Any] = {}
        self.refresh_interval = timedelta(minutes=15)
        self.last_refresh = None
    
    def add_platform(self, platform: Platform, credentials: Dict):
        """Add platform connector"""
        if platform == Platform.GOOGLE:
            self.connectors[platform] = GoogleAdsConnector(credentials)
        elif platform == Platform.META:
            self.connectors[platform] = MetaAdsConnector(credentials)
        elif platform == Platform.TIKTOK:
            self.connectors[platform] = TikTokAdsConnector(credentials)
        else:
            self.connectors[platform] = PlatformConnector(platform, credentials)
    
    async def sync_all_platforms(self,
                               start_date: datetime,
                               end_date: datetime) -> Dict[Platform, List[CampaignMetrics]]:
        """Sync data from all connected platforms"""
        all_metrics = {}
        
        # Fetch from all platforms concurrently
        tasks = []
        for platform, connector in self.connectors.items():
            task = connector.fetch_metrics(start_date, end_date)
            tasks.append((platform, task))
        
        # Execute all tasks
        for platform, task in tasks:
            try:
                metrics = await task
                all_metrics[platform] = metrics
            except Exception as e:
                print(f"Error fetching {platform.value}: {e}")
                all_metrics[platform] = []
        
        self.last_refresh = datetime.now()
        self.cached_data = all_metrics
        
        return all_metrics
    
    def get_unified_metrics(self) -> pd.DataFrame:
        """Get unified metrics across all platforms"""
        all_data = []
        
        for platform, metrics_list in self.cached_data.items():
            for metric in metrics_list:
                all_data.append({
                    'platform': platform.value,
                    'campaign_id': metric.campaign_id,
                    'campaign_name': metric.campaign_name,
                    'timestamp': metric.timestamp,
                    'impressions': metric.impressions,
                    'clicks': metric.clicks,
                    'conversions': metric.conversions,
                    'spend': metric.spend,
                    'revenue': metric.revenue,
                    'ctr': metric.ctr,
                    'cpc': metric.cpc,
                    'cpa': metric.cpa,
                    'roas': metric.roas,
                    'cpm': metric.cpm
                })
        
        return pd.DataFrame(all_data)
    
    def get_platform_comparison(self) -> Dict:
        """Compare performance across platforms"""
        df = self.get_unified_metrics()
        
        if df.empty:
            return {}
        
        comparison = {}
        
        for platform in df['platform'].unique():
            platform_data = df[df['platform'] == platform]
            
            comparison[platform] = {
                'total_spend': platform_data['spend'].sum(),
                'total_revenue': platform_data['revenue'].sum(),
                'total_conversions': platform_data['conversions'].sum(),
                'avg_cpa': platform_data['spend'].sum() / platform_data['conversions'].sum() if platform_data['conversions'].sum() > 0 else 0,
                'avg_roas': platform_data['revenue'].sum() / platform_data['spend'].sum() if platform_data['spend'].sum() > 0 else 0,
                'avg_ctr': platform_data['ctr'].mean(),
                'campaign_count': len(platform_data)
            }
        
        return comparison
    
    def get_top_performers(self, 
                          metric: MetricType = MetricType.ROAS,
                          limit: int = 10) -> pd.DataFrame:
        """Get top performing campaigns across platforms"""
        df = self.get_unified_metrics()
        
        if df.empty:
            return pd.DataFrame()
        
        # Sort by specified metric
        sorted_df = df.sort_values(by=metric.value, ascending=False)
        
        return sorted_df.head(limit)
    
    def get_attribution_view(self) -> Dict:
        """Get cross-platform attribution insights"""
        df = self.get_unified_metrics()
        
        if df.empty:
            return {}
        
        # Calculate platform contribution to conversions
        total_conversions = df['conversions'].sum()
        platform_contribution = {}
        
        for platform in df['platform'].unique():
            platform_conversions = df[df['platform'] == platform]['conversions'].sum()
            platform_contribution[platform] = {
                'conversions': platform_conversions,
                'percentage': (platform_conversions / total_conversions * 100) if total_conversions > 0 else 0
            }
        
        # Calculate conversion paths (simplified)
        conversion_paths = {
            'single_touch': total_conversions * 0.4,
            'multi_touch': total_conversions * 0.6,
            'avg_touchpoints': 2.3
        }
        
        return {
            'platform_contribution': platform_contribution,
            'conversion_paths': conversion_paths,
            'total_conversions': total_conversions
        }


class DashboardVisualizer:
    """Generate dashboard visualizations and reports"""
    
    def __init__(self, dashboard: UnifiedDashboard):
        self.dashboard = dashboard
    
    def generate_executive_summary(self) -> Dict:
        """Generate executive summary view"""
        df = self.dashboard.get_unified_metrics()
        
        if df.empty:
            return {"error": "No data available"}
        
        summary = {
            'total_spend': df['spend'].sum(),
            'total_revenue': df['revenue'].sum(),
            'total_conversions': df['conversions'].sum(),
            'overall_roas': df['revenue'].sum() / df['spend'].sum() if df['spend'].sum() > 0 else 0,
            'overall_cpa': df['spend'].sum() / df['conversions'].sum() if df['conversions'].sum() > 0 else 0,
            'active_campaigns': len(df),
            'platform_breakdown': self.dashboard.get_platform_comparison(),
            'top_campaigns': self.dashboard.get_top_performers(limit=5).to_dict('records'),
            'period': {
                'start': df['timestamp'].min().isoformat() if not df.empty else None,
                'end': df['timestamp'].max().isoformat() if not df.empty else None
            }
        }
        
        return summary
    
    def generate_performance_report(self) -> Dict:
        """Generate detailed performance report"""
        df = self.dashboard.get_unified_metrics()
        
        if df.empty:
            return {"error": "No data available"}
        
        # Calculate performance metrics
        performance = {
            'by_platform': {},
            'trends': {},
            'anomalies': []
        }
        
        # Platform performance
        for platform in df['platform'].unique():
            platform_df = df[df['platform'] == platform]
            
            performance['by_platform'][platform] = {
                'metrics': {
                    'impressions': platform_df['impressions'].sum(),
                    'clicks': platform_df['clicks'].sum(),
                    'conversions': platform_df['conversions'].sum(),
                    'spend': platform_df['spend'].sum(),
                    'revenue': platform_df['revenue'].sum()
                },
                'ratios': {
                    'ctr': platform_df['ctr'].mean(),
                    'cvr': (platform_df['conversions'].sum() / platform_df['clicks'].sum() * 100) if platform_df['clicks'].sum() > 0 else 0,
                    'cpa': platform_df['cpa'].mean(),
                    'roas': platform_df['roas'].mean()
                },
                'best_campaign': platform_df.nlargest(1, 'roas').iloc[0]['campaign_name'] if not platform_df.empty else None,
                'worst_campaign': platform_df.nsmallest(1, 'roas').iloc[0]['campaign_name'] if not platform_df.empty else None
            }
        
        # Detect anomalies
        for col in ['ctr', 'cpa', 'roas']:
            mean = df[col].mean()
            std = df[col].std()
            
            anomalies = df[abs(df[col] - mean) > 2 * std]
            for _, row in anomalies.iterrows():
                performance['anomalies'].append({
                    'campaign': row['campaign_name'],
                    'platform': row['platform'],
                    'metric': col,
                    'value': row[col],
                    'deviation': abs(row[col] - mean) / std
                })
        
        return performance
    
    def generate_predictive_insights(self) -> Dict:
        """Generate predictive insights based on current data"""
        df = self.dashboard.get_unified_metrics()
        
        if df.empty:
            return {"error": "No data available"}
        
        insights = {
            'predictions': {},
            'opportunities': [],
            'risks': []
        }
        
        # Simple trend prediction (in production, use proper ML models)
        for platform in df['platform'].unique():
            platform_df = df[df['platform'] == platform]
            
            # Calculate simple growth rate
            if len(platform_df) > 1:
                spend_growth = (platform_df['spend'].iloc[-1] - platform_df['spend'].iloc[0]) / platform_df['spend'].iloc[0]
                revenue_growth = (platform_df['revenue'].iloc[-1] - platform_df['revenue'].iloc[0]) / platform_df['revenue'].iloc[0]
                
                insights['predictions'][platform] = {
                    'spend_trend': 'increasing' if spend_growth > 0 else 'decreasing',
                    'revenue_trend': 'increasing' if revenue_growth > 0 else 'decreasing',
                    'projected_next_period_spend': platform_df['spend'].sum() * (1 + spend_growth),
                    'projected_next_period_revenue': platform_df['revenue'].sum() * (1 + revenue_growth)
                }
        
        # Identify opportunities
        high_performers = df[df['roas'] > df['roas'].quantile(0.75)]
        for _, campaign in high_performers.iterrows():
            if campaign['spend'] < df['spend'].median():
                insights['opportunities'].append({
                    'campaign': campaign['campaign_name'],
                    'platform': campaign['platform'],
                    'recommendation': 'Scale high-performing campaign',
                    'potential_revenue': campaign['revenue'] * 1.5
                })
        
        # Identify risks
        low_performers = df[df['roas'] < df['roas'].quantile(0.25)]
        for _, campaign in low_performers.iterrows():
            insights['risks'].append({
                'campaign': campaign['campaign_name'],
                'platform': campaign['platform'],
                'issue': 'Low ROAS',
                'recommendation': 'Review targeting or pause campaign'
            })
        
        return insights
    
    def generate_competitive_view(self) -> Dict:
        """Generate competitive intelligence view"""
        # In production, this would integrate with competitive intelligence tools
        return {
            'market_share_estimate': {
                'our_spend': self.dashboard.get_unified_metrics()['spend'].sum(),
                'estimated_market_size': self.dashboard.get_unified_metrics()['spend'].sum() * 10,
                'estimated_share': 10.0
            },
            'competitive_metrics': {
                'our_avg_cpa': self.dashboard.get_unified_metrics()['cpa'].mean(),
                'industry_avg_cpa': self.dashboard.get_unified_metrics()['cpa'].mean() * 1.2,
                'position': 'below average'
            },
            'recommendations': [
                "Increase investment in top-performing platforms",
                "Test new creative formats on underperforming campaigns",
                "Explore emerging platforms for first-mover advantage"
            ]
        }


class RealTimeDashboard:
    """Real-time monitoring and alerting"""
    
    def __init__(self, dashboard: UnifiedDashboard):
        self.dashboard = dashboard
        self.alert_rules = []
        self.alert_history = []
    
    def add_alert_rule(self, 
                      name: str,
                      condition: Dict,
                      action: str):
        """Add alert rule for real-time monitoring"""
        self.alert_rules.append({
            'name': name,
            'condition': condition,
            'action': action,
            'created_at': datetime.now()
        })
    
    async def monitor(self):
        """Monitor campaigns in real-time"""
        while True:
            # Refresh data
            await self.dashboard.sync_all_platforms(
                datetime.now() - timedelta(hours=1),
                datetime.now()
            )
            
            # Check alert rules
            df = self.dashboard.get_unified_metrics()
            
            for rule in self.alert_rules:
                if self._check_condition(df, rule['condition']):
                    self._trigger_alert(rule)
            
            # Wait before next check
            await asyncio.sleep(300)  # 5 minutes
    
    def _check_condition(self, df: pd.DataFrame, condition: Dict) -> bool:
        """Check if alert condition is met"""
        metric = condition.get('metric')
        threshold = condition.get('threshold')
        operator = condition.get('operator', '>')
        
        if metric in df.columns:
            if operator == '>':
                return any(df[metric] > threshold)
            elif operator == '<':
                return any(df[metric] < threshold)
        
        return False
    
    def _trigger_alert(self, rule: Dict):
        """Trigger alert action"""
        alert = {
            'rule_name': rule['name'],
            'triggered_at': datetime.now(),
            'action': rule['action']
        }
        
        self.alert_history.append(alert)
        
        # In production, send notifications
        print(f"ALERT: {rule['name']} triggered at {alert['triggered_at']}")


# Example usage
async def main():
    """Example dashboard usage"""
    # Create dashboard
    dashboard = UnifiedDashboard()
    
    # Add platforms
    dashboard.add_platform(Platform.GOOGLE, {"client_id": "xxx", "client_secret": "yyy"})
    dashboard.add_platform(Platform.META, {"access_token": "zzz"})
    dashboard.add_platform(Platform.TIKTOK, {"api_key": "aaa"})
    
    # Sync data
    await dashboard.sync_all_platforms(
        datetime.now() - timedelta(days=7),
        datetime.now()
    )
    
    # Create visualizer
    visualizer = DashboardVisualizer(dashboard)
    
    # Generate reports
    executive_summary = visualizer.generate_executive_summary()
    print("Executive Summary:")
    print(f"Total Spend: ${executive_summary['total_spend']:,.2f}")
    print(f"Total Revenue: ${executive_summary['total_revenue']:,.2f}")
    print(f"Overall ROAS: {executive_summary['overall_roas']:.2f}")
    
    # Performance report
    performance = visualizer.generate_performance_report()
    print("\nPlatform Performance:")
    for platform, metrics in performance['by_platform'].items():
        print(f"\n{platform}:")
        print(f"  ROAS: {metrics['ratios']['roas']:.2f}")
        print(f"  CPA: ${metrics['ratios']['cpa']:.2f}")
    
    # Predictive insights
    insights = visualizer.generate_predictive_insights()
    print("\nPredictive Insights:")
    print(f"Opportunities: {len(insights['opportunities'])}")
    print(f"Risks: {len(insights['risks'])}")
    
    # Set up real-time monitoring
    realtime = RealTimeDashboard(dashboard)
    realtime.add_alert_rule(
        "High CPA Alert",
        {"metric": "cpa", "threshold": 100, "operator": ">"},
        "notify_team"
    )
    
    # Would run continuously in production
    # await realtime.monitor()


if __name__ == "__main__":
    asyncio.run(main())