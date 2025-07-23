#!/usr/bin/env python3
"""
Real Campaign Analysis with Comprehensive Features
Analyzes the 32 real campaigns using extracted features for immediate insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RealCampaignAnalyzer:
    """Analyze real campaigns with comprehensive features"""
    
    def __init__(self, db_path="real_campaigns_features.db"):
        self.db_path = db_path
        self.df = None
        self.output_dir = Path("output/campaign_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load the comprehensive campaign dataset"""
        try:
            conn = sqlite3.connect(self.db_path)
            self.df = pd.read_sql_query("SELECT * FROM real_campaigns_comprehensive", conn)
            conn.close()
            
            print(f"✅ Loaded {len(self.df)} campaigns with {len(self.df.columns)} features")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return False
    
    def analyze_award_recognition_impact(self):
        """Analyze the impact of award recognition on effectiveness"""
        print("\n🏆 AWARD RECOGNITION IMPACT ANALYSIS")
        print("="*50)
        
        # Group by award show
        award_analysis = self.df.groupby('award_show').agg({
            'creative_effectiveness_score': ['mean', 'std', 'count'],
            'award_prestige_score': 'mean',
            'roi_multiplier': 'mean',
            'brand_lift_percentage': 'mean'
        }).round(2)
        
        print("\nEffectiveness by Award Show:")
        print(award_analysis)
        
        # Award prestige correlation
        correlation = self.df['award_prestige_score'].corr(self.df['creative_effectiveness_score'])
        print(f"\nAward Prestige vs Effectiveness Correlation: {correlation:.3f}")
        
        # Top performing campaigns by award show
        print("\nTop Performing Campaign by Award Show:")
        for show in self.df['award_show'].unique():
            show_campaigns = self.df[self.df['award_show'] == show]
            top_campaign = show_campaigns.loc[show_campaigns['creative_effectiveness_score'].idxmax()]
            print(f"   {show}: {top_campaign['name']} (CES: {top_campaign['creative_effectiveness_score']:.1f})")
        
        return award_analysis
    
    def analyze_csr_effectiveness(self):
        """Analyze CSR campaign effectiveness"""
        print("\n💡 CSR EFFECTIVENESS ANALYSIS")
        print("="*40)
        
        csr_campaigns = self.df[self.df['csr_presence_binary'] == 1]
        non_csr_campaigns = self.df[self.df['csr_presence_binary'] == 0]
        
        if len(csr_campaigns) > 0:
            print(f"\nCSR Campaigns: {len(csr_campaigns)}")
            print(f"Non-CSR Campaigns: {len(non_csr_campaigns)}")
            
            csr_avg = csr_campaigns['creative_effectiveness_score'].mean()
            non_csr_avg = non_csr_campaigns['creative_effectiveness_score'].mean()
            
            print(f"\nAverage Effectiveness Scores:")
            print(f"   CSR Campaigns: {csr_avg:.1f}")
            print(f"   Non-CSR Campaigns: {non_csr_avg:.1f}")
            print(f"   CSR Impact: {'+' if csr_avg > non_csr_avg else ''}{csr_avg - non_csr_avg:.1f} points")
            
            # CSR by category
            if 'csr_category' in self.df.columns:
                csr_by_category = csr_campaigns.groupby('csr_category')['creative_effectiveness_score'].mean().sort_values(ascending=False)
                print(f"\nCSR Effectiveness by Category:")
                for category, score in csr_by_category.items():
                    if category != 'None':
                        print(f"   {category}: {score:.1f}")
            
            # CSR authenticity impact
            if len(csr_campaigns) > 1:
                auth_correlation = csr_campaigns['csr_authenticity_score'].corr(csr_campaigns['creative_effectiveness_score'])
                print(f"\nCSR Authenticity vs Effectiveness Correlation: {auth_correlation:.3f}")
            
            return {
                'csr_count': len(csr_campaigns),
                'csr_avg_effectiveness': csr_avg,
                'non_csr_avg_effectiveness': non_csr_avg,
                'csr_impact': csr_avg - non_csr_avg
            }
        else:
            print("No CSR campaigns found in dataset")
            return None
    
    def analyze_cultural_relevance(self):
        """Analyze cultural relevance impact"""
        print("\n🌏 CULTURAL RELEVANCE ANALYSIS")
        print("="*40)
        
        # Cultural relevance vs effectiveness
        cultural_corr = self.df['cultural_relevance_score'].corr(self.df['creative_effectiveness_score'])
        print(f"\nCultural Relevance vs Effectiveness Correlation: {cultural_corr:.3f}")
        
        # Localization depth impact
        if 'localization_depth' in self.df.columns:
            local_analysis = self.df.groupby('localization_depth')['creative_effectiveness_score'].agg(['mean', 'count']).round(2)
            print(f"\nEffectiveness by Localization Depth:")
            print(local_analysis)
        
        # High cultural relevance campaigns
        high_cultural = self.df[self.df['cultural_relevance_score'] > 0.8]
        print(f"\nHigh Cultural Relevance Campaigns (score > 0.8): {len(high_cultural)}")
        if len(high_cultural) > 0:
            avg_effectiveness = high_cultural['creative_effectiveness_score'].mean()
            print(f"   Average Effectiveness: {avg_effectiveness:.1f}")
            
            print("\n   Top Culturally Relevant Campaigns:")
            for _, campaign in high_cultural.nlargest(3, 'cultural_relevance_score').iterrows():
                print(f"     • {campaign['name']} (Cultural: {campaign['cultural_relevance_score']:.2f}, CES: {campaign['creative_effectiveness_score']:.1f})")
        
        return cultural_corr
    
    def analyze_innovation_impact(self):
        """Analyze innovation and technology impact"""
        print("\n🚀 INNOVATION IMPACT ANALYSIS")
        print("="*35)
        
        # Innovation level correlation
        innovation_corr = self.df['innovation_level'].corr(self.df['creative_effectiveness_score'])
        print(f"\nInnovation Level vs Effectiveness Correlation: {innovation_corr:.3f}")
        
        # Technology integration impact
        tech_corr = self.df['technology_integration'].corr(self.df['creative_effectiveness_score'])
        print(f"Technology Integration vs Effectiveness Correlation: {tech_corr:.3f}")
        
        # High innovation campaigns
        high_innovation = self.df[self.df['innovation_level'] > 0.7]
        print(f"\nHigh Innovation Campaigns (level > 0.7): {len(high_innovation)}")
        if len(high_innovation) > 0:
            avg_effectiveness = high_innovation['creative_effectiveness_score'].mean()
            regular_effectiveness = self.df[self.df['innovation_level'] <= 0.7]['creative_effectiveness_score'].mean()
            print(f"   High Innovation Avg: {avg_effectiveness:.1f}")
            print(f"   Regular Innovation Avg: {regular_effectiveness:.1f}")
            print(f"   Innovation Boost: {avg_effectiveness - regular_effectiveness:.1f} points")
        
        return innovation_corr
    
    def analyze_brand_performance(self):
        """Analyze performance by brand"""
        print("\n🏢 BRAND PERFORMANCE ANALYSIS")
        print("="*35)
        
        brand_analysis = self.df.groupby('brand').agg({
            'creative_effectiveness_score': ['mean', 'count'],
            'award_prestige_score': 'mean',
            'csr_presence_binary': 'sum',
            'innovation_level': 'mean'
        }).round(2)
        
        brand_analysis.columns = ['Avg_CES', 'Campaign_Count', 'Avg_Award_Prestige', 'CSR_Campaigns', 'Avg_Innovation']
        brand_analysis = brand_analysis.sort_values('Avg_CES', ascending=False)
        
        print("\nBrand Performance Summary:")
        print(brand_analysis)
        
        # Multi-campaign brands
        multi_brands = brand_analysis[brand_analysis['Campaign_Count'] > 1]
        if len(multi_brands) > 0:
            print(f"\nBrands with Multiple Campaigns: {len(multi_brands)}")
            for brand, row in multi_brands.iterrows():
                print(f"   {brand}: {row['Campaign_Count']} campaigns, avg CES {row['Avg_CES']:.1f}")
        
        return brand_analysis
    
    def analyze_feature_importance(self):
        """Analyze which features are most important for effectiveness"""
        print("\n📊 FEATURE IMPORTANCE ANALYSIS")
        print("="*40)
        
        # Select numeric features for correlation analysis
        numeric_features = [
            'award_prestige_score', 'csr_message_prominence', 'visual_complexity_score',
            'aesthetic_score', 'message_sentiment', 'message_clarity', 'innovation_level',
            'cultural_relevance_score', 'brand_asset_visibility', 'memorability_score'
        ]
        
        # Filter features that exist in the dataset
        available_features = [f for f in numeric_features if f in self.df.columns]
        
        if available_features:
            correlations = self.df[available_features + ['creative_effectiveness_score']].corr()['creative_effectiveness_score'][:-1].sort_values(ascending=False)
            
            print("\nFeature Correlations with Creative Effectiveness:")
            for feature, corr in correlations.items():
                print(f"   {feature:<25}: {corr:>6.3f}")
            
            # Feature combinations analysis
            print(f"\nTop Feature Combinations:")
            
            # High-performing campaigns analysis
            top_campaigns = self.df.nlargest(5, 'creative_effectiveness_score')
            print(f"\nTop 5 Campaigns - Feature Profiles:")
            for _, campaign in top_campaigns.iterrows():
                print(f"\n   {campaign['name']} (CES: {campaign['creative_effectiveness_score']:.1f})")
                for feature in available_features[:5]:  # Top 5 features
                    print(f"     {feature}: {campaign[feature]:.2f}")
            
            return correlations
        else:
            print("No suitable numeric features found for correlation analysis")
            return None
    
    def generate_insights_summary(self):
        """Generate comprehensive insights summary"""
        print("\n💡 KEY INSIGHTS SUMMARY")
        print("="*30)
        
        insights = []
        
        # Overall performance
        avg_ces = self.df['creative_effectiveness_score'].mean()
        high_performers = len(self.df[self.df['creative_effectiveness_score'] > 75])
        insights.append(f"📈 Average effectiveness score: {avg_ces:.1f}")
        insights.append(f"🌟 High performers (CES > 75): {high_performers}/{len(self.df)} ({high_performers/len(self.df)*100:.1f}%)")
        
        # Award recognition insights
        if 'award_prestige_score' in self.df.columns:
            top_award_show = self.df.groupby('award_show')['creative_effectiveness_score'].mean().idxmax()
            top_score = self.df.groupby('award_show')['creative_effectiveness_score'].mean().max()
            insights.append(f"🏆 Top performing award show: {top_award_show} (avg CES: {top_score:.1f})")
        
        # CSR insights
        csr_campaigns = self.df[self.df['csr_presence_binary'] == 1] if 'csr_presence_binary' in self.df.columns else pd.DataFrame()
        if len(csr_campaigns) > 0:
            csr_avg = csr_campaigns['creative_effectiveness_score'].mean()
            non_csr_avg = self.df[self.df['csr_presence_binary'] == 0]['creative_effectiveness_score'].mean()
            if csr_avg > non_csr_avg:
                insights.append(f"💚 CSR campaigns outperform by {csr_avg - non_csr_avg:.1f} points")
            else:
                insights.append(f"⚠️ CSR campaigns underperform by {non_csr_avg - csr_avg:.1f} points")
        
        # Innovation insights
        if 'innovation_level' in self.df.columns:
            innovation_corr = self.df['innovation_level'].corr(self.df['creative_effectiveness_score'])
            if innovation_corr > 0.3:
                insights.append(f"🚀 Strong innovation-effectiveness correlation ({innovation_corr:.2f})")
            elif innovation_corr > 0.1:
                insights.append(f"📊 Moderate innovation-effectiveness correlation ({innovation_corr:.2f})")
        
        # Cultural insights
        if 'cultural_relevance_score' in self.df.columns:
            cultural_corr = self.df['cultural_relevance_score'].corr(self.df['creative_effectiveness_score'])
            if cultural_corr > 0.2:
                insights.append(f"🌏 Cultural relevance positively impacts effectiveness ({cultural_corr:.2f})")
        
        # Brand insights
        brand_performance = self.df.groupby('brand')['creative_effectiveness_score'].mean()
        top_brand = brand_performance.idxmax()
        top_brand_score = brand_performance.max()
        insights.append(f"🏢 Top performing brand: {top_brand} (avg CES: {top_brand_score:.1f})")
        
        # Display insights
        for insight in insights:
            print(f"   {insight}")
        
        return insights
    
    def create_visualization_summary(self):
        """Create visualization summary (text-based for now)"""
        print("\n📊 VISUALIZATION SUMMARY")
        print("="*30)
        
        print("\n📈 Recommended Visualizations:")
        print("   1. Award Show vs Effectiveness (Bar Chart)")
        print("   2. CSR vs Non-CSR Effectiveness (Box Plot)")
        print("   3. Innovation Level vs Effectiveness (Scatter Plot)")
        print("   4. Cultural Relevance Distribution (Histogram)")
        print("   5. Brand Performance Comparison (Horizontal Bar)")
        print("   6. Feature Correlation Heatmap")
        print("   7. Top Campaigns Performance Dashboard")
        
        # Create simple text-based distribution
        print(f"\n📊 Effectiveness Score Distribution:")
        ces_bins = pd.cut(self.df['creative_effectiveness_score'], bins=5, labels=['Low', 'Below Avg', 'Average', 'Good', 'Excellent'])
        distribution = ces_bins.value_counts().sort_index()
        
        for category, count in distribution.items():
            bar_length = int(count / len(self.df) * 40)  # Scale to 40 chars
            bar = "█" * bar_length
            print(f"   {category:<10}: {bar} ({count})")
    
    def save_analysis_report(self, award_analysis, csr_analysis, cultural_corr, innovation_corr, brand_analysis, feature_correlations, insights):
        """Save comprehensive analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            "analysis_metadata": {
                "timestamp": timestamp,
                "total_campaigns": len(self.df),
                "analysis_date": datetime.now().isoformat(),
                "database_source": self.db_path
            },
            "award_recognition_analysis": award_analysis.to_dict() if award_analysis is not None else None,
            "csr_effectiveness_analysis": csr_analysis,
            "cultural_relevance_correlation": float(cultural_corr) if cultural_corr is not None else None,
            "innovation_impact_correlation": float(innovation_corr) if innovation_corr is not None else None,
            "brand_performance": brand_analysis.to_dict() if brand_analysis is not None else None,
            "feature_importance": feature_correlations.to_dict() if feature_correlations is not None else None,
            "key_insights": insights,
            "summary_statistics": {
                "mean_effectiveness": float(self.df['creative_effectiveness_score'].mean()),
                "std_effectiveness": float(self.df['creative_effectiveness_score'].std()),
                "high_performers_count": len(self.df[self.df['creative_effectiveness_score'] > 75]),
                "csr_campaigns_count": len(self.df[self.df['csr_presence_binary'] == 1]) if 'csr_presence_binary' in self.df.columns else 0
            }
        }
        
        report_path = self.output_dir / f"campaign_analysis_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Analysis report saved: {report_path}")
        return report_path
    
    def run_comprehensive_analysis(self):
        """Run the complete analysis pipeline"""
        print("🔍 REAL CAMPAIGN COMPREHENSIVE ANALYSIS")
        print("="*50)
        
        if not self.load_data():
            return False
        
        print(f"\n📊 Dataset Overview:")
        print(f"   Total Campaigns: {len(self.df)}")
        print(f"   Award Shows: {', '.join(self.df['award_show'].unique())}")
        print(f"   Brands: {len(self.df['brand'].unique())}")
        print(f"   Industries: {', '.join(self.df['industry'].unique())}")
        
        # Run all analyses
        award_analysis = self.analyze_award_recognition_impact()
        csr_analysis = self.analyze_csr_effectiveness()
        cultural_corr = self.analyze_cultural_relevance()
        innovation_corr = self.analyze_innovation_impact()
        brand_analysis = self.analyze_brand_performance()
        feature_correlations = self.analyze_feature_importance()
        insights = self.generate_insights_summary()
        
        # Create visualizations summary
        self.create_visualization_summary()
        
        # Save report
        report_path = self.save_analysis_report(
            award_analysis, csr_analysis, cultural_corr, innovation_corr, 
            brand_analysis, feature_correlations, insights
        )
        
        print("\n✅ ANALYSIS COMPLETE!")
        print(f"   📋 Report saved: {report_path}")
        print(f"   📊 {len(insights)} key insights generated")
        print(f"   🎯 Ready for strategic recommendations")
        
        return True


def main():
    """Run the comprehensive campaign analysis"""
    analyzer = RealCampaignAnalyzer()
    
    success = analyzer.run_comprehensive_analysis()
    
    if success:
        print("\n🚀 Next steps:")
        print("   1. Review the analysis report")
        print("   2. Create visualizations in your preferred tool")
        print("   3. Generate strategic recommendations")
        print("   4. Share insights with stakeholders")
    else:
        print("\n❌ Analysis failed. Check data and try again.")


if __name__ == "__main__":
    main()
