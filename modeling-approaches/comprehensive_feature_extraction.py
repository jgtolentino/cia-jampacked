#!/usr/bin/env python3
"""
Comprehensive Feature Extraction for Creative Effectiveness Analysis
Extract ALL variables from 8 modeling approaches into one rich dataset
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import random
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class ComprehensiveFeatureExtractor:
    """Extract all features from 8 modeling approaches into unified dataset"""
    
    def __init__(self, n_campaigns: int = 1000):
        self.n_campaigns = n_campaigns
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("output/comprehensive_extraction")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_campaign_metadata(self) -> pd.DataFrame:
        """Generate basic campaign metadata"""
        campaigns = []
        
        for i in range(self.n_campaigns):
            campaign = {
                'campaign_id': f'CAMP_{i:04d}',
                'brand': np.random.choice(['Brand_A', 'Brand_B', 'Brand_C', 'Brand_D', 'Brand_E']),
                'category': np.random.choice(['CPG', 'Tech', 'Retail', 'Finance', 'Auto']),
                'launch_date': datetime.now() - timedelta(days=np.random.randint(1, 730)),
                'region': np.random.choice(['North_America', 'Europe', 'Asia', 'LATAM', 'APAC']),
                'campaign_type': np.random.choice(['Brand', 'Performance', 'Hybrid']),
                'primary_channel': np.random.choice(['TV', 'Digital', 'Social', 'OOH', 'Radio']),
            }
            campaigns.append(campaign)
            
        return pd.DataFrame(campaigns)
    
    def extract_visual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all visual-related features"""
        # Basic visual complexity
        df['visual_complexity_score'] = np.random.beta(2, 5, self.n_campaigns)
        df['edge_density'] = np.random.uniform(0.1, 0.9, self.n_campaigns)
        df['spatial_frequency'] = np.random.uniform(0.2, 0.8, self.n_campaigns)
        df['texture_variance'] = np.random.uniform(0.1, 0.7, self.n_campaigns)
        
        # Object and scene features
        df['object_count'] = np.random.poisson(4, self.n_campaigns)
        df['scene_complexity'] = np.random.beta(3, 3, self.n_campaigns)
        df['depth_layers'] = np.random.randint(1, 5, self.n_campaigns)
        df['motion_intensity'] = np.random.uniform(0, 1, self.n_campaigns)
        
        # Color features
        df['color_palette_diversity'] = np.random.uniform(0.2, 0.9, self.n_campaigns)
        df['dominant_color_hue'] = np.random.uniform(0, 360, self.n_campaigns)
        df['color_saturation_mean'] = np.random.uniform(0.3, 0.9, self.n_campaigns)
        df['color_contrast_ratio'] = np.random.uniform(2, 21, self.n_campaigns)
        
        # Face and emotion
        df['face_presence_binary'] = np.random.binomial(1, 0.7, self.n_campaigns)
        df['face_count'] = np.where(df['face_presence_binary'], 
                                    np.random.poisson(2, self.n_campaigns), 0)
        df['dominant_emotion'] = np.random.choice(['joy', 'trust', 'anticipation', 'neutral'], 
                                                 self.n_campaigns)
        df['emotion_intensity'] = np.random.uniform(0.3, 0.9, self.n_campaigns)
        
        # Brand elements
        df['brand_asset_visibility'] = np.random.beta(4, 2, self.n_campaigns)
        df['logo_size_ratio'] = np.random.uniform(0.01, 0.15, self.n_campaigns)
        df['brand_color_prominence'] = np.random.uniform(0.1, 0.8, self.n_campaigns)
        df['product_visibility_score'] = np.random.beta(3, 3, self.n_campaigns)
        
        # Aesthetic and memorability
        df['aesthetic_score'] = np.random.beta(3, 2, self.n_campaigns)
        df['memorability_score'] = np.random.beta(3, 3, self.n_campaigns)
        df['distinctiveness_score'] = np.random.uniform(0.3, 0.95, self.n_campaigns)
        df['visual_hierarchy_score'] = np.random.uniform(0.4, 0.9, self.n_campaigns)
        
        return df
    
    def extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all text and messaging features"""
        # Message characteristics
        df['message_sentiment'] = np.random.uniform(-0.5, 0.9, self.n_campaigns)
        df['message_urgency'] = np.random.beta(2, 3, self.n_campaigns)
        df['emotional_tone_score'] = np.random.uniform(0.2, 0.9, self.n_campaigns)
        df['benefit_orientation'] = np.random.choice(['functional', 'emotional', 'social'], 
                                                   self.n_campaigns)
        
        # Readability and clarity
        df['readability_score'] = np.random.uniform(40, 80, self.n_campaigns)  # Flesch score
        df['message_clarity'] = np.random.beta(4, 2, self.n_campaigns)
        df['jargon_density'] = np.random.uniform(0, 0.3, self.n_campaigns)
        df['word_count'] = np.random.poisson(25, self.n_campaigns)
        
        # Call-to-action
        df['cta_presence'] = np.random.binomial(1, 0.8, self.n_campaigns)
        df['cta_strength'] = np.where(df['cta_presence'], 
                                     np.random.uniform(0.4, 0.9, self.n_campaigns), 0)
        df['cta_clarity'] = np.where(df['cta_presence'], 
                                    np.random.beta(4, 2, self.n_campaigns), 0)
        
        # Headlines and copy
        df['headline_strength'] = np.random.beta(3, 2, self.n_campaigns)
        df['headline_length'] = np.random.poisson(6, self.n_campaigns)
        df['value_proposition_clarity'] = np.random.uniform(0.3, 0.95, self.n_campaigns)
        df['unique_selling_point_strength'] = np.random.beta(3, 3, self.n_campaigns)
        
        # Social proof and credibility
        df['social_proof_elements'] = np.random.poisson(1, self.n_campaigns)
        df['credibility_indicators'] = np.random.binomial(3, 0.4, self.n_campaigns)
        df['testimonial_presence'] = np.random.binomial(1, 0.3, self.n_campaigns)
        
        return df
    
    def extract_audio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all audio-related features"""
        # Basic audio properties
        df['audio_presence'] = np.random.binomial(1, 0.7, self.n_campaigns)
        df['audio_tempo'] = np.where(df['audio_presence'], 
                                    np.random.uniform(60, 180, self.n_campaigns), 0)
        df['audio_energy'] = np.where(df['audio_presence'], 
                                     np.random.uniform(0.2, 0.9, self.n_campaigns), 0)
        df['audio_valence'] = np.where(df['audio_presence'], 
                                      np.random.uniform(0.1, 0.9, self.n_campaigns), 0)
        
        # Voice and music
        df['voice_presence'] = np.where(df['audio_presence'], 
                                       np.random.binomial(1, 0.8, self.n_campaigns), 0)
        df['music_presence'] = np.where(df['audio_presence'], 
                                       np.random.binomial(1, 0.9, self.n_campaigns), 0)
        df['voice_gender'] = np.where(df['voice_presence'], 
                                     np.random.choice(['male', 'female', 'mixed'], self.n_campaigns), 
                                     'none')
        df['music_genre'] = np.where(df['music_presence'], 
                                    np.random.choice(['pop', 'classical', 'electronic', 'ambient'], 
                                                   self.n_campaigns), 'none')
        
        # Audio branding
        df['sonic_logo_presence'] = np.random.binomial(1, 0.3, self.n_campaigns)
        df['audio_brand_mentions'] = np.random.poisson(2, self.n_campaigns)
        df['audio_memorability'] = np.where(df['audio_presence'], 
                                          np.random.beta(3, 3, self.n_campaigns), 0)
        
        return df
    
    def extract_campaign_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract campaign-level features"""
        # Duration and timing
        df['campaign_duration_days'] = np.random.randint(7, 180, self.n_campaigns)
        df['days_since_launch'] = (datetime.now() - df['launch_date']).dt.days
        df['quarter_launched'] = df['launch_date'].dt.quarter
        df['is_holiday_campaign'] = np.random.binomial(1, 0.2, self.n_campaigns)
        
        # Budget and investment
        df['budget_allocated'] = np.random.lognormal(11, 1.5, self.n_campaigns)
        df['budget_vs_category_avg'] = np.random.normal(1, 0.3, self.n_campaigns)
        df['share_of_voice'] = np.random.uniform(0.05, 0.35, self.n_campaigns)
        df['competitive_pressure'] = np.random.uniform(0.2, 0.9, self.n_campaigns)
        
        # Channel mix
        df['channel_mix_diversity'] = np.random.uniform(0.2, 0.8, self.n_campaigns)
        df['digital_allocation_pct'] = np.random.uniform(0.1, 0.9, self.n_campaigns)
        df['traditional_allocation_pct'] = 1 - df['digital_allocation_pct']
        df['social_media_weight'] = np.random.uniform(0, 0.5, self.n_campaigns)
        
        # Market conditions
        df['market_saturation'] = np.random.uniform(0.3, 0.9, self.n_campaigns)
        df['category_growth_rate'] = np.random.normal(0.05, 0.1, self.n_campaigns)
        df['economic_sentiment'] = np.random.uniform(-0.5, 0.5, self.n_campaigns)
        df['consumer_confidence'] = np.random.uniform(0.4, 0.8, self.n_campaigns)
        
        # Campaign strategy
        df['targeting_precision'] = np.random.beta(3, 2, self.n_campaigns)
        df['message_consistency'] = np.random.uniform(0.5, 0.95, self.n_campaigns)
        df['creative_distinctiveness'] = np.random.beta(3, 3, self.n_campaigns)
        df['strategic_clarity'] = np.random.uniform(0.4, 0.9, self.n_campaigns)
        
        return df
    
    def extract_cultural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract cultural and contextual features"""
        # Cultural relevance
        df['cultural_relevance_score'] = np.random.beta(3, 3, self.n_campaigns)
        df['cultural_symbol_count'] = np.random.poisson(1, self.n_campaigns)
        df['local_language_accuracy'] = np.random.uniform(0.7, 1.0, self.n_campaigns)
        df['cultural_sensitivity_score'] = np.random.uniform(0.6, 1.0, self.n_campaigns)
        
        # Trend alignment
        df['trend_alignment'] = np.random.beta(2, 3, self.n_campaigns)
        df['meme_usage'] = np.random.binomial(1, 0.2, self.n_campaigns)
        df['zeitgeist_relevance'] = np.random.uniform(0.2, 0.8, self.n_campaigns)
        df['social_movement_alignment'] = np.random.binomial(1, 0.3, self.n_campaigns)
        
        # Localization
        df['localization_depth'] = np.random.choice(['global', 'regional', 'local'], 
                                                  self.n_campaigns)
        df['adaptation_quality'] = np.random.uniform(0.5, 0.95, self.n_campaigns)
        df['cultural_authenticity'] = np.random.beta(3, 2, self.n_campaigns)
        
        return df
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-related features"""
        # Time decay effects
        df['time_since_launch'] = df['days_since_launch']
        df['recency_weight'] = np.exp(-df['days_since_launch'] / 365)
        df['wear_out_factor'] = 1 - np.exp(-df['days_since_launch'] / 180)
        
        # Seasonal effects
        df['seasonality_alignment'] = np.random.uniform(0.3, 0.9, self.n_campaigns)
        df['day_of_week_effect'] = np.random.choice([0.8, 0.9, 1.0, 1.1, 1.2], 
                                                   self.n_campaigns)
        df['month_of_year_effect'] = np.random.uniform(0.8, 1.2, self.n_campaigns)
        
        # Campaign lifecycle
        df['lifecycle_stage'] = pd.cut(df['days_since_launch'], 
                                     bins=[0, 30, 90, 180, 730], 
                                     labels=['launch', 'growth', 'mature', 'decline'])
        df['momentum_score'] = np.where(df['lifecycle_stage'] == 'launch', 
                                       np.random.uniform(0.7, 1.0, self.n_campaigns),
                                       np.random.uniform(0.3, 0.7, self.n_campaigns))
        
        return df
    
    def extract_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for Graph Neural Network approaches"""
        # Asset relationships
        df['asset_count'] = np.random.poisson(5, self.n_campaigns)
        df['asset_diversity'] = np.random.uniform(0.3, 0.9, self.n_campaigns)
        df['asset_coherence'] = np.random.beta(4, 2, self.n_campaigns)
        df['cross_asset_synergy'] = np.random.uniform(0.4, 0.9, self.n_campaigns)
        
        # Campaign connections
        df['campaign_family_size'] = np.random.poisson(3, self.n_campaigns)
        df['brand_heritage_score'] = np.random.uniform(0.3, 0.9, self.n_campaigns)
        df['campaign_evolution_score'] = np.random.beta(3, 3, self.n_campaigns)
        
        # Influence metrics
        df['influencer_involvement'] = np.random.binomial(1, 0.3, self.n_campaigns)
        df['viral_potential'] = np.random.beta(2, 5, self.n_campaigns)
        df['network_reach_potential'] = np.random.lognormal(10, 1, self.n_campaigns)
        
        return df
    
    def extract_hierarchical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for hierarchical models"""
        # Brand-level features
        brand_effects = {
            'Brand_A': 0.1, 'Brand_B': 0.05, 'Brand_C': -0.05, 
            'Brand_D': 0.15, 'Brand_E': -0.1
        }
        df['brand_baseline_effect'] = df['brand'].map(brand_effects)
        df['brand_reputation_score'] = df['brand'].map(
            lambda x: np.random.uniform(0.6, 0.9)
        )
        
        # Category-level features
        category_effects = {
            'CPG': 0.05, 'Tech': 0.1, 'Retail': 0.0, 
            'Finance': -0.05, 'Auto': 0.08
        }
        df['category_baseline_effect'] = df['category'].map(category_effects)
        df['category_maturity'] = df['category'].map(
            lambda x: np.random.choice(['emerging', 'growth', 'mature', 'declining'])
        )
        
        # Region-level features
        region_effects = {
            'North_America': 0.1, 'Europe': 0.05, 'Asia': 0.15, 
            'LATAM': -0.05, 'APAC': 0.12
        }
        df['region_baseline_effect'] = df['region'].map(region_effects)
        df['region_digital_maturity'] = df['region'].map(
            lambda x: np.random.uniform(0.5, 0.95)
        )
        
        return df
    
    def extract_survival_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for survival analysis"""
        # Survival time (how long campaign remained effective)
        df['survival_time'] = np.random.weibull(2, self.n_campaigns) * 180
        df['event_indicator'] = (df['survival_time'] < df['days_since_launch']).astype(int)
        df['censored'] = 1 - df['event_indicator']
        
        # Risk factors
        df['creative_fatigue_risk'] = np.random.beta(2, 3, self.n_campaigns)
        df['competitive_threat_level'] = np.random.uniform(0.2, 0.8, self.n_campaigns)
        df['market_disruption_risk'] = np.random.uniform(0.1, 0.5, self.n_campaigns)
        
        # Durability factors
        df['creative_durability_score'] = np.random.beta(3, 2, self.n_campaigns)
        df['message_timelessness'] = np.random.uniform(0.3, 0.8, self.n_campaigns)
        df['execution_quality'] = np.random.beta(4, 2, self.n_campaigns)
        
        return df
    
    def generate_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate multiple target variables for different modeling approaches"""
        # Primary target: Creative Effectiveness Score (0-100)
        # Complex function of multiple features
        base_score = (
            0.15 * df['visual_complexity_score'] +
            0.10 * df['brand_asset_visibility'] +
            0.10 * df['message_clarity'] +
            0.08 * df['emotional_tone_score'] +
            0.12 * df['strategic_clarity'] +
            0.10 * df['creative_distinctiveness'] +
            0.08 * df['cultural_relevance_score'] +
            0.07 * df['memorability_score'] +
            0.05 * df['audio_memorability'] +
            0.05 * df['viral_potential'] +
            0.10 * np.random.normal(0, 0.1, self.n_campaigns)  # Random noise
        )
        
        # Add interaction effects
        interaction_bonus = (
            0.05 * df['visual_complexity_score'] * df['message_clarity'] +
            0.03 * df['brand_asset_visibility'] * df['memorability_score'] +
            0.02 * df['emotional_tone_score'] * df['cultural_relevance_score']
        )
        
        # Scale to 0-100
        df['creative_effectiveness_score'] = np.clip(
            (base_score + interaction_bonus) * 100, 0, 100
        )
        
        # Secondary targets
        df['roi_multiplier'] = np.exp(df['creative_effectiveness_score'] / 50) * \
                              np.random.uniform(0.8, 1.2, self.n_campaigns)
        
        df['brand_lift_percentage'] = (df['creative_effectiveness_score'] / 5 + 
                                      np.random.normal(0, 2, self.n_campaigns))
        
        df['engagement_rate'] = np.clip(
            df['creative_effectiveness_score'] / 200 + 
            np.random.beta(2, 5, self.n_campaigns) * 0.1,
            0, 1
        )
        
        df['purchase_intent_lift'] = (df['creative_effectiveness_score'] / 10 + 
                                     np.random.normal(0, 1.5, self.n_campaigns))
        
        # Binary classification target
        df['binary_success'] = (df['creative_effectiveness_score'] > 65).astype(int)
        
        # Multi-class target
        df['ces_categorical'] = pd.cut(df['creative_effectiveness_score'],
                                      bins=[0, 40, 60, 80, 100],
                                      labels=['Poor', 'Average', 'Good', 'Excellent'])
        
        # Time-to-event targets (for survival analysis)
        df['time_to_peak_effectiveness'] = np.random.gamma(2, 30, self.n_campaigns)
        df['campaign_longevity'] = df['survival_time']
        
        # Network targets (for GNN)
        df['asset_effectiveness'] = np.random.beta(3, 2, self.n_campaigns)
        df['campaign_synergy_score'] = np.random.uniform(0.4, 0.9, self.n_campaigns)
        
        return df
    
    def create_variable_definitions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive variable definitions"""
        variable_defs = {
            "metadata": {
                "creation_date": datetime.now().isoformat(),
                "n_campaigns": self.n_campaigns,
                "n_features": len(df.columns),
                "modeling_approaches": [
                    "Dynamic Hierarchical Empirical Bayesian (DHEB)",
                    "Bayesian Neural Additive Models (LA-NAMs)",
                    "Ensemble Learning (RF, XGBoost, Stacking)",
                    "Graph Neural Networks (GNN)",
                    "Multi-Level Mixed Effects Models",
                    "Time-Varying Coefficient Models",
                    "Bayesian Network Models",
                    "Survival Analysis Models"
                ]
            },
            "feature_categories": {
                "visual": [col for col in df.columns if 'visual' in col or 'color' in col or 
                          'face' in col or 'aesthetic' in col],
                "text": [col for col in df.columns if 'message' in col or 'headline' in col or 
                        'cta' in col or 'readability' in col],
                "audio": [col for col in df.columns if 'audio' in col or 'voice' in col or 
                         'music' in col or 'sonic' in col],
                "campaign": [col for col in df.columns if 'campaign' in col or 'budget' in col or 
                           'channel' in col or 'market' in col],
                "cultural": [col for col in df.columns if 'cultural' in col or 'trend' in col or 
                           'local' in col],
                "temporal": [col for col in df.columns if 'time' in col or 'recency' in col or 
                           'lifecycle' in col],
                "network": [col for col in df.columns if 'asset' in col or 'network' in col or 
                          'viral' in col],
                "hierarchical": [col for col in df.columns if 'brand_' in col or 'category_' in col or 
                               'region_' in col],
                "survival": [col for col in df.columns if 'survival' in col or 'event' in col or 
                           'risk' in col],
                "targets": [col for col in df.columns if 'effectiveness' in col or 'roi' in col or 
                          'lift' in col or 'success' in col]
            },
            "variable_types": {
                "numeric": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                "datetime": df.select_dtypes(include=['datetime']).columns.tolist(),
                "binary": [col for col in df.columns if df[col].nunique() == 2]
            },
            "target_variables": {
                "primary": "creative_effectiveness_score",
                "regression": ["creative_effectiveness_score", "roi_multiplier", 
                             "brand_lift_percentage", "engagement_rate"],
                "classification": ["binary_success", "ces_categorical"],
                "survival": ["survival_time", "event_indicator"],
                "network": ["asset_effectiveness", "campaign_synergy_score"]
            }
        }
        
        return variable_defs
    
    def create_data_splits(self, df: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create multiple train/test splits for different modeling approaches"""
        splits = {}
        
        # 1. Random split (80/20)
        train_idx = np.random.choice(df.index, size=int(0.8 * len(df)), replace=False)
        test_idx = df.index.difference(train_idx)
        splits['random'] = (df.loc[train_idx], df.loc[test_idx])
        
        # 2. Temporal split (older campaigns for training)
        df_sorted = df.sort_values('launch_date')
        split_point = int(0.8 * len(df_sorted))
        splits['temporal'] = (df_sorted.iloc[:split_point], df_sorted.iloc[split_point:])
        
        # 3. Brand-based split (hold out one brand)
        test_brand = 'Brand_E'
        splits['brand_holdout'] = (
            df[df['brand'] != test_brand],
            df[df['brand'] == test_brand]
        )
        
        # 4. Region-based split (hold out one region)
        test_region = 'APAC'
        splits['region_holdout'] = (
            df[df['region'] != test_region],
            df[df['region'] == test_region]
        )
        
        # 5. Stratified split (maintain target distribution)
        from sklearn.model_selection import train_test_split
        X = df.drop(columns=['creative_effectiveness_score'])
        y = df['creative_effectiveness_score']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=pd.qcut(y, 5, labels=False), random_state=42
        )
        splits['stratified'] = (
            pd.concat([X_train, y_train], axis=1),
            pd.concat([X_test, y_test], axis=1)
        )
        
        return splits
    
    def export_data(self, df: pd.DataFrame, variable_defs: Dict[str, Any], 
                   splits: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]):
        """Export data in multiple formats"""
        # Main dataset
        csv_path = self.output_dir / f"comprehensive_creative_effectiveness_{self.timestamp}.csv"
        parquet_path = self.output_dir / f"comprehensive_creative_effectiveness_{self.timestamp}.parquet"
        json_path = self.output_dir / f"comprehensive_creative_effectiveness_{self.timestamp}.json"
        
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        df.to_json(json_path, orient='records', date_format='iso')
        
        # Variable definitions
        var_def_path = self.output_dir / f"variable_definitions_{self.timestamp}.json"
        with open(var_def_path, 'w') as f:
            json.dump(variable_defs, f, indent=2, default=str)
        
        # Export splits
        splits_dir = self.output_dir / "splits"
        splits_dir.mkdir(exist_ok=True)
        
        for split_name, (train_df, test_df) in splits.items():
            train_df.to_csv(splits_dir / f"{split_name}_train.csv", index=False)
            test_df.to_csv(splits_dir / f"{split_name}_test.csv", index=False)
        
        # Create summary report
        summary = {
            "extraction_timestamp": self.timestamp,
            "dataset_shape": list(df.shape),
            "numeric_features": len(variable_defs['variable_types']['numeric']),
            "categorical_features": len(variable_defs['variable_types']['categorical']),
            "target_variables": len(variable_defs['target_variables']['regression']) + 
                              len(variable_defs['target_variables']['classification']),
            "file_locations": {
                "csv": str(csv_path),
                "parquet": str(parquet_path),
                "json": str(json_path),
                "variable_definitions": str(var_def_path),
                "splits_directory": str(splits_dir)
            },
            "splits_created": list(splits.keys())
        }
        
        summary_path = self.output_dir / f"extraction_summary_{self.timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def run_extraction(self):
        """Run the complete feature extraction pipeline"""
        print("🚀 Starting Comprehensive Feature Extraction...")
        print(f"   Generating {self.n_campaigns} synthetic campaigns")
        
        # Generate base campaign data
        print("\n📊 Extracting features from 8 modeling approaches:")
        df = self.generate_campaign_metadata()
        print("   ✓ Campaign metadata")
        
        # Extract all feature categories
        df = self.extract_visual_features(df)
        print("   ✓ Visual features (25+ variables)")
        
        df = self.extract_text_features(df)
        print("   ✓ Text features (15+ variables)")
        
        df = self.extract_audio_features(df)
        print("   ✓ Audio features (10+ variables)")
        
        df = self.extract_campaign_features(df)
        print("   ✓ Campaign features (20+ variables)")
        
        df = self.extract_cultural_features(df)
        print("   ✓ Cultural features (10+ variables)")
        
        df = self.extract_temporal_features(df)
        print("   ✓ Temporal features (8+ variables)")
        
        df = self.extract_network_features(df)
        print("   ✓ Network features (10+ variables)")
        
        df = self.extract_hierarchical_features(df)
        print("   ✓ Hierarchical features (8+ variables)")
        
        df = self.extract_survival_features(df)
        print("   ✓ Survival features (6+ variables)")
        
        # Generate target variables
        df = self.generate_target_variables(df)
        print("   ✓ Target variables (12+ targets)")
        
        print(f"\n✅ Total features extracted: {len(df.columns)}")
        
        # Create variable definitions
        print("\n📋 Creating variable definitions...")
        variable_defs = self.create_variable_definitions(df)
        
        # Create data splits
        print("\n🔄 Creating train/test splits...")
        splits = self.create_data_splits(df)
        print(f"   Created {len(splits)} different split strategies")
        
        # Export everything
        print("\n💾 Exporting data...")
        summary = self.export_data(df, variable_defs, splits)
        
        print("\n✨ Extraction Complete!")
        print(f"\n📁 Output directory: {self.output_dir}")
        print(f"📊 Dataset shape: {summary['dataset_shape'][0]} campaigns × {summary['dataset_shape'][1]} features")
        print(f"🎯 Target variables: {summary['target_variables']} different targets")
        print("\n🚀 Ready for modeling with any approach!")
        
        return df, variable_defs, splits, summary


def main():
    """Run the comprehensive feature extraction"""
    extractor = ComprehensiveFeatureExtractor(n_campaigns=1000)
    df, variable_defs, splits, summary = extractor.run_extraction()
    
    # Print sample of the data
    print("\n📊 Sample of extracted data:")
    print(df.head())
    
    print("\n🎯 Target variable statistics:")
    print(df['creative_effectiveness_score'].describe())
    
    print("\n✅ Extraction complete! You can now use this data with any modeling approach.")


if __name__ == "__main__":
    main()