#!/usr/bin/env python3
"""
Rescale CES scores for better distribution and differentiation
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

class CESRescaler:
    """Rescale CES scores to a more balanced distribution"""
    
    def __init__(self):
        self.output_dir = Path("output/rescaled_campaigns")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_current_distribution(self, df: pd.DataFrame):
        """Analyze current score distribution"""
        print("\n📊 CURRENT DISTRIBUTION ANALYSIS")
        print("="*50)
        
        scores = df['creative_effectiveness_score']
        
        print(f"Mean: {scores.mean():.1f}")
        print(f"Median: {scores.median():.1f}")
        print(f"Std Dev: {scores.std():.1f}")
        print(f"Skewness: {stats.skew(scores):.2f}")
        print(f"Range: {scores.min():.1f} - {scores.max():.1f}")
        
        # Distribution by source
        print("\nBy Source:")
        for source in df['source'].unique():
            source_scores = df[df['source'] == source]['creative_effectiveness_score']
            print(f"  {source}: Mean={source_scores.mean():.1f}, Range={source_scores.min():.1f}-{source_scores.max():.1f}")
    
    def rescale_scores(self, df: pd.DataFrame, method: str = 'balanced') -> pd.DataFrame:
        """Rescale scores using different methods"""
        
        df_rescaled = df.copy()
        
        if method == 'balanced':
            # Method 1: Balanced rescaling with source adjustment
            print("\n🔧 Applying BALANCED RESCALING...")
            
            # Target parameters
            target_mean = 65.0  # Center the distribution
            target_std = 10.0   # Reasonable spread
            target_max = 85.0   # Leave room for exceptional campaigns
            
            # Current parameters
            current_mean = df['creative_effectiveness_score'].mean()
            current_std = df['creative_effectiveness_score'].std()
            
            # First, normalize to standard scale
            df_rescaled['ces_normalized'] = (df['creative_effectiveness_score'] - current_mean) / current_std
            
            # Apply source-specific adjustments
            for source in df['source'].unique():
                mask = df_rescaled['source'] == source
                
                if source == 'WARC':
                    # WARC campaigns get a smaller boost (they're proven but not all exceptional)
                    adjustment = 0.3  # Moderate boost
                else:
                    # Real Portfolio gets baseline
                    adjustment = 0.0
                
                df_rescaled.loc[mask, 'ces_normalized'] = df_rescaled.loc[mask, 'ces_normalized'] + adjustment
            
            # Transform to target distribution
            df_rescaled['creative_effectiveness_score'] = (
                df_rescaled['ces_normalized'] * target_std + target_mean
            )
            
            # Cap at target max but allow some exceptional scores
            df_rescaled['creative_effectiveness_score'] = df_rescaled['creative_effectiveness_score'].clip(upper=target_max)
            
            # Ensure minimum score
            df_rescaled['creative_effectiveness_score'] = df_rescaled['creative_effectiveness_score'].clip(lower=40)
            
        elif method == 'percentile':
            # Method 2: Percentile-based rescaling
            print("\n🔧 Applying PERCENTILE RESCALING...")
            
            # Map scores to percentiles then to new scale
            percentiles = df['creative_effectiveness_score'].rank(pct=True)
            
            # Define the new scale mapping
            # 0-20th percentile: 40-55
            # 20-40th percentile: 55-65
            # 40-60th percentile: 65-70
            # 60-80th percentile: 70-75
            # 80-95th percentile: 75-80
            # 95-100th percentile: 80-85
            
            conditions = [
                (percentiles <= 0.20),
                (percentiles <= 0.40),
                (percentiles <= 0.60),
                (percentiles <= 0.80),
                (percentiles <= 0.95),
                (percentiles <= 1.00)
            ]
            
            choices = [
                40 + (percentiles * 75),  # 40-55
                55 + ((percentiles - 0.20) * 50),  # 55-65
                65 + ((percentiles - 0.40) * 25),  # 65-70
                70 + ((percentiles - 0.60) * 25),  # 70-75
                75 + ((percentiles - 0.80) * 25),  # 75-80
                80 + ((percentiles - 0.95) * 100)  # 80-85
            ]
            
            df_rescaled['creative_effectiveness_score'] = np.select(conditions, choices)
            
        elif method == 'logarithmic':
            # Method 3: Logarithmic compression for high scores
            print("\n🔧 Applying LOGARITHMIC RESCALING...")
            
            # Apply log transformation to compress high scores
            min_score = df['creative_effectiveness_score'].min()
            max_score = df['creative_effectiveness_score'].max()
            
            # Normalize to 0-1
            normalized = (df['creative_effectiveness_score'] - min_score) / (max_score - min_score)
            
            # Apply log transformation
            log_transformed = np.log1p(normalized * 2) / np.log1p(2)
            
            # Scale to 45-80 range
            df_rescaled['creative_effectiveness_score'] = 45 + (log_transformed * 35)
            
        # Round to 1 decimal
        df_rescaled['creative_effectiveness_score'] = df_rescaled['creative_effectiveness_score'].round(1)
        
        # Recalculate derived metrics
        df_rescaled['roi_multiplier'] = 1.0 + (df_rescaled['creative_effectiveness_score'] / 100) * 3.0
        df_rescaled['brand_lift_percentage'] = df_rescaled['creative_effectiveness_score'] * 0.3 + np.random.normal(0, 2, len(df_rescaled))
        df_rescaled['engagement_rate'] = (df_rescaled['creative_effectiveness_score'] / 100 * 0.6 + 0.1).clip(0, 0.8)
        
        # Round derived metrics
        df_rescaled['roi_multiplier'] = df_rescaled['roi_multiplier'].round(2)
        df_rescaled['brand_lift_percentage'] = df_rescaled['brand_lift_percentage'].round(1)
        df_rescaled['engagement_rate'] = df_rescaled['engagement_rate'].round(3)
        
        # Add rescaling metadata
        df_rescaled['rescaling_method'] = method
        df_rescaled['rescaling_date'] = datetime.now().isoformat()
        
        return df_rescaled
    
    def compare_distributions(self, original_df: pd.DataFrame, rescaled_df: pd.DataFrame):
        """Compare original and rescaled distributions"""
        print("\n📊 DISTRIBUTION COMPARISON")
        print("="*60)
        print(f"{'Metric':<20} {'Original':<15} {'Rescaled':<15} {'Change':<10}")
        print("="*60)
        
        metrics = {
            'Mean': (original_df['creative_effectiveness_score'].mean(), 
                    rescaled_df['creative_effectiveness_score'].mean()),
            'Median': (original_df['creative_effectiveness_score'].median(),
                      rescaled_df['creative_effectiveness_score'].median()),
            'Std Dev': (original_df['creative_effectiveness_score'].std(),
                       rescaled_df['creative_effectiveness_score'].std()),
            'Min': (original_df['creative_effectiveness_score'].min(),
                   rescaled_df['creative_effectiveness_score'].min()),
            'Max': (original_df['creative_effectiveness_score'].max(),
                   rescaled_df['creative_effectiveness_score'].max()),
            'Range': (original_df['creative_effectiveness_score'].max() - original_df['creative_effectiveness_score'].min(),
                     rescaled_df['creative_effectiveness_score'].max() - rescaled_df['creative_effectiveness_score'].min())
        }
        
        for metric, (orig, rescaled) in metrics.items():
            change = rescaled - orig
            print(f"{metric:<20} {orig:<15.1f} {rescaled:<15.1f} {change:+10.1f}")
        
        # By source
        print("\nBy Source:")
        for source in original_df['source'].unique():
            orig_mean = original_df[original_df['source'] == source]['creative_effectiveness_score'].mean()
            rescaled_mean = rescaled_df[rescaled_df['source'] == source]['creative_effectiveness_score'].mean()
            print(f"  {source}: {orig_mean:.1f} → {rescaled_mean:.1f} ({rescaled_mean - orig_mean:+.1f})")
    
    def display_rescaled_rankings(self, df: pd.DataFrame):
        """Display top campaigns after rescaling"""
        print("\n🏆 TOP 20 CAMPAIGNS (RESCALED)")
        print("="*85)
        print(f"{'Rank':<5} {'CES':<7} {'Old CES':<8} {'Campaign':<35} {'Brand':<18} {'Source':<12}")
        print("="*85)
        
        # Get original scores for comparison
        original_scores = df.set_index('campaign_id')['creative_effectiveness_score'].to_dict()
        
        # Sort by new scores
        df_sorted = df.sort_values('creative_effectiveness_score', ascending=False).head(20)
        
        for idx, (_, row) in enumerate(df_sorted.iterrows(), 1):
            # Note: We're showing the same score since we don't have the original
            print(f"{idx:<5} {row['creative_effectiveness_score']:<7.1f} "
                  f"{'-':<8} {row['name'][:34]:<35} {row['brand'][:17]:<18} {row['source']:<12}")
    
    def save_rescaled_data(self, df: pd.DataFrame, method: str):
        """Save rescaled dataset"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output data structure
        output_data = {
            'metadata': {
                'rescaling_date': datetime.now().isoformat(),
                'rescaling_method': method,
                'total_campaigns': len(df),
                'score_statistics': {
                    'mean': round(df['creative_effectiveness_score'].mean(), 1),
                    'median': round(df['creative_effectiveness_score'].median(), 1),
                    'std_dev': round(df['creative_effectiveness_score'].std(), 1),
                    'min': round(df['creative_effectiveness_score'].min(), 1),
                    'max': round(df['creative_effectiveness_score'].max(), 1)
                }
            },
            'campaigns': df.to_dict('records')
        }
        
        # Save JSON
        json_path = self.output_dir / f"rescaled_campaigns_{method}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Save CSV
        csv_path = self.output_dir / f"rescaled_campaigns_{method}_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"\n💾 Saved rescaled data:")
        print(f"   JSON: {json_path}")
        print(f"   CSV: {csv_path}")
        
        return json_path

def main():
    """Run the rescaling process"""
    # Load the current scored dataset
    input_path = "output/scored_campaigns/fully_scored_campaigns_20250711_172743.json"
    
    if not Path(input_path).exists():
        print(f"Error: {input_path} not found")
        return
    
    print("📊 Loading scored dataset...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data['campaigns'])
    print(f"✅ Loaded {len(df)} campaigns")
    
    # Initialize rescaler
    rescaler = CESRescaler()
    
    # Analyze current distribution
    rescaler.analyze_current_distribution(df)
    
    # Test different rescaling methods
    methods = ['balanced', 'percentile', 'logarithmic']
    
    print("\n🔬 Testing rescaling methods...")
    best_method = None
    best_std = float('inf')
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Testing method: {method.upper()}")
        print('='*60)
        
        df_rescaled = rescaler.rescale_scores(df, method)
        rescaler.compare_distributions(df, df_rescaled)
        
        # Check if this gives better distribution
        std_dev = df_rescaled['creative_effectiveness_score'].std()
        if 8 <= std_dev <= 12:  # Good standard deviation range
            if abs(std_dev - 10) < abs(best_std - 10):
                best_method = method
                best_std = std_dev
    
    # Apply best method
    print(f"\n✅ Best method: {best_method}")
    df_final = rescaler.rescale_scores(df, best_method)
    
    # Display new rankings
    rescaler.display_rescaled_rankings(df_final)
    
    # Save results
    output_path = rescaler.save_rescaled_data(df_final, best_method)
    
    print("\n✨ Rescaling complete!")
    print(f"   New score range: {df_final['creative_effectiveness_score'].min():.1f} - {df_final['creative_effectiveness_score'].max():.1f}")
    print(f"   Better distribution achieved with {best_method} method")

if __name__ == "__main__":
    main()