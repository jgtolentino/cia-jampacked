#!/usr/bin/env python3
"""
Display all campaigns ranked by CES score
"""

import json
import pandas as pd
from pathlib import Path

def display_all_ces_scores():
    """Display all campaigns ranked by CES"""
    
    # Load the fully scored dataset
    json_path = 'output/scored_campaigns/fully_scored_campaigns_20250711_172743.json'
    
    if not Path(json_path).exists():
        print(f"Error: {json_path} not found")
        return
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data['campaigns'])
    
    # Sort by CES score descending
    df_sorted = df.sort_values('creative_effectiveness_score', ascending=False).reset_index(drop=True)
    
    # Print header
    print('🏆 ALL 52 CAMPAIGNS RANKED BY CES (Creative Effectiveness Score)')
    print('='*85)
    print(f"{'Rank':<5} {'CES':<7} {'Campaign Name':<40} {'Brand':<20} {'Source':<15}")
    print('='*85)
    
    # Print all campaigns
    for idx, row in df_sorted.iterrows():
        rank = idx + 1
        ces = row['creative_effectiveness_score']
        name = row['name'][:39]
        brand = row['brand'][:19]
        source = row['source']
        print(f"{rank:<5} {ces:<7.1f} {name:<40} {brand:<20} {source:<15}")
    
    # Score distribution
    print('\n📊 SCORE DISTRIBUTION')
    print('='*50)
    
    ranges = [
        (80, 100, 'Exceptional'),
        (70, 79.9, 'Excellent'),
        (60, 69.9, 'Good'),
        (50, 59.9, 'Average'),
        (0, 49.9, 'Below Average')
    ]
    
    for low, high, label in ranges:
        count = len(df[(df['creative_effectiveness_score'] >= low) & 
                      (df['creative_effectiveness_score'] <= high)])
        if count > 0:
            print(f"{label} ({low:.0f}-{high:.0f}): {count} campaigns")
    
    # Summary statistics by source
    print('\n📈 SUMMARY BY SOURCE')
    print('='*50)
    
    for source in sorted(df['source'].unique()):
        source_df = df[df['source'] == source]
        print(f"\n{source}:")
        print(f"  Count: {len(source_df)}")
        print(f"  Mean CES: {source_df['creative_effectiveness_score'].mean():.1f}")
        print(f"  Range: {source_df['creative_effectiveness_score'].min():.1f} - {source_df['creative_effectiveness_score'].max():.1f}")
        
        # Top 3 for this source
        print(f"  Top 3:")
        top3 = source_df.nlargest(3, 'creative_effectiveness_score')
        for _, row in top3.iterrows():
            print(f"    {row['creative_effectiveness_score']:.1f} - {row['name']}")
    
    # Overall statistics
    print('\n📊 OVERALL STATISTICS')
    print('='*50)
    print(f"Total Campaigns: {len(df)}")
    print(f"Mean CES: {df['creative_effectiveness_score'].mean():.1f}")
    print(f"Median CES: {df['creative_effectiveness_score'].median():.1f}")
    print(f"Std Dev: {df['creative_effectiveness_score'].std():.1f}")
    print(f"Range: {df['creative_effectiveness_score'].min():.1f} - {df['creative_effectiveness_score'].max():.1f}")
    
    # Percentile breakdown
    print('\n📊 PERCENTILE BREAKDOWN')
    print('='*50)
    percentiles = [10, 25, 50, 75, 90, 95]
    for p in percentiles:
        value = df['creative_effectiveness_score'].quantile(p/100)
        print(f"{p}th percentile: {value:.1f}")

if __name__ == "__main__":
    display_all_ces_scores()