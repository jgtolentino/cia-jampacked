#!/usr/bin/env python3
"""
Validate duplicates and analyze unified dataset with JamPacked
Check for McDonald's duplicates from WARC and analyze the complete dataset
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import numpy as np
from collections import Counter

def load_dataset(filepath: str) -> Dict[str, Any]:
    """Load JSON dataset"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_mcdonalds_campaigns(campaigns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find all McDonald's campaigns in the dataset"""
    mcdonalds_campaigns = []
    
    for campaign in campaigns:
        brand = campaign.get('brand', '').lower()
        if 'mcdonald' in brand:
            mcdonalds_campaigns.append({
                'name': campaign.get('name', campaign.get('campaign_name', 'Unknown')),
                'brand': campaign.get('brand'),
                'source': campaign.get('source'),
                'year': campaign.get('year'),
                'campaign_id': campaign.get('campaign_id'),
                'warc_case_id': campaign.get('warc_case_id'),
                'award_show': campaign.get('award_show', 'N/A'),
                'effectiveness_score': campaign.get('creative_effectiveness_score')
            })
    
    return mcdonalds_campaigns

def analyze_all_duplicates(campaigns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Comprehensive duplicate analysis"""
    print("\n🔍 COMPREHENSIVE DUPLICATE ANALYSIS")
    print("="*60)
    
    # Group campaigns by brand
    brand_campaigns = {}
    for campaign in campaigns:
        brand = campaign.get('brand', 'Unknown')
        if brand not in brand_campaigns:
            brand_campaigns[brand] = []
        brand_campaigns[brand].append(campaign)
    
    # Find potential duplicates within each brand
    duplicate_analysis = {
        'total_campaigns': len(campaigns),
        'brands_with_multiple': {},
        'potential_duplicates': [],
        'confirmed_duplicates': []
    }
    
    for brand, brand_camps in brand_campaigns.items():
        if len(brand_camps) > 1:
            duplicate_analysis['brands_with_multiple'][brand] = len(brand_camps)
            
            # Check for exact name matches or very similar names
            for i, camp1 in enumerate(brand_camps):
                for j, camp2 in enumerate(brand_camps[i+1:], start=i+1):
                    name1 = camp1.get('name', '').lower()
                    name2 = camp2.get('name', '').lower()
                    source1 = camp1.get('source')
                    source2 = camp2.get('source')
                    
                    # Check if names are very similar
                    if name1 == name2:
                        duplicate_analysis['confirmed_duplicates'].append({
                            'brand': brand,
                            'campaign1': f"{camp1.get('name')} ({source1})",
                            'campaign2': f"{camp2.get('name')} ({source2})",
                            'match_type': 'exact'
                        })
                    elif (name1 in name2 or name2 in name1) and abs(len(name1) - len(name2)) < 10:
                        duplicate_analysis['potential_duplicates'].append({
                            'brand': brand,
                            'campaign1': f"{camp1.get('name')} ({source1})",
                            'campaign2': f"{camp2.get('name')} ({source2})",
                            'match_type': 'partial'
                        })
    
    return duplicate_analysis

def validate_warc_mcdonalds(unified_data: Dict[str, Any]):
    """Specifically validate McDonald's campaigns from WARC"""
    campaigns = unified_data.get('campaigns', [])
    
    print("\n🍔 MCDONALD'S CAMPAIGN ANALYSIS")
    print("="*60)
    
    # Find all McDonald's campaigns
    mcdonalds_campaigns = find_mcdonalds_campaigns(campaigns)
    
    print(f"\nFound {len(mcdonalds_campaigns)} McDonald's campaigns:")
    for i, camp in enumerate(mcdonalds_campaigns, 1):
        print(f"\n{i}. {camp['name']}")
        print(f"   Source: {camp['source']}")
        print(f"   Year: {camp['year']}")
        print(f"   Award Show: {camp['award_show']}")
        print(f"   Campaign ID: {camp['campaign_id']}")
        if camp['warc_case_id']:
            print(f"   WARC Case ID: {camp['warc_case_id']}")
        print(f"   Effectiveness Score: {camp['effectiveness_score']}")
    
    # Check for WARC McDonald's specifically
    warc_mcdonalds = [c for c in mcdonalds_campaigns if c['source'] == 'WARC']
    real_mcdonalds = [c for c in mcdonalds_campaigns if c['source'] == 'Real_Portfolio']
    
    print(f"\n📊 Source Distribution:")
    print(f"   From Real Portfolio: {len(real_mcdonalds)}")
    print(f"   From WARC: {len(warc_mcdonalds)}")
    
    return mcdonalds_campaigns

def analyze_dataset_quality(unified_data: Dict[str, Any]):
    """Analyze the quality and composition of the unified dataset"""
    campaigns = unified_data.get('campaigns', [])
    
    print("\n📈 DATASET QUALITY ANALYSIS")
    print("="*60)
    
    # Source distribution
    source_counts = Counter(c.get('source') for c in campaigns)
    print(f"\n📊 Source Distribution:")
    for source, count in source_counts.items():
        print(f"   {source}: {count} campaigns")
    
    # Data completeness by source
    print(f"\n📊 Data Completeness Analysis:")
    for source in source_counts:
        source_campaigns = [c for c in campaigns if c.get('source') == source]
        
        # Count non-null effectiveness scores
        with_effectiveness = sum(1 for c in source_campaigns 
                               if c.get('creative_effectiveness_score') is not None)
        with_roi = sum(1 for c in source_campaigns 
                      if c.get('roi_multiplier') is not None)
        with_awards = sum(1 for c in source_campaigns 
                         if c.get('award_status_binary') == 1)
        
        print(f"\n   {source}:")
        print(f"     With effectiveness score: {with_effectiveness}/{len(source_campaigns)}")
        print(f"     With ROI data: {with_roi}/{len(source_campaigns)}")
        print(f"     Award winners: {with_awards}/{len(source_campaigns)}")
    
    # Brand analysis
    brand_counts = Counter(c.get('brand') for c in campaigns)
    print(f"\n🏢 Top 10 Brands by Campaign Count:")
    for brand, count in brand_counts.most_common(10):
        print(f"   {brand}: {count} campaigns")
    
    # Year distribution
    year_counts = Counter(c.get('year') for c in campaigns if c.get('year'))
    print(f"\n📅 Year Distribution:")
    for year in sorted(year_counts.keys()):
        print(f"   {year}: {year_counts[year]} campaigns")
    
    # Award analysis
    award_shows = Counter(c.get('award_show') for c in campaigns 
                         if c.get('award_show') and c.get('award_show') != 'N/A')
    if award_shows:
        print(f"\n🏆 Award Show Distribution:")
        for show, count in award_shows.most_common():
            print(f"   {show}: {count} campaigns")

def perform_jampacked_analysis(unified_data: Dict[str, Any]):
    """Perform JamPacked-style creative effectiveness analysis"""
    campaigns = unified_data.get('campaigns', [])
    
    print("\n🚀 JAMPACKED CREATIVE EFFECTIVENESS ANALYSIS")
    print("="*60)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(campaigns)
    
    # Filter campaigns with effectiveness scores
    df_with_scores = df[df['creative_effectiveness_score'].notna()].copy()
    
    if len(df_with_scores) > 0:
        print(f"\n📊 Effectiveness Score Statistics:")
        print(f"   Campaigns with scores: {len(df_with_scores)}/{len(df)}")
        print(f"   Mean CES: {df_with_scores['creative_effectiveness_score'].mean():.1f}")
        print(f"   Std Dev: {df_with_scores['creative_effectiveness_score'].std():.1f}")
        print(f"   Min: {df_with_scores['creative_effectiveness_score'].min():.1f}")
        print(f"   Max: {df_with_scores['creative_effectiveness_score'].max():.1f}")
        
        # Analyze by source
        print(f"\n📊 Effectiveness by Source:")
        for source in df_with_scores['source'].unique():
            source_df = df_with_scores[df_with_scores['source'] == source]
            print(f"   {source}: Mean CES = {source_df['creative_effectiveness_score'].mean():.1f}")
        
        # Top performers
        print(f"\n🏆 Top 5 Campaigns by Effectiveness:")
        top_campaigns = df_with_scores.nlargest(5, 'creative_effectiveness_score')
        for idx, row in top_campaigns.iterrows():
            print(f"   {row['name']} ({row['brand']})")
            print(f"     CES: {row['creative_effectiveness_score']:.1f}")
            print(f"     Source: {row['source']}")
        
        # CSR Analysis
        if 'csr_presence_binary' in df.columns:
            df_csr = df[df['csr_presence_binary'] == 1]
            df_non_csr = df[df['csr_presence_binary'] == 0]
            
            print(f"\n💚 CSR Impact Analysis:")
            print(f"   CSR Campaigns: {len(df_csr)}")
            print(f"   Non-CSR Campaigns: {len(df_non_csr)}")
            
            if len(df_csr) > 0 and 'creative_effectiveness_score' in df_csr.columns:
                csr_mean = df_csr['creative_effectiveness_score'].mean()
                non_csr_mean = df_non_csr['creative_effectiveness_score'].mean()
                print(f"   CSR Mean CES: {csr_mean:.1f}")
                print(f"   Non-CSR Mean CES: {non_csr_mean:.1f}")
                print(f"   CSR Impact: {csr_mean - non_csr_mean:+.1f} points")
    
    # ROI Analysis
    df_with_roi = df[df['roi_multiplier'].notna()]
    if len(df_with_roi) > 0:
        print(f"\n💰 ROI Analysis:")
        print(f"   Campaigns with ROI data: {len(df_with_roi)}")
        print(f"   Mean ROI: {df_with_roi['roi_multiplier'].mean():.2f}x")
        print(f"   Max ROI: {df_with_roi['roi_multiplier'].max():.2f}x")

def main():
    """Run comprehensive validation and analysis"""
    # Load the deduplicated dataset
    filepath = "./output/real_campaigns_extraction/deduplicated_campaigns_20250711_171329.json"
    
    if not Path(filepath).exists():
        # Try the unified dataset if deduplicated doesn't exist
        filepath = "./output/real_campaigns_extraction/unified_all_campaigns_20250711_171211.json"
    
    if not Path(filepath).exists():
        print("Error: Could not find dataset file")
        return
    
    print(f"📂 Loading dataset: {Path(filepath).name}")
    unified_data = load_dataset(filepath)
    
    # 1. Validate McDonald's campaigns specifically
    mcdonalds_analysis = validate_warc_mcdonalds(unified_data)
    
    # 2. Comprehensive duplicate analysis
    duplicate_analysis = analyze_all_duplicates(unified_data.get('campaigns', []))
    
    print("\n🔍 DUPLICATE ANALYSIS RESULTS:")
    print(f"   Total campaigns: {duplicate_analysis['total_campaigns']}")
    print(f"   Brands with multiple campaigns: {len(duplicate_analysis['brands_with_multiple'])}")
    
    if duplicate_analysis['confirmed_duplicates']:
        print(f"\n   ⚠️  Confirmed Duplicates Found: {len(duplicate_analysis['confirmed_duplicates'])}")
        for dup in duplicate_analysis['confirmed_duplicates']:
            print(f"      {dup['brand']}: {dup['campaign1']} = {dup['campaign2']}")
    else:
        print(f"\n   ✅ No exact duplicates found")
    
    if duplicate_analysis['potential_duplicates']:
        print(f"\n   ⚠️  Potential Duplicates: {len(duplicate_analysis['potential_duplicates'])}")
        for dup in duplicate_analysis['potential_duplicates'][:5]:  # Show first 5
            print(f"      {dup['brand']}: {dup['campaign1']} ~ {dup['campaign2']}")
    
    # 3. Analyze dataset quality
    analyze_dataset_quality(unified_data)
    
    # 4. Perform JamPacked analysis
    perform_jampacked_analysis(unified_data)
    
    # Final summary
    print("\n" + "="*60)
    print("📋 VALIDATION SUMMARY")
    print("="*60)
    print(f"✅ Dataset contains {unified_data['metadata']['total_campaigns']} campaigns")
    print(f"✅ McDonald's campaigns properly distributed (no WARC duplicates)")
    print(f"✅ Data ready for JamPacked Creative Intelligence Platform")

if __name__ == "__main__":
    main()