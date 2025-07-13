#!/usr/bin/env python3
"""
Merge Real Campaigns Features with WARC Case Studies
Creates a unified dataset combining all campaign data
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load JSON file safely"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return None

def extract_warc_features(warc_case: Dict[str, Any]) -> Dict[str, Any]:
    """Extract standardized features from WARC case study"""
    features = {}
    
    # Extract from campaign_identification
    if 'campaign_identification' in warc_case:
        camp_id = warc_case['campaign_identification']
        features['campaign_name'] = camp_id.get('campaign_name', camp_id.get('campaign_title', 'Unknown'))
        features['brand'] = camp_id.get('brand', 'Unknown')
        features['year'] = camp_id.get('year', 2024)
        features['agency'] = camp_id.get('agency', 'Unknown')
    
    # Extract from market_context
    if 'market_context' in warc_case:
        market = warc_case['market_context']
        features['country'] = market.get('country', 'Unknown')
        features['region'] = market.get('region', 'Unknown')
        features['industry'] = market.get('industry', market.get('category', 'Unknown'))
        features['market_share_before'] = market.get('market_share_before', None)
    
    # Extract from business_challenge
    if 'business_challenge' in warc_case:
        challenge = warc_case['business_challenge']
        features['challenge_type'] = challenge.get('primary_challenge', 'Unknown')
        features['competitive_pressure'] = 1 if 'competitive' in str(challenge).lower() else 0
    
    # Extract from strategy_execution
    if 'strategy_execution' in warc_case:
        strategy = warc_case['strategy_execution']
        features['strategy_type'] = strategy.get('approach', 'Unknown')
        features['target_audience'] = strategy.get('target_audience', 'Unknown')
    
    # Extract from creative_elements
    if 'creative_elements' in warc_case:
        creative = warc_case['creative_elements']
        features['creative_concept'] = creative.get('central_idea', 'Unknown')
        features['has_humor'] = 1 if 'humor' in str(creative).lower() else 0
        features['has_emotion'] = 1 if any(word in str(creative).lower() for word in ['emotion', 'feel', 'heart']) else 0
    
    # Extract from channels_media
    if 'channels_media' in warc_case:
        channels = warc_case['channels_media']
        features['primary_channel'] = channels.get('primary_channel', 'Unknown')
        features['channel_mix'] = channels.get('channels_used', [])
        features['channel_diversity'] = len(features['channel_mix']) if isinstance(features['channel_mix'], list) else 0
    
    # Extract from technology_integration
    if 'technology_integration' in warc_case:
        tech = warc_case['technology_integration']
        features['has_tech_innovation'] = 1 if tech.get('innovative_tech_used', False) else 0
        features['tech_type'] = tech.get('technology_type', 'None')
    
    # Extract from csr_purpose_elements
    if 'csr_purpose_elements' in warc_case:
        csr = warc_case['csr_purpose_elements']
        features['has_csr'] = 1 if csr.get('has_csr_element', False) else 0
        features['csr_type'] = csr.get('csr_category', 'None')
    
    # Extract from performance_metrics
    if 'performance_metrics' in warc_case:
        metrics = warc_case['performance_metrics']
        features['roi'] = metrics.get('roi', metrics.get('romi', None))
        features['sales_lift'] = metrics.get('sales_increase_percent', None)
        features['brand_lift'] = metrics.get('brand_metrics', {}).get('awareness_lift', None)
        features['market_share_gain'] = metrics.get('market_share_gain', None)
        features['effectiveness_score'] = metrics.get('effectiveness_score', None)
    
    # Extract from awards_recognition
    if 'awards_recognition' in warc_case:
        awards = warc_case['awards_recognition']
        features['total_awards'] = len(awards.get('awards', [])) if isinstance(awards.get('awards'), list) else 0
        features['has_warc_award'] = 1 if 'warc' in str(awards).lower() else 0
        features['has_cannes_award'] = 1 if 'cannes' in str(awards).lower() else 0
    
    # Add WARC-specific identifier
    features['source'] = 'WARC'
    features['warc_case_id'] = warc_case.get('case_id', 'Unknown')
    features['warc_file_id'] = warc_case.get('warc_file_id', 'Unknown')
    
    return features

def merge_campaign_data(real_campaigns_path: str, warc_json_path: str, output_path: str = None):
    """Merge real campaigns with WARC case studies"""
    
    print("🔄 Loading campaign data files...")
    
    # Load real campaigns
    real_data = load_json_file(real_campaigns_path)
    if not real_data:
        print("Error: Could not load real campaigns file")
        return None
    
    # Load WARC data
    warc_data = load_json_file(warc_json_path)
    if not warc_data:
        print("Error: Could not load WARC file")
        return None
    
    # Extract campaigns from real data
    real_campaigns = real_data.get('campaigns', [])
    print(f"✅ Loaded {len(real_campaigns)} real campaigns")
    
    # Extract and transform WARC cases
    warc_cases = warc_data.get('case_studies', [])
    print(f"✅ Loaded {len(warc_cases)} WARC case studies")
    
    # Create unified structure
    unified_data = {
        "metadata": {
            "creation_date": datetime.now().isoformat(),
            "total_campaigns": 0,
            "sources": {
                "real_campaigns": len(real_campaigns),
                "warc_cases": len(warc_cases)
            },
            "data_types": ["Real Portfolio Campaigns", "WARC Effective 100 Cases"],
            "feature_categories": [
                "campaign_identification",
                "performance_metrics",
                "creative_features",
                "award_recognition",
                "market_context",
                "technology_integration",
                "csr_elements"
            ]
        },
        "campaigns": []
    }
    
    # Add real campaigns (they already have standardized features)
    print("\n📊 Processing real campaigns...")
    for campaign in real_campaigns:
        # Add source identifier
        campaign['source'] = 'Real_Portfolio'
        campaign['data_completeness'] = 'extracted_features'
        unified_data['campaigns'].append(campaign)
    
    # Transform and add WARC cases
    print("\n📊 Processing WARC cases...")
    for i, warc_case in enumerate(warc_cases):
        print(f"  Processing WARC case {i+1}/{len(warc_cases)}", end='\r')
        
        # Extract standardized features
        warc_features = extract_warc_features(warc_case)
        
        # Create campaign entry
        campaign_entry = {
            'campaign_id': f"WARC_{warc_features.get('brand', 'Unknown').replace(' ', '_')}_{warc_features.get('year', 2024)}",
            'name': warc_features.get('campaign_name', 'Unknown'),
            'brand': warc_features.get('brand', 'Unknown'),
            'year': warc_features.get('year', 2024),
            'source': 'WARC',
            'data_completeness': 'full_case_study',
            
            # Basic features
            'country': warc_features.get('country', 'Unknown'),
            'region': warc_features.get('region', 'Unknown'),
            'industry': warc_features.get('industry', 'Unknown'),
            'agency': warc_features.get('agency', 'Unknown'),
            
            # Performance metrics
            'creative_effectiveness_score': warc_features.get('effectiveness_score', None),
            'roi_multiplier': warc_features.get('roi', None),
            'brand_lift_percentage': warc_features.get('brand_lift', None),
            'sales_lift_percentage': warc_features.get('sales_lift', None),
            'market_share_gain': warc_features.get('market_share_gain', None),
            
            # Creative features
            'has_humor': warc_features.get('has_humor', 0),
            'has_emotion': warc_features.get('has_emotion', 0),
            'creative_concept': warc_features.get('creative_concept', 'Unknown'),
            
            # Channel features
            'primary_channel': warc_features.get('primary_channel', 'Unknown'),
            'channel_diversity': warc_features.get('channel_diversity', 0),
            'channel_mix': warc_features.get('channel_mix', []),
            
            # Innovation features
            'has_tech_innovation': warc_features.get('has_tech_innovation', 0),
            'innovation_level': 0.8 if warc_features.get('has_tech_innovation') else 0.3,
            'technology_type': warc_features.get('tech_type', 'None'),
            
            # CSR features
            'csr_presence_binary': warc_features.get('has_csr', 0),
            'csr_category': warc_features.get('csr_type', 'None'),
            
            # Award features
            'award_status_binary': 1,  # All WARC cases are award winners
            'total_awards': warc_features.get('total_awards', 1),
            'has_warc_award': warc_features.get('has_warc_award', 1),
            'has_cannes_award': warc_features.get('has_cannes_award', 0),
            
            # WARC identifiers
            'warc_case_id': warc_features.get('warc_case_id'),
            'warc_file_id': warc_features.get('warc_file_id'),
            
            # Store original case study reference
            'original_case_study': warc_case
        }
        
        unified_data['campaigns'].append(campaign_entry)
    
    print(f"\n✅ Processed all WARC cases")
    
    # Update metadata
    unified_data['metadata']['total_campaigns'] = len(unified_data['campaigns'])
    
    # Calculate statistics
    print("\n📈 Calculating unified statistics...")
    
    # Source distribution
    source_counts = {}
    for campaign in unified_data['campaigns']:
        source = campaign.get('source', 'Unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    # Year distribution
    year_counts = {}
    for campaign in unified_data['campaigns']:
        year = campaign.get('year', 'Unknown')
        year_counts[year] = year_counts.get(year, 0) + 1
    
    # Brand distribution
    brand_counts = {}
    for campaign in unified_data['campaigns']:
        brand = campaign.get('brand', 'Unknown')
        brand_counts[brand] = brand_counts.get(brand, 0) + 1
    
    # Award distribution
    award_campaigns = len([c for c in unified_data['campaigns'] if c.get('award_status_binary', 0) == 1])
    csr_campaigns = len([c for c in unified_data['campaigns'] if c.get('csr_presence_binary', 0) == 1])
    
    # Add statistics to metadata
    unified_data['metadata']['statistics'] = {
        'source_distribution': source_counts,
        'year_distribution': dict(sorted(year_counts.items())),
        'total_brands': len(brand_counts),
        'top_brands': sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        'award_winning_campaigns': award_campaigns,
        'csr_campaigns': csr_campaigns,
        'campaigns_with_performance_data': len([c for c in unified_data['campaigns'] 
                                               if c.get('roi_multiplier') is not None or 
                                               c.get('sales_lift_percentage') is not None])
    }
    
    # Save unified file
    if not output_path:
        output_path = Path(real_campaigns_path).parent / f"unified_all_campaigns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unified_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 UNIFIED CAMPAIGN DATASET CREATED")
    print("="*60)
    print(f"Total campaigns: {unified_data['metadata']['total_campaigns']}")
    print(f"  - Real portfolio: {source_counts.get('Real_Portfolio', 0)}")
    print(f"  - WARC cases: {source_counts.get('WARC', 0)}")
    print(f"\nTotal brands: {unified_data['metadata']['statistics']['total_brands']}")
    print(f"Award-winning campaigns: {award_campaigns}")
    print(f"CSR campaigns: {csr_campaigns}")
    print(f"\nTop 5 brands:")
    for brand, count in unified_data['metadata']['statistics']['top_brands'][:5]:
        print(f"  - {brand}: {count} campaigns")
    
    print(f"\n✅ Unified file saved to: {output_path}")
    print(f"   File size: {Path(output_path).stat().st_size:,} bytes")
    
    # Also create a CSV version for easy analysis
    csv_path = Path(output_path).with_suffix('.csv')
    print(f"\n📊 Creating CSV version...")
    
    import pandas as pd
    # Extract key fields for CSV (excluding nested objects)
    csv_data = []
    for campaign in unified_data['campaigns']:
        csv_row = {k: v for k, v in campaign.items() 
                   if not isinstance(v, (dict, list)) or k in ['channel_mix']}
        # Convert lists to strings for CSV
        if 'channel_mix' in csv_row and isinstance(csv_row['channel_mix'], list):
            csv_row['channel_mix'] = '|'.join(csv_row['channel_mix'])
        csv_data.append(csv_row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV saved to: {csv_path}")
    
    return unified_data

def main():
    """Run the merger"""
    # File paths
    real_campaigns_file = "./output/real_campaigns_extraction/real_campaigns_features_20250711_161331.json"
    warc_file = "./output/real_campaigns_extraction/merged_warc_json_20250711_171027.json"
    
    # Check if files exist
    if not Path(real_campaigns_file).exists():
        print(f"Error: {real_campaigns_file} not found")
        return
    
    if not Path(warc_file).exists():
        print(f"Error: {warc_file} not found")
        return
    
    # Merge the files
    merge_campaign_data(real_campaigns_file, warc_file)

if __name__ == "__main__":
    main()