#!/usr/bin/env python3
"""
Remove duplicate campaigns from unified dataset
Identifies duplicates by brand + campaign name similarity
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from difflib import SequenceMatcher
import re

def normalize_name(name: str) -> str:
    """Normalize campaign name for comparison"""
    # Convert to lowercase
    name = name.lower()
    # Remove special characters
    name = re.sub(r'[^a-z0-9\s]', ' ', name)
    # Remove extra spaces
    name = ' '.join(name.split())
    return name

def calculate_similarity(name1: str, name2: str) -> float:
    """Calculate similarity between two campaign names"""
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)
    return SequenceMatcher(None, norm1, norm2).ratio()

def find_duplicates(campaigns: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """Find duplicate campaigns based on brand and name similarity"""
    duplicates = {}
    processed = set()
    
    print("🔍 Searching for duplicates...")
    
    for i, camp1 in enumerate(campaigns):
        if i in processed:
            continue
            
        brand1 = camp1.get('brand', '').lower()
        name1 = camp1.get('name', camp1.get('campaign_name', ''))
        
        # Find similar campaigns
        similar_indices = [i]
        
        for j, camp2 in enumerate(campaigns[i+1:], start=i+1):
            if j in processed:
                continue
                
            brand2 = camp2.get('brand', '').lower()
            name2 = camp2.get('name', camp2.get('campaign_name', ''))
            
            # Check if same brand
            if brand1 == brand2 or calculate_similarity(brand1, brand2) > 0.85:
                # Check name similarity
                name_similarity = calculate_similarity(name1, name2)
                
                # Also check for exact substring matches
                norm1 = normalize_name(name1)
                norm2 = normalize_name(name2)
                is_substring = norm1 in norm2 or norm2 in norm1
                
                if name_similarity > 0.75 or is_substring:
                    similar_indices.append(j)
                    processed.add(j)
        
        if len(similar_indices) > 1:
            key = f"{brand1}_{normalize_name(name1)[:20]}"
            duplicates[key] = similar_indices
            processed.update(similar_indices)
    
    return duplicates

def select_best_version(campaigns: List[Dict[str, Any]], indices: List[int]) -> int:
    """Select the most complete version from duplicate campaigns"""
    best_idx = indices[0]
    best_score = 0
    
    for idx in indices:
        campaign = campaigns[idx]
        score = 0
        
        # Score based on data completeness
        # Count non-null, non-empty fields
        for key, value in campaign.items():
            if value is not None and value != "" and value != "Unknown":
                score += 1
                # Extra points for important fields
                if key in ['creative_effectiveness_score', 'roi_multiplier', 'sales_lift_percentage']:
                    score += 2
                if key == 'original_case_study':  # WARC full case study
                    score += 5
        
        # Prefer WARC source for detailed data
        if campaign.get('source') == 'WARC':
            score += 3
        
        # Prefer more recent campaigns
        year = campaign.get('year', 0)
        if isinstance(year, (int, float)):
            score += (year - 2000) * 0.1
        
        if score > best_score:
            best_score = score
            best_idx = idx
    
    return best_idx

def remove_duplicates(input_file: str, output_file: str = None):
    """Remove duplicates from unified campaign file"""
    
    print(f"📂 Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    campaigns = data.get('campaigns', [])
    original_count = len(campaigns)
    print(f"✅ Loaded {original_count} campaigns")
    
    # Find duplicates
    duplicates = find_duplicates(campaigns)
    
    if not duplicates:
        print("✨ No duplicates found!")
        return data
    
    print(f"\n🔍 Found {len(duplicates)} groups of duplicates:")
    
    # Show duplicate groups
    all_duplicate_indices = set()
    for group_key, indices in duplicates.items():
        all_duplicate_indices.update(indices)
        print(f"\n  Group: {group_key}")
        for idx in indices:
            camp = campaigns[idx]
            print(f"    - [{idx}] {camp.get('name', 'Unknown')} ({camp.get('source', 'Unknown')}, {camp.get('year', 'Unknown')})")
    
    # Remove duplicates
    print("\n🧹 Removing duplicates (keeping best version of each)...")
    
    # Create new campaign list
    deduplicated_campaigns = []
    indices_to_skip = set()
    
    # First, add the best version from each duplicate group
    for group_key, indices in duplicates.items():
        best_idx = select_best_version(campaigns, indices)
        deduplicated_campaigns.append(campaigns[best_idx])
        indices_to_skip.update(indices)
        print(f"  Kept campaign [{best_idx}] from group {group_key}")
    
    # Then add all non-duplicate campaigns
    for i, campaign in enumerate(campaigns):
        if i not in indices_to_skip:
            deduplicated_campaigns.append(campaign)
    
    # Update data
    data['campaigns'] = deduplicated_campaigns
    
    # Update metadata
    removed_count = original_count - len(deduplicated_campaigns)
    data['metadata']['total_campaigns'] = len(deduplicated_campaigns)
    data['metadata']['deduplication'] = {
        'original_count': original_count,
        'duplicates_removed': removed_count,
        'final_count': len(deduplicated_campaigns),
        'deduplication_date': datetime.now().isoformat()
    }
    
    # Recalculate statistics
    print("\n📊 Recalculating statistics...")
    
    # Source distribution
    source_counts = {}
    for campaign in deduplicated_campaigns:
        source = campaign.get('source', 'Unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    # Brand distribution
    brand_counts = {}
    for campaign in deduplicated_campaigns:
        brand = campaign.get('brand', 'Unknown')
        brand_counts[brand] = brand_counts.get(brand, 0) + 1
    
    # Update statistics
    data['metadata']['statistics']['source_distribution'] = source_counts
    data['metadata']['statistics']['total_brands'] = len(brand_counts)
    data['metadata']['statistics']['top_brands'] = sorted(
        brand_counts.items(), key=lambda x: x[1], reverse=True
    )[:10]
    
    # Save deduplicated file
    if not output_file:
        output_file = Path(input_file).parent / f"deduplicated_campaigns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Also create CSV version
    csv_path = Path(output_file).with_suffix('.csv')
    print(f"\n📊 Creating CSV version...")
    
    import pandas as pd
    csv_data = []
    for campaign in deduplicated_campaigns:
        csv_row = {k: v for k, v in campaign.items() 
                   if not isinstance(v, (dict, list)) or k in ['channel_mix']}
        if 'channel_mix' in csv_row and isinstance(csv_row['channel_mix'], list):
            csv_row['channel_mix'] = '|'.join(csv_row['channel_mix'])
        csv_data.append(csv_row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("✅ DEDUPLICATION COMPLETE")
    print("="*60)
    print(f"Original campaigns: {original_count}")
    print(f"Duplicates removed: {removed_count}")
    print(f"Final campaigns: {len(deduplicated_campaigns)}")
    print(f"\nSource distribution:")
    for source, count in source_counts.items():
        print(f"  - {source}: {count}")
    print(f"\nTop 5 brands:")
    for brand, count in data['metadata']['statistics']['top_brands'][:5]:
        print(f"  - {brand}: {count} campaigns")
    
    print(f"\n📁 Saved to: {output_file}")
    print(f"📊 CSV saved to: {csv_path}")
    
    return data

def main():
    """Run deduplication on unified campaign file"""
    input_file = "./output/real_campaigns_extraction/unified_all_campaigns_20250711_171211.json"
    
    if not Path(input_file).exists():
        print(f"Error: {input_file} not found")
        return
    
    remove_duplicates(input_file)

if __name__ == "__main__":
    main()