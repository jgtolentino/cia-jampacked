#!/usr/bin/env python3
"""
Merge WARC JSON files
Combines case studies from multiple WARC JSON exports
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load JSON file safely"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return None

def merge_warc_jsons(file1_path: str, file2_path: str, output_path: str = None):
    """Merge two WARC JSON files, removing duplicates"""
    
    # Load both files
    print(f"Loading {file1_path}...")
    data1 = load_json_file(file1_path)
    
    print(f"Loading {file2_path}...")
    data2 = load_json_file(file2_path)
    
    if not data1 or not data2:
        print("Error: Could not load one or both files")
        return None
    
    # Create merged structure
    merged_data = {
        "metadata": {
            "extraction_date": datetime.now().strftime("%Y-%m-%d"),
            "merge_date": datetime.now().isoformat(),
            "source_files": [
                Path(file1_path).name,
                Path(file2_path).name
            ],
            "source_authority": "WARC Effective 100 (2021-2025) + WARC Awards for Effectiveness",
            "feature_categories": []
        },
        "case_studies": []
    }
    
    # Merge metadata feature categories
    categories_set = set()
    if 'metadata' in data1 and 'feature_categories' in data1['metadata']:
        categories_set.update(data1['metadata']['feature_categories'])
    if 'metadata' in data2 and 'feature_categories' in data2['metadata']:
        categories_set.update(data2['metadata']['feature_categories'])
    merged_data['metadata']['feature_categories'] = sorted(list(categories_set))
    
    # Track unique case studies by campaign name
    unique_cases = {}
    
    # Process case studies from both files
    for data in [data1, data2]:
        if 'case_studies' in data:
            for case in data['case_studies']:
                # Use campaign_name as unique identifier
                campaign_name = case.get('campaign_name', case.get('campaign', 'Unknown'))
                
                # If we haven't seen this campaign, add it
                if campaign_name not in unique_cases:
                    unique_cases[campaign_name] = case
                else:
                    # Merge data if needed (keep the more complete version)
                    existing = unique_cases[campaign_name]
                    
                    # Compare completeness (count non-null fields)
                    existing_fields = sum(1 for v in existing.values() if v is not None and v != "")
                    new_fields = sum(1 for v in case.values() if v is not None and v != "")
                    
                    if new_fields > existing_fields:
                        unique_cases[campaign_name] = case
                    elif new_fields == existing_fields:
                        # Merge fields, preferring non-null values
                        for key, value in case.items():
                            if value is not None and value != "" and (existing.get(key) is None or existing.get(key) == ""):
                                existing[key] = value
    
    # Convert back to list
    merged_data['case_studies'] = list(unique_cases.values())
    
    # Update metadata counts
    merged_data['metadata']['total_cases'] = len(merged_data['case_studies'])
    merged_data['metadata']['unique_campaigns'] = len(set(
        case.get('campaign_name', case.get('campaign', 'Unknown')) 
        for case in merged_data['case_studies']
    ))
    
    # Stats from original files
    stats1_count = len(data1.get('case_studies', []))
    stats2_count = len(data2.get('case_studies', []))
    merged_count = len(merged_data['case_studies'])
    
    print(f"\nMerge Statistics:")
    print(f"  File 1: {stats1_count} case studies")
    print(f"  File 2: {stats2_count} case studies")
    print(f"  Merged: {merged_count} unique case studies")
    print(f"  Duplicates removed: {stats1_count + stats2_count - merged_count}")
    
    # Calculate additional statistics
    if merged_data['case_studies']:
        # Brand distribution
        brands = {}
        for case in merged_data['case_studies']:
            brand = case.get('brand', 'Unknown')
            brands[brand] = brands.get(brand, 0) + 1
        
        # Year distribution
        years = {}
        for case in merged_data['case_studies']:
            year = case.get('year', 'Unknown')
            years[year] = years.get(year, 0) + 1
        
        # Add to metadata
        merged_data['metadata']['brand_distribution'] = brands
        merged_data['metadata']['year_distribution'] = years
        merged_data['metadata']['top_brands'] = sorted(
            brands.items(), key=lambda x: x[1], reverse=True
        )[:10]
    
    # Save merged file
    if not output_path:
        output_path = Path(file1_path).parent / f"merged_warc_json_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Merged file saved to: {output_path}")
    
    return merged_data

def main():
    """Run the merger"""
    # File paths
    file1 = "./output/real_campaigns_extraction/warc_comprehensive_json.json"
    file2 = "./output/real_campaigns_extraction/complete_warc_json.json"
    
    # Check if files exist
    if not Path(file1).exists():
        print(f"Error: {file1} not found")
        return
    
    if not Path(file2).exists():
        print(f"Error: {file2} not found")
        return
    
    # Merge the files
    merge_warc_jsons(file1, file2)

if __name__ == "__main__":
    main()