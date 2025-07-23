#!/usr/bin/env python3
"""
Merge WARC JSON files with proper nested structure handling
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

def get_campaign_identifier(case: Dict[str, Any]) -> str:
    """Extract campaign identifier from nested structure"""
    # Try to get campaign name from nested structure
    if 'campaign_identification' in case and isinstance(case['campaign_identification'], dict):
        campaign_id = case['campaign_identification']
        # Try different possible field names
        return (campaign_id.get('campaign_name') or 
                campaign_id.get('campaign_title') or 
                campaign_id.get('campaign') or 
                campaign_id.get('name') or 
                f"Case_{case.get('case_id', 'Unknown')}")
    
    # Fallback to case_id
    return case.get('case_id', f"Unknown_{hash(str(case))}")

def merge_warc_jsons(file1_path: str, file2_path: str, output_path: str = None):
    """Merge two WARC JSON files with nested structure support"""
    
    # Load both files
    print(f"Loading {file1_path}...")
    with open(file1_path, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    
    print(f"Loading {file2_path}...")
    with open(file2_path, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
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
    
    # Track unique case studies by identifier
    unique_cases = {}
    case_sources = {}  # Track which file each case came from
    
    # Process case studies from both files
    for idx, (data, source_name) in enumerate([(data1, 'file1'), (data2, 'file2')]):
        if 'case_studies' in data:
            print(f"\nProcessing {source_name}: {len(data['case_studies'])} cases")
            
            for case in data['case_studies']:
                # Get unique identifier
                case_id = get_campaign_identifier(case)
                
                # If we haven't seen this case, add it
                if case_id not in unique_cases:
                    unique_cases[case_id] = case
                    case_sources[case_id] = source_name
                else:
                    # Check if the new case has more data
                    existing = unique_cases[case_id]
                    
                    # Count non-empty fields recursively
                    def count_fields(obj, depth=0):
                        if depth > 3:  # Prevent infinite recursion
                            return 0
                        count = 0
                        if isinstance(obj, dict):
                            for v in obj.values():
                                if v is not None and v != "" and v != {}:
                                    count += 1
                                    if isinstance(v, (dict, list)):
                                        count += count_fields(v, depth + 1)
                        elif isinstance(obj, list):
                            count += len([x for x in obj if x is not None and x != ""])
                        return count
                    
                    existing_fields = count_fields(existing)
                    new_fields = count_fields(case)
                    
                    print(f"  Duplicate found: {case_id}")
                    print(f"    Existing: {existing_fields} fields from {case_sources[case_id]}")
                    print(f"    New: {new_fields} fields from {source_name}")
                    
                    if new_fields > existing_fields:
                        print(f"    → Keeping new version (more complete)")
                        unique_cases[case_id] = case
                        case_sources[case_id] = source_name
                    else:
                        print(f"    → Keeping existing version")
    
    # Convert back to list
    merged_data['case_studies'] = list(unique_cases.values())
    
    # Update metadata counts
    merged_data['metadata']['total_cases'] = len(merged_data['case_studies'])
    merged_data['metadata']['unique_campaigns'] = len(unique_cases)
    
    # Stats from original files
    stats1_count = len(data1.get('case_studies', []))
    stats2_count = len(data2.get('case_studies', []))
    merged_count = len(merged_data['case_studies'])
    
    print(f"\n{'='*50}")
    print(f"MERGE STATISTICS:")
    print(f"{'='*50}")
    print(f"File 1: {stats1_count} case studies")
    print(f"File 2: {stats2_count} case studies")
    print(f"Merged: {merged_count} unique case studies")
    print(f"Duplicates removed: {stats1_count + stats2_count - merged_count}")
    
    # Show source distribution
    source_counts = {}
    for source in case_sources.values():
        source_counts[source] = source_counts.get(source, 0) + 1
    print(f"\nSource distribution:")
    for source, count in source_counts.items():
        print(f"  From {source}: {count} cases")
    
    # Extract statistics from merged data
    if merged_data['case_studies']:
        # Brand distribution
        brands = {}
        years = {}
        countries = {}
        
        for case in merged_data['case_studies']:
            # Extract brand
            if 'campaign_identification' in case:
                brand = case['campaign_identification'].get('brand', 'Unknown')
                brands[brand] = brands.get(brand, 0) + 1
                
                # Extract year
                year = case['campaign_identification'].get('year', 'Unknown')
                years[year] = years.get(year, 0) + 1
            
            # Extract country from market context
            if 'market_context' in case:
                country = case['market_context'].get('country', 'Unknown')
                countries[country] = countries.get(country, 0) + 1
        
        # Add to metadata
        merged_data['metadata']['brand_distribution'] = dict(sorted(brands.items()))
        merged_data['metadata']['year_distribution'] = dict(sorted(years.items()))
        merged_data['metadata']['country_distribution'] = dict(sorted(countries.items()))
        merged_data['metadata']['top_brands'] = sorted(
            brands.items(), key=lambda x: x[1], reverse=True
        )[:10]
        
        print(f"\nTop brands in merged data:")
        for brand, count in merged_data['metadata']['top_brands'][:5]:
            print(f"  {brand}: {count} campaigns")
    
    # Save merged file
    if not output_path:
        output_path = Path(file1_path).parent / f"merged_warc_json_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Merged file saved to: {output_path}")
    print(f"   File size: {Path(output_path).stat().st_size:,} bytes")
    
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
    
    print(f"File 1 size: {Path(file1).stat().st_size:,} bytes")
    print(f"File 2 size: {Path(file2).stat().st_size:,} bytes")
    
    # Merge the files
    merge_warc_jsons(file1, file2)

if __name__ == "__main__":
    main()