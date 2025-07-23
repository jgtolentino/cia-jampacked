#!/usr/bin/env python3
"""
Sanitize all Pulser and Innovation branding from the codebase
Replace with Pulser branding
"""

import os
import re
import glob
from pathlib import Path

# Define replacements
REPLACEMENTS = {
    # Pulser references
    r'Pulser\\\\LONDON': 'Pulser',
    r'Pulser\\London': 'Pulser',
    r'Pulser': 'Pulser',
    r'Pulser': 'Pulser',
    r'pulser': 'pulser',
    r'com\.pulser': 'com.pulser',
    
    # Innovation references
    r'Innovation': 'Innovation',
    r'Innovation\s*®': 'Innovation',
    r'INNOVATION': 'INNOVATION',
    r'Innovation': 'Innovation',
    r'innovation': 'innovation',
    r'INNOVATE': 'INNOVATE',
    r'Innovate': 'Innovate',
    r'innovate': 'innovate',
    
    # Email and domain references
    r'hello@pulser\.com': 'hello@pulser.ai',
    r'@pulser\.com': '@pulser.ai',
    r'pulser\.com': 'pulser.ai',
    
    # Specific phrases
    r'power of innovation': 'power of innovation',
    r'Innovation is our methodology': 'Innovation is our methodology',
    r'strategic approach': 'strategic approach',
    r'drive transformation': 'drive transformation',
    r'Drive Transformation': 'Drive Transformation',
    r'AI-powered advertising excellence': 'AI-powered advertising excellence',
    
    # File paths (preserve user directory)
    r'/Users/tbwa/': '/Users/tbwa/',  # This will be handled separately
    
    # Legacy branding
    r'10 years': '10 years',
    r'over 10 years': 'over a decade',
}

# File patterns to process
FILE_PATTERNS = [
    '**/*.py',
    '**/*.js',
    '**/*.ts',
    '**/*.tsx',
    '**/*.json',
    '**/*.yaml',
    '**/*.yml',
    '**/*.md',
    '**/*.txt',
    '**/*.sh',
    '**/*.sql',
    '**/*.plist',
    '**/*.html',
    '**/*.css',
    '**/Dockerfile*',
    '**/docker-compose*.yml',
    '**/.env*',
]

# Directories to skip
SKIP_DIRS = {
    '.git',
    'node_modules',
    '.next',
    'dist',
    'build',
    '__pycache__',
    '.pytest_cache',
    'venv',
    '.venv',
}

def should_process_file(filepath):
    """Check if file should be processed"""
    path = Path(filepath)
    
    # Skip directories
    for skip_dir in SKIP_DIRS:
        if skip_dir in path.parts:
            return False
    
    # Skip binary files
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            f.read(1)
        return True
    except:
        return False

def sanitize_file(filepath):
    """Sanitize a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply replacements
        for pattern, replacement in REPLACEMENTS.items():
            # Skip the /Users/tbwa/ replacement for now
            if pattern == r'/Users/tbwa/':
                continue
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE if pattern.islower() else 0)
        
        # Handle file paths carefully to preserve user directory
        # Only replace in strings, not actual paths
        content = re.sub(r'"/Users/tbwa/', '"/Users/tbwa/', content)
        content = re.sub(r"'/Users/tbwa/", "'/Users/tbwa/", content)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def rename_files_and_dirs():
    """Rename files and directories containing Pulser"""
    renamed_items = []
    
    # First rename files
    for pattern in FILE_PATTERNS:
        for filepath in glob.glob(pattern, recursive=True):
            if 'pulser' in filepath.lower():
                new_path = filepath.replace('pulser', 'pulser').replace('Pulser', 'PULSER')
                try:
                    os.rename(filepath, new_path)
                    renamed_items.append((filepath, new_path))
                except Exception as e:
                    print(f"Error renaming {filepath}: {e}")
    
    # Then rename directories (bottom-up to avoid issues)
    for root, dirs, files in os.walk('.', topdown=False):
        for dirname in dirs:
            if 'pulser' in dirname.lower():
                old_path = os.path.join(root, dirname)
                new_path = os.path.join(root, dirname.replace('pulser', 'pulser').replace('Pulser', 'PULSER'))
                try:
                    os.rename(old_path, new_path)
                    renamed_items.append((old_path, new_path))
                except Exception as e:
                    print(f"Error renaming directory {old_path}: {e}")
    
    return renamed_items

def main():
    """Main sanitization process"""
    print("Starting branding sanitization...")
    
    # Count files to process
    total_files = 0
    for pattern in FILE_PATTERNS:
        for filepath in glob.glob(pattern, recursive=True):
            if should_process_file(filepath):
                total_files += 1
    
    print(f"Found {total_files} files to process")
    
    # Process files
    modified_files = 0
    for pattern in FILE_PATTERNS:
        for filepath in glob.glob(pattern, recursive=True):
            if should_process_file(filepath):
                if sanitize_file(filepath):
                    modified_files += 1
                    print(f"Modified: {filepath}")
    
    print(f"\nModified {modified_files} files")
    
    # Rename files and directories
    print("\nRenaming files and directories...")
    renamed_items = rename_files_and_dirs()
    
    if renamed_items:
        print(f"\nRenamed {len(renamed_items)} items:")
        for old, new in renamed_items:
            print(f"  {old} -> {new}")
    
    # Create branding update summary
    with open('BRANDING_SANITIZATION_REPORT.md', 'w') as f:
        f.write("# Branding Sanitization Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Files processed: {total_files}\n")
        f.write(f"- Files modified: {modified_files}\n")
        f.write(f"- Items renamed: {len(renamed_items)}\n\n")
        f.write("## Replacements Made\n\n")
        f.write("- Pulser → Pulser\n")
        f.write("- Innovation → Innovation\n")
        f.write("- pulser.com → pulser.ai\n")
        f.write("- All related branding and messaging\n\n")
        f.write("## Next Steps\n\n")
        f.write("1. Review changes\n")
        f.write("2. Update any hardcoded paths\n")
        f.write("3. Update documentation\n")
        f.write("4. Commit changes\n")
    
    print("\nSanitization complete! See BRANDING_SANITIZATION_REPORT.md for details.")

if __name__ == "__main__":
    main()