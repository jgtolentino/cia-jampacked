#!/usr/bin/env python3
"""
Revert file paths back to /Users/tbwa/ while keeping other branding changes
"""

import os
import re
import glob
from pathlib import Path

# Define replacements - only fix the paths
PATH_FIXES = {
    # Fix paths back to tbwa
    r'/Users/tbwa/': '/Users/tbwa/',
    r'"/Users/tbwa/': '"/Users/tbwa/',
    r"'/Users/tbwa/": "'/Users/tbwa/",
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

def fix_file_paths(filepath):
    """Fix paths in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply path fixes
        for pattern, replacement in PATH_FIXES.items():
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Main path fixing process"""
    print("Starting path fixes...")
    
    # Count files to process
    total_files = 0
    for pattern in FILE_PATTERNS:
        for filepath in glob.glob(pattern, recursive=True):
            if should_process_file(filepath):
                total_files += 1
    
    print(f"Found {total_files} files to check")
    
    # Process files
    modified_files = 0
    for pattern in FILE_PATTERNS:
        for filepath in glob.glob(pattern, recursive=True):
            if should_process_file(filepath):
                if fix_file_paths(filepath):
                    modified_files += 1
                    print(f"Fixed paths in: {filepath}")
    
    print(f"\nFixed paths in {modified_files} files")
    
    # Create fix summary
    with open('PATH_FIX_REPORT.md', 'w') as f:
        f.write("# Path Fix Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Files checked: {total_files}\n")
        f.write(f"- Files fixed: {modified_files}\n\n")
        f.write("## Changes Made\n\n")
        f.write("- Reverted `/Users/tbwa/` back to `/Users/tbwa/`\n")
        f.write("- Kept all other Pulser branding\n")
        f.write("- System paths now point to correct directories\n\n")
        f.write("## Important\n\n")
        f.write("The `/Users/tbwa/` path is the actual system directory and must remain unchanged.\n")
        f.write("Only branding references were changed, not system paths.\n")
    
    print("\nPath fixes complete! See PATH_FIX_REPORT.md for details.")

if __name__ == "__main__":
    main()