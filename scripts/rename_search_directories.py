#!/usr/bin/env python3
"""
Rename existing search directories from method-specific prefixes to search_ prefix.
Changes probe_*, random_*, grid_* to search_* while keeping the timestamp.
"""
import os
import re

def rename_search_directories():
    """Rename search directories to use consistent search_ prefix."""
    searches_dir = os.path.join('runs', 'searches')
    
    if not os.path.exists(searches_dir):
        print("No runs/searches directory found")
        return
    
    renamed_count = 0
    for old_name in os.listdir(searches_dir):
        # Skip if already has search_ prefix
        if old_name.startswith('search_'):
            continue
            
        # Match patterns like probe_YYYYMMDD_HHMMSS, random_YYYYMMDD_HHMMSS, etc.
        match = re.match(r'^(probe|random|grid)_(\d{8}_\d{6})$', old_name)
        if match:
            method, timestamp = match.groups()
            new_name = f'search_{timestamp}'
            
            old_path = os.path.join(searches_dir, old_name)
            new_path = os.path.join(searches_dir, new_name)
            
            # Check if new name already exists
            if os.path.exists(new_path):
                print(f"Warning: {new_name} already exists, skipping {old_name}")
                continue
            
            # Rename the directory
            os.rename(old_path, new_path)
            print(f"Renamed: {old_name} -> {new_name}")
            renamed_count += 1
    
    print(f"\nRenamed {renamed_count} search directories")
    
    # Show final state
    print("\nCurrent search directories:")
    for name in sorted(os.listdir(searches_dir)):
        print(f"  {name}")

if __name__ == "__main__":
    rename_search_directories()