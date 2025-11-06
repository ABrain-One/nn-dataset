import json
import os
from pathlib import Path
from typing import Any, List, Dict


def filter_items_with_duration(data: Any) -> tuple[Any, int, int]:
    removed_count = 0
    kept_count = 0
    
    if isinstance(data, list):
        filtered_list = []
        for item in data:
            if isinstance(item, dict):
                if 'duration' in item and item['duration'] != 0:
                    filtered_list.append(item)
                    kept_count += 1
                else:
                    removed_count += 1
            else:
                filtered_list.append(item)
                kept_count += 1
        return filtered_list, removed_count, kept_count
    
    elif isinstance(data, dict):
        filtered_dict = {}
        for key, value in data.items():
            filtered_value, rem, kept = filter_items_with_duration(value)
            filtered_dict[key] = filtered_value
            removed_count += rem
            kept_count += kept
        return filtered_dict, removed_count, kept_count
    
    else:
        return data, 0, 1


def process_json_file(file_path: Path) -> tuple[bool, int, int]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        filtered_data, removed_count, kept_count = filter_items_with_duration(data)
        
        if removed_count > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=4)
            return True, removed_count, kept_count
        
        return False, 0, kept_count
            
    except json.JSONDecodeError as e:
        print(f"  ⚠ Error reading JSON: {e}")
        return False, 0, 0
    except Exception as e:
        print(f"  ⚠ Error processing file: {e}")
        return False, 0, 0


def find_all_json_files(base_path: Path) -> List[Path]:
    json_files = []
    
    if not base_path.exists():
        print(f"Error: Path does not exist: {base_path}")
        return json_files
    
    for directory in base_path.iterdir():
        if directory.is_dir():
            for json_file in directory.glob("*.json"):
                json_files.append(json_file)
    
    return sorted(json_files)


def main():
    script_dir = Path(__file__).parent
    base_path = script_dir / "ab" / "nn" / "stat" / "train"
    
    print(f"Scanning directory: {base_path}")
    print("=" * 70)
    
    json_files = find_all_json_files(base_path)
    
    if not json_files:
        print(f"\nNo JSON files found in {base_path}")
        return
    
    print(f"\nFound {len(json_files)} JSON files\n")
    
    total_files_processed = 0
    total_files_modified = 0
    total_items_removed = 0
    total_items_kept = 0
    
    modified_files = []
    
    for json_file in json_files:
        relative_path = f"{json_file.parent.name}/{json_file.name}"
        print(f"Processing: {relative_path}")
        
        was_modified, removed_count, kept_count = process_json_file(json_file)
        
        total_files_processed += 1
        total_items_kept += kept_count
        
        if was_modified:
            total_files_modified += 1
            total_items_removed += removed_count
            modified_files.append((relative_path, removed_count, kept_count))
            print(f"  ✓ Modified: Removed {removed_count} item(s), kept {kept_count} item(s)")
        else:
            if kept_count > 0:
                print(f"  ✓ No changes needed: All {kept_count} item(s) have valid duration")
            else:
                print(f"  ℹ Empty or no items to process")
        print()
    
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Total JSON files processed: {total_files_processed}")
    print(f"  Files modified: {total_files_modified}")
    print(f"  Files unchanged: {total_files_processed - total_files_modified}")
    print(f"  Total items removed (no duration or duration=0): {total_items_removed}")
    print(f"  Total items kept (valid duration): {total_items_kept}")
    
    if modified_files:
        print("\n" + "=" * 70)
        print(f"\nModified files (showing first 20):")
        for i, (file_path, removed, kept) in enumerate(modified_files[:20], 1):
            print(f"  {i}. {file_path} - Removed: {removed}, Kept: {kept}")
        if len(modified_files) > 20:
            print(f"  ... and {len(modified_files) - 20} more")
    
    print("\n✓ Processing complete!")


if __name__ == "__main__":
    main()