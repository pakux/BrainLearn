#!/usr/bin/env python3
"""
Script to generate ssl-multi.json from image files in specified directories
"""
# %%
#!/usr/bin/env python3
"""
Script to generate ssl-multi.json from image files in specified directories
with 80/20 train/validation split
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple

def find_images(base_path: str, directories: List[str], extensions: List[str] = ['.npy']) -> List[Dict[str, str]]:
    """
    Find all image files in the specified directories
    
    Args:
        base_path: Base path to search from
        directories: List of directory names to search in (e.g., ['mspaths', 'adni1', 'nifd', 'ppmi'])
        extensions: List of file extensions to include (default: ['.npy'])
    
    Returns:
        List of dictionaries with 'image' key containing file paths
    """
    all_images = []
    
    for directory in directories:
        dir_path = Path(base_path) / directory
        
        if not dir_path.exists():
            print(f"Warning: Directory not found: {dir_path}")
            continue
        
        print(f"Searching in: {dir_path}")
        
        # Find all files with specified extensions
        for ext in extensions:
            for file_path in dir_path.rglob(f"*{ext}"):
                if file_path.is_file():
                    all_images.append({
                        "image": str(file_path)
                    })
        
        print(f"  Found {len([d for d in all_images if directory in d['image']])} files in {directory}")
    
    return all_images


def split_train_val(data: List[Dict[str, str]], 
                     train_ratio: float = 0.8,
                     seed: int = 42) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Split data into training and validation sets
    
    Args:
        data: List of image dictionaries
        train_ratio: Ratio of data to use for training (default: 0.8)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (training_data, validation_data)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle the data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split index
    split_idx = int(len(shuffled_data) * train_ratio)
    
    # Split the data
    train_data = shuffled_data[:split_idx]
    val_data = shuffled_data[split_idx:]
    
    return train_data, val_data


def generate_ssl_multi_json(base_path: str, 
                            directories: List[str],
                            output_file: str = "ssl-multi.json",
                            extensions: List[str] = ['.npy'],
                            train_ratio: float = 0.8,
                            seed: int = 42):
    """
    Generate ssl-multi.json file with image paths from specified directories
    Split into 80/20 train/validation sets
    
    Args:
        base_path: Base path where directories are located
        directories: List of directory names to search
        output_file: Output JSON file name
        extensions: File extensions to include
        train_ratio: Ratio of data for training (default: 0.8 for 80%)
        seed: Random seed for reproducibility
    """
    print(f"Generating {output_file}...")
    print(f"Base path: {base_path}")
    print(f"Directories: {directories}")
    print(f"Extensions: {extensions}")
    print(f"Train/Val split: {int(train_ratio*100)}/{int((1-train_ratio)*100)}")
    print(f"Random seed: {seed}")
    print("-" * 60)
    
    # Find all images
    all_images = find_images(base_path, directories, extensions)
    
    if len(all_images) == 0:
        print("ERROR: No images found!")
        return
    
    # Split into train and validation
    train_data, val_data = split_train_val(all_images, train_ratio, seed)
    
    # Create the final structure
    output_data = {
        "training": train_data,
        "validation": val_data
    }
    
    # Write to JSON file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("-" * 60)
    print(f"Total images found: {len(all_images)}")
    print(f"Training images: {len(train_data)} ({len(train_data)/len(all_images)*100:.1f}%)")
    print(f"Validation images: {len(val_data)} ({len(val_data)/len(all_images)*100:.1f}%)")
    print(f"Output written to: {output_file}")
    
    # Print summary by directory
    print("\nSummary by directory:")
    for directory in directories:
        total_count = len([d for d in all_images if directory in d['image']])
        train_count = len([d for d in train_data if directory in d['image']])
        val_count = len([d for d in val_data if directory in d['image']])
        print(f"  {directory}: {total_count} total ({train_count} train, {val_count} val)")


if __name__ == "__main__":
    # Configuration
    BASE_PATH = "/mnt/bulk-neptune/radhika/project/images"
    DIRECTORIES = ["ukb", "mspaths", "adni1", "nifd", "ppmi"]
    OUTPUT_FILE = "../data/ssl_data/ssl-multi.json"
    EXTENSIONS = ['.npy']  # Add other extensions if needed, e.g., ['.npy', '.png', '.jpg']
    TRAIN_RATIO = 0.8  # 80% training, 20% validation
    RANDOM_SEED = 42  # For reproducibility
    
    # Generate the JSON file
    generate_ssl_multi_json(
        base_path=BASE_PATH,
        directories=DIRECTORIES,
        output_file=OUTPUT_FILE,
        extensions=EXTENSIONS,
        train_ratio=TRAIN_RATIO,
        seed=RANDOM_SEED
    )
    
    # Optional: Print sample entries
    print("\nSample training entries (first 3):")
    with open(OUTPUT_FILE, 'r') as f:
        data = json.load(f)
        for i, entry in enumerate(data['training'][:3]):
            print(f"  {i+1}. {entry['image']}")
    
    print("\nSample validation entries (first 3):")
    for i, entry in enumerate(data['validation'][:3]):
        print(f"  {i+1}. {entry['image']}")
# %%
